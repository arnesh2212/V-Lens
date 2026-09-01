# SPDX-License-Identifier: Apache-2.0
"""A two-step question, watched as it is worked out.

"What colour is the sphere to the left of the large metal cube?"  The model has
to find the cube first, then the sphere beside it, then say that sphere's
colour.  The picture and the wording are held fixed and only the question's
target changes, so the two steps can be told apart from each other and from the
picture.  This is the visual version of watching an arithmetic answer form.

Three things are measured on the same scenes.

(a) When the answer arrives.  At the answer spot, at every depth, the read
    gives the log-odds of each of the eight colours.  We track the answer's
    colour, the landmark's colour (the trap: naming the cube instead of the
    sphere) and a third colour that is in the picture but in neither role.  The
    same is read with no map, which is the logit lens, for comparison.

(b) Where the model is looking, and when.  The per-item write over image
    positions for the score "the answer's colour", summed over the landmark's
    patches, the referent's patches, a third object's, and an equal number of
    bare floor patches.

(c) What each region is actually worth.  Cover each region at one depth with
    the average state for those positions, run the rest of the network, and
    measure how far the answer's log-odds move.  This is the truth that (b) is
    graded against, and it answers "when does the model stop needing the cube".

Then the swap: exchange the answer colour's push coordinate with another
colour's at the answer spot, and see whether the spoken answer follows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--min-sep", type=float, default=0.06)
ap.add_argument("--balance", type=float, default=5.0)
ap.add_argument("--max-obj", type=int, default=7)
ap.add_argument("--n-mu", type=int, default=40, help="scenes for the average state per position")
ap.add_argument("--n-fit", type=int, default=60, help="scenes to fit the colour write directions")
ap.add_argument("--swap-alphas", type=float, nargs="*", default=[0.1, 0.2, 0.4, 0.8])
ap.add_argument("--out", default="/workspace/vlm/results/twostep")
ap.add_argument("--figs", default="/workspace/vlm/figures/twostep")
a = ap.parse_args()

import jvlm  # noqa: E402
from jvlm import axes, clevrdata, corpus, depth, plot, reader, readout, tasks, verdict  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

name = a.repo.split("/")[-1]
os.makedirs(a.out, exist_ok=True); os.makedirs(a.figs, exist_ok=True)
say = lambda *x: print(*x, flush=True)
T0 = time.time()
m = jvlm.load_vlm(a.repo); sc = tasks.Scorer(m)
L6 = depth.layers_at(m, depth.STANDARD)
ALLL = list(range(max(m.n_deepstack, 1), m.n_layers - 1))
frac = {l: depth.to_frac(m, l) for l in range(m.n_layers)}
plot.use()
LAST = m.n_layers - 1

COLOURS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
CTOK = {}
for c in COLOURS:
    t = None
    for form in (" " + c.capitalize(), " " + c, c.capitalize(), c):
        ids = m.tokenizer.encode(form, add_special_tokens=False)
        if len(ids) == 1:
            t = int(ids[0]); break
    CTOK[c] = t
COLOURS = [c for c in COLOURS if CTOK[c] is not None]
say(f"colours with a single token: {COLOURS}")

GRID = None
# the two-step scenes are rare: only a few hundred CLEVR renders carry an unambiguous
# landmark-and-referent pair.  The average state and the colour directions do not need
# that structure, so they are fitted on ordinary renders that are not in the test set.
ev = []
for it in clevrdata.pointer_items(14, n_obj=(3, a.max_obj), min_sep=a.min_sep, balance=a.balance, limit=a.n * 4):
    if it["answer_main"] in COLOURS and it["answer_control"] in COLOURS and it["colour_distractor"] in COLOURS:
        ev.append(it)
    if len(ev) >= a.n:
        break
used = {it["image_index"] for it in ev}
SHAPES = ["cube", "sphere", "cylinder"]; SIZES = ["large", "small"]; MATS = ["metal", "rubber"]
rng0 = np.random.default_rng(3)
plain = []
for sc_ in clevrdata.scenes():
    if sc_["image_index"] in used or not clevrdata.have(sc_):
        continue
    q = f"What colour is the {rng0.choice(SIZES)} {rng0.choice(MATS)} {rng0.choice(SHAPES)}?"
    plain.append(dict(path=clevrdata.image_path(sc_), question_main=q, question_control=q))
    if len(plain) >= a.n_mu + a.n_fit:
        break
mu_items = plain[: a.n_mu]; fit_items = plain[a.n_mu: a.n_mu + a.n_fit]
say(f"{len(ev)} two-step scenes to read, {len(mu_items)} plain renders for the average state, {len(fit_items)} to fit the write directions")


def img(it): return corpus.load_image(it["path"], square=448)
ONE_WORD = " Answer with one word."
def batch(it, which="main"):
    # the reading is taken at the first answer token, so the answer has to start there:
    # without this the model opens with "The", and every colour score is about the word after it
    return m.chat(it[f"question_{which}"] + ONE_WORD, [img(it)])


# ---- the average state per image position, on CLEVR renders
MU, n_ = {l: None for l in L6}, 0
with torch.no_grad():
    for it in mu_items:
        b = batch(it); pos = readout.image_positions(m, b)
        with ActivationRecorder(m.layers, at=L6) as rec:
            m.forward(b)
        for l in L6:
            h = rec.activations[l][0, pos.to(rec.activations[l].device)].float().cpu()
            if MU[l] is None: MU[l] = torch.zeros_like(h)
            if MU[l].shape == h.shape: MU[l] += h
        n_ += 1
MU = {l: MU[l] / n_ for l in L6}
P = MU[L6[0]].shape[0]; GRID = int(round(np.sqrt(P)))
say(f"{P} image positions = {GRID}x{GRID}; averages over {n_} renders")

# ---- the read, fitted on CLEVR answer spots with no labels
ANS_fit = []
with torch.no_grad():
    for it in fit_items:
        b = batch(it, "main" if len(ANS_fit) % 2 == 0 else "control")
        with ActivationRecorder(m.layers, at=list(range(m.n_layers))) as rec:
            m.forward(b)
        ANS_fit.append(np.stack([rec.activations[l][0, -1].float().cpu().numpy() for l in range(m.n_layers)]))
RD = reader.fit(np.stack(ANS_fit).astype(np.float32), ALLL, meta=dict(repo=a.repo, corpus="CLEVR answer spots", n=len(ANS_fit)))
RD.save(f"{a.out}/{name}__read_clevr.pt")
say(f"read fitted on {len(ANS_fit)} answer spots")

# ---- the write direction for each colour at the answer spot, for the swap
AXC = [axes.Axis(f"colour:{c}", (lambda lg, h, t=CTOK[c]: lg[t].float()), "word", dict(token=CTOK[c])) for c in COLOURS]
def fit_batches():
    for it in fit_items:
        yield batch(it)
F = verdict.fit(m, fit_batches(), AXC, ALLL, every=100)
VL = verdict.VerdictLens.from_fit(F, dict(repo=a.repo, colours=COLOURS))
say("colour write directions fitted")

TOKS = [CTOK[c] for c in COLOURS]


def colour_logodds(H):
    """[n, len(COLOURS)] log-softmax over the whole vocabulary, restricted to the colour tokens."""
    with torch.no_grad():
        lg = m.unembed(torch.as_tensor(H).to(m.device).to(torch.bfloat16)).float()
        return torch.log_softmax(lg, -1)[..., TOKS].cpu().numpy()


# ---- (a) when the answer arrives, and (b) where the write is
def read_item(it, which):
    b = batch(it, which); pos = readout.image_positions(m, b); S = b["input_ids"].shape[1]
    ans_c = it["answer_main"] if which == "main" else it["answer_control"]
    other_c = it["answer_control"] if which == "main" else it["answer_main"]
    t_ans = CTOK[ans_c]
    REC_AT = sorted(set([*L6, *ALLL, LAST]))
    with ActivationRecorder(m.layers, at=REC_AT, start_graph_at=min(REC_AT)) as rec, torch.enable_grad():
        m.forward(b)
        A = [rec.activations[l] for l in L6]
        hL = rec.activations[LAST][0, S - 1]
        lg = m.unembed(hL[None])[0]
        val = torch.log_softmax(lg.float(), -1)[t_ans]
        g = torch.autograd.grad(val, A)
        ANS = {l: rec.activations[l][0, S - 1].detach().float().cpu().numpy() for l in REC_AT}
    maps = {}
    for i, l in enumerate(L6):
        gp = g[i][0, pos.to(g[i].device)].float().cpu()
        hp = A[i][0, pos.to(A[i].device)].detach().float().cpu() - MU[l]
        maps[l] = (gp * hp).sum(-1).numpy()
    return {l: v.astype(np.float32) for l, v in ANS.items()}, maps, dict(ans=ans_c, other=other_c, dist=it["colour_distractor"]), pos, b


def region_share(mp, it):
    """Share of the map's absolute mass on each region, over that region's share of the frame."""
    out = {}
    tot = np.abs(mp).sum() + 1e-9
    for key, tk in (("landmark", it["tok_landmark"]), ("referent", it["tok_referent"]),
                    ("distractor", it["tok_distractor"]), ("floor", it["tok_floor"])):
        tk = [t for t in tk if t < len(mp)]
        if not tk:
            continue
        out[key] = float(np.abs(mp[tk]).sum() / tot / (len(tk) / len(mp)))
    return out


rows, shares, t0 = [], [], time.time()
for n2, it in enumerate(ev):
    for which in ("main", "control"):
        ANS, maps, cols, pos, b = read_item(it, which)
        i_ans, i_oth, i_dst = COLOURS.index(cols["ans"]), COLOURS.index(cols["other"]), COLOURS.index(cols["dist"])
        for l in ALLL:
            for tname, H in (("read", RD.to_final(ANS[l][None], l)), ("logit lens", ANS[l][None])):
                lo = colour_logodds(H)[0]
                rows.append(dict(item=n2, which=which, layer=l, frac=frac[l], transport=tname,
                                 answer=float(lo[i_ans]), landmark=float(lo[i_oth]), distractor=float(lo[i_dst]),
                                 top_is_answer=bool(int(lo.argmax()) == i_ans), top_is_landmark=bool(int(lo.argmax()) == i_oth)))
        for l in L6:
            shares.append(dict(item=n2, which=which, layer=l, frac=frac[l], **region_share(maps[l], it)))
    if (n2 + 1) % 10 == 0:
        say(f"  read {n2 + 1}/{len(ev)}  ({(time.time() - t0) / 60:.1f} min)")

R = {(w, t): {l: [r for r in rows if r["which"] == w and r["transport"] == t and r["layer"] == l] for l in ALLL} for w in ("main", "control") for t in ("read", "logit lens")}
say(f"\nthe two-step question: how often the top colour is the right one, by depth")
say(f"{'depth':>6} {'read':>9} {'logit lens':>11}   | names the landmark instead: {'read':>9} {'logit lens':>11}   | the one-step control, read: {'right':>7}")
for l in ALLL:
    a_p = np.mean([r["top_is_answer"] for r in R[("main", "read")][l]])
    a_i = np.mean([r["top_is_answer"] for r in R[("main", "logit lens")][l]])
    l_p = np.mean([r["top_is_landmark"] for r in R[("main", "read")][l]])
    l_i = np.mean([r["top_is_landmark"] for r in R[("main", "logit lens")][l]])
    c_p = np.mean([r["top_is_answer"] for r in R[("control", "read")][l]])
    say(f"{frac[l]:>6.2f} {a_p:>9.2f} {a_i:>11.2f}   | {'':>26} {l_p:>9.2f} {l_i:>11.2f}   | {'':>29} {c_p:>7.2f}")

say(f"\nwhere the write sits, over chance for the region's size (main question)")
say(f"{'depth':>6} {'landmark':>10} {'referent':>10} {'distractor':>11} {'floor':>8}")
for l in L6:
    sl = [s for s in shares if s["which"] == "main" and s["layer"] == l]
    say(f"{frac[l]:>6.2f} " + " ".join(f"{np.median([s[k] for s in sl if k in s]):>10.2f}" for k in ("landmark", "referent", "distractor", "floor")))


# ---- (c) what each region is worth, by depth
def cover(it, l, keys, which="main"):
    b = batch(it, which); pos = readout.image_positions(m, b); S = b["input_ids"].shape[1]
    t_ans = CTOK[it["answer_main"] if which == "main" else it["answer_control"]]
    with torch.no_grad(), ActivationRecorder(m.layers, at=[l, LAST]) as rec:
        m.forward(b)
        h = rec.activations[l][0, pos.to(rec.activations[l].device)].float()
        base = float(torch.log_softmax(m.unembed(rec.activations[LAST][0, S - 1][None]).float(), -1)[0, t_ans])
    out = {}
    for key in keys:
        tk = [t for t in it[f"tok_{key}"] if t < len(h)]
        if not tk:
            continue
        X = h.clone(); X[tk] = MU[l][tk].to(X.device)
        def hook(mod, inp, o, _x=X[None], _p=pos):
            tup = not torch.is_tensor(o)
            hh = (o[0] if tup else o).clone()
            hh[:, _p.to(hh.device)] = _x.to(hh.device, hh.dtype)
            return (hh, *o[1:]) if tup else hh
        hd = m.layers[l].register_forward_hook(hook)
        try:
            with torch.no_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec2:
                m.forward(b)
                v = float(torch.log_softmax(m.unembed(rec2.activations[LAST][0, S - 1][None]).float(), -1)[0, t_ans])
        finally:
            hd.remove()
        out[key] = v - base
    return out


KEYS = ["landmark", "referent", "distractor", "floor"]
cov, t0 = [], time.time()
for n2, it in enumerate(ev):
    for l in L6:
        d = cover(it, l, KEYS)
        cov.append(dict(item=n2, layer=l, frac=frac[l], **d))
    if (n2 + 1) % 10 == 0:
        say(f"  covering {n2 + 1}/{len(ev)}  ({(time.time() - t0) / 60:.1f} min)")

say(f"\nwhat covering each region costs the answer's log-odds, by depth (median)")
say(f"{'depth':>6} " + " ".join(f"{k:>11}" for k in KEYS))
for l in L6:
    cl = [c for c in cov if c["layer"] == l]
    say(f"{frac[l]:>6.2f} " + " ".join(f"{np.median([c[k] for c in cl if k in c]):>11.2f}" for k in KEYS))
sp = {}
for l in L6:
    cl = [c for c in cov if c["layer"] == l]
    sl = [s for s in shares if s["which"] == "main" and s["layer"] == l]
    pairs = [(abs(c[k]), s[k]) for c, s in zip(cl, sl) for k in KEYS if k in c and k in s]
    if pairs:
        A_, B_ = np.array(pairs).T
        sp[l] = float(np.corrcoef(np.argsort(np.argsort(A_)), np.argsort(np.argsort(B_)))[0, 1])
say("rank agreement between the write's share and what covering costs: " + ", ".join(f"{frac[l]:.2f}: {sp.get(l, float('nan')):.2f}" for l in L6))


# ---- the swap: exchange the answer colour's coordinate with another colour's
def unit(v):
    v = torch.as_tensor(v).float(); return v / v.norm().clamp_min(1e-8)

swap_rows = []
lmid = L6[len(L6) // 2]
for n2, it in enumerate(ev[: min(len(ev), 24)]):
    b = batch(it); base = sc.generate(b, max_new_tokens=6).strip().lower()
    src, tgt = it["answer_main"], it["colour_distractor"]
    for dname in ("write", "output-layer row", "random"):
        if dname == "write":
            d_src, d_tgt = unit(VL.dir(f"colour:{src}", "ans", lmid)), unit(VL.dir(f"colour:{tgt}", "ans", lmid))
        elif dname == "output-layer row":
            W = m._lm_head.weight
            d_src, d_tgt = unit(W[CTOK[src]].float().cpu()), unit(W[CTOK[tgt]].float().cpu())
        else:
            g1 = torch.Generator().manual_seed(n2); g2 = torch.Generator().manual_seed(1000 + n2)
            d_src = unit(torch.randn(m.d_model, generator=g1)); d_tgt = unit(torch.randn(m.d_model, generator=g2))
        for al in a.swap_alphas:
            for chan in ("img", "all"):
                iv = readout.Intervention(m, {lmid: torch.stack([d_src, d_tgt])}, channel=chan, mode="clamp", alpha=al)
                iv.set_batch(b)
                with iv:
                    out_swap = sc.generate(b, max_new_tokens=6).strip().lower()
                iv2 = readout.Intervention(m, {lmid: d_tgt}, channel=chan, mode="add", alpha=al)
                iv2.set_batch(b)
                with iv2:
                    out_add = sc.generate(b, max_new_tokens=6).strip().lower()
                swap_rows.append(dict(item=n2, direction=dname, alpha=al, channel=chan, base=base, src=src, tgt=tgt,
                                      said_tgt_add=bool(tgt in out_add), said_src_add=bool(src in out_add),
                                      said_tgt_swap=bool(tgt in out_swap), said_src_swap=bool(src in out_swap),
                                      add=out_add[:24], swap=out_swap[:24]))
base_right = np.mean([tgt in r["base"] for r in swap_rows if r["alpha"] == a.swap_alphas[0] and r["direction"] == "write"])
for chan in ("img", "all"):
    say(f"\nwriting another colour at depth {frac[lmid]:.2f}, on {'the picture' if chan == 'img' else 'every position'}; the spoken answer becomes that colour:")
    say(f"{'direction':<22} " + " ".join(f"{'a=%g' % al:>8}" for al in a.swap_alphas) + "   | swap instead of add, largest dose | still says the true colour")
    for dname in ("write", "output-layer row", "random"):
        cells = []
        for al in a.swap_alphas:
            v = [r["said_tgt_add"] for r in swap_rows if r["direction"] == dname and r["alpha"] == al and r["channel"] == chan]
            cells.append(f"{np.mean(v):>8.2f}" if v else f"{'-':>8}")
        sw = [r["said_tgt_swap"] for r in swap_rows if r["direction"] == dname and r["alpha"] == a.swap_alphas[-1] and r["channel"] == chan]
        keep = [r["said_src_add"] for r in swap_rows if r["direction"] == dname and r["alpha"] == a.swap_alphas[-1] and r["channel"] == chan]
        say(f"{dname:<22} " + " ".join(cells) + f"   | {np.mean(sw) if sw else float('nan'):>32.2f} | {np.mean(keep) if keep else float('nan'):.2f}")

# ---- figures
xs = [frac[l] for l in ALLL]
fig, axs = plt.subplots(1, 2, figsize=(7.8, 2.9), sharey=True)
for ax, t in zip(axs, ("read", "logit lens")):
    lab_ = []
    for key, lab, c in (("top_is_answer", "the sphere, the answer", plot.SERIES[0]),
                        ("top_is_landmark", "the cube, the trap", plot.SERIES[1])):
        y = [np.mean([r[key] for r in R[("main", t)][l]]) for l in ALLL]
        ax.plot(xs, y, color=c, lw=1.5); lab_.append((y[-1], lab, c))
    y = [np.mean([r["top_is_answer"] for r in R[("control", t)][l]]) for l in ALLL]
    ax.plot(xs, y, color=plot.SERIES[2], lw=1.1, ls="--"); lab_.append((y[-1], "one-step control", plot.SERIES[2]))
    lab_.sort(key=lambda z: -z[0])
    for (yv, lab, c), yy in zip(lab_, plot.stagger([z[0] for z in lab_], 0.07)):
        plot.end_label(ax, xs[-1], yy, lab, c)
    ax.set_title(t, loc="left", fontsize=9.5, color=plot.INK2); ax.set_xlabel("depth"); plot.despine(ax)
axs[0].set_ylabel("top colour is this one"); axs[0].set_ylim(0, 1.02)
fig.tight_layout(); plot.save(fig, f"{a.figs}/{name}_twostep_answer.png"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(7.8, 2.9))
xs6 = [frac[l] for l in L6]
E0, E1 = [], []
for k, c in zip(KEYS, plot.SERIES):
    y = [np.median([s[k] for s in shares if s["which"] == "main" and s["layer"] == l and k in s]) for l in L6]
    axs[0].plot(xs6, y, color=c, lw=1.4); E0.append((y[-1], k, c))
    y2 = [np.median([abs(cc[k]) for cc in cov if cc["layer"] == l and k in cc]) for l in L6]
    axs[1].plot(xs6, y2, color=c, lw=1.4); E1.append((y2[-1], k, c))
for ax_, E in ((axs[0], E0), (axs[1], E1)):
    E.sort(key=lambda z: -z[0])
    span = max(z[0] for z in E) - min(z[0] for z in E) or 1.0
    for (yv, k, c), yy in zip(E, plot.stagger([z[0] for z in E], 0.13 * span)):
        plot.end_label(ax_, xs6[-1], yy, k, c)
plot.chance_line(axs[0], 1.0)
axs[0].set_title("where the write sits", loc="left", fontsize=9.5, color=plot.INK2); axs[0].set_ylabel("share over chance")
axs[1].set_title("what covering the region costs", loc="left", fontsize=9.5, color=plot.INK2); axs[1].set_ylabel("|change| in the answer’s log-odds")
for ax in axs: ax.set_xlabel("depth"); plot.despine(ax)
fig.tight_layout(); plot.save(fig, f"{a.figs}/{name}_twostep_regions.png"); plt.close(fig)

json.dump(dict(repo=a.repo, n=len(ev), colours=COLOURS, depth_swap=frac[lmid],
               answer_curves={t: {str(frac[l]): dict(main_answer=float(np.mean([r["top_is_answer"] for r in R[("main", t)][l]])),
                                                     main_landmark=float(np.mean([r["top_is_landmark"] for r in R[("main", t)][l]])),
                                                     control_answer=float(np.mean([r["top_is_answer"] for r in R[("control", t)][l]])))
                                  for l in ALLL} for t in ("read", "logit lens")},
               write_share={str(frac[l]): {k: float(np.median([s[k] for s in shares if s["which"] == "main" and s["layer"] == l and k in s])) for k in KEYS} for l in L6},
               covering={str(frac[l]): {k: float(np.median([c[k] for c in cov if c["layer"] == l and k in c])) for k in KEYS} for l in L6},
               agreement={str(frac[l]): sp.get(l) for l in L6},
               swap=swap_rows[:400]),
          open(f"{a.out}/{name}_twostep.json", "w"), indent=1, default=float)
say(f"\nwrote {a.out}/{name}_twostep.json  ({(time.time() - T0) / 60:.1f} min)")
