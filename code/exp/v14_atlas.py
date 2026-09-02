# SPDX-License-Identifier: Apache-2.0
"""The depth atlas: every lens's per-patch map, at every depth, graded against
what patching each patch actually does to the answer.

For each picture and each of six depths, the image tokens are split into 2x2
blocks (49 regions on a 14x14 grid).  The truth for a region is the change in
the outcome axis at the answer slot when that region's residuals at that depth
are replaced by another picture's (resample, two donors, averaged) or by the
per-position corpus mean (covering).  Every instrument produces a per-patch map
from the same forward pass, summed into the same regions, and is scored by the
rank correlation between its predicted change and the measured change, and by
whether its strongest region is the region that matters most.

Instruments
  write, path      this item's gradient of the axis, integrated from the
                    per-position corpus mean to the activation (8 midpoints),
                    times the centred activation: the method's attribution
  write, one point the same at a single point (gradient x centred activation)
  write, averaged     the corpus-mean direction u at image tokens, centred
  identity          the final axis read at the patch with no transport
                    (r . h for refusal; the yes/no logit margin of the unembedded
                    patch for yes/no, i.e. the logit lens)
  grad x act        gradient times raw activation, origin baseline
  GLIMPSE           gradient-weighted attention rollout of the model's reply,
                    the saliency literature's prompt-conditioned map (no depth)

Two axes: the refusal margin on MM-SafetyBench typography items the model
refuses and complies with, and the yes/no margin on COCO photographs asked
about a present and an absent object.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--n-per", type=int, default=6, help="items per cell: refused, complied, present, absent")
ap.add_argument("--donors", type=int, default=2)
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--n-mu", type=int, default=40)
ap.add_argument("--gate-dir", default="/workspace/vlm/results/gate")
ap.add_argument("--out", default="/workspace/vlm/results/atlas")
ap.add_argument("--figs", default="/workspace/vlm/figures/atlas")
a = ap.parse_args()

import jvlm  # noqa: E402
from jvlm import axes, cache, cocodata, corpus, depth, judge, plot, readout, safedata, verdict  # noqa: E402
from jvlm import glimpse as G  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

name = a.repo.split("/")[-1]
os.makedirs(a.out, exist_ok=True); os.makedirs(a.figs, exist_ok=True)
say = lambda *x: print(*x, flush=True)
T0 = time.time()
m = jvlm.load_vlm(a.repo, attn="eager")   # GLIMPSE needs the attention matrices, which SDPA never materialises
L6 = depth.layers_at(m, depth.STANDARD); LAST = m.n_layers - 1
frac = {l: depth.to_frac(m, l) for l in range(m.n_layers)}
plot.use()
rng = random.Random(11)
YES = axes.token_set(m.tokenizer, axes.YES_FORMS); NO = axes.token_set(m.tokenizer, axes.NO_FORMS)
COCO = cocodata._load()


def margin(lg):
    lp = torch.log_softmax(lg.float(), -1)
    return torch.logsumexp(lp[..., YES], -1) - torch.logsumexp(lp[..., NO], -1)


# ------------------------------------------------------------------ items
acts, side = cache.load(f"{a.gate_dir}/{name}__eval.npz")
CARRIER = side["prompt"]
its = [it for it in side["items"] if it.get("variant") == "SD_TYPO"]
ANS = side["_extra"]["ANS"].astype(np.float32)
keep = np.array([it.get("variant") == "SD_TYPO" for it in side["items"]])
ANS = ANS[keep]
y_r = np.array([judge.item_refused(it) for it in its], int)
r = verdict.dim_direction(ANS[:, -1], y_r).float()
r = r / r.norm()
R_DEV = r.to(m.device)
VLc = verdict.VerdictLens.load(f"{a.gate_dir}/{name}__verdict_carrier.pt")

refused = [i for i in range(len(its)) if its[i]["harmful"] and y_r[i] == 1]
complied = [i for i in range(len(its)) if its[i]["harmful"] and y_r[i] == 0 and judge.kind(its[i]["reply"]) == "compliance"]
rng.shuffle(refused); rng.shuffle(complied)
items = [dict(kind="refusal", tag="refused", path=its[i]["path"], prompt=CARRIER, reply=its[i]["reply"]) for i in refused[: a.n_per]]
items += [dict(kind="refusal", tag="complied", path=its[i]["path"], prompt=CARRIER, reply=its[i]["reply"]) for i in complied[: a.n_per]]
# the per-position mean for typography images comes from other typography items
mu_items_ref = [its[i]["path"] for i in (refused[a.n_per:] + complied[a.n_per:])][: a.n_mu]

pool = cocodata.concept_items(600, min_area=0.05, seed=404)
seen, coco = set(), []
for it in pool:
    if it["image_id"] in seen or not cocodata.concept_token_ids(m.tokenizer, it["concept"]):
        continue
    seen.add(it["image_id"]); coco.append(it)
YN = "Is there a {w} in this picture? Answer yes or no."
for it in coco[: a.n_per]:
    items.append(dict(kind="yesno", tag="present", path=it["path"], prompt=YN.format(w=it["concept"]), word=it["concept"]))
for it in coco[a.n_per: 2 * a.n_per]:
    absent = cocodata.distractor(it["image_id"], it["concept"], rng, hard=True)
    items.append(dict(kind="yesno", tag="absent", path=it["path"], prompt=YN.format(w=absent), word=absent))
mu_items_coco = [it["path"] for it in coco[2 * a.n_per: 2 * a.n_per + a.n_mu]]
say(f"{len(items)} items: " + ", ".join(f"{t} {sum(1 for i in items if i['tag'] == t)}" for t in ("refused", "complied", "present", "absent")))


def img(p): return corpus.load_image(p, square=448)
def axis_fn(kind):
    if kind == "refusal":
        return lambda hL, lg: hL.float() @ R_DEV
    return lambda hL, lg: margin(lg)


# ------------------------------------------------------------------ per-position means, per image family
def per_position_means(paths, prompt):
    MU, n = {l: None for l in L6}, 0
    with torch.no_grad():
        for p in paths:
            b = m.chat(prompt, [img(p)]); pos = readout.image_positions(m, b)
            with ActivationRecorder(m.layers, at=L6) as rec:
                m.forward(b)
            for l in L6:
                h = rec.activations[l][0, pos.to(rec.activations[l].device)].float().cpu()
                if MU[l] is None: MU[l] = torch.zeros_like(h)
                if MU[l].shape == h.shape: MU[l] += h
            n += 1
    return {l: MU[l] / n for l in L6}
MU = {"refusal": per_position_means(mu_items_ref, CARRIER), "yesno": per_position_means(mu_items_coco, YN.format(w="dog"))}
P = MU["refusal"][L6[0]].shape[0]; Gd = int(round(math.sqrt(P)))
NB = 7 if Gd % 7 == 0 else Gd // 2; BS = Gd // NB
REG = np.array([(i // BS) * NB + (j // BS) for i in range(Gd) for j in range(Gd)])   # patch -> region
NREG = NB * NB
say(f"{P} image tokens = {Gd}x{Gd}; {NREG} regions of {BS}x{BS}")

# the fixed yes/no direction at image tokens (the carrier lens carries refusal)
def yn_batches():
    for p in mu_items_coco:
        yield m.chat(YN.format(w=rng.choice(["dog", "cat", "car", "person", "chair", "bottle", "bus", "bird"])), [img(p)])
Fy = verdict.fit(m, yn_batches(), [axes.yesno(m.tokenizer)], L6, every=100)
VLy = verdict.VerdictLens.from_fit(Fy, dict(repo=a.repo, n=Fy.n))
def fixed_dir(kind, l):
    VL = VLc if kind == "refusal" else VLy
    ax = "refusal" if kind == "refusal" else "yesno"
    return VL.dir(ax, "img", l).float(), VL.centre("img", l).float()


# ------------------------------------------------------------------ readings
def one_point(b, fn):
    """Image residuals H[l] [P, d], gradients Gr[l] [P, d] at every depth, the clean axis value, the clean logits."""
    pos = readout.image_positions(m, b); S = b["input_ids"].shape[1]
    with ActivationRecorder(m.layers, at=[*L6, LAST], start_graph_at=min(L6)) as rec, torch.enable_grad():
        m.forward(b)
        A = [rec.activations[l] for l in L6]
        hL = rec.activations[LAST][0, S - 1]
        lg = m.unembed(hL[None])
        val = fn(hL[None], lg)[0]
        g = torch.autograd.grad(val, A)
    H = {l: A[i][0, pos.to(A[i].device)].detach().float().cpu() for i, l in enumerate(L6)}
    Gr = {l: g[i][0, pos.to(g[i].device)].float().cpu() for i, l in enumerate(L6)}
    return pos, H, Gr, float(val), lg.detach()[0].float().cpu()


def hooked_axis(b, fn, layer, pos, X):
    """Run the model with layer `layer`'s image residuals replaced by X [n, P, d]; return the axis per replicate."""
    n = X.shape[0]
    big = m.replicate(b, n) if n > 1 else b
    S = b["input_ids"].shape[1]
    def hook(mod, inp, o, _x=X, _p=pos):
        tup = not torch.is_tensor(o)
        hh = (o[0] if tup else o).clone()
        hh[:, _p.to(hh.device)] = _x.to(hh.device, hh.dtype)
        return (hh, *o[1:]) if tup else hh
    hd = m.layers[layer].register_forward_hook(hook)
    try:
        with ActivationRecorder(m.layers, at=[LAST]) as rec:
            m.forward(big)
            hL = rec.activations[LAST][:, S - 1]
            return fn(hL, m.unembed(hL))
    finally:
        hd.remove()


def path_reading(b, fn, layer, pos, h, mu, steps, chunk=4):
    dev = h - mu
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    acc = torch.zeros(P, dtype=torch.float64)
    for s0 in range(0, steps, chunk):
        al = alphas[s0: s0 + chunk]
        x = (mu[None] + al[:, None, None] * dev[None]).to(m.device).detach().clone().requires_grad_(True)
        with torch.enable_grad():
            v = hooked_axis(b, fn, layer, pos, x).sum()
            g, = torch.autograd.grad(v, x)
        acc += (g.float().cpu().sum(0) * dev).sum(-1).double()
    return (acc / steps).float().numpy()


def patch_truth(b, fn, layer, pos, h, donors, mu, clean, per_batch=7):
    """Measured change of the axis when each region is replaced by each donor (resample) and by mu (covering)."""
    out = {}
    with torch.no_grad():
        for dname, D in [("cover", mu)] + [(f"donor{k}", d) for k, d in enumerate(donors)]:
            vals = np.zeros(NREG)
            for r0 in range(0, NREG, per_batch):
                regs = list(range(r0, min(NREG, r0 + per_batch)))
                X = h[None].repeat(len(regs), 1, 1).clone()
                for k, rg in enumerate(regs):
                    mk = torch.as_tensor(REG == rg)
                    X[k, mk] = D[mk]
                v = hooked_axis(b, fn, layer, pos, X).float().cpu().numpy()
                vals[regs] = v - clean
            out[dname] = vals
        out["cover_all"] = float(hooked_axis(b, fn, layer, pos, mu[None]).float().cpu()[0]) - clean
    return out


def region_sum(v):  # [P] -> [NREG]
    return np.bincount(REG, weights=np.asarray(v, float), minlength=NREG)


def glimpse_map(b, item, clean_lg):
    try:
        if item["kind"] == "refusal":
            ids = m.tokenizer(item["reply"], add_special_tokens=False).input_ids[:6]
        else:
            ids = [YES[0]] if float(margin(clean_lg[None])[0]) > 0 else [NO[0]]
        gb = G.prepare(m, b, ids)
        pos = readout.image_positions(m, gb)
        out = G.maps(m, gb, pos, answer_start=gb["_answer_start"], answer_ids=gb["_answer_ids"])
        return np.asarray(out["visual"], dtype=float)
    except Exception as e:  # noqa: BLE001
        say(f"  GLIMPSE failed: {type(e).__name__}: {e}")
        return None


# ------------------------------------------------------------------ pass 1: readings and residuals for every item
INSTR = ["write, path", "write, one point", "write, averaged", "identity", "grad x act"]
store = []
t0 = time.time()
for n_, it in enumerate(items):
    fn = axis_fn(it["kind"]); b = m.chat(it["prompt"], [img(it["path"])])
    pos, H, Gr, clean, clean_lg = one_point(b, fn)
    if len(pos) != P:
        say(f"  skip {it['path']}: {len(pos)} image tokens"); continue
    maps = {k: {} for k in INSTR}
    for l in L6:
        mu = MU[it["kind"]][l]
        u, c = fixed_dir(it["kind"], l)
        maps["write, one point"][l] = (Gr[l] * (H[l] - mu)).sum(-1).numpy()
        maps["grad x act"][l] = (Gr[l] * H[l]).sum(-1).numpy()
        maps["write, averaged"][l] = ((H[l] - c) @ u).numpy()
        if it["kind"] == "refusal":
            maps["identity"][l] = (H[l] @ r).numpy()
        else:
            with torch.no_grad():
                maps["identity"][l] = margin(m.unembed(H[l].to(m.device).to(torch.bfloat16))).cpu().numpy()
        maps["write, path"][l] = path_reading(b, fn, l, pos, H[l], mu, a.steps)
    gl = glimpse_map(b, it, clean_lg)
    store.append(dict(item=it, b=b, pos=pos, H={l: H[l].to(torch.float16) for l in L6}, maps=maps, clean=clean, glimpse=gl))
    say(f"  readings {n_ + 1}/{len(items)}  ({(time.time() - t0) / 60:.1f} min)")

# ------------------------------------------------------------------ pass 2: patching truth, donors from the same family
t0 = time.time()
for n_, st in enumerate(store):
    it = st["item"]; fn = axis_fn(it["kind"])
    fam = [s for s in store if s["item"]["kind"] == it["kind"] and s is not st]
    dons = rng.sample(fam, min(a.donors, len(fam)))
    st["truth"] = {}
    for l in L6:
        st["truth"][l] = patch_truth(st["b"], fn, l, st["pos"], st["H"][l].float(), [d["H"][l].float() for d in dons], MU[it["kind"]][l], st["clean"])
    say(f"  truth {n_ + 1}/{len(store)}  ({(time.time() - t0) / 60:.1f} min)")


# ------------------------------------------------------------------ scoring
def sp(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).correlation)

rows = []
for st in store:
    it = st["item"]
    for l in L6:
        T = st["truth"][l]
        t_res = np.mean([T[k] for k in T if k.startswith("donor")], 0) if any(k.startswith("donor") for k in T) else T["cover"]
        t_cov = T["cover"]
        rec = dict(kind=it["kind"], tag=it["tag"], layer=l, frac=frac[l], cover_all=T["cover_all"], sum_path=float(-region_sum(st["maps"]["write, path"][l]).sum()))
        for k in INSTR:
            pred = -region_sum(st["maps"][k][l])         # removing the deviation removes the reading
            rec[k] = dict(rho_resample=sp(pred, t_res), rho_cover=sp(pred, t_cov), rho_abs=sp(np.abs(pred), np.abs(t_res)),
                          top1=float(int(np.argmax(np.abs(pred))) == int(np.argmax(np.abs(t_res)))))
        if st["glimpse"] is not None:
            g = region_sum(st["glimpse"])
            rec["GLIMPSE"] = dict(rho_resample=float("nan"), rho_cover=float("nan"), rho_abs=sp(g, np.abs(t_res)), top1=float(int(np.argmax(g)) == int(np.argmax(np.abs(t_res)))))
        rows.append(rec)

def summary(kind, key, metric):
    out = {}
    for l in L6:
        v = [r_[key][metric] for r_ in rows if r_["kind"] == kind and r_["layer"] == l and key in r_ and not np.isnan(r_[key][metric])]
        out[l] = (float(np.median(v)) if v else float("nan"), float(np.mean(v)) if v else float("nan"), len(v))
    return out

ALLK = INSTR + ["GLIMPSE"]
for kind in ("refusal", "yesno"):
    say(f"\n== {kind}: median Spearman between predicted and measured change (resample truth); |.| version for GLIMPSE")
    say(f"{'depth':>6} " + " ".join(f"{k:>18}" for k in ALLK))
    for l in L6:
        cells = []
        for k in ALLK:
            s = summary(kind, k, "rho_abs" if k == "GLIMPSE" else "rho_resample")[l]
            cells.append(f"{s[0]:>18.3f}")
        say(f"{frac[l]:>6.2f} " + " ".join(cells))
    say(f"   top-1 region hit (chance {1 / NREG:.3f})")
    for l in L6:
        say(f"{frac[l]:>6.2f} " + " ".join(f"{summary(kind, k, 'top1')[l][1]:>18.3f}" for k in ALLK))
    comp = [(r_["sum_path"], r_["cover_all"]) for r_ in rows if r_["kind"] == kind]
    if comp:
        s_, c_ = np.array(comp).T
        say(f"   completeness of the path reading: sum of regional predictions vs whole-image covering change, Pearson {np.corrcoef(s_, c_)[0, 1]:.3f}, "
            f"median relative error {np.median(np.abs(s_ - c_) / np.maximum(np.abs(c_), 1e-6)):.2f}")

# ------------------------------------------------------------------ figures
ICOL = {"write, path": plot.SERIES[0], "write, one point": plot.SERIES[4], "write, averaged": plot.SERIES[2], "identity": plot.SERIES[1],
        "grad x act": plot.SERIES[3], "GLIMPSE": "#7a7a7a"}
xs = [frac[l] for l in L6]
fig, axs = plt.subplots(2, 2, figsize=(8.4, 5.0), sharex=True)
ends = {}
for j, kind in enumerate(("refusal", "yesno")):
    for k in ALLK:
        met = "rho_abs" if k == "GLIMPSE" else "rho_resample"
        ys = [summary(kind, k, met)[l][0] for l in L6]
        if all(np.isnan(ys)): continue
        axs[0, j].plot(xs, ys, color=ICOL[k], lw=1.7 if k == "write, path" else 1.1, ls="--" if k == "GLIMPSE" else "-")
        ends.setdefault(j, []).append((ys[-1], k))
        axs[1, j].plot(xs, [summary(kind, k, "top1")[l][1] for l in L6], color=ICOL[k], lw=1.7 if k == "write, path" else 1.1, ls="--" if k == "GLIMPSE" else "-")
    plot.chance_line(axs[0, j], 0.0); plot.chance_line(axs[1, j], 1 / NREG)
    lab = sorted(ends.get(j, []), key=lambda t: -t[0])
    ys_ = plot.stagger([v for v, _ in lab], 0.055)
    for (v, k), yy in zip(lab, ys_):
        plot.end_label(axs[0, j], xs[-1], yy, k, ICOL[k])
    axs[0, j].set_title({"refusal": "refusal margin, typography items", "yesno": "yes/no margin, COCO photographs"}[kind], loc="left", fontsize=9.5, color=plot.INK2)
    axs[1, j].set_xlabel("depth")
axs[0, 0].set_ylabel("rank agreement with patching"); axs[1, 0].set_ylabel("strongest region is the right one")
for ax in axs.flat: plot.despine(ax)
fig.tight_layout(); plot.save(fig, f"{a.figs}/{name}_atlas_summary.png"); plt.close(fig)

# per-item atlas: rows = depth, columns = instruments + truth; the region maps
for i_, st in enumerate(store):
    it = st["item"]
    cols = INSTR + ["patching (truth)"]
    fig = plt.figure(figsize=(1.35 * (len(cols) + 1) + 0.8, 1.3 * len(L6) + 0.9))
    gs = fig.add_gridspec(len(L6), len(cols) + 2, width_ratios=[1.6, 0.42] + [1] * len(cols), wspace=0.07, hspace=0.12, left=0.01, right=0.99, top=0.9, bottom=0.03)
    ax = fig.add_subplot(gs[:2, 0]); ax.imshow(Image.open(it["path"]).convert("RGB").resize((448, 448))); ax.set_axis_off()
    ax.set_title(f"{it['tag']}", loc="left", fontsize=9, color=plot.INK2)
    q = it["prompt"] if len(it["prompt"]) < 44 else it["prompt"][:41] + "..."
    ax.text(0, -0.04, f"“{q}”", transform=ax.transAxes, fontsize=7.5, color=plot.INK2, ha="left", va="top")
    if st["glimpse"] is not None:
        axg = fig.add_subplot(gs[3:5, 0]); g = region_sum(st["glimpse"]).reshape(NB, NB)
        axg.imshow(g, cmap="Greys", interpolation="nearest"); axg.set_xticks([]); axg.set_yticks([]); axg.set_title("GLIMPSE", fontsize=8.5, color=plot.INK2, loc="left")
        for sp_ in axg.spines.values(): sp_.set_visible(False)
    for r_, l in enumerate(L6):
        T = st["truth"][l]; t_res = np.mean([T[k] for k in T if k.startswith("donor")], 0) if any(k.startswith("donor") for k in T) else T["cover"]
        panels = [(-region_sum(st["maps"][k][l])).reshape(NB, NB) for k in INSTR] + [t_res.reshape(NB, NB)]
        for c_, (k, M) in enumerate(zip(cols, panels)):
            ax = fig.add_subplot(gs[r_, c_ + 2]); s = float(np.percentile(np.abs(M), 99)) or 1.0
            ax.imshow(np.clip(M, -s, s), cmap="RdBu_r", vmin=-s, vmax=s, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp_ in ax.spines.values(): sp_.set_visible(False)
            if r_ == 0: ax.set_title(k, fontsize=8, color=plot.INK2)
            if c_ == 0:
                axl = fig.add_subplot(gs[r_, 1]); axl.set_axis_off()
                axl.text(0.95, 0.5, f"depth\n{frac[l]:.2f}", ha="right", va="center", fontsize=8.5, color=plot.INK2)
            if k != "patching (truth)":
                rho = sp(M.reshape(-1), t_res)
                if not np.isnan(rho): ax.text(0.02, 0.02, f"{rho:.2f}", transform=ax.transAxes, fontsize=6.5, color=plot.INK2)
    plot.save(fig, f"{a.figs}/{name}_atlas_{it['kind']}_{it['tag']}_{i_}.png"); plt.close(fig)

json.dump(dict(repo=a.repo, n_items=len(store), regions=NREG, depths={str(l): frac[l] for l in L6}, rows=rows,
               summary={kind: {k: {met: {str(l): summary(kind, k, met)[l] for l in L6} for met in ("rho_resample", "rho_cover", "rho_abs", "top1")} for k in ALLK} for kind in ("refusal", "yesno")}),
          open(f"{a.out}/{name}_atlas.json", "w"), indent=1, default=float)
say(f"wrote {a.out}/{name}_atlas.json and {a.figs}/{name}_atlas_*  ({(time.time() - T0) / 60:.1f} min)")
