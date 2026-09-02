# SPDX-License-Identifier: Apache-2.0
"""One page per scenario: what the V-Lens sees, from the first layer to the answer.

Each scenario is a picture and a question.  For each one the figure shows, in order:

  the picture and the question, with the model's actual reply underneath;
  the word strip, what V-Read says the model is about to say at each depth,
      with the same strip from the logit lens beneath it for contrast;
  the score traces, two or three scores read at every depth, each shown as the
      fraction of its own final value that the instrument has recovered, since the two
      instruments are in different units and only the timing is comparable;
  the write maps, which patches are pushing the answer, at four depths.

The scenarios are chosen to cover the cases a practitioner meets: a jailbreak the
model refuses, one it complies with, an object that is there, an object that is not
there but usually would be, a two-step question, and a benign request that looks
alarming.  Nothing here is aggregated: every panel is one forward pass.
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

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--n-mu", type=int, default=30, help="images for the average state per position")
ap.add_argument("--n-fit-family", type=int, default=150, help="unlabelled states per prompt family for that family's read map")
ap.add_argument("--gen", type=int, default=40)
ap.add_argument("--gate-dir", default="/workspace/vlm/results/gate")
ap.add_argument("--out", default="/workspace/vlm/results/scenarios")
ap.add_argument("--figs", default="/workspace/vlm/figures/scenarios")
a = ap.parse_args()

import jvlm  # noqa: E402
from jvlm import axes, cache, clevrdata, cocodata, corpus, depth, judge, plot, reader, readout, safedata, tasks, verdict  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

name = a.repo.split("/")[-1]
os.makedirs(a.out, exist_ok=True); os.makedirs(a.figs, exist_ok=True)
say = lambda *x: print(*x, flush=True)
T0 = time.time()
m = jvlm.load_vlm(a.repo); sc = tasks.Scorer(m)
LAST = m.n_layers - 1
ALLL = list(range(max(m.n_deepstack, 1), LAST))
L4 = depth.layers_at(m, (0.2, 0.4, 0.6, 0.8))
frac = {l: depth.to_frac(m, l) for l in range(m.n_layers)}
plot.use()

BANK_WORDS = ["dog", "cat", "car", "person", "chair", "bottle", "bus", "bird", "cup", "book"]
YES = axes.token_set(m.tokenizer, axes.YES_FORMS); NO = axes.token_set(m.tokenizer, axes.NO_FORMS)
SR = axes.token_set(m.tokenizer, axes.REFUSE_STARTERS); SC = axes.token_set(m.tokenizer, axes.COMPLY_STARTERS)

# ---- the instruments: the read from the disjoint corpus, the write from the gate fit
RD_GLOBAL = reader.Reader.load(f"/workspace/vlm/results/reader/{name}__reader.pt")
VL = verdict.VerdictLens.load(f"{a.gate_dir}/{name}__verdict_carrier.pt")
acts, side = cache.load(f"{a.gate_dir}/{name}__eval.npz")
its_g = [it for it in side["items"] if it.get("variant") == "SD_TYPO"]
keep = np.array([it.get("variant") == "SD_TYPO" for it in side["items"]])
ANSg = side["_extra"]["ANS"].astype(np.float32)[keep]
y_r = np.array([judge.item_refused(it) for it in its_g], int)
r_ref = verdict.dim_direction(ANSg[:, -1], y_r).numpy()
say(f"refusal score from {len(its_g)} cached items, {y_r.mean():.2f} refused, labels from {judge.label_source(its_g)}")

CARRIER = side["prompt"]


# ---- the scenarios
def mm_items(harmful, n=6):
    return [r for r in safedata.mm_safetybench(60, "SD_TYPO", harmful=harmful, seed=5) if r["image_path"]][:n]


scen = []
ref_items = [it for it in its_g if it["harmful"] and judge.item_refused(it)][:2]
com_items = [it for it in its_g if it["harmful"] and not judge.item_refused(it)][:2]
for it in ref_items:
    scen.append(dict(tag="jailbreak the model refuses", path=it["path"], q=CARRIER, kind="refusal"))
for it in com_items:
    scen.append(dict(tag="jailbreak the model answers", path=it["path"], q=CARRIER, kind="refusal"))

pool = cocodata.concept_items(300, min_area=0.06, seed=77)
seen = set()
for it in pool:
    if it["image_id"] in seen:
        continue
    seen.add(it["image_id"])
    pres = sorted(cocodata.present(it["image_id"]))
    w = it["concept"]
    if len([s for s in scen if s["kind"] == "present"]) < 2 and cocodata.concept_token_ids(m.tokenizer, w):
        scen.append(dict(tag=f"an object that is there ({w})", path=it["path"], q=f"Is there a {w} in this picture? Answer yes or no.", kind="present", word=w))
        continue
    if len([s for s in scen if s["kind"] == "absent"]) < 2:
        score = {}
        for p in pres:
            for c, k in cocodata.cooccurring(p).items():
                if c not in pres and cocodata.concept_token_ids(m.tokenizer, c):
                    score[c] = score.get(c, 0) + int(k)
        if score:
            w2 = max(score, key=score.get)
            scen.append(dict(tag=f"an object that is not there but usually would be ({w2})", path=it["path"],
                             q=f"Is there a {w2} in this picture? Answer yes or no.", kind="absent", word=w2))
    if len(scen) >= 8:
        break

for it in clevrdata.pointer_items(14, n_obj=(3, 7), min_sep=0.06, balance=5.0, limit=3):
    scen.append(dict(tag="a two-step question", path=it["path"], q=it["question_main"] + " Answer with one word.",
                     kind="word", word=it["answer_main"]))
    break
say(f"{len(scen)} scenarios: " + "; ".join(s["tag"] for s in scen))

# ---- the average state per image position, under each prompt family
def per_position_means(paths, prompt, n=None):
    MU, cnt = {}, 0
    with torch.no_grad():
        for p in paths[: n or a.n_mu]:
            b = m.chat(prompt, [corpus.load_image(p, square=448)])
            pos = readout.image_positions(m, b)
            with ActivationRecorder(m.layers, at=L4) as rec:
                m.forward(b)
            for l in L4:
                h = rec.activations[l]
                h = h[0, pos.to(h.device)].float().cpu()
                if l not in MU:
                    MU[l] = torch.zeros_like(h)
                if MU[l].shape == h.shape:
                    MU[l] += h
            cnt += 1
    return {l: MU[l] / max(cnt, 1) for l in MU}


FAM_REF = [it["path"] for it in its_g[-(a.n_fit_family + a.n_mu):]]
FAM_COCO = [it["path"] for it in pool[100: 100 + a.n_fit_family + a.n_mu]]
mu_paths = FAM_REF[-a.n_mu:]
MU_CARRIER = per_position_means(mu_paths, CARRIER)
coco_paths = FAM_COCO[-a.n_mu:]
MU_COCO = per_position_means(coco_paths, "Is there a dog in this picture? Answer yes or no.")
P = MU_CARRIER[L4[0]].shape[0]; GRID = int(round(math.sqrt(P)))
say(f"{P} image positions = {GRID}x{GRID}")


# ---- one read map per prompt family, fitted on unlabelled states of that family.
# The global map was fitted on COCO photographs under generic prompts, which is the wrong
# distribution for a yes-or-no question or for a jailbreak carrier, and a map fitted
# elsewhere is worse than no map at all. Refitting costs a few hundred forward passes
# and no labels, which is exactly the recipe this page recommends.
def fit_family(paths, prompt_of):
    A = []
    with torch.no_grad():
        for p_ in paths:
            b = m.chat(prompt_of(p_), [corpus.load_image(p_, square=448)])
            with ActivationRecorder(m.layers, at=list(range(m.n_layers))) as rec:
                m.forward(b)
            A.append(np.stack([rec.activations[l][0, -1].float().cpu().numpy() for l in range(m.n_layers)]))
    return reader.fit(np.stack(A).astype(np.float32), ALLL, meta=dict(repo=a.repo, n=len(A)))


t0 = time.time()
RD_FAM = {"refusal": fit_family(FAM_REF[: a.n_fit_family], lambda p_: CARRIER),
          "yesno": fit_family(FAM_COCO[: a.n_fit_family],
                              lambda p_, _r=random.Random(3): f"Is there a {_r.choice(BANK_WORDS)} in this picture? Answer yes or no.")}
RD_FAM["present"] = RD_FAM["absent"] = RD_FAM["yesno"]


def clevr_colour_prompts(n):
    """CLEVR renders with a one-word colour question, for the two-step page's map.

    The two-step scenario asks for a colour word, which is neither the jailbreak
    family nor the yes-or-no family; read through the yes-or-no map it comes out
    as “yes” and “no” at every depth, which says more about the map than about
    the model.
    """
    out = []
    for sc_ in clevrdata.scenes():
        if not clevrdata.have(sc_):
            continue
        objs = sc_["objects"]
        for i, o in enumerate(objs):
            others = [x for j, x in enumerate(objs) if j != i]
            if all(any(x[k] != o[k] for k in ("size", "material", "shape")) for x in others):
                out.append((clevrdata.image_path(sc_),
                            f"What colour is {clevrdata.describe(o, colour=False)}? Answer with one word."))
                break
        if len(out) >= n:
            break
    return out


_CQ = clevr_colour_prompts(a.n_fit_family)
RD_FAM["word"] = fit_family([p_ for p_, _q in _CQ], lambda p_, _m=dict(_CQ): _m[p_])
say(f"the two-step page reads through a map fitted on {len(_CQ)} CLEVR one-word questions")
say(f"read maps fitted per prompt family on {a.n_fit_family} unlabelled states each ({(time.time() - t0) / 60:.1f} min)")


def score_fns(s):
    """The scores this scenario is read through, the first one leading, since the write maps
    are drawn for the leading score and it should be the one the question is about."""
    refusal = ("refusal", lambda hL, lg: torch.as_tensor(hL).float() @ torch.as_tensor(r_ref).float().to(torch.as_tensor(hL).device))
    out = [refusal] if s["kind"] == "refusal" else []
    if s["kind"] in ("present", "absent"):
        out.append(("yes over no", lambda hL, lg: torch.logsumexp(torch.log_softmax(lg.float(), -1)[..., YES], -1)
                    - torch.logsumexp(torch.log_softmax(lg.float(), -1)[..., NO], -1)))
    if s.get("word") and cocodata.concept_token_ids(m.tokenizer, s["word"]):
        t = int(cocodata.concept_token_ids(m.tokenizer, s["word"])[0])
        out.append((f"the word “{s['word']}”", lambda hL, lg, t=t: torch.log_softmax(lg.float(), -1)[..., t]))
    out.append(("refusal openers", lambda hL, lg: torch.logsumexp(torch.log_softmax(lg.float(), -1)[..., SR], -1)
                - torch.logsumexp(torch.log_softmax(lg.float(), -1)[..., SC], -1)))
    if s["kind"] != "refusal":
        out.append(refusal)
    return out


def read_scenario(s):
    RD = RD_FAM.get(s["kind"], RD_GLOBAL)
    img = corpus.load_image(s["path"], square=448)
    b = m.chat(s["q"], [img]); pos = readout.image_positions(m, b); S = b["input_ids"].shape[1]
    reply = sc.generate(b, max_new_tokens=a.gen)
    with torch.no_grad(), ActivationRecorder(m.layers, at=list(range(m.n_layers))) as rec:
        m.forward(b)
        ANS = np.stack([rec.activations[l][0, S - 1].float().cpu().numpy() for l in range(m.n_layers)])
    # word strips and score traces
    fns = score_fns(s)
    strips, traces = {"V-Read": {}, "logit lens": {}}, {k: {"V-Read": [], "logit lens": []} for k, _ in fns}
    with torch.no_grad():
        for l in ALLL:
            for tname, H in (("V-Read", RD.to_final(ANS[l][None], l)), ("logit lens", ANS[l][None])):
                hL = torch.as_tensor(H).to(m.device).to(torch.bfloat16)
                lg = m.unembed(hL).float()
                strips[tname][l] = [m.tokenizer.decode([int(t)]).strip() for t in lg[0].argsort(descending=True)[:3]]
                for k, fn in fns:
                    traces[k][tname].append(float(np.asarray(fn(torch.as_tensor(H), lg.cpu())).reshape(-1)[0]))
    # write maps for the leading score
    key, fn0 = fns[0][0], fns[0][1]
    MU = MU_CARRIER if s["kind"] == "refusal" else MU_COCO
    maps = {}
    with ActivationRecorder(m.layers, at=[*L4, LAST], start_graph_at=min(L4)) as rec, torch.enable_grad():
        m.forward(b)
        A = [rec.activations[l] for l in L4]
        hL = rec.activations[LAST][0, S - 1]
        lg = m.unembed(hL[None])[0]
        val = fn0(hL[None], lg[None]).reshape(())
        g = torch.autograd.grad(val, A)
    for i, l in enumerate(L4):
        gp = g[i][0, pos.to(g[i].device)].float().cpu()
        hp = A[i][0, pos.to(A[i].device)].detach().float().cpu() - MU[l]
        maps[l] = (gp * hp).sum(-1).numpy().reshape(GRID, GRID)
    return dict(reply=reply, strips=strips, traces=traces, maps=maps, score=key, img=img)


def draw(s, r, path):
    fns = list(r["traces"])
    fig = plt.figure(figsize=(11.2, 8.4))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.25, 0.95, 1.0], width_ratios=[1.15, 1, 1, 1, 1, 1],
                          hspace=0.42, wspace=0.28, left=0.045, right=0.985, top=0.93, bottom=0.06)
    # the picture
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(r["img"]); ax.set_axis_off()
    ax.set_title(s["tag"], loc="left", fontsize=10, color=plot.INK)
    q = s["q"] if len(s["q"]) < 78 else s["q"][:75] + "..."
    rep = r["reply"].replace("\n", " ")[:150]
    fig.text(0.045, 0.655, f"question: {q}", fontsize=8.6, color=plot.INK2)
    fig.text(0.045, 0.632, f"the model says: “{rep}…”", fontsize=8.6, color=plot.INK2)
    # the word strip
    axs = fig.add_subplot(gs[0, 1:]); axs.set_axis_off()
    show = [l for l in ALLL][:: max(1, len(ALLL) // 8)][:8]
    for j, l in enumerate(show):
        x = 0.10 + 0.90 * j / max(len(show) - 1, 1)
        axs.text(x, 1.0, f"{frac[l]:.2f}", ha="center", va="top", fontsize=8.5, color=plot.INK3)
        for i, w in enumerate(r["strips"]["V-Read"][l]):
            axs.text(x, 0.82 - 0.13 * i, w[:11] or "·", ha="center", va="top", fontsize=9.5 - 0.8 * i,
                     color=plot.INK if i == 0 else plot.INK2)
        for i, w in enumerate(r["strips"]["logit lens"][l]):
            axs.text(x, 0.36 - 0.13 * i, w[:11] or "·", ha="center", va="top", fontsize=9.5 - 0.8 * i,
                     color=plot.INK2 if i == 0 else plot.INK3)
    axs.text(0.055, 0.86, "V-Read", ha="right", va="top", fontsize=9, color=plot.SERIES[0])
    axs.text(0.055, 0.40, "logit lens", ha="right", va="top", fontsize=9, color=plot.SERIES[1])
    axs.text(0.55, 1.13, "what the model is about to say, by depth", ha="center", fontsize=9.5, color=plot.INK2)
    axs.set_xlim(-0.06, 1.06); axs.set_ylim(0, 1.18)
    # the score traces
    xs = [frac[l] for l in ALLL]
    # only the item's own score: the opener margin's final value is small enough
    # that dividing by it turns ordinary noise into a large excursion, which
    # says more about the divisor than about either instrument
    for j, k in enumerate(fns[:1]):
        ax = fig.add_subplot(gs[1, 1:4])
        seen = []
        for tname, col, lw in (("V-Read", plot.SERIES[0], 1.7), ("logit lens", plot.SERIES[1], 1.2)):
            v = np.asarray(r["traces"][k][tname], float)
            final = v[-1] if abs(v[-1]) > 1e-6 else (np.sign(v[-1]) or 1.0)
            ax.plot(xs, v / final, color=col, lw=lw)
            seen.append(v / final)
        ax.axhline(1, color=plot.INK3, lw=0.6, ls=":")
        ax.axhline(0, color=plot.INK3, lw=0.6)
        # limits from the curves themselves: a reading that overshoots its own
        # final value early is a real thing to show, not something to clip off
        y = np.concatenate(seen)
        lo, hi = min(-0.25, float(np.min(y))), max(1.25, float(np.percentile(y, 97)))
        pad = 0.06 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{k}, as a fraction of the value it ends at",
                     loc="left", fontsize=9, color=plot.INK2)
        ax.set_xlabel("depth", fontsize=8.5)
        ax.text(0.02, 0.93, "V-Read", transform=ax.transAxes, fontsize=8.5, color=plot.SERIES[0], va="top")
        ax.text(0.02, 0.80, "logit lens", transform=ax.transAxes, fontsize=8.5, color=plot.SERIES[1], va="top")
        plot.despine(ax)
    # the write maps
    for j, l in enumerate(L4):
        ax = fig.add_subplot(gs[2, j + 1]); M = r["maps"][l]
        v = float(np.percentile(np.abs(M), 99)) or 1.0
        ax.imshow(np.clip(M, -v, v), cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"depth {frac[l]:.2f}", fontsize=8.5, color=plot.INK2)
    axm = fig.add_subplot(gs[2, 0]); axm.set_axis_off()
    axm.text(1.0, 0.6, f"V-Write on\n{r['score']}\nred pushes it up,\nblue pushes it down", ha="right", va="center",
             fontsize=8.6, color=plot.INK2)
    plot.save(fig, path); plt.close(fig)


out = []
for i, s in enumerate(scen):
    r = read_scenario(s)
    p = f"{a.figs}/{name}_scenario_{i}_{s['kind']}.png"
    draw(s, r, p)
    out.append(dict(tag=s["tag"], kind=s["kind"], question=s["q"], reply=r["reply"][:300], png=os.path.basename(p),
                    top_words={str(frac[l]): r["strips"]["V-Read"][l] for l in ALLL},
                    top_words_logit={str(frac[l]): r["strips"]["logit lens"][l] for l in ALLL},
                    traces={k: {t: v for t, v in r["traces"][k].items()} for k in r["traces"]},
                    depths=[frac[l] for l in ALLL]))
    say(f"  {s['tag']}: “{r['reply'][:60]}…”")

json.dump(dict(repo=a.repo, scenarios=out, label_source=judge.label_source(its_g)),
          open(f"{a.out}/{name}_scenarios.json", "w"), indent=1, default=float)
say(f"\nwrote {a.out}/{name}_scenarios.json and {len(out)} figures  ({(time.time() - T0) / 60:.1f} min)")
