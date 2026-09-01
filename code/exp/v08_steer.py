# SPDX-License-Identifier: Apache-2.0
"""Swapping and steering: does writing along the lens direction move the answer?

The workspace paper's test of a coordinate is to write to it and watch the
report follow.  Here: for a COCO photograph whose main object is A and a word
B that is not in it, add B's direction at the image tokens at one layer,
scaled to the local residual norm, and ask "What is the main object in this
image? Answer with one word."  Sweep the dose.  Then the swap: remove the
component along A's direction and add B's.

Four directions per layer, all at the same norm:
  lens        u_B^img, the corpus-mean gradient of B's answer logit at image tokens
  DIM         mean(h | B in picture) - mean(h | B absent) at image tokens (labelled)
  logit-lens  W_U[B], the unembedding row
  random      a matched-norm Gaussian direction

Measured: the first-token log-odds of B over A at the answer slot; the rate
at which the one-word answer becomes B; and benign retention, the rate at
which "Is there a C in the image?" for a present object C is still answered
yes.  Everything at three depths and five doses.  Writes JSON and figures.
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
ap.add_argument("--n", type=int, default=60, help="photographs")
ap.add_argument("--n-fit", type=int, default=60, help="prompts to fit the word directions")
ap.add_argument("--fracs", type=float, nargs="*", default=[0.35, 0.5, 0.65])
ap.add_argument("--alphas", type=float, nargs="*", default=[0.0, 0.5, 1.0, 2.0, 4.0])
ap.add_argument("--targets", nargs="*", default=["dog", "cat", "car", "bus", "pizza", "bird", "horse", "boat"])
ap.add_argument("--out", default="/workspace/vlm/results/steer")
ap.add_argument("--figdir", default="/workspace/vlm/figures/steer")
a = ap.parse_args()

import jvlm  # noqa: E402
from jvlm import axes, cocodata, corpus, depth, plot, readout, tasks, verdict  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402

name = a.repo.split("/")[-1]
os.makedirs(a.out, exist_ok=True); os.makedirs(a.figdir, exist_ok=True)
say = lambda *x: print(*x, flush=True)
m = jvlm.load_vlm(a.repo); sc = tasks.Scorer(m)
LAYERS = depth.layers_at(m, a.fracs)
Q = "What is the main object in this image? Answer with one word."
T0 = time.time()

targets = [w for w in a.targets if cocodata.concept_token_ids(m.tokenizer, w)]
tok_of = {w: int(cocodata.concept_token_ids(m.tokenizer, w)[0]) for w in targets}

# ---- items: photographs whose main object A is in the word bank and B is absent
pool = cocodata.concept_items(3000, min_area=0.10, seed=303)
items = [it for it in pool if it["concept"] in targets and len([w for w in targets if w not in it["present"]]) >= 2][: a.n]
rng = np.random.default_rng(0)
for it in items:
    it["B"] = str(rng.choice([w for w in targets if w not in it["present"]]))
    others = [c for c in it["present"] if c != it["concept"] and cocodata.concept_token_ids(m.tokenizer, c)]
    it["C"] = others[0] if others else None
say(f"{len(items)} photographs; targets {targets}")

# ---- directions
AX = [axes.Axis(f"word:{w}", (lambda lg, h, t=tok_of[w]: lg[t].float()), "word", dict(token=tok_of[w])) for w in targets]
fit_items = cocodata.concept_items(a.n_fit * 3, min_area=0.0, seed=101)[: a.n_fit]
def batches():
    for it in fit_items:
        yield m.chat(str(rng.choice(corpus.GENERIC_PROMPTS)), [corpus.load_image(it["path"], square=448)])
F = verdict.fit(m, batches(), AX, LAYERS, every=30)
VL = verdict.VerdictLens.from_fit(F, dict(repo=a.repo, words=targets))
say(f"word directions fitted in {(time.time() - T0) / 60:.1f} min")

# labelled DIM per word at image tokens: mean over images containing B minus mean over images without
def img_means(items_, layers):
    acc = {l: [] for l in layers}
    for it in items_:
        b = m.chat("Describe this image.", [corpus.load_image(it["path"], square=448)])
        ip = readout.image_positions(m, b)
        A_ = readout.residuals(m, b, layers)
        for l in layers:
            acc[l].append(A_[l][ip].float().mean(0).cpu())
    return {l: torch.stack(v) for l, v in acc.items()}
dim_pool = cocodata.concept_items(2000, min_area=0.05, seed=404)
DIM = {}
for w in targets:
    pos = [it for it in dim_pool if w in it["present"]][:40]; neg = [it for it in dim_pool if w not in it["present"]][:40]
    if len(pos) < 8:
        continue
    Mp, Mn = img_means(pos, LAYERS), img_means(neg, LAYERS)
    DIM[w] = {l: (Mp[l].mean(0) - Mn[l].mean(0)) for l in LAYERS}
say(f"DIM directions for {len(DIM)} words")

def unit(v): return v / v.norm().clamp_min(1e-8)
def directions(w, l):
    d = {"lens": unit(VL.dir(f"word:{w}", "img", l)),
         "logit-lens": unit(readout.unembed_direction(m, tok_of[w]).float().cpu())}
    if w in DIM: d["DIM (labelled)"] = unit(DIM[w][l])
    g = torch.Generator().manual_seed(hash((w, l)) % (2**31)); d["random"] = unit(torch.randn(d["lens"].shape, generator=g))
    return d

# ---- measure
def first_token_margin(batch, wB, wA):
    lg = sc.last_logits(batch)
    return float(lg[tok_of[wB]] - lg[tok_of[wA]])

def one_word(batch):
    return sc.generate(batch, max_new_tokens=4).strip().lower().strip(".,!\"' ")

results = []
for k, it in enumerate(items):
    img = corpus.load_image(it["path"], square=448); A_, B_ = it["concept"], it["B"]
    bq = m.chat(Q, [img])
    base_margin = first_token_margin(bq, B_, A_); base_ans = one_word(bq)
    bc = m.chat(f"Is there a {it['C']} in this image? Answer yes or no.", [img]) if it["C"] else None
    base_ret = (sc.yesno_margin(sc.last_logits(bc)) > 0) if bc is not None else None
    rec = dict(image_id=it["image_id"], A=A_, B=B_, C=it["C"], base_margin=base_margin, base_answer=base_ans, base_retained=base_ret, runs=[])
    for l in LAYERS:
        for dname, d in directions(B_, l).items():
            for alpha in a.alphas:
                if alpha == 0.0:
                    rec["runs"].append(dict(layer=l, frac=depth.to_frac(m, l), direction=dname, mode="add", alpha=0.0,
                                            margin=base_margin, answer=base_ans, said_B=(B_ in base_ans), retained=base_ret)); continue
                for mode in ("add", "swap"):
                    if mode == "add":
                        iv = readout.Intervention(m, {l: d}, channel="img", mode="add", alpha=alpha)
                    else:
                        # swap: erase A's lens component, then add B's, both at the image tokens
                        spec = {l: torch.stack([unit(VL.dir(f"word:{A_}", "img", l)), d])}
                        iv = readout.Intervention(m, {l: unit(VL.dir(f"word:{A_}", "img", l))}, channel="img", mode="ablate", alpha=1.0)
                    with iv:
                        iv.set_batch(bq); mg = first_token_margin(bq, B_, A_); ans = one_word(bq)
                        ret = None
                        if bc is not None:
                            iv.set_batch(bc); ret = bool(sc.yesno_margin(sc.last_logits(bc)) > 0)
                    if mode == "swap":
                        with readout.Intervention(m, {l: d}, channel="img", mode="add", alpha=alpha) as iv2:
                            with readout.Intervention(m, {l: unit(VL.dir(f"word:{A_}", "img", l))}, channel="img", mode="ablate", alpha=1.0) as iv1:
                                iv1.set_batch(bq); iv2.set_batch(bq); mg = first_token_margin(bq, B_, A_); ans = one_word(bq)
                    rec["runs"].append(dict(layer=l, frac=depth.to_frac(m, l), direction=dname, mode=mode, alpha=alpha,
                                            margin=mg, answer=ans, said_B=(B_ in ans), retained=ret,
                                            energy=float(np.mean(iv.energy)) if iv.energy else None))
    results.append(rec)
    if (k + 1) % 10 == 0:
        say(f"  {k + 1}/{len(items)}  {(time.time() - T0) / 60:.1f} min")

json.dump(dict(repo=a.repo, items=results, layers=LAYERS, alphas=a.alphas, targets=targets), open(f"{a.out}/{name}_steer.json", "w"), indent=1, default=float)

# ---- summary tables
say(f"\n{name}: rate at which the one-word answer becomes B (add), by direction, depth and dose:")
import collections
agg = collections.defaultdict(list)
for rec in results:
    for r in rec["runs"]:
        agg[(r["mode"], r["direction"], r["frac"], r["alpha"])].append((r["said_B"], r["margin"] - rec["base_margin"], r["retained"]))
for mode in ("add", "swap"):
    say(f"\n[{mode}]  {'direction':<16}{'depth':>6}" + "".join(f"{'a=%g'%al:>9}" for al in a.alphas) + "   |  d(margin) at max dose  |  retention at max dose")
    for l in LAYERS:
        f = depth.to_frac(m, l)
        for dname in ("lens", "DIM (labelled)", "logit-lens", "random"):
            cells = []
            for al in a.alphas:
                v = agg.get((mode if al else "add", dname, f, al)) or agg.get(("add", dname, f, 0.0))
                cells.append(np.mean([x[0] for x in v]) if v else float("nan"))
            top = agg.get((mode, dname, f, a.alphas[-1]), [])
            dm = np.mean([x[1] for x in top]) if top else float("nan")
            ret = np.mean([x[2] for x in top if x[2] is not None]) if top else float("nan")
            say(f"        {dname:<16}{f:>6.2f}" + "".join(f"{c:>9.2f}" for c in cells) + f"   |  {dm:+7.2f}                |  {ret:5.2f}")

# ---- figure: dose-response at the middle depth, add mode
plot.use()
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw=dict(wspace=0.3))
lmid = LAYERS[len(LAYERS) // 2]; fmid = depth.to_frac(m, lmid)
cols = {"lens": plot.BLUE, "DIM (labelled)": plot.MUTED, "logit-lens": plot.ORANGE, "random": plot.AXIS}
for dname, c in cols.items():
    ys = [np.mean([x[0] for x in (agg.get(("add", dname, fmid, al)) if al else agg.get(("add", dname, fmid, 0.0)))]) for al in a.alphas]
    rs = [np.mean([x[2] for x in (agg.get(("add", dname, fmid, al)) if al else agg.get(("add", dname, fmid, 0.0))) if x[2] is not None]) for al in a.alphas]
    axs[0].plot(a.alphas, ys, color=c, marker="o", label=dname); axs[1].plot(a.alphas, rs, color=c, marker="o", label=dname)
axs[0].set_xlabel("dose (multiples of the local residual norm)"); axs[0].set_ylabel(f"answer becomes the written word\n(depth {fmid:.2f})"); axs[0].set_ylim(-0.02, 1.02)
axs[1].set_xlabel("dose"); axs[1].set_ylabel("unrelated yes/no still correct"); axs[1].set_ylim(-0.02, 1.02)
h, l_ = axs[0].get_legend_handles_labels(); fig.legend(h, l_, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncols=4, frameon=False)
plot.save(fig, f"{a.figdir}/{name}_steer_dose.png")
say(f"wrote {a.out}/{name}_steer.json and {a.figdir}/{name}_steer_dose.png   ({(time.time() - T0) / 60:.1f} min)")
