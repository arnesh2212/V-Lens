"""The ablation ladder from the reference lens to V-Lens.

Four terms separate them -- the target of the cotangent (T), the positions the
average runs over (S), the point the expansion is taken about (M), and whether
the final norm is differentiated or re-applied (N).  T and S are properties of
the fit; M and N are properties of the readout, so they can be switched on the
same transport at no cost.

The measurement is the obvious one and it has ground truth: read the lens at the
image tokens of a COCO photograph and ask where the photographed object's own
name sits in the vocabulary.  A lens that answers the question "what is this
patch about to make the model say" should put it near the top; a lens that is
measuring something else should not.

Also reported: how early in depth each lens surfaces the object, and how well it
separates an object that is in the picture from one that is not.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, "code")
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch

import jvlm
from jvlm import cocodata, corpus, readout
from jvlm.vlens import VLens

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens", required=True, help="the reference fit")
ap.add_argument("--vlens", required=True, help="the answer-targeted, image-sourced fit")
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--mean-n", type=int, default=40)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/ladder")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
J = VLens.from_jacobian_lens(
    jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
    else __import__("jlens").lens.JacobianLens.load(a.jlens))
V = (VLens.load(a.vlens) if not a.vlens.endswith(".ckpt")
     else VLens.from_jacobian_lens(jvlm.lens_from_checkpoint(a.vlens)))
LAYERS = [l for l in J.source_layers if l in V.jacobians][::2]
print(m, f"\nladder layers {LAYERS}", flush=True)

# One mean, estimated on images disjoint from the evaluation set and shared by
# both lenses, so the centring term is not an advantage handed to one of them.
mean_items = cocodata.concept_items(a.mean_n, min_area=0.06, seed=777)
acc = {l: torch.zeros(m.d_model) for l in LAYERS}
nm = 0
for it in mean_items:
    b = m.chat("Describe this image.", [corpus.load_image(it["path"])])
    ipos = readout.image_positions(m, b)
    if len(ipos) == 0:
        continue
    acts = readout.residuals(m, b, LAYERS)
    for l in LAYERS:
        acc[l] += acts[l][ipos].mean(0).cpu()
    nm += 1
MEAN = {l: acc[l] / max(nm, 1) for l in LAYERS}
for lens in (J, V):
    lens.means = {l: MEAN[l] for l in LAYERS}
print(f"mean image-token activation from {nm} held-out photographs", flush=True)

RUNGS = [
    ("reference",        J, False, False),   # W_U norm(J h)               -- as published
    ("+centre",          J, True,  False),   # W_U norm(J (h - mean))      -- M
    ("+centre +norm",    J, True,  True),    # W_U D_x J (h - mean)        -- M, N
    ("+target +source",  V, False, False),   # W_U norm(A h)               -- T, S
    ("+T +S +centre",    V, True,  False),   # W_U norm(A (h - mean))      -- T, S, M
    ("V-Lens",           V, True,  True),    # W_U D_x A (h - mean)        -- all four
]

CATS = cocodata.SINGLE_WORD
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
FLAT = sorted({t for v in CAT_IDS.values() for t in v})
POS = {t: i for i, t in enumerate(FLAT)}

items = cocodata.concept_items(a.n, min_area=0.08, seed=7, tokenizer=m.tokenizer)
rows = []
t0 = time.time()
for k, it in enumerate(items):
    b = m.chat("Describe this image.", [corpus.load_image(it["path"])])
    ipos = readout.image_positions(m, b)
    if len(ipos) == 0:
        continue
    acts = readout.residuals(m, b, LAYERS)
    h_final = acts[m.n_layers - 1][-1]
    here = set(it["present"])

    for name, lens, centre, exact in RUNGS:
        for l in LAYERS:
            lg = lens.read(m, acts[l][ipos], l, h_final,
                           centre=centre, exact_norm=exact).cpu()
            best = lg.max(0).values                       # [vocab]
            sub = best[FLAT]
            ranks = (best.unsqueeze(0) > sub.unsqueeze(1)).sum(1)
            for c in CATS:
                r = int(min(ranks[POS[t]] for t in CAT_IDS[c]))
                rows.append(dict(i=k, image_id=it["image_id"], rung=name,
                                 layer=l, depth=l / (m.n_layers - 1), cat=c,
                                 present=int(c in here),
                                 target=int(c == it["concept"]), rank=r))
    if k % 20 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    rk = np.empty_like(order, dtype=float)
    rk[order] = np.arange(len(allv))
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2)
                 / (len(pos) * len(neg)))


summ = []
for name, *_ in RUNGS:
    d = df[df.rung == name]
    for l in LAYERS:
        dl = d[d.layer == l]
        tgt = dl[dl.target == 1]["rank"]
        summ.append(dict(
            rung=name, layer=l, depth=l / (m.n_layers - 1),
            median_rank_target=float(tgt.median()),
            top10_target=float((tgt < 10).mean()),
            top1_target=float((tgt == 0).mean()),
            auc=auc(-dl[dl.present == 1]["rank"].values.astype(float),
                    -dl[dl.present == 0]["rank"].values.astype(float))))
S = pd.DataFrame(summ)
S.to_csv(f"{a.out}/{tag}_summary.csv", index=False)

best = S.loc[S.groupby("rung").auc.idxmax()].set_index("rung")
tab = S.groupby("rung").agg(best_auc=("auc", "max"),
                            best_top10=("top10_target", "max"),
                            best_rank=("median_rank_target", "min"))
tab["best_depth"] = best["depth"]
tab = tab.reindex([r[0] for r in RUNGS])
print("\n", tab.round(4).to_string())

earliest = {}
for name, *_ in RUNGS:
    d = S[(S.rung == name) & (S.top10_target > 0.25)]
    earliest[name] = float(d.depth.min()) if len(d) else float("nan")
print("\nearliest depth putting the object in the readout's top 10 on a "
      "quarter of photographs:")
for k_, v in earliest.items():
    print(f"  {k_:18s} {v}")
json.dump({"repo": a.repo, "n": int(df.i.nunique()),
           "table": tab.round(4).to_dict(), "earliest_depth": earliest,
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
