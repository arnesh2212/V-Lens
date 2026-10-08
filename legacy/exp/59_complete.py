"""Does the lens's arithmetic close?

The claim the integrated form makes is not a correlation.  It is that the
readings over all image positions *sum to* the change in the logit that
replacing those positions with the baseline actually causes.  That is a
statement with a residual, so it can be checked rather than asserted.

For each photograph, each layer, and each of several vocabulary items:

    truth   the real change in the token's logit when every image position is
            replaced by the corpus mean, from a forward pass
    claim   the sum over positions of that token's reading

reported as the relative error |claim - truth| / mean|truth|, for the
first-order reading (``steps = 1``, which is the gradient-times-input form) and
for the integrated reading at several path lengths.  If the integral is doing
what the calculus says, the error falls with the number of steps and the
first-order form is the worst case.

The same run also asks whether the extra fidelity buys anything a reader cares
about: pointing and energy against the object's box, at each path length.
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
from jvlm import cocodata, corpus, readout, stats as st, vmap

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--mean-images", type=int, default=80)
ap.add_argument("--n-layers", type=int, default=3)
ap.add_argument("--steps", type=int, nargs="*", default=[1, 2, 4, 8])
ap.add_argument("--vocab", type=int, default=48)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-frac", type=float, default=0.45)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/complete")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
LAYERS = [int(round(f * (m.n_layers - 1)))
          for f in np.linspace(0.20, 0.85, a.n_layers)]
CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
print(m, f"\nlayers {LAYERS}   path lengths {a.steps}", flush=True)

PROMPT = "Describe this image."


def prep(path):
    return m.chat(PROMPT, [corpus.load_image(path, square=a.square)])


pool = cocodata.concept_items(6000, categories=CATS, min_area=0.04, seed=7)
b0 = prep(pool[0]["path"])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
nm = 0
for it in pool[: a.mean_images]:
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    A = readout.residuals(m, b, LAYERS)
    for l in LAYERS:
        acc[l] += A[l][ip].float().cpu().double()
    nm += 1
MEANS = {l: (acc[l] / max(nm, 1)).float() for l in LAYERS}
print(f"means from {nm}; {P} tokens on a {G}x{G} grid ({time.time()-t0:.0f}s)",
      flush=True)

items = cocodata.concept_items(a.n, categories=CATS, min_area=0.07, seed=808,
                               tokenizer=m.tokenizer)
rows, seen = [], 0
for k, it in enumerate(items):
    c = it["concept"]
    if c not in CAT_IDS:
        continue
    boxes = cocodata.object_boxes(it["image_id"], c)
    if not boxes:
        continue
    inside = sorted({t for bx in boxes for t in cocodata.box_tokens(bx, G)})
    if not (3 <= len(inside) <= int(a.max_frac * P)):
        continue
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    # a vocabulary holding this photograph's object plus a fixed spread of
    # others, so the z-score has something to be a score against
    rng = np.random.default_rng(400 + k)
    others = [x for x in CATS if x != c]
    words = [c] + list(rng.choice(others, size=min(a.vocab - 1, len(others)),
                                  replace=False))
    ids, col = [], []
    for w in words:
        wi = CAT_IDS[w]
        if w == c:
            col = list(range(len(ids), len(ids) + len(wi)))
        ids += wi
    for l in LAYERS:
        for s in a.steps:
            t1 = time.time()
            sc = (vmap.do_map(m, b, ids, [l], ip.tolist(), means={l: MEANS[l]})[l]
                  if s == 1 else
                  vmap.ig_map(m, b, ids, [l], ip.tolist(), means={l: MEANS[l]},
                              steps=s)[l])
            comp = vmap.completeness(m, b, ids, l, ip.tolist(), MEANS[l], sc)
            heat = vmap.normalise(sc)[col].max(0).values
            pt = vmap.pointing(heat, set(inside), G)
            rows.append(dict(i=k, concept=c, layer=l, steps=s,
                             depth=round(l / (m.n_layers - 1), 3),
                             n_tok=len(inside), secs=time.time() - t1,
                             rel_error=comp["rel_error"], corr=comp["corr"],
                             **pt))
    seen += 1
    if k % 5 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  kept {seen}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
agg = df.groupby(["steps", "layer"]).agg(
    n=("i", "count"), rel_error=("rel_error", "mean"), corr=("corr", "mean"),
    point=("hit", "mean"), point5=("hit5", "mean"),
    energy=("mass_in", "mean"), chance=("chance", "mean"),
    secs=("secs", "mean")).reset_index()
agg.to_csv(f"{a.out}/{tag}_summary.csv", index=False)
print("\nCLOSURE AND POINTING BY PATH LENGTH\n",
      agg.round(4).to_string(index=False), flush=True)

over = df.groupby("steps").agg(
    rel_error=("rel_error", "mean"), corr=("corr", "mean"),
    point=("hit", "mean"), energy=("mass_in", "mean"),
    secs=("secs", "mean"))
print("\nOVER ALL LAYERS\n", over.round(4).to_string(), flush=True)

out = {"repo": a.repo, "n": int(df.i.nunique()), "grid": G, "P": P,
       "layers": LAYERS, "n_layers": m.n_layers, "steps": a.steps,
       "by_step_layer": agg.round(5).to_dict("records"),
       "over": over.round(5).reset_index().to_dict("records"),
       "minutes": round((time.time() - t0) / 60, 1)}
json.dump(st.jsonable(out), open(f"{a.out}/{tag}_stats.json", "w"), indent=2,
          default=float)
print("done", flush=True)
