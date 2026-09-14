"""Can a position-averaged transport make a map?

The Jacobian lens transports an activation with one matrix per layer, averaged
over source positions.  Applied to a sentence that is reasonable.  Applied to a
photograph it means every patch is carried by the same operator, and the only
thing distinguishing one patch's reading from another's is the patch's own
activation.

That is a testable claim, and it is the whole argument for putting a region back
into the lens.  For each photograph we compute all three readings at every image
token and ask how much they actually vary from place to place:

    distinct   how many different words are top-ranked across the grid
    agreement  how often two random patches give the same top word
    spread     the standard deviation, across patches, of the photographed
               object's score

A lens whose readings barely move across a picture is not reading the picture.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, "/workspace/vlm/code")
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch

import jvlm
from jvlm import cocodata, corpus, readout, stats as st, vmap

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens", default=None)
ap.add_argument("--n", type=int, default=90)
ap.add_argument("--mean-images", type=int, default=60)
ap.add_argument("--n-layers", type=int, default=5)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/posvar")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lo, hi = max(m.n_deepstack, 1), m.n_layers - 2
LAYERS = sorted({int(round(lo + (hi - lo) * f))
                 for f in np.linspace(0.15, 0.9, a.n_layers)})
J = None
if a.jlens and os.path.exists(a.jlens):
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))
CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
FLAT = sorted({t for v in CAT_IDS.values() for t in v})
POSI = {t: i for i, t in enumerate(FLAT)}
print(m, f"\nlayers {LAYERS}   J-lens: {'yes' if J else 'no'}", flush=True)

PROMPT = "Describe this image."


def prep(p):
    return m.chat(PROMPT, [corpus.load_image(p, square=a.square)])


pool = cocodata.concept_items(6000, min_area=0.05, seed=7)
b0 = prep(pool[0]["path"])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
bank = {l: [] for l in LAYERS}
for it in pool[: a.mean_images]:
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    A = readout.residuals(m, b, LAYERS)
    for l in LAYERS:
        if len(bank[l]) < 12:
            bank[l].append(A[l][ip].float().cpu())
    if all(len(bank[l]) >= 12 for l in LAYERS):
        break
print(f"{P} image tokens; donors ready ({time.time()-t0:.0f}s)", flush=True)

items = cocodata.concept_items(a.n, min_area=0.08, seed=404,
                               tokenizer=m.tokenizer)
rng = np.random.default_rng(5)
rows = []
for k, it in enumerate(items):
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    ids = CAT_IDS[it["concept"]]
    col = [POSI[t] for t in ids]
    A = readout.residuals(m, b, LAYERS)
    DO = vmap.do_map(m, b, FLAT, LAYERS, ip.tolist(),
                     means={l: bank[l][k % len(bank[l])] for l in LAYERS})
    for l in LAYERS:
        h = A[l][ip].float().cpu()
        lg = m.unembed(h.to(m.device)).float().cpu()[:, FLAT].T
        maps = {"logit lens": lg}
        if J is not None and l in J.jacobians:
            jl, _ = readout.lens_logits(m, J, b, layers=[l],
                                        positions=ip.tolist())
            maps["J-lens"] = jl[l].float().cpu()[:, FLAT].T
        maps["VJ-Lens"] = DO[l]
        for name, mm in maps.items():
            top = mm.argmax(0)                       # top word at each patch
            pair = rng.choice(P, size=(400, 2))
            agree = float((top[pair[:, 0]] == top[pair[:, 1]]).float().mean())
            sc_ = mm[col].max(0).values
            rows.append(dict(i=k, lens=name, layer=l,
                             depth=round(l / (m.n_layers - 1), 3),
                             distinct=int(len(torch.unique(top))),
                             agree=agree,
                             spread=float(sc_.std() / sc_.abs().mean()
                                          .clamp_min(1e-6))))
    if k % 15 == 0:
        print(f"  {k}/{len(items)}  {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
agg = df.groupby(["lens", "layer"]).agg(
    n=("i", "count"), distinct=("distinct", "mean"),
    agree=("agree", "mean"), spread=("spread", "mean")).reset_index()
agg.to_csv(f"{a.out}/{tag}_summary.csv", index=False)
print(f"\nHOW MUCH DOES EACH LENS VARY ACROSS A PICTURE?  ({P} patches)\n",
      agg.round(3).to_string(index=False))
best = agg.groupby("lens").agg(distinct=("distinct", "mean"),
                               agree=("agree", "mean"),
                               spread=("spread", "mean"))
print("\n", best.round(3).to_string())
json.dump({"repo": a.repo, "layers": LAYERS, "n_patches": P,
           "n": int(df.i.nunique()),
           "by_layer": st.jsonable(agg.round(4).to_dict("index")),
           "overall": st.jsonable(best.round(4).to_dict("index")),
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
print("done", flush=True)
