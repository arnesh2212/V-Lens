"""How big should a region be?

The region is the only parameter Atlas has that the lenses it generalises do
not, so it deserves its own measurement.  At one extreme a region is a single
image token, which is too small: neighbouring patches carry the same object and
covering one changes almost nothing.  At the other it is the whole frame, which
is the Jacobian lens's choice and has no place in it at all.

We sweep the region from one token to the whole grid and measure, at each size,
how well the reading predicts what covering that same region really does.  If
there is a sweet spot it is the paper's answer to "what is the right unit for a
vision lens", and it should sit where an object sits.
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
from scipy import stats as sps

import jvlm
from jvlm import cocodata, corpus, readout, stats as st, vmap

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--n", type=int, default=110)
ap.add_argument("--mean-images", type=int, default=80)
ap.add_argument("--sizes", type=int, nargs="*", default=[1, 2, 4, 9, 16, 36, 64])
ap.add_argument("--n-layers", type=int, default=3)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/region")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lo, hi = max(m.n_deepstack, 1), m.n_layers - 2
LAYERS = sorted({int(round(lo + (hi - lo) * f))
                 for f in np.linspace(0.3, 0.7, a.n_layers)})
CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
FLAT = sorted({t for v in CAT_IDS.values() for t in v})
POSI = {t: i for i, t in enumerate(FLAT)}
print(m, f"\nlayers {LAYERS}   sizes {a.sizes}", flush=True)

PROMPT = "Describe this image."


def prep(p):
    return m.chat(PROMPT, [corpus.load_image(p, square=a.square)])


def margin(lg, ids):
    v = lg[FLAT].float()
    return float(max(v[POSI[t]] for t in ids) - v.mean())


pool = cocodata.concept_items(6000, min_area=0.05, seed=7)
b0 = prep(pool[0]["path"])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
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
SIZES = [s for s in a.sizes if s <= P]
print(f"{P} tokens on a {G}x{G} grid ({time.time()-t0:.0f}s)", flush=True)

items = cocodata.concept_items(a.n, min_area=0.08, seed=2027,
                               tokenizer=m.tokenizer)
rng = np.random.default_rng(3)
rows = []
for k, it in enumerate(items):
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    ids = CAT_IDS[it["concept"]]
    col = [POSI[t] for t in ids]
    A = readout.residuals(m, b, LAYERS + [m.n_layers - 1])
    base = margin(m.unembed(A[m.n_layers - 1][-1:].to(m.device))[0], ids)
    if base <= 0:
        continue
    DO = vmap.do_map(m, b, FLAT, LAYERS, ip.tolist(),
                     means={l: bank[l][k % len(bank[l])] for l in LAYERS})
    for s in SIZES:
        # a square block of the grid, so a region is a place and not a scatter
        side = int(round(s ** 0.5))
        side = max(1, min(side, G))
        r0 = int(rng.integers(0, G - side + 1))
        c0 = int(rng.integers(0, G - side + 1))
        tk = [(r0 + dr) * G + (c0 + dc)
              for dr in range(side) for dc in range(side)]
        for l in LAYERS:
            pred = float(vmap.normalise(DO[l])[col][:, tk].max(0).values.mean())
            pos = ip[torch.tensor(tk)].to(m.device)
            vals = bank[l][k % len(bank[l])][torch.tensor(tk)].to(m.device)

            def hook(mod, inp, out, _p=pos, _v=vals):
                tup = not torch.is_tensor(out)
                h = (out[0] if tup else out).clone()
                h[0, _p] = _v.to(h.dtype)
                return (h, *out[1:]) if tup else h

            hd = m.layers[l].register_forward_hook(hook)
            try:
                with torch.no_grad():
                    AA = readout.residuals(m, b, [m.n_layers - 1])
                    after = margin(m.unembed(
                        AA[m.n_layers - 1][-1:].to(m.device))[0], ids)
            finally:
                hd.remove()
            rows.append(dict(i=k, size=side * side, side=side, layer=l,
                             pred=pred, drop=base - after,
                             frac=side * side / P))
    if k % 15 == 0:
        print(f"  {k}/{len(items)}  {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
out_rows = []
for s in sorted(df["size"].unique()):
    for l in LAYERS:
        d = df[(df["size"] == s) & (df.layer == l)]
        if len(d) > 10:
            out_rows.append(dict(
                size=int(s), side=int(d.side.iloc[0]), layer=l,
                n=len(d), frac=float(d.frac.iloc[0]),
                spearman=float(sps.spearmanr(d.pred, d["drop"]).statistic),
                mean_drop=float(d["drop"].mean())))
sm = pd.DataFrame(out_rows)
sm.to_csv(f"{a.out}/{tag}_summary.csv", index=False)
print("\nHOW WELL A REGION'S READING PREDICTS COVERING THAT REGION\n",
      sm.pivot_table(index="size", columns="layer", values="spearman")
      .round(3).to_string())
print("\nand how much covering it actually does\n",
      sm.pivot_table(index="size", columns="layer", values="mean_drop")
      .round(3).to_string())
json.dump({"repo": a.repo, "layers": LAYERS, "grid": G, "n_patches": P,
           "n": int(df.i.nunique()),
           "by_size": st.jsonable(sm.round(4).to_dict("index")),
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
print("done", flush=True)
