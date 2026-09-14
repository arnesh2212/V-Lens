"""Where the Jacobian earns its keep.

Near the output the lens direction and the raw unembedding row are almost the
same vector, so erasing either does the same thing.  Early in the network they
diverge.  So restrict the erasure to a band of layers and compare the two
directions band by band: if the transport is real, erasing the lens direction in
an early band should cost naming where erasing the unembedding row in the same
band does not, and the two should converge in the late bands where the cosine
does.

The crossing depth is predicted in advance by the geometry, before any behaviour
is measured, which is what makes this more than a description.
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
from jvlm import cocodata, corpus, readout, tasks

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/depthcross")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lens = (jvlm.lens_from_checkpoint(a.lens) if a.lens.endswith(".ckpt")
        else __import__("jlens").lens.JacobianLens.load(a.lens))
sc = tasks.Scorer(m)
L = list(lens.source_layers)


def band(lo, hi):
    return [l for l in L if lo <= l / (m.n_layers - 1) < hi]


BANDS = {f"{lo:.2f}-{hi:.2f}": band(lo, hi)
         for lo, hi in ((0.10, 0.35), (0.35, 0.55), (0.55, 0.75), (0.75, 1.01))}
BANDS = {k: v for k, v in BANDS.items() if v}
print(m, lens, "\nbands:", {k: (v[0], v[-1]) for k, v in BANDS.items()}, flush=True)

items = cocodata.concept_items(a.n, min_area=0.08, seed=11,
                               hard=True, tokenizer=m.tokenizer)
rows, geo = [], []
t0 = time.time()
_cache: dict = {}


def dirs(c):
    if c in _cache:
        return _cache[c]
    ids = cocodata.concept_token_ids(m.tokenizer, c)
    W = m._lm_head.weight[ids].detach().float()
    Wn = W / W.norm(dim=-1, keepdim=True)
    lensv, cos = {}, {}
    for l in L:
        V = readout.lens_vectors(m, lens, ids, l)
        lensv[l] = V
        cos[l] = float(torch.nn.functional.cosine_similarity(V, Wn, dim=-1).mean())
    _cache[c] = (lensv, {l: Wn for l in L}, cos)
    return _cache[c]


for k, it in enumerate(items):
    c, d = it["concept"], it["distractor"]
    im = corpus.load_image(it["path"])
    lensv, unemb, cos = dirs(c)
    for bname, layers in BANDS.items():
        geo.append(dict(concept=c, band=bname,
                        cos=float(np.mean([cos[l] for l in layers]))))

    def multi(prompts, _iv):
        return sc.multi_last_logits(_iv.set_batch(m.chat_batch(prompts, [im])))

    for bname, layers in [("none", None)] + list(BANDS.items()):
        for which, spec in (("lens", lensv), ("unembed", unemb)):
            if bname == "none" and which == "unembed":
                continue
            iv = (readout.NoIntervention() if bname == "none" else
                  readout.Intervention(m, {l: spec[l] for l in layers},
                                       channel="all", mode="ablate", alpha=1.0))
            with iv:
                r = tasks.run_two_channels(sc, c, d, lambda p, _iv=iv: multi(p, _iv))
            rows.append(dict(i=k, concept=c, distractor=d, band=bname,
                             direction=which, cond=f"{which}_{bname}",
                             cos=float(np.mean([cos[l] for l in layers]))
                             if layers else float("nan"), **r))
    if k % 20 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
pd.DataFrame(geo).to_parquet(f"{a.out}/{tag}_geometry.parquet")
agg = df.groupby(["band", "direction"]).agg(
    n=("i", "count"), cos=("cos", "mean"),
    word_correct=("word_correct", "mean"),
    letter_correct=("letter_correct", "mean"),
    word_margin=("word_margin", "mean"))
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", agg.round(3).to_string())

clean = df[df.band == "none"].word_correct.mean()
gap = {}
for b in BANDS:
    wl = df[(df.band == b) & (df.direction == "lens")].word_correct.mean()
    wu = df[(df.band == b) & (df.direction == "unembed")].word_correct.mean()
    gap[b] = dict(cos=float(df[(df.band == b)].cos.mean()),
                  lens_word=float(wl), unembed_word=float(wu),
                  advantage=float(wu - wl))
json.dump({"repo": a.repo, "clean_word": float(clean), "bands": gap,
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_stats.json", "w"), indent=2)
print("\n", json.dumps(gap, indent=2))
