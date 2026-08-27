"""Is the direct estimator the same object, and how much cheaper?

Three checks.

EXACTNESS.  On a single prompt the two routes must agree to numerical precision:
building the site-to-answer Jacobian and contracting it with the norm-corrected
unembedding row is, by the chain rule, the same as differentiating that row's
logit directly.  If they disagree, one of them is wrong.

COST.  The matrix route needs ``d_model`` backward passes per prompt; the direct
route needs one per token.  Measured, not asserted.

FAITHFULNESS.  Over a corpus the two are *not* the same: the matrix route
averages the transport and then contracts, the direct route averages the
contraction.  They differ by a covariance between the transport and the operating
point, and the direct one is the object that gets used.  We report the angle
between them and the per-layer stability the direct route makes visible.
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
import torch.nn.functional as F

import jvlm
from jvlm import cocodata, corpus, direct, readout
from jvlm.vlens import VLens, norm_jacobian_apply

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--vlens", default=None, help="matrix-route V-Lens, if fitted")
ap.add_argument("--concepts", type=int, default=10)
ap.add_argument("--per-concept", type=int, default=24)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/direct")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
LAYERS = [l for l in range(max(m.n_deepstack, 1), m.n_layers - 1)][::3]
print(m, f"\nlayers {LAYERS}", flush=True)
report: dict = {"repo": a.repo, "layers": LAYERS}

# ------------------------------------------------------------------ exactness
it0 = cocodata.concept_items(1, min_area=0.1, seed=5, tokenizer=m.tokenizer)[0]
im0 = corpus.load_image(it0["path"])
b0 = m.chat("Describe this image.", [im0])
ids0 = cocodata.concept_token_ids(m.tokenizer, it0["concept"])[:3]
mask0 = m.image_mask(b0["input_ids"])[0]

t0 = time.time()
g_direct = direct.logit_gradients(m, b0, ids0, LAYERS)
t_direct_one = time.time() - t0

t0 = time.time()
J1, _, _, _ = jvlm.jacobian_for_batch(m, b0, LAYERS, dim_batch=8,
                                      position_policy="image",
                                      target_policy="answer")
t_matrix_one = time.time() - t0

acts0 = readout.residuals(m, b0, [m.n_layers - 1])
h_final0 = acts0[m.n_layers - 1][-1]
W0 = norm_jacobian_apply(m, m._lm_head.weight[ids0].detach().float(), h_final0)
cos_exact = {}
for l in LAYERS:
    v_mat = (W0 @ J1[l].to(W0.device).float())
    v_dir = g_direct[l].to(W0.device)
    cos_exact[l] = float(F.cosine_similarity(v_mat, v_dir, dim=-1).mean())
report["exactness_cosine"] = cos_exact
report["seconds_one_prompt"] = {"direct": round(t_direct_one, 2),
                                "matrix": round(t_matrix_one, 2),
                                "speedup": round(t_matrix_one / max(t_direct_one, 1e-6), 1)}
print("\nEXACTNESS  cosine(matrix, direct) per layer, one prompt:")
print("  " + "  ".join(f"L{l}:{c:.4f}" for l, c in cos_exact.items()))
print(f"COST  one prompt, {len(ids0)} tokens: direct {t_direct_one:.2f}s  "
      f"matrix {t_matrix_one:.2f}s  ({t_matrix_one/max(t_direct_one,1e-6):.0f}x)",
      flush=True)

# ------------------------------------------------- corpus directions, directly
pool = cocodata.concept_items(6000, min_area=0.05, seed=3)
by_c: dict[str, list] = {}
for it in pool:
    by_c.setdefault(it["concept"], []).append(it)
concepts = sorted(by_c, key=lambda c: -len(by_c[c]))[: a.concepts]
print("concepts:", concepts, flush=True)

matrix_lens = None
if a.vlens and os.path.exists(a.vlens):
    matrix_lens = VLens.load(a.vlens)

rows, dirs = [], {}
t_start = time.time()
for c in concepts:
    ids = cocodata.concept_token_ids(m.tokenizer, c)
    acc = direct.DirectDirections(LAYERS)
    for it in by_c[c][: a.per_concept]:
        b = m.chat("Describe this image.", [corpus.load_image(it["path"])])
        try:
            acc.add(m, b, ids)
        except ValueError:
            continue
    dirs[c] = acc.mean()
    stab = acc.stability()
    for l in LAYERS:
        row = dict(concept=c, layer=l, depth=l / (m.n_layers - 1),
                   n_prompts=acc.n, stability=stab[l])
        if matrix_lens is not None and l in matrix_lens.jacobians:
            v_mat = matrix_lens.vectors(m, ids, l, h_final0)
            row["cos_to_matrix"] = float(
                F.cosine_similarity(v_mat.to(dirs[c][l].device),
                                    dirs[c][l], dim=-1).mean())
        rows.append(row)
    print(f"  {c}: {acc.n} prompts  {time.time()-t_start:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
torch.save({c: {l: v for l, v in d.items()} for c, d in dirs.items()},
           f"{a.out}/{tag}_directions.pt")

cols = [c for c in ("stability", "cos_to_matrix") if c in df.columns]
band = df.assign(b=(df.depth * 4).astype(int) / 4).groupby("b")[cols].mean()
print("\nBY DEPTH:\n", band.round(4).to_string())
report["by_depth"] = band.round(4).to_dict()
report["n_prompts_per_concept"] = a.per_concept
report["total_backward_passes_direct"] = a.concepts * a.per_concept * 5
report["total_backward_passes_matrix"] = a.concepts * a.per_concept * m.d_model
report["minutes"] = round((time.time() - t_start) / 60, 1)
print("\n", json.dumps({k: v for k, v in report.items() if k != "by_depth"},
                       indent=2, default=float))
json.dump(report, open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
