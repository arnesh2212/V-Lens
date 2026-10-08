# SPDX-License-Identifier: Apache-2.0
"""Fit a tuned lens for one model from a cache, and report its KL against the model.

The tuned lens is the strongest lens baseline, so it is fitted once per model
on a fixed corpus of images under a fixed prompt, saved beside the Jacobian
lens, and read by everything else through :mod:`vlens.lenses`.  The ridge
solution is closed form; ``--kl-epochs`` refines it on the GPU with the
original paper's objective.

Writes ``results/lenses/<model>__tuned.pt`` and a JSON with the per-layer KL
of the logit lens (identity) and the tuned lens, so the improvement the tuned
lens buys on this model is on record before it is used as a baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--n", type=int, default=400, help="COCO images to fit on")
ap.add_argument("--every", type=int, default=1, help="every k-th usable layer")
ap.add_argument("--prompt", default="Describe this image.")
ap.add_argument("--lam", type=float, default=1e-2)
ap.add_argument("--max-rows", type=int, default=150_000)
ap.add_argument("--kl-epochs", type=int, default=0)
ap.add_argument("--out", default="/workspace/vlm/results/lenses")
ap.add_argument("--cache-dir", default="/workspace/vlm/results/cache")
ap.add_argument("--reuse-cache", action="store_true")
a = ap.parse_args()

import vlens  # noqa: E402
from vlens.lens import cache, depth, tuned  # noqa: E402
from vlens.data import coco  # noqa: E402

name = a.repo.split("/")[-1]
m = vlens.load_vlm(a.repo)
L = depth.every(m, a.every)
cpath = f"{a.cache_dir}/{name}__tunedfit_n{a.n}.npz"
if not (a.reuse_cache and os.path.exists(cpath)):
    # held-out from every evaluation set: seed 101 is reserved for lens fitting
    items = coco.concept_items(a.n * 3, min_area=0.0, seed=101)[: a.n]
    cache.collect(m, items, L, a.prompt, cpath, pool=None)
acts, side = cache.load(cpath)
FL = side["final_layer"]
print(f"{name}: {side['n']} images, {side['image_tokens']} tokens, layers {L[0]}..{L[-1]} step {a.every}", flush=True)

lens = tuned.fit_from_cache(acts, FL, lam=a.lam, max_rows=a.max_rows)
kl_ridge = tuned.kl_to_model(lens, m, acts, FL)
if a.kl_epochs:
    lens = tuned.refine_kl(lens, m, acts, FL, epochs=a.kl_epochs)
    kl_final = tuned.kl_to_model(lens, m, acts, FL)
else:
    kl_final = kl_ridge
d = acts[FL].shape[-1]
ident = tuned.TunedLens({l: (torch.eye(d), torch.zeros(d)) for l in lens.layers})
kl_logit = tuned.kl_to_model(ident, m, acts, FL)

print(f"\n{'layer':>6} {'depth':>6} {'logit lens':>11} {'tuned':>8}")
for l in lens.layers:
    print(f"{l:>6} {depth.to_frac(m, l):>6.2f} {kl_logit[l]:>11.3f} {kl_final[l]:>8.3f}")

lens.meta.update(repo=a.repo, prompt=a.prompt, n_images=side["n"], layers=lens.layers,
                 kl_logit_lens=kl_logit, kl_ridge=kl_ridge, kl_final=kl_final,
                 depth_frac={l: depth.to_frac(m, l) for l in lens.layers})
p = lens.save(f"{a.out}/{name}__tuned.pt")
print(f"\nwrote {p}")
