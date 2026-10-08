"""The inversion, with the answer vocabulary disjoint from the readout's.

Two objections to the inversion share one defusal.

The first is circularity: the secant is fit on the fidelity criterion, so of
course it wins that and loses the other.  The second is deeper -- the legibility
metric asks whether the object's *name* outranks other names, which is itself a
"what will it say" measurement, so the anti-correlation might be definitional
rather than empirical.

Both die if the inversion holds (a) among lenses that were never fit on either
criterion, and (b) when the thing being predicted is an answer the object's name
does not appear in.  So here the model is asked a yes/no question -- "is there a
cow in this image?" -- and every lens is scored on how well it predicts what
resampling an image site does to the *yes-minus-no margin*.  The readout
vocabulary (65 object names) and the answer vocabulary (two words) share no
tokens.

Legibility comes from the main run and is not recomputed.  What this adds is the
fidelity axis, measured somewhere the legibility axis cannot reach.
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
import torch

import jvlm
from jvlm import cocodata, corpus, direct, readout
from jvlm.patchfit import PatchCollector, gauge_fix

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens", default=None)
ap.add_argument("--fit-images", type=int, default=500)
ap.add_argument("--samples", type=int, default=8)
ap.add_argument("--donors", type=int, default=32)
ap.add_argument("--tangent-images", type=int, default=40)
ap.add_argument("--n-layers", type=int, default=4)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/disjoint")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
J = None
if a.jlens and os.path.exists(a.jlens):
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))
lo, hi = max(m.n_deepstack, 1), m.n_layers - 2
LAYERS = sorted({int(round(lo + (hi - lo) * f))
                 for f in np.linspace(0.2, 0.85, a.n_layers)})


def ids_for(words):
    out = []
    for w in words:
        for form in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            t = m.tokenizer.encode(form, add_special_tokens=False)
            if len(t) == 1:
                out.append(t[0])
    return sorted(set(out))


YES, NO = ids_for(["yes"]), ids_for(["no"])
ANSWER = YES + NO
NAMES = sorted({t for c in cocodata.SINGLE_WORD
                for t in cocodata.concept_token_ids(m.tokenizer, c)})
overlap = set(ANSWER) & set(NAMES)
print(m, f"\nlayers {LAYERS}   yes={len(YES)} no={len(NO)} tokens; "
      f"overlap with the 65 object names: {len(overlap)}", flush=True)
assert not overlap, f"answer and readout vocabularies share {overlap}"

# the readout is the yes-minus-no contrast and nothing else: a uniform shift of
# the two is invisible to the choice, so it is projected out
W = gauge_fix(m._lm_head.weight.detach().float().cpu())
R = (W[YES].mean(0) - W[NO].mean(0))[None]     # [1, d]: the yes-minus-no axis

items = cocodata.concept_items(a.fit_images, min_area=0.06, seed=515,
                               tokenizer=m.tokenizer)


def prep(it, present=True):
    q = (f"Is there a {it['concept'] if present else it['distractor']} "
         "in this image? Answer yes or no.")
    return m.chat(q, [corpus.load_image(it["path"], square=a.square)])


b0 = prep(items[0])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
DON = {l: [] for l in LAYERS}
nm = 0
for it in items[: max(a.donors, 60)]:
    b = prep(it)
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    A = readout.residuals(m, b, LAYERS)
    for l in LAYERS:
        if len(DON[l]) < a.donors:
            DON[l].append(A[l][ip].float().cpu())
    nm += 1
print(f"{P} image tokens, {len(DON[LAYERS[0]])} donors  "
      f"({time.time()-t0:.0f}s)", flush=True)

tan = direct.DirectDirections(LAYERS)
for it in items[: a.tangent_images]:
    try:
        tan.add(m, prep(it), ANSWER)
    except (ValueError, torch.cuda.OutOfMemoryError):
        torch.cuda.empty_cache()
        continue
raw = tan.mean("img", normalize=False)
ny, nn_ = len(YES), len(NO)
# the tangent for the contrast, in the same one-dimensional readout
TAN = {l: (v[:ny].mean(0) - v[ny:ny + nn_].mean(0))[None].float()
       for l, v in raw.items()}
print(f"tangent from {tan.n} photographs  ({time.time()-t0:.0f}s)", flush=True)

gen = torch.Generator().manual_seed(0)
SIZES = np.array([1, 1, 2, 4, 4, 8, 8, 16])
cols = {l: PatchCollector(m, l, R) for l in LAYERS}
for gi, it in enumerate(items):
    b = prep(it, present=(gi % 2 == 0))
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    sizes = SIZES[torch.randperm(len(SIZES), generator=gen).numpy()][: a.samples]
    for l in LAYERS:
        cols[l].add(b, ip.cpu(), DON[l], n_samples=a.samples, group=gi,
                    generator=gen, sizes=sizes)
    if gi % 50 == 0:
        print(f"  patch {gi}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

report = {}
NAMESET = ["logit lens", "Jacobian lens", "our tangent", "the secant"]
print("\npredicting what a patch does to the yes/no margin:")
print(" " * 12 + "".join(f"{n:>16}" for n in NAMESET))
for l in LAYERS:
    c = cols[l]
    L, _, rep = c.fit(prior=TAN[l])
    cand = {"logit lens": R, "our tangent": TAN[l], "the secant": L}
    if J is not None and l in J.jacobians:
        cand["Jacobian lens"] = R @ J.jacobians[l].float().cpu()
    ev = c.evaluate(cand)
    report[str(l)] = {"fit": rep, "ladder": ev}
    cells = [f"{ev[n]['fve']:+.3f}" if n in ev else "--" for n in NAMESET]
    print(f"  layer {l:<5}" + "".join(f"{x:>16}" for x in cells)
          + f"    (effect {ev['_n']['effect_rms']:.2f} logits)", flush=True)

json.dump({"repo": a.repo, "layers": LAYERS, "n_yes": len(YES),
           "n_no": len(NO), "overlap": sorted(overlap),
           "by_layer": report,
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
print("\nIf the ordering here matches the one on object-name readout, the "
      "inversion is definitional. If it inverts, it is not.", flush=True)
