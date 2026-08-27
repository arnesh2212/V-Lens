"""V-Lens, fit to counterfactuals that mean something.

The first version of the do head was fit on the wrong question.  It replaced a
random handful of image tokens with another photograph's and tried to predict
what happened to 225 category words at once.  Both sides of that regression are
mostly nuisance -- a random patch of sky carries no object, and 224 of the 225
words are irrelevant to any given picture -- so the fit collapsed onto one
direction and read the same word off every photograph.

This asks the question the lens is actually for.  COCO says where each object
is.  So: cover the tokens the object occupies, replace them with the same
positions from a different photograph, and record what that does to the model's
logit for that object's name.  Now the regressor carries "the cow is gone" and
the target carries "the model stopped saying cow", which is signal on both
sides.  A matched number of background tokens is patched the same way, as the
control that says the effect is about the object and not about damage.

Circularity is avoided by splitting the *categories*, not only the photographs.
The map is fit using photographs of one set of objects and evaluated on
photographs of a disjoint set, so it has to generalise to words it never saw
moved.

Three things are measured on the held-out categories:

  predicts   held-out variance explained in what patching the object really did
  reads      does the object's name outrank the categories that are absent
  points     does the map peak inside the object's own bounding box -- the
             pointing game from the saliency literature, which is the version of
             "is this lens any good" that you can look at
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
from jvlm import cocodata, corpus, direct, readout, stats as st
from jvlm.patchfit import PatchCollector, SecantLens, gauge_fix

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--fit-images", type=int, default=600)
ap.add_argument("--eval-n", type=int, default=140)
ap.add_argument("--donors", type=int, default=32)
ap.add_argument("--tangent-images", type=int, default=20)
ap.add_argument("--n-layers", type=int, default=5)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--min-area", type=float, default=0.05)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/objectlens")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lo, hi = max(m.n_deepstack, 1), m.n_layers - 2
LAYERS = sorted({int(round(lo + (hi - lo) * f))
                 for f in np.linspace(0.15, 0.85, a.n_layers)})

CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
rng = np.random.default_rng(20260826)
perm = rng.permutation(len(CATS))
TEST_CATS = sorted(CATS[i] for i in perm[: len(CATS) // 3])
FIT_CATS = sorted(CATS[i] for i in perm[len(CATS) // 3:])
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
FLAT = sorted({t for v in CAT_IDS.values() for t in v})
POSI = {t: i for i, t in enumerate(FLAT)}
R = gauge_fix(m._lm_head.weight.detach().float().cpu())[FLAT]
R = R - R.mean(0, keepdim=True)
print(m, f"\nlayers {LAYERS}\nfit on {len(FIT_CATS)} categories, held out "
      f"{len(TEST_CATS)}: {', '.join(TEST_CATS[:8])}...", flush=True)

PROMPT = "Describe this image."


def prep(path):
    return m.chat(PROMPT, [corpus.load_image(path, square=a.square)])


def pick(n, cats, seed, min_area):
    """Photographs whose main object is one of ``cats`` and is big enough."""
    pool = cocodata.concept_items(6000, min_area=min_area, seed=seed)
    out = [it for it in pool if it["concept"] in cats]
    return out[:n]


fit_items = pick(a.fit_images, set(FIT_CATS), 4242, a.min_area)
print(f"{len(fit_items)} fitting photographs", flush=True)

b0 = prep(fit_items[0]["path"])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
assert G * G == P, f"{P} image tokens is not a square grid"
print(f"{P} image tokens, a {G}x{G} grid", flush=True)

DON = {l: [] for l in LAYERS}
for it in fit_items[: max(a.donors, 40)]:
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    A = readout.residuals(m, b, LAYERS)
    for l in LAYERS:
        if len(DON[l]) < a.donors:
            DON[l].append(A[l][ip].float().cpu())
print(f"{len(DON[LAYERS[0]])} donors  ({time.time()-t0:.0f}s)", flush=True)

tan = direct.DirectDirections(LAYERS)
for it in fit_items[: a.tangent_images]:
    try:
        tan.add(m, prep(it["path"]), FLAT)
    except (ValueError, torch.cuda.OutOfMemoryError):
        torch.cuda.empty_cache()
TAN = {l: (v.float() - v.float().mean(0, keepdim=True))
       for l, v in tan.mean("img", normalize=False).items()}
print(f"derivative from {tan.n} photographs  ({time.time()-t0:.0f}s)", flush=True)

# ---- object-aware counterfactuals -------------------------------------------
gen = torch.Generator().manual_seed(0)
cols = {l: PatchCollector(m, l, R) for l in LAYERS}
kept = 0
for gi, it in enumerate(fit_items):
    boxes = cocodata.object_boxes(it["image_id"], it["concept"])
    if not boxes:
        continue
    tok = sorted({t for bx in boxes for t in cocodata.box_tokens(bx, G)})
    if not (3 <= len(tok) <= P // 2):
        continue                      # an object filling the frame has no control
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    # a background set of the same size, so the control differs from the object
    # arm only in *where* it is, never in how much of the picture it covers
    bg = [t for t in range(P) if t not in set(tok)]
    n_bg = min(len(tok), len(bg))
    if n_bg < 3:
        continue
    sel = torch.tensor(np.asarray(bg)[
        rng.choice(len(bg), size=n_bg, replace=False)])
    on = torch.tensor(tok)
    for l in LAYERS:
        cols[l].add(b, ip.cpu(), DON[l], n_samples=2, group=gi,
                    generator=gen, index_sets=[on, sel])
    kept += 1
    if gi % 60 == 0:
        print(f"  patch {gi}/{len(fit_items)}  kept {kept}  "
              f"{time.time()-t0:.0f}s", flush=True)
print(f"{kept} photographs, {len(cols[LAYERS[0]].X)} counterfactuals per layer "
      f"({time.time()-t0:.0f}s)", flush=True)

maps, inters, reports = {}, {}, {}
for l in LAYERS:
    L, b_, rep = cols[l].fit(prior=TAN[l])
    cand = {"logit lens": R, "the derivative": TAN[l], "V-Lens": L}
    rep["ladder"] = cols[l].evaluate(cand)
    maps[l], inters[l], reports[l] = L, b_, rep
    lad = rep["ladder"]
    print(f"  L{l}: lambda={rep['chosen_lambda']:g}  "
          + "  ".join(f"{n} {lad[n]['fve_scaled']:+.3f}" for n in cand)
          + f"   (effect {lad['_n']['effect_rms']:.2f})", flush=True)

MEANS = {l: torch.stack([d for d in DON[l]]).mean(0) for l in LAYERS}
lens = SecantLens(maps, inters, MEANS, FLAT, reports)
lens.save(f"{a.out}/{tag}_vlens.pt")
torch.save({"tangent": TAN, "tokens": FLAT}, f"{a.out}/{tag}_tangent.pt")

# ---- evaluation, on categories the map never saw -----------------------------
ev = pick(a.eval_n, set(TEST_CATS), 99, 0.08)
rows = []
for k, it in enumerate(ev):
    boxes = cocodata.object_boxes(it["image_id"], it["concept"])
    if not boxes:
        continue
    tok = set(t for bx in boxes for t in cocodata.box_tokens(bx, G))
    if not (3 <= len(tok) <= P // 2):
        continue                      # an object filling the frame has no control
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    A = readout.residuals(m, b, LAYERS)
    here = set(it["present"])
    col = [POSI[t] for t in CAT_IDS[it["concept"]] if t in POSI]
    for l in LAYERS:
        h = A[l][ip].float().cpu()
        d = h - MEANS[l]
        views = {"V-Lens": d @ maps[l].T + inters[l],
                 "the derivative": d @ TAN[l].T,
                 "logit lens": m.unembed(h.to(m.device)).float().cpu()[:, FLAT]}
        for name, sc_ in views.items():
            best = sc_.max(0).values
            order = best.argsort(descending=True)
            rk = torch.empty_like(order)
            rk[order] = torch.arange(len(order))
            # where in the frame does this lens put the object?
            heat = sc_[:, col].max(1).values
            peak = int(heat.argmax())
            top5 = set(heat.topk(min(5, P)).indices.tolist())
            rows.append(dict(
                i=k, concept=it["concept"], lens=name, layer=l,
                depth=round(l / (m.n_layers - 1), 3),
                rank=int(min(rk[POSI[t]] for t in CAT_IDS[it["concept"]]
                             if t in POSI)),
                n_absent=len([c for c in TEST_CATS if c not in here]),
                worse_than=int(sum(
                    1 for c in TEST_CATS if c not in here
                    and min(rk[POSI[t]] for t in CAT_IDS[c] if t in POSI)
                    < min(rk[POSI[t]] for t in CAT_IDS[it["concept"]]
                          if t in POSI))),
                hit=int(peak in tok),
                hit5=int(bool(top5 & tok)),
                frac=len(tok) / P))
    if k % 25 == 0:
        print(f"  eval {k}/{len(ev)}  {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
agg = df.groupby(["lens", "layer"]).agg(
    n=("i", "count"), median_rank=("rank", "median"),
    top1=("rank", lambda s: float((s == 0).mean())),
    points_at_1=("hit", "mean"), points_at_5=("hit5", "mean"),
    chance=("frac", "mean")).reset_index()
agg.to_csv(f"{a.out}/{tag}_summary.csv", index=False)
print("\n", agg.round(3).to_string(index=False))
best = agg.groupby("lens").agg(best_rank=("median_rank", "min"),
                               best_top1=("top1", "max"),
                               best_point=("points_at_1", "max"),
                               chance=("chance", "mean"))
print("\n", best.round(3).to_string())
json.dump({"repo": a.repo, "layers": LAYERS, "grid": G,
           "fit_categories": FIT_CATS, "test_categories": TEST_CATS,
           "n_fit_photographs": kept, "n_eval": int(df.i.nunique()),
           "by_lens": st.jsonable(best.round(4).to_dict("index")),
           "fit": {str(l): {k: v for k, v in reports[l].items() if k != "path"}
                   for l in LAYERS},
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
print("done", flush=True)
