# SPDX-License-Identifier: Apache-2.0
"""Every attribution method on one ruler, each at its own best aggregation.

The comparison in this paper has been three lenses pooled by summing, which is
the operation our reading is built for and not the one either prior lens is
built for.  That is a real confound and it is measurable: on synthetic data
where both methods are perfect, holding an additive method and an intensive one
to the same pooling rule moves top-1 accuracy by 0.2 purely through the choice.

So this runs the whole comparison through ``jvlm.bench``, which treats
aggregation as a free axis and reports every method at the setting that suits
*it*.  The methods are:

  VJ-Lens          the path reading, our method
  logit lens       W_U applied at the image positions
  J-lens           W_U J_l, the fitted transport
  gradient norm    the strongest classical competitor we have measured
  gradient x act   what our reading reduces to with a one-point path
  random           the measured floor, not an assumed one
  wrong prompt     our reading taken under a different question, which
                   separates "reads the image" from "reads this question"

The last of these is the control that matters most for our claim.  A method
that scores well under the wrong prompt is reading the picture; the gap between
the right-prompt and wrong-prompt columns is the part of our score that is
actually about the question.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import collections
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jvlm
from jvlm import bench, cocodata, corpus, readout, vmap

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens", default=None)
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--mean-images", type=int, default=80)
ap.add_argument("--n-layers", type=int, default=4)
ap.add_argument("--ig-steps", type=int, default=8)
ap.add_argument("--pool", type=int, default=2)
ap.add_argument("--vocab", type=int, default=60)
ap.add_argument("--out", default="results/baselines")
ap.add_argument("--tag", default="qwen2b")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
m = jvlm.load_vlm(a.repo)
J = None
if a.jlens:
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))

    # Read where the lens exists.  A fitted Jacobian lens only has a transport
    # at the layers it was fitted at; chosen from the model's geometry instead,
    # the depths miss it and the arm is dropped in silence.
    LAYERS, _ = readout.snap_to_lens(LAYERS, J)

items = cocodata.concept_items(a.n, tokenizer=m.tokenizer)
print(f"{len(items)} items", flush=True)

b0 = m.chat("Describe this image.", [corpus.load_image(items[0]["path"], square=448)])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
side = int(round(P ** 0.5))
LAST = m.n_layers - 1
LAYERS = sorted({int(round(f * LAST)) for f in
                 np.linspace(0.25, 0.85, a.n_layers)})
print(f"{P} image tokens on {side}x{side}; layers {LAYERS}", flush=True)

# the per-position corpus mean, accumulated the way every other experiment
# here does it: there is no shared helper, and inventing one that disagreed
# with the runs we are grading against would be worse than repeating it
_acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
_nm = 0
for _p in corpus.coco_images(a.mean_images, seed=11):
    _b = m.chat("Describe this image.", [corpus.load_image(_p, square=448)])
    if len(readout.image_positions(m, _b)) != P:
        continue
    _A = readout.residuals(m, _b, LAYERS)
    for l in LAYERS:
        _acc[l] += _A[l][IPOS].float().cpu().double()
    _nm += 1
MEANS = {l: (_acc[l] / max(_nm, 1)).float() for l in LAYERS}
print(f"means from {_nm} held-out photographs", flush=True)

# regions: a coarse grid over the image tokens, the same partition for everyone
K = side // a.pool
REGIONS = []
for r in range(K):
    for c in range(K):
        cells = [rr * side + cc
                 for rr in range(r * a.pool, min((r + 1) * a.pool, side))
                 for cc in range(c * a.pool, min((c + 1) * a.pool, side))]
        if cells:
            REGIONS.append(cells)
print(f"{len(REGIONS)} regions of {a.pool}x{a.pool} patches", flush=True)


def cover_truth(batch, tok_ids, region_cells):
    """What covering each region really costs the answered word."""
    sets = [(cells, IPOS[torch.tensor(cells, dtype=torch.long)])
            for cells in region_cells]
    big = m.replicate(batch, len(sets))
    layer = LAYERS[len(LAYERS) // 2]

    def hook(mod, inp, out, _s=sets, _l=layer):
        tup = not torch.is_tensor(out)
        h = (out[0] if tup else out).clone()
        for j, (cells, seq) in enumerate(_s):
            h[j, seq.to(h.device)] = MEANS[_l][torch.tensor(cells)].to(h.device).to(h.dtype)
        return (h, *out[1:]) if tup else h

    L = m.n_layers - 1
    apos = batch["input_ids"].shape[1] - 1
    hd = m.layers[layer].register_forward_hook(hook)
    try:
        hb = readout.residuals_batched(m, big, [L])[L]
        with torch.no_grad():
            lg = m.unembed(hb[:, apos] if hb.dim() == 3 else hb[apos][None]).float().cpu()
    finally:
        hd.remove()
    h0 = readout.residuals(m, batch, [L])[L]
    with torch.no_grad():
        base = m.unembed(h0[apos][None])[0].float().cpu()
    col = torch.tensor(tok_ids, dtype=torch.long)
    return (base[col].max() - lg[:, col].max(-1).values).numpy()


tasks, methods = [], {n: [] for n in
                      ("VJ-Lens", "logit lens", "J-lens", "gradient norm",
                       "gradient x activation", "random", "wrong prompt")}
rng = np.random.default_rng(0)
t0 = time.time()
kept = 0
skips = collections.Counter()

for it in items:
    img = corpus.load_image(it["path"], square=448)
    ids = cocodata.concept_token_ids(m.tokenizer, it["concept"])
    if not ids:
        skips["no concept token"] += 1
        continue
    b = m.chat("Describe this image.", [img])
    if len(readout.image_positions(m, b)) != P:
        skips["image-token count differs"] += 1
        continue
    # a different question over the same picture: the wrong-prompt control
    bw = m.chat("How many objects are in this photograph?", [img])

    truth = cover_truth(b, ids, REGIONS)
    if not np.isfinite(truth).all() or float(np.std(truth)) < 1e-9:
        skips[f"flat/non-finite truth (len {np.size(truth)})"] += 1
        continue

    layer = LAYERS[len(LAYERS) // 2]
    VOC = sorted(set(ids))
    ig = vmap.ig_map(m, b, VOC, [layer], IPOS.tolist(),
                     means={layer: MEANS[layer]}, steps=a.ig_steps)
    igw = vmap.ig_map(m, bw, VOC, [layer], IPOS.tolist(),
                      means={layer: MEANS[layer]}, steps=a.ig_steps)
    g = vmap.raw_gradients(m, b, VOC, [layer], IPOS.tolist())
    A = readout.residuals(m, b, [layer])
    h = A[layer][IPOS].float().cpu()

    per = {}
    per["VJ-Lens"] = ig[layer].max(0).values.numpy()
    per["wrong prompt"] = igw[layer].max(0).values.numpy()
    gr = g[layer][0].float().cpu()                       # [P, d]
    per["gradient norm"] = gr.norm(dim=-1).numpy()
    per["gradient x activation"] = (gr * (h - MEANS[layer])).sum(-1).numpy()
    with torch.no_grad():
        ll = m.unembed(h.to(m.device))[:, torch.tensor(ids)].float().cpu()
    per["logit lens"] = ll.max(-1).values.numpy()
    if J is not None and layer in J.jacobians:
        jl, _ = readout.lens_logits(m, J, b, layers=[layer],
                                    positions=IPOS.tolist(), cached=A)
        per["J-lens"] = jl[layer][:, torch.tensor(ids)].max(-1).values.numpy()
    per["random"] = rng.normal(size=P)

    tasks.append(bench.Task(regions=REGIONS, truth=truth, kind="localise",
                            image_id=str(it.get("image_id"))))
    for name in methods:
        methods[name].append(per.get(name, np.full(P, np.nan)))
    kept += 1
    if kept % 10 == 0:
        el = time.time() - t0
        print(f"  {kept}/{a.n}  {el:.0f}s  eta {el / kept * (a.n - kept) / 60:.1f}m",
              flush=True)

methods = {n: v for n, v in methods.items()
           if v and np.isfinite(v[0]).any()}
if not tasks:
    raise SystemExit(
        "no item survived: " + ", ".join(f"{k} x{v}" for k, v in skips.items()))
res = bench.sweep(methods, tasks)
best = bench.best_per_method(res)

print("\ntop-1 accuracy, method x aggregation\n")
print(bench.table(res))
if skips:
    print("skipped: " + ", ".join(f"{k} x{v}" for k, v in skips.items()))
print(f"\nchance = {tasks[0].chance_top1():.3f} over {len(REGIONS)} regions, "
      f"{len(tasks)} photographs\n")
print(f"{'method':<24}{'best rule':>10}{'top-1':>8}{'lift':>7}{'rho':>8}")
for n, r in sorted(best.items(), key=lambda kv: -kv[1].top1):
    print(f"{n:<24}{r.aggregation:>10}{r.top1:>8.3f}{r.lift():>6.1f}x{r.spearman:>8.3f}")

out = {
    "repo": a.repo, "n": len(tasks), "n_regions": len(REGIONS),
    "layer": int(LAYERS[len(LAYERS) // 2]), "ig_steps": a.ig_steps,
    "chance": tasks[0].chance_top1(),
    "grid": [{"method": r.method, "aggregation": r.aggregation,
              "top1": r.top1, "spearman": r.spearman, "n": r.n} for r in res],
    "best": {n: {"aggregation": r.aggregation, "top1": r.top1,
                 "lift": r.lift(), "spearman": r.spearman}
             for n, r in best.items()},
}
p = os.path.join(a.out, f"{a.tag}_stats.json")
json.dump(out, open(p, "w"), indent=1)
print(f"\nwrote {p}")
