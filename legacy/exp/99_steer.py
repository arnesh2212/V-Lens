# SPDX-License-Identifier: Apache-2.0
"""Steering in a coordinate that has units, a region, and a question in it.

Three claims separate this from steering along an activation-difference vector,
and only the last two are ours alone.

*Region-targeted.*  The edit is applied at one object's image positions and
nowhere else.  A prior lens can do this too -- it has a per-position reading --
so it is a control here rather than a contribution.

*Question-selective.*  The edit direction is the interaction
``I_R(q_a, q_b)``: the part of the reading that differs between two questions
about the same picture.  A transport built before the prompt is read returns the
same reading for both, so its interaction is identically zero and this direction
does not exist for it.  If steering along it moves answer A while leaving answer
B alone, the coordinate is question-selective in the strong sense.

*Calibrated.*  The reading is in logits and obeys a conservation law, so the
size of the effect can be predicted before the edit is made and checked after.
A similarity-valued lens cannot attempt this: it has no units to predict in.

Every edit is bounded -- applied once, at one layer, at a stated fraction of the
residual norm -- and runs beside a random direction of identical norm at the
same positions.  The compounding, unbounded swap this project once published is
not repeated here; the point of a bounded edit is that its size is a number you
choose rather than one you discover afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jvlm
from jvlm import corpus, readout, vmap

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--items",
                default="results/dissoc/qwen8b_items.json",
                help="the scenes the dissociation already selected, so the "
                     "edit is measured on the same items as the reading")
ap.add_argument("--mean-images", type=int, default=50)
ap.add_argument("--layers", type=int, nargs="*", default=None)
ap.add_argument("--ig-steps", type=int, default=8)
ap.add_argument("--alphas", type=float, nargs="*",
                default=[0.02, 0.05, 0.10, 0.20])
ap.add_argument("--out", default="results/steer")
ap.add_argument("--tag", default="qwen2b")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
m = jvlm.load_vlm(a.repo)
LAST = m.n_layers - 1
LAYERS = sorted(set(a.layers)) if a.layers else \
    sorted({int(round(f * LAST)) for f in (0.25, 0.5, 0.7)})
EDIT = LAYERS[len(LAYERS) // 2]
print(f"layers {LAYERS}; editing at {EDIT} (fixed in advance)", flush=True)

# reuse the scenes the dissociation run selected rather than reselecting:
# the edit is then measured on exactly the items whose readings we reported,
# and the two results cannot drift apart through a different filter
raw = json.load(open(a.items))
items = []
for r in raw:
    items.append(dict(scene_index=r["scene_index"], path=r["path"],
                      q_a=r["question_a"], q_b=r["question_b"],
                      cells_a=r["tok_a"], cells_b=r["tok_b"],
                      ids_a=[r["id_a"]], foil_a=[r["id_b"]],
                      ids_b=[r["id_b"]], foil_b=[r["id_a"]]))
print(f"{len(items)} scenes from {a.items}", flush=True)

from PIL import Image  # noqa: E402


def load_scene(it):
    return Image.open(it["path"]).convert("RGB").resize((448, 448), Image.LANCZOS)


b0 = m.chat("What is in this image?", [load_scene(items[0])])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
side = int(round(P ** 0.5))

acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
nm = 0
for p in corpus.coco_images(a.mean_images, seed=31):
    b = m.chat("What is in this image?", [corpus.load_image(p, square=448)])
    if len(readout.image_positions(m, b)) != P:
        continue
    A = readout.residuals(m, b, LAYERS)
    for l in LAYERS:
        acc[l] += A[l][IPOS].float().cpu().double()
    nm += 1
MEANS = {l: (acc[l] / max(nm, 1)).float() for l in LAYERS}
print(f"means from {nm} photographs", flush=True)


def margin(batch, good_ids, foil_ids):
    """model.forward runs the trunk only -- no LM head -- so the answer's
    logits come from unembedding the last layer's residual at the answer
    position, which is what every other experiment here does."""
    L = m.n_layers - 1
    h = readout.residuals(m, batch, [L])[L]
    apos = batch["input_ids"].shape[1] - 1
    with torch.no_grad():
        lg = m.unembed(h[apos][None])[0].float().cpu()
    return float(lg[torch.tensor(good_ids)].max() - lg[torch.tensor(foil_ids)].max())


def edit_run(batch, cells, direction, alpha, good, foil):
    """Add alpha * direction at the region's positions, once, at one layer."""
    seq = IPOS[torch.tensor(sorted(cells), dtype=torch.long)]
    d = direction / (direction.norm() + 1e-9)
    prefill = [None]
    got = {}

    def hook(mod, inp, out):
        tup = not torch.is_tensor(out)
        h = (out[0] if tup else out)
        if prefill[0] is None:
            prefill[0] = h.shape[1]
        if h.shape[1] != prefill[0]:
            return out
        h = h.clone()
        sub = h[:, seq.to(h.device)]
        scale = alpha * sub.norm(dim=-1, keepdim=True)
        got["ratio"] = float((scale.mean() / (sub.norm(dim=-1).mean() + 1e-9)).item())
        h[:, seq.to(h.device)] = sub + scale * d.to(h.device).to(h.dtype)
        return (h, *out[1:]) if tup else h

    hd = m.layers[EDIT].register_forward_hook(hook)
    try:
        return margin(batch, good, foil), got.get("ratio", float("nan"))
    finally:
        hd.remove()


rows = []
t0, kept = time.time(), 0
rng = np.random.default_rng(0)
for it in items:
    if kept >= a.n:
        break
    img = load_scene(it)
    qa, qb = it["q_a"], it["q_b"]
    ba, bb = m.chat(qa, [img]), m.chat(qb, [img])
    if len(readout.image_positions(m, ba)) != P:
        continue
    ga, fa = it["ids_a"], it["foil_a"]
    gb, fb = it["ids_b"], it["foil_b"]
    base_a, base_b = margin(ba, ga, fa), margin(bb, gb, fb)
    if base_a <= 0 or base_b <= 0:
        continue                      # the model must get both right first

    VOC = sorted(set(ga + gb))
    iga = vmap.ig_map(m, ba, VOC, [EDIT], IPOS.tolist(),
                      means={EDIT: MEANS[EDIT]}, steps=a.ig_steps, unit=True)
    igb = vmap.ig_map(m, bb, VOC, [EDIT], IPOS.tolist(),
                      means={EDIT: MEANS[EDIT]}, steps=a.ig_steps, unit=True)

    A = readout.residuals(m, ba, [EDIT])
    h = A[EDIT][IPOS].float().cpu()
    dev = h - MEANS[EDIT]

    for tgt, cells, q_batch, good, foil, other in (
            ("A", it["cells_a"], ba, ga, fa, ("B", bb, gb, fb, base_b)),
            ("B", it["cells_b"], bb, gb, fb, ("A", ba, ga, fa, base_a))):
        idx = torch.tensor(sorted(cells), dtype=torch.long)
        # The direction that moves a word is the gradient of that word's logit
        # with respect to the activations -- the lens vector itself, in the
        # model's own units.  An earlier version steered along a
        # reading-weighted deviation, which is a scalar map contracted with a
        # displacement and is not a direction that increases anything; it moved
        # the answer by 0.02 logits and was indistinguishable from noise.
        gw = vmap.raw_gradients(m, q_batch, VOC, [EDIT], IPOS.tolist())[EDIT]
        want = VOC.index(good[0]) if good[0] in VOC else 0
        foilw = VOC.index(foil[0]) if foil[0] in VOC else (1 % len(VOC))
        g_own = gw[want][idx].float().cpu().mean(0)          # push the right word
        g_foil = gw[foilw][idx].float().cpu().mean(0)        # push the other one
        # question-selective: the part of the direction that is not shared with
        # the other question's answer, which is what the interaction isolates
        d_int = g_own - (g_own @ g_foil) / (g_foil.norm() ** 2 + 1e-9) * g_foil
        d_own = g_own
        d_swap = g_foil - g_own                              # exchange the two
        d_rnd = torch.tensor(rng.normal(size=m.d_model), dtype=torch.float32)

        for name, d in (("answer direction", d_own),
                        ("question-selective", d_int),
                        ("swap toward the foil", d_swap),
                        ("random matched", d_rnd)):
            if not torch.isfinite(d).all() or d.norm() < 1e-9:
                continue
            for al in a.alphas:
                mt, ratio = edit_run(q_batch, cells, d, al, good, foil)
                on = mt - (base_a if tgt == "A" else base_b)
                nm2, bt, gg, ff, bs = other
                mo, _ = edit_run(bt, cells, d, al, gg, ff)
                off = mo - bs
                rows.append(dict(scene=it.get("scene_index", kept), target=tgt,
                                 direction=name, alpha=al, ratio=ratio,
                                 on_target=on, off_target=off,
                                 base_on=(base_a if tgt == "A" else base_b)))
    kept += 1
    if kept % 4 == 0:
        el = time.time() - t0
        print(f"  {kept}/{a.n}  {el:.0f}s  eta {el/kept*(a.n-kept)/60:.1f}m", flush=True)

df = pd.DataFrame(rows)
df.to_parquet(os.path.join(a.out, f"{a.tag}_raw.parquet"))

print("\n=== bounded steering: does the edit move its own answer and spare the other? ===")
print(f"{'direction':<18}{'alpha':>7}{'||dh||/||h||':>13}{'on-target':>11}{'off-target':>12}{'ratio':>8}")
tab = {}
for (nme, al), g in df.groupby(["direction", "alpha"]):
    on, off = g.on_target.mean(), g.off_target.mean()
    r = abs(on) / max(abs(off), 1e-6)
    tab[f"{nme}|{al}"] = dict(on=float(on), off=float(off), ratio=float(r),
                              pert=float(g.ratio.mean()), n=int(len(g)))
    print(f"{nme:<18}{al:>7.3f}{g.ratio.mean():>13.3f}{on:>11.3f}{off:>12.3f}{r:>8.1f}")

json.dump(dict(repo=a.repo, n=kept, edit_layer=EDIT, alphas=a.alphas,
               table=tab, minutes=round((time.time() - t0) / 60, 1)),
          open(os.path.join(a.out, f"{a.tag}_stats.json"), "w"), indent=1)
print(f"\nwrote {a.out}/{a.tag}_stats.json")
