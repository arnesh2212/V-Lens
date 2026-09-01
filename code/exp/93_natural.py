# SPDX-License-Identifier: Apache-2.0
r"""The double dissociation on PHOTOGRAPHS.

92_dissoc.py establishes the headline result on CLEVR, where object masks are
exact because the scene was rendered from them.  The first question any reader
asks is whether it survives on real photographs, where the "region" is a COCO
bounding box, the lighting is not controlled, and the two objects sit in a
cluttered frame with dozens of unannotated things between them.  This script
answers that question and nothing else: same three measurements, same
estimators, same statistics, same controls, a different image distribution.

THE ITEM.  A COCO val2017 photograph containing two annotated categories that
lie on opposite halves of the frame -- left/right or top/bottom -- each the
largest annotated thing on its own side, with a clear gap across the midline
and disjoint token footprints.  The two questions are

    q_a = "What object is in the {lo} half of this photo?"
    q_b = "What object is in the {hi} half of this photo?"

whose correct answers are the two categories' own names, which differ by
construction.  Both are verified against the model before the item is kept, the
same bar 92_dissoc.py sets: the model's best-scoring COCO category word must be
the true one under each question, or the photograph is discarded.  The margin
objective for q_a is then log P(name_a) - log P(name_b) and for q_b the reverse,
so each question is scored on its own natural correct-versus-foil contrast and
the foil is the *other object's own answer* -- exactly the CLEVR construction,
with category names where CLEVR had colour words.

WHY A SPATIAL REFERRING EXPRESSION RATHER THAN "what colour is the dog".  COCO
has no colour annotation, so a colour question has no ground truth; the only
attribute the annotations certify is *which category is where*.  The cost is
that the question names a place rather than a thing, and the benefit is that
the ground-truth region is the annotation itself.  Both axes are used, and the
localisation table is broken out by axis, because a reader's first objection to
a left/right design is that it might be a left/right prior rather than a
reading of the picture; a top/bottom item cannot be explained that way.

THE THREE MEASUREMENTS are those of 92_dissoc.py and are computed by the same
calls into the same modules.

(1) LOCALISATION, two ways.  Patch level: the pointing game
    (:func:`jvlm.vmap.pointing`) on VJ-Lens's per-patch reading, against the
    box's token footprint, with chance reported from the grid.  Region level:
    does the region with the largest summed attribution equal the questioned
    object's region?  Chance for *that* is NOT 1/R -- the six regions have very
    different sizes and a summed reading favours a big one -- so it is measured
    per item by permuting the per-patch scores across the frame and re-running
    the same argmax, which is the null of "the reading carries no spatial
    information but has the same marginal distribution".

(2) BEHAVIOUR.  Ablating a region -- replacing its tokens with the per-position
    corpus mean at HEAD_LAYER, the rest of the photograph left real -- and
    reading the change in the *answer* margin.  Region A should cost answer A
    much more than answer B and vice versa.  Reported beside a random region
    matched in size and drawn from neither object, and beside the two
    prompt-blind lenses' own top-k patches.

(3) DOES THE INTERACTION PREDICT THE BEHAVIOUR.  I_R = A_R(q_a) - A_R(q_b) read
    off the intact photograph with no ablation at all, correlated across
    regions against the behavioural difference-in-differences measured in (2),
    bootstrapped over IMAGES and floored by a shuffled-item null.

WHAT IS FIXED IN ADVANCE.  Layers (``--layers``, or three sampled across the
usable depth), HEAD_LAYER as the middle one, steps=8 for token readings and
steps=32 for the margin objective, and the partition {A, B, four quadrants of
the remainder}.  Nothing is chosen after seeing a result.

MEMORY.  ``jvlm.vmap.ig_map`` runs all ``steps`` interpolants as one batch,
which is what put the 8B model over 24 GB in the 92_dissoc run.  Every reading
here goes through :func:`jvlm.objective.ig_map`, which takes a ``chunk`` and is
asserted equal to ``vmap.ig_map`` token-for-token in objective.py's own
selftest, so the estimate is unchanged and only the peak memory moves.

Smoke first: ``--n 2 --scan 40 --mean-images 6 --n-layers 2 --gen-n 1
--steps-margin 4 --null-perm 20 --boot 200``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from math import comb

sys.path.insert(0, "/workspace/vlm/code")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens",
                default="/workspace/vlm/results/lenses/"
                        "Qwen3-VL-2B-Instruct__mm__all__n120.pt")
ap.add_argument("--n", type=int, default=24, help="items kept")
ap.add_argument("--scan", type=int, default=400, help="candidate items examined")
ap.add_argument("--mean-images", type=int, default=60)
ap.add_argument("--layers", type=int, nargs="*", default=None)
ap.add_argument("--n-layers", type=int, default=3)
ap.add_argument("--steps-token", type=int, default=8)
ap.add_argument("--steps-margin", type=int, default=32)
ap.add_argument("--chunk-token", type=int, default=4)
ap.add_argument("--chunk-margin", type=int, default=4)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--suffix", default=" Answer with one word.")
ap.add_argument("--strict", action="store_true",
                help="require the category word to be the single top token, "
                     "not merely the top category")
# --- item selection, all fixed before any model is run -----------------------
ap.add_argument("--margin-frac", type=float, default=0.02,
                help="gap each object must keep from the midline")
ap.add_argument("--min-area", type=float, default=0.015)
ap.add_argument("--dominance", type=float, default=0.7,
                help="runner-up on the same side may be at most this times "
                     "the target's area")
ap.add_argument("--straddle", type=float, default=1.0,
                help="largest midline-straddling category may be at most this "
                     "times the smaller target's area")
ap.add_argument("--balance", type=float, default=4.0)
ap.add_argument("--min-tok", type=int, default=3)
ap.add_argument("--axes", nargs="*", default=["lr", "tb"])
# --- reporting ---------------------------------------------------------------
ap.add_argument("--gen-n", type=int, default=12)
ap.add_argument("--gen-tokens", type=int, default=6)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--null-perm", type=int, default=200)
ap.add_argument("--region-null", type=int, default=400,
                help="permutations per item for the measured region-argmax "
                     "chance floor")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="/workspace/vlm/results/natdissoc")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

import jvlm
from jvlm import cocodata as cx
from jvlm import corpus, interact, objective, readout, stats as st, vmap

# ============================================================== model + fixed
m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
if a.layers:
    LAYERS = sorted(set(int(l) for l in a.layers))
else:
    lo, hi = max(m.n_deepstack, 1), m.n_layers - 2
    LAYERS = sorted({int(round(lo + (hi - lo) * f))
                     for f in np.linspace(0.2, 0.85, a.n_layers)})
HEAD_LAYER = LAYERS[len(LAYERS) // 2]
J = None
if a.jlens and os.path.exists(a.jlens):
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))
print(m, f"\nLAYERS {LAYERS}  HEAD_LAYER {HEAD_LAYER} (fixed in advance)  "
      f"steps_token={a.steps_token}/chunk{a.chunk_token} "
      f"steps_margin={a.steps_margin}/chunk{a.chunk_margin}  "
      f"jlens={'yes' if J else 'no'}", flush=True)

# --------------------------------------------------------- the answer words
# A handful of COCO category names have an everyday synonym the model is more
# likely to emit ("sofa" for couch, "fridge" for refrigerator).  Admitting them
# only makes the *selection* filter easier to pass; the reported quantities are
# unaffected, and both questions face the same vocabulary.
SYN = {
    "person": ["person", "man", "woman", "people", "boy", "girl"],
    "tv": ["tv", "television", "monitor", "screen"],
    "couch": ["couch", "sofa"],
    "motorcycle": ["motorcycle", "motorbike"],
    "airplane": ["airplane", "plane", "jet"],
    "refrigerator": ["refrigerator", "fridge"],
    "donut": ["donut", "doughnut"],
    "oven": ["oven", "stove"],
}
DROP = {"teddy", "mouse", "remote"}    # ambiguous words, not a category read


def variant_forms(w: str) -> list[str]:
    plural = w + "es" if w.endswith(("s", "x", "ch", "sh")) else w + "s"
    out = []
    for f in (w, plural):
        out += [f, f.capitalize(), " " + f, " " + f.capitalize()]
    return out


def single_token_variants(cat: str) -> list[tuple[str, int]]:
    out = []
    for word in SYN.get(cat, [cat]):
        for s in variant_forms(word):
            ids = m.tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                out.append((s, int(ids[0])))
    return out


CATVAR = {c: single_token_variants(c) for c in cx.SINGLE_WORD if c not in DROP}
CATVAR = {c: v for c, v in CATVAR.items() if v}
print(f"{len(CATVAR)}/{len(cx.SINGLE_WORD)} COCO categories readable as a "
      f"single token", flush=True)


def prep(path, question):
    return m.chat(question + a.suffix, [corpus.load_image(path, square=a.square)])


def clean_logits(batch):
    h = readout.residuals(m, batch, [m.n_layers - 1])[m.n_layers - 1]
    apos = batch["input_ids"].shape[1] - 1
    with torch.no_grad():
        return m.unembed(h[apos][None])[0].float().cpu()


def cat_choice(lg, cat):
    """(text, id, logit) of ``cat``'s best-scoring single-token surface form."""
    return max(((t, i, float(lg[i])) for t, i in CATVAR[cat]), key=lambda x: x[2])


def top_cat(lg):
    per = {c: cat_choice(lg, c) for c in CATVAR}
    return max(per, key=lambda c: per[c][2]), per


# ------------------------------------------------------------------ the grid
_probe_id = sorted(cx._load()["boxes"])[0]
b0 = prep(str(cx.image_path(_probe_id)), "What is this?")
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
if G * G != P:
    raise SystemExit(f"{P} image tokens is not a square grid")
print(f"{P} image tokens on {G}x{G}", flush=True)


def half_cells(axis: str, which: str) -> list[int]:
    """The cells of the half a question names.  Every item's two objects are
    required to sit on opposite halves of this axis, so this is the half the
    question points at before the picture is consulted at all."""
    out = []
    for c in range(P):
        r, col = divmod(c, G)
        i = col if axis == "lr" else r
        if (i < G // 2) == (which == "lo"):
            out.append(c)
    return out


def pointing_within(heat: torch.Tensor, inside: set, cells: list[int]):
    """The pointing game restricted to one half of the frame.

    The obvious objection to a spatial referring expression is that the lens
    might be reading the word "left" and lighting up the left half wholesale
    rather than finding the object in it.  Restricting the argmax to the named
    half removes that degree of freedom entirely: inside the half, chance is the
    object's share of the HALF, and a lens that only knows which side to look at
    scores exactly chance here."""
    idx = torch.as_tensor(sorted(cells), dtype=torch.long)
    top = int(idx[int(heat[idx].argmax())])
    n_in = len(set(inside) & set(cells))
    return int(top in inside), (n_in / len(cells) if cells else float("nan"))


def quadrant(c: int) -> int:
    r, col = divmod(c, G)
    return (r >= G // 2) * 2 + (col >= G // 2)


def chance5(n_inside: int, P_total: int, k: int = 5) -> float:
    if n_inside <= 0:
        return 0.0
    if k > P_total - n_inside:
        return 1.0
    return 1.0 - comb(P_total - n_inside, k) / comb(P_total, k)


# ------------------------------------------------------------- item selection
AXIS_WORDS = {"lr": ("left", "right"), "tb": ("top", "bottom")}
_BOXES = cx._load()["boxes"]
ALL_IDS = sorted(_BOXES)
# the per-position baseline is built from the tail of the id order; those
# photographs are held out of the scan so no image is both control and item
HELD_OUT = set(ALL_IDS[-(a.mean_images + 40):])


def _area(bs):
    return sum(b[2] * b[3] for b in bs)


def _side(bs, axis, mg):
    i = 0 if axis == "lr" else 1
    lo = min(b[i] for b in bs)
    hi = max(b[i] + b[i + 2] for b in bs)
    if hi <= 0.5 - mg:
        return "lo"
    if lo >= 0.5 + mg:
        return "hi"
    return None


def two_object_items():
    """COCO photographs with one dominant annotated category on each half."""
    for iid in ALL_IDS:
        if iid in HELD_OUT:
            continue
        per_cat = _BOXES[iid]
        for axis in a.axes:
            los, his, mid = [], [], []
            for c, bs in per_cat.items():
                s = _side(bs, axis, a.margin_frac)
                (los if s == "lo" else his if s == "hi" else mid).append(
                    (_area(bs), c))
            for L in (los, his, mid):
                L.sort(reverse=True)
            if not los or not his:
                continue
            aA, cA = los[0]
            aB, cB = his[0]
            if cA == cB or cA not in CATVAR or cB not in CATVAR:
                continue
            if aA < a.min_area or aB < a.min_area:
                continue
            if len(los) > 1 and los[1][0] > a.dominance * aA:
                continue
            if len(his) > 1 and his[1][0] > a.dominance * aB:
                continue
            if mid and mid[0][0] > a.straddle * min(aA, aB):
                continue
            ida = {i for _, i in CATVAR[cA]}
            idb = {i for _, i in CATVAR[cB]}
            if ida & idb:                 # a clipped margin carries no signal
                continue
            tA, tB = set(), set()
            for b in per_cat[cA]:
                tA |= set(cx.box_tokens(b, G))
            for b in per_cat[cB]:
                tB |= set(cx.box_tokens(b, G))
            if (tA & tB) or len(tA) < a.min_tok or len(tB) < a.min_tok:
                continue
            if max(len(tA), len(tB)) > a.balance * min(len(tA), len(tB)):
                continue
            w_lo, w_hi = AXIS_WORDS[axis]
            yield dict(
                image_id=iid, axis=axis, path=str(cx.image_path(iid)),
                cat_a=cA, cat_b=cB, area_a=round(aA, 4), area_b=round(aB, 4),
                tok_a=sorted(tA), tok_b=sorted(tB),
                question_a=f"What object is in the {w_lo} half of this photo?",
                question_b=f"What object is in the {w_hi} half of this photo?")
            break          # at most one item per photograph, so the bootstrap
                           # over images is a bootstrap over independent units


# -------------------------------------------------------------- the baseline
# A per-position mean over held-out COCO photographs -- the same distribution
# the items come from, so covering a region injects "an average photograph
# here" rather than "a different kind of picture here".
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
nrm = {l: torch.zeros(P, dtype=torch.float64) for l in LAYERS}
nm = 0
for iid in sorted(HELD_OUT, reverse=True):
    if nm >= a.mean_images:
        break
    p = cx.image_path(iid)
    if not p.exists():
        continue
    b = m.chat("Describe this image.", [corpus.load_image(p, square=a.square)])
    if len(readout.image_positions(m, b)) != P:
        continue
    A = readout.residuals(m, b, LAYERS)
    for l in LAYERS:
        v = A[l][IPOS].float().cpu()
        acc[l] += v.double()
        nrm[l] += v.norm(dim=-1).double()
    nm += 1
MEANS = {l: (acc[l] / max(nm, 1)).float() for l in LAYERS}
NORM = {l: (nrm[l] / max(nm, 1)).float() for l in LAYERS}
REG = int((NORM[LAYERS[-1]] > NORM[LAYERS[-1]].mean() + 3 * NORM[LAYERS[-1]].std())
          .sum())
print(f"means from {nm} held-out COCO photographs; {REG} register-like tokens "
      f"by norm (kept in every region)  {time.time()-t0:.0f}s", flush=True)


# ------------------------------------------------------------------ ablation
def ipos_of(cells) -> torch.Tensor:
    return IPOS[torch.as_tensor(sorted(cells), dtype=torch.long)]


def ablate_effect(obj, layer, cells, f_intact: float) -> float:
    """F(intact) - F(region ablated); positive = ablation hurt this answer."""
    if not cells:
        return 0.0
    idx = torch.as_tensor(sorted(cells), dtype=torch.long)
    f0 = float(objective.value(m, obj, replace=MEANS[layer][idx], layer=layer,
                               positions=ipos_of(cells))[0])
    return f_intact - f0


def _gen_hook(layer: int, cells):
    idx = torch.as_tensor(sorted(cells), dtype=torch.long)
    pos = ipos_of(cells)
    rep = MEANS[layer][idx].to(m.device)

    def hook(mod, inp, o, _v=rep, _p=pos):
        tup = not torch.is_tensor(o)
        hh = (o[0] if tup else o).clone()
        if hh.shape[1] <= int(_p.max()):
            return o                      # a cached decode step: no image
        hh[:, _p.to(hh.device)] = _v.to(hh.device).to(hh.dtype)
        return (hh, *o[1:]) if tup else hh

    return hook


def generate_text(batch, layer=None, cells=None, max_new: int = 6) -> str:
    hd = (m.layers[layer].register_forward_hook(_gen_hook(layer, cells))
          if cells is not None else None)
    try:
        ids = objective.generated_answer(m, batch, max_new_tokens=max_new)
    finally:
        if hd is not None:
            hd.remove()
    return m.tokenizer.decode(ids).strip().lower()


# ---------------------------------------------- token-level reading, chunked
def token_ig(batch, tid, layers):
    """vmap.ig_map's reading, taken through objective.ig_map so the ``steps``
    interpolants can be held ``chunk`` at a time.  objective.py's selftest
    asserts the two agree to floating point for a token_logit objective; the
    only reason to route through the second is that the first materialises all
    ``steps`` rows at once and an 8B model does not fit that in 24 GB."""
    obj = objective.token_logit(batch, [int(tid)])
    return objective.ig_map(m, obj, layers, IPOS, MEANS,
                            steps=a.steps_token, chunk=a.chunk_token)


# --------------------------------- measured chance for a region-level argmax
def region_null(scores: np.ndarray, region_cells, n_draws: int, rng) -> np.ndarray:
    """P(each region wins the summed-attribution argmax) when the same values
    are scattered at random over the frame.  This is the honest floor: regions
    differ enormously in size and a *sum* favours the big ones, so 1/R is not
    chance and neither is the largest region's share."""
    idx = np.argsort(rng.random((n_draws, scores.size)), axis=1)
    perm = scores[idx]                                   # [n_draws, P]
    sums = np.stack([perm[:, c].sum(1) for c in region_cells], axis=1)
    win = np.argmax(sums, axis=1)
    return np.bincount(win, minlength=len(region_cells)) / float(n_draws)


REGION_NAMES = ["A", "B", "Q0", "Q1", "Q2", "Q3"]

# ============================================================== the main loop
point_rows: list[dict] = []
region_rows: list[dict] = []
ablate_rows: list[dict] = []
flip_rows: list[dict] = []
rarg_rows: list[dict] = []
items_kept: list[dict] = []
n_cand = n_ok_a = n_ok_b = n_ok_both = 0
blind_delta: list[float] = []
blind_IR: list[float] = []
wrong_a: Counter = Counter()
wrong_b: Counter = Counter()
lens_available = {"logit lens": True, "J-lens": bool(J)}
rng = np.random.default_rng(a.seed)

pq, rq, aq, fq, nq = (f"{a.out}/{tag}_point.parquet", f"{a.out}/{tag}_region.parquet",
                      f"{a.out}/{tag}_ablate.parquet", f"{a.out}/{tag}_flips.parquet",
                      f"{a.out}/{tag}_rargmax.parquet")


def checkpoint():
    pd.DataFrame(point_rows).to_parquet(pq)
    pd.DataFrame(region_rows).to_parquet(rq)
    if ablate_rows:
        pd.DataFrame(ablate_rows).to_parquet(aq)
    if flip_rows:
        pd.DataFrame(flip_rows).to_parquet(fq)
    if rarg_rows:
        pd.DataFrame(rarg_rows).to_parquet(nq)
    json.dump(st.jsonable(items_kept), open(f"{a.out}/{tag}_items.json", "w"),
              indent=2, default=float)


for it_i, it in enumerate(two_object_items()):
    if len(items_kept) >= a.n or n_cand >= a.scan:
        break
    n_cand += 1
    if not os.path.exists(it["path"]):
        continue
    ba = prep(it["path"], it["question_a"])
    bb = prep(it["path"], it["question_b"])
    if (len(readout.image_positions(m, ba)) != P
            or len(readout.image_positions(m, bb)) != P):
        continue

    lg_a, lg_b = clean_logits(ba), clean_logits(bb)
    top_a, sc_a = top_cat(lg_a)
    top_b, sc_b = top_cat(lg_b)
    ok_a = (top_a == it["cat_a"]) and (
        not a.strict or int(lg_a.argmax()) == sc_a[it["cat_a"]][1])
    ok_b = (top_b == it["cat_b"]) and (
        not a.strict or int(lg_b.argmax()) == sc_b[it["cat_b"]][1])
    n_ok_a += int(ok_a)
    n_ok_b += int(ok_b)
    if not ok_a:
        wrong_a[f"{it['cat_a']}->{top_a}"] += 1
    if not ok_b:
        wrong_b[f"{it['cat_b']}->{top_b}"] += 1
    if not (ok_a and ok_b):
        continue
    n_ok_both += 1
    ii = len(items_kept)

    text_a, id_a, _ = sc_a[it["cat_a"]]
    text_b, id_b, _ = sc_b[it["cat_b"]]
    tok_a, tok_b = set(it["tok_a"]), set(it["tok_b"])

    obj_a = objective.margin(m, ba, [(text_a, text_b)])
    obj_b = objective.margin(m, bb, [(text_b, text_a)])

    # ------------------------- the prompt-blind claim, measured not asserted
    h_all = readout.residuals(m, ba, LAYERS)
    h_img = h_all[HEAD_LAYER][IPOS].float().cpu()
    h_img_b = readout.residuals(m, bb, [HEAD_LAYER])[HEAD_LAYER][IPOS].float().cpu()
    blind_delta.append(float((h_img - h_img_b).abs().max()))
    jacH = (J.jacobians.get(HEAD_LAYER)
            if J is not None and HEAD_LAYER in J.jacobians else None)
    fa = interact.fixed_lens_patches(m, h_img, [id_a], jacobian=jacH)
    fb = interact.fixed_lens_patches(m, h_img_b, [id_a], jacobian=jacH)
    blind_IR.append(max(float((fa[k] - fb[k]).abs().max()) for k in fa))

    # ---------------------------------------------------- (1) localisation
    HALF_LO, HALF_HI = half_cells(it["axis"], "lo"), half_cells(it["axis"], "hi")
    # every layer in one call: the path integral is per layer but the clean
    # forward that seeds it is not, and on an 8B model that forward is not free
    HA = token_ig(ba, id_a, LAYERS)
    HB = token_ig(bb, id_b, LAYERS)
    for l in LAYERS:
        heat_a, heat_b = HA[l][0], HB[l][0]
        pt_a = vmap.pointing(heat_a, tok_a, G)
        pt_b = vmap.pointing(heat_b, tok_b, G)
        cross_a = vmap.pointing(heat_a, tok_b, G)["hit"]
        cross_b = vmap.pointing(heat_b, tok_a, G)["hit"]
        hh_a, ch_a = pointing_within(heat_a, tok_a, HALF_LO)
        hh_b, ch_b = pointing_within(heat_b, tok_b, HALF_HI)
        point_rows.append(dict(item=ii, image_id=it["image_id"], axis=it["axis"],
                               layer=l, lens="VJ-Lens", q="a", n_inside=len(tok_a),
                               hit=pt_a["hit"], hit5=pt_a["hit5"],
                               chance=pt_a["chance"], chance5=chance5(len(tok_a), P),
                               mass_in=pt_a["mass_in"], cross_hit=cross_a,
                               hit_half=hh_a, chance_half=ch_a))
        point_rows.append(dict(item=ii, image_id=it["image_id"], axis=it["axis"],
                               layer=l, lens="VJ-Lens", q="b", n_inside=len(tok_b),
                               hit=pt_b["hit"], hit5=pt_b["hit5"],
                               chance=pt_b["chance"], chance5=chance5(len(tok_b), P),
                               mass_in=pt_b["mass_in"], cross_hit=cross_b,
                               hit_half=hh_b, chance_half=ch_b))
        hL = h_all[l][IPOS].float().cpu() if l != HEAD_LAYER else h_img
        jacL = J.jacobians.get(l) if J is not None and l in J.jacobians else None
        for lens_name, hh in interact.fixed_lens_patches(
                m, hL, [id_a, id_b], jacobian=jacL).items():
            pa = vmap.pointing(hh[0], tok_a, G)
            pb = vmap.pointing(hh[1], tok_b, G)
            fa_h = pointing_within(hh[0], tok_a, HALF_LO)
            fb_h = pointing_within(hh[1], tok_b, HALF_HI)
            for q, pp, nin, wh in (("a", pa, len(tok_a), fa_h),
                                   ("b", pb, len(tok_b), fb_h)):
                point_rows.append(dict(
                    item=ii, image_id=it["image_id"], axis=it["axis"], layer=l,
                    lens=lens_name, q=q, n_inside=nin, hit=pp["hit"],
                    hit5=pp["hit5"], chance=pp["chance"], chance5=chance5(nin, P),
                    mass_in=pp["mass_in"], cross_hit=None,
                    hit_half=wh[0], chance_half=wh[1]))

    # ------------------------------------------- (3) the interaction object
    scores_a = objective.ig_map(m, obj_a, [HEAD_LAYER], IPOS, MEANS,
                                steps=a.steps_margin,
                                chunk=a.chunk_margin)[HEAD_LAYER]
    scores_b = objective.ig_map(m, obj_b, [HEAD_LAYER], IPOS, MEANS,
                                steps=a.steps_margin,
                                chunk=a.chunk_margin)[HEAD_LAYER]

    quad_cells: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
    for c in range(P):
        if c not in tok_a and c not in tok_b:
            quad_cells[quadrant(c)].append(c)
    region_cells = [sorted(tok_a), sorted(tok_b),
                    quad_cells[0], quad_cells[1], quad_cells[2], quad_cells[3]]
    REGIONS = [torch.tensor(c, dtype=torch.long) for c in region_cells]
    interact.check_partition(REGIONS, P)

    ra = interact.region_sums(scores_a, REGIONS)[0]
    rb = interact.region_sums(scores_b, REGIONS)[0]
    I_R = (ra - rb).tolist()

    # region-level localisation, with its chance floor MEASURED per item
    sa = scores_a[0].numpy().astype(np.float64)
    sb = scores_b[0].numpy().astype(np.float64)
    null_a = region_null(sa, region_cells, a.region_null, rng)
    null_b = region_null(sb, region_cells, a.region_null, rng)
    win_a = int(np.argmax(ra.numpy()))
    win_b = int(np.argmax(rb.numpy()))
    rarg_rows.append(dict(
        item=ii, image_id=it["image_id"], axis=it["axis"],
        win_a=REGION_NAMES[win_a], win_b=REGION_NAMES[win_b],
        hit_a=int(win_a == 0), hit_b=int(win_b == 1),
        chance_a=float(null_a[0]), chance_b=float(null_b[1]),
        both=int(win_a == 0 and win_b == 1),
        chance_both=float(null_a[0] * null_b[1]),
        frac_a=len(tok_a) / P, frac_b=len(tok_b) / P))

    Fa1 = float(objective.value(m, obj_a)[0])
    Fb1 = float(objective.value(m, obj_b)[0])
    Fa0 = float(objective.value(m, obj_a, replace=MEANS[HEAD_LAYER],
                                layer=HEAD_LAYER, positions=IPOS)[0])
    Fb0 = float(objective.value(m, obj_b, replace=MEANS[HEAD_LAYER],
                                layer=HEAD_LAYER, positions=IPOS)[0])
    did_truth = (Fa1 - Fa0) - (Fb1 - Fb0)
    did_claim = float(sum(I_R))
    comp_rel_err = abs(did_claim - did_truth) / max(abs(did_truth), 1e-6)

    # ------------------------------- (2) behavioural ablation, per region --
    for name, cells in zip(REGION_NAMES, region_cells):
        ea = ablate_effect(obj_a, HEAD_LAYER, cells, Fa1)
        eb = ablate_effect(obj_b, HEAD_LAYER, cells, Fb1)
        region_rows.append(dict(
            item=ii, image_id=it["image_id"], axis=it["axis"], region=name,
            n_tok=len(cells), I_R=I_R[REGION_NAMES.index(name)],
            effect_a=ea, effect_b=eb, did_measured=ea - eb))

    # ---------------------------------- each lens NOMINATES its own k patches
    # The fair comparison is not "the annotation versus a lens" but "one lens's
    # pick versus another's": every lens gets the same budget k = |A|, nominates
    # k patches for question a and k for question b, and both nominations are
    # ablated against both answers.  A lens dissociates when its q_a pick costs
    # answer a more than answer b AND its q_b pick costs answer b more than a.
    # VJ-Lens's two picks come from the two questions.  The prompt-blind lenses
    # have no question to read, so their two picks come from the two ANSWER
    # WORDS -- a concession they are not entitled to in practice (you do not
    # know the answer before you ask) and which is granted here anyway, so the
    # control cannot be dismissed as having been given only one shot.
    k = max(len(tok_a), 1)
    free_cells = [c for c in range(P) if c not in tok_a and c not in tok_b]
    rgen = np.random.default_rng(a.seed * 100_003 + it["image_id"])
    rand_cells = sorted(rgen.choice(free_cells, size=min(k, len(free_cells)),
                                    replace=False).tolist())
    rand_cells_b = sorted(rgen.choice(free_cells, size=min(k, len(free_cells)),
                                      replace=False).tolist())

    _memo: dict = {}

    def eff(which: str, cells) -> float:
        key = (which, tuple(sorted(cells)))
        if key not in _memo:
            _memo[key] = ablate_effect(obj_a if which == "a" else obj_b,
                                       HEAD_LAYER, cells,
                                       Fa1 if which == "a" else Fb1)
        return _memo[key]

    nominations: dict[str, tuple] = {
        "VJ-Lens top-k (question-conditional)":
            (torch.topk(HA[HEAD_LAYER][0], min(k, P)).indices.tolist(),
             torch.topk(HB[HEAD_LAYER][0], min(k, P)).indices.tolist())}
    for lens_name, heat2 in interact.fixed_lens_patches(
            m, h_img, [id_a, id_b], jacobian=jacH).items():
        nominations[f"{lens_name} top-k (answer word; I_R=0)"] = (
            torch.topk(heat2[0], min(k, P)).indices.tolist(),
            torch.topk(heat2[1], min(k, P)).indices.tolist())
    nominations["random matched"] = (rand_cells, rand_cells_b)
    nominations["annotated object box"] = (sorted(tok_a), sorted(tok_b))

    for name, (ca, cb) in nominations.items():
        ea, eb = eff("a", ca), eff("b", ca)        # the q_a nomination
        e2a, e2b = eff("a", cb), eff("b", cb)      # the q_b nomination
        ablate_rows.append(dict(
            item=ii, image_id=it["image_id"], row=name, n_tok=len(ca),
            effect_a=ea, effect_b=eb, effect_a2=e2a, effect_b2=e2b,
            did_a=ea - eb, did_b=e2b - e2a,
            cross=int((ea - eb) > 0 and (e2b - e2a) > 0),
            same_pick=int(sorted(ca) == sorted(cb)),
            overlap=len(set(ca) & set(cb)) / max(len(set(ca) | set(cb)), 1),
            in_a=len(set(ca) & tok_a) / max(len(ca), 1),
            in_b=len(set(cb) & tok_b) / max(len(cb), 1)))

    # ------------------------------------------------ full-answer flips (few)
    if ii < a.gen_n:
        try:
            gen_a0 = generate_text(ba, max_new=a.gen_tokens)
            gen_b0 = generate_text(bb, max_new=a.gen_tokens)
            for rname, cells in (("A", sorted(tok_a)), ("B", sorted(tok_b)),
                                 ("random", rand_cells)):
                ga1 = generate_text(ba, HEAD_LAYER, cells, a.gen_tokens)
                gb1 = generate_text(bb, HEAD_LAYER, cells, a.gen_tokens)
                flip_rows.append(dict(
                    item=ii, image_id=it["image_id"], region=rname,
                    gen_a0=gen_a0, gen_a1=ga1, flip_a=int(gen_a0 != ga1),
                    gen_b0=gen_b0, gen_b1=gb1, flip_b=int(gen_b0 != gb1)))
        except Exception as e:                                   # noqa: BLE001
            print(f"  [warn] generation flips failed on item {ii}: "
                  f"{type(e).__name__}: {str(e)[:120]}", flush=True)

    items_kept.append(dict(
        it, top_a=top_a, top_b=top_b, text_a=text_a, text_b=text_b,
        id_a=id_a, id_b=id_b, n_tok_a=len(tok_a), n_tok_b=len(tok_b),
        F_a=Fa1, F_b=Fb1, F_a0=Fa0, F_b0=Fb0, did_truth=did_truth,
        did_claim=did_claim, completeness_rel_err=comp_rel_err))

    el = time.time() - t0
    if len(items_kept) % 4 == 0 or len(items_kept) == a.n:
        print(f"  {len(items_kept)}/{a.n} kept  cand {n_cand}  "
              f"{it['cat_a']}/{it['cat_b']} ({it['axis']})  "
              f"did_truth={did_truth:+.3f} did_claim={did_claim:+.3f} "
              f"rel_err={comp_rel_err:.3f}  {el:.0f}s  "
              f"eta {el/max(len(items_kept),1)*(a.n-len(items_kept))/60:.1f}m",
              flush=True)
        checkpoint()

checkpoint()
print(f"\nkept {len(items_kept)} of {n_cand} candidate photographs "
      f"(q_a correct {n_ok_a}, q_b correct {n_ok_b}, both {n_ok_both})", flush=True)
print(f"most common q_a misreads: {wrong_a.most_common(6)}")
print(f"most common q_b misreads: {wrong_b.most_common(6)}", flush=True)

if len(items_kept) < 3:
    print("TOO FEW ITEMS -- nothing to summarise; this is the honest outcome, "
          "not a bug to route around", flush=True)
    json.dump({"repo": a.repo, "kept": len(items_kept), "candidates": n_cand,
               "n_ok_a": n_ok_a, "n_ok_b": n_ok_b, "n_ok_both": n_ok_both},
              open(f"{a.out}/{tag}_stats.json", "w"), indent=2)
    raise SystemExit(0)

# ==================================================================== report
pt = pd.DataFrame(point_rows)
rg = pd.DataFrame(region_rows)
ab = pd.DataFrame(ablate_rows)
fl = pd.DataFrame(flip_rows) if flip_rows else pd.DataFrame()
ra_df = pd.DataFrame(rarg_rows)


def boot_mean(values, n=a.boot, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    rg_ = np.random.default_rng(seed)
    boots = values[rg_.integers(0, len(values), (n, len(values)))].mean(1)
    return dict(mean=float(values.mean()), lo=float(np.percentile(boots, 2.5)),
                hi=float(np.percentile(boots, 97.5)), n=int(len(values)))


def boot_by_item(df, col, item_col="item", seed=0):
    return boot_mean(df.groupby(item_col)[col].mean().to_numpy(), seed=seed)


def boot_ratio_by_item(df, num_col, den_col, item_col="item", n=a.boot, seed=0):
    per_item = df.groupby(item_col)[[num_col, den_col]].mean()
    num, den = per_item[num_col].to_numpy(), per_item[den_col].to_numpy()
    if len(num) == 0:
        return dict(ratio=float("nan"), lo=float("nan"), hi=float("nan"))
    rg_ = np.random.default_rng(seed)
    idx = rg_.integers(0, len(num), (n, len(num)))
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = num[idx].mean(1) / den[idx].mean(1)
    boots = boots[np.isfinite(boots)]
    real = float(num.mean() / den.mean()) if den.mean() != 0 else float("nan")
    if len(boots) == 0:
        return dict(ratio=real, lo=float("nan"), hi=float("nan"))
    return dict(ratio=real, lo=float(np.percentile(boots, 2.5)),
                hi=float(np.percentile(boots, 97.5)))


def boot_diff(x, y, n=a.boot, seed=0):
    """Paired bootstrap over items of mean(x) - mean(y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rg_ = np.random.default_rng(seed)
    idx = rg_.integers(0, len(x), (n, len(x)))
    b = x[idx].mean(1) - y[idx].mean(1)
    return dict(diff=float(x.mean() - y.mean()),
                lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                p_ge0=float((b <= 0).mean()))


print("\n" + "=" * 90)
print("(0) THE PROMPT-BLIND LENSES CANNOT DISSOCIATE -- measured, not assumed")
print("=" * 90)
print(f"image-token residuals at layer {HEAD_LAYER} under q_a vs q_b: "
      f"max |delta| over {len(blind_delta)} photographs = {max(blind_delta):.3e} "
      f"(the image tokens precede the question under a causal mask)")
print(f"logit-lens / J-lens per-patch reading of the SAME word under q_a vs "
      f"q_b: max |delta| = {max(blind_IR):.3e}, so I_R(q_a,q_b) = 0 exactly "
      f"for both prior lenses at every region of every item.  They appear "
      f"below only as localisation and ablation controls, scored on the one "
      f"reading they can produce.")

print("\n" + "=" * 90)
print("(1a) LOCALISATION, PATCH LEVEL -- pointing game vs the two prompt-blind lenses")
print("=" * 90)
loc = (pt.groupby(["lens", "layer"])
       .agg(n=("item", "count"), hit=("hit", "mean"), hit5=("hit5", "mean"),
            chance=("chance", "mean"), chance5=("chance5", "mean"),
            hit_half=("hit_half", "mean"), chance_half=("chance_half", "mean"),
            mass_in=("mass_in", "mean")).reset_index())
print(loc.round(4).to_string(index=False))
vj_head = pt[(pt.lens == "VJ-Lens") & (pt.layer == HEAD_LAYER)]
own_hit = boot_by_item(vj_head, "hit")
own_hit5 = boot_by_item(vj_head, "hit5")
cross = vj_head.cross_hit.dropna()
cross_rate = float(cross.mean()) if len(cross) else float("nan")
print(f"\nVJ-Lens @ layer {HEAD_LAYER}: own-region hit {own_hit['mean']:.3f} "
      f"[{own_hit['lo']:.3f},{own_hit['hi']:.3f}]  hit5 {own_hit5['mean']:.3f} "
      f"[{own_hit5['lo']:.3f},{own_hit5['hi']:.3f}]  measured chance(top1) "
      f"{vj_head.chance.mean():.3f}  chance(top5) {vj_head.chance5.mean():.3f}  "
      f"cross-region top1 (should be low) {cross_rate:.3f}")
hw = boot_by_item(vj_head, "hit_half", seed=15)
print(f"...and with the argmax RESTRICTED to the half the question names -- so "
      f"a lens that merely reads the word \"left\" scores chance -- VJ-Lens "
      f"hits the object's own box {hw['mean']:.3f} [{hw['lo']:.3f},"
      f"{hw['hi']:.3f}] of the time against a measured within-half chance of "
      f"{vj_head.chance_half.mean():.3f}")
if pt.axis.nunique() > 1:
    print("\nby axis (a left/right prior cannot explain a top/bottom item):")
    print(vj_head.groupby("axis").agg(n=("hit", "size"), hit=("hit", "mean"),
                                      hit5=("hit5", "mean"),
                                      chance=("chance", "mean")
                                      ).round(4).to_string())

print("\n" + "=" * 90)
print("(1b) LOCALISATION, REGION LEVEL -- argmax over the 6-region partition, "
      "chance MEASURED")
print("=" * 90)
ha = boot_mean(ra_df.hit_a.to_numpy(), seed=1)
hb = boot_mean(ra_df.hit_b.to_numpy(), seed=2)
bo = boot_mean(ra_df.both.to_numpy(), seed=3)
d_a = boot_diff(ra_df.hit_a.to_numpy(), ra_df.chance_a.to_numpy(), seed=4)
d_b = boot_diff(ra_df.hit_b.to_numpy(), ra_df.chance_b.to_numpy(), seed=5)
d_bo = boot_diff(ra_df.both.to_numpy(), ra_df.chance_both.to_numpy(), seed=6)
print(f"region A wins under q_a: {ha['mean']:.3f} [{ha['lo']:.3f},{ha['hi']:.3f}]"
      f"   measured chance {ra_df.chance_a.mean():.3f}"
      f"   excess {d_a['diff']:+.3f} [{d_a['lo']:+.3f},{d_a['hi']:+.3f}] "
      f"p(<=0)={d_a['p_ge0']:.4f}")
print(f"region B wins under q_b: {hb['mean']:.3f} [{hb['lo']:.3f},{hb['hi']:.3f}]"
      f"   measured chance {ra_df.chance_b.mean():.3f}"
      f"   excess {d_b['diff']:+.3f} [{d_b['lo']:+.3f},{d_b['hi']:+.3f}] "
      f"p(<=0)={d_b['p_ge0']:.4f}")
print(f"BOTH (the dissociation itself): {bo['mean']:.3f} "
      f"[{bo['lo']:.3f},{bo['hi']:.3f}]   measured chance "
      f"{ra_df.chance_both.mean():.3f}   excess {d_bo['diff']:+.3f} "
      f"[{d_bo['lo']:+.3f},{d_bo['hi']:+.3f}] p(<=0)={d_bo['p_ge0']:.4f}")
print(f"\n(a prompt-blind lens reads the same map under both questions, so its "
      f"BOTH rate is 0 whenever the two questions want different regions -- "
      f"which is every item here, by construction)")
print("\nwhere the argmax actually landed:")
print(pd.crosstab(ra_df.win_a, ra_df.win_b, rownames=["argmax under q_a"],
                  colnames=["argmax under q_b"]).to_string())

print("\n" + "=" * 90)
print(f"(2) BEHAVIOURAL ABLATION @ layer {HEAD_LAYER}, margin objective, "
      f"steps={a.steps_margin}")
print("=" * 90)
table_rows = []
for name, sub in (("object region A", rg[rg.region == "A"]),
                  ("object region B", rg[rg.region == "B"])):
    ea, eb = boot_by_item(sub, "effect_a"), boot_by_item(sub, "effect_b")
    table_rows.append(dict(row=name, n=ea["n"], n_tok=sub.n_tok.mean(),
                           effect_a=ea["mean"], a_lo=ea["lo"], a_hi=ea["hi"],
                           effect_b=eb["mean"], b_lo=eb["lo"], b_hi=eb["hi"]))
for name, sub in ab.groupby("row"):
    ea, eb = boot_by_item(sub, "effect_a"), boot_by_item(sub, "effect_b")
    table_rows.append(dict(row=f"[q_a pick] {name}", n=ea["n"],
                           n_tok=sub.n_tok.mean(),
                           effect_a=ea["mean"], a_lo=ea["lo"], a_hi=ea["hi"],
                           effect_b=eb["mean"], b_lo=eb["lo"], b_hi=eb["hi"]))
tdf = pd.DataFrame(table_rows)
print(tdf.round(4).to_string(index=False))

print("\n" + "-" * 90)
print("EVERY LENS NOMINATES ITS OWN k PATCHES PER QUESTION -- does its pair of "
      "picks dissociate?")
print("  did_a = (q_a pick's cost to answer a) - (its cost to answer b)")
print("  did_b = (q_b pick's cost to answer b) - (its cost to answer a)")
print("  cross = both positive on the same photograph; coin-flip floor 0.25")
print("-" * 90)
nom_rows = []
for name, sub in ab.groupby("row"):
    da_ = boot_by_item(sub, "did_a", seed=11)
    db_ = boot_by_item(sub, "did_b", seed=12)
    cr = boot_by_item(sub, "cross", seed=13)
    nom_rows.append(dict(
        lens=name, n=cr["n"], did_a=da_["mean"], a_lo=da_["lo"], a_hi=da_["hi"],
        did_b=db_["mean"], b_lo=db_["lo"], b_hi=db_["hi"],
        cross=cr["mean"], c_lo=cr["lo"], c_hi=cr["hi"],
        same_pick=sub.same_pick.mean(), overlap=sub.overlap.mean(),
        in_box_a=sub.in_a.mean(), in_box_b=sub.in_b.mean()))
ndf = pd.DataFrame(nom_rows).sort_values("cross", ascending=False)
print(ndf.round(4).to_string(index=False))

ratio_a = boot_ratio_by_item(rg[rg.region == "A"], "effect_a", "effect_b")
ratio_b = boot_ratio_by_item(rg[rg.region == "B"], "effect_b", "effect_a")
sel_a = rg[rg.region == "A"].groupby("item")[["effect_a", "effect_b"]].mean()
sel_b = rg[rg.region == "B"].groupby("item")[["effect_b", "effect_a"]].mean()
da = boot_diff(sel_a.effect_a.to_numpy(), sel_a.effect_b.to_numpy(), seed=7)
db = boot_diff(sel_b.effect_b.to_numpy(), sel_b.effect_a.to_numpy(), seed=8)
print(f"\nselective effect ratio, region A (effect on A / effect on B): "
      f"{ratio_a['ratio']:.2f} [{ratio_a['lo']:.2f},{ratio_a['hi']:.2f}]")
print(f"selective effect ratio, region B (effect on B / effect on A): "
      f"{ratio_b['ratio']:.2f} [{ratio_b['lo']:.2f},{ratio_b['hi']:.2f}]")
print(f"paired difference, region A: {da['diff']:+.4f} "
      f"[{da['lo']:+.4f},{da['hi']:+.4f}] p(<=0)={da['p_ge0']:.4f}")
print(f"paired difference, region B: {db['diff']:+.4f} "
      f"[{db['lo']:+.4f},{db['hi']:+.4f}] p(<=0)={db['p_ge0']:.4f}")
per_item_cross = (sel_a.effect_a.to_numpy() - sel_a.effect_b.to_numpy()) > 0
per_item_crossb = (sel_b.effect_b.to_numpy() - sel_b.effect_a.to_numpy()) > 0
both_ok = boot_mean((per_item_cross & per_item_crossb).astype(float), seed=9)
print(f"photographs showing the FULL crossover (A hurts a more than b AND B "
      f"hurts b more than a): {both_ok['mean']:.3f} "
      f"[{both_ok['lo']:.3f},{both_ok['hi']:.3f}] of {both_ok['n']}  "
      f"(coin-flip floor 0.25)")

print("\n" + "=" * 90)
print("(3) DOES I_R PREDICT THE MEASURED DiD -- per-region, images resampled")
print("=" * 90)
real = interact.error_stats(rg["I_R"].to_numpy(), rg["did_measured"].to_numpy())
print(f"pooled over {len(rg)} (item, region) pairs, {rg.item.nunique()} "
      f"photographs, {len(REGION_NAMES)} regions each: "
      f"pearson={real['pearson']:.4f} spearman={real['spearman']:.4f} "
      f"r2_identity={real['r2_identity']:.4f} rel_err={real['rel_err']:.4f}")
items_ = rg.item.unique()
groups = {i: g for i, g in rg.groupby("item")}
boots = []
for _ in range(a.boot):
    chosen = rng.choice(items_, size=len(items_), replace=True)
    sub = pd.concat([groups[c] for c in chosen], ignore_index=True)
    if sub["I_R"].std() > 0 and sub["did_measured"].std() > 0:
        boots.append(float(np.corrcoef(sub["I_R"], sub["did_measured"])[0, 1]))
boots = np.array(boots)
if len(boots):
    print(f"paired bootstrap over photographs ({len(boots)}/{a.boot} valid): "
          f"pearson r 95% CI [{np.percentile(boots, 2.5):.4f}, "
          f"{np.percentile(boots, 97.5):.4f}]")
piv_I = rg.pivot(index="item", columns="region", values="I_R")[REGION_NAMES]
piv_D = rg.pivot(index="item", columns="region", values="did_measured")[REGION_NAMES]
common = piv_I.index.intersection(piv_D.index)
I_mat, D_mat = piv_I.loc[common].to_numpy(), piv_D.loc[common].to_numpy()
null_rs = []
for _ in range(a.null_perm):
    perm = rng.permutation(len(common))
    if D_mat[perm].std() > 0 and I_mat.std() > 0:
        null_rs.append(float(np.corrcoef(I_mat.flatten(),
                                         D_mat[perm].flatten())[0, 1]))
null_rs = np.array(null_rs)
if len(null_rs):
    print(f"null floor (photograph labels shuffled, {len(null_rs)} perms): "
          f"mean r={null_rs.mean():.4f}  p95={np.percentile(null_rs, 95):.4f}  "
          f"max={null_rs.max():.4f}")
# The shuffled-item null above is NOT a floor of zero and must not be read as
# one: every photograph has the same region template (A positive, B negative,
# quadrants near nothing), so a wrongly paired item still gets that template
# right.  The question that template does not already answer is whether I_R
# tracks the part of the DiD that VARIES from photograph to photograph.  Two
# estimators of that, both computed from data already in hand:
piv_I2 = piv_I.loc[common]
piv_D2 = piv_D.loc[common]
Ic = (piv_I2 - piv_I2.mean(axis=0)).to_numpy()      # region main effect removed
Dc = (piv_D2 - piv_D2.mean(axis=0)).to_numpy()
r_dm = float(np.corrcoef(Ic.flatten(), Dc.flatten())[0, 1])
dm_null = []
for _ in range(a.null_perm):
    perm = rng.permutation(len(common))
    dm_null.append(float(np.corrcoef(Ic.flatten(), Dc[perm].flatten())[0, 1]))
dm_null = np.array(dm_null)
dm_boot = []
for _ in range(a.boot):
    ch = rng.integers(0, len(common), len(common))
    x, y = Ic[ch].flatten(), Dc[ch].flatten()
    if x.std() > 0 and y.std() > 0:
        dm_boot.append(float(np.corrcoef(x, y)[0, 1]))
dm_boot = np.array(dm_boot)
print(f"\nregion-demeaned (the A/B template subtracted from both sides, so "
      f"only photograph-specific variation is left): pearson r={r_dm:.4f} "
      + (f"95% CI [{np.percentile(dm_boot, 2.5):.4f},"
         f"{np.percentile(dm_boot, 97.5):.4f}]  " if len(dm_boot) else "")
      + f"shuffled-item null mean {dm_null.mean():.4f} p95 "
        f"{np.percentile(dm_null, 95):.4f}")
within = []
for i2 in range(len(common)):
    x, y = piv_I2.to_numpy()[i2], piv_D2.to_numpy()[i2]
    if np.std(x) > 0 and np.std(y) > 0:
        within.append(float(np.corrcoef(x, y)[0, 1]))
w = boot_mean(within, seed=14)
print(f"within-photograph pearson r across the 6 regions, averaged over "
      f"photographs: {w['mean']:.4f} [{w['lo']:.4f},{w['hi']:.4f}] "
      f"(n={w['n']}; each is a 6-point correlation)")

comp = np.array([x["completeness_rel_err"] for x in items_kept])
print(f"\ncompleteness of the analytic decomposition (sum_R I_R vs the "
      f"full-image mean-swap DiD), steps={a.steps_margin}: median rel_err "
      f"{np.median(comp):.4f}  max {comp.max():.4f}")

flip_summary = None
if not fl.empty:
    print("\n" + "=" * 90)
    print("FULL GENERATED-ANSWER FLIPS")
    print("=" * 90)
    flip_summary = fl.groupby("region")[["flip_a", "flip_b"]].mean().reset_index()
    print(flip_summary.round(3).to_string(index=False))
    print("\ncounts:\n",
          fl.groupby("region")[["flip_a", "flip_b"]].sum().to_string())

# ------------------------------------------------------------------ dump json
rep = dict(
    repo=a.repo, tag=tag, dataset="COCO val2017", jlens=a.jlens if J else None,
    n_items=len(items_kept), n_candidates=n_cand, n_ok_a=n_ok_a, n_ok_b=n_ok_b,
    n_ok_both=n_ok_both, layers=LAYERS, head_layer=HEAD_LAYER, grid=G,
    n_patches=P, steps_token=a.steps_token, steps_margin=a.steps_margin,
    chunk_token=a.chunk_token, chunk_margin=a.chunk_margin, mean_images=nm,
    register_tokens=REG, region_names=REGION_NAMES,
    axes=st.jsonable(dict(Counter(x["axis"] for x in items_kept))),
    pairs=st.jsonable([f"{x['cat_a']}|{x['cat_b']}" for x in items_kept]),
    prompt_blind=dict(max_residual_delta=max(blind_delta),
                      max_lens_map_delta=max(blind_IR), I_R=0.0),
    localisation_patch=st.jsonable(loc.round(6).to_dict("records")),
    localisation_within_half=dict(hit=hw,
                                  chance=float(vj_head.chance_half.mean())),
    localisation_patch_head=dict(hit=own_hit, hit5=own_hit5,
                                 chance1=float(vj_head.chance.mean()),
                                 chance5=float(vj_head.chance5.mean()),
                                 cross_region_hit=cross_rate),
    localisation_region=dict(
        hit_a=ha, hit_b=hb, both=bo,
        chance_a=float(ra_df.chance_a.mean()),
        chance_b=float(ra_df.chance_b.mean()),
        chance_both=float(ra_df.chance_both.mean()),
        excess_a=d_a, excess_b=d_b, excess_both=d_bo),
    ablation_table=st.jsonable(tdf.round(6).to_dict("records")),
    nomination_table=st.jsonable(ndf.round(6).to_dict("records")),
    selective_effect_ratio=dict(region_a=ratio_a, region_b=ratio_b,
                                paired_a=da, paired_b=db,
                                full_crossover_rate=both_ok),
    interaction_correlation=dict(
        n_pairs=len(rg), n_items=int(rg.item.nunique()),
        pearson=real["pearson"], spearman=real["spearman"],
        r2_identity=real["r2_identity"], rel_err=real["rel_err"],
        pearson_ci=[float(np.percentile(boots, 2.5)),
                    float(np.percentile(boots, 97.5))] if len(boots)
        else [float("nan")] * 2,
        null_mean=float(null_rs.mean()) if len(null_rs) else float("nan"),
        null_p95=float(np.percentile(null_rs, 95)) if len(null_rs) else float("nan"),
        demeaned_pearson=r_dm,
        demeaned_ci=[float(np.percentile(dm_boot, 2.5)),
                     float(np.percentile(dm_boot, 97.5))] if len(dm_boot)
        else [float("nan")] * 2,
        demeaned_null_mean=float(dm_null.mean()),
        demeaned_null_p95=float(np.percentile(dm_null, 95)),
        within_item=w),
    completeness=dict(steps=a.steps_margin, median_rel_err=float(np.median(comp)),
                      max_rel_err=float(comp.max())),
    flips=st.jsonable(flip_summary.round(4).to_dict("records"))
    if flip_summary is not None else None,
    lens_available=lens_available,
    minutes=round((time.time() - t0) / 60, 1))
json.dump(rep, open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
print(f"\nwrote {a.out}/{tag}_stats.json  ({rep['minutes']} min)", flush=True)
