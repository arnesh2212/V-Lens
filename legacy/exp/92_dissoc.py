# SPDX-License-Identifier: Apache-2.0
r"""The paper's headline result: a double dissociation on one image.

Two objects, two questions, one photograph.  q_a = "What colour is the
{shape_a}?", q_b = "What colour is the {shape_b}?" on a CLEVR scene where
shape_a and shape_b each name exactly one object, so the image activations at
every image token are byte-identical between the two prompts (causal mask,
verified the same way 91_interaction.py verifies it) and only the question
changes.  Region A is object a's token footprint, region B is object b's.

THE THREE MEASUREMENTS, in the order the RUN loop makes them.

(1) LOCALISATION.  Does VJ-Lens's per-patch reading of "the colour of a" under
    q_a point at region A, and of "the colour of b" under q_b point at region
    B?  Scored with :func:`jvlm.vmap.pointing` -- the saliency literature's
    pointing game -- against the scene graph's true object footprint, chance
    reported from the grid rather than assumed.  Token targets, steps=8 (fixed,
    see below).

(2) BEHAVIOUR.  Ablating region A (replacing its tokens with the per-position
    corpus mean, everything else in the picture left real) should damage the
    model's *answer* to q_a much more than its answer to q_b, and ablating
    region B should show the opposite pattern.  "Answer" here is the margin
    objective from :mod:`jvlm.objective` -- log P(correct) - log P(foil),
    teacher-forced over the whole generated word, which is what a forced choice
    actually is -- not a single-token logit.  The foil for q_a is object b's
    own colour and vice versa, so the margin is literally "does the picture
    make the model prefer a's colour over b's, for this question": exactly the
    quantity a cross-modal interaction should predict.  Reported beside a
    random region matched in size and drawn from neither object, and beside
    the two prompt-blind lenses' own top-k patches (logit lens, J-lens), which
    are read once -- under q_a's colour token, since neither lens can read a
    question at all -- and scored the same way.  Margin objective, steps=32
    (fixed, see below).

(3) DOES THE INTERACTION PREDICT THE BEHAVIOUR.  I_R(q_a,q_b) = A_R(q_a) -
    A_R(q_b), the per-region attribution difference of :mod:`jvlm.interact`'s
    object, generalised here to the margin objective by composing
    :func:`jvlm.objective.ig_map` with :func:`jvlm.interact.region_sums` (no
    line of either module is edited; every number here is produced by calling
    into them).  Completeness -- sum_R I_R equal to the *analytic*
    difference-in-differences obtained by replacing the whole image with the
    mean image -- is checked per item.  The number that matters is a different
    one: does I_R, read off the real image with no ablation at all, correlate
    across regions with the *behavioural*, single-region-ablation
    difference-in-differences measured in (2)?  That correlation is not
    implied by completeness (a single-region ablation is not the same
    counterfactual as the path-integral baseline swap the law is about) and is
    therefore the paper's actual empirical claim.

WHAT IS FIXED IN ADVANCE, AND WHY.
  * LAYERS: three source layers spread across the usable depth (20%-85% of
    n_deepstack..n_layers-2), the same recipe 91_interaction.py and
    80_pointer.py use.  HEAD_LAYER is the middle one of the three.  All
    behavioural and interaction numbers use HEAD_LAYER alone; LAYERS is kept
    only for the (cheap, token-level) localisation robustness table.  Nothing
    here is chosen after looking at a result.
  * steps=8 for token-level readings (do_map/vmap.ig_map, item 1) and steps=32
    for the margin objective (item 2/3), per the measured convergence numbers
    in the task brief (rel error 0.357 at steps=4, 0.154 at steps=8; the
    log-softmax margin is more curved and needs the higher count).  Both are
    CLI-overridable but default to the measured-sufficient values and every
    reported number uses the default.
  * The partition per item is {A, B, four quadrants of everything else}: the
    two object regions plus a fixed, image-geometry-only split of the
    remainder (never chosen from a region's own attribution, which would be
    selecting on the dependent variable).

Run ``--n 3 --scan 200 --mean-images 8 --n-layers 2 --gen-n 2`` first: the
smoke test the brief asks for, end to end including ablation and the answer
comparison, before anything is queued at full size.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from math import comb

sys.path.insert(0, "code")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens",
                default="results/lenses/"
                        "Qwen3-VL-2B-Instruct__mm__all__n120.pt")
ap.add_argument("--n", type=int, default=24, help="items kept")
ap.add_argument("--scan", type=int, default=3000, help="CLEVR scenes scanned")
ap.add_argument("--mean-images", type=int, default=90)
ap.add_argument("--layers", type=int, nargs="*", default=None,
                help="read at these layers instead of sampling across depth, "
                     "so they can be made to match a fitted J-lens")
ap.add_argument("--n-layers", type=int, default=3,
                help="VJ-Lens source layers sampled across depth")
ap.add_argument("--steps-token", type=int, default=8)
ap.add_argument("--steps-margin", type=int, default=32)
ap.add_argument("--chunk-margin", type=int, default=8)
ap.add_argument("--n-obj-lo", type=int, default=3)
ap.add_argument("--n-obj-hi", type=int, default=6)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--suffix", default=" Answer with one word.")
ap.add_argument("--strict", action="store_true",
                help="require the colour word to be the single top token, "
                     "not merely the top colour")
ap.add_argument("--gen-n", type=int, default=12,
                help="items scored for full generated-answer flips")
ap.add_argument("--gen-tokens", type=int, default=6)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--null-perm", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/dissoc")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

import jvlm
from jvlm import clevrdata as cd
from jvlm import corpus, interact, objective, readout, stats as st, vmap

# ============================================================== model + fixed
m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lo, hi = max(m.n_deepstack, 1), m.n_layers - 2
LAYERS = sorted({int(round(lo + (hi - lo) * f))
                 for f in np.linspace(0.2, 0.85, a.n_layers)})
HEAD_LAYER = LAYERS[len(LAYERS) // 2]
J = None
if a.jlens and os.path.exists(a.jlens):
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))

    # Read where the lens exists.  A fitted Jacobian lens only has a transport
    # at the layers it was fitted at; chosen from the model's geometry instead,
    # the depths miss it and the arm is dropped in silence.
    LAYERS, _ = readout.snap_to_lens(LAYERS, J)
print(m, f"\nLAYERS {LAYERS}  HEAD_LAYER {HEAD_LAYER} (fixed in advance, not "
      f"chosen from a result)  steps_token={a.steps_token} "
      f"steps_margin={a.steps_margin}  jlens={'yes' if J else 'no'}",
      flush=True)

# ------------------------------------------------------------- colour tokens
def variant_forms(w: str) -> list[str]:
    return [w, w.capitalize(), " " + w, " " + w.capitalize()]


def single_token_variants(word: str) -> list[tuple[str, int]]:
    out = []
    for s in variant_forms(word):
        ids = m.tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            out.append((s, int(ids[0])))
    return out


CVAR = {c: single_token_variants(c) for c in cd.COLOURS}
CVAR = {c: v for c, v in CVAR.items() if v}
print(f"{len(CVAR)}/{len(cd.COLOURS)} colours single-token: "
      + ", ".join(f"{c}:{len(v)}" for c, v in CVAR.items()), flush=True)


def prep(path, question):
    return m.chat(question + a.suffix, [corpus.load_image(path, square=a.square)])


def clean_logits(batch):
    h = readout.residuals(m, batch, [m.n_layers - 1])[m.n_layers - 1]
    apos = batch["input_ids"].shape[1] - 1
    with torch.no_grad():
        return m.unembed(h[apos][None])[0].float().cpu()


def colour_choice(lg, colour):
    """(text, id, logit) of ``colour``'s best-scoring single-token surface form."""
    scored = [(txt, tid, float(lg[tid])) for txt, tid in CVAR[colour]]
    return max(scored, key=lambda x: x[2])


def top_colour(lg):
    """(best colour name, {colour: (text, id, logit)}) over every CVAR colour."""
    per = {c: colour_choice(lg, c) for c in CVAR}
    top = max(per, key=lambda c: per[c][2])
    return top, per


# ------------------------------------------------------------------ the grid
first_scene = next(s for s in cd.scenes() if cd.have(s))
b0 = prep(cd.image_path(first_scene), "What is this?")
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
if G * G != P:
    raise SystemExit(f"{P} image tokens is not a square grid")
print(f"{P} image tokens on {G}x{G}", flush=True)


def quadrant(c: int) -> int:
    r, col = divmod(c, G)
    return (r >= G // 2) * 2 + (col >= G // 2)


def chance5(n_inside: int, P_total: int, k: int = 5) -> float:
    if n_inside <= 0:
        return 0.0
    if k > P_total - n_inside:
        return 1.0
    return 1.0 - comb(P_total - n_inside, k) / comb(P_total, k)


# -------------------------------------------------------------- the baseline
# A per-position mean over CLEVR renders drawn from the far end of the image
# index -- held out from the forward scan below, which starts at scene 0 --
# per the warning in 80_pointer.py: a COCO mean would inject "this is a
# synthetic render" into every covered position and swamp the thing measured.
pool = sorted(cd.scenes(), key=lambda s: -s["image_index"])
pool = [cd.image_path(s) for s in pool if cd.have(s)][: a.mean_images + 30]
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
nrm = {l: torch.zeros(P, dtype=torch.float64) for l in LAYERS}
nm = 0
for p in pool:
    if nm >= a.mean_images:
        break
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
print(f"means from {nm} held-out CLEVR renders (tail of the val index); "
      f"{REG} register-like tokens by norm (kept in every region -- excluding "
      f"them would put a hole in the conservation law)  {time.time()-t0:.0f}s",
      flush=True)


# ------------------------------------------------------------- item selection
def two_object_items(n_obj, scan, balance=4.0, min_tok=2, skip=0):
    """CLEVR scenes with two objects, each named unambiguously by its shape
    alone, disjoint token footprints, and different (single-token) colours.

    'What colour is the {shape}?' answers in one glance -- no relation, no
    landmark to resolve first -- which is what a *localisation-and-ablation*
    dissociation wants: two independently addressable regions on one frame and
    nothing in the question text to shortcut either of them.  Every object in
    the scene, not just the chosen pair, is required to have a clean,
    non-overlapping box: the same quality bar 80_pointer.py's constructors use,
    because a messy occlusion elsewhere in frame makes every box in the scene
    suspect.
    """
    seen = 0
    for si, s in enumerate(cd.scenes()):
        if si < skip:
            continue
        if seen >= scan:
            return
        seen += 1
        objs = s["objects"]
        if not (n_obj[0] <= len(objs) <= n_obj[1]) or not cd.have(s):
            continue
        bx = cd.object_boxes(s)
        if any(bx[j] is None or not bx[j]["clean"] for j in range(len(objs))):
            continue
        toks = {j: set(cd.box_tokens(bx[j]["box"], G)) for j in range(len(objs))}
        if any(not toks[j] for j in toks):
            continue
        if any(toks[i] & toks[j] for i in toks for j in toks if i < j):
            continue
        shape_owner: dict[str, list[int]] = {}
        for j, o in enumerate(objs):
            shape_owner.setdefault(o["shape"], []).append(j)
        uniq = [j for js in shape_owner.values() if len(js) == 1 for j in js]
        best = None
        for oa in uniq:
            for ob in uniq:
                if oa == ob or objs[oa]["color"] == objs[ob]["color"]:
                    continue
                ca, cb = objs[oa]["color"], objs[ob]["color"]
                if ca not in CVAR or cb not in CVAR:
                    continue
                na, nb = len(toks[oa]), len(toks[ob])
                if na < min_tok or nb < min_tok:
                    continue
                if max(na, nb) > balance * min(na, nb):
                    continue
                best = (oa, ob)
                break
            if best:
                break
        if not best:
            continue
        oa, ob = best
        yield dict(
            scene_index=si, image_index=s["image_index"], path=cd.image_path(s),
            n_objects=len(objs), obj_a=oa, obj_b=ob,
            tok_a=sorted(toks[oa]), tok_b=sorted(toks[ob]),
            colour_a=objs[oa]["color"], colour_b=objs[ob]["color"],
            shape_a=objs[oa]["shape"], shape_b=objs[ob]["shape"],
            question_a=f"What colour is the {cd.SAY[objs[oa]['shape']]}?",
            question_b=f"What colour is the {cd.SAY[objs[ob]['shape']]}?")


# ------------------------------------------------------------------ ablation
def ipos_of(cells) -> torch.Tensor:
    return IPOS[torch.as_tensor(sorted(cells), dtype=torch.long)]


def ablate_effect(obj, layer, cells, f_intact: float) -> tuple[float, float]:
    """F(intact) - F(region ablated); positive = ablation hurt this answer."""
    if not cells:
        return 0.0, f_intact
    idx = torch.as_tensor(sorted(cells), dtype=torch.long)
    rep = MEANS[layer][idx]
    f0 = float(objective.value(m, obj, replace=rep, layer=layer,
                               positions=ipos_of(cells))[0])
    return f_intact - f0, f0


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
    """The model's own greedy words, optionally with a region ablated for the
    whole generation (the hook is applied at prefill and rides the KV cache
    through decoding, so it need not re-fire on the single-token steps)."""
    hd = (m.layers[layer].register_forward_hook(_gen_hook(layer, cells))
          if cells is not None else None)
    try:
        ids = objective.generated_answer(m, batch, max_new_tokens=max_new)
    finally:
        if hd is not None:
            hd.remove()
    return m.tokenizer.decode(ids).strip().lower()


REGION_NAMES = ["A", "B", "Q0", "Q1", "Q2", "Q3"]

# ============================================================== the main loop
point_rows: list[dict] = []
region_rows: list[dict] = []
ablate_rows: list[dict] = []
flip_rows: list[dict] = []
items_kept: list[dict] = []
n_cand = n_ok_both = 0
lens_available = {"logit lens": True, "J-lens": bool(J)}

pq, rq, aq, fq = (f"{a.out}/{tag}_point.parquet", f"{a.out}/{tag}_region.parquet",
                  f"{a.out}/{tag}_ablate.parquet", f"{a.out}/{tag}_flips.parquet")


def checkpoint():
    pd.DataFrame(point_rows).to_parquet(pq)
    pd.DataFrame(region_rows).to_parquet(rq)
    if ablate_rows:
        pd.DataFrame(ablate_rows).to_parquet(aq)
    if flip_rows:
        pd.DataFrame(flip_rows).to_parquet(fq)
    json.dump(st.jsonable(items_kept), open(f"{a.out}/{tag}_items.json", "w"),
              indent=2, default=float)


for it_i, it in enumerate(two_object_items((a.n_obj_lo, a.n_obj_hi), a.scan)):
    if len(items_kept) >= a.n:
        break
    n_cand += 1
    ba = prep(it["path"], it["question_a"])
    bb = prep(it["path"], it["question_b"])
    if (len(readout.image_positions(m, ba)) != P
            or len(readout.image_positions(m, bb)) != P):
        continue

    lg_a, lg_b = clean_logits(ba), clean_logits(bb)
    top_a, sc_a = top_colour(lg_a)
    top_b, sc_b = top_colour(lg_b)
    ok_a = (top_a == it["colour_a"]) and (
        not a.strict or int(lg_a.argmax()) == sc_a[it["colour_a"]][1])
    ok_b = (top_b == it["colour_b"]) and (
        not a.strict or int(lg_b.argmax()) == sc_b[it["colour_b"]][1])
    if not (ok_a and ok_b):
        continue
    n_ok_both += 1

    text_a, id_a, _ = sc_a[it["colour_a"]]     # colour_a's own surface form
    text_b, id_b, _ = sc_b[it["colour_b"]]     # colour_b's own surface form
    tok_a, tok_b = set(it["tok_a"]), set(it["tok_b"])

    # each question scored on its own natural correct-vs-foil margin: q_a's
    # correct is colour_a and its foil is object b's own colour, and vice
    # versa -- the pointer.py-precedent "wrong but salient" foil, here made
    # exact because both objects genuinely sit in the one photograph.
    obj_a = objective.margin(m, ba, [(text_a, text_b)])
    obj_b = objective.margin(m, bb, [(text_b, text_a)])

    # ---------------------------------------------------- (1) localisation
    h_all = readout.residuals(m, ba, LAYERS)          # one forward, every layer
    h_img = h_all[HEAD_LAYER][IPOS].float().cpu()
    for l in LAYERS:
        heat_a = vmap.ig_map(m, ba, [id_a], [l], IPOS, means=MEANS,
                             steps=a.steps_token)[l][0]
        heat_b = vmap.ig_map(m, bb, [id_b], [l], IPOS, means=MEANS,
                             steps=a.steps_token)[l][0]
        pt_a, pt_b = vmap.pointing(heat_a, tok_a, G), vmap.pointing(heat_b, tok_b, G)
        cross_a = vmap.pointing(heat_a, tok_b, G)["hit"]   # specificity: should be low
        cross_b = vmap.pointing(heat_b, tok_a, G)["hit"]
        point_rows.append(dict(item=it_i, scene=it["scene_index"], layer=l,
                                lens="VJ-Lens", q="a", n_inside=len(tok_a),
                                hit=pt_a["hit"], hit5=pt_a["hit5"],
                                chance=pt_a["chance"],
                                chance5=chance5(len(tok_a), P),
                                mass_in=pt_a["mass_in"], cross_hit=cross_a))
        point_rows.append(dict(item=it_i, scene=it["scene_index"], layer=l,
                                lens="VJ-Lens", q="b", n_inside=len(tok_b),
                                hit=pt_b["hit"], hit5=pt_b["hit5"],
                                chance=pt_b["chance"],
                                chance5=chance5(len(tok_b), P),
                                mass_in=pt_b["mass_in"], cross_hit=cross_b))
        hL = h_all[l][IPOS].float().cpu() if l != HEAD_LAYER else h_img
        jacL = J.jacobians.get(l) if J is not None and l in J.jacobians else None
        fl = interact.fixed_lens_patches(m, hL, [id_a, id_b], jacobian=jacL)
        for lens_name, hh in fl.items():
            pa, pb = vmap.pointing(hh[0], tok_a, G), vmap.pointing(hh[1], tok_b, G)
            point_rows.append(dict(item=it_i, scene=it["scene_index"], layer=l,
                                    lens=lens_name, q="a", n_inside=len(tok_a),
                                    hit=pa["hit"], hit5=pa["hit5"],
                                    chance=pa["chance"], chance5=chance5(len(tok_a), P),
                                    mass_in=pa["mass_in"], cross_hit=None))
            point_rows.append(dict(item=it_i, scene=it["scene_index"], layer=l,
                                    lens=lens_name, q="b", n_inside=len(tok_b),
                                    hit=pb["hit"], hit5=pb["hit5"],
                                    chance=pb["chance"], chance5=chance5(len(tok_b), P),
                                    mass_in=pb["mass_in"], cross_hit=None))

    # ------------------------------------------- (3) the interaction object
    # I_R = A_R(q_a; margin_a) - A_R(q_b; margin_b): a different NATURAL
    # objective per question, which is still exactly the interact.py law by
    # linearity of completeness (each term sums to its own effect regardless
    # of what the other objective is) -- verified in objective.py's own
    # selftest, "(12) the interaction object", not re-derived here.
    scores_a = objective.ig_map(m, obj_a, [HEAD_LAYER], IPOS, MEANS,
                                steps=a.steps_margin, chunk=a.chunk_margin)[HEAD_LAYER]
    scores_b = objective.ig_map(m, obj_b, [HEAD_LAYER], IPOS, MEANS,
                                steps=a.steps_margin, chunk=a.chunk_margin)[HEAD_LAYER]

    quad_cells: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
    for c in range(P):
        if c in tok_a or c in tok_b:
            continue
        quad_cells[quadrant(c)].append(c)
    region_cells = [sorted(tok_a), sorted(tok_b),
                    quad_cells[0], quad_cells[1], quad_cells[2], quad_cells[3]]
    REGIONS = [torch.tensor(c, dtype=torch.long) for c in region_cells]
    interact.check_partition(REGIONS, P)

    ra = interact.region_sums(scores_a, REGIONS)[0]
    rb = interact.region_sums(scores_b, REGIONS)[0]
    I_R = (ra - rb).tolist()

    Fa1 = float(objective.value(m, obj_a)[0])
    Fb1 = float(objective.value(m, obj_b)[0])
    Fa0 = float(objective.value(m, obj_a, replace=MEANS[HEAD_LAYER], layer=HEAD_LAYER,
                                positions=IPOS)[0])
    Fb0 = float(objective.value(m, obj_b, replace=MEANS[HEAD_LAYER], layer=HEAD_LAYER,
                                positions=IPOS)[0])
    did_truth = (Fa1 - Fa0) - (Fb1 - Fb0)
    did_claim = float(sum(I_R))
    comp_rel_err = abs(did_claim - did_truth) / max(abs(did_truth), 1e-6)

    # ------------------------------- (2) behavioural ablation, per region --
    # this is the SAME quantity item 3 correlates I_R against, but measured
    # by really ablating just that region (rest of the image left real), not
    # by the analytic full-image mean swap above.
    for name, cells in zip(REGION_NAMES, region_cells):
        eff_a, _ = ablate_effect(obj_a, HEAD_LAYER, cells, Fa1)
        eff_b, _ = ablate_effect(obj_b, HEAD_LAYER, cells, Fb1)
        region_rows.append(dict(
            item=it_i, scene=it["scene_index"], region=name, n_tok=len(cells),
            I_R=I_R[REGION_NAMES.index(name)], effect_a=eff_a, effect_b=eff_b,
            did_measured=eff_a - eff_b))

    # ------------------------------------------ extra rows for item 2's table
    k = max(len(tok_a), 1)
    free_cells = [c for c in range(P) if c not in tok_a and c not in tok_b]
    rgen = np.random.default_rng(a.seed * 100_003 + it["scene_index"])
    rand_cells = sorted(rgen.choice(free_cells, size=min(k, len(free_cells)),
                                    replace=False).tolist())
    eff_a, _ = ablate_effect(obj_a, HEAD_LAYER, rand_cells, Fa1)
    eff_b, _ = ablate_effect(obj_b, HEAD_LAYER, rand_cells, Fb1)
    ablate_rows.append(dict(item=it_i, scene=it["scene_index"], row="random matched",
                            n_tok=len(rand_cells), effect_a=eff_a, effect_b=eff_b))

    # the two prompt-blind lenses cannot define an interaction (I_R is
    # identically zero for both: neither reading is a function of the
    # question), so they are scored here by their OWN top-k patches for
    # colour_a's token under q_a -- the only reading they can produce -- and
    # that same fixed region's effect on both answers is reported.
    lens_maps = interact.fixed_lens_patches(m, h_img, [id_a],
                                            jacobian=J.jacobians.get(HEAD_LAYER)
                                            if J is not None
                                            and HEAD_LAYER in J.jacobians else None)
    for lens_name, heat in lens_maps.items():
        topk = torch.topk(heat[0], min(k, P)).indices.tolist()
        eff_a, _ = ablate_effect(obj_a, HEAD_LAYER, topk, Fa1)
        eff_b, _ = ablate_effect(obj_b, HEAD_LAYER, topk, Fb1)
        # the row label deliberately omits the per-item k (n_tok carries it):
        # k = len(tok_a) varies by item, and grouping by a k-suffixed label
        # would fragment one lens's rows across items into separate buckets.
        ablate_rows.append(dict(item=it_i, scene=it["scene_index"],
                                row=f"{lens_name} top-k (under q_a; I_R=0)",
                                n_tok=len(topk), effect_a=eff_a, effect_b=eff_b))

    # ------------------------------------------------ full-answer flips (few)
    if it_i < a.gen_n:
        try:
            gen_a0 = generate_text(ba, max_new=a.gen_tokens)
            gen_b0 = generate_text(bb, max_new=a.gen_tokens)
            for rname, cells in (("A", sorted(tok_a)), ("B", sorted(tok_b)),
                                 ("random", rand_cells)):
                gen_a1 = generate_text(ba, HEAD_LAYER, cells, a.gen_tokens)
                gen_b1 = generate_text(bb, HEAD_LAYER, cells, a.gen_tokens)
                flip_rows.append(dict(
                    item=it_i, scene=it["scene_index"], region=rname,
                    gen_a0=gen_a0, gen_a1=gen_a1, flip_a=int(gen_a0 != gen_a1),
                    gen_b0=gen_b0, gen_b1=gen_b1, flip_b=int(gen_b0 != gen_b1)))
        except Exception as e:                                   # noqa: BLE001
            print(f"  [warn] generation flips failed on item {it_i}: "
                  f"{type(e).__name__}: {str(e)[:120]}", flush=True)

    items_kept.append(dict(
        it, top_a=top_a, top_b=top_b, text_a=text_a, text_b=text_b,
        id_a=id_a, id_b=id_b, F_a=Fa1, F_b=Fb1, F_a0=Fa0, F_b0=Fb0,
        did_truth=did_truth, did_claim=did_claim, completeness_rel_err=comp_rel_err))

    el = time.time() - t0
    if len(items_kept) % 4 == 0 or len(items_kept) == a.n:
        print(f"  {len(items_kept)}/{a.n} kept  cand {n_cand}  "
              f"did_truth={did_truth:+.3f} did_claim={did_claim:+.3f} "
              f"rel_err={comp_rel_err:.3f}  {el:.0f}s  "
              f"eta {el/max(len(items_kept),1)*(a.n-len(items_kept))/60:.1f}m",
              flush=True)
        checkpoint()

checkpoint()
print(f"\nkept {len(items_kept)}/{n_cand} candidates scanned "
      f"(both answers correct: {n_ok_both})", flush=True)

if not items_kept:
    print("NOTHING KEPT -- the model does not answer these questions, or none "
          "were found; nothing to summarise", flush=True)
    json.dump({"repo": a.repo, "kept": 0, "candidates": n_cand}, open(
        f"{a.out}/{tag}_stats.json", "w"), indent=2)
    raise SystemExit(0)

# ==================================================================== report
pt = pd.DataFrame(point_rows)
rg = pd.DataFrame(region_rows)
ab = pd.DataFrame(ablate_rows) if ablate_rows else pd.DataFrame()
fl = pd.DataFrame(flip_rows) if flip_rows else pd.DataFrame()
rng = np.random.default_rng(a.seed)


def boot_mean(values, n=a.boot, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    rg_ = np.random.default_rng(seed)
    boots = values[rg_.integers(0, len(values), (n, len(values)))].mean(1)
    return dict(mean=float(values.mean()), lo=float(np.percentile(boots, 2.5)),
               hi=float(np.percentile(boots, 97.5)), n=int(len(values)))


def boot_by_item(df, value_col, item_col="item", seed=0):
    per_item = df.groupby(item_col)[value_col].mean()
    return boot_mean(per_item.to_numpy(), seed=seed)


def boot_ratio_by_item(df, num_col, den_col, item_col="item", n=a.boot, seed=0):
    """Bootstrap CI of mean(num)/mean(den), resampling ITEMS with replacement."""
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


# --------------------------------------------------- (1) localisation report
print("\n" + "=" * 88)
print("(1) LOCALISATION -- pointing game, VJ-Lens vs the two prompt-blind lenses")
print("=" * 88)
loc = (pt.groupby(["lens", "layer"])
      .agg(n=("item", "count"), hit=("hit", "mean"), hit5=("hit5", "mean"),
           chance=("chance", "mean"), chance5=("chance5", "mean"),
           mass_in=("mass_in", "mean")).reset_index())
print(loc.round(4).to_string(index=False))
vj_head = pt[(pt.lens == "VJ-Lens") & (pt.layer == HEAD_LAYER)]
own_hit = boot_by_item(vj_head, "hit")
own_hit5 = boot_by_item(vj_head, "hit5")
cross = vj_head.cross_hit.dropna()
cross_rate = float(cross.mean()) if len(cross) else float("nan")
print(f"\nVJ-Lens @ HEAD_LAYER={HEAD_LAYER}: own-region hit "
      f"{own_hit['mean']:.3f} [{own_hit['lo']:.3f},{own_hit['hi']:.3f}]  "
      f"hit5 {own_hit5['mean']:.3f} [{own_hit5['lo']:.3f},{own_hit5['hi']:.3f}]  "
      f"chance(top1) {vj_head.chance.mean():.3f}  chance(top5) "
      f"{vj_head.chance5.mean():.3f}  cross-region top1 hit rate "
      f"(should be low): {cross_rate:.3f}", flush=True)

# ------------------------------------------------ (2) behavioural ablation table
print("\n" + "=" * 88)
print(f"(2) BEHAVIOURAL ABLATION @ HEAD_LAYER={HEAD_LAYER}, margin objective, "
      f"steps={a.steps_margin}")
print("=" * 88)
table_rows = []
for name, sub in (("VJ region for A", rg[rg.region == "A"]),
                  ("VJ region for B", rg[rg.region == "B"])):
    ea = boot_by_item(sub, "effect_a")
    eb = boot_by_item(sub, "effect_b")
    table_rows.append(dict(row=name, n=ea["n"],
                           effect_a=ea["mean"], effect_a_lo=ea["lo"], effect_a_hi=ea["hi"],
                           effect_b=eb["mean"], effect_b_lo=eb["lo"], effect_b_hi=eb["hi"]))
if not ab.empty:
    for name, sub in ab.groupby("row"):
        ea = boot_by_item(sub, "effect_a")
        eb = boot_by_item(sub, "effect_b")
        table_rows.append(dict(row=name, n=ea["n"],
                               effect_a=ea["mean"], effect_a_lo=ea["lo"],
                               effect_a_hi=ea["hi"], effect_b=eb["mean"],
                               effect_b_lo=eb["lo"], effect_b_hi=eb["hi"]))
tdf = pd.DataFrame(table_rows)
print(tdf.round(4).to_string(index=False))

ratio_a = boot_ratio_by_item(rg[rg.region == "A"], "effect_a", "effect_b")
ratio_b = boot_ratio_by_item(rg[rg.region == "B"], "effect_b", "effect_a")
print(f"\nselective effect ratio, region A (effect on A / effect on B): "
      f"{ratio_a['ratio']:.2f} [{ratio_a['lo']:.2f},{ratio_a['hi']:.2f}]")
print(f"selective effect ratio, region B (effect on B / effect on A): "
      f"{ratio_b['ratio']:.2f} [{ratio_b['lo']:.2f},{ratio_b['hi']:.2f}]")

# ---------------------------------------------------------- (3) correlation
print("\n" + "=" * 88)
print("(3) DOES I_R PREDICT THE MEASURED DiD -- per-region correlation, "
      "images resampled")
print("=" * 88)
real = interact.error_stats(rg["I_R"].to_numpy(), rg["did_measured"].to_numpy())
print(f"pooled over {len(rg)} (item, region) pairs, {rg.item.nunique()} items, "
      f"{len(REGION_NAMES)} regions/item: pearson={real['pearson']:.4f} "
      f"spearman={real['spearman']:.4f} r2_identity={real['r2_identity']:.4f} "
      f"rel_err={real['rel_err']:.4f}")

items_ = rg.item.unique()
groups = {i: g for i, g in rg.groupby("item")}
boots = []
for _ in range(a.boot):
    chosen = rng.choice(items_, size=len(items_), replace=True)
    sub = pd.concat([groups[c] for c in chosen], ignore_index=True)
    if sub["I_R"].std() > 0 and sub["did_measured"].std() > 0:
        boots.append(float(np.corrcoef(sub["I_R"], sub["did_measured"])[0, 1]))
boots = np.array(boots)
print(f"paired bootstrap over images ({len(boots)}/{a.boot} valid resamples): "
      f"pearson r 95% CI [{np.percentile(boots, 2.5):.4f}, "
      f"{np.percentile(boots, 97.5):.4f}]" if len(boots) else
      "paired bootstrap: no valid resamples (degenerate variance)")

# the wrong-pairing floor: same estimator, same scale, item labels shuffled
piv_I = rg.pivot(index="item", columns="region", values="I_R")[REGION_NAMES]
piv_D = rg.pivot(index="item", columns="region", values="did_measured")[REGION_NAMES]
common = piv_I.index.intersection(piv_D.index)
I_mat, D_mat = piv_I.loc[common].to_numpy(), piv_D.loc[common].to_numpy()
null_rs = []
for _ in range(a.null_perm):
    perm = rng.permutation(len(common))
    if D_mat[perm].std() > 0 and I_mat.std() > 0:
        null_rs.append(float(np.corrcoef(I_mat.flatten(), D_mat[perm].flatten())[0, 1]))
null_rs = np.array(null_rs)
print(f"null floor (item labels shuffled, {len(null_rs)} permutations): "
      f"mean r={null_rs.mean():.4f}  p95={np.percentile(null_rs, 95):.4f}"
      if len(null_rs) else "null floor: no valid permutations")

comp = np.array([it["completeness_rel_err"] for it in items_kept])
print(f"\ncompleteness of the analytic decomposition (sum_R I_R vs the "
      f"full-image mean-swap DiD), steps={a.steps_margin}: median rel_err "
      f"{np.median(comp):.4f}  max {comp.max():.4f}")

# ---------------------------------------------------------------- (4) flips
flip_summary = None
if not fl.empty:
    print("\n" + "=" * 88)
    print("FULL GENERATED-ANSWER FLIPS")
    print("=" * 88)
    flip_summary = fl.groupby("region")[["flip_a", "flip_b"]].mean().reset_index()
    counts = fl.groupby("region")[["flip_a", "flip_b"]].sum().reset_index()
    print(flip_summary.round(3).to_string(index=False))
    print("\ncounts:\n", counts.to_string(index=False))

# ------------------------------------------------------------------ dump json
rep = dict(
    repo=a.repo, tag=tag, jlens=a.jlens if J is not None else None,
    n_items=len(items_kept), n_candidates=n_cand, n_ok_both=n_ok_both,
    layers=LAYERS, head_layer=HEAD_LAYER, grid=G, n_patches=P,
    steps_token=a.steps_token, steps_margin=a.steps_margin,
    chunk_margin=a.chunk_margin, mean_images=nm, register_tokens=REG,
    colours_single_token=list(CVAR.keys()),
    region_names=REGION_NAMES,
    localisation=st.jsonable(loc.round(6).to_dict("records")),
    localisation_head_layer=dict(hit=own_hit, hit5=own_hit5,
                                 chance1=float(vj_head.chance.mean()),
                                 chance5=float(vj_head.chance5.mean()),
                                 cross_region_hit=cross_rate),
    ablation_table=st.jsonable(tdf.round(6).to_dict("records")),
    selective_effect_ratio=dict(region_a=ratio_a, region_b=ratio_b),
    interaction_correlation=dict(
        n_pairs=len(rg), n_items=int(rg.item.nunique()),
        pearson=real["pearson"], spearman=real["spearman"],
        r2_identity=real["r2_identity"], rel_err=real["rel_err"],
        pearson_ci=[float(np.percentile(boots, 2.5)),
                    float(np.percentile(boots, 97.5))] if len(boots)
        else [float("nan"), float("nan")],
        null_mean=float(null_rs.mean()) if len(null_rs) else float("nan"),
        null_p95=float(np.percentile(null_rs, 95)) if len(null_rs) else float("nan")),
    completeness=dict(steps=a.steps_margin, median_rel_err=float(np.median(comp)),
                      max_rel_err=float(comp.max())),
    flips=st.jsonable(flip_summary.round(4).to_dict("records"))
    if flip_summary is not None else None,
    lens_available=lens_available,
    minutes=round((time.time() - t0) / 60, 1))
json.dump(rep, open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
print(f"\nwrote {a.out}/{tag}_stats.json  ({rep['minutes']} min)", flush=True)
