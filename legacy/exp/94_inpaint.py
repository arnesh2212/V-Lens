# SPDX-License-Identifier: Apache-2.0
r"""An INDEPENDENT ruler: delete the object from the photograph, not from the
residual stream.

THE GAP THIS CLOSES.  Every causal grading in this project so far replaces
intermediate activations at a region's tokens with the per-position corpus mean
:math:`\bar h`.  That is a defensible intervention, but it is *also* the
intervention our reading is mathematically designed to approximate --

.. math::
   \sum_{p \in R} (h_p - \bar h_p)^{\!\top}\!\!\int_0^1\!
   \nabla_{h_p} F(\bar h + \alpha (h - \bar h), q)\,\mathrm{d}\alpha
   \;\longrightarrow\; F(h,q) - F(\bar h, q)

is the conservation law -- so grading against it grades with our own ruler.  The
grey-block pixel check we already have is independent but weak (rho = 0.056),
because a grey rectangle is not a photograph: the vision tower reacts to the
rectangle itself, so the measured effect is mostly "there is a hole here".

Here the object is REMOVED and the hole INPAINTED with LaMa (big-lama, run on
the CPU so it never competes with the VLM for VRAM), giving a plausible natural
photograph with one thing missing.  The image is then re-encoded from scratch --
new pixels, new vision tower activations, new everything.  Nothing about this
intervention lives in the space any lens reads.  That is the entire point: it is
the one ruler in the paper that cannot be accused of being our own.

WHAT IS MEASURED, per image.

  Objective.  A two-alternative forced choice over a full answer sequence,
  ``objective.margin``: ``log P(" concept") - log P(" distractor")`` given
  "Which of these appears in the photograph: {c} or {d}? Answer with one word."
  The distractor is a COCO category absent from the image (``cocodata``'s own
  matched-foil sampler).  Items where the model does not already prefer the
  correct answer are dropped before any lens is read.

  Regions.  ``target`` is the union of the target category's instance masks
  (dilated, so the object's own boundary and contact shadow go with it);
  ``other`` is the largest instance of a *different* category; ``bg1``/``bg2``
  are the target's own mask rigidly translated into annotated-object-free
  background, so they are matched in area and in shape and differ only in
  *where* they are.  Each is inpainted separately.

  Truth.  ``truth_R = F(intact) - F(inpainted away R)`` in nats per token.
  Positive means removing R hurt the answer.  This is the only quantity any
  method is graded against and it never touches an activation.

  Sanity.  An item whose ``truth_target`` is below ``--min-signal`` carries no
  causal signal at all -- the model's answer survives the object's removal --
  and is reported as its own line rather than averaged in silently.

  Chance.  Region top-1 is graded against a MEASURED floor: ``--null-draws``
  random per-position score vectors per item, pooled by the same aggregation,
  because on a ragged region set a summed reading favours the biggest region
  and 1/R is not the floor.  A ``region size`` method is carried in the table
  for the same reason.

  Deletion / insertion.  Both directions use the same pixel operator, so the
  curves are on one ruler: deletion inpaints the top-``f`` cells of a method's
  own ranking; insertion inpaints everything *except* them.  ``f=0`` (deletion)
  and ``f=1`` (insertion) are the intact image and are shared, as is the
  fully-inpainted endpoint.  Low deletion AUC and high insertion AUC are good.

  Flips.  The model's own greedy words, intact vs with the region inpainted out.

Every confidence interval bootstraps over IMAGES, never over regions.

Run ``--n 2 --mean-images 4 --fracs 0.1 --gen-n 1`` first: the tiny end-to-end
smoke test, including a real inpaint and a real re-encode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zlib
import logging
import os
import sys
import time

sys.path.insert(0, "code")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _n in ("httpx", "httpcore", "urllib3", "filelock", "PIL"):
    logging.getLogger(_n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens",
                default="results/lenses/"
                        "Qwen3-VL-2B-Instruct__mm__all__n120.pt")
ap.add_argument("--n", type=int, default=24, help="items kept")
ap.add_argument("--pool-n", type=int, default=400, help="candidate items scanned")
ap.add_argument("--mean-images", type=int, default=60)
ap.add_argument("--layers", type=int, nargs="*", default=None,
                help="read at these layers instead of sampling across depth, "
                     "so they can match a J-lens fitted at only a few")
ap.add_argument("--n-layers", type=int, default=3)
ap.add_argument("--steps-margin", type=int, default=32,
                help="measured-sufficient for the margin objective; 4 is not")
ap.add_argument("--chunk-margin", type=int, default=8)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--min-area", type=float, default=0.03)
ap.add_argument("--max-area", type=float, default=0.35,
                help="an object covering more than this leaves LaMa nothing "
                     "to reconstruct from, so the inpaint stops being plausible")
ap.add_argument("--cell-thresh", type=float, default=0.25,
                help="fraction of a token cell the mask must cover for the "
                     "cell to belong to the region (matches cocodata.box_tokens)")
ap.add_argument("--dilate", type=int, default=9,
                help="pixels the object mask is grown by before inpainting, so "
                     "the silhouette edge and contact shadow go with the object")
ap.add_argument("--inpainter", default="lama", choices=("lama", "telea", "ns"))
ap.add_argument("--lama-device", default="cpu")
ap.add_argument("--n-other", type=int, default=2,
                help="how many OTHER-category objects join the region set")
ap.add_argument("--n-bg", type=int, default=3,
                help="how many shape-matched background translations join it")
ap.add_argument("--min-signal", type=float, default=0.02,
                help="|truth_target| below this = the item carries no signal")
ap.add_argument("--fracs", type=float, nargs="*",
                default=[0.05, 0.15, 0.30, 0.50],
                help="deletion/insertion fractions of the token grid")
ap.add_argument("--gen-n", type=int, default=12, help="items scored for flips")
ap.add_argument("--gen-tokens", type=int, default=6)
ap.add_argument("--null-draws", type=int, default=200,
                help="random score vectors per item for the MEASURED chance floor")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--save-examples", type=int, default=3)
ap.add_argument("--out", default="results/inpaint")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

import cv2
from PIL import Image
from pycocotools import mask as maskutil

import jvlm
from jvlm import bench, cocodata, corpus, interact, objective, readout, vmap
from jvlm import stats as st
from jlens.hooks import ActivationRecorder

# ============================================================== the inpainter
_LAMA = None
if a.inpainter == "lama":
    from simple_lama_inpainting import SimpleLama
    _LAMA = SimpleLama(device=a.lama_device)
INPAINTER_NAME = {"lama": f"LaMa (big-lama) on {a.lama_device}",
                  "telea": "OpenCV Telea",
                  "ns": "OpenCV Navier-Stokes"}[a.inpainter]

_CACHE: dict[bytes, Image.Image] = {}
_N_INPAINT = [0, 0]        # calls, cache hits


def inpaint(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """Remove ``mask`` from ``img`` and fill the hole plausibly.

    The cache is keyed on the mask alone because within one item the picture
    never changes, and deletion curves for different methods agree on their
    first few cells far more often than not.
    """
    if not mask.any():
        return img
    key = hashlib.blake2b(np.ascontiguousarray(mask, dtype=np.uint8).tobytes(),
                          digest_size=16).digest()
    hit = _CACHE.get(key)
    if hit is not None:
        _N_INPAINT[1] += 1
        return hit
    _N_INPAINT[0] += 1
    m8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if _LAMA is not None:
        out = _LAMA(img, Image.fromarray(m8))
        out = out.resize(img.size) if out.size != img.size else out
    else:
        flag = cv2.INPAINT_TELEA if a.inpainter == "telea" else cv2.INPAINT_NS
        bgr = cv2.inpaint(np.array(img)[:, :, ::-1].copy(), m8, 5, flag)
        out = Image.fromarray(bgr[:, :, ::-1])
    _CACHE[key] = out
    return out


# ============================================================== model + fixed
m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
if a.layers:
    LAYERS = sorted(set(a.layers))
else:
    lo, hi = max(m.n_deepstack, 1), m.n_layers - 2
    LAYERS = sorted({int(round(lo + (hi - lo) * f))
                     for f in np.linspace(0.2, 0.85, a.n_layers)})
HEAD_LAYER = LAYERS[len(LAYERS) // 2]
J = None
if a.jlens and os.path.exists(a.jlens):
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))
    if HEAD_LAYER not in J.jacobians:
        print(f"[warn] J-lens has no layer {HEAD_LAYER} "
              f"(has {sorted(J.jacobians)}); the J-lens arm is DROPPED",
              flush=True)
        J = None
print(m, f"\nLAYERS {LAYERS}  HEAD_LAYER {HEAD_LAYER} (fixed in advance)  "
      f"steps_margin={a.steps_margin}  inpainter={INPAINTER_NAME}  "
      f"jlens={'yes' if J else 'no'}", flush=True)

# ---------------------------------------------------------- COCO segmentation
# cocodata keeps boxes, not masks; the masks are what this experiment is for,
# so the annotation file is re-read here rather than a helper being invented
# that the already-banked runs never saw.
_inst = json.load(open(cocodata.INSTANCES))
_CATNAME = {c["id"]: c["name"] for c in _inst["categories"]}
_SIZES = {im["id"]: (im["height"], im["width"]) for im in _inst["images"]}
_ANNS: dict[int, list] = {}
for _an in _inst["annotations"]:
    _ANNS.setdefault(_an["image_id"], []).append(_an)
del _inst
print(f"COCO instances loaded: {len(_ANNS)} annotated images  "
      f"{time.time()-t0:.0f}s", flush=True)


def ann_mask(ann, h, w) -> np.ndarray:
    """One annotation's binary mask at the photograph's native resolution."""
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = maskutil.frPyObjects(seg, h, w)
        rle = maskutil.merge(rles)
    elif isinstance(seg.get("counts"), list):
        rle = maskutil.frPyObjects(seg, h, w)
    else:
        rle = dict(seg)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode()
    return maskutil.decode(rle).astype(bool)


def cat_mask(image_id: int, category: str, *, square: int) -> np.ndarray:
    """Union of a category's non-crowd instances, resized to the square frame."""
    h, w = _SIZES[image_id]
    acc = np.zeros((h, w), bool)
    for an in _ANNS.get(image_id, []):
        if an.get("iscrowd") or _CATNAME[an["category_id"]] != category:
            continue
        acc |= ann_mask(an, h, w)
    if not acc.any():
        return np.zeros((square, square), bool)
    return cv2.resize(acc.astype(np.uint8), (square, square),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def all_object_mask(image_id: int, *, square: int) -> np.ndarray:
    """Every annotated object, for deciding what counts as background."""
    h, w = _SIZES[image_id]
    acc = np.zeros((h, w), bool)
    for an in _ANNS.get(image_id, []):
        try:
            acc |= ann_mask(an, h, w)
        except Exception:                                        # noqa: BLE001
            continue
    return cv2.resize(acc.astype(np.uint8), (square, square),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not mask.any():
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def shifted_background(mask: np.ndarray, occupied: np.ndarray,
                       rng: np.random.Generator, tries: int = 300):
    """The same mask, rigidly translated into empty background.

    Matched in area and in shape to the object's own mask, so the only thing
    that differs from the ``target`` region is *where the pixels are*.  Returns
    ``(mask, overlap_fraction)``; the caller decides whether the overlap is
    small enough to call it background.
    """
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None, 1.0
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    best, best_ov = None, 1.0
    for _ in range(tries):
        dy = int(rng.integers(-y0, H - y1)) if H - y1 > -y0 else 0
        dx = int(rng.integers(-x0, W - x1)) if W - x1 > -x0 else 0
        if abs(dy) + abs(dx) < 8:
            continue
        sh = np.zeros_like(mask)
        sh[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)] = \
            mask[max(0, -dy):H + min(0, -dy), max(0, -dx):W + min(0, -dx)]
        n = int(sh.sum())
        if n < 0.8 * int(mask.sum()):
            continue
        ov = float((sh & occupied).sum()) / max(n, 1)
        if ov < best_ov:
            best, best_ov = sh, ov
            if ov < 0.02:
                break
    return best, best_ov


# ------------------------------------------------------------------ the grid
_probe_items = cocodata.concept_items(1, min_area=a.min_area, seed=3,
                                      tokenizer=m.tokenizer)
b0 = m.chat("Describe this image.",
            [corpus.load_image(_probe_items[0]["path"], square=a.square)])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
if G * G != P:
    raise SystemExit(f"{P} image tokens is not a square grid")
CELL = a.square / G
print(f"{P} image tokens on {G}x{G}; {CELL:.1f}px per cell", flush=True)

# cell -> pixel block, precomputed once
_CELL_BOX = [(int(round(r * CELL)), int(round((r + 1) * CELL)),
              int(round(c * CELL)), int(round((c + 1) * CELL)))
             for r in range(G) for c in range(G)]


def mask_to_cells(mask: np.ndarray, thresh: float) -> list[int]:
    """Which token cells the mask covers, by area fraction of the cell."""
    small = cv2.resize(mask.astype(np.float32), (G, G),
                       interpolation=cv2.INTER_AREA).reshape(-1)
    cells = np.nonzero(small >= thresh)[0].tolist()
    if not cells:                       # a thin object still owns its best cell
        cells = np.nonzero(small >= max(thresh / 4, 1e-3))[0].tolist()
    if not cells:
        cells = [int(small.argmax())]
    return sorted(cells)


def cells_to_mask(cells) -> np.ndarray:
    """The pixel footprint of a set of token cells."""
    mk = np.zeros((a.square, a.square), bool)
    for c in cells:
        r0, r1, c0, c1 = _CELL_BOX[int(c)]
        mk[r0:r1, c0:c1] = True
    return mk


# -------------------------------------------------------------- the baseline
# per-position corpus mean, built inline the way every other experiment here
# does it (there is no corpus.position_means helper, and inventing one that
# disagreed with the banked runs would be worse than repeating six lines)
_acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
_nm = 0
for _p in corpus.coco_images(a.mean_images + 30, seed=11):
    if _nm >= a.mean_images:
        break
    _b = m.chat("Describe this image.", [corpus.load_image(_p, square=a.square)])
    if len(readout.image_positions(m, _b)) != P:
        continue
    _A = readout.residuals(m, _b, LAYERS)
    for l in LAYERS:
        _acc[l] += _A[l][IPOS].float().cpu().double()
    _nm += 1
MEANS = {l: (_acc[l] / max(_nm, 1)).float() for l in LAYERS}
print(f"means from {_nm} held-out photographs  {time.time()-t0:.0f}s", flush=True)


# =============================================================== the readings
def first_order(obj, layer, positions) -> tuple[torch.Tensor, torch.Tensor]:
    r"""``(grad x (h - hbar), ||grad||)`` at :math:`\alpha = 1`.

    The one-point form of the same reading -- attribution patching's estimator
    and, with the baseline discarded, plain gradient saliency.  Both are the
    ablations our path integral is supposed to beat, so both are in the table.

    ``vmap.do_map`` cannot be pointed at a multi-pass objective (it takes the
    gradient at the recorder's activations, which hold only the *last* variant),
    so the objective engine's own write-in pass is used, with a single
    interpolant pinned at the real activation instead of a path.
    """
    pos = torch.as_tensor(positions, dtype=torch.long)
    with torch.no_grad(), ActivationRecorder(m.layers, at=[layer]) as rec:
        m.forward(obj.variants[0].batch)
        h = rec.activations[layer][0, pos.to(rec.activations[layer].device)] \
            .detach().float()
    dev = h - MEANS[layer].to(h.device).float()
    x = h.clone()[None].detach().requires_grad_(True)          # [1, P, d]
    gxa = torch.zeros(obj.k, len(pos), dtype=torch.float64)
    gnm = torch.zeros(obj.k, len(pos), dtype=torch.float64)

    def sink(j, g, _d=dev):
        gg = g.float().sum(0)
        gxa[j] += (gg * _d).sum(-1).double().cpu()
        gnm[j] += gg.norm(dim=-1).double().cpu()

    objective._pass(m, obj, layer, pos, x, cols=range(obj.k), sink=sink,
                    want_vals=False)
    return gxa.float(), gnm.float()


def _trapz(y, x):
    """numpy 2 renamed trapz; the rule is the same either way."""
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(np.asarray(y, float), np.asarray(x, float))


def clean_logits(batch) -> torch.Tensor:
    """Answer-position logits, one forward."""
    h = readout.residuals(m, batch, [m.n_layers - 1])[m.n_layers - 1]
    with torch.no_grad():
        return m.unembed(h[batch["input_ids"].shape[1] - 1][None])[0].float().cpu()


def best_form(lg: torch.Tensor, word: str) -> str:
    """The surface form of ``word`` the model is most ready to emit here.

    After a chat template's assistant header ``"cat"`` and ``" cat"`` are
    different first tokens and only one of them is what the model would
    actually produce; scoring the wrong one makes the margin a measurement of
    the tokenizer.  The form is chosen ONCE, on the intact photograph, and then
    held fixed for every inpainted variant -- re-choosing it per intervention
    would be selecting on the outcome.
    """
    cands = [word, word.capitalize(), " " + word, " " + word.capitalize()]
    scored = []
    for f in cands:
        ids = m.tokenizer.encode(f, add_special_tokens=False)
        if ids:
            scored.append((float(lg[ids[0]]), f))
    return max(scored)[1] if scored else word


def prep(img, question):
    return m.chat(question + " Answer with one word.", [img])


def question_for(c, d):
    return f"Which of these appears in the photograph: {c} or {d}?"


WRONG_Q = "How many objects are in this photograph?"

METHODS = ["VJ-Lens", "VJ-Lens (wrong prompt)", "logit lens", "J-lens",
           "gradient x activation", "gradient norm", "activation deviation",
           "region size", "random"]
#: the curves cost a re-encode per point, so the two pure ranking controls
#: ("region size" is a constant map, "activation deviation" is image-only)
#: are left out of them; "random" stays, because a curve needs its floor.
CURVE_METHODS = ["VJ-Lens", "logit lens", "J-lens", "gradient x activation",
                 "gradient norm", "random"]

# ============================================================== the main loop
rows: list[dict] = []          # one per (item, region)
curve_rows: list[dict] = []    # one per (item, method, direction, fraction)
flip_rows: list[dict] = []
item_rows: list[dict] = []
tasks: list[bench.Task] = []
per_task: dict[str, list] = {k: [] for k in METHODS}
null_hits: dict[str, list] = {}

rng = np.random.default_rng(a.seed)
items = cocodata.concept_items(a.pool_n, min_area=a.min_area, seed=7,
                               tokenizer=m.tokenizer, hard=True)
print(f"{len(items)} candidate items", flush=True)

n_seen = n_area = n_pref = n_mask = 0
pq = f"{a.out}/{tag}_regions.parquet"
cq = f"{a.out}/{tag}_curves.parquet"
fq = f"{a.out}/{tag}_flips.parquet"


def checkpoint():
    # the per-position vectors are what the aggregation sweep needs and they
    # are nowhere else on disk: 196 floats x 9 methods x n items is nothing,
    # and it makes a run that is cut short by the GPU clock still reportable.
    if per_task["VJ-Lens"]:
        np.savez_compressed(
            f"{a.out}/{tag}_scores.npz",
            regions=np.array([json.dumps([list(map(int, r)) for r in t.regions])
                              for t in tasks], dtype=object),
            truth=np.array([json.dumps(list(map(float, t.truth)))
                            for t in tasks], dtype=object),
            **{f"m::{k}": np.stack([np.asarray(v, float) for v in per_task[k]])
               for k in METHODS})
    if rows:
        pd.DataFrame(rows).to_parquet(pq)
    if curve_rows:
        pd.DataFrame(curve_rows).to_parquet(cq)
    if flip_rows:
        pd.DataFrame(flip_rows).to_parquet(fq)
    if item_rows:
        json.dump(st.jsonable(item_rows), open(f"{a.out}/{tag}_items.json", "w"),
                  indent=2, default=float)


for it in items:
    if len(item_rows) >= a.n:
        break
    n_seen += 1
    iid, c, d = it["image_id"], it["concept"], it["distractor"]
    if iid not in _SIZES or iid not in _ANNS:
        continue
    _CACHE.clear()

    # ------------------------------------------------------------ the masks
    try:
        m_target_raw = cat_mask(iid, c, square=a.square)
    except Exception as e:                                       # noqa: BLE001
        print(f"  [skip {iid}] mask decode failed: {type(e).__name__}", flush=True)
        continue
    frac = float(m_target_raw.mean())
    if not (a.min_area <= frac <= a.max_area):
        continue
    n_area += 1
    m_target = dilate(m_target_raw, a.dilate)

    img = corpus.load_image(it["path"], square=a.square)
    b = prep(img, question_for(c, d))
    if len(readout.image_positions(m, b)) != P:
        continue

    # the model must already prefer the right answer, or there is nothing for
    # removing the object to break
    LG = clean_logits(b)
    PAIR = (best_form(LG, c), best_form(LG, d))
    obj = objective.margin(m, b, [PAIR])
    F1 = float(objective.value(m, obj)[0])
    if F1 <= 0:
        continue
    n_pref += 1

    # -------------------------------------------------------- the regions
    occupied = all_object_mask(iid, square=a.square)
    others = {}
    for an in _ANNS.get(iid, []):
        nm = _CATNAME[an["category_id"]]
        if nm == c or an.get("iscrowd"):
            continue
        others[nm] = others.get(nm, 0.0) + float(an.get("area", 0.0))
    region_masks: list[tuple[str, np.ndarray]] = [("target", m_target)]
    for nm, _ in sorted(others.items(), key=lambda kv: -kv[1])[:6]:
        if len(region_masks) > a.n_other:
            break
        try:
            mk = dilate(cat_mask(iid, nm, square=a.square), a.dilate)
        except Exception:                                        # noqa: BLE001
            continue
        if not mk.any() or mk.mean() > a.max_area:
            continue
        if float((mk & m_target).sum()) / max(int(mk.sum()), 1) > 0.25:
            continue
        region_masks.append((f"other:{nm}", mk))
    irng = np.random.default_rng(a.seed * 7919 + iid)
    for j in range(a.n_bg):
        sh, ov = shifted_background(m_target, occupied | m_target, irng)
        if sh is None or ov > 0.15:
            continue
        region_masks.append((f"bg{j+1}", sh))
        occupied = occupied | sh
    if len(region_masks) < 3:
        continue
    n_mask += 1

    region_cells = [mask_to_cells(mk, a.cell_thresh) for _, mk in region_masks]
    names = [nm for nm, _ in region_masks]

    # ------------------------------------------------- the PIXEL ground truth
    truth, gen_after = [], []
    for (nm, mk) in region_masks:
        im2 = inpaint(img, mk)
        b2 = prep(im2, question_for(c, d))
        if len(readout.image_positions(m, b2)) != P:
            truth.append(float("nan"))
            continue
        obj2 = objective.margin(m, b2, [PAIR])
        truth.append(F1 - float(objective.value(m, obj2)[0]))
    truth = np.asarray(truth, float)
    if not np.isfinite(truth).all():
        continue
    t_target = float(truth[0])

    # ------------------------------------------------------------ the lenses
    bw = prep(img, WRONG_Q)
    objw = objective.margin(m, bw, [PAIR])
    vj = objective.ig_map(m, obj, [HEAD_LAYER], IPOS, MEANS,
                          steps=a.steps_margin, chunk=a.chunk_margin)[HEAD_LAYER][0]
    vjw = objective.ig_map(m, objw, [HEAD_LAYER], IPOS, MEANS,
                           steps=a.steps_margin, chunk=a.chunk_margin)[HEAD_LAYER][0]
    gxa, gnm = first_order(obj, HEAD_LAYER, IPOS)

    A = readout.residuals(m, b, [HEAD_LAYER])
    h = A[HEAD_LAYER][IPOS].float().cpu()
    ids = cocodata.concept_token_ids(m.tokenizer, c) or \
        [cocodata.primary_token_id(m.tokenizer, c)]
    fl = interact.fixed_lens_patches(
        m, h, ids, jacobian=(J.jacobians.get(HEAD_LAYER) if J else None))

    per = {
        "VJ-Lens": vj.numpy(),
        "VJ-Lens (wrong prompt)": vjw.numpy(),
        "gradient x activation": gxa[0].numpy(),
        "gradient norm": gnm[0].numpy(),
        "logit lens": fl["logit lens"].max(0).values.numpy(),
        "activation deviation": (h - MEANS[HEAD_LAYER]).norm(dim=-1).numpy(),
        "region size": np.ones(P, dtype=float),
        "random": rng.normal(size=P),
    }
    if "J-lens" in fl:
        per["J-lens"] = fl["J-lens"].max(0).values.numpy()

    # completeness of the reading against its OWN (activation-mean) law, kept
    # only as a health check on steps -- it is not the ruler being reported
    comp = objective.completeness(m, obj, HEAD_LAYER, IPOS,
                                  MEANS[HEAD_LAYER], vj[None])

    # Regions are BUILT target-first, and numpy's argmax returns the first
    # maximum, so any method whose map is constant over the frame -- which is
    # exactly what the "region size" size-control is at the mean/max/rms
    # aggregations -- would score a free top-1 hit on every item.  Shuffling
    # the order (regions and their measured truth permuted together) removes
    # the tie-break and changes no non-degenerate method's answer.
    _perm = np.random.default_rng(a.seed * 104729 + iid).permutation(
        len(region_cells))
    region_cells = [region_cells[j] for j in _perm]
    names = [names[j] for j in _perm]
    region_masks = [region_masks[j] for j in _perm]
    truth = truth[_perm]
    task = bench.Task(regions=region_cells, truth=truth, kind="rank",
                      image_id=str(iid))
    tasks.append(task)
    for k in METHODS:
        per_task[k].append(per.get(k, np.full(P, np.nan)))
    for j, nm in enumerate(names):
        rows.append(dict(item=len(item_rows), image_id=iid, concept=c,
                         distractor=d, region=nm, n_cells=len(region_cells[j]),
                         mask_frac=float(region_masks[j][1].mean()),
                         truth=float(truth[j]),
                         **{f"score::{k}": float(np.sum(
                             per.get(k, np.full(P, np.nan))[region_cells[j]]))
                            for k in METHODS}))

    # ------------------------------------------- deletion / insertion curves
    n_cells = [max(1, int(round(f * P))) for f in a.fracs]
    full_mask = np.ones((a.square, a.square), bool)
    try:
        b_none = prep(inpaint(img, full_mask), question_for(c, d))
        F_none = float(objective.value(m, objective.margin(m, b_none, [PAIR]))[0])
    except Exception as e:                                       # noqa: BLE001
        print(f"  [warn] full-frame inpaint failed on {iid}: "
              f"{type(e).__name__}; item dropped", flush=True)
        rows[:] = [r for r in rows if r["item"] != len(item_rows)]
        tasks.pop()
        for k in METHODS:
            per_task[k].pop()
        continue
    for k in CURVE_METHODS:
        v = per.get(k)
        if v is None or not np.isfinite(v).all():
            continue
        order = np.argsort(-v)
        for direction in ("deletion", "insertion"):
            xs, ys = [0.0], [F1 if direction == "deletion" else F_none]
            for f, nk in zip(a.fracs, n_cells):
                top = order[:nk]
                mk = cells_to_mask(top)
                if direction == "insertion":
                    mk = ~mk
                b2 = prep(inpaint(img, mk), question_for(c, d))
                Fv = float(objective.value(
                    m, objective.margin(m, b2, [PAIR]))[0])
                xs.append(float(f))
                ys.append(Fv)
                curve_rows.append(dict(item=len(item_rows), image_id=iid,
                                       method=k, direction=direction,
                                       frac=float(f), F=Fv, F_intact=F1,
                                       F_none=F_none))
            # The AUC covers only the fractions actually measured.  Carrying
            # the curve out to f = 1 would make three quarters of the area a
            # straight line drawn between two points nobody evaluated, and the
            # ranking would then be decided by that interpolation rather than
            # by the method.
            den = (F1 - F_none) if abs(F1 - F_none) > 1e-6 else 1.0
            yn = [(y - F_none) / den for y in ys]
            curve_rows.append(dict(item=len(item_rows), image_id=iid, method=k,
                                   direction=direction, frac=-1.0,
                                   F=float(_trapz(yn, xs) / max(xs[-1], 1e-9)), F_intact=F1,
                                   F_none=F_none))

    # ----------------------------------------------------- generated answers
    if len(item_rows) < a.gen_n:
        try:
            g0 = m.tokenizer.decode(objective.generated_answer(
                m, b, max_new_tokens=a.gen_tokens)).strip().lower()
            for j, (nm, mk) in enumerate(region_masks):
                b2 = prep(inpaint(img, mk), question_for(c, d))
                g1 = m.tokenizer.decode(objective.generated_answer(
                    m, b2, max_new_tokens=a.gen_tokens)).strip().lower()
                flip_rows.append(dict(item=len(item_rows), image_id=iid,
                                      region=nm, concept=c, distractor=d,
                                      gen0=g0, gen1=g1, flip=int(g0 != g1),
                                      said_c0=int(c in g0), said_c1=int(c in g1),
                                      truth=float(truth[j])))
        except Exception as e:                                   # noqa: BLE001
            print(f"  [warn] flips failed on {iid}: {type(e).__name__}", flush=True)

    if len(item_rows) < a.save_examples:
        os.makedirs(f"{a.out}/examples", exist_ok=True)
        img.save(f"{a.out}/examples/{tag}_{iid}_intact.png")
        inpaint(img, m_target).save(f"{a.out}/examples/{tag}_{iid}_{c}_removed.png")

    item_rows.append(dict(
        item=len(item_rows), image_id=iid, concept=c, distractor=d,
        area=frac, F_intact=F1, F_none=F_none, truth_target=t_target,
        truth_max_other=float(np.delete(truth, names.index("target")).max()),
        n_regions=len(names),
        regions=names, no_signal=int(abs(t_target) < a.min_signal),
        comp_rel_err=float(comp["rel_error"]),
        comp_truth=float(comp["truth"][0]), comp_claim=float(comp["claim"][0]),
        comp_F_mean_image=float(comp["F0"][0])))

    el = time.time() - t0
    print(f"  {len(item_rows)}/{a.n} {c}/{d} R={len(names)}  "
          f"truth_target={t_target:+.3f} best_other="
          f"{np.delete(truth, names.index('target')).max():+.3f}  "
          f"F={F1:+.2f}->{F_none:+.2f}  inpaints {_N_INPAINT[0]}"
          f"(+{_N_INPAINT[1]}c)  {el:.0f}s  eta "
          f"{el/max(len(item_rows),1)*(a.n-len(item_rows))/60:.1f}m", flush=True)
    if len(item_rows) % 2 == 0 or len(item_rows) == a.n:
        checkpoint()

checkpoint()
print(f"\nkept {len(item_rows)} items from {n_seen} scanned "
      f"(area ok {n_area}, model prefers correct {n_pref}, masks ok {n_mask})",
      flush=True)
if not item_rows:
    json.dump({"repo": a.repo, "kept": 0}, open(f"{a.out}/{tag}_stats.json", "w"))
    raise SystemExit("NOTHING KEPT -- nothing to summarise")

# ==================================================================== report
R = pd.DataFrame(rows)
IT = pd.DataFrame(item_rows)
CV = pd.DataFrame(curve_rows) if curve_rows else pd.DataFrame()
FL = pd.DataFrame(flip_rows) if flip_rows else pd.DataFrame()
brng = np.random.default_rng(a.seed + 1)
OUT: dict = {"repo": a.repo, "tag": tag, "inpainter": INPAINTER_NAME,
             "head_layer": HEAD_LAYER, "layers": LAYERS, "P": P, "grid": G,
             "steps_margin": a.steps_margin, "n_items": len(IT),
             "n_scanned": n_seen, "jlens": bool(J)}


def boot_over_images(values_by_item: dict, fn, n=a.boot, seed=0):
    """Bootstrap a statistic, resampling IMAGES with replacement."""
    keys = list(values_by_item)
    if not keys:
        return dict(stat=float("nan"), lo=float("nan"), hi=float("nan"))
    g = np.random.default_rng(seed)
    real = fn([values_by_item[k] for k in keys])
    bs = []
    for _ in range(n):
        pick = g.integers(0, len(keys), len(keys))
        v = fn([values_by_item[keys[i]] for i in pick])
        if v == v:
            bs.append(v)
    if not bs:
        return dict(stat=real, lo=float("nan"), hi=float("nan"))
    return dict(stat=float(real), lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)), n_boot=len(bs))


# ------------------------------------------ (0) did the inpainting do anything
print("\n" + "=" * 90)
print("(0) SANITY -- did removing the object actually move the answer?")
print("=" * 90)
sig = IT[IT.no_signal == 0]
print(f"inpainter: {INPAINTER_NAME};  {len(IT)} items, "
      f"{int(IT.no_signal.sum())} with |truth_target| < {a.min_signal} "
      f"(NO causal signal -- the answer survives the object's removal)")
tt = IT.truth_target.to_numpy()
bt = boot_over_images({i: v for i, v in enumerate(tt)},
                      lambda v: float(np.mean(v)), seed=1)
print(f"truth_target (nats/token lost by removing the object): mean "
      f"{bt['stat']:+.4f} [{bt['lo']:+.4f}, {bt['hi']:+.4f}]  "
      f"median {np.median(tt):+.4f}  positive in "
      f"{float((tt > 0).mean()):.3f} of items")
other = IT.truth_max_other.to_numpy()
bo = boot_over_images({i: v for i, v in enumerate(other)},
                      lambda v: float(np.mean(v)), seed=2)
print(f"best NON-target region, same ruler:                    mean "
      f"{bo['stat']:+.4f} [{bo['lo']:+.4f}, {bo['hi']:+.4f}]")
bd = boot_over_images({i: v for i, v in enumerate(tt - other)},
                      lambda v: float(np.mean(v)), seed=3)
print(f"target minus best other:                               mean "
      f"{bd['stat']:+.4f} [{bd['lo']:+.4f}, {bd['hi']:+.4f}]  "
      f"target is strictly largest in {float((tt > other).mean()):.3f} of items")
print(f"F(intact) mean {IT.F_intact.mean():+.3f}, "
      f"F(everything inpainted) mean {IT.F_none.mean():+.3f}; "
      f"F(mean-activation image) mean {IT.comp_F_mean_image.mean():+.3f}")
print(f"VJ-Lens completeness against its OWN activation-mean law (a health "
      f"check on --steps-margin, NOT the ruler being reported): truth "
      f"{IT.comp_truth.mean():+.4f}, claim {IT.comp_claim.mean():+.4f}, "
      f"median |gap| {np.median((IT.comp_claim - IT.comp_truth).abs()):.4f}, "
      f"median rel-err {IT.comp_rel_err.median():.3f}")
OUT["sanity"] = dict(truth_target=bt, truth_max_other=bo, target_minus_other=bd,
                     n_no_signal=int(IT.no_signal.sum()),
                     frac_target_largest=float((tt > other).mean()),
                     F_intact=float(IT.F_intact.mean()),
                     F_none=float(IT.F_none.mean()),
                     comp_truth=float(IT.comp_truth.mean()),
                     comp_claim=float(IT.comp_claim.mean()),
                     comp_rel_err_median=float(IT.comp_rel_err.median()))

RT = (R.assign(kind=R.region.str.replace(r"^bg\d+$", "bg", regex=True)
                       .str.split(":").str[0])
       .groupby("kind")
       .agg(n=("truth", "size"), truth=("truth", "mean"),
            truth_median=("truth", "median"), n_cells=("n_cells", "mean"),
            mask_frac=("mask_frac", "mean")).reset_index()
       .sort_values("truth", ascending=False))
print("\nwhat the PIXEL ruler says each kind of region is worth:")
print(RT.round(4).to_string(index=False))
OUT["truth_by_region_kind"] = st.jsonable(RT.to_dict(orient="records"))

# ------------------------------------------------- (1) the measured chance floor
print("\n" + "=" * 90)
print("(1) MEASURED CHANCE -- a random scorer on these very regions")
print("=" * 90)
for agg in bench.AGGREGATIONS:
    hits = []
    for t in tasks:
        arg = int(np.argmax(t.truth))
        for _ in range(a.null_draws):
            v = brng.normal(size=P)
            hits.append(int(np.argmax(bench.pool(v, t, agg)) == arg))
    null_hits[agg] = hits
    print(f"  {agg:>5}: top-1 = {np.mean(hits):.4f} "
          f"(1/R would be {np.mean([t.chance_top1() for t in tasks]):.4f}, "
          f"{a.null_draws} draws x {len(tasks)} items)")
OUT["measured_chance"] = {k: float(np.mean(v)) for k, v in null_hits.items()}
OUT["nominal_chance"] = float(np.mean([t.chance_top1() for t in tasks]))

# ------------------------------------------------------- (2) the lens ranking
print("\n" + "=" * 90)
print("(2) EVERY LENS vs THE PIXEL RULER -- each at its own best aggregation")
print("=" * 90)
have = {k: v for k, v in per_task.items()
        if all(np.isfinite(np.asarray(x)).all() for x in v)}
res = bench.sweep(have, tasks)
best = bench.best_per_method(res, by="top1")
table = []
for k, r in sorted(best.items(), key=lambda kv: -kv[1].top1):
    ch = float(np.mean(null_hits[r.aggregation]))
    # pooled correlation across every (item, region) pair, at that aggregation
    pooled = np.concatenate([bench.pool(np.asarray(v, float), t, r.aggregation)
                             for v, t in zip(have[k], tasks)])
    tru = np.concatenate([t.truth for t in tasks])
    pear = float(np.corrcoef(pooled, tru)[0, 1]) if pooled.std() > 0 else float("nan")
    by_item = {i: (bench.pool(np.asarray(v, float), t, r.aggregation), t.truth)
               for i, (v, t) in enumerate(zip(have[k], tasks))}

    def _rho(vs):
        A_ = np.concatenate([x[0] for x in vs])
        B_ = np.concatenate([x[1] for x in vs])
        if A_.std() < 1e-12 or B_.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(np.argsort(np.argsort(A_)),
                                 np.argsort(np.argsort(B_)))[0, 1])

    bb = boot_over_images(by_item, _rho, seed=zlib.crc32(k.encode()) % 9999)
    table.append(dict(method=k, agg=r.aggregation, top1=r.top1,
                      chance=ch, lift=r.top1 / ch if ch else float("nan"),
                      within_rho=r.spearman, pooled_r=pear,
                      pooled_rho=bb["stat"], rho_lo=bb["lo"], rho_hi=bb["hi"],
                      n=r.n))
TB = pd.DataFrame(table)
print(TB.round(4).to_string(index=False))
print("\n  top1        = fraction of images whose most-damaging region the "
      "method ranks first\n  chance      = MEASURED floor for that aggregation "
      "(random scorer, same regions)\n  within_rho  = mean within-image Spearman "
      "(score vs measured margin loss)\n  pooled_rho  = Spearman over all "
      "(image, region) pairs, 95% CI bootstrapped over IMAGES")
OUT["ranking"] = st.jsonable(TB.to_dict(orient="records"))
print("\nfull aggregation grid (top-1):")
print(bench.table(res))
OUT["grid"] = st.jsonable([r.__dict__ | {"by_image": None} for r in res])

# --------------------------------------------- (3) deletion / insertion AUCs
if not CV.empty:
    print("\n" + "=" * 90)
    print("(3) DELETION / INSERTION AUC -- same pixel operator in both "
          "directions")
    print("=" * 90)
    auc = CV[CV.frac < 0]
    lines = []
    for k, g in auc.groupby("method"):
        row = {"method": k}
        for direction in ("deletion", "insertion"):
            sub = g[g.direction == direction]
            if sub.empty:
                continue
            per_item = sub.groupby("item")["F"].mean().to_dict()
            bb = boot_over_images(per_item, lambda v: float(np.mean(v)),
                                  seed=zlib.crc32((k + direction).encode()) % 9999)
            row[direction] = bb["stat"]
            row[f"{direction}_lo"] = bb["lo"]
            row[f"{direction}_hi"] = bb["hi"]
        if "deletion" in row and "insertion" in row:
            row["ins_minus_del"] = row["insertion"] - row["deletion"]
        lines.append(row)
    AU = pd.DataFrame(lines).sort_values("ins_minus_del", ascending=False)
    print(AU.round(4).to_string(index=False))
    print("\n  curves are normalised so 1.0 = the intact answer and 0.0 = the "
          "fully-inpainted one.\n  LOW deletion AUC and HIGH insertion AUC are "
          "good; the last column is the summary.")
    OUT["auc"] = st.jsonable(AU.to_dict(orient="records"))

# ------------------------------------------------------------------ (4) flips
if not FL.empty:
    print("\n" + "=" * 90)
    print("(4) GENERATED-ANSWER FLIPS -- the model's own greedy words")
    print("=" * 90)
    fs = FL.groupby("region").agg(n=("flip", "size"), flip=("flip", "mean"),
                                  said_c_before=("said_c0", "mean"),
                                  said_c_after=("said_c1", "mean"),
                                  truth=("truth", "mean")).reset_index()
    print(fs.round(4).to_string(index=False))
    tgt = FL[FL.region == "target"]
    if len(tgt):
        bf = boot_over_images({i: v for i, v in enumerate(tgt.flip.to_numpy())},
                              lambda v: float(np.mean(v)), seed=5)
        print(f"\nremoving the TARGET flips the generated answer in "
              f"{bf['stat']:.3f} [{bf['lo']:.3f}, {bf['hi']:.3f}] of items; "
              f"the answer still names the concept in "
              f"{tgt.said_c1.mean():.3f} (was {tgt.said_c0.mean():.3f})")
    OUT["flips"] = st.jsonable(fs.to_dict(orient="records"))

# ------------------------------------------------- (5) signal-carrying subset
if len(sig) and len(sig) < len(IT):
    keep = set(sig.item.tolist())
    sub_tasks = [t for i, t in enumerate(tasks) if i in keep]
    sub = {k: [v for i, v in enumerate(have[k]) if i in keep] for k in have}
    print("\n" + "=" * 90)
    print(f"(5) THE SAME TABLE ON THE {len(sub_tasks)} ITEMS THAT CARRY SIGNAL "
          f"(|truth_target| >= {a.min_signal})")
    print("=" * 90)
    b2 = bench.best_per_method(bench.sweep(sub, sub_tasks), by="top1")
    print(pd.DataFrame([dict(method=k, agg=r.aggregation, top1=r.top1,
                             within_rho=r.spearman, n=r.n)
                        for k, r in sorted(b2.items(), key=lambda kv: -kv[1].top1)]
                       ).round(4).to_string(index=False))
    OUT["ranking_signal_only"] = st.jsonable(
        [dict(method=k, agg=r.aggregation, top1=r.top1, rho=r.spearman, n=r.n)
         for k, r in b2.items()])

json.dump(st.jsonable(OUT), open(f"{a.out}/{tag}_stats.json", "w"), indent=2,
          default=float)
print(f"\nwrote {a.out}/{tag}_stats.json  ({time.time()-t0:.0f}s total, "
      f"{_N_INPAINT[0]} real inpaints, {_N_INPAINT[1]} cache hits)", flush=True)
