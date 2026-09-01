# SPDX-License-Identifier: Apache-2.0
"""CLEVR val: scenes, object extents, and two-step relational questions.

CLEVR is the right source for watching a pointer resolve.  Every scene carries
each object's colour, shape, size, material and pixel position, plus the four
spatial relations between every pair, so a question can be *constructed* with a
guaranteed unique answer rather than hoped to be unambiguous.  What the release
does not carry is a mask or a box, and a covering intervention needs one.

Boxes come from a background plate.  Every val render shares one camera and one
floor, so the per-pixel median over a hundred renders is the empty room; a
render minus that plate is the objects.  Connected components of the difference
are split between the object centres they contain, which the release does give,
and the result is a tight box per object -- verified by eye on a contact sheet
before anything was measured with it.

The download is a range-read of the official 19 GB zip: the central directory
sits at the end of the archive, so ``images/val`` and the val scene graph can be
pulled out without the other 17 GB ever touching disk.  See ``fetch()``.
"""
from __future__ import annotations

import glob
import json
import os
from functools import lru_cache

import numpy as np
from PIL import Image

ROOT = "/workspace/vlm/data/clevr"
IMG_DIR = f"{ROOT}/images/val"
SCENES = f"{ROOT}/scenes/CLEVR_val_scenes.json"
PLATE = f"{ROOT}/bg.npy"

COLOURS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
SHAPES = ["cube", "sphere", "cylinder"]
#: the word the question should use for each shape -- "sphere" is what CLEVR
#: calls it and "ball" is what a person calls it; the model is asked in the
#: plainer word, which is also the one the target figure's title uses
SAY = {"cube": "cube", "sphere": "ball", "cylinder": "cylinder"}
RAD = {"large": 0.7, "small": 0.35}
#: pixel radius ~= KPIX * world_radius / depth, calibrated on isolated objects
KPIX = 560.0


# --------------------------------------------------------------------- loading

@lru_cache(maxsize=1)
def scenes() -> list[dict]:
    return json.load(open(SCENES))["scenes"]


@lru_cache(maxsize=1)
def plate() -> np.ndarray:
    """The empty room: per-pixel median over the val renders on disk."""
    if os.path.exists(PLATE):
        return np.load(PLATE)
    ps = sorted(glob.glob(f"{IMG_DIR}/*.png"))[:120]
    a = np.stack([np.asarray(Image.open(p).convert("RGB"), np.float32)
                  for p in ps])
    bg = np.median(a, 0)
    np.save(PLATE, bg)
    return bg


def image_path(scene: dict) -> str:
    return f"{IMG_DIR}/{scene['image_filename']}"


def have(scene: dict) -> bool:
    return os.path.exists(image_path(scene))


# ----------------------------------------------------------------------- boxes

def object_boxes(scene: dict, thr: float = 25.0, minpix: int = 45,
                 trim: float = 1.5) -> dict:
    """``{index: {box, npix, clean}}`` in fractions of the frame, or ``None``.

    ``clean`` is false when the extracted extent is far larger than the size the
    object's radius and depth predict, which is what a merge with a neighbour or
    with a shadow looks like.  Selection only ever uses clean objects.
    """
    from scipy import ndimage

    im = np.asarray(Image.open(image_path(scene)).convert("RGB"), np.float32)
    H, W, _ = im.shape
    m = np.abs(im - plate()).max(-1) > thr
    m = ndimage.binary_opening(m, np.ones((3, 3)))
    lab, n = ndimage.label(m)
    if n:
        sizes = np.concatenate([[0.0], ndimage.sum(m, lab, range(1, n + 1))])
        lab[sizes[lab] < minpix] = 0
    cen = np.array([o["pixel_coords"][:2] for o in scene["objects"]], float)
    ci = np.clip(cen.astype(int), [0, 0], [W - 1, H - 1])
    comp = lab[ci[:, 1], ci[:, 0]]
    ys, xs = np.nonzero(lab > 0)
    L = lab[ys, xs]
    out: dict[int, dict | None] = {}
    for j in range(len(cen)):
        c = comp[j]
        if c == 0:
            out[j] = None
            continue
        here = list(np.nonzero(comp == c)[0])
        sel = L == c
        px, py = xs[sel], ys[sel]
        if len(here) > 1:
            d2 = ((px[:, None] - cen[here, 0][None]) ** 2
                  + (py[:, None] - cen[here, 1][None]) ** 2)
            mine = d2.argmin(1) == here.index(j)
            px, py = px[mine], py[mine]
        if len(px) < minpix:
            out[j] = None
            continue
        x0, x1 = np.percentile(px, [trim, 100 - trim])
        y0, y1 = np.percentile(py, [trim, 100 - trim])
        o = scene["objects"][j]
        rp = KPIX * RAD[o["size"]] / o["pixel_coords"][2]
        out[j] = dict(box=(x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H),
                      npix=int(len(px)),
                      clean=bool((x1 - x0) < 2.6 * rp and (y1 - y0) < 3.0 * rp))
    return out


def box_tokens(box, grid: int, thresh: float = 0.2) -> list[int]:
    """Cells of a ``grid x grid`` token layout the box covers by ``thresh``."""
    x0, y0, w, h = box
    x1, y1 = x0 + w, y0 + h
    out = []
    for r in range(grid):
        for c in range(grid):
            cx0, cx1 = c / grid, (c + 1) / grid
            cy0, cy1 = r / grid, (r + 1) / grid
            ov = (max(0.0, min(x1, cx1) - max(x0, cx0))
                  * max(0.0, min(y1, cy1) - max(y0, cy0)))
            if ov * grid * grid >= thresh:
                out.append(r * grid + c)
    return out


# ------------------------------------------------------------------- questions

def describe(o: dict, colour: bool = False) -> str:
    """A noun phrase for one object.

    The landmark is named *without* its colour, so that the control question --
    which asks for that colour -- cannot be answered from the question text.
    """
    bits = [o["size"], o["material"] if o["material"] != "rubber" else "matte"]
    if colour:
        bits.append(o["color"])
    return "the " + " ".join(bits + [SAY[o["shape"]]])


def _unique(objs, keys, j) -> bool:
    k = tuple(objs[j][x] for x in keys)
    return sum(tuple(o[x] for x in keys) == k for o in objs) == 1


def near_items(grid: int, n_obj=(4, 6), min_same: int = 3,
               margin: float = 0.70, balance: float = 4.0,
               limit: int | None = None, skip: int = 0):
    """The same construction with a relation no heuristic can shortcut.

    "The ball to the left of the cylinder" has a hole in it.  Where exactly one
    ball stands left of the landmark, that ball is necessarily the leftmost
    ball, so "say the colour of the leftmost ball" answers every such question
    without ever finding the landmark -- and the first run showed the model
    doing precisely that, at zero causal cost for covering the landmark.  Any
    half-plane relation has this hole: uniqueness forces the referent to the
    extreme of its own shape along the relation's axis.

    "The ball nearest the cylinder" has no such hole.  Nearest to *what* cannot
    be evaluated without the landmark, and the referent is required here to be
    neither the leftmost nor the rightmost ball, so no single-axis heuristic
    reaches it.  Three objects share the referent's shape, and the one furthest
    from the landmark serves as the distractor -- same shape, same question,
    wrong object.
    """
    kept = 0
    for si, s in enumerate(scenes()):
        if limit is not None and kept >= limit:
            return
        if si < skip:
            continue
        objs = s["objects"]
        if not (n_obj[0] <= len(objs) <= n_obj[1]) or not have(s):
            continue
        bx = object_boxes(s)
        if any(bx[j] is None or not bx[j]["clean"] for j in range(len(objs))):
            continue
        toks = {j: set(box_tokens(bx[j]["box"], grid)) for j in range(len(objs))}
        if any(not toks[j] for j in toks):
            continue
        if any(toks[i] & toks[j] for i in toks for j in toks if i < j):
            continue
        px = np.array([o["pixel_coords"][:2] for o in objs], float)
        w3 = np.array([o["3d_coords"] for o in objs], float)
        hit = None
        for li in range(len(objs)):
            if not _unique(objs, ("size", "material", "shape"), li):
                continue
            for sh in SHAPES:
                if sh == objs[li]["shape"]:
                    continue
                same = [j for j in range(len(objs))
                        if objs[j]["shape"] == sh and j != li]
                if len(same) < min_same:
                    continue
                d2 = np.array([np.hypot(*(px[j] - px[li])) for j in same])
                d3 = np.array([np.linalg.norm(w3[j] - w3[li]) for j in same])
                if d2.argmin() != d3.argmin():
                    continue           # "nearest" must not depend on the metric
                o2, o3 = np.sort(d2), np.sort(d3)
                if o2[0] > margin * o2[1] or o3[0] > margin * o3[1]:
                    continue           # and it must be nearest by a clear margin
                ri = same[int(d2.argmin())]
                # the referent must not be the extreme of its own shape in x,
                # or "say the leftmost one" would reach it without the landmark
                xs = px[same, 0]
                if not (xs.min() < px[ri, 0] < xs.max()):
                    continue
                if objs[ri]["color"] == objs[li]["color"]:
                    continue
                nl, nr = len(toks[li]), len(toks[ri])
                if max(nl, nr) > balance * min(nl, nr) or min(nl, nr) < 2:
                    continue
                di = same[int(d2.argmax())]
                if di == ri or objs[di]["color"] in (objs[li]["color"],
                                                     objs[ri]["color"]):
                    continue
                used = set().union(*toks.values())
                free = [t for t in range(grid * grid) if t not in used
                        and grid // 3 <= t // grid < grid - 1]
                if len(free) < nl:
                    continue
                rng = np.random.default_rng(2000 + si)
                floor = sorted(int(t) for t in rng.choice(free, size=nl,
                                                          replace=False))
                hit = dict(
                    scene_index=si, image_index=s["image_index"],
                    path=image_path(s), n_objects=len(objs), rel="near",
                    landmark=li, referent=ri, distractor=di,
                    tok_landmark=sorted(toks[li]),
                    tok_referent=sorted(toks[ri]),
                    tok_distractor=sorted(toks[di]),
                    tok_floor=floor,
                    box_landmark=bx[li]["box"], box_referent=bx[ri]["box"],
                    box_distractor=bx[di]["box"],
                    answer_main=objs[ri]["color"],
                    answer_control=objs[li]["color"],
                    colour_distractor=objs[di]["color"],
                    desc_landmark=describe(objs[li]),
                    question_main=(f"What colour is the {SAY[sh]} nearest "
                                   f"{describe(objs[li])}?"),
                    question_control=(
                        f"What colour is {describe(objs[li])}?"),
                    shape_referent=sh, shape_landmark=objs[li]["shape"],
                    n_same_shape=len(same),
                    near_ratio=round(float(o2[0] / o2[1]), 3))
                break
            if hit:
                break
        if hit:
            kept += 1
            yield hit


def pointer_items(grid: int, n_obj=(3, 5), rel: str = "left",
                  min_sep: float = 0.10, balance: float = 3.0,
                  limit: int | None = None, skip: int = 0):
    """Scenes that carry an unambiguous two-step question, one dict each.

    The construction, in the order the constraints bite:

    *landmark*   an object named uniquely by size, material and shape -- no
                 colour, because the control question asks for its colour and
                 a question must never contain its own answer.
    *referent*   the only object of its shape standing ``rel`` of the landmark,
                 while at least one further object of that same shape does not.
                 That second one is what makes the question two-step: the shape
                 alone does not pick the answer out, so the landmark has to be
                 read first.
    *distractor* a third object, in neither role, as far from both as the scene
                 allows, and a third colour.
    *floor*      as many token cells of bare floor as the landmark occupies,
                 which is what separates "this place mattered" from "covering
                 anywhere with an average render pulls the answer".

    The two questions share the render and the landmark phrase and differ only
    in which object the answer is about, so the roles exchange with the question
    and nothing else about the picture changes.
    """
    sgn = -1.0 if rel == "left" else 1.0
    kept = 0
    for si, s in enumerate(scenes()):
        if limit is not None and kept >= limit:
            return
        if si < skip:
            continue
        objs = s["objects"]
        if not (n_obj[0] <= len(objs) <= n_obj[1]) or not have(s):
            continue
        bx = object_boxes(s)
        if any(bx[j] is None or not bx[j]["clean"] for j in range(len(objs))):
            continue
        toks = {j: set(box_tokens(bx[j]["box"], grid)) for j in range(len(objs))}
        if any(not toks[j] for j in toks):
            continue
        # every object must own its cells outright, or covering one place
        # covers two objects and the reading is about neither
        if any(toks[i] & toks[j] for i in toks for j in toks if i < j):
            continue
        xs = [o["pixel_coords"][0] for o in objs]
        cl = np.array([o["pixel_coords"][:2] for o in objs], float)
        R = s["relationships"]
        hit = None
        for li in range(len(objs)):
            if not _unique(objs, ("size", "material", "shape"), li):
                continue
            side = set(R[rel][li])
            for sh in SHAPES:
                if sh == objs[li]["shape"]:
                    continue
                same = [j for j in range(len(objs))
                        if objs[j]["shape"] == sh and j != li]
                cand = [j for j in same if j in side]
                if len(same) < 2 or len(cand) != 1:
                    continue
                ri = cand[0]
                if objs[ri]["color"] == objs[li]["color"]:
                    continue
                # the alternatives of the same shape must sit clearly on the
                # other side, so that a reader would not hesitate either
                if sgn * (xs[ri] - xs[li]) < min_sep * 480:
                    continue
                if any(sgn * (xs[j] - xs[li]) > -min_sep * 480
                       for j in same if j != ri):
                    continue
                # the two lines that have to cross must be covering regions
                # of comparable size, or the crossing is a fact about how many
                # tokens each object happens to occupy
                nl, nr = len(toks[li]), len(toks[ri])
                if max(nl, nr) > balance * min(nl, nr) or min(nl, nr) < 2:
                    continue
                others = [j for j in range(len(objs)) if j not in (li, ri)]
                if not others:
                    continue
                di = max(others, key=lambda j: min(
                    float(np.hypot(*(cl[j] - cl[li]))),
                    float(np.hypot(*(cl[j] - cl[ri])))))
                if objs[di]["color"] in (objs[li]["color"], objs[ri]["color"]):
                    continue
                used = set().union(*toks.values())
                free = [t for t in range(grid * grid) if t not in used
                        and grid // 3 <= t // grid < grid - 1]
                if len(free) < len(toks[li]):
                    continue
                rng = np.random.default_rng(1000 + si)
                floor = sorted(int(t) for t in rng.choice(
                    free, size=len(toks[li]), replace=False))
                hit = dict(
                    scene_index=si, image_index=s["image_index"],
                    path=image_path(s), n_objects=len(objs), rel=rel,
                    landmark=li, referent=ri, distractor=di,
                    tok_landmark=sorted(toks[li]),
                    tok_referent=sorted(toks[ri]),
                    tok_distractor=sorted(toks[di]),
                    tok_floor=floor,
                    box_landmark=bx[li]["box"], box_referent=bx[ri]["box"],
                    box_distractor=bx[di]["box"],
                    answer_main=objs[ri]["color"],
                    answer_control=objs[li]["color"],
                    colour_distractor=objs[di]["color"],
                    desc_landmark=describe(objs[li]),
                    question_main=(
                        f"What colour is the {SAY[sh]} that is to the "
                        f"{rel} of {describe(objs[li])}?"),
                    question_control=(
                        f"What colour is {describe(objs[li])}?"),
                    shape_referent=sh, shape_landmark=objs[li]["shape"],
                    n_same_shape=len(same))
                break
            if hit:
                break
        if hit:
            kept += 1
            yield hit


def fetch(out: str = ROOT, n_images: int = 700,
          url: str = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"):
    """Range-read ``images/val`` and the val scene graph out of the zip."""
    import io
    import zipfile

    import requests

    class HTTPFile(io.RawIOBase):
        def __init__(self, url):
            self.url, self.s = url, requests.Session()
            r = self.s.head(url, timeout=60)
            r.raise_for_status()
            self.size, self.pos = int(r.headers["content-length"]), 0

        def seekable(self):
            return True

        def readable(self):
            return True

        def seek(self, off, whence=0):
            self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
            return self.pos

        def tell(self):
            return self.pos

        def read(self, n=-1):
            n = self.size - self.pos if n is None or n < 0 else n
            n = min(n, self.size - self.pos)
            if n <= 0:
                return b""
            for k in range(5):
                try:
                    r = self.s.get(self.url, timeout=180, headers={
                        "Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                    r.raise_for_status()
                    b = r.content
                    break
                except Exception:
                    if k == 4:
                        raise
            self.pos += len(b)
            return b

        def readinto(self, b):
            d = self.read(len(b))
            b[:len(d)] = d
            return len(d)

    os.makedirs(f"{out}/images/val", exist_ok=True)
    os.makedirs(f"{out}/scenes", exist_ok=True)
    zf = zipfile.ZipFile(io.BufferedReader(HTTPFile(url), buffer_size=1 << 20))
    names = zf.namelist()
    dst = f"{out}/scenes/CLEVR_val_scenes.json"
    if not os.path.exists(dst):
        e = [n for n in names if n.endswith("CLEVR_val_scenes.json")][0]
        with zf.open(e) as src, open(dst, "wb") as f:
            while (c := src.read(1 << 22)):
                f.write(c)
    for n in sorted(x for x in names
                    if "/images/val/" in x and x.endswith(".png"))[:n_images]:
        d = f"{out}/images/val/{os.path.basename(n)}"
        if not (os.path.exists(d) and os.path.getsize(d)):
            open(d, "wb").write(zf.open(n).read())
