# SPDX-License-Identifier: Apache-2.0
r"""The object lens: read the picture in objects, not in tiles.

Every map in this paper so far reported its readings over a square grid of
token tiles, because the tiles are where the activations live.  Nothing in the
mathematics wants the grid.  The conservation law,

.. math::
   \sum_R A_R(q) = F(h, q) - F(\bar h, q),

asks only that the regions :math:`R` *partition* the image positions: each
position in exactly one region, no position missing.  A partition drawn from a
segmentation -- this dog, that sofa, and one "rest of the scene" remainder --
satisfies it exactly as well as a 7x7 grid does, and the readings then speak
the language a person asks questions in: *the answer leans this much on the
dog, this much on the sofa, this much on everything else*.  That coordinate
change is the whole module.  No new estimator is introduced; the per-patch
readings come from :mod:`jvlm.vmap` and :mod:`jvlm.objective` unchanged, and
this module only decides which patches belong to which thing and adds them up.

Three rules keep the partition honest.

*Ownership is exclusive.*  A token cell goes to the object whose mask covers
the most of it, and only if that coverage clears a threshold; every unclaimed
cell -- sky, floor, whatever the annotator did not outline -- goes to a single
remainder segment called the *scene*.  So the segments are genuinely disjoint
and genuinely exhaustive, which is what the law needs, and the scene's share
is reported rather than hidden: on a real photograph the context often carries
real weight, and pretending the objects account for everything would be a lie
the arithmetic itself would expose.

*Pixels are not tokens.*  The token partition drives the readings; the pixel
masks drive the counterfactuals and the drawings.  Greying out "the dog" greys
the dog's own pixels, overlap and all, because that is the counterfactual a
reader means by covering the dog.

*Crowd annotations are dropped, and said so.*  COCO's ``iscrowd`` masks are
compressed RLE over unresolved groups; their pixels fall to the scene segment
here.  A photograph whose only annotation is a crowd never becomes an item.
"""
from __future__ import annotations

import base64
import json
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
from PIL import Image, ImageDraw

from jvlm import cocodata

# ---------------------------------------------------------------- COCO masks


@lru_cache(maxsize=1)
def _inst() -> tuple[dict, dict]:
    """Per-image instance polygons.  :mod:`jvlm.cocodata` keeps only boxes, and
    it is owned by other experiments, so the polygons are loaded here."""
    inst = json.load(open(cocodata.INSTANCES))
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    per: dict[int, list] = defaultdict(list)
    for a in inst["annotations"]:
        if a.get("iscrowd"):
            continue                       # RLE over a group; see module doc
        per[a["image_id"]].append(
            (cats[a["category_id"]], a["segmentation"], float(a["area"])))
    sizes = {im["id"]: (im["width"], im["height"]) for im in inst["images"]}
    return dict(per), sizes


def polygon_mask(polys: list, wh: tuple[int, int], res: int) -> np.ndarray:
    """Rasterise COCO polygons (original pixel coords) onto a ``res x res``
    square -- the same resize every photograph gets before the model sees it."""
    w, h = wh
    im = Image.new("1", (res, res), 0)
    dr = ImageDraw.Draw(im)
    for poly in polys:
        if len(poly) < 6:
            continue
        pts = [(poly[i] / w * res, poly[i + 1] / h * res)
               for i in range(0, len(poly), 2)]
        dr.polygon(pts, fill=1)
    return np.asarray(im, dtype=bool)


def category_masks(image_id: int, res: int = 448,
                   cats: list[str] | None = None) -> dict[str, np.ndarray]:
    """``{category: bool [res, res]}`` -- the union of that category's
    instances, on the square frame.  Categories outside ``cats`` are ignored
    (they will belong to the scene segment)."""
    per, sizes = _inst()
    out: dict[str, np.ndarray] = {}
    for cat, seg, _area in per.get(image_id, []):
        if cats is not None and cat not in cats:
            continue
        if not isinstance(seg, list):
            continue
        mk = polygon_mask(seg, sizes[image_id], res)
        out[cat] = out[cat] | mk if cat in out else mk
    return {c: mk for c, mk in out.items() if mk.any()}


# ---------------------------------------------------------------- partition


class Segment:
    """One region of the partition: a name, its token cells, its pixel mask."""

    def __init__(self, name: str, tokens: list[int], pix: np.ndarray | None,
                 is_object: bool):
        self.name = name
        self.tokens = sorted(int(t) for t in tokens)
        self.pix = pix                    # bool [res, res]; None for the scene
        self.is_object = is_object

    def __repr__(self) -> str:
        return f"Segment({self.name}, {len(self.tokens)} tokens)"


def partition(masks: dict[str, np.ndarray], G: int, *, thresh: float = 0.25,
              min_tokens: int = 2, max_obj: int = 4,
              max_frac: float = 0.75) -> list[Segment] | None:
    """Turn pixel masks into an exclusive, exhaustive partition of the
    ``G x G`` token cells.

    A cell belongs to the mask that covers the most of it, provided that
    coverage is at least ``thresh`` of the cell; otherwise it belongs to the
    scene.  Masks that end up with fewer than ``min_tokens`` cells are
    dissolved into the scene, largest ``max_obj`` masks are kept, and the
    result is refused (``None``) if a single mask claims more than
    ``max_frac`` of the frame -- covering it would be covering the picture.

    The returned list ends with the scene segment.  The partition property --
    every cell in exactly one segment -- is asserted, not assumed.
    """
    if not masks:
        return None
    names = sorted(masks, key=lambda c: -masks[c].sum())[: max_obj]
    res = masks[names[0]].shape[0]
    if res % G:                       # resample so cells pool exact blocks
        r2 = G * max(1, round(res / G))
        masks = {c: np.asarray(Image.fromarray(masks[c]).resize(
            (r2, r2), Image.NEAREST), dtype=bool) for c in names}
        res = r2
    s = res // G
    cov = np.stack([
        masks[c].reshape(G, s, G, s).mean(axis=(1, 3)).reshape(-1)
        for c in names])                                    # [n_masks, G*G]
    owner = np.where(cov.max(0) >= thresh, cov.argmax(0), -1)
    segs: list[Segment] = []
    for i, c in enumerate(names):
        tok = np.flatnonzero(owner == i)
        if len(tok) < min_tokens:
            owner[tok] = -1
            continue
        if len(tok) > max_frac * G * G:
            return None
        segs.append(Segment(c, tok.tolist(), masks[c], True))
    if not segs:
        return None
    segs.sort(key=lambda sg: -len(sg.tokens))
    segs.append(Segment("scene", np.flatnonzero(owner == -1).tolist(),
                        None, False))
    got = sorted(t for sg in segs for t in sg.tokens)
    assert got == list(range(G * G)), "not a partition of the token cells"
    return segs


def seg_sum(per_patch: torch.Tensor, segs: list[Segment]) -> torch.Tensor:
    """``[k, P] -> [k, S]`` by SUM: the reading the conservation law defines.
    Disjoint segments add, and the segment sums add to the total."""
    return torch.stack([per_patch[:, sg.tokens].sum(-1) for sg in segs], -1)


def seg_mean(per_patch: torch.Tensor, segs: list[Segment]) -> torch.Tensor:
    """``[k, P] -> [k, S]`` by MEAN: the matched aggregation for the content
    lenses, whose per-patch score has no units and no additivity -- summing it
    would score big segments high for being big."""
    return torch.stack([per_patch[:, sg.tokens].mean(-1) for sg in segs], -1)


# ------------------------------------------------------------- ground truths


def cover_truth(model, batch, voc: list[int], layer: int,
                segs: list[Segment], mean: torch.Tensor, ip: torch.Tensor,
                *, chunk: int = 6, answer_pos: int | None = None
                ) -> torch.Tensor:
    """``[S, k]`` -- the true drop in each token's logit when each segment's
    activations are replaced by the corpus mean at ``layer``.  Segments run as
    batch rows, ``chunk`` at a time."""
    from jlens.hooks import ActivationRecorder

    if answer_pos is None:
        answer_pos = batch["input_ids"].shape[1] - 1
    last = model.n_layers - 1
    with torch.no_grad(), ActivationRecorder(model.layers, at=[last]) as rec:
        model.forward(batch)
        base = model.unembed(
            rec.activations[last][0, answer_pos][None])[0, voc].float().cpu()
    drops = torch.zeros(len(segs), len(voc))
    for s0 in range(0, len(segs), chunk):
        grp = segs[s0: s0 + chunk]
        big = model.replicate(batch, len(grp)) if len(grp) > 1 else batch

        def hook(mod, inp, o, _g=grp):
            tup = not torch.is_tensor(o)
            hh = (o[0] if tup else o).clone()
            for j, sg in enumerate(_g):
                t = torch.as_tensor(sg.tokens, dtype=torch.long)
                hh[j, ip[t].to(hh.device)] = \
                    mean[t].to(hh.device).to(hh.dtype)
            return (hh, *o[1:]) if tup else hh

        hd = model.layers[layer].register_forward_hook(hook)
        try:
            with torch.no_grad(), \
                    ActivationRecorder(model.layers, at=[last]) as rec:
                model.forward(big)
                lg = model.unembed(
                    rec.activations[last][:, answer_pos])[:, voc].float().cpu()
        finally:
            hd.remove()
        drops[s0: s0 + len(grp)] = base[None] - lg
    return drops


def grey_image(img: Image.Image, mask: np.ndarray | None,
               segs: list[Segment] | None = None) -> Image.Image:
    """Grey out a pixel mask; for the scene segment (``mask is None``) grey
    everything *outside* the objects' masks."""
    arr = np.array(img)
    if mask is None:
        mk = np.zeros(arr.shape[:2], dtype=bool)
        for sg in segs or []:
            if sg.pix is not None:
                mk |= _at(sg.pix, arr.shape[0])
        mk = ~mk
    else:
        mk = _at(mask, arr.shape[0])
    arr[mk] = 128
    return Image.fromarray(arr)


def _at(mask: np.ndarray, res: int) -> np.ndarray:
    if mask.shape[0] == res:
        return mask
    return np.asarray(Image.fromarray(mask).resize((res, res), Image.NEAREST),
                      dtype=bool)


def pixel_truth(model, img: Image.Image, prompt: str, voc: list[int],
                segs: list[Segment]) -> torch.Tensor:
    """``[S, k]`` -- the drop in each token's logit when each segment's own
    *pixels* are greyed out and the photograph is re-encoded from scratch.
    Nothing about the lens's baseline is involved, so this ground truth is one
    the lens has never seen."""
    from jlens.hooks import ActivationRecorder

    last = model.n_layers - 1

    def logits(im):
        b = model.chat(prompt, [im])
        ap = b["input_ids"].shape[1] - 1
        with torch.no_grad(), ActivationRecorder(model.layers, at=[last]) as rec:
            model.forward(b)
            return model.unembed(
                rec.activations[last][0, ap][None])[0, voc].float().cpu()

    base = logits(img)
    out = torch.zeros(len(segs), len(voc))
    for j, sg in enumerate(segs):
        out[j] = base - logits(grey_image(img, sg.pix, segs))
    return out


# ------------------------------------------------------------------ serialise


def rle(mask: np.ndarray) -> dict:
    """Pack a boolean mask for a JSON card (packbits + base64)."""
    return {"shape": list(mask.shape),
            "bits": base64.b64encode(
                np.packbits(mask.astype(np.uint8))).decode()}


def unrle(d: dict) -> np.ndarray:
    flat = np.unpackbits(np.frombuffer(base64.b64decode(d["bits"]),
                                       dtype=np.uint8))
    n = int(np.prod(d["shape"]))
    return flat[:n].reshape(d["shape"]).astype(bool)


def card_segments(segs: list[Segment]) -> list[dict]:
    return [{"name": sg.name, "tokens": sg.tokens, "object": sg.is_object,
             "pix": (rle(sg.pix) if sg.pix is not None else None)}
            for sg in segs]


# ------------------------------------------------------------ automatic masks

_SAM = {}


def sam_masks(img: Image.Image, *, device: str = "cuda",
              model_id: str = "facebook/sam-vit-base",
              points_per_side: int = 16, min_frac: float = 0.02,
              max_masks: int = 5, dedup_iou: float = 0.75) -> dict:
    """``{'region 1': bool [res, res], ...}`` from SAM's automatic mask
    generation -- the partition with no ground-truth annotations at all.
    The lens then *names* each region itself, which is the demonstration that
    the method does not lean on COCO's labels.

    Masks are kept largest-first, dropping any that mostly repeats an already
    kept one (IoU above ``dedup_iou``) or is smaller than ``min_frac`` of the
    frame.  Loaded once per process; the pipeline object is cached.
    """
    key = (model_id, device)
    if key not in _SAM:
        from transformers import pipeline
        _SAM[key] = pipeline("mask-generation", model=model_id, device=device)
    out = _SAM[key](img, points_per_side=points_per_side)
    res = img.size[0]
    masks = [np.asarray(mk, dtype=bool) for mk in out["masks"]]
    masks = [mk for mk in masks if mk.shape == (res, res)]
    masks.sort(key=lambda mk: -int(mk.sum()))
    kept: list[np.ndarray] = []
    for mk in masks:
        area = mk.sum() / mk.size
        if area < min_frac or area > 0.85:
            continue
        if any((mk & kv).sum() / max((mk | kv).sum(), 1) > dedup_iou
               for kv in kept):
            continue
        kept.append(mk)
        if len(kept) >= max_masks:
            break
    return {f"region {i + 1}": mk for i, mk in enumerate(kept)}


# ------------------------------------------------------- counterfactual pairs

#: An open question per COCO supercategory that any of its members answers.
SUPER_Q = {
    "animal": "What animal is in this picture?",
    "vehicle": "What vehicle is in this picture?",
    "furniture": "What piece of furniture is in this picture?",
    "food": "What food is in this picture?",
    "appliance": "What appliance is in this picture?",
    "electronic": "What electronic device is in this picture?",
    "kitchen": "What kitchen item is in this picture?",
    "accessory": "What accessory is in this picture?",
    "sports": "What piece of sports equipment is in this picture?",
    "indoor": "What household object is in this picture?",
}
ONE_WORD = " Answer with one word."


def open_question(cat: str) -> str | None:
    sup = cocodata._load()["super_of"].get(cat)
    q = SUPER_Q.get(sup)
    return (q + ONE_WORD) if q else None


# ------------------------------------------------------------------- figures
#
# Everything below is drawing.  It imports matplotlib lazily so the GPU job
# pays nothing for it, and it reads only what the cards carry, so the figures
# can be re-rendered without a model or a card in the machine.

_SEG_COLORS = None


def _mpl():
    global _SEG_COLORS
    from jvlm import style
    style.use()
    import matplotlib.pyplot as plt
    _SEG_COLORS = [style.BLUE, style.ORANGE, style.AQUA, style.VIOLET]
    return plt, style


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _centroid(mk: np.ndarray) -> tuple[float, float]:
    """Centre of the mask's largest connected piece, so the label sits on the
    object rather than at the mean of two disconnected halves."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mk)
        if n > 1:
            big = 1 + int(np.argmax(ndimage.sum(mk, lab, range(1, n + 1))))
            mk = lab == big
    except ImportError:
        pass
    ys, xs = np.nonzero(mk)
    return float(xs.mean()), float(ys.mean())


def _draw_card(ax, img: Image.Image, card_segs: list[dict],
               shares: list[float], plt, style, fs: float = 15.0,
               focus: list[int] | None = None) -> None:
    """One photograph with objects tinted and labelled by their share.

    The tint's opacity carries the share, so between two panels the reader
    *sees* the weight move from one object to the other; the exact number is
    printed on the object.  ``focus`` limits tint and labels to the named
    segments (the two the questions ask about); everything else stays bare.
    """
    res = img.size[0]
    arr = np.asarray(img).astype(float) / 255.0
    over = np.zeros((res, res, 4))
    objs = [(j, s) for j, s in enumerate(card_segs)
            if s["object"] and (focus is None or j in focus)]
    if focus is not None:
        objs.sort(key=lambda t: focus.index(t[0]))     # stable colours
    import matplotlib.colors as mc
    marks = []
    for rank, (j, s) in enumerate(objs):
        mk = _at(unrle(s["pix"]), res)
        col = mc.to_rgba(_SEG_COLORS[rank % len(_SEG_COLORS)])
        al = 0.16 + 0.55 * float(np.clip(shares[j], 0.0, 1.0))
        over[mk] = [*col[:3], al]
        marks.append((j, s, mk, _SEG_COLORS[rank % len(_SEG_COLORS)]))
    ax.imshow(arr)
    ax.imshow(over)
    spots = []
    for j, s, mk, col in marks:
        ax.contour(mk.astype(float), levels=[0.5], colors=[col],
                   linewidths=2.4)
        cx, cy = _centroid(mk)
        # keep labels apart: nudge down past any earlier label nearby
        for px, py in spots:
            if abs(cx - px) < 0.42 * res and abs(cy - py) < 0.11 * res:
                cy = py + 0.11 * res
        cx = float(np.clip(cx, 0.14 * res, 0.86 * res))
        cy = float(np.clip(cy, 0.06 * res, 0.94 * res))
        spots.append((cx, cy))
        ax.annotate(f"{s['name']}  {_pct(shares[j])}", (cx, cy),
                    ha="center", va="center", fontsize=fs + 2,
                    color="#ffffff",
                    bbox=dict(boxstyle="round,pad=0.35", fc=col, ec="#ffffff",
                              lw=1.2, alpha=0.95))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_flip(card: dict, out_png: str, fs: float = 15.0) -> None:
    """THE payoff figure: one photograph, two questions, the object shares
    reorder.  ``card`` comes from the flip stage of ``exp/87_objlens.py``."""
    plt, style = _mpl()
    img = Image.open(card["path"]).convert("RGB").resize(
        (card["res"], card["res"]), Image.LANCZOS)
    focus = card.get("obj_idx")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.6))
    for ax, q in zip(axes, card["questions"]):
        _draw_card(ax, img, card["segments"], q["shares"], plt, style,
                   fs=fs, focus=focus)
        ax.set_title(q["question"].replace(ONE_WORD, "").strip(),
                     fontsize=fs + 2, pad=10)
        rest = 1.0 - sum(q["shares"][j] for j in (focus or []))
        ax.set_xlabel(f"answer: {q['answer']}        "
                      f"everything else: {_pct(rest)}", fontsize=fs + 1)
    fig.suptitle("Same picture, two questions", y=0.995, fontsize=fs + 8)
    fig.text(0.5, 0.925,
             "each object's share of the model's answer, read by the "
             "object lens",
             ha="center", fontsize=fs + 1, color=style.INK2)
    fig.tight_layout(rect=(0, 0.01, 1, 0.90))
    fig.savefig(out_png)
    plt.close(fig)


def fig_examples(cards: list[dict], out_png: str, fs: float = 14.0) -> None:
    """A row of photographs, each with its objects tinted by their share of
    the model's own answer to an open question."""
    plt, style = _mpl()
    cards = cards[:3]
    fig, axes = plt.subplots(1, len(cards), figsize=(4.7 * len(cards) + 0.4,
                                                     6.1))
    axes = np.atleast_1d(axes)
    for ax, cd in zip(axes, cards):
        img = Image.open(cd["path"]).convert("RGB").resize(
            (cd["res"], cd["res"]), Image.LANCZOS)
        q = cd["questions"][0]
        _draw_card(ax, img, cd["segments"], q["shares"], plt, style, fs=fs,
                   focus=cd.get("obj_idx"))
        ax.set_title(q["question"].replace(ONE_WORD, "").strip(),
                     fontsize=fs, pad=8)
        ax.set_xlabel(f"answer: {q['answer']}", fontsize=fs + 1)
    fig.suptitle("What the answer rests on", y=0.99, fontsize=fs + 7)
    fig.tight_layout(rect=(0, 0.01, 1, 0.92))
    fig.savefig(out_png)
    plt.close(fig)


_LENS_ORDER = ["object lens", "object lens (point)", "J-lens (mean)",
               "J-lens (sum)", "logit lens (mean)", "logit lens (sum)"]


def fig_grade(df, out_png: str, fs: float = 15.0) -> None:
    """Mean Spearman of each lens's object ranking against both causal
    truths, with cluster-bootstrap CIs over concepts."""
    plt, style = _mpl()
    from jvlm import stats as st
    truths = [("cover", "Cover the object's activations"),
              ("pixel", "Grey the object's pixels, re-encode")]
    lenses = [x for x in _LENS_ORDER if x in set(df.lens)] \
        + [x for x in sorted(set(df.lens)) if x not in _LENS_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 0.62 * len(lenses) + 2.6),
                             sharey=True)
    for ax, (tname, ttitle) in zip(axes, truths):
        d = df[df.truth == tname]
        ys = np.arange(len(lenses))[::-1]
        for y, lens in zip(ys, lenses):
            dd = d[d.lens == lens]
            if not len(dd):
                continue
            mu, lo, hi = st.cluster_bootstrap_ci(dd, "spearman", "concept")
            col = style.BLUE if lens.startswith("object") else style.MUTED
            ax.plot([lo, hi], [y, y], color=col, lw=3, solid_capstyle="round")
            ax.plot([mu], [y], "o", color=col, ms=9)
            ax.annotate(f"{mu:.2f}", (hi, y), xytext=(8, 0),
                        textcoords="offset points", va="center",
                        fontsize=fs - 1, color=style.INK2)
        ax.axvline(0, color=style.GRID, lw=1.2, zorder=0)
        ax.set_yticks(ys)
        ax.set_yticklabels(lenses, fontsize=fs)
        ax.set_xlim(-0.35, 1.05)
        ax.set_title(ttitle, fontsize=fs + 1, pad=8)
        ax.grid(axis="x", color=style.GRID, lw=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Spearman vs the true drops, over objects",
                      fontsize=fs - 1)
    fig.suptitle("Which lens ranks the objects as the interventions do",
                 y=1.02, fontsize=fs + 5)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def fig_complete(dc, out_png: str, fs: float = 15.0) -> None:
    """Claim against truth, one point per word per photograph per layer:
    the conservation law over the object partition, checked."""
    plt, style = _mpl()
    xs, ys = [], []
    for _, r in dc.iterrows():
        xs += list(r["truth"])
        ys += list(r["claim"])
    x, y = np.asarray(xs), np.asarray(ys)
    r_val = float(np.corrcoef(x, y)[0, 1])
    err = float(np.abs(x - y).max())
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    lim = float(max(np.abs(x).max(), np.abs(y).max())) * 1.08
    ax.plot([-lim, lim], [-lim, lim], color=style.GRID, lw=1.5, zorder=0)
    ax.scatter(x, y, s=16, alpha=0.35, color=style.BLUE, linewidths=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("measured effect of replacing the picture (logits)",
                  fontsize=fs)
    ax.set_ylabel("object readings, summed (logits)", fontsize=fs)
    ax.set_title("The object readings add up to the intervention",
                 fontsize=fs + 3, pad=10)
    ax.annotate(f"r = {r_val:.2f}   max gap {err:.2f}\n"
                f"{len(x)} words over {dc.i.nunique()} photographs",
                (0.04, 0.96), xycoords="axes fraction", va="top",
                fontsize=fs, color=style.INK2)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def fig_track(dflip, out_png: str, fs: float = 15.0) -> None:
    """Does the reading move to the object the question asks about?"""
    plt, style = _mpl()
    from jvlm import stats as st
    d = dflip[dflip.both_yes] if "both_yes" in dflip else dflip
    arms = [("object lens, reading the answer", "vj_track", style.BLUE),
            ("fixed lenses, reading the answer", "static_track", style.MUTED),
            ("logit lens, handed the question's word", "word_track",
             style.MUTED)]
    fig, ax = plt.subplots(figsize=(9.8, 3.9))
    ys = np.arange(len(arms))[::-1]
    ax.set_ylim(-0.6, len(arms) - 0.4)
    for y, (label, col_name, col) in zip(ys, arms):
        if col_name == "static_track":
            mu, lo, hi = 0.5, 0.5, 0.5      # byte-identical readings: pinned
        else:
            mu, lo, hi = st.cluster_bootstrap_ci(d, col_name, "concept")
        ax.plot([lo, hi], [y, y], color=col, lw=3, solid_capstyle="round")
        ax.plot([mu], [y], "o", color=col, ms=10)
        ax.annotate(f"{mu:.2f}", (hi, y), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=fs,
                    color=style.INK2)
    ax.axvline(0.5, color=style.GRID, lw=1.4, zorder=0)
    ax.annotate("chance", (0.5, -0.52), ha="center", va="bottom",
                fontsize=fs - 2, color=style.MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels([a[0] for a in arms], fontsize=fs)
    ax.set_xlim(0.3, 1.05)
    ax.grid(axis="x", color=style.GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of questions whose named object gets the larger "
                  "reading", fontsize=fs - 1)
    ax.set_title("Ask about a different object and the reading follows",
                 fontsize=fs + 3, pad=10)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# -------------------------------------------------------------------- checks


def selftest() -> dict:
    """CPU checks with no model: the partition is a partition, the pooled
    aggregations are exact, and the mask round-trips."""
    out = {}
    rng = np.random.default_rng(0)
    res, G = 448, 14
    a = np.zeros((res, res), bool)
    a[40:200, 60:260] = True
    b = np.zeros((res, res), bool)
    b[150:420, 200:440] = True          # overlaps a
    segs = partition({"dog": a, "sofa": b}, G)
    out["partition"] = (segs is not None
                        and segs[-1].name == "scene"
                        and sorted(t for sg in segs for t in sg.tokens)
                        == list(range(G * G)), "exclusive and exhaustive")
    ov = set(segs[0].tokens) & set(segs[1].tokens)
    out["disjoint"] = (len(ov) == 0, f"{len(ov)} shared cells")
    x = torch.from_numpy(rng.normal(size=(3, G * G))).float()
    out["sum_additive"] = (
        bool(torch.allclose(seg_sum(x, segs).sum(-1), x.sum(-1), atol=1e-5)),
        "segment sums equal the total")
    mk = rng.random((res, res)) > 0.5
    out["rle"] = (bool((unrle(rle(mk)) == mk).all()), "mask round-trip")
    # ragged resolution: res not divisible by G
    m2 = {"cat": np.asarray(Image.fromarray(a).resize((450, 450),
                                                      Image.NEAREST), bool)}
    s2 = partition(m2, G)
    out["ragged"] = (s2 is not None and sorted(
        t for sg in s2 for t in sg.tokens) == list(range(G * G)),
        "non-divisible resolution still partitions")
    return out


if __name__ == "__main__":
    R = selftest()
    w = max(len(k) for k in R)
    for k, (ok, note) in R.items():
        print(f"  {k:<{w}}  {'ok' if ok else 'FAIL'}  {note}")
    bad = [k for k, v in R.items() if not v[0]]
    raise SystemExit(1 if bad else 0)
