# SPDX-License-Identifier: Apache-2.0
"""Watching one photograph being thought about, layer by layer.

Every other layerwise figure in this paper is a summary statistic against
depth: one number per layer, averaged over photographs.  That says a lens gets
better or worse with depth; it cannot say *what moves*.  These figures keep the
photograph fixed and let depth be the axis, so a reader watches the causal mass
travel across the frame and can see, column by column, which reading follows it.

Three constructions, all off the same file -- the causal map recomputed at every
one of twelve layers, with each lens read at the same regions and the same
layers:

    layermap     one photograph; rows are the truth and the lenses, columns are
                 the layers, each cell a 7x7 map over the frame
    trajectory   one region -- the one that ends up mattering most -- followed
                 up the stack, as its rank among the 49
    readable     when a lens first prints something true at an image token

Conventions are :mod:`jvlm.figcausal`'s, because a reader should not have to
learn a second set: square tiles, the heading pad computed from the line count,
words clipped rather than allowed to run into their neighbours, end labels
staggered when two series finish together, and the axis's two ends named under
it.  The one addition is that each row here wears its lens's paper colour as a
single-hue ramp -- twelve columns of the same blue would leave the row name as
the only thing telling four rows apart.
"""
from __future__ import annotations

import os

import numpy as np

from jvlm import style
from jvlm.figcausal import LENS_COLOUR
from jvlm.style import GRID, INK, INK2, MUTED, SURFACE

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

HOT = "#b8481f"
#: the truth is not a lens, so it wears no lens hue
TRUTH_INK = "#3c3a36"

#: the three lenses drawn against the truth, in the paper's order
ORDER = ["logit lens", "J-lens", "VJ-Lens"]
TRUTH = "what covering it\nreally does"


def _ramp(colour, n: int = 9):
    """A light-to-dark ramp in one hue, for a magnitude.

    The light end is the page, so an empty cell reads as empty; the dark end is
    the lens's own colour, so the row is identifiable from six feet away.  The
    ramp is built rather than listed because there are four of them and a
    hand-tuned list per lens would drift apart.
    """
    r, g, b = to_rgb(colour)
    # The steps are not evenly spaced between white and the hue.  A lens colour
    # chosen to be legible as a line is already dark, so an even ramp spends
    # seven of its nine steps in shades a reader cannot tell apart and the
    # whole panel comes out a solid block.  The exponent pushes the range into
    # the light half, where the differences are visible.
    t = np.linspace(0.04, 1.0, n) ** 1.45
    out = [tuple(1 - x * (1 - c) for c in (r, g, b)) for x in t]
    out[-1] = tuple(c * 0.74 for c in (r, g, b))   # past the hue, for the peak
    return out


def _shade(ramp, q: float):
    if q != q:
        return SURFACE
    return ramp[max(0, min(len(ramp) - 1, int(q * len(ramp))))]


def _dark(colour) -> bool:
    r, g, b = to_rgb(colour)[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b < 0.55


def _quant(v, scale: str = "rank"):
    """Where each region stands inside its own column.

    Two things have to be true at once.  A scale shared down the whole strip
    would leave the early columns blank -- covering a region does a few
    hundredths of a logit at layer 5 and a quarter of one at layer 13 -- and
    the figure would say nothing about where the influence is at the layers it
    was drawn to show.  But a column's own min and max are not enough either:
    the contribution reading is heavily peaked, so one region takes the dark
    end and the other forty-eight land in the first shade, and the row comes
    out blank for the opposite reason.

    So a tile's shade is its RANK among the forty-nine, not its value.  That is
    also the quantity every agreement number in this paper is computed from --
    the Spearman correlations are rank correlations -- so the picture and the
    number are finally about the same thing.  What a column is worth in logits
    is the bar strip above it, which is where the magnitude went.
    """
    v = np.asarray(v, dtype=float)
    if scale == "minmax":
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        return (v - lo) / ((hi - lo) or 1.0)
    o = np.argsort(np.argsort(v))
    return o / max(len(v) - 1, 1)


# ------------------------------------------------------------------ figure A
#
# One photograph, the truth and the lenses down the rows, the layers across the
# columns.  Everything about the layout is computed in inches: the one thing
# that must hold is that the photograph, the row names and the strip do not
# overlap, and a layout in figure fractions stops holding the moment the number
# of layers changes.

def layermap(cards, out, name="f83_layermap.png", order=None, img=None,
             n_layers=None, per_cell=1.55, gap=0.62, photo_max=3.4,
             gutter=2.10, key=True, bars=True, top_k=1, max_cols=6):
    """Twelve layers of one photograph's causal map, with each lens beside it.

    ``cards`` are the cards for a single photograph, one per layer, exactly as
    ``60_causalmap.py --cards-every-layer`` writes them.

    A ring marks the region the truth calls most damaging at that layer; a dot
    marks the region that row calls strongest.  A lens tracks the influence
    exactly when its dot sits inside the ring, and that is a thing a reader can
    check twelve times in one glance rather than take on trust.
    """
    cards = sorted(cards, key=lambda c: int(c["layer"]))
    if not cards:
        return None
    # Twelve columns of 7x7 tiles cannot be read at page width; the travel of
    # the causal mass is just as visible over six evenly spaced depths, drawn
    # at twice the size.  The first and last measured layers always stay.
    if max_cols and len(cards) > max_cols:
        idx = sorted({int(round(t)) for t in
                      np.linspace(0, len(cards) - 1, max_cols)})
        cards = [cards[i] for i in idx]
    order = [l for l in (order or ORDER) if l in cards[0]["lenses"]]
    rows = [None] + order                     # None is the truth row
    C = int(cards[0]["coarse"])
    keep = list(cards[0]["keep"])
    layers = [int(c["layer"]) for c in cards]
    L, R = len(cards), len(rows)
    n_layers = n_layers or (max(layers) + 2)
    box = set(cards[0].get("inside", []))
    concept = cards[0]["concept"]

    style.use()
    unit = per_cell / C
    wide = L * C + (L - 1) * gap
    tall = R * C + (R - 1) * gap
    strip_w, strip_h = unit * wide, unit * tall
    photo_in = min(strip_h, photo_max)
    left, right = 0.10, 0.16
    below = 1.18                              # layer numbers and axis names
    bar_h = 0.34 if bars else 0.0
    key_h = 1.10 if key else 0.0
    # the key stands a third of an inch higher than it used to, to clear the
    # bar's own value; the band above it grows by the same amount so its
    # caption does not meet the figure's heading
    above = 0.98 + key_h + bar_h
    W = left + photo_in + gutter + strip_w + right
    H = strip_h + below + above
    fig = plt.figure(figsize=(W, H))

    ax = fig.add_axes([(left + photo_in + gutter) / W, below / H,
                       strip_w / W, strip_h / H])
    ax.set_xlim(0, wide)
    ax.set_ylim(tall, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ramps = {None: _ramp(TRUTH_INK)}
    for l in order:
        ramps[l] = _ramp(LENS_COLOUR.get(l, MUTED))

    peak = []
    for j, c in enumerate(cards):
        truth = np.asarray(c["truth"], dtype=float)
        peak.append(float(np.nanmax(truth)))
        tru_top = set(np.argsort(-truth)[:top_k].tolist())
        for r, lens in enumerate(rows):
            v = truth if lens is None else np.asarray(
                c["lenses"][lens]["score"], dtype=float)
            q = _quant(v)
            ramp = ramps[lens]
            bx, by = j * (C + gap), r * (C + gap)
            for i, cell in enumerate(keep):
                y, x = divmod(int(cell), C)
                ax.add_patch(Rectangle((bx + x, by + y), 1.0, 1.0,
                                       facecolor=_shade(ramp, q[i]),
                                       edgecolor="none", zorder=2))
            # where the truth is, in every row, so the rows can be compared
            for t in tru_top:
                y, x = divmod(int(keep[t]), C)
                ax.add_patch(Rectangle((bx + x + .07, by + y + .07), .86, .86,
                                       fill=False, edgecolor=HOT, lw=1.5,
                                       zorder=6))
            # where this row points
            a = int(np.argmax(v))
            y, x = divmod(int(keep[a]), C)
            ax.plot([bx + x + .5], [by + y + .5], marker="o", ms=3.4, mew=0,
                    color="#ffffff" if _dark(_shade(ramp, q[a])) else INK,
                    zorder=7)
            # the human's box, on the truth row only: it is the same cells in
            # every row, and four outlines a column would leave no white in the
            # strip at all.  One perimeter rather than an outline per cell --
            # forty-nine dashed squares are a texture, not a boundary -- and a
            # white line under the dark one, or it disappears into the tiles it
            # is drawn over
            if lens is None and box:
                _perimeter(ax, box, C, bx, by)

    # ---- row names, in the gutter, each with its own colour beside it
    for r, lens in enumerate(rows):
        y = r * (C + gap) + C / 2
        ax.annotate(TRUTH if lens is None else lens, xy=(0, y),
                    xytext=(-13, 0), textcoords="offset points", ha="right",
                    va="center", color=INK, fontsize=20, linespacing=1.3,
                    annotation_clip=False)
        if lens is not None:
            ax.plot([0], [y], marker="s", ms=6.5, mew=0,
                    color=LENS_COLOUR.get(lens, MUTED), clip_on=False,
                    transform=ax.transData + _off(ax, -6, 0), zorder=8)

    # ---- under the strip: the layer, and what the two ends of the axis are
    for j, l in enumerate(layers):
        ax.annotate(str(l), xy=(j * (C + gap) + C / 2, tall), xytext=(0, -6),
                    textcoords="offset points", ha="center", va="top",
                    color=INK2, fontsize=20.3, annotation_clip=False)
    for xp, ha, txt in ((0.0, "left", "just after the picture is read in"),
                        (float(wide), "right", "just before the answer")):
        ax.annotate(txt, xy=(xp, tall), xytext=(0, -28),
                    textcoords="offset points", ha=ha, va="top", color=MUTED,
                    fontsize=18.8, annotation_clip=False)
    d0 = 100 * layers[0] / max(n_layers - 1, 1)
    d1 = 100 * layers[-1] / max(n_layers - 1, 1)
    ax.annotate(f"layer of the language model   (layer {layers[0]} is "
                f"{d0:.0f}% of the way through, layer {layers[-1]} is "
                f"{d1:.0f}%)", xy=(wide / 2, tall), xytext=(0, -54),
                textcoords="offset points", ha="center", va="top", color=INK,
                fontsize=21.8, annotation_clip=False)

    # ---- above the strip: how much covering anything is worth at that layer
    if bars:
        ab = fig.add_axes([(left + photo_in + gutter) / W,
                           (below + strip_h + 0.14) / H, strip_w / W,
                           bar_h / H])
        ab.set_xlim(0, wide)
        ab.set_ylim(0, max(peak) * 1.12 or 1.0)
        ab.set_xticks([]); ab.set_yticks([])
        ab.set_facecolor("none")
        for sp in ab.spines.values():
            sp.set_visible(False)
        ab.plot([0, wide], [0, 0], color=GRID, lw=0.9, zorder=1,
                clip_on=False)
        top = int(np.argmax(peak))
        for j, p in enumerate(peak):
            ab.add_patch(Rectangle((j * (C + gap) + C * 0.12, 0), C * 0.76, p,
                                   facecolor=MUTED, edgecolor="none",
                                   zorder=2))
        ab.annotate("what the worst cover costs, in logits", xy=(0, 0),
                    xytext=(-13, 1), textcoords="offset points", ha="right",
                    va="bottom", color=INK2, fontsize=18.8,
                    annotation_clip=False)
        ab.annotate(f"{peak[top]:.2f}",
                    xy=(top * (C + gap) + C * 0.5, peak[top]), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    color=INK2, fontsize=18.8, annotation_clip=False)

    # ---- the key, on one row.  The two marks used to sit under the shade
    # swatches, which put them exactly where the bar strip prints its largest
    # value; one row of key cannot collide with anything above it
    if key:
        n, unit_in, fs = 8, 0.26, 13.0
        # the entries are laid out from the widths of their own words, not
        # from constants: guessing where a thirty-character label ends is how
        # the two marks came to be printed on top of each other
        wide_of = lambda t: len(t) * fs * 0.50 / 72.0 / unit_in
        t1 = "the truly most damaging region"
        t2 = "where that row points"
        x_ring = n + 2.0
        x_dot = x_ring + 1.2 + wide_of(t1) + 1.8
        # the swatch also has to be wide enough for the two words that name its
        # ends, which is a wider requirement than the ramp itself has
        span = x_dot + 0.9 + wide_of(t2)
        kx = (left + photo_in + gutter) / W
        # the gap holds the bar's largest value, printed over the tallest
        # bar, and the two words hanging under the key
        ky = (below + strip_h + 0.14 + bar_h + 0.86) / H
        ak = fig.add_axes([kx, ky, span * unit_in / W, 0.17 / H])
        ak.set_xlim(0, span); ak.set_ylim(0, 1)
        ak.set_xticks([]); ak.set_yticks([]); ak.set_facecolor("none")
        for sp in ak.spines.values():
            sp.set_visible(False)
        for i in range(n):
            ak.add_patch(Rectangle((i + .02, 0), .96, 1, edgecolor="none",
                                   facecolor=_shade(ramps[None],
                                                    (i + .5) / n)))
        # The two words flank the gradient rather than hanging under its ends.
        # The swatch is only n units of an axes that is much wider, so set
        # under it they had about a centimetre each and printed as one word;
        # beside it they have the whole gutter and stay at the shared size.
        # ...on their own row under the swatch, where the legend above is not
        # competing for the space, and running outward from its two ends
        # rather than inward, which is what made one word of them.
        # sized off the swatch they name: it is n * unit_in wide, not as wide
        # as the axes that holds it, and the two words have to share it
        _w_pt = n * unit_in * 72.0
        _kfs = round(min(18.1, _w_pt / (len("weakeststrongest") + 3) / 0.50), 1)
        for x, ha, dx, t in ((0, "left", -2, "weakest"),
                             (n, "right", 2, "strongest")):
            ak.annotate(t, xy=(x, 0), xytext=(dx, -5),
                        textcoords="offset points", ha=ha, va="top",
                        color=MUTED, fontsize=_kfs, annotation_clip=False)
        ak.annotate("where a region ranks inside its own column", xy=(0, 1),
                    xytext=(0, 5), textcoords="offset points", ha="left",
                    va="bottom", color=INK2, fontsize=18.8,
                    annotation_clip=False)
        ak.add_patch(Rectangle((x_ring, 0.02), 0.96, 0.96, fill=False,
                               edgecolor=HOT, lw=1.6))
        ak.annotate(t1, xy=(x_ring + 1.2, 0.5), xytext=(0, 0),
                    textcoords="offset points", ha="left", va="center",
                    color=INK2, fontsize=fs, annotation_clip=False)
        ak.plot([x_dot], [0.5], marker="o", ms=4.0, mew=0, color=INK,
                clip_on=False)
        ak.annotate(t2, xy=(x_dot + 0.7, 0.5), xytext=(0, 0),
                    textcoords="offset points", ha="left", va="center",
                    color=INK2, fontsize=fs, annotation_clip=False)

    # ---- the photograph
    ap = fig.add_axes([left / W, (below + (strip_h - photo_in) / 2) / H,
                       photo_in / W, photo_in / H])
    if img is not None:
        ap.imshow(img)
    ap.set_xticks([]); ap.set_yticks([])
    ap.set_aspect("equal")
    for sp in ap.spines.values():
        sp.set_visible(False)
    if img is None:
        ap.set_facecolor(GRID)
    fig.text(left / W, (below + (strip_h - photo_in) / 2 - 0.24) / H,
             f"the model answers “{concept}”", ha="left", va="top",
             color=INK2, fontsize=20.3)

    fig.text(0.5, 1 - 6 / (72 * H), "Where the answer is coming from, "
             "layer by layer", ha="center", va="top", color=INK, fontsize=23.5)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def _perimeter(ax, cells, C, bx=0.0, by=0.0, lw=1.0):
    """One dashed outline around a block of cells, with a white line under it.

    ``cells`` come from a bounding box, so they are a rectangle in the coarse
    grid and the perimeter is the four sides of that rectangle.
    """
    ys = [int(c) // C for c in cells]
    xs = [int(c) % C for c in cells]
    x0, x1 = min(xs), max(xs) + 1
    y0, y1 = min(ys), max(ys) + 1
    ax.add_patch(Rectangle((bx + x0, by + y0), x1 - x0, y1 - y0, fill=False,
                           edgecolor="#ffffff", lw=lw * 2.6, alpha=0.85,
                           zorder=4))
    ax.add_patch(Rectangle((bx + x0, by + y0), x1 - x0, y1 - y0, fill=False,
                           edgecolor=INK, lw=lw, linestyle=(0, (3, 2)),
                           zorder=5))


def _off(ax, x, y):
    from matplotlib.transforms import ScaledTranslation
    return ScaledTranslation(x / 72.0, y / 72.0, ax.figure.dpi_scale_trans)


# ------------------------------------------------------------------ figure B
#
# The map figure shows the whole frame at every layer and asks the reader to
# find the movement in it.  This is the same movement as one line: pick the
# region that ends up mattering most, and follow it.
#
# Rank rather than value, because the four series have no common unit -- a
# logit drop, a logit, a fitted logit and a contribution -- and a panel with
# four y axes on it is not a panel.  Rank among the 49 regions is what the
# agreement number in every other causal figure is computed from anyway, so
# this is that number's own picture.

def _stand(v, j: int) -> float:
    """How far region ``j`` stands above the rest of the frame, in sd.

    Four series with no common unit -- a logit drop, a logit, a fitted logit
    and a contribution -- cannot share an axis as they are, and four y axes on
    one panel is not a panel.  Standardising each map across its own
    forty-nine regions puts all four in the same unit and keeps the only thing
    the panel is about: how far this place stands out from the rest of the
    picture, at that layer, under that reading.

    The rank would be the other candidate and is the wrong one here.  The true
    drops arrive quantised in sixteenths of a logit, so a third of the frame
    ties at every layer and a tied region's rank jumps between first and
    twelfth on changes of a sixteenth -- which is a property of the storage
    format, not of the model.
    """
    v = np.asarray(v, dtype=float)
    sd = float(v.std())
    return float((v[j] - v.mean()) / sd) if sd > 1e-12 else 0.0


def trajectory(groups, out, name="f84_traject.png", order=None, n_layers=None,
               photos=None, n_max=3, top_at="peak", which=None):
    """One region per photograph, followed up the stack.

    The region is the worst single thing that can be covered anywhere in the
    frame at any depth: the one whose covering costs the answer most, at the
    layer where covering it costs most.  Taking instead whatever happens to be
    strongest at the last layer measured picks a region out of the layers where
    covering *anything* has stopped doing much, and the panel is then about
    noise.  Each panel asks, of that one place: when does it start mattering,
    and when does each lens notice?
    """
    keys = list(groups)
    keys = [keys[i] for i in which] if which else keys[:n_max]
    if not keys:
        return None
    order = [l for l in (order or ORDER) if l in groups[keys[0]][0]["lenses"]]
    style.use()
    n = len(keys)
    # the photographs sit in a row of their own above the panels.  An inset in
    # the corner of a panel is drawn over the data, and on a rank axis the
    # corner a line is not in changes from photograph to photograph, so there
    # is no corner that is safe to put it in
    fig, gaxes = plt.subplots(2, n, figsize=(5.6 * n, 6.9), squeeze=False,
                              height_ratios=[1.0, 4.15], sharey="row")
    axes, paxes = gaxes[1], gaxes[0]
    R = len(groups[keys[0]][0]["truth"])

    for ax, pax, path in zip(axes, paxes, keys):
        g = groups[path]
        layers = [int(c["layer"]) for c in g]
        T = np.array([c["truth"] for c in g], dtype=float)      # [layer, R]
        j = (int(np.argmax(T.max(0))) if top_at == "peak"
             else int(np.argmax(T[-1])))
        at = int(layers[int(np.argmax(T[:, j]))])
        cell = int(g[0]["keep"][j])
        C = int(g[0]["coarse"])
        inside = cell in set(g[0].get("inside", []))

        ends = []
        series = [(None, TRUTH_INK, [_stand(c["truth"], j) for c in g])]
        for lens in order:
            series.append((lens, LENS_COLOUR.get(lens, MUTED),
                           [_stand(c["lenses"][lens]["score"], j) for c in g]))
        for lens, colour, y in series:
            lw = 2.6 if lens is None else 2.0
            ax.plot(layers, y, color=colour, lw=lw, marker="o",
                    ms=6.0 if lens is None else 5.0,
                    zorder=4 if lens is None else 3, clip_on=False)
            ends.append([layers[-1], float(y[-1]),
                         "the truth" if lens is None else lens, colour])
        _layer_axis(ax, layers)
        # the zero line stops short of the label column: drawn across the whole
        # axes it is struck through whichever name lands nearest zero
        x0, x1 = ax.get_xlim()
        ax.plot([x0, layers[-1] + (x1 - layers[-1]) * 0.30], [0, 0],
                color=INK, lw=1.0, ls=(0, (2, 3)), zorder=1)
        ax.margins(y=0.16)
        _stagger(ax, ends, log=False)
        style.despine(ax)
        ax.grid(axis="y", color=GRID, lw=0.9)
        if ax is axes[0]:
            ax.set_ylabel("how far this one region stands above\n"
                          "the rest of the frame  (standard deviations)")
        _head(ax, f"“{g[0]['concept']}”  ·  "
                  + ("inside the object's box" if inside
                     else "outside the object's box"),
              f"covering it costs {T[:, j].max():.2f} logits at layer {at}")
        _photo(pax, (photos or {}).get(path), cell, C)

    fig.text(0.5, 0.985, "One region, followed up the stack",
             ha="center", va="top", color=INK, fontsize=23.5)
    fig.subplots_adjust(left=0.058, right=0.988, top=0.905, bottom=0.088,
                        wspace=0.30, hspace=0.34)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def _layer_axis(ax, layers, room=0.30):
    """Layers across, with room at the right for the end labels.

    The room is part of the axis, so the spine and any tick on it would say
    the measurement runs six layers past the last one measured.  The ticks are
    therefore placed inside the measured range and nowhere else.
    """
    lo, hi = layers[0], layers[-1]
    ax.set_xlim(lo - 0.9, hi + max(2.0, (hi - lo) * room))
    step = 5 if (hi - lo) <= 30 else 10
    ax.set_xticks([t for t in range(0, hi + 1, step) if lo <= t <= hi])
    ax.set_xlabel("layer of the language model")


def _head(ax, title, sub=None, fs=20.6):
    # the pad has to clear the subtitle's OWN font size (18.8) plus its 7pt
    # offset from the axes edge, not the panel's body-text size -- at the
    # smaller stand-in size the subtitle's top line ran into the title above
    # it
    sub_fs = 18.8
    lines = (str(sub).count("\n") + 1) if sub else 0
    ax.set_title(title, loc="center", color=INK,
                 pad=14 + 7 + lines * sub_fs * 1.35, fontsize=fs)
    if sub:
        ax.annotate(sub, xy=(0.5, 1), xytext=(0, 7), xycoords="axes fraction",
                    textcoords="offset points", ha="center", va="bottom",
                    color=INK2, fontsize=sub_fs, linespacing=1.35)


def _photo(ax, img, cell, C):
    """The photograph, with the followed region ringed on it.

    A region index is not a place until it is shown as one; without this the
    panel is about a number and the reader has no way to see where in the
    frame that number is.
    """
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if img is None:
        return
    ax.imshow(img)
    ax.set_aspect("equal")
    ax.set_anchor("SW")              # against the panel's own left edge,
    #                                  and down against the heading under it
    w, h = img.size
    y, x = divmod(int(cell), C)
    ax.add_patch(Rectangle((x * w / C, y * h / C), w / C, h / C, fill=False,
                           edgecolor=HOT, lw=2.2))


# ------------------------------------------------------------------ figure D
#
# The strip figure shades a cell by where the photographed object's own name
# ranks there.  That is the right picture for one photograph and it is not a
# number, so it cannot be read across photographs or compared between the two
# runs.  These are the same two questions as quantities: does the lens name the
# thing at all, and does it name it in the right place.

#: the word-grid run reserves the bare name for the single-point reading and
#: calls the path-integrated arm -- the one the paper means -- something else
STRIP_NAME = {"VJ-Lens": "VJ-Lens (path)"}


def _names_it(card, lens):
    """Share of the box's tiles, and of the rest, printing the object's name."""
    C = int(card["coarse"])
    keep = card["keep"]
    box = set(card.get("inside", []))
    w = card["lenses"][lens]["word"]
    c = card["concept"]
    ins = [w[j] == c for j in range(len(keep)) if int(keep[j]) in box]
    out = [w[j] == c for j in range(len(keep)) if int(keep[j]) not in box]
    return (float(np.mean(ins)) if ins else np.nan,
            float(np.mean(out)) if out else np.nan)


def readable(strip, groups, out, name="f85_readable.png", order=None,
             n_layers=None, vocab_causal=30):
    """When does a lens first read something true at an image token?

    The first two panels are one photograph each, at twenty-four layers: where
    the photographed object's own name ranks in a two-hundred-word vocabulary,
    read at the tiles the human's box covers.  There is no average over the two
    because there is nothing sensible to average -- one of them is at first
    place and the other at two hundredth, and the mean of those is a rank no
    photograph ever had.

    The third panel is eight photographs at twelve layers, and a different
    question of the same kind: what share of the box's tiles print that name at
    all, against the share of every other tile that does, because a lens
    printing the name over the whole frame has not found anything.
    """
    order = list(order or ORDER)
    style.use()
    cards = [c for c in (strip or {}).get("cards", [])
             if any(STRIP_NAME.get(l, l) in c["lenses"] for l in order)]
    cards = cards[:2]
    nv = len((strip or {}).get("vocab", [])) or 200
    n = len(cards) + 1
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.9), squeeze=False)
    axes = axes[0]
    if len(cards) > 1:
        axes[1].sharey(axes[0])

    # ---------------------------------------------- one photograph at a time
    for ax, card in zip(axes, cards):
        ends = []
        # Two lenses can hold first place at every layer, and one line is then
        # drawn exactly under the other and cannot be seen at all.  Each lens
        # is offset by a couple of points along y -- in points, not in ranks,
        # so no line is drawn at a rank it does not have.
        for k, lens in enumerate(order):
            dy = (k - (len(order) - 1) / 2) * 3.0
            fn = STRIP_NAME.get(lens, lens)
            per = card["lenses"].get(fn)
            if not per:
                continue
            colour = LENS_COLOUR.get(lens, MUTED)
            layers = sorted(int(q) for q in per)
            y = [per[str(l)]["box_rank"] + 1 for l in layers]
            tr = ax.transData + _off(ax, 0, dy)
            ax.plot(layers, y, color=colour, lw=2.2, zorder=3, transform=tr)
            first = next((q for q, l in enumerate(layers) if y[q] == 1), None)
            if first is not None:
                ax.plot([layers[first]], [1], marker="o", ms=10.0,
                        mfc=SURFACE, mec=colour, mew=2.2, zorder=5,
                        clip_on=False, transform=tr)
            ends.append([layers[-1], float(y[-1]), lens, colour])
        _rank_axis(ax, nv, layers)
        if ax is axes[0]:
            ax.set_ylabel("where the object's own name ranks,\n"
                          "read at the tiles its box covers")
        else:
            # not set_yticklabels([]): these two axes share a y axis, so a
            # formatter cleared on one is cleared on both and the figure loses
            # the labels it does want
            ax.tick_params(labelleft=False)
        _stagger(ax, ends, log=True)
        _head(ax, f"\u201c{card.get('caption', card['concept'])}\u201d "
                  f"\u00b7  one photograph, {len(layers)} layers")

    # ------------------------------------------------- and eight photographs
    ax = axes[-1]
    cc = [c for g in groups.values() for c in g]
    layers = sorted({int(c["layer"]) for c in cc})
    n_photo = len(groups)
    ends = []
    for lens in order:
        if not cc or lens not in cc[0]["lenses"]:
            continue
        colour = LENS_COLOUR.get(lens, MUTED)
        ins, outs = [], []
        for l in layers:
            v = [_names_it(c, lens) for c in cc if int(c["layer"]) == l]
            ins.append(float(np.nanmean([x[0] for x in v])))
            outs.append(float(np.nanmean([x[1] for x in v])))
        ax.plot(layers, outs, color=colour, lw=1.6, alpha=0.55,
                ls=(0, (2, 2)), zorder=2)
        ax.plot(layers, ins, color=colour, lw=2.4, marker="o", ms=5.5,
                zorder=3)
        ends.append([layers[-1], float(ins[-1]), lens, colour])
    ax.axhline(1.0 / vocab_causal, color=INK, lw=1.0, ls=(0, (2, 3)), zorder=1)
    # under the line and at its right-hand end: over the line is where both
    # dashed curves run, and at the left-hand end the solid ones start
    ax.annotate("naming it by chance", xy=(0.99, 1.0 / vocab_causal),
                xytext=(0, -7), xycoords=("axes fraction", "data"),
                textcoords="offset points", ha="right", va="top",
                color=INK2, fontsize=18.8)
    ax.set_ylim(-0.062, 0.45)
    _layer_axis(ax, layers)
    _stagger(ax, ends, log=False)
    ax.set_ylabel("share of the box's tiles that\nprint the object's own name")
    style.despine(ax)
    ax.plot([], [], color=MUTED, lw=2.4, label="inside the box")
    ax.plot([], [], color=MUTED, lw=1.6, alpha=0.55, ls=(0, (2, 2)),
            label="everywhere else")
    style.legend(ax, loc="upper left")
    _head(ax, f"{n_photo} photographs  \u00b7  {len(layers)} layers")

    fig.text(0.5, 0.985, "When a picture becomes readable",
             ha="center", va="top", color=INK, fontsize=23.5)
    fig.subplots_adjust(left=0.062, right=0.978, top=0.845, bottom=0.115,
                        wspace=0.40)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def _rank_axis(ax, nv, layers):
    """A rank axis: first place at the top, log-spaced, named in ordinals."""
    ax.set_yscale("log")
    ax.set_ylim(nv * 1.35, 0.78)
    ax.set_yticks([1, 3, 10, 30, 100, nv])
    ax.set_yticklabels(["1st", "3rd", "10th", "30th", "100th", f"{nv}th"])
    ax.minorticks_off()
    _layer_axis(ax, layers, room=0.26)
    ax.grid(axis="y", color=GRID, lw=0.9)
    style.despine(ax)


def _stagger(ax, ends, log=False, frac=0.24, fs=19.6):
    """End-of-line labels, pushed apart when two series finish together."""
    if not ends:
        return
    y0, y1 = ax.get_ylim()
    f = (lambda v: np.log10(max(v, 1e-9))) if log else (lambda v: v)
    span = abs(f(y1) - f(y0))
    gap = span * frac
    ys = [f(e[1]) for e in ends]
    for e in ends:
        e.append(e[1])
    idx = list(np.argsort(ys))
    for a_, b_ in zip(idx[:-1], idx[1:]):
        if ys[b_] - ys[a_] < gap:
            ys[b_] = ys[a_] + gap
    # the axis may be inverted -- a rank axis has 1st at the top -- so the two
    # limits are sorted rather than assumed, and the stack is clamped at both
    # ends: pushing labels apart without a floor is what drops the last one
    # off the bottom of the panel
    lo, hi = sorted((f(y0), f(y1)))
    lo, hi = lo + gap * 0.35, hi - gap * 0.35
    over = max(ys) - hi
    if over > 0:
        ys = [y - over for y in ys]
    under = lo - min(ys)
    if under > 0:
        ys = [y + under for y in ys]
    for e, y in zip(ends, ys):
        e[1] = (10 ** y) if log else y
    for x, y, lens, colour, y_true in ends:
        style.label_end(ax, x, y, lens, colour, y_true=y_true, fontsize=fs)


# ------------------------------------------------------------------ figure C
#
# The same measurement on two architectures that put the picture into the
# language model in different ways.  Qwen3-VL injects visual features in place
# into its first layers, so content spreads across positions; LLaVA projects the
# picture into ordinary token positions and leaves it there.  If that difference
# is real, the causal mass should stay inside the object on one and travel on
# the other, and that is a thing to be looked at rather than asserted.

def _truth_row(ax, cards, C, keep, box, gap, ramp, y0=0.0):
    """One model's truth maps, left to right, into an existing axes."""
    for j, c in enumerate(cards):
        v = np.asarray(c["truth"], dtype=float)
        q = _quant(v)
        bx = j * (C + gap)
        for i, cell in enumerate(keep):
            y, x = divmod(int(cell), C)
            ax.add_patch(Rectangle((bx + x, y0 + y), 1.0, 1.0,
                                   facecolor=_shade(ramp, q[i]),
                                   edgecolor="none", zorder=2))
        if box:
            _perimeter(ax, box, C, bx, y0)
        t = int(np.argmax(v))
        y, x = divmod(int(keep[t]), C)
        ax.add_patch(Rectangle((bx + x + .07, y0 + y + .07), .86, .86,
                               fill=False, edgecolor=HOT, lw=1.6, zorder=6))


def _box_share(card) -> float:
    """Share of the frame's total damage that the human's box accounts for.

    A share of magnitude, not of a signed sum: covering some regions raises the
    answer's logit, and a signed total can pass through zero and take the ratio
    with it.
    """
    v = np.abs(np.asarray(card["truth"], dtype=float))
    box = set(card.get("inside", []))
    keep = card["keep"]
    m = np.array([int(keep[j]) in box for j in range(len(keep))])
    return float(v[m].sum() / max(v.sum(), 1e-9))


def two_models(models, out, name="f86_arch.png", n_layers=None,
               photos=None, per_cell=1.02, gap=0.62, strip_w=12.3,
               colours=None, pick="strongest", concept=None, max_cols=6):
    """Two architectures' layerwise causal maps, and the number under them.

    ``models`` maps a display name to that model's cards grouped by
    photograph, and to the number of layers in its language model, as
    ``{name: (groups, n_layers)}``.  One photograph is drawn per model; the
    curve below is over all of that model's photographs.

    Both strips are of the same object word -- the one both models were run on
    whose strongest example is strongest -- so the two are comparable rather
    than merely adjacent.  Within that word the photograph drawn is the one on
    which covering a region costs the most,
    so that the strip is a map and not a ranking of noise.  Picking by eye
    would be picking the photograph that makes the claim; picking on the size of
    the effect keeps the rule clear of what the figure is claiming, which is
    about where the effect is.  The strips are one photograph each and the curve
    under them is the measurement.
    """
    style.use()
    names = list(models)
    # Neither colour may be a lens's.  No lens appears in this figure, but the
    # paper's four lens hues are learned once and read everywhere, and a model
    # drawn in the contribution reading's blue would be read as one.
    colours = colours or [style.MAGENTA, style.GREEN]
    # the same object word on both models, so the two strips are comparable
    if concept is None:
        have = [{c["concept"] for g in models[nm][0].values() for c in g}
                for nm in names]
        both = set.intersection(*have) if have else set()
        best = None
        for w in sorted(both):
            v = min(max(max(float(np.max(c["truth"])) for c in g)
                        for g in models[nm][0].values()
                        if g[0]["concept"] == w) for nm in names)
            if best is None or v > best[0]:
                best = (v, w)
        concept = best[1] if best else None
    rows = []
    for nm in names:
        groups, nl = models[nm]
        keys = [k for k in groups
                if concept is None or groups[k][0]["concept"] == concept]
        keys = keys or list(groups)
        if pick and len(keys) > 1:
            # the photograph on which covering a region does the most, so the
            # strip is a map rather than a ranking of noise.  Choosing on the
            # size of the effect rather than on where the effect is keeps the
            # rule clear of the claim the figure is making
            pk = [max(float(np.max(c["truth"])) for c in groups[k])
                  for k in keys]
            key = keys[int(np.argmax(pk))]
        else:
            key = keys[0]
        g = groups[key]
        # the drawn strip keeps at most ``max_cols`` evenly spaced depths so
        # its maps are large enough to read; the curve below still uses every
        # measured layer
        if max_cols and len(g) > max_cols:
            idx = sorted({int(round(t)) for t in
                          np.linspace(0, len(g) - 1, max_cols)})
            g = [g[i] for i in idx]
        rows.append(dict(name=nm, cards=g, key=key, n_layers=nl,
                         groups=groups))

    left, right, gut = 0.10, 0.18, 2.75
    photo_in = 1.95
    strip_h, unit = [], []
    for r in rows:
        C = int(r["cards"][0]["coarse"])
        L = len(r["cards"])
        wide = L * C + (L - 1) * gap
        u = strip_w / wide
        unit.append(u)
        strip_h.append(u * C)
    plot_h = 3.30
    pads = 0.92                                   # under each strip
    above = 0.58
    W = left + photo_in + gut + strip_w + right
    H = above + sum(strip_h) + 2 * pads + plot_h + 0.86
    fig = plt.figure(figsize=(W, H))

    ramp = _ramp(TRUTH_INK)
    y = H - above
    for k, r in enumerate(rows):
        C = int(r["cards"][0]["coarse"])
        L = len(r["cards"])
        wide = L * C + (L - 1) * gap
        y -= strip_h[k]
        ax = fig.add_axes([(left + photo_in + gut) / W, y / H,
                           strip_w / W, strip_h[k] / H])
        ax.set_xlim(0, wide); ax.set_ylim(C, 0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        _truth_row(ax, r["cards"], C, r["cards"][0]["keep"],
                   set(r["cards"][0].get("inside", [])), gap, ramp)
        layers = [int(c["layer"]) for c in r["cards"]]
        for j, l in enumerate(layers):
            d = 100 * l / max(r["n_layers"] - 1, 1)
            ax.annotate(f"{d:.0f}%", xy=(j * (C + gap) + C / 2, C),
                        xytext=(0, -6), textcoords="offset points",
                        ha="center", va="top", color=INK2, fontsize=18.8,
                        annotation_clip=False)
        ax.annotate("depth through the language model", xy=(wide / 2, C),
                    xytext=(0, -27), textcoords="offset points", ha="center",
                    va="top", color=INK, fontsize=20.3, annotation_clip=False)
        ax.annotate(r["name"], xy=(0, C / 2), xytext=(-22, 0),
                    textcoords="offset points", ha="right", va="center",
                    color=INK, fontsize=20, annotation_clip=False)
        # the same mark the curve below wears, so a strip and a line can be
        # read onto each other without counting rows
        ax.plot([0], [C / 2], marker="s", ms=7.0, mew=0, color=colours[k],
                clip_on=False, transform=ax.transData + _off(ax, -11, 0),
                zorder=8)
        # over two lines: on one it was wider than the gutter and printed
        # over the photograph on its left
        # dropped clear of the model's name, which is set on the same side
        ax.annotate(f"“{r['cards'][0]['concept']}”\n{C * C} regions",
                    xy=(0, C / 2), xytext=(-22, -44),
                    textcoords="offset points", ha="right", va="center",
                    color=INK2, fontsize=19.6, annotation_clip=False)
        img = (photos or {}).get(r["key"])
        if img is not None:
            ph = min(photo_in, strip_h[k])
            ap = fig.add_axes([left / W, (y + (strip_h[k] - ph) / 2) / H,
                               ph / W, ph / H])
            ap.imshow(img)
            ap.set_xticks([]); ap.set_yticks([]); ap.set_aspect("equal")
            for sp in ap.spines.values():
                sp.set_visible(False)
        y -= pads

    # ---- the number under the pictures
    #
    # The share alone will not carry the claim: LLaVA's boxes cover a quarter of
    # the frame and Qwen's a third, so a box holding a third of the damage means
    # something different on each.  What is comparable is the share against the
    # box's own area -- one is what a box would hold if damage were spread
    # evenly, and one reference line then serves both models.
    # the plot spans the whole canvas rather than only the strip's columns:
    # aligning it with them would leave the two inches under the photographs
    # empty, and it is a different chart, so nothing has to line up
    axp = fig.add_axes([(left + 1.20) / W, 0.66 / H,
                        (W - left - right - 1.36) / W, plot_h / H])
    axp.axhline(1.0, color=INK, lw=1.1, ls=(0, (2, 3)), zorder=1)
    # under its own rule: above it the words sit in the band the flatter of
    # the two models runs through
    axp.annotate("the box is worth exactly its own area", xy=(0, 1.0),
                 xytext=(2, -7), xycoords=("axes fraction", "data"),
                 textcoords="offset points", ha="left", va="top",
                 color=INK2, fontsize=18.8)
    ends, xs = [], []
    for k, r in enumerate(rows):
        cards = [c for g in r["groups"].values() for c in g]
        layers = sorted({int(c["layer"]) for c in cards})
        dep = [100 * l / max(r["n_layers"] - 1, 1) for l in layers]
        mu = []
        for l in layers:
            at = [c for c in cards if int(c["layer"]) == l]
            sh = float(np.mean([_box_share(c) for c in at]))
            ar = float(np.mean([len(set(c.get("inside", []))) /
                                (int(c["coarse"]) ** 2) for c in at]))
            mu.append(sh / max(ar, 1e-9))
        axp.plot(dep, mu, color=colours[k], lw=2.4, marker="o", ms=6, zorder=3)
        ends.append([dep[-1], mu[-1], r["name"], colours[k]])
        xs += dep
    axp.set_ylim(0, max(2.2, max(e[1] for e in ends) * 1.2))
    lo, hi = min(xs), max(xs)
    axp.set_xlim(lo - 3, hi + (hi - lo) * 0.30)
    axp.set_xticks([t for t in range(0, 101, 20) if lo - 2 <= t <= hi + 2])
    _stagger(axp, ends, log=False)
    axp.set_xlabel("depth through the language model  (%)")
    axp.set_ylabel("what the object's box is worth,\nagainst its own share of the frame")
    style.despine(axp)
    axp.grid(axis="y", color=GRID, lw=0.9)

    fig.text(0.5, 1 - 6 / (72 * H),
             "Two architectures, the same measurement", ha="center", va="top",
             color=INK, fontsize=23.5)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p
