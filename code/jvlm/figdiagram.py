# SPDX-License-Identifier: Apache-2.0
"""The two schematics: where the reading happens, and which space the edit moves.

Drawn rather than written as SVG so they can be rendered and looked at before
they ship, which is the only way the no-overlapping-text rule is enforceable.
"""
from __future__ import annotations

import os

import numpy as np

from jvlm import style
from jvlm.style import BLUE, GRID, INK, INK2, MUTED, ORANGE

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

SOFT = "#dde9f8"
PANEL = "#eef2f6"


def _box(ax, x, y, w, h, fc=PANEL, ec="#c9d3dd", lw=1.0, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw))


def _arrow(ax, p0, p1, colour=MUTED, lw=1.4, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 color=colour, linewidth=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def forward_pass(out, name="f70_forward.png"):
    """The forward pass, the region, and the one backward pass that reads it.

    Laid out in strict horizontal bands -- arrow, strip, labels, bracket, boxes
    -- because the first version put the backward arrow through the middle of a
    caption.
    """
    style.use()
    fig, ax = plt.subplots(figsize=(13.6, 4.62))
    ax.set_xlim(0, 120); ax.set_ylim(0, 50); ax.axis("off")
    # the axes are placed before anything is drawn on them, because the
    # backward arrow's crown has to be measured through transData and that
    # only answers correctly once the axes have their final shape
    fig.subplots_adjust(left=0.01, right=0.99, top=0.928, bottom=0.02)

    Y_STRIP = 27.0          # the token squares
    H = 3.1
    # the bands are set from the type they hold: at the paper's size the
    # caption under the strip reached the bracket drawn beneath it
    Y_LABEL = 25.2          # what each strip is
    Y_BRACKET = 20.4        # the region bracket
    Y_REGION = 18.6         # its name
    Y_ARC = 40.0            # the backward arrow's label
    Y_BOX = 5.0             # read here / written here

    # the photograph
    _box(ax, 3, Y_STRIP - 4.0, 13, 11, fc="#e8eef4")
    ax.add_patch(plt.Circle((7.2, Y_STRIP + 4.4), 1.25, color="#c9d3dd"))
    ax.add_patch(Polygon([[4.4, Y_STRIP - 2.8], [8.6, Y_STRIP + 2.4],
                          [11.4, Y_STRIP - 0.8], [13.2, Y_STRIP + 1.0],
                          [15.0, Y_STRIP - 2.8]], color="#c9d3dd"))
    ax.text(9.5, 21.4, "the photograph", ha="center", va="top",
            fontsize=19.6, color=INK2)

    # image tokens; three of them are the region
    x0, tw, gap = 21.0, 3.1, 0.55
    for i in range(10):
        hot = 3 <= i <= 5
        _box(ax, x0 + i * (tw + gap), Y_STRIP, tw, H,
             fc=SOFT if hot else PANEL, ec=BLUE if hot else "#c9d3dd",
             lw=1.3 if hot else 1.0)
    xim1 = x0 + 10 * (tw + gap) - gap
    xr0 = x0 + 3 * (tw + gap)
    xr1 = x0 + 6 * (tw + gap) - gap
    ax.text((x0 + xim1) / 2, Y_LABEL, "image tokens, one per patch",
            ha="center", va="top", fontsize=19.6, color=INK2)
    ax.plot([xr0, xr0, xr1, xr1],
            [Y_BRACKET, Y_BRACKET - 0.9, Y_BRACKET - 0.9, Y_BRACKET],
            color=BLUE, lw=1.3)
    ax.text((xr0 + xr1) / 2, Y_REGION, "region R", ha="center", va="top",
            fontsize=20.3, color=BLUE)

    # text tokens
    x1 = xim1 + 3.6
    for i in range(5):
        _box(ax, x1 + i * (tw + gap), Y_STRIP, tw, H)
    xtx1 = x1 + 5 * (tw + gap) - gap
    ax.text((x1 + xtx1) / 2, Y_LABEL, "the question", ha="center", va="top",
            fontsize=19.6, color=INK2)

    # the answer
    x2 = xtx1 + 4.4
    _box(ax, x2, Y_STRIP - 0.6, 16, H + 1.2)
    ax.text(x2 + 8, Y_STRIP + H / 2, "answer", ha="center", va="center",
            fontsize=20.3, color=INK)
    ax.text(x2 + 8, Y_LABEL, "logit of word $t$", ha="center", va="top",
            fontsize=19.6, color=INK2)

    # forward
    ym = Y_STRIP + H / 2
    for aa, bb in ((16.4, 20.4), (xim1 + 0.6, x1 - 0.5),
                   (xtx1 + 0.6, x2 - 0.5)):
        _arrow(ax, (aa, ym), (bb, ym))

    # the one backward pass.  Its label goes ABOVE the arc's own crown, which
    # is computed rather than guessed: a fixed height put the curve straight
    # through the middle of the words.  matplotlib's arc3 is a quadratic Bezier
    # whose control point sits at the chord's midpoint offset by rad, so the
    # crown is a quarter of that offset above the chord.
    p0 = (x2 + 3, Y_STRIP + H + 1.2)
    p1 = ((xr0 + xr1) / 2, Y_STRIP + H + 1.0)
    rad = 0.30
    _arrow(ax, p0, p1, colour=BLUE, lw=1.9, rad=rad)
    # FancyArrowPatch builds its curve in DISPLAY coordinates, so rad is a
    # fraction of the chord in pixels and the bulge is taller in data units
    # than in x units whenever the axes are wider than they are tall.  Measure
    # it there and bring the answer back.
    (ax0, ay0), (ax1, ay1) = ax.transData.transform([p0, p1])
    cy_px = (ay0 + ay1) / 2 - rad * (ax1 - ax0)
    crown_px = (ay0 + ay1 + 2 * cy_px) / 4
    crown = ax.transData.inverted().transform(
        ((ax0 + ax1) / 2, crown_px))[1]
    ax.text((x2 + xr1) / 2, max(Y_ARC, crown + 1.1),
            "one backward pass: how word $t$'s logit moves with this patch",
            ha="center", va="bottom", fontsize=20.3, color=BLUE)

    # read here / written here
    _box(ax, 20, Y_BOX, 44, 8.6)
    ax.text(42, Y_BOX + 5.8, "read here", ha="center", va="center",
            fontsize=21, color=INK)
    ax.text(42, Y_BOX + 2.6, "a score for every word, at this region",
            ha="center", va="center", fontsize=18.8, color=INK2)
    _box(ax, 72, Y_BOX, 44, 8.6)
    ax.text(94, Y_BOX + 5.8, "and written here", ha="center", va="center",
            fontsize=21, color=INK)
    ax.text(94, Y_BOX + 2.6, "the same vectors, edited in place", ha="center",
            va="center", fontsize=18.8, color=INK2)
    ax.plot([64.6, 71.4], [Y_BOX + 4.3, Y_BOX + 4.3], color="#c9d3dd", lw=1.2)
    ax.plot([(xr0 + xr1) / 2, (xr0 + xr1) / 2], [Y_REGION - 3.4, Y_BOX + 8.6],
            color="#c9d3dd", lw=1.2)

    style.title(ax, "Where the reading happens")
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def edit_space(out, name="f71_space.png"):
    """The subspace an edit moves, and the five things an edit can be."""
    style.use()
    # wider than it was: the right-hand column's longest line needs the room,
    # and so does the note that used to run across the rule down the middle
    fig, ax = plt.subplots(figsize=(16.0, 4.66))
    ax.set_xlim(0, 136); ax.set_ylim(0, 50); ax.axis("off")

    # the span, as a plane seen edge-on
    plane = np.array([[6, 20], [34, 30], [58, 22], [30, 12]])
    ax.add_patch(Polygon(plane, closed=True, facecolor=SOFT, edgecolor=BLUE,
                         alpha=0.85, linewidth=1.1))
    ax.text(6, 8.4, "the span of the lens vectors at this region",
            ha="left", va="center", fontsize=20.3, color=BLUE)

    o = np.array([26.0, 20.5])
    _arrow(ax, o, (42.5, 25.0), colour=BLUE, lw=1.8)
    ax.text(43.6, 25.4, r"$\hat{v}_{\rm knife}$", fontsize=20.3, color=BLUE,
            va="center")
    _arrow(ax, o, (17.0, 15.6), colour=BLUE, lw=1.8)
    ax.text(15.6, 14.2, r"$\hat{v}_{\rm banana}$", fontsize=20.3, color=BLUE,
            ha="right", va="center")

    # the activation and its two parts
    tip = np.array([33.0, 42.0])
    _arrow(ax, o, tip, colour=INK2, lw=2.2)
    ax.text(33.8, 43.2, r"$h_\ell[p]$", fontsize=21.8, color=INK, va="bottom")
    ax.text(33.8, 40.2, "what this patch holds", fontsize=18.8, color=MUTED,
            va="top")
    par = np.array([37.0, 23.5])
    _arrow(ax, o, par, colour=ORANGE, lw=2.0)
    ax.plot([par[0], tip[0]], [par[1], tip[1]], color=MUTED, lw=1.1,
            linestyle=(0, (4, 3)))
    # under the plane, not across it: beside its own arrow the name is wider
    # than the wedge it sat in and the plane's own edge ran through it, so it
    # stands clear below with a hairline back to the arrow it names
    ax.plot([44.6, 38.6], [15.4, 22.4], color=ORANGE, lw=0.9, alpha=0.55,
            ls=(0, (1, 2)), zorder=2)
    ax.text(44.0, 15.0, "the part we can name", fontsize=18.8, color=ORANGE,
            va="top")
    ax.text(38.5, 33.5, "everything else, untouched", fontsize=18.8,
            color=MUTED, va="center")

    # the five edits
    ax.plot([70, 70], [7, 45], color=GRID, lw=1.2)
    ax.text(74, 44.4, "every edit moves only the named part", fontsize=21.8,
            color=INK, va="top")
    rows = [("erase", "drop the coordinate to zero"),
            ("amplify", r"scale it by $\alpha$"),
            ("inject", "write in a word that is not there"),
            ("swap", "exchange two words' coordinates"),
            ("transplant", "take a coordinate from another photograph")]
    for k, (nm, txt) in enumerate(rows):
        y = 37.0 - k * 6.2
        ax.text(74, y, nm, fontsize=20.3, color=INK2, va="center")
        ax.text(88, y, txt, fontsize=18.8, color=MUTED, va="center")

    style.title(ax, "Which space is altered")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.925, bottom=0.02)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def gallery(cards, lenses, out, name="f72_gallery.png", layer=None,
            per_row=6, fs=15.9):
    """Many photographs at once: the word each lens reads inside the object's box.

    A breadth figure.  One photograph tells a reader what a lens does; a page of
    them tells them how often.  Each column is a photograph -- the picture, then
    one row per lens carrying the word that lens scores highest inside the
    object's own box, tinted when it is the right word.
    """
    import base64
    import io

    from PIL import Image

    style.use()
    cards = list(cards)
    n = min(len(cards), per_row * 2)
    cards = cards[:n]
    ncol = min(per_row, n)
    nrow = int(np.ceil(n / ncol))
    # a blank row closes each band: without it the second band's photograph
    # titles sit directly under the first band's last row of words and read as
    # if they belonged to it.  One uniform hspace cannot make the gap between
    # two bands larger than the gap inside one.
    gap = 1 if nrow > 1 else 0
    rows = len(lenses) + 1 + gap

    fig_h = nrow * (2.05 + 0.42 * len(lenses)) + 0.30 * gap * max(nrow - 1, 0) + 1.5
    # the closing row of the LAST band is a strip of white at the foot of the
    # figure that nothing sits in, so it is flattened to nothing
    ratios = []
    for b in range(nrow):
        ratios += [2.6] + [0.62] * len(lenses)
        ratios += [0.55 if b < nrow - 1 else 0.01] * gap
    fig, axes = plt.subplots(nrow * rows, ncol,
                             figsize=(2.1 * ncol, fig_h),
                             gridspec_kw={"height_ratios": ratios})
    axes = np.atleast_2d(axes)
    if gap:
        for b in range(nrow):
            for cc in range(ncol):
                axes[b * rows + rows - 1, cc].axis("off")

    for k, c in enumerate(cards):
        band, col = divmod(k, ncol)
        C = int(c["coarse"])
        inside = set(c.get("inside", []))
        # the picture
        ax = axes[band * rows, col]
        try:
            raw = c["img"].split(",", 1)[1]
            im = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
            ax.imshow(im)
        except Exception:
            ax.set_facecolor(PANEL)
        ax.set_xticks([]); ax.set_yticks([])
        # imshow shrinks the axes box to the picture's own aspect and centres
        # it, which puts the title of a wide photograph below the title of a
        # tall one; anchored north, a row of names is one line
        ax.set_anchor("N")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(c["concept"], fontsize=19.6, color=INK, pad=4)

        for j, lens in enumerate(lenses):
            ax = axes[band * rows + 1 + j, col]
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            d = c["lenses"].get(lens)
            if not d:
                continue
            key = str(layer) if layer is not None and str(layer) in d \
                else sorted(d, key=lambda x: int(x))[-1]
            words = d[key]["words"]
            # the word this lens reads most often inside the object's box
            from collections import Counter
            got = Counter(words[t] for t in inside if t < len(words))
            w = got.most_common(1)[0][0] if got else "·"
            right = w == c["concept"]
            ax.add_patch(Rectangle((0.02, 0.10), 0.96, 0.80,
                                   facecolor=SOFT if right else PANEL,
                                   edgecolor=BLUE if right else "#dfe4ea",
                                   linewidth=1.2 if right else 0.8))
            ax.text(0.5, 0.5, w if len(w) <= 10 else w[:9] + "\u2026",
                    ha="center", va="center", fontsize=fs + 0.5,
                    color=INK if right else INK2)
            if col == 0:
                ax.text(-0.06, 0.5, lens, ha="right", va="center",
                        fontsize=18.8, color=INK2, transform=ax.transAxes)

    for k in range(len(cards), nrow * ncol):
        band, col = divmod(k, ncol)
        for j in range(rows):
            axes[band * rows + j, col].axis("off")

    # the header owns the top strip of the canvas and nothing else is drawn in
    # it; the two lines are placed from the top down, in figure fractions, so
    # they cannot land on each other however many rows there are
    head = 0.50 / fig_h                      # one line, in figure fractions
    fig.text(0.5, 1 - head * 0.9, "What each lens calls the object, over "
             "many photographs", ha="center", va="top", fontsize=25, color=INK)
    fig.subplots_adjust(left=0.115, right=0.995, top=1 - head * 2.3,
                        bottom=0.012, hspace=0.26, wspace=0.10)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p
