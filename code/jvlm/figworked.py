# SPDX-License-Identifier: Apache-2.0
"""Worked examples a reader can check line by line.

Two figures, both built out of single trials rather than averages.

``editcards`` prints one real case per edit operation: the photograph, the
question, what the model said before and after, and the two lens coordinates
the edit is supposed to move.  The cases are chosen by a rule stated on the
figure -- the answer had to change *and* the edited concept's own reading had
to rise -- because a trial whose answer flips while the reading falls flipped
for some other reason and would show a reader the opposite of the claim.

``promptcase`` prints one photograph under all four framings of the
cross-modal experiment, with the model's actual replies and each lens's
reading of the dangerous word to four decimal places.  The aggregate figure
elsewhere shows two flat lines, and a flat line is easy to read as "small".
The digits are the point: two of the three lenses return the *identical* value
under a description request, a false claim and a request for harm, because a
fixed transport of the image tokens is a function of the picture alone.

Three conventions here were arrived at the hard way and are worth keeping.

* Text is placed with a cursor that walks down the axes in points, converted
  with that axes' own height in inches read back off the layout.  Guessing the
  height, or placing at hand-chosen fractions, is what printed one paragraph
  on top of another the first time round.
* Widths of strings are *measured* with the renderer, never estimated from the
  character count.  An estimate put the arrow of "yellow" -> "orange" straight
  through the word after it.
* Photographs are padded to their cell's own aspect with the page colour, so
  every picture fills its cell exactly and a row of captions stays a row.
  COCO ships everything from 1.58 to 0.79 wide, and anchoring alone leaves the
  captions at five different heights.
"""
from __future__ import annotations

import os
import textwrap

import numpy as np

from jvlm import style
from jvlm.style import BLUE, GRID, INK, INK2, MUTED, ORANGE, SURFACE, VIOLET

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#: the same map :mod:`jvlm.figcausal` fixes, so a lens keeps one colour
LENS_COLOUR = {"logit lens": ORANGE, "J-lens": VIOLET, "VJ-Lens": BLUE}

C_BEFORE = MUTED       # the unedited state, everywhere in this module
C_AFTER = BLUE         # what the edit did
PANEL = "#eef2f6"
SOFT = "#eef3fa"

OP_DOES = {
    "erase": "take the source concept's own coordinate away",
    "amplify": "scale the target concept's own coordinate up",
    "inject": "add the target direction, sized by the local norm",
    "swap": "exchange the two concepts' coordinates",
    "swapdev": "exchange the two coordinates' deviations from the baseline",
    "transplant": "copy the target's coordinate off a real photograph of it",
}
#: which coordinate each operation is actually made in -- the mark goes there
OP_ACTS_ON = {"erase": "src", "amplify": "dst", "inject": "dst",
              "swap": "both", "swapdev": "both", "transplant": "dst"}


# ------------------------------------------------------------------ furniture

def _num(v, dp=2, sign=True):
    """A number with a real minus sign, matching the tick labels elsewhere."""
    s = f"{v:+.{dp}f}" if sign else f"{v:.{dp}f}"
    return s.replace("-", "−")


def _fighead(fig, main, sub, x=0.007, fs=17.0, fss=11.3, pad=7.0, gap=9.0,
             clear=13.0):
    """Figure title and a quieter block under it; returns where plots may start.

    Measured in points against the figure's own height, so one call gives the
    same visual gap on a short figure and a tall one.  A fraction of the figure
    height does not.
    """
    unit = 1.0 / (72.0 * fig.get_figheight())
    y = 1 - pad * unit
    fig.text(x, y, main, ha="left", va="top", color=INK, fontsize=fs)
    y -= (fs * 1.24 + gap) * unit
    if sub:
        fig.text(x, y, sub, ha="left", va="top", color=INK2, fontsize=fss,
                 linespacing=1.42)
        y -= (sub.count("\n") + 1) * fss * 1.42 * unit
    return y - clear * unit


def _ax_h(ax):
    """The axes' height in inches, off the layout rather than off a guess."""
    return ax.get_position(original=True).height * ax.figure.get_figheight()


def _cell_aspect(ax):
    p = ax.get_position(original=True)
    f = ax.figure
    return (p.width * f.get_figwidth()) / (p.height * f.get_figheight())


def _tw(ax, s, fs):
    """Width of a string in axes-fraction units, measured with the renderer."""
    fig = ax.figure
    try:
        r = fig.canvas.get_renderer()
    except AttributeError:                       # pragma: no cover
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
    t = ax.text(0, 0, s, fontsize=fs, alpha=0.0)
    w = t.get_window_extent(renderer=r).width
    t.remove()
    return w / max(ax.get_window_extent(renderer=r).width, 1.0)


class _Text:
    """A top-down text cursor on a unit axes whose height in inches is known.

    Every card in a figure gets the same cursor, so the same content lands at
    the same height in all of them without a hand-tuned constant, and a line
    that grows from one wrapped row to two pushes what follows down instead of
    printing on top of it.
    """

    def __init__(self, ax, h_in=None, y=1.0):
        self.ax = ax
        self.u = 1.0 / (72.0 * (h_in if h_in else _ax_h(ax)))
        self.y = y

    def gap(self, pts):
        self.y -= pts * self.u

    def line(self, x, text, fs=10.5, colour=INK, lead=1.34, ha="left", **kw):
        self.ax.text(x, self.y, text, ha=ha, va="top", fontsize=fs,
                     color=colour, linespacing=lead, **kw)
        self.y -= (str(text).count("\n") + 1) * fs * lead * self.u
        return self.y

    def wrapped(self, x, text, width, **kw):
        return self.line(x, "\n".join(textwrap.wrap(text, width)), **kw)


def _blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def _pad_to(im, aspect, bg=SURFACE):
    """Pad a photograph out to ``aspect`` with the page colour, centred.

    Nothing is cropped and nothing is stretched; the picture simply sits in a
    box of the shape its cell has, which is what keeps a row of captions level.
    """
    from PIL import Image, ImageColor

    w, h = im.size
    W, H = (w, int(round(w / aspect))) if w / h > aspect else \
           (int(round(h * aspect)), h)
    if (W, H) == (w, h):
        return im
    out = Image.new("RGB", (max(W, w), max(H, h)), ImageColor.getrgb(bg))
    out.paste(im, ((max(W, w) - w) // 2, (max(H, h) - h) // 2))
    return out


def _photo(ax, path, caption, thumb=900, fit=True, fs=9.4):
    from PIL import Image

    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((thumb, thumb), Image.LANCZOS)
        if fit:
            im = _pad_to(im, _cell_aspect(ax))
        ax.imshow(im)
    except Exception:
        ax.set_facecolor(PANEL)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    for sp in ax.spines.values():
        sp.set_visible(False)
    if caption:
        ax.annotate(caption, xy=(0.5, 0), xytext=(0, -6),
                    xycoords="axes fraction", textcoords="offset points",
                    ha="center", va="top", color=INK2, fontsize=fs,
                    linespacing=1.3)


# ------------------------------------------------------------- the edit cards

def _bars(ax, rows, x_lab=0.168, x0=0.188, x1=0.885, bar=8.0, pitch=14.0):
    """Four thin bars: two concepts, before and after, on one shared scale.

    Both readings are in the same units, so they share an axis; each card keeps
    its own range, because the operations differ by an order of magnitude in
    how far they push and that is itself one of the findings.
    """
    _blank(ax)
    h = _ax_h(ax)
    u = 1.0 / (72.0 * h)
    vals = [v for _, _, v, _ in rows]
    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
    span = (hi - lo) or 1.0
    lo, hi = lo - 0.15 * span, hi + 0.15 * span
    fx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    zero = fx(0.0)
    top = 1.0 - 5 * u
    bot = top - len(rows) * pitch * u
    ax.plot([zero, zero], [bot, top], color="#c9c8c3", lw=0.9, zorder=1)
    for k, (label, mark, v, colour) in enumerate(rows):
        yc = top - (k + 0.5) * pitch * u
        ax.add_patch(Rectangle((min(zero, fx(v)), yc - bar * u / 2),
                               abs(fx(v) - zero), bar * u, facecolor=colour,
                               edgecolor="none", zorder=3))
        ax.text(x_lab, yc, label, ha="right", va="center", fontsize=9.4,
                color=INK if mark else INK2)
        if mark:
            ax.plot([0.004], [yc], marker="o", ms=4.6, mew=0, color=BLUE,
                    zorder=4)
        dx, ha = (0.007, "left") if v >= 0 else (-0.007, "right")
        ax.text(fx(v) + dx, yc, _num(v), ha=ha, va="center", fontsize=9.4,
                color=INK2)


def _card(fig, cell, c):
    """One operation, one trial: photograph, question, answer, readings."""
    donor = c["op"] == "transplant"
    inner = cell.subgridspec(2, 3, height_ratios=[2.10, 0.86],
                             width_ratios=[1.0, 1.0, 3.05], wspace=0.075,
                             hspace=0.190)
    if donor:
        _photo(fig.add_subplot(inner[0, 0]), c["path"],
               f"the photograph  ·  #{c['image_id']}")
        _photo(fig.add_subplot(inner[0, 1]), c["donor_path"],
               f"the donor  ·  a real {c['dst']}")
    else:
        _photo(fig.add_subplot(inner[0, 0:2]), c["path"],
               f"the photograph  ·  #{c['image_id']}")

    ax = _blank(fig.add_subplot(inner[0, 2]))
    cur = _Text(ax, y=0.995)
    cur.line(0, c["op"], fs=15.0, colour=INK)
    ax.text(1.0, 0.995, f"dose {c['dose']:g}    ‖Δh‖ / ‖h‖ = "
            f"{c['disturb']:.4f}", ha="right", va="top", fontsize=9.6,
            color=INK2)
    cur.gap(2)
    cur.wrapped(0, OP_DOES.get(c["op"], ""), 62, fs=10.0, colour=INK2)
    cur.gap(8)
    cur.line(0, "asked of the photograph", fs=9.2, colour=MUTED)
    cur.gap(1)
    cur.wrapped(0, "“" + c["question"] + "”", 55, fs=10.4, colour=INK)
    cur.gap(8)
    cur.line(0, "the model answered", fs=9.2, colour=MUTED)
    cur.gap(2)
    y = cur.y
    a, b = f"“{c['answer_before']}”", f"“{c['answer_after']}”"
    ax.text(0, y, a, ha="left", va="top", fontsize=15.0, color=INK2)
    x = _tw(ax, a, 15.0) + 0.022
    ax.text(x, y, "→", ha="left", va="top", fontsize=15.0, color=MUTED)
    x += _tw(ax, "→", 15.0) + 0.022
    ax.text(x, y, b, ha="left", va="top", fontsize=15.0, color=C_AFTER)
    cur.y = y - 15.0 * 1.28 / (72 * _ax_h(ax))
    cur.gap(2)
    cur.line(0, "the target answer's logit lead over the true one: "
             f"{_num(c['margin_before'])}  →  {_num(c['margin_after'])}",
             fs=9.6, colour=INK2)

    acts = OP_ACTS_ON.get(c["op"], "dst")
    _bars(fig.add_subplot(inner[1, 0:3]), [
        (f"“{c['dst']}” before", acts in ("dst", "both"),
         c["read_dst_before"], C_BEFORE),
        (f"“{c['dst']}” after", acts in ("dst", "both"),
         c["read_dst_after"], C_AFTER),
        (f"“{c['src']}” before", acts in ("src", "both"),
         c["read_src_before"], C_BEFORE),
        (f"“{c['src']}” after", acts in ("src", "both"),
         c["read_src_after"], C_AFTER)])


def editcards(picks, out, name="f74_editcards.png", ncol=2, card_w=8.05,
              card_h=3.24):
    """Six operations, one checkable trial each, on one sheet."""
    style.use()
    n = len(picks)
    nrow = int(np.ceil(n / ncol))
    vgap, hgap, head = 0.50, 0.55, 2.05
    fw = ncol * card_w + (ncol - 1) * hgap + 0.30
    fh = nrow * card_h + (nrow - 1) * vgap + head
    fig = plt.figure(figsize=(fw, fh))
    top = _fighead(
        fig,
        "Six edits, one trial each, printed so a reader can check them",
        "One case per operation, chosen by a rule and not by eye: the model's "
        "answer had to change AND the edited concept's own reading had to "
        "rise.  A trial whose answer changes while that\nreading falls has "
        "moved for some other reason, and there are such trials; they are "
        "left out here and counted in the dose curves instead.  A blue dot "
        "marks the coordinate the operation\nis made in — one of the two for "
        "most operations, both for the two that exchange a pair.  The bars "
        "are that lens coordinate at the image positions, grey before the "
        "edit and blue after,\neach card on its own scale.  ‖Δh‖/‖h‖ is how "
        "far the edit moved the residual stream there: every one of these "
        "six is under four per cent of it.")
    gs = fig.add_gridspec(nrow, ncol, left=0.008, right=0.995, top=top,
                          bottom=0.010, wspace=hgap / card_w,
                          hspace=vgap / card_h)
    for k, c in enumerate(picks):
        _card(fig, gs[k // ncol, k % ncol], c)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=190)
    plt.close(fig)
    return p


# --------------------------------------------------------- the prompt example

#: the four framings, in the order the aggregate figure uses
FRAMINGS = [
    ("plain", "just describe it"),
    ("control", "a harmless false claim"),
    ("substitute", "the dangerous false claim"),
    ("ask", "the false claim, then a request for harm"),
]
KEY = {"logit lens": "say", "J-lens": "jlens", "VJ-Lens": "atlas"}


def promptcase(rows, meta, out, name="f75_promptcase.png", layer=18,
               layers=(8, 13, 18, 23), danger="knife"):
    """One photograph, four framings, and the readings to four decimals."""
    style.use()
    fig = plt.figure(figsize=(16.7, 7.62))
    top = _fighead(
        fig,
        "The same picture, four different things said about it, and two "
        "lenses that cannot tell",
        "One photograph and one word — “" + danger + "” — read at the image "
        "positions under four framings of the same request.  The reply column "
        "is what the model actually said; the three\nnumber columns are the "
        f"three lenses at layer {layer}.  Read down them.  The first two "
        "repeat one value to the last digit, and the third does not.")
    gs = fig.add_gridspec(2, 2, height_ratios=[3.95, 1.70],
                          width_ratios=[1.0, 3.05], left=0.008, right=0.994,
                          top=top, bottom=0.018, wspace=0.05, hspace=0.12)
    gl = gs[0, 0].subgridspec(2, 1, height_ratios=[1.0, 0.38], hspace=0.30)

    _photo(fig.add_subplot(gl[0, 0]), meta["path"],
           f"COCO val2017  ·  #{meta['image_id']}")
    ax = _blank(fig.add_subplot(gl[1, 0]))
    cur = _Text(ax, y=0.88)
    cur.line(0, "what is in the photograph", fs=9.4, colour=MUTED)
    cur.gap(2)
    cur.wrapped(0, f"The object the framings name is the {meta['obj']}.  The "
                "annotation lists " + ", ".join(meta["present"])
                + f".  There is no {danger} in the picture and none in the "
                "annotation; the word enters only through the text.", 52,
                fs=10.2, colour=INK)

    # ---- the table
    ax = _blank(fig.add_subplot(gs[0, 1]))
    h = _ax_h(ax)
    u = 1.0 / (72.0 * h)
    XP, XR = 0.0, 0.345
    NX = {"logit lens": 0.788, "J-lens": 0.893, "VJ-Lens": 1.000}

    y = 0.995
    for nm, x in NX.items():
        ax.text(x, y, nm, ha="right", va="top", fontsize=10.5, color=INK)
        ax.plot([x - _tw(ax, nm, 10.5) - 0.016], [y - 5.4 * u], marker="o",
                ms=5.5, mew=0, color=LENS_COLOUR[nm], clip_on=False)
    ax.text(XP, y, "the framing, and the text actually sent", ha="left",
            va="top", fontsize=10.5, color=INK)
    ax.text(XR, y, "what the model replied", ha="left", va="top",
            fontsize=10.5, color=INK)
    y -= 10.5 * 1.3 * u + 6 * u
    ax.plot([0, 1], [y, y], color="#c9c8c3", lw=0.9, clip_on=False)
    y -= 10 * u

    xl, xr = NX["logit lens"] - 0.100, NX["J-lens"] + 0.006
    y_top = y
    for j, (key, label) in enumerate(FRAMINGS):
        r = rows[key]
        c1 = _Text(ax, h, y=y)
        c1.line(XP, label, fs=10.5, colour=INK)
        c1.gap(1)
        c1.wrapped(XP, "“" + r["prompt"] + "”", 46, fs=9.9, colour=INK2)
        c2 = _Text(ax, h, y=y)
        c2.line(XR, "refused" if r["refused"] else "answered", fs=10.5,
                colour=INK)
        c2.gap(1)
        said = r["text"].replace("\n", " ").strip()
        # every reply was cut at twenty-four tokens; the ones that finished a
        # sentence inside that budget are whole, the rest are not
        c2.wrapped(XR, "“" + said + ("”" if said[-1:] in ".!?" else "…”"), 45,
                   fs=9.9, colour=INK2)
        for nm, x in NX.items():
            ax.text(x, y, f"{r[f'{KEY[nm]}_{layer}']:.4f}", ha="right",
                    va="top", fontsize=12.0,
                    color=INK if nm == "VJ-Lens" else INK2)
        y = min(c1.y, c2.y) - 13 * u
        if j < len(FRAMINGS) - 1:
            ax.plot([0, 1], [y + 6 * u, y + 6 * u], color=GRID, lw=0.9)
    ax.add_patch(Rectangle((xl, y + 4 * u), xr - xl, y_top + 7 * u - y - 4 * u,
                           facecolor=SOFT, edgecolor="none", zorder=0))
    ax.text((xl + xr) / 2, y - 1 * u, "the same number,\nfour times over",
            ha="center", va="top", fontsize=9.9, color=INK2, linespacing=1.32)
    ax.text(1.0, y - 1 * u, "the only one of the three\nthat can see the "
            "question", ha="right", va="top", fontsize=9.9, color=INK2,
            linespacing=1.32)

    # ---- the footer: the same story at every depth, and why
    ax = _blank(fig.add_subplot(gs[1, :]))
    h = _ax_h(ax)
    u = 1.0 / (72.0 * h)
    W = 0.615
    y = 0.97
    ax.text(0, y, "the same photograph at every depth the lenses were fitted "
            "at, in the same units", ha="left", va="top", fontsize=10.5,
            color=INK)
    y -= 10.5 * 1.3 * u + 6 * u
    ax.plot([0, W], [y, y], color="#c9c8c3", lw=0.9)
    y -= 11 * u
    xs = [0.235 + (W - 0.255) / (len(layers) - 1) * k
          for k in range(len(layers))]
    for x, l in zip(xs, layers):
        ax.text(x, y, f"layer {l}", ha="center", va="top", fontsize=10.2,
                color=INK2)
    y -= 10.2 * 1.3 * u + 8 * u
    for nm in ("logit lens", "J-lens", "VJ-Lens"):
        ax.text(0.017, y, nm, ha="left", va="top", fontsize=10.5, color=INK)
        ax.plot([0.004], [y - 5.6 * u], marker="o", ms=5.5, mew=0,
                color=LENS_COLOUR[nm], clip_on=False)
        for x, l in zip(xs, layers):
            v = [rows[k][f"{KEY[nm]}_{l}"] for k, _ in FRAMINGS]
            same = max(v) - min(v) < 5e-5
            ax.text(x, y, f"{v[0]:.4f}" if same
                    else f"{min(v):.3f} – {max(v):.3f}", ha="center",
                    va="top", fontsize=11.4, color=INK2 if same else INK)
        y -= 11.4 * 1.34 * u + 5 * u
    ax.text(0.004, y - 3 * u, "One value per depth for the first two, because "
            "all four framings return it unchanged; a range for the third, "
            "because they do not.", ha="left", va="top", fontsize=9.8,
            color=INK2)

    cur = _Text(ax, h, y=0.97)
    cur.line(W + 0.035, "why two of the three cannot move", fs=10.5,
             colour=INK)
    cur.gap(4)
    cur.wrapped(W + 0.035,
                "The image tokens come before the text tokens and attention "
                "only looks backwards, so their vectors are the same four "
                "times over.  A logit lens reads those vectors through one "
                "fixed matrix and a Jacobian lens through another, so both "
                "are constant here by construction rather than by accident, "
                "and no amount of fitting can change it.  The third lens "
                "perturbs the tokens and re-runs the model, which is why it "
                "has anything to report at all.", 62, fs=10.2, colour=INK2)

    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=190)
    plt.close(fig)
    return p
