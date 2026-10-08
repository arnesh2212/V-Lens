# SPDX-License-Identifier: Apache-2.0
"""Worked examples a reader can check line by line.

Two figures, both built out of single trials rather than averages.

``editcards`` prints one real case per edit operation: the photograph, the
question, what the model said before and after, and the two lens coordinates
the edit is supposed to move.  The cases are chosen by a rule stated on the
figure -- the answer had to change *and* the reading of the concept the answer
moves *toward* had to rise -- because a trial whose answer flips while that
reading falls flipped for some other reason and would show a reader the
opposite of the claim.  "The edited concept" is the wrong phrase for it and was
the phrase here for one round: erase takes the *source* concept away, so the
edited concept's reading falls by construction, and the rule the driver applies
has always been ``read_dst_after > read_dst_before``.

``promptcase`` prints one photograph under all four framings of the
cross-modal experiment, with the model's actual replies and each lens's
reading of the dangerous word to four decimal places.  The aggregate figure
elsewhere shows two flat lines, and a flat line is easy to read as "small".
The digits are the point: two of the three lenses return the *identical* value
under a description request, a false claim and a request for harm, because a
fixed transport of the image tokens is a function of the picture alone.

Both figures print the trial and nothing else.  The in-figure text that IS
the data -- the prompt, the model's reply, the two answers, the readings -- is
the whole point and stays.  The explanatory header that used to sit above it,
and the paragraph that used to explain why two lenses cannot move, are in the
captions these figures are published under, in ``paper/atlas.html``.

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
#: the band behind the two fixed-transport columns.  It was a blue tint, which
#: is VJ-Lens's colour in this paper, sitting behind the two columns that are
#: not VJ-Lens.  Neutral grey says "these two go together" and nothing else.
SOFT = "#f1f0ec"

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
#: what the strength parameter *is*, per operation.  Mirrors ``OP_X`` in
#: :mod:`jvlm.figcausal`, which owns the dose axes.  A bare "dose 8" beside a
#: bare "dose 0.0125" invites a comparison the two numbers do not support:
#: one counts multiples of a coordinate removed, the other is a fraction of the
#: local norm added.  Naming the parameter on each card removes the invitation.
OP_DOSE = {"erase": "{v:g}× the coordinate removed",
           "amplify": "gain {v:g}",
           "inject": "strength {v:g}",
           "swap": "gain {v:g}",
           "swapdev": "gain {v:g}",
           "transplant": "gain {v:g} on the donor"}


# ------------------------------------------------------------------ furniture

def _num(v, dp=2, sign=True):
    """A number with a real minus sign, matching the tick labels elsewhere."""
    s = f"{v:+.{dp}f}" if sign else f"{v:.{dp}f}"
    return s.replace("-", "−")


def _fighead(fig, main, x=0.5, fs=23.5, pad=7.0, clear=13.0):
    """The figure's one-line heading; returns where the cards may start.

    Measured in points against the figure's own height, so one call gives the
    same visual gap on a short figure and a tall one.  A fraction of the figure
    height does not.

    This drew a six-line block under the heading until the reader asked for the
    prose to come out of the pictures.  Everything it said is in the caption.
    """
    unit = 1.0 / (72.0 * fig.get_figheight())
    if not style.FIGURE_TITLES:
        return 1 - 6.0 * unit          # the caption names the figure
    fig.text(x, 1 - pad * unit, main, ha="center" if x == 0.5 else "left",
             va="top", color=INK, fontsize=fs)
    return 1 - (pad + fs * 1.28 + clear) * unit


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


def _wrap_to(ax, text, fs, maxw):
    """Break ``text`` so that every line measures at most ``maxw``."""
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if cur and _tw(ax, trial, fs) > maxw:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines or [""]


def _fit_fs(ax, s, fs, maxw, floor=12.0):
    """Largest size at or below ``fs`` at which ``s`` fits on one line."""
    while fs > floor and _tw(ax, s, fs) > maxw:
        fs -= 0.4
    return round(fs, 1)


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

    def line(self, x, text, fs=18.8, colour=INK, lead=1.34, ha="left", **kw):
        self.ax.text(x, self.y, text, ha=ha, va="top", fontsize=fs,
                     color=colour, linespacing=lead, **kw)
        self.y -= (str(text).count("\n") + 1) * fs * lead * self.u
        return self.y

    def wrapped(self, x, text, width, **kw):
        return self.line(x, "\n".join(textwrap.wrap(text, width)), **kw)

    def fitted(self, x, text, maxw, fs=18.8, **kw):
        """Wrapped to a measured width rather than a character count.

        Palatino is proportional, so "will it fit" and "how many characters"
        are different questions: a forty-character wrap fits one column and
        overruns the next one along.  Every case where prose ran through the
        table beside it came from answering the second question.
        """
        return self.line(x, "\n".join(_wrap_to(self.ax, text, fs, maxw)),
                         fs=fs, **kw)

    def balanced(self, x, text, width, **kw):
        """Wrapped, but a line that must break is broken near its middle.

        Wrapping a forty-eight character gloss at forty-six leaves the last
        word alone on a line of its own, which looks like a mistake.  When the
        text has to run to two lines, split it into two of a similar length.
        """
        w = width if len(text) <= width else max(22, (len(text) + 1) // 2 + 4)
        return self.wrapped(x, text, w, **kw)


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


def _photo(ax, path, caption, thumb=900, fit=True, fs=18.1):
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
        # on the transplant card two photographs sit side by side, each about a
        # fifth of the card wide, and two captions set at the shared size ran
        # into each other in the gap between them
        ax.annotate(caption, xy=(0.5, 0), xytext=(0, -6),
                    xycoords="axes fraction", textcoords="offset points",
                    ha="center", va="top", color=INK2,
                    fontsize=_fit_fs(ax, caption, fs, 0.98, floor=10.0),
                    linespacing=1.3)


# ------------------------------------------------------------- the edit cards

def _bars(ax, rows, x_lab=0.160, x_mark=0.172, x0=0.190, x1=0.885,
          bar=9.0, pitch=16.5):
    """Four thin bars: two concepts, before and after, on one shared scale.

    Both readings are in the same units, so they share an axis; each card keeps
    its own range, because the operations differ by an order of magnitude in
    how far they push and that is itself one of the findings.
    """
    _blank(ax)
    h = _ax_h(ax)
    u = 1.0 / (72.0 * h)
    # A 16.5pt pitch cannot hold an 18.1pt label: one line of it is 24pt tall,
    # so every row's name overlapped the row below it by construction.  Spread
    # the rows over the height the axes actually has, then set the labels to
    # the pitch that leaves.
    pitch = max(pitch, (72.0 * h - 5.0) / max(len(rows), 1))
    fs_lab = min(18.1, pitch / 1.34 * 0.97)
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
        ax.text(x_lab, yc, label, ha="right", va="center", fontsize=fs_lab,
                color=INK if mark else INK2)
        if mark:
            # beside the row it belongs to, in the gap between the name and the
            # zero line.  Parked at the card's left edge it read as a stray
            # column of bullets sixty points from anything it could mark.
            ax.plot([x_mark], [yc], marker="o", ms=4.6, mew=0, color=BLUE,
                    zorder=4)
        dx, ha = (0.007, "left") if v >= 0 else (-0.007, "right")
        ax.text(fx(v) + dx, yc, _num(v), ha=ha, va="center", fontsize=fs_lab,
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

    # How tall this column comes out depends on how many lines the question,
    # the gloss and the margin line wrap to, and that differs card by card.
    # Drawn at fixed sizes the longer cards ran off the bottom of the column
    # and printed across the bar chart underneath.  So the sizes are scaled
    # down, together, until the column fits the height it actually has.
    u_c = 1.0 / (72.0 * _ax_h(ax))

    def _col_need(k):
        gloss = OP_DOES.get(c["op"], "")
        q = "“" + c["question"] + "”"
        marg = ("the target answer's logit lead over the true one: "
                f"{_num(c['margin_before'])}  →  {_num(c['margin_after'])}")
        n_g = len(_wrap_to(ax, gloss, 18.8 * k, 0.99))
        n_q = len(_wrap_to(ax, q, 19.6 * k, 0.99))
        n_m = len(_wrap_to(ax, marg, 18.8 * k, 0.99))
        pts = (21.2 * k * 1.34 + 2 + n_g * 18.8 * k * 1.34 + 5
               + 18.1 * k * 1.34 + 1 + n_q * 19.6 * k * 1.34 + 6
               + 18.1 * k * 1.34 + 2 + 17.0 * k * 1.28 + 2
               + n_m * 18.8 * k * 1.34)
        return pts * u_c

    K = 1.0
    while K > 0.60 and _col_need(K) > 0.985:
        K -= 0.02

    cur = _Text(ax, y=0.995)
    # The operation name is left-aligned and its dose right-aligned on the same
    # line.  "erase" is short and "8× the coordinate removed  ‖Δh‖/‖h‖ = 3.9%"
    # is not, so on the widest cards the second ran back through the first.
    # Measure what the name leaves, set the dose to fit it, and if even the
    # smallest size will not fit, give the dose its own line.
    op_fs = 21.2 * K
    dose = (OP_DOSE.get(c["op"], "dose {v:g}").format(v=c["dose"])
            + f"    ‖Δh‖ / ‖h‖ = {c['disturb']:.1%}")
    room = 1.0 - _tw(ax, c["op"], op_fs) - 0.030
    d_fs = _fit_fs(ax, dose, 18.1 * K, room, floor=12.0)
    cur.line(0, c["op"], fs=op_fs, colour=INK)
    if _tw(ax, dose, d_fs) <= room:
        ax.text(1.0, 0.995, dose, ha="right", va="top", fontsize=d_fs,
                color=INK2)
    else:
        cur.gap(1)
        cur.line(0, dose, fs=18.1 * K, colour=INK2)
    cur.gap(2)
    cur.fitted(0, OP_DOES.get(c["op"], ""), 0.99, fs=18.8 * K, colour=INK2)
    cur.gap(5)
    cur.line(0, "asked of the photograph", fs=18.1 * K, colour=MUTED)
    cur.gap(1)
    cur.fitted(0, "“" + c["question"] + "”", 0.99, fs=19.6 * K, colour=INK)
    cur.gap(6)
    y_ans = cur.y
    cur.line(0, "the model answered", fs=18.1 * K, colour=MUTED)
    # how rare this card is.  Six cards side by side read as six equals; they
    # are not -- erase passes the stated rule twice in thirteen trials.  It
    # goes on this row because it is the one row that is short and
    # left-aligned on all six cards, so nothing can grow into it.
    if c.get("pass_m"):
        rule = f"{c['pass_n']} of {c['pass_m']} trials pass the rule"
        r_room = 1.0 - _tw(ax, "the model answered", 18.1 * K) - 0.030
        ax.text(1.0, y_ans, rule, ha="right", va="top",
                fontsize=_fit_fs(ax, rule, 18.1 * K, r_room, floor=11.5),
                color=MUTED)
    cur.gap(2)
    y = cur.y
    a, b = f"“{c['answer_before']}”", f"“{c['answer_after']}”"
    # the arrow was advanced by the width of the text at 17pt while the text
    # was set at 21.2, so it landed inside the word before it
    ax.text(0, y, a, ha="left", va="top", fontsize=21.2 * K, color=INK2)
    x = _tw(ax, a, 21.2 * K) + 0.022
    ax.text(x, y, "→", ha="left", va="top", fontsize=21.2 * K, color=MUTED)
    x += _tw(ax, "→", 21.2 * K) + 0.022
    ax.text(x, y, b, ha="left", va="top", fontsize=21.2 * K, color=C_AFTER)
    cur.y = y - 17.0 * K * 1.28 / (72 * _ax_h(ax))
    cur.gap(2)
    # this line is the longest on the card and used to run off the right edge
    # and across the neighbouring card's bars
    cur.fitted(0, "the target answer's logit lead over the true one: "
               f"{_num(c['margin_before'])}  →  {_num(c['margin_after'])}",
               0.99, fs=18.8 * K, colour=INK2)

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


def editcards(picks, out, name="f74_editcards.png", ncol=2, card_w=9.3,
              card_h=3.95):
    """Six operations, one checkable trial each, on one sheet."""
    style.use()
    n = len(picks)
    nrow = int(np.ceil(n / ncol))
    vgap, hgap, head = 0.55, 0.60, 0.72
    fw = ncol * card_w + (ncol - 1) * hgap + 0.30
    fh = nrow * card_h + (nrow - 1) * vgap + head
    fig = plt.figure(figsize=(fw, fh))
    # the selection rule, what the blue dot marks, what the bars are and what
    # ‖Δh‖/‖h‖ means are in this figure's caption in paper/atlas.html.  The
    # cards themselves are the data and are untouched.
    top = _fighead(
        fig, "Six edits, one trial each, printed so a reader can check them")
    gs = fig.add_gridspec(nrow, ncol, left=0.008, right=0.995, top=top,
                          bottom=0.010, wspace=hgap / card_w,
                          hspace=vgap / card_h)
    for k, c in enumerate(picks):
        _card(fig, gs[k // ncol, k % ncol], c)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    # the default pad leaves eight pixels between the ink and the frame, which
    # reads as a clipped figure however carefully nothing is actually clipped
    fig.savefig(p, dpi=190, bbox_inches="tight", pad_inches=0.17)
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
    fig = plt.figure(figsize=(18.0, 7.9))
    # what this figure is measuring, which lens is in which column, that the
    # reply column is the model's own words, and why a fixed transport cannot
    # move: all of it is in this figure's caption in paper/atlas.html.  What is
    # left in the picture is the trial itself, which is the data.
    top = _fighead(
        fig, "The same picture, four different things said about it, and two "
        "lenses that cannot tell")
    gs = fig.add_gridspec(2, 2, height_ratios=[3.95, 1.52],
                          width_ratios=[1.0, 3.05], left=0.008, right=0.994,
                          top=top, bottom=0.018, wspace=0.05, hspace=0.12)
    gl = gs[0, 0].subgridspec(2, 1, height_ratios=[1.0, 0.38], hspace=0.30)

    _photo(fig.add_subplot(gl[0, 0]), meta["path"],
           f"COCO val2017  ·  #{meta['image_id']}")
    ax = _blank(fig.add_subplot(gl[1, 0]))
    cur = _Text(ax, y=0.88)
    cur.line(0, "what is in the photograph", fs=18.1, colour=MUTED)
    cur.gap(2)
    # the same measure-then-set rule as the table: this block grows with the
    # length of the COCO annotation list, and set at a fixed size it ran out of
    # the bottom of its own panel and into the depth table's heading
    gloss = (f"The object the framings name is the {meta['obj']}.  COCO "
             "annotates this photograph as " + ", ".join(meta["present"])
             + f".  No {danger} is visible in it.")
    gu = 1.0 / (72.0 * _ax_h(ax))
    fs_g = 18.8
    while fs_g > 11.0 and (
            len(_wrap_to(ax, gloss, fs_g, 0.97)) * fs_g * 1.34 * gu) > 0.62:
        fs_g -= 0.4
    cur.fitted(0, gloss, 0.97, fs=fs_g, colour=INK)

    # ---- the table
    ax = _blank(fig.add_subplot(gs[0, 1]))
    h = _ax_h(ax)
    u = 1.0 / (72.0 * h)
    XP, XR = 0.0, 0.345
    NX = {"logit lens": 0.788, "J-lens": 0.893, "VJ-Lens": 1.000}

    xl_head = NX["logit lens"] - 0.100
    y = 0.995
    for nm, x in NX.items():
        ax.text(x, y, nm, ha="right", va="top", fontsize=18.8, color=INK)
        ax.plot([x - _tw(ax, nm, 13.0) - 0.016], [y - 6.2 * u], marker="o",
                ms=5.5, mew=0, color=LENS_COLOUR[nm], clip_on=False)
    # the headings are sized against the columns they head.  Set flat at 18.8
    # the first ran a third of the way into the reply column.
    hp = _fit_fs(ax, "the framing, and the text actually sent", 18.8,
                 XR - XP - 0.012)
    hr = _fit_fs(ax, "what the model replied", 18.8, xl_head - XR - 0.012)
    ax.text(XP, y, "the framing, and the text actually sent", ha="left",
            va="top", fontsize=hp, color=INK)
    ax.text(XR, y, "what the model replied", ha="left", va="top",
            fontsize=hr, color=INK)
    y -= 13.0 * 1.3 * u + 6 * u
    ax.plot([0, 1], [y, y], color="#c9c8c3", lw=0.9, clip_on=False)
    y -= 10 * u

    xl, xr = NX["logit lens"] - 0.100, NX["J-lens"] + 0.006

    # How tall the four rows come out depends on how many lines each prompt and
    # each reply wrap to, which depends on the size they are set at.  Drawn at
    # a fixed size the last row ran off the bottom of the axes and printed on
    # top of the depth table below, because text is not clipped to its axes.
    # So the size is chosen first, by measuring, and the rows always fit.
    def _bottom(fs_lab, fs_body):
        yy = y
        last = yy
        for key, _lab in FRAMINGS:
            r = rows[key]
            said = r["text"].replace("\n", " ").strip()
            reply = "“" + said + ("”" if said[-1:] in ".!?" else "…”")
            n1 = len(_wrap_to(ax, "“" + r["prompt"] + "”", fs_body,
                              XR - XP - 0.012))
            n2 = len(_wrap_to(ax, reply, fs_body, xl - XR - 0.012))
            head = fs_lab * 1.34 * u + 1 * u
            col = head + max(n1, n2) * fs_body * 1.34 * u
            last = yy
            yy -= col + 13 * u
        y_bot = last - (13.5 * 1.30 + 7) * u
        return y_bot - 8 * u - 4 * fs_lab * 1.32 * u   # the two notes, stacked

    FS_LAB, FS_BODY = 18.8, 18.1
    while FS_LAB > 11.5 and _bottom(FS_LAB, FS_BODY) < 0.015:
        FS_LAB, FS_BODY = FS_LAB - 0.4, FS_BODY - 0.4

    y_top = y
    for j, (key, label) in enumerate(FRAMINGS):
        r = rows[key]
        c1 = _Text(ax, h, y=y)
        c1.line(XP, label, fs=FS_LAB, colour=INK)
        c1.gap(1)
        c1.fitted(XP, "“" + r["prompt"] + "”", XR - XP - 0.012,
                  fs=FS_BODY, colour=INK2)
        c2 = _Text(ax, h, y=y)
        c2.line(XR, "refused" if r["refused"] else "answered", fs=FS_LAB,
                colour=INK)
        c2.gap(1)
        said = r["text"].replace("\n", " ").strip()
        # every reply was cut at twenty-four tokens; the ones that finished a
        # sentence inside that budget are whole, the rest are not
        c2.fitted(XR, "“" + said + ("”" if said[-1:] in ".!?" else "…”"),
                  xl - XR - 0.012, fs=FS_BODY, colour=INK2)
        for nm, x in NX.items():
            ax.text(x, y, f"{r[f'{KEY[nm]}_{layer}']:.2f}", ha="right",
                    va="top", fontsize=19.6,
                    color=INK if nm == "VJ-Lens" else INK2)
        y_num = y
        y = min(c1.y, c2.y) - 13 * u
        if j < len(FRAMINGS) - 1:
            ax.plot([0, 1], [y + 6 * u, y + 6 * u], color=GRID, lw=0.9)
    # The band ends under the last NUMBER, not under the last row.  The last
    # row's prompt runs to three lines and nothing at all is printed under the
    # numbers there, so a band drawn to the row's foot finished as an empty
    # box a hundred and thirty pixels deep.  The two notes then sit below it,
    # level with each other; placed inside, the band's right edge cut through
    # the middle of the right-hand one.
    y_bot = y_num - (13.5 * 1.30 + 7) * u
    ax.add_patch(Rectangle((xl, y_bot), xr - xl, y_top + 7 * u - y_bot,
                           facecolor=SOFT, edgecolor="none", zorder=0))
    # The two notes are stacked, not set side by side.  One is centred under
    # the grey band and the other right-aligned on the VJ-Lens column; those
    # two spans are only a tenth of the width apart, so on one line the tail of
    # the first ran into the head of the second.
    yn = y_bot - 8 * u
    ax.text((xl + xr) / 2, yn, "the same number, four times over",
            ha="center", va="top", fontsize=FS_LAB, color=INK2)
    ax.text(1.0, yn - FS_LAB * 1.42 * u, "the only lens that can see the "
            "question", ha="right", va="top", fontsize=FS_LAB, color=INK2)

    # ---- the footer: the same story at every depth.  The "why two of the
    # three cannot move" paragraph that used to stand to the right of this
    # table is in the caption; the table takes the width back.
    ax = _blank(fig.add_subplot(gs[1, :]))
    h = _ax_h(ax)
    u = 1.0 / (72.0 * h)
    W = 1.0
    y = 0.97
    ax.text(0, y, "the same photograph at every depth the lenses were fitted "
            "at", ha="left", va="top", fontsize=19.6, color=INK)
    y -= 13.5 * 1.3 * u + 6 * u
    ax.plot([0, W], [y, y], color="#c9c8c3", lw=0.9)
    y -= 11 * u
    xs = [0.300 + (W - 0.360) / (len(layers) - 1) * k
          for k in range(len(layers))]
    for x, l in zip(xs, layers):
        ax.text(x, y, f"layer {l}", ha="center", va="top", fontsize=18.8,
                color=INK2)
    y -= 13.0 * 1.3 * u + 8 * u
    for nm in ("logit lens", "J-lens", "VJ-Lens"):
        ax.text(0.017, y, nm, ha="left", va="top", fontsize=18.8, color=INK)
        ax.plot([0.004], [y - 6.2 * u], marker="o", ms=5.5, mew=0,
                color=LENS_COLOUR[nm], clip_on=False)
        for x, l in zip(xs, layers):
            v = [rows[k][f"{KEY[nm]}_{l}"] for k, _ in FRAMINGS]
            same = max(v) - min(v) < 5e-5
            ax.text(x, y, f"{v[0]:.2f}" if same
                    else f"{min(v):.2f} – {max(v):.2f}", ha="center",
                    va="top", fontsize=19.6, color=INK2 if same else INK)
        y -= 13.5 * 1.34 * u + 5 * u
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=190, bbox_inches="tight", pad_inches=0.17)
    plt.close(fig)
    return p
