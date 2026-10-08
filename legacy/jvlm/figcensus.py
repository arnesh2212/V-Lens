# SPDX-License-Identifier: Apache-2.0
"""The hallucination funding census: one wrong answer, one dot, and a price.

Because a reading is in logits and the readings over the whole picture sum to
what replacing the picture does to a token's logit, a wrong answer carries a
signed number -- how much the photograph itself contributed to the answer the
model gave.  Positive and the image paid for the error; negative and the
picture pushed the other way and was overruled.  The scatter is that number
against the model's own yes-over-no margin, and the two worked rows spend the
number over the frame so a reader can see which regions paid.

Two conventions, both borrowed from :mod:`jvlm.figcausal`:

* tiles are square and carry their value, because a translucent wash over a
  photograph can show that two maps differ but never what differs;
* a panel carries a heading and nothing else -- what the panel means belongs in
  the caption in ``paper/atlas.html``.

The ramp here is the one place this module departs from that file.  A
contribution has a sign, so the ramp diverges about zero: blue for a region
that funded the answer the model gave, red for one that argued against it, and
the page colour where a region did neither.  A single-hue ramp would print
"pushed hardest against" and "did nothing" in the same pale shade.
"""
from __future__ import annotations

import os

import numpy as np

from jvlm import style
from jvlm.style import BLUE, GRID, INK, INK2, MUTED, RED, SURFACE

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from jvlm.figworked import _blank, _cell_aspect, _num, _pad_to

#: blue funds the answer the model gave, red argues against it
DIVERGE = LinearSegmentedColormap.from_list(
    "vj_signed",
    ["#b3253f", "#d4635f", "#eb9b93", "#f7d3cd", SURFACE,
     "#d3e3f5", "#a3c6ea", "#6ba1dd", "#1f63b8"])

C_FUND = BLUE          # the picture paid for the answer
C_OUT = RED            # the picture voted the other way and lost
C_OK = MUTED           # the correct refusals, as a reference cloud


def _signed_tiles(ax, vals, C, out=None, fs=18.1, dp=2, top=None):
    """A C x C grid of signed tiles; only the strong ones carry their price.

    Forty-nine six-point numbers were unreadable and mostly said ±0.00.
    The shade carries the sign and size everywhere; the number is printed only
    where a region actually paid, at a size a reader can see.
    """
    v = np.asarray(vals, dtype=float)
    lim = float(np.abs(v).max()) or 1.0
    ax.set_xlim(0, C)
    ax.set_ylim(C, 0)
    ax.set_aspect("equal")
    ax.set_anchor("N")          # both panels of a row hang from its heading
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for j in range(C * C):
        y, x = divmod(j, C)
        q = 0.5 + 0.5 * float(v[j]) / lim
        rgba = DIVERGE(q)
        ax.add_patch(Rectangle((x + .015, y + .015), .97, .97,
                               facecolor=rgba, edgecolor="none"))
        ink = "#ffffff" if abs(q - 0.5) > 0.36 else INK
        if abs(float(v[j])) >= 0.25 * lim:
            ax.text(x + .5, y + .52, _num(v[j], dp), ha="center", va="center",
                    color=ink, fontsize=fs)
    if top is not None:
        for j in np.atleast_1d(top):
            y, x = divmod(int(j), C)
            ax.add_patch(Rectangle((x + .05, y + .05), .90, .90, fill=False,
                                   edgecolor=INK, lw=1.6))


def _head(ax, text, fs=20.6, pad=9):
    ax.set_title(text, loc="center", color=INK, pad=pad, fontsize=fs)


def _rowhead(fig, ax_left, ax_right, line1, line2, fs=21):
    """One heading over a row of two panels.

    A title on each panel of a row of this width runs into its neighbour --
    "the model answers yes" printed straight through "regions sum to" in the
    first version.  The row gets one heading spanning both instead, placed in
    figure coordinates off the two panels' own extents.
    """
    x = (ax_left.get_position().x0 + ax_right.get_position().x1) / 2
    y = max(ax_left.get_position().y1, ax_right.get_position().y1)
    u = 1.0 / (72.0 * fig.get_figheight())
    fig.text(x, y + 10 * u + fs * 1.30 * u, line1, ha="center", va="baseline",
             color=INK, fontsize=fs)
    fig.text(x, y + 8 * u, line2, ha="center", va="baseline", color=INK2,
             fontsize=fs * 0.92)


def _photo(ax, path, thumb=900):
    from PIL import Image
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((thumb, thumb), Image.LANCZOS)
        im = _pad_to(im, _cell_aspect(ax))
        ax.imshow(im)
    except Exception:
        ax.set_facecolor("#eef2f6")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    for sp in ax.spines.values():
        sp.set_visible(False)


def xlimits(df, q=1.0, pad=0.05):
    """x-range: all of the wrong answers, and the control's middle 98%."""
    h = df[df.hallucinated == 1].total
    c = df[df.hallucinated == 0].total
    lo = min(h.min(), np.percentile(c, q)) if len(h) else np.percentile(c, q)
    hi = max(h.max(), np.percentile(c, 100 - q)) if len(h) else np.percentile(c, 100 - q)
    m = pad * (hi - lo)
    return lo - m, hi + m


def off_axis(df, **kw):
    """How many questions fall outside :func:`xlimits`, for the caption."""
    lo, hi = xlimits(df, **kw)
    return int(((df.total < lo) | (df.total > hi)).sum())


def _label(rec, tau):
    """What this case actually is, from its own number.

    An earlier version fixed the two row labels in advance and looked for a case
    to fit each.  When one regime is nearly empty that prints a label the case
    does not deserve, so the label is derived from the price instead.
    """
    if rec["total"] > tau:
        return "misperception"
    if rec["total"] < -tau:
        return "outvoted perception"
    return "too close to call"


def _extremes(cards, tau):
    """The two clearest hallucinations: the biggest price of either sign.

    Stated so a reader knows the rows were not chosen for prettiness -- among
    the wrong answers, the photograph that contributed most to the answer given
    and the one that contributed least.
    """
    hs = [c for c in cards if c["said"] == "yes"]
    if not hs:
        return [(None, "misperception"), (None, "outvoted perception")]
    top = max(hs, key=lambda c: c["total"])
    bot = min(hs, key=lambda c: c["total"])
    if top is bot:
        return [(top, _label(top, tau)), (None, "")]
    return [(top, _label(top, tau)), (bot, _label(bot, tau))]


def census(df, cards, stats, out, name="f88_census.png"):
    """Left: every wrong answer as a dot, over the control's distribution.
    Right: two worked cases, the photograph beside the priced map."""
    style.use(1.0)
    cen = stats["census"]
    tau = float(cen["tau"])
    h = df[df.hallucinated == 1]
    c = df[df.hallucinated == 0]
    funded = cen["hallucinations"]["reading"]["funded"]
    fs_lab = plt.rcParams["axes.labelsize"] * 0.92

    # the maps are square, so their tiles are as large as the row is tall:
    # at the paper's type size a tile of the old canvas was narrower than the
    # price printed in it, and two neighbouring prices ran together.  The
    # canvas is taller than the row needs for that reason alone -- the tile is
    # sized by the height here, never by the width
    fig = plt.figure(figsize=(20.0, 14.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.90, 1.0],
                          left=0.044, right=0.990, top=0.925, bottom=0.058,
                          wspace=0.11)
    gl = gs[0, 0].subgridspec(2, 1, height_ratios=[4.4, 1.0], hspace=0.10)
    ax = fig.add_subplot(gl[0])
    axr = fig.add_subplot(gl[1], sharex=ax)

    for b in (ax, axr):
        b.axvspan(-tau, tau, color="#e7e4db", zorder=0, lw=0)
        b.axvline(0, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=1)
    hf = h[h.total > 0]
    ho = h[h.total <= 0]
    ax.scatter(ho.total, ho.conf, s=44, c=C_OUT, alpha=0.80, lw=0, zorder=4)
    ax.scatter(hf.total, hf.conf, s=44, c=C_FUND, alpha=0.80, lw=0, zorder=4)
    # every wrong answer is shown; the control's outermost percent is allowed
    # off the axis rather than squashing the answers the figure is about, and
    # the caption says how many that is
    lo_, hi_ = xlimits(df)
    ax.set_xlim(lo_, hi_)
    ax.set_ylabel("the model\u2019s yes \u2212 no margin (logits)")
    ax.tick_params(labelbottom=False)
    style.title(ax, "Every wrong answer, priced")
    ytop = ax.get_ylim()[1]
    ax.annotate(f"outvoted perception  {1-funded:.0%}", xy=(0, ytop),
                xytext=(-9, -4), textcoords="offset points", ha="right",
                va="top", color=C_OUT, fontsize=fs_lab)
    ax.annotate(f"misperception  {funded:.0%}", xy=(0, ytop),
                xytext=(9, -4), textcoords="offset points", ha="left",
                va="top", color=C_FUND, fontsize=fs_lab)

    # the control on the same axis: the same reading on the questions the model
    # got right, which is what a picture pushing the other way looks like
    axr.hist(c.total, bins=64, color=C_OK, alpha=0.55, lw=0, zorder=3)
    axr.set_yticks([])
    axr.grid(False)
    axr.spines["left"].set_visible(False)
    axr.set_xlabel("the picture\u2019s contribution to the answer the model "
                   "gave (logits)")
    axr.annotate("correct refusals", xy=(0.015, 0.80), xycoords="axes fraction",
                 ha="left", va="top", color=INK2, fontsize=fs_lab)

    rows = _extremes(cards, tau)
    # ring the two cases in the scatter, so the worked rows can be found in it
    for rec, _ in rows:
        if rec is not None:
            ax.scatter([rec["total"]], [rec["conf"]], s=190, facecolors="none",
                       edgecolors=INK, lw=1.3, zorder=6)
    gr = gs[0, 1].subgridspec(2, 1, hspace=0.14)
    for r, (rec, label) in enumerate(rows):
        cell = gr[r].subgridspec(1, 2, width_ratios=[0.86, 1.14], wspace=0.06)
        axp = fig.add_subplot(cell[0])
        axt = fig.add_subplot(cell[1])
        if rec is None:
            _blank(axp)
            _blank(axt)
            continue
        _photo(axp, rec["path"])
        C = int(rec["coarse"])
        v = np.asarray(rec["tiles"], float)
        _signed_tiles(axt, v, C, top=int(np.argmax(np.abs(v))))
        # one heading for the row, over both of its panels; the two numbers are
        # the ones completeness says must agree
        _rowhead(fig, axp, axt,
                 f"{label}: \u201cis there a {rec['obj']}?\u201d, "
                 f"answered {rec['said']}",
                 f"{C*C} regions sum to {_num(rec['total'], 2)} logits "
                 f"\u00b7 ablation {_num(rec['truth'], 2)}")
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p
