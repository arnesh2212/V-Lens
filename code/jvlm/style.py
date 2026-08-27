# SPDX-License-Identifier: Apache-2.0
"""Figure style: Palatino, no bold, generous type, recessive furniture.

Colours are the validated categorical order; only the first three slots are
used where every pair can appear together (scatter, small multiples), which is
the documented all-pairs-safe prefix.  Text always wears an ink colour, never a
series colour -- identity is carried by the mark beside it.
"""
from __future__ import annotations

import glob

import matplotlib
import matplotlib.font_manager as fm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

for _p in glob.glob("/usr/share/fonts/truetype/texgyre/*.otf"):
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass

SERIF = ["TeX Gyre Pagella", "Palatino", "Palatino Linotype", "P052", "DejaVu Serif"]

# categorical slots, in the fixed order
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, MAGENTA, GREEN = "#eda100", "#e87ba4", "#008300"
VIOLET, RED = "#4a3aa7", "#e34948"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e6e5e1"

# role colours used throughout the paper, so a reader learns them once
C_WORD = ORANGE      # the verbal-report channel
C_KEY = BLUE         # the forced-choice channel
C_CTRL = MUTED       # controls and reference lines
C_JLENS = BLUE
C_LOGIT = ORANGE


def use(scale: float = 1.0) -> None:
    """Install the style.  ``scale`` multiplies every type size at once."""
    s = lambda x: round(x * scale, 1)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF,
        "mathtext.fontset": "dejavuserif",
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "figure.titleweight": "normal",

        "font.size": s(13),
        "axes.titlesize": s(15),
        "axes.labelsize": s(13.5),
        "xtick.labelsize": s(12),
        "ytick.labelsize": s(12),
        "legend.fontsize": s(12),
        "figure.titlesize": s(16),

        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.edgecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#cfcec9",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,

        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,

        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.borderaxespad": 0.0,
        "legend.columnspacing": 1.6,

        "figure.dpi": 130,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.prop_cycle": plt.cycler(color=SERIES),
    })


def title(ax, text: str, sub: str | None = None, pad: float = 5,
          legend_rows: int = 0, width: int = 74) -> None:
    """Left-aligned title with an optional quieter second line.

    The space above the axes is sized to the text that is actually there.  The
    previous version reserved a fixed block for a legend whether one was drawn
    or not, which is where the white bands above every panel came from; legends
    now sit inside the axes, where they belong on a plot with room in it.

    ``sub`` is wrapped rather than left to run past the axes and collide with
    the next panel.
    """
    import textwrap

    fs = plt.rcParams["axes.labelsize"] * 0.9
    lfs = plt.rcParams["axes.labelsize"] * 0.88
    lines = textwrap.wrap(sub, width) if sub else []
    block = len(lines) * fs * 1.32
    # A legend anchored just above the axes lands in exactly the strip the
    # subtitle grows into, and matplotlib does not arbitrate between them --
    # that is the overprinted first line visible in half a dozen older
    # figures.  Reserve the legend's rows here and the subtitle sits above it.
    strip = legend_rows * lfs * 1.9 if legend_rows else 0.0
    ax._sty_legend_gap = strip
    ax.set_title(text, loc="left", color=INK,
                 pad=pad + strip + block + (3 if lines else 0))
    if lines:
        ax.annotate("\n".join(lines), xy=(0, 1), xytext=(0, pad + strip),
                    xycoords="axes fraction", textcoords="offset points",
                    ha="left", va="bottom", color=INK2, fontsize=fs,
                    linespacing=1.32)


def legend_above(ax, ncols: int = 2, **kw):
    """A legend in the strip above the axes, under the title and subtitle.

    Pass ``legend_rows=1`` to :func:`title` on the same axes first; that is
    what opens the strip this sits in.  Anchoring a legend above the axes
    without reserving the room is what printed the legend on top of the
    subtitle in the first version of this paper's figures.
    """
    kw.setdefault("loc", "lower left")
    kw.setdefault("bbox_to_anchor", (0, 1.005))
    kw.setdefault("ncols", ncols)
    kw.setdefault("frameon", False)
    kw.setdefault("borderaxespad", 0.0)
    kw.setdefault("handlelength", 1.4)
    kw.setdefault("handletextpad", 0.5)
    kw.setdefault("columnspacing", 1.8)
    kw.setdefault("fontsize", plt.rcParams["axes.labelsize"] * 0.88)
    return ax.legend(**kw)


def panel(ax, text: str, loc: str = "upper right") -> None:
    """A per-panel label placed INSIDE the axes.

    ``ax.set_title`` on a panel that also carries the figure's title collides
    with it -- the two are drawn in the same strip of space above the axes and
    matplotlib does not arbitrate.  That collision is visible in several older
    figures in this project, so the panel label goes inside instead, where it
    cannot fight anything.
    """
    x, ha = (0.98, "right") if "right" in loc else (0.02, "left")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            color=INK2, fontsize=plt.rcParams["axes.labelsize"] * 0.85)


def legend(ax, **kw):
    """A compact legend inside the axes, with no frame and tight rows."""
    kw.setdefault("loc", "best")
    kw.setdefault("frameon", False)
    kw.setdefault("handlelength", 1.4)
    kw.setdefault("handletextpad", 0.5)
    kw.setdefault("borderaxespad", 0.3)
    kw.setdefault("labelspacing", 0.35)
    kw.setdefault("fontsize", plt.rcParams["axes.labelsize"] * 0.88)
    return ax.legend(**kw)


def despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def label_end(ax, x, y, text, color, dx=6, dy=0, y_true=None, marker="o",
              **kw):
    """Direct label at the end of a series, in ink -- never in the series hue.

    The house rule is that text wears an ink colour and identity is carried by
    the mark beside it.  There has to *be* a mark: where two lines finish close
    together the labels are pushed apart along y, and a stack of grey names
    beside a bundle of coloured lines cannot be read back onto them at all.  So
    a dot in the series hue is drawn at the label's left, and, when the label
    has been moved off its line, a hairline leads back to where the line
    actually ends.
    """
    from matplotlib.transforms import offset_copy

    tr = offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units="points")
    if y_true is not None and y_true != y:
        ax.plot([x, x], [y_true, y], color=color, lw=0.8, alpha=0.35,
                ls=(0, (1, 2)), zorder=1, clip_on=False)
    ax.plot([x], [y], marker=marker, color=color, ms=5.5, mew=0, zorder=5,
            clip_on=False, transform=tr)
    ax.annotate(text, xy=(x, y), xytext=(dx + 8.5, dy), va="center",
                textcoords="offset points", ha="left", color=INK2, **kw)
