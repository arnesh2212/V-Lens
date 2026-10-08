# SPDX-License-Identifier: Apache-2.0
"""Figure style: Palatino, no bold, generous type, recessive furniture.

Colours are the validated categorical order; only the first three slots are
used where every pair can appear together (scatter, small multiples), which is
the documented all-pairs-safe prefix.  Text always wears an ink colour, never a
series colour -- identity is carried by the mark beside it.
"""
from __future__ import annotations

import glob
import os as _os

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
INK2 = "#3f3e3b"
MUTED = "#6f6d66"
GRID = "#e6e5e1"

# role colours used throughout the paper, so a reader learns them once
C_WORD = ORANGE      # the verbal-report channel
C_KEY = BLUE         # the forced-choice channel
C_CTRL = MUTED       # controls and reference lines
C_JLENS = BLUE
C_LOGIT = ORANGE


#: Draw a title inside the figure itself?
#:
#: A paper's figure carries no heading of its own: the caption names it, and a
#: sentence set across the top of the image duplicates the caption in a place
#: the typesetter does not control.  These figures were written with editorial
#: headings, so the switch turns them off in one place rather than in fifteen
#: modules, and each figure-level helper returns the space back to the panels
#: instead of leaving a band of white where the heading used to be.
FIGURE_TITLES = False


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

        "font.size": s(17),
        "axes.titlesize": s(20),
        "axes.labelsize": s(17),
        "xtick.labelsize": s(15),
        "ytick.labelsize": s(15),
        "legend.fontsize": s(15),
        "figure.titlesize": s(22),

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
    _install_overlap_check()


# --------------------------------------------------------------------------
# Overlap checking
#
# Reviewers of this artifact kept finding collided text -- a tick label under a
# legend, two end-labels on top of each other -- and the only way anyone caught
# them was by looking at every figure, which does not scale past a few dozen.
# So the check runs where the figure is written, from the same renderer that
# lays the text out, and reports in the coordinates that actually matter.
#
# savefig is wrapped rather than each module edited: every figure module here
# already calls style.use(), so one wrap reaches all of them and no new figure
# can be added that skips the check.
_WRAPPED = False
#: intersection must cover this fraction of the smaller box before it counts.
#: Adjacent tick labels routinely share a pixel or two of padding without
#: reading as collided, and flagging those buries the real ones.  Set at
#: 0.14 this missed two captions whose tails genuinely ran together, so it
#: sits low enough to catch a clipped word and high enough to stay quiet on
#: a clean chart.
_MIN_FRAC = 0.07
#: written beside the figures, wherever the checkout happens to live
OVERLAP_LOG = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
    "figures", "overlaps.log")


def _boxes(fig):
    """Visible, non-empty text artists with their rendered extents.

    The draw is forced first.  savefig renders at savefig.dpi while the cached
    canvas renderer sits at figure.dpi, and measuring against the stale one puts
    axis labels on top of tick labels that are nowhere near them.
    """
    import matplotlib.text as mtext
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    # Tick labels for ticks outside the axes' view still exist as artists and
    # still report a window extent, but nothing is drawn for them: they sit
    # above the panel, right where the panel's title is, and read as a
    # collision that is not on the page.  Drop them before comparing.
    ghosts = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            try:
                lo, hi = sorted(axis.get_view_interval())
                pad = (hi - lo) * 1e-6
                for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                    if not (lo - pad <= tk.get_loc() <= hi + pad):
                        ghosts.add(id(tk.label1))
                        ghosts.add(id(tk.label2))
            except Exception:
                continue

    out = []
    for t in fig.findobj(mtext.Text):
        try:
            if id(t) in ghosts:
                continue
            if not t.get_visible() or not t.get_text().strip():
                continue
            bb = t.get_window_extent(renderer=r)
            if bb.width <= 0 or bb.height <= 0:
                continue
            out.append((t.get_text().strip().replace("\n", " "), bb))
        except Exception:
            continue
    return out


def overlaps(fig):
    """Pairs of text artists whose rendered boxes genuinely collide."""
    items, hits = _boxes(fig), []
    for i in range(len(items)):
        ti, bi = items[i]
        for j in range(i + 1, len(items)):
            tj, bj = items[j]
            w = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            h = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if w <= 0 or h <= 0:
                continue
            small = min(bi.width * bi.height, bj.width * bj.height)
            if small > 0 and (w * h) / small >= _MIN_FRAC:
                hits.append((ti[:38], tj[:38], (w * h) / small))
    return sorted(hits, key=lambda x: -x[2])


def _install_overlap_check():
    global _WRAPPED
    if _WRAPPED:
        return
    import os

    from matplotlib.figure import Figure
    inner = Figure.savefig

    def savefig(self, fname, *a, **kw):
        res = inner(self, fname, *a, **kw)
        try:
            hits = overlaps(self)
            if hits:
                name = os.path.basename(str(fname))
                head = f"{name}: {len(hits)} overlapping text pair(s)"
                print("  ! " + head, flush=True)
                for x, y, f in hits[:6]:
                    print(f"      {f:4.0%}  {x!r} <> {y!r}", flush=True)
                try:
                    os.makedirs(os.path.dirname(OVERLAP_LOG), exist_ok=True)
                    with open(OVERLAP_LOG, "a") as fh:
                        fh.write(head + "\n")
                        for x, y, f in hits:
                            fh.write(f"    {f:4.0%}  {x!r} <> {y!r}\n")
                except Exception:
                    pass
        except Exception:
            pass                      # a check must never lose a figure
        return res

    # A figure-level heading drawn with suptitle is the same duplication of the
    # caption as one drawn by hand, and there are a dozen call sites.  Silencing
    # it here reaches all of them; suptitle reserves no layout space of its own,
    # so nothing moves when it stops drawing.
    if not FIGURE_TITLES:
        Figure.suptitle = lambda self, *a, **k: None

    Figure.savefig = savefig
    _WRAPPED = True


def title(ax, text: str, sub: str | None = None, pad: float = 5,
          legend_rows: int = 0, width: int | None = None) -> None:
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
    if width is None:
        # wrapped to the axes it belongs to, not to a fixed column count: a
        # count that suits a full-width single plot is half again too wide for
        # one panel of three, and the overflow prints into the next panel
        w_in = ax.get_position().width * ax.figure.get_figwidth()
        width = max(18, int(w_in * 72.0 / (fs * 0.50)))
    lines = textwrap.wrap(sub, width) if sub else []
    block = len(lines) * fs * 1.32
    # A legend anchored just above the axes lands in exactly the strip the
    # subtitle grows into, and matplotlib does not arbitrate between them --
    # that is the overprinted first line visible in half a dozen older
    # figures.  Reserve the legend's rows here and the subtitle sits above it.
    strip = legend_rows * lfs * 1.9 if legend_rows else 0.0
    ax._sty_legend_gap = strip
    ax.set_title(text, loc="center", color=INK,
                 pad=pad + strip + block + (3 if lines else 0))
    if lines:
        ax.annotate("\n".join(lines), xy=(0.5, 1), xytext=(0, pad + strip),
                    xycoords="axes fraction", textcoords="offset points",
                    ha="center", va="bottom", color=INK2, fontsize=fs,
                    linespacing=1.32)


def legend_above(ax, ncols: int = 2, **kw):
    """A legend in the strip above the axes, under the title and subtitle.

    Pass ``legend_rows=1`` to :func:`title` on the same axes first; that is
    what opens the strip this sits in.  Anchoring a legend above the axes
    without reserving the room is what printed the legend on top of the
    subtitle in the first version of this paper's figures.
    """
    kw.setdefault("loc", "lower center")
    kw.setdefault("bbox_to_anchor", (0.5, 1.005))
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


def stagger_xticks(ax, drop_lines: float = 2.2, fs: float | None = None):
    """Drop every other x tick label, and return the room that took, in inches.

    Long category names standing side by side are wider than the slots the axis
    gives them, and at this paper's type size several of them met.  Dropping
    alternate names doubles the room each one has without taking a point off
    any of them; the caller adds the returned height to the bottom margin.
    """
    fs = fs or plt.rcParams["xtick.labelsize"]
    dy = drop_lines * fs * 1.30                      # points
    base = plt.rcParams["xtick.major.pad"]
    for t in ax.xaxis.get_major_ticks()[1::2]:
        t.set_pad(base + dy)
    return dy / 72.0


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
