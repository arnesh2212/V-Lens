# SPDX-License-Identifier: Apache-2.0
"""A quiet plotting style for the write-up: Palatino, no bold, little ink.

This is deliberately separate from :mod:`jvlm.style`, which the older figure
modules depend on.  The rules here are the ones the reader asked for after
seeing the first round: no heading inside the image, no paragraph inside the
image, one idea per panel, thin marks, a hairline grid, and text that stays in
ink colours while the marks carry the colour.

Colour is the reference categorical order validated for this surface (light
``#fcfcfb``); the first three slots also clear the all-pairs gate, so a scatter
or a set of small multiples uses those three and no more.  Sequential is one
blue ramp.  Diverging is blue against red with a grey middle.  Text never wears
a series colour.
"""
from __future__ import annotations

import glob
import os

import matplotlib
import matplotlib.font_manager as fm
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

for _d in ("/usr/share/fonts/truetype/texgyre", os.path.expanduser("~/.local/share/fonts/texgyre")):
    for _p in glob.glob(os.path.join(_d, "*.otf")):
        try:
            fm.fontManager.addfont(_p)
        except Exception:
            pass

SERIF = ["TeX Gyre Pagella", "Palatino", "Palatino Linotype", "P052", "DejaVu Serif"]

# ---- tokens ------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: fixed categorical order; never cycled past what is listed
#:
#: Chosen so the first four (the ones that actually appear together in a
#: single panel -- the read, the write, the logit lens, a fourth condition)
#: stay apart three ways at once: as ordinary colour, under a red-green or
#: blue-yellow confusion (protanopia, deuteranopia, tritanopia simulated),
#: and as plain grayscale luminance, where blue/orange/aqua used to crowd
#: into the same mid-grey. Aqua now sits dark, yellow sits light, and blue
#: and orange fall in between with real daylight between them.
SERIES = ["#2a78d6", "#e97135", "#174f48", "#dab24e", "#cf638a", "#39672a", "#3b2785", "#ce2138"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SERIES
#: the three that survive every pairing; a scatter uses these and no more
TRIO = SERIES[:3]
#: a third, more recessive ink -- for a caption-like line inside a figure
#: that should read as quieter than INK2 (axis labels) but is still text,
#: never a series colour
INK3 = MUTED

#: one hue, light to dark, for magnitude
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
         "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
#: warm against cool with a neutral middle, for signed quantities
DIVERGING = ["#0d366b", "#2a78d6", "#86b6ef", "#f0efec", "#f0a6a5", "#e34948", "#8f1d1d"]

LINE = 1.8          # points; the "2px" of the spec at figure scale
MARK = 6.5          # marker size, with a surface ring so it survives crossings


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

        # small, even type: the figure is read beside a caption, not instead of one
        "font.size": s(11.5),
        "axes.titlesize": s(12),
        "axes.labelsize": s(11.5),
        "xtick.labelsize": s(10.5),
        "ytick.labelsize": s(10.5),
        "legend.fontsize": s(10.5),

        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.edgecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.axisbelow": True,

        "lines.linewidth": LINE,
        "lines.markersize": MARK,
        "lines.markeredgewidth": 1.2,
        "lines.markeredgecolor": SURFACE,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",

        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.2,
        "legend.columnspacing": 1.4,
        "legend.labelcolor": INK2,

        "figure.dpi": 130,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "axes.prop_cycle": plt.cycler(color=SERIES),
    })
    # a heading belongs in the caption, never in the image
    from matplotlib.figure import Figure
    Figure.suptitle = lambda self, *a, **k: None


# ---- helpers -----------------------------------------------------------------
def despine(ax, left: bool = True) -> None:
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)


def end_label(ax, x, y, text, color=INK2, dx=5, dy=0, **kw):
    """Name a series at its last point.  Text in ink; the line carries the colour."""
    return ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                       ha="left", va="center", color=color, fontsize=kw.pop("fontsize", None)
                       or plt.rcParams["legend.fontsize"], **kw)


def stagger(ys, min_gap):
    """Push a sorted list of label heights apart so none sit closer than ``min_gap``.

    Returns the adjusted heights in the input order.  Used for end labels on
    lines that converge at the right edge; past four series, use small
    multiples instead of this.
    """
    order = np.argsort(ys)
    out = np.array(ys, float)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def chance_line(ax, y, label="chance", **kw):
    """A measured floor, drawn as a hairline and named once at the right edge."""
    ax.axhline(y, color=AXIS, lw=0.9, zorder=1, **kw)
    x1 = ax.get_xlim()[1]
    ax.annotate(label, xy=(x1, y), xytext=(-3, 3), textcoords="offset points",
                ha="right", va="bottom", color=MUTED,
                fontsize=plt.rcParams["xtick.labelsize"])


def small_multiples(n, cols=None, panel=(2.6, 2.2), sharey=True):
    """A grid of equally sized panels.  Returns ``(fig, axes_flat)``."""
    cols = cols or min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(panel[0] * cols, panel[1] * rows),
                             sharey=sharey, squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def panel_label(ax, text):
    """A short label for a panel, set in the corner in ink, never a sentence."""
    ax.text(0.0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom",
            color=INK2, fontsize=plt.rcParams["axes.titlesize"])


def save(fig, path, dpi=None):
    """Write the figure and run the overlap check the older style module owns."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi or plt.rcParams["savefig.dpi"])
    try:
        from jvlm import style as _old
        hits = _old.overlaps(fig)
        if hits:
            print(f"  ! {os.path.basename(path)}: {len(hits)} overlapping text pair(s)",
                  flush=True)
            for a, b, f in hits[:5]:
                print(f"      {f:4.0%}  {a!r} <> {b!r}", flush=True)
    except Exception:
        pass
    plt.close(fig)
    return path
