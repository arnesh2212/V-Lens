# SPDX-License-Identifier: Apache-2.0
"""The house style for every figure in this paper.

Rules this enforces, from the brief:

* Palatino (TeX Gyre Pagella) everywhere, and large: a two-column paper
  scales a figure down hard, so the type has to start bigger than looks right.
* No bold, anywhere, ever. No titles either: the story belongs in the caption.
* Every piece of text is ink. Colour lives in the marks, never in the words,
  so a label is read as a label and a colour means one thing per figure.
* A warm palette: clay, ochre, slate, sage, on a plain white ground.
* Wide rather than tall, at least about two to one, and cropped to the ink.
* No bar charts. Use position, line, ribbon, dot and ridge instead.
* No em dashes, in the figure or in its caption.
"""
from __future__ import annotations

import os
import warnings

import matplotlib
import matplotlib.font_manager as fm
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# ---------------------------------------------------------------- the palette
#: warm sequence, ordered so neighbours stay distinguishable in greyscale too
CLAY = "#BF5B3D"          #: ours: V-Read, V-Write, the effect being claimed
CLAY_DK = "#8F3F28"
CLAY_SOFT = "#E2B7A3"
SLATE = "#456076"         #: the blind or missed component
SAGE = "#77854F"
OCHRE = "#C98A2B"         #: what was seen, what was reported
GRAY = "#A9A198"
SERIES = [CLAY, OCHRE, SLATE, SAGE, CLAY_DK, "#8C6A1F", GRAY, CLAY_SOFT]
LENS = CLAY               #: V-Read / V-Write always take this
BASE = OCHRE              #: the strongest baseline
MUTED = GRAY              #: controls, random directions, chance
INK = "#20201D"           #: every piece of text in every figure
INK2 = INK                #: there is no secondary ink; text is text
SUBINK = "#5C574F"        #: faint lines only, never text
GRID = "#ECE7DE"
AXIS = "#5C574F"

#: the model a figure is of, stamped quietly in the margin. Empty for a figure
#: that pools several, which says nothing about it and lets the file name carry it
STAMP = ""
SURFACE = "#FFFFFF"       #: plain white ground, as asked
PAPER = "#FBF8F3"
BAND = "#F3E9E1"
#: warm sequential ramp for heat maps, light to dark
WARM = ["#FFF6EA", "#FBE3C2", "#F3C48E", "#E29A5C", "#C9703A", "#A54B23", "#7A2F13"]
#: warm diverging ramp, cool-warm but kept inside the family
DIVERGING = ["#3E5C57", "#7E9B92", "#C7CFC6", "#F4EDE2", "#EFC79A", "#D08A49", "#A54B23"]

_PAGELLA = ["TeX Gyre Pagella", "Palatino", "Palatino Linotype", "P052", "URW Palladio L", "DejaVu Serif"]


def _install_fonts() -> None:
    """Register the bundled Pagella files so the figure does not fall back silently."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in (os.path.join(here, "assets"), os.path.expanduser("~/.local/share/fonts/texgyre")):
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith((".otf", ".ttf")):
                try:
                    fm.fontManager.addfont(os.path.join(d, f))
                except Exception:  # noqa: BLE001
                    pass


def use(scale: float = 1.0) -> None:
    """Install the style.  ``scale`` multiplies every type size at once."""
    _install_fonts()
    have = {f.name for f in fm.fontManager.ttflist}
    fam = [f for f in _PAGELLA if f in have] or _PAGELLA
    if fam[0] not in have:
        warnings.warn("Pagella/Palatino not found; the figure will not match the paper")
    s = lambda x: round(x * scale, 1)
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": fam,
        "mathtext.fontset": "custom",
        "mathtext.rm": fam[0], "mathtext.it": f"{fam[0]}:italic", "mathtext.bf": fam[0],
        # a two-column paper scales a figure down hard, so the type starts
        # bigger than it looks like it should on screen
        "font.size": s(19),
        "axes.titlesize": s(19.5), "axes.labelsize": s(19.5),
        "xtick.labelsize": s(17), "ytick.labelsize": s(17),
        "legend.fontsize": s(16.5),
        "figure.titlesize": s(20),
        # no bold anywhere
        "font.weight": "normal", "axes.titleweight": "normal", "axes.labelweight": "normal",
        "figure.labelweight": "normal",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS,
        "xtick.color": INK, "ytick.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.9, "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "axes.linewidth": 1.1,
        "lines.linewidth": 2.6, "lines.markersize": 7.0, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        # no wasted margin
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.03, "figure.constrained_layout.wspace": 0.03,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "savefig.dpi": 300,
        "figure.dpi": 110,
    })


#: the widest a figure may be, in inches. A two-column template gives a
#: table*-width figure about seven inches, so a figure drawn seventeen inches
#: wide has its type divided by two and a half: seventeen point ticks land at
#: under seven point on the page, and at a single column they land at three.
#: Capping the width is what keeps the type legible; the aspect is preserved,
#: so a figure that was wide stays wide.
MAX_WIDTH = 10.5


def wide(rows: int = 1, cols: int = 1, unit=(6.6, 3.0), **kw):
    """A figure sized wide rather than tall, with no outer margin to spare."""
    w, h = unit[0] * cols, unit[1] * rows
    if w > MAX_WIDTH:
        h *= MAX_WIDTH / w
        w = MAX_WIDTH
    return plt.subplots(rows, cols, figsize=(w, h), **kw)


def despine(ax, left: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["left"].set_color(AXIS)


def label_end(ax, x, y, text, color=INK, dx=6, fontsize=None, va="center"):
    """Name a line at its right end instead of spending a legend on it.

    The text is ink whatever the line is. Colour belongs to the mark, and the
    mark here is the line the label is sitting at the end of; ``color`` is
    kept in the signature because callers pass it, and is used for the leader
    when a label has to be moved away from its own line.
    """
    t = ax.annotate(text, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
                    color=INK, va=va, ha="left", fontsize=fontsize,
                    annotation_clip=False)
    t._series_colour = color
    # the layout is not settled until the figure is drawn, so the final nudge
    # apart happens in ``save``; this is the list it works from
    if va == "center":
        ax.__dict__.setdefault("_end_labels", []).append(t)
    return t


def stagger(ys, min_gap):
    """Nudge end labels apart, keeping their order, so none overlap."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for k in range(1, len(order)):
        i, j = order[k - 1], order[k]
        if out[j] - out[i] < min_gap:
            out[j] = out[i] + min_gap
    return out


def chance(ax, y, text="chance", x=None):
    """A reference line, drawn quietly and named once where nothing is drawn.

    The label goes at whichever end of the axis has the most room below the
    line, so it does not land on top of a curve.
    """
    ax.axhline(y, color=AXIS, lw=1.0, zorder=1)
    if not text:
        return
    x0, x1 = ax.get_xlim()
    if x is None:
        # pick the end whose nearest data point sits furthest from the line
        room_l, room_r = [], []
        for ln in ax.lines:
            xd, yd = ln.get_xdata(), ln.get_ydata()
            if len(xd) < 2:
                continue
            room_l.append(abs(float(yd[0]) - y))
            room_r.append(abs(float(yd[-1]) - y))
        left = (min(room_l) if room_l else 1) >= (min(room_r) if room_r else 0)
        x, ha, dx = (x0, "left", 2) if left else (x1, "right", -2)
    else:
        ha, dx = "left", 2
    # the reference line, and often a mean line, pass straight through this
    # word; a white ground under it is the only thing that keeps it readable
    ax.annotate(text, xy=(x, y), xytext=(dx, 3), textcoords="offset points",
                color=INK, fontsize=matplotlib.rcParams["xtick.labelsize"] * 0.92,
                ha=ha, va="bottom", zorder=8,
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=1.0))


def key_labels(ax, pairs, loc="right", gap_pts: float = 3.0):
    """A compact key in the margin: a stroke of colour, then the name in ink.

    A block of names with no colour beside them is a guess, and a block of
    names drawn inside the panel lands on the lines it is naming. This puts
    both right: the key goes outside the right spine, and every entry carries
    a sample of its own line.

    ``pairs`` is ``[(text, colour, dashes), ...]`` in the order to show them.
    """
    if not pairs:
        return
    ax.figure.canvas.draw()
    fs = matplotlib.rcParams["legend.fontsize"]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    h_px = max(ax.get_window_extent().height, 1)
    per_pt = (y1 - y0) / h_px * ax.figure.dpi / 72.0
    step = (fs + gap_pts) * per_pt
    top = y1 - step * 0.35
    xs = x1 + (x1 - x0) * 0.025
    xe = xs + (x1 - x0) * 0.055
    for i, (text, colour, dash) in enumerate(pairs):
        y = top - i * step
        ax.plot([xs, xe], [y, y], color=colour, lw=2.6, dashes=dash or (),
                solid_capstyle="butt", clip_on=False, zorder=9)
        ax.annotate(text, xy=(xe, y), xytext=(5, 0), textcoords="offset points",
                    color=INK, va="center", ha="left", annotation_clip=False,
                    fontsize=fs)


def fit_y(ax, lo=None, hi=None, pad=0.08):
    """Scale the axis to what was actually drawn, with a little air.

    A panel scaled nought to one when the data lives between nought and three
    tenths is seven tenths of empty page.
    """
    ys = []
    for ln in ax.lines:
        d = np.asarray(ln.get_ydata(), float)
        d = d[np.isfinite(d)]
        if d.size:
            ys.append((d.min(), d.max()))
    if not ys:
        return
    a = min(v[0] for v in ys) if lo is None else lo
    b = max(v[1] for v in ys) if hi is None else hi
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        return
    m = (b - a) * pad
    ax.set_ylim(a - m, b + m)


def end_labels(ax, x, pairs, gap_pts: float = 2.0, outside: bool = True):
    """Name several lines at their right ends, nudged apart so none collide.

    ``pairs`` is ``[(y, text, colour), ...]``. The minimum spacing is worked out
    from the type size and the axis height rather than guessed in data units,
    so it holds whatever the figure is scaled to.

    The labels go outside the right spine by default, and the axis is widened
    to make room for them. A block of names sitting inside the panel lands on
    the very lines it is naming, so the margin is the right place for it.
    """
    if not pairs:
        return
    if outside:
        x0, x1 = ax.get_xlim()
        if x > x1 - (x1 - x0) * 1e-9:
            ax.set_xlim(x0, x)
        x1 = ax.get_xlim()[1]
    ax.figure.canvas.draw()
    h_px = ax.get_window_extent().height
    y0, y1 = ax.get_ylim()
    fs = matplotlib.rcParams["legend.fontsize"]
    per_pt = (y1 - y0) / max(h_px, 1) * ax.figure.dpi / 72.0
    min_gap = (fs + gap_pts) * per_pt
    ys = stagger([p[0] for p in pairs], min_gap)
    moved = min_gap * 0.45
    for (y_true, text, colour), yy in zip(pairs, ys):
        label_end(ax, x, float(yy), text, color=colour)
        # a label nudged clear of its own line is no longer beside it, so a
        # leader in the line's colour carries the identity the text may not
        if abs(float(yy) - float(y_true)) > moved:
            x0, x1 = ax.get_xlim()
            lead = (x1 - x0) * 0.022
            ax.plot([x, x + lead], [y_true, yy], color=colour, lw=1.1,
                    solid_capstyle="butt", zorder=2, clip_on=False)


def ribbon(ax, x, lo, hi, color, alpha=0.18, **kw):
    """A confidence band under a line."""
    return ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, zorder=2, **kw)


def depth_axis(ax, label="depth through the network"):
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["input", "", "half", "", "output"])
    # "input" sits at the corner, where the lowest y tick also is; a little
    # room below keeps the two off each other whatever the y range is
    ax.tick_params(axis="x", pad=5)
    ax.set_xlabel(label)


def warm_cmap(diverging: bool = False):
    from matplotlib.colors import LinearSegmentedColormap
    name = "vlens_div" if diverging else "vlens_warm"
    return LinearSegmentedColormap.from_list(name, DIVERGING if diverging else WARM)


def image_panel(ax, im, title=None, boxes=None, box_color=INK):
    """An image on a plain white ground, no axes, optional outlined patches."""
    ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    if boxes:
        for (x, y, w, h) in boxes:
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=box_color, lw=1.4))
    if title:
        ax.set_title(title, loc="left", color=INK2, pad=4)


# ---------------------------------------------------------------- saving
OVERLAP_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "figures", "overlaps.log")


def _settle_end_labels(fig) -> None:
    """Nudge end labels apart once the layout is final, measuring in pixels.

    Spacing worked out before the figure is drawn is spacing against the wrong
    axes height, so the spacing holds whatever the figure is scaled to.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        items = [t for t in getattr(ax, "_end_labels", []) if t.get_visible()]
        if len(items) < 2:
            continue
        boxes = [t.get_window_extent(renderer=r) for t in items]
        ys = [(b.y0 + b.y1) / 2.0 for b in boxes]
        h = max(b.height for b in boxes)
        box = ax.get_window_extent()
        # One name a long way from the rest reads as a stray rather than as a
        # label on a line, and the eye has to hunt for what it belongs to. When
        # the lines finish that far apart, the names go together as a block in
        # the order the lines are in, and the colours do the pointing.
        srt = sorted(ys)
        apart = max((b - a for a, b in zip(srt, srt[1:])), default=0.0)
        if len(items) > 2 and apart > 3 * h:
            top = box.y1 - h / 2
            block = np.array([top - k * (h + 1.5) for k in range(len(items))], float)
            moved = np.empty(len(items), float)
            moved[np.argsort(-np.asarray(ys))] = block
        else:
            moved = stagger(ys, h + 1.5)
            moved = moved - (moved.mean() - float(np.mean(ys)))   # keep them centred
        # and inside the panel: a column of names floating above the frame
        # reads as part of the figure rather than as labels on its lines
        over = moved.max() + h / 2 - box.y1
        if over > 0:
            moved = moved - over
        under = box.y0 - (moved.min() - h / 2)
        if under > 0:
            moved = moved + min(under, max(0.0, box.y1 - (moved.max() + h / 2)))
        inv = ax.transData.inverted()
        for t, was, now in zip(items, ys, moved):
            x_d, y_d = t.xy
            t.xy = (x_d, y_d + inv.transform((0.0, now))[1] - inv.transform((0.0, was))[1])
    fig.canvas.draw()


def _text_boxes(fig):
    fig.canvas.draw()
    out = []
    for ax in fig.axes:
        # ticks, axis labels and titles live outside the axes by construction and
        # a tight bounding box grows to include them, so only annotations placed
        # by hand are checked for running off the canvas
        # A tick matplotlib has kept but placed outside the view is not drawn
        # and must not be checked: it sits at the corner and collides with
        # everything, so only text the reader sees is checked.
        def shown(labels, locs, lo, hi):
            return [t for t, v in zip(labels, locs) if lo - 1e-9 <= v <= hi + 1e-9]

        xl, yl = ax.get_xlim(), ax.get_ylim()
        xt = shown(ax.get_xticklabels(), ax.get_xticks(), min(xl), max(xl))
        yt = shown(ax.get_yticklabels(), ax.get_yticks(), min(yl), max(yl))
        ticks = {id(t) for t in [*xt, *yt, ax.title, ax.xaxis.label, ax.yaxis.label]
                 if t is not None}
        for t in [*ax.texts, *xt, *yt, ax.title, ax.xaxis.label, ax.yaxis.label]:
            if t is None or not t.get_text().strip() or not t.get_visible():
                continue
            try:
                out.append((t.get_text(), t.get_window_extent(renderer=fig.canvas.get_renderer()),
                            id(t) in ticks))
            except Exception:  # noqa: BLE001
                pass
    return out


def save(fig, path, dpi=None, check: bool = True, tight: bool = False):
    """Write the figure, then report any text that overlaps or runs off the canvas.

    ``tight`` crops to the ink. A 3d axes fills its own box whatever is drawn
    in it, so the margin it leaves cannot be taken out from inside.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # No stamp. Whichever model a figure is of, the file name says so, and a
    # line of grey in the margin buys a band of empty page in every figure.
    problems = []
    tight_save = tight or matplotlib.rcParams.get("savefig.bbox") == "tight"
    _settle_end_labels(fig)
    if check:
        boxes = _text_boxes(fig)
        W, H = fig.canvas.get_width_height()
        for i, (t1, b1, is_tick) in enumerate(boxes):
            # tick labels sit on the edge by construction and a tight bounding
            # box grows to include them, so only other text is checked
            # the figure is saved with a tight bounding box, which grows the
            # canvas to hold anything placed outside the axes; measuring
            # against the untightened canvas reports that as falling off it
            if (not tight_save and not is_tick
                    and (b1.x0 < -2 or b1.y0 < -2 or b1.x1 > W + 2 or b1.y1 > H + 2)):
                problems.append(f"off canvas: {t1[:40]!r}")
            for t2, b2, _ in boxes[i + 1:]:
                ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
                oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
                if ox > 1 and oy > 1 and min(ox * oy / max(b1.width * b1.height, 1), 1) > 0.12:
                    problems.append(f"overlap: {t1[:28]!r} <> {t2[:28]!r}")
    fig.savefig(path, dpi=dpi or matplotlib.rcParams["savefig.dpi"],
                **(dict(bbox_inches="tight", pad_inches=0.02) if tight else {}))
    # the sync redraws every ten minutes and pyplot holds on to every figure
    # it is handed until told otherwise
    plt.close(fig)
    if problems:
        os.makedirs(os.path.dirname(OVERLAP_LOG) or ".", exist_ok=True)
        with open(OVERLAP_LOG, "a") as fh:
            fh.write(f"{path}\n" + "\n".join("   " + p for p in dict.fromkeys(problems)) + "\n")
        print(f"  ! {os.path.basename(path)}: " + "; ".join(list(dict.fromkeys(problems))[:3]))
    return path


def key_band(fig, handles, labels, rows: int = 1, ncol=None, fontsize=None,
             gap_in: float = 0.08, x=None, **kw):
    """Put the key in a band of its own above every panel.

    A key drawn inside a panel, or level with a column of rows beside one,
    reads as more data. This settles the layout as it stands, freezes every
    panel and every piece of figure-placed text where it is in inches, makes
    the figure taller by exactly the key's height, and sets the key as one row
    (or ``rows`` rows) above the highest ink. Nothing is squeezed to make room.
    """
    from matplotlib.lines import Line2D as _L
    from matplotlib.patches import Rectangle as _R, FancyBboxPatch as _F, Polygon as _P
    from matplotlib.text import Text as _T
    fs = fontsize or matplotlib.rcParams["legend.fontsize"]
    n = len(handles)
    asked = ncol
    ncol = ncol or int(np.ceil(n / max(rows, 1)))
    # a layout engine can take more than one pass to settle, so the figure is
    # drawn until the panels stop moving before they are frozen
    for _ in range(4):
        before = [tuple(ax.get_position(original=False).bounds) for ax in fig.axes]
        fig.canvas.draw()
        if [tuple(ax.get_position(original=False).bounds) for ax in fig.axes] == before:
            break
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches()
    pos = [(ax, ax.get_position(original=False)) for ax in fig.axes]
    tb = fig.get_tightbbox(r)
    fig.set_layout_engine("none")
    for ax, p in pos:
        ax.set_position(p, which="both")
    base = dict(frameon=False, fontsize=fs, labelcolor=INK, borderaxespad=0.0,
                columnspacing=1.4, handlelength=2.0)
    base.update(kw)
    fixed = asked is not None
    while True:
        leg = fig.legend(handles, labels, loc="lower center", ncol=ncol,
                         bbox_to_anchor=(0.5, 0.0), **base)
        fig.canvas.draw()
        lb = leg.get_window_extent(r)
        # a key wider than the panels would widen the page; it takes one more
        # row instead, never more rows than it has entries
        if fixed or lb.width / fig.dpi <= float(tb.width) + 0.02 or ncol <= 1:
            break
        leg.remove()
        rows += 1
        ncol = int(np.ceil(n / rows))
    lh = lb.height / fig.dpi
    H2 = float(tb.y1) + gap_in + lh + 0.03
    s = H / H2
    figtr = {id(fig.transFigure), id(fig.transSubfigure)}
    for ax, p in pos:
        ax.set_position([p.x0, p.y0 * s, p.width, p.height * s], which="both")
    for a in fig.findobj():
        if a is leg or a is fig:
            continue
        try:
            tr = a.get_transform()
        except Exception:  # noqa: BLE001
            continue
        if id(tr) not in figtr:
            continue
        if isinstance(a, _T):
            xx, yy = a.get_position()
            a.set_y(yy * s)
        elif isinstance(a, _L):
            a.set_ydata(np.asarray(a.get_ydata(), float) * s)
        elif isinstance(a, (_R, _F)):
            a.set_y(a.get_y() * s)
            a.set_height(a.get_height() * s)
        elif isinstance(a, _P):
            xy = np.asarray(a.get_xy(), float)
            xy[:, 1] *= s
            a.set_xy(xy)
    for other in fig.legends:
        if other is leg:
            continue
        bb = other.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        other.set_bbox_to_anchor((bb.x0, bb.y0 * s, bb.width, bb.height * s),
                                 transform=fig.transFigure)
    fig.set_size_inches(W, H2)
    xc = x if x is not None else (float(tb.x0) + float(tb.x1)) / 2.0 / W
    leg.set_bbox_to_anchor((xc, (float(tb.y1) + gap_in) / H2), transform=fig.transFigure)
    return leg


def key_band_lines(fig, pairs, lw: float = 2.6, **kw):
    """:func:`key_band` for a key of line strokes: ``pairs`` is
    ``[(text, colour, dashes), ...]``, the same form :func:`key_labels` takes."""
    from matplotlib.lines import Line2D as _L
    hs = [_L([], [], color=c, lw=lw, dashes=d or (None, None)) for _t, c, d in pairs]
    return key_band(fig, hs, [t for t, _c, _d in pairs], **kw)
