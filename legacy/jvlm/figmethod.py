# SPDX-License-Identifier: Apache-2.0
"""The method's one parameter, its resolution, its models, its depth profile.

Four figures that belong to the method and to the model-level claims rather
than to any single result, and that the paper was making in prose only.

  ``region_ladder``   what a region of 1, 4, 16, 64 and 196 patches actually
                      covers on a photograph, with the measured agreement
                      printed under each -- the two prior lenses are the two
                      ends of this strip.
  ``resolution``      how finely each lens tells one part of a picture from
                      another, as a distribution over photographs rather than
                      as a single averaged number.
  ``models``          the flagship causal measures, per model, ours against the
                      best prior lens, with chance marked.
  ``depth_profile``   every measure the paper reports "at the best depth", laid
                      out against depth, so a reader can see where in the
                      network each claim lives.

Kept out of :mod:`jvlm.figcausal`, :mod:`jvlm.figatlas` and
:mod:`jvlm.figdiagram` deliberately; those modules belong to other tracks.  The
layout rules they enforce are restated at the top of this file rather than
imported, so editing either module cannot silently reshape figures here:

  * a title's pad is computed from how many lines its subtitle has, never a
    fixed pad, and a row of panels reserves the room its tallest head needs;
  * a word never shares a place with another word or with a mark -- where two
    would land together, one is moved and the move is in the code;
  * the ends of an axis are named below the axis, not at the end of a line;
  * text wears an ink colour; identity is carried by the coloured mark beside
    it;
  * a grid of tiles over a photograph is square, or nothing in it fits.
"""
from __future__ import annotations

import os

import numpy as np

from jvlm import style
from jvlm.style import GRID, INK, INK2, MUTED

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

#: one colour per lens, used everywhere in the paper
LENS_COLOUR = {
    "logit lens": style.ORANGE,
    "J-lens": style.VIOLET,
    "VJ-Lens": style.BLUE,
    "VJ-Lens (point)": style.AQUA,
    "VJ-Lens (path)": style.BLUE,
}
#: the run files spell the same lens three ways
CANON = {"Atlas": "VJ-Lens", "ATLAS": "VJ-Lens", "V-Lens": "VJ-Lens",
         "atlas": "VJ-Lens", "vj": "VJ-Lens", "jlens": "J-lens",
         "logit": "logit lens", "say": "logit lens"}

HOT = "#b8481f"
MINUS = "−"


def _num(v, dp=2):
    """A number with a real minus sign in it, not a hyphen."""
    return f"{v:.{dp}f}".replace("-", MINUS)


# ---------------------------------------------------------------- furniture

def _head(ax, title, sub=None, sub_colour=None, fs_title=20.6, fs=18.8,
          min_lines=0):
    """Panel title with a quieter block under it, both above the axes.

    The pad is computed from the number of lines the subtitle actually has, or
    from ``min_lines`` where a row of panels has to keep its titles level.
    """
    n_sub = str(sub).count("\n") + 1 if sub else 0
    step = fs * 1.35
    pad = 10 + max(n_sub, min_lines) * step
    ax.set_title(title, loc="center", color=INK, pad=pad, fontsize=fs_title)
    if sub:
        ax.annotate(sub, xy=(0.5, 1), xytext=(0, 7), xycoords="axes fraction",
                    textcoords="offset points", ha="center", va="bottom",
                    color=sub_colour or INK2, fontsize=fs, linespacing=1.35)


def _figure_head(fig, main, sub=None, x=0.5, fs=23.5, fss=21.8, gap=7.0,
                 clear=11.0):
    """Centred figure title; returns where panels may start.

    Explanatory blocks belong in the caption, so ``sub`` is normally None and
    the head is one centred line of type.
    """
    unit = 1.0 / (72.0 * fig.get_figheight())
    y_main = 1.0 - 6.0 * unit
    if not style.FIGURE_TITLES:
        return y_main                  # the caption names the figure
    fig.text(x, y_main, main, ha="center", va="top", color=INK, fontsize=fs)
    if not sub:
        return y_main - (fs * 1.22 + clear) * unit
    y_sub = y_main - (fs * 1.22 + gap) * unit
    fig.text(x, y_sub, sub, ha="center", va="top", color=INK2, fontsize=fss,
             linespacing=1.42)
    lines = str(sub).count("\n") + 1
    return y_sub - lines * fss * 1.42 * unit - clear * unit


def _wrap_to_axes(fig, ax, s, fs, frac=0.98):
    """Break ``s`` on spaces so every line fits inside ``ax`` at ``fs`` points.

    Measured, not guessed: the type scale moved once and every hand-counted
    break moved with it.  The axes have to be positioned before this is
    called, which is why the row's ``subplots_adjust`` runs first.
    """
    r = fig.canvas.get_renderer()
    avail = ax.get_window_extent(renderer=r).width * frac
    probe = fig.text(0, 0, "", fontsize=fs)
    lines, cur = [], ""
    for w in str(s).split():
        trial = (cur + " " + w).strip()
        probe.set_text(trial)
        if cur and probe.get_window_extent(renderer=r).width > avail:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    probe.remove()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _panel_top(fig, head_top, lines, fs=18.8, fs_title=20.6, clear=8.0,
               title_lines=1):
    """Leave the row of panel heads exactly the room its tallest member needs.

    ``title_lines`` is how many lines the tallest heading broke into: a
    heading wrapped to the panel's width is taller than one line, and the row
    has to be pushed down by the difference or the key lands on it.
    """
    unit = 1.0 / (72.0 * fig.get_figheight())
    return head_top - (10 + lines * fs * 1.35
                       + title_lines * fs_title * 1.25 + clear) * unit


def _key(fig, y, names, x=0.5, fs=20.3, ncols=None):
    """One key for a whole row of panels: a coloured mark, then an ink name.

    Five panels carrying the same three series do not need five legends, and a
    legend inside a panel has to be placed somewhere a line is not.
    """
    handles = [Line2D([], [], color=LENS_COLOUR.get(n, MUTED), lw=2.2,
                      marker="o", ms=6, label=n) for n in names]
    return fig.legend(handles=handles, loc="upper center",
                      bbox_to_anchor=(x, y), frameon=False,
                      ncols=ncols or len(names), fontsize=fs,
                      labelcolor=INK2, handlelength=1.5, handletextpad=0.5,
                      columnspacing=2.0, borderaxespad=0.0)


def _dot_label(ax, x, y, text, colour, side="right", fs=None, dy=0.0):
    """A name beside a coloured mark, on either side of the mark."""
    fs = fs or plt.rcParams["axes.labelsize"] * 0.88
    ax.plot([x], [y], marker="o", color=colour, ms=6, mew=0, zorder=6,
            clip_on=False)
    dx, ha = (10, "left") if side == "right" else (-10, "right")
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="center", color=INK2, fontsize=fs, zorder=6,
                clip_on=False)


def _panel_key(ax, rows, loc="upper right", fs=19.6):
    """A key inside one panel: a coloured mark, then an ink name, one per row.

    Used where three distributions overlap and there is nowhere beside a mound
    to put its name without landing on another one.
    """
    handles = [Line2D([], [], color=LENS_COLOUR.get(l, MUTED), lw=0,
                      marker="o", ms=7, label=t) for l, t in rows]
    return style.legend(ax, handles=handles, loc=loc, fontsize=fs,
                        handlelength=0.8, handletextpad=0.5, labelspacing=0.5)


def _save(fig, out, name, dpi=200):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p


# ============================================== A. the one-parameter family

def region_ladder(rows, card, out, name="f80_ladder.png",
                  sizes=(1, 4, 16, 64, 196), grid=14, square=448,
                  fig_w=17.0):
    """What a region of each size covers, with the measured agreement under it.

    The sweep already has a line chart.  A curve says the middle of the axis is
    the useful part; it does not say what the middle of the axis *is*.  Here the
    same five sizes are drawn on a real photograph, at the resolution the model
    sees it, so a reader can see the object being read at each scale.

    The photograph is shown square because that is how it reaches the vision
    tower, and the grid drawn over it is the model's own token grid.  The blocks
    are centred on the annotated object; in the sweep itself every region is
    placed at random, which is what the numbers underneath are measured over.
    """
    from PIL import Image

    style.use()
    by = {}
    for r in rows:
        by.setdefault(int(r["size"]), []).append(r)
    sizes = [s for s in sizes if s in by]

    im = Image.open(card["path"]).convert("RGB").resize(
        (square, square), Image.LANCZOS)
    arr = np.asarray(im)

    C = int(card.get("coarse", 7))
    scale = grid / C
    inside = list(card.get("inside") or [])
    if inside:
        rc = np.array([divmod(int(c), C) for c in inside], dtype=float)
        cy = (rc[:, 0].mean() + 0.5) * scale
        cx = (rc[:, 1].mean() + 0.5) * scale
    else:
        cy = cx = grid / 2.0

    n = len(sizes)
    # a square photograph panel means the axes box is square in display units,
    # so the figure's height is not free: it is the panel's width plus whatever
    # the head and the foot need.  Solving for it is what keeps the strip from
    # opening a band of white above and below every picture.
    left, right, ws = 0.008, 0.995, 0.055
    panel_w = fig_w * (right - left) / (n + (n - 1) * ws)
    head_in, foot_in = 1.55, 1.20
    fig_h = head_in + panel_w + foot_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = [fig.add_subplot(1, n, i + 1) for i in range(n)]

    for ax, s in zip(axes, sizes):
        side = max(1, min(grid, int(round(s ** 0.5))))
        r0 = int(min(max(round(cy - side / 2.0), 0), grid - side))
        c0 = int(min(max(round(cx - side / 2.0), 0), grid - side))

        ax.imshow(arr, extent=(0, grid, grid, 0), interpolation="antialiased")
        ax.add_patch(Rectangle((0, 0), grid, grid, facecolor=style.SURFACE,
                               edgecolor="none", alpha=0.72, zorder=2))
        lit = ax.imshow(arr, extent=(0, grid, grid, 0), zorder=3,
                        interpolation="antialiased")
        lit.set_clip_path(Rectangle((c0, r0), side, side,
                                    transform=ax.transData))
        for k in range(1, grid):
            ax.plot([k, k], [0, grid], color="#ffffff", lw=0.5, alpha=0.30,
                    zorder=4)
            ax.plot([0, grid], [k, k], color="#ffffff", lw=0.5, alpha=0.30,
                    zorder=4)
        ax.add_patch(Rectangle((c0, r0), side, side, fill=False, zorder=5,
                               edgecolor=style.BLUE, lw=2.4))
        ax.set_xlim(0, grid)
        ax.set_ylim(grid, 0)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

        frac = 100.0 * s / (grid * grid)
        sub = (f"{frac:.1f}% of the frame" if frac < 1
               else f"{frac:.0f}% of the frame")
        if s == sizes[0]:
            sub += "\nthe finest a reading can be"
        elif s == sizes[-1]:
            sub += "\nthe Jacobian lens's own corner"
        _head(ax, f"{s} patch" + ("" if s == 1 else "es"), sub, min_lines=2)

        rr = [float(x["spearman"]) for x in by[s]]
        dd = [float(x["mean_drop"]) for x in by[s]]
        drop = 100.0 * float(np.mean(dd))
        foot = (f"costs {drop:.0f}% of the answer\n"
                f"predicted at ρ = {_num(np.mean(rr))}")
        ax.annotate(foot, xy=(0.5, 0), xytext=(0, -13),
                    xycoords="axes fraction", textcoords="offset points",
                    ha="center", va="top", color=INK2, fontsize=19.6,
                    linespacing=1.5)

    n_ph = int(by[sizes[0]][0].get("n", 0))
    print(f"  region_ladder: {n_ph} photographs, grid {grid}x{grid}",
          flush=True)
    head_top = _figure_head(
        fig, "Both ends of the region dial are worse than its middle")
    top = _panel_top(fig, head_top, 2)
    bottom = top - panel_w / fig_h
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom,
                        wspace=ws)
    return _save(fig, out, name)


# ================================================ B. spatial resolution seen

def resolution(df, out, name="f81_resolution.png"):
    """How finely each lens tells one part of a picture from another.

    The paper reports three averaged numbers.  An average cannot say whether a
    gap is a shift of the whole population or a few photographs dragging a
    mean, and that is the only interesting question about a gap this size, so
    the distributions are drawn instead -- and then the same thing paired,
    photograph by photograph, which is the form no average can fake.
    """
    style.use()
    d = df.copy()
    d["lens"] = d.lens.replace(CANON)
    order = [l for l in ("logit lens", "J-lens", "VJ-Lens")
             if l in set(d.lens)]
    per = d.groupby(["i", "lens"])[["distinct", "agree"]].mean()
    n_ph = int(d.i.nunique())
    n_patch = 196

    fig, axes = plt.subplots(1, 3, figsize=(17.8, 5.5))

    # ---- 1. how many different words a picture gets
    ax = axes[0]
    v = {l: per.xs(l, level="lens").distinct.values for l in order}
    bins = np.arange(28, 132, 3.0)
    top = 0.0
    for l in order:
        c = LENS_COLOUR[l]
        h, _ = np.histogram(v[l], bins=bins)
        top = max(top, float(h.max()))
        ax.hist(v[l], bins=bins, histtype="stepfilled", color=c, alpha=0.16,
                zorder=2)
        ax.hist(v[l], bins=bins, histtype="step", color=c, lw=1.8, zorder=3)
    ax.set_ylim(0, top * 1.62)
    ax.set_xlim(bins[0], bins[-1])
    prior = [l for l in order if l != "VJ-Lens"]
    # three names beside three mounds, two of which are on top of each other:
    # the only place all three fit is the empty strip above every mound
    _panel_key(ax, [(l, f"{l}, median {np.median(v[l]):.0f}") for l in order])
    ax.set_xlabel(f"different top-ranked words, out of {n_patch} patches")
    ax.set_ylabel("photographs")
    _head(ax, "how many words one picture gets", min_lines=1)
    style.despine(ax)

    # ---- 2. how often the same word comes back
    ax = axes[1]
    a = {l: 100.0 * per.xs(l, level="lens").agree.values for l in order}
    bins = np.arange(0, 13.5, 0.4)
    top = 0.0
    for l in order:
        c = LENS_COLOUR[l]
        h, _ = np.histogram(a[l], bins=bins)
        top = max(top, float(h.max()))
        ax.hist(a[l], bins=bins, histtype="stepfilled", color=c, alpha=0.16,
                zorder=2)
        ax.hist(a[l], bins=bins, histtype="step", color=c, lw=1.8, zorder=3)
    ax.set_ylim(0, top * 1.62)
    ax.set_xlim(0, bins[-1])
    _panel_key(ax, [(l, f"{l}, {np.median(a[l]):.1f}%") for l in order])
    ax.set_xlabel("two patches picked at random read the same word  (%)")
    ax.set_ylabel("photographs")
    _head(ax, "how often the same word comes back", min_lines=1)
    style.despine(ax)

    # ---- 3. the same thing paired, photograph by photograph
    ax = axes[2]
    if "VJ-Lens" in v and prior:
        best = np.max(np.vstack([v[l] for l in prior]), axis=0)
        gap = np.sort(v["VJ-Lens"] - best)
        x = np.arange(1, len(gap) + 1)
        ax.axhline(0, color=INK, lw=1.0, zorder=2)
        ax.fill_between(x, 0, gap, color=style.BLUE, alpha=0.16, zorder=3)
        ax.plot(x, gap, color=style.BLUE, lw=2.2, zorder=4)
        ax.set_ylim(0, float(gap.max()) * 1.34)
        ax.set_xlim(0, len(gap) + 1)
        ax.annotate("the two lenses would agree here", xy=(2, 0),
                    xytext=(4, 6), textcoords="offset points", ha="left",
                    va="bottom", color=INK2, fontsize=18.8)
        mid = len(gap) // 2
        # both names go under their own dot, into the empty part of the fill;
        # above the curve they would cross it
        _dot_label(ax, float(x[0]), float(gap[0]),
                   f"smallest gap, {gap[0]:.0f} words", style.BLUE,
                   side="right", fs=19.6, dy=-14)
        # over its own dot and back towards the axis, on two short lines:
        # on one line the name reached past the spine and printed on the tick
        # beside it, and under the dot it lay along the curve itself
        _dot_label(ax, float(x[mid]), float(gap[mid]),
                   f"median,\n{np.median(gap):.0f} words", style.BLUE,
                   side="left", fs=19.6, dy=26)
        ax.set_xlabel(f"the {n_ph} photographs, ordered by the gap")
        ax.set_ylabel("more words than the better prior lens")
        _head(ax, "and it is every photograph",
              f"ahead on {int((gap > 0).sum())} of {len(gap)} photographs",
              min_lines=1)
        style.despine(ax)

    head_top = _figure_head(fig, "A picture is worth about seventy words to "
                                 "one lens and twenty to the others")
    top = _panel_top(fig, head_top, 1)
    # the middle panel's axis name is wider than its panel; the gutter is what
    # keeps its ends off the panels either side of it
    fig.subplots_adjust(left=0.052, right=0.994, top=top, bottom=0.135,
                        wspace=0.40)
    return _save(fig, out, name)


# ============================================================== C. models

def _rate(raw, lens, col):
    s = raw[raw.lens == lens].groupby("i")[col].mean()
    return s.values


def _boot(v, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (reps, len(v)))
    bs = v[idx].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def models(runs, out, name="f82_models.png", regions=49):
    """The flagship causal measures, per model, ours against the prior lenses.

    ``runs`` is a list of ``{"label", "raw", "n"}``.  Nothing is fitted, so a
    second model is one command; the only thing worth drawing is whether the
    gap survives the change of architecture, and against what chance level.
    """
    style.use()
    MEAS = [("hit_cause", "names the single most\ndamaging region", 1.0 / regions),
            ("top3_cause", "names one of the three\nmost damaging", 3.0 / regions)]
    ours = "VJ-Lens"
    fig, ax = plt.subplots(figsize=(2.9 + 3.4 * len(runs), 7.6))

    xs, ticks = [], []
    for k, run in enumerate(runs):
        raw = run["raw"].copy()
        raw["lens"] = raw.lens.replace(CANON)
        priors = [l for l in ("logit lens", "J-lens") if l in set(raw.lens)]
        for j, (col, lab, chance) in enumerate(MEAS):
            x = k * 1.0 + (-0.20 if j == 0 else 0.20)
            xs.append(x); ticks.append(lab)
            best, best_m = None, -1.0
            for l in priors:
                m = float(_rate(raw, l, col).mean())
                if m > best_m:
                    best, best_m = l, m
            v_o = _rate(raw, ours, col)
            m_o = float(v_o.mean())
            ax.plot([x, x], [best_m, m_o], color=MUTED, lw=1.4, zorder=2,
                    alpha=0.8)
            ax.plot([x - 0.09, x + 0.09], [chance, chance], color=MUTED,
                    lw=1.4, ls=(0, (3, 2)), zorder=2)
            for l in priors:
                v = _rate(raw, l, col)
                m = float(v.mean())
                lo, hi = _boot(v, seed=k * 10 + j)
                ax.plot([x, x], [lo, hi], color=LENS_COLOUR[l], lw=1.4,
                        alpha=0.55, zorder=3)
                ax.plot([x], [m], marker="o", color=LENS_COLOUR[l],
                        ms=9 if l == best else 7, mew=0, zorder=4)
            lo, hi = _boot(v_o, seed=100 + k * 10 + j)
            ax.plot([x, x], [lo, hi], color=style.BLUE, lw=1.6, alpha=0.6,
                    zorder=3)
            ax.plot([x], [m_o], marker="o", color=style.BLUE, ms=10, mew=0,
                    zorder=5)
            ax.annotate(f"{m_o / max(best_m, 1e-9):.1f}× the best prior",
                        xy=(x, hi), xytext=(0, 9), textcoords="offset points",
                        ha="center", va="bottom", color=INK2, fontsize=18.8)

    ax.set_xlim(-0.55, len(runs) - 0.45)
    ax.set_ylim(0, 0.57)
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks, fontsize=18.8)
    # the two measure names are far wider than the pair of slots they stand
    # over, so every other one is dropped clear of its neighbour
    style.stagger_xticks(ax, 2.4, fs=18.8)
    ax.set_ylabel("share of photographs")
    ax.annotate("chance", xy=(xs[0] - 0.10, MEAS[0][2]), xytext=(-6, 0),
                textcoords="offset points", ha="right", va="center",
                color=INK2, fontsize=18.8)
    # the model's name goes below the axis, under the pair it owns
    for k, run in enumerate(runs):
        ax.annotate(f"{run['label']}\n{run['n']} photographs",
                    xy=(k, 0), xytext=(0, -122),
                    xycoords=("data", "axes fraction"),
                    textcoords="offset points", ha="center", va="top",
                    color=INK, fontsize=21.8, linespacing=1.4)
    style.despine(ax)
    ax.margins(y=0.05)

    head_top = _figure_head(fig, "The gap, across checkpoints")
    have = set()
    for run in runs:
        have |= set(run["raw"].lens.replace(CANON).unique())
    _key(fig, head_top,
         [l for l in ("logit lens", "J-lens", "VJ-Lens") if l in have])
    unit = 1.0 / (72.0 * fig.get_figheight())
    fig.subplots_adjust(left=0.075, right=0.985, top=head_top - 30 * unit,
                        bottom=0.345)
    return _save(fig, out, name)


# ======================================================= D. depth profile

def depth_profile(panels, out, name="f83_depth.png", fig_w=17.6):
    """Every measure the paper reports at one depth, drawn against depth.

    ``panels`` is a list of ``{"title", "sub", "ylabel", "series", ...}`` where
    ``series`` maps a lens name to ``(depth_percent, value)``.  Any panel whose
    file was missing is simply absent: this figure never invents a measure.
    """
    style.use()
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 5.6))
    axes = np.atleast_1d(axes)
    # the panels take their final width before anything is written above them,
    # so a heading can be broken to the width it actually has
    fig.subplots_adjust(left=0.045, right=0.995, wspace=0.34)
    for ax, p in zip(axes, panels):
        if p.get("zero"):
            ax.axhline(0.0, color=GRID, lw=1.2, zorder=1)
        ref = p.get("ref")
        if ref is not None:
            ax.axhline(ref[0], color=MUTED, lw=1.2, ls=(0, (3, 2)), zorder=1)
            # named at the right-hand end: at the left the two prior lenses run
            # along the rule itself, and a name there lands on both of them
            ax.annotate(ref[1], xy=(0.985, ref[0]), xytext=(0, 6),
                        xycoords=("axes fraction", "data"),
                        textcoords="offset points", ha="right", va="bottom",
                        color=INK2, fontsize=18.8)
        for lens, (x, y) in p["series"].items():
            c = LENS_COLOUR.get(lens, MUTED)
            ls = p.get("dashed", {}).get(lens, "-")
            ax.plot(x, y, color=c, lw=2.2, ls=ls, marker="o", ms=6,
                    zorder=p.get("z", {}).get(lens, 3))
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("depth  (%)")
        ax.set_ylabel(p["ylabel"])
        if p.get("ylim"):
            ax.set_ylim(*p["ylim"])
        else:
            ax.margins(y=0.22)
        style.despine(ax)

    # five headings side by side, each as long as a short sentence: at the
    # paper's type size they ran into one another, so each one is broken to
    # the width of the panel it names and the row is given the room that costs
    heads = [(_wrap_to_axes(fig, ax, p["title"], 20.6),
              _wrap_to_axes(fig, ax, p["sub"], 18.8) if p.get("sub") else None)
             for ax, p in zip(axes, panels)]
    t_lines = max(t.count("\n") + 1 for t, _ in heads)
    s_lines = max((u.count("\n") + 1 for _, u in heads if u), default=0)
    for ax, (title, sub) in zip(axes, heads):
        _head(ax, title, sub, min_lines=s_lines)

    head_top = _figure_head(
        fig, "Where in the network each of these claims lives")
    key = _key(fig, head_top + 0.004,
               ["logit lens", "J-lens", "VJ-Lens"])
    unit = 1.0 / (72.0 * fig.get_figheight())
    top = _panel_top(fig, head_top - 22 * unit, s_lines,
                     title_lines=t_lines)
    fig.subplots_adjust(left=0.045, right=0.995, top=top, bottom=0.125,
                        wspace=0.34)
    return _save(fig, out, name)
