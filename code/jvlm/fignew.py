# SPDX-License-Identifier: Apache-2.0
"""Figures for the results that have no picture yet.

Four of them, one per result, and nothing here touches a model or a GPU:
every value drawn is read straight out of a result file that already exists,
or is a difference between two numbers in that file.  The one derived
quantity is named where it is drawn: the safety strip divides each lens's
per-pair change by that lens's own spread across images, because the three
lenses live on wildly different scales (the logit lens's readings have a
standard deviation of about 2.4e3 and the J-lens's about 3.0e3, against about
24 for ours) and a shared raw axis would flatter whichever lens happens to be
largest.  The raw maxima are printed by the driver and stated in the caption.

Layout notes, since the shared type scale is large: every panel here is sized
from the text it has to hold rather than the other way round.  Long row names
get a wide left margin, series are labelled at their ends instead of in a box
that would eat the plotting area, and where two panels share a y-axis the
right-hand one carries no end labels at all.
"""
from __future__ import annotations

import os

import numpy as np

from jvlm import style
from jvlm.style import AQUA, BLUE, GRID, INK, INK2, MUTED, ORANGE, RED, VIOLET

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator


# ------------------------------------------------------------------ furniture

def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def _headline(fig, head, sub=None, fs=25.0, fs_sub=16.0, gap=15.0):
    """Figure heading, one quieter line under it, and the top the axes get."""
    unit = 1.0 / (72.0 * fig.get_figheight())
    if not style.FIGURE_TITLES:
        return 1 - 9 * unit            # the caption names the figure
    fig.text(0.5, 1 - 9 * unit, head, ha="center", va="top", color=INK,
             fontsize=fs)
    y = 1 - (9 + fs * 1.30) * unit
    if sub:
        fig.text(0.5, y - 5 * unit, sub, ha="center", va="top", color=INK2,
                 fontsize=fs_sub)
        y -= (5 + fs_sub * 1.35) * unit
    return y - gap * unit


def _head_block(fs=19.0, sfs=14.5, sub=False):
    """Points of space a panel heading occupies above its own axes box."""
    return 9.0 + fs * 1.35 + (sfs * 1.5 if sub else 0.0)


def _axes_top(fig, top, sub=False, extra=0.0, fs=19.0, sfs=14.5):
    """Where the axes box must stop so its heading clears what is above it.

    ``subplots_adjust(top=...)`` sets the axes BOX; a title is drawn outside
    it.  Passing the headline's own bottom straight into ``top`` therefore
    prints every panel heading into the figure's subtitle, which is the
    collision this exists to stop.
    """
    unit = 1.0 / (72.0 * fig.get_figheight())
    return top - (_head_block(fs, sfs, sub) + extra) * unit


def _head(ax, text, sub=None, fs=19.0, sfs=14.5):
    """A panel heading, and exactly the room its own lines need."""
    ax.set_title(text, loc="center", color=INK, fontsize=fs,
                 pad=9 + (sfs * 1.45 if sub else 0.0))
    if sub:
        ax.annotate(sub, xy=(0.5, 1.0), xytext=(0, 7), xycoords="axes fraction",
                    textcoords="offset points", ha="center", va="bottom",
                    color=INK2, fontsize=sfs)


def _foot(fig, text, fs=13.0, y=0.012):
    fig.text(0.5, y, text, ha="center", va="bottom", color=MUTED, fontsize=fs)


# ============================================ 1. the pair, and the exact 0.500

def pair_safety(strips, auroc, out, name="f143_pairsafety.png",
                null_mean=0.5, null_p95=0.54, foot=None, seed=0):
    """Same image, two requests: whose reading moves, and what that scores.

    ``strips`` is ``[(lens, values, colour), ...]`` top to bottom, where
    ``values`` is the per-pair change in that lens's reading expressed in that
    lens's own across-image standard deviation.  ``auroc`` is
    ``[{label, value, n, group}, ...]`` top to bottom; ``group`` only decides
    where the extra white space falls.

    A bar at 0.5 reads as a weak classifier, which is the opposite of the
    point, so the fixed lenses are never given a bar.  They are drawn as a
    ring threaded on the 0.500 line in every row, and the left panel shows
    why they are there: their reading for the two requests is the same
    number, so the score cannot be anything else.
    """
    style.use()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(21.0, 8.8),
                                   gridspec_kw={"width_ratios": [1.0, 1.18]})

    # ---- left: does the reading move at all?
    rng = np.random.default_rng(seed)
    n = len(strips)
    for i, (lab, vals, c) in enumerate(strips):
        y = n - i
        v = np.asarray(vals, float)
        flat = bool(np.all(v == v[0]))
        axL.scatter(v, y + rng.uniform(-0.27, 0.27, size=len(v)), s=26,
                    color=c, alpha=0.45, linewidths=0, zorder=3)
        if flat:
            axL.plot([v[0]], [y], marker="o", ms=15, mfc="none", mec=c,
                     mew=2.6, zorder=5)
            axL.annotate(f"all {len(v)} pairs at exactly {v[0]:.2f}",
                         xy=(v[0], y), xytext=(17, 0),
                         textcoords="offset points", ha="left", va="center",
                         fontsize=15.5, color=c)
        else:
            med = float(np.median(v))
            axL.plot([med, med], [y - 0.38, y + 0.38], color=INK, lw=2.6,
                     zorder=5)
            axL.annotate(f"median {med:.2f}", xy=(med, y + 0.42),
                         xytext=(0, 7), textcoords="offset points",
                         ha="center", va="bottom", fontsize=15.5, color=INK)
    axL.axvline(0.0, color=INK2, lw=1.1, zorder=1)
    axL.set_yticks(list(range(n, 0, -1)))
    axL.set_yticklabels([s[0] for s in strips], fontsize=16.5)
    axL.set_ylim(0.30, n + 0.95)
    lo = min(float(np.min(s[1])) for s in strips)
    hi = max(float(np.max(s[1])) for s in strips)
    span = hi - lo
    axL.set_xlim(lo - 0.10 * span, hi + 0.10 * span)
    axL.set_xlabel("change in the reading when only the request changes\n"
                   "(in that lens's own spread across images)")
    axL.grid(axis="x")
    axL.grid(axis="y", visible=False)
    style.despine(axL)
    _head(axL, "Change under a new request")

    # ---- right: what that is worth as a score
    ys, y, prev = [], 0.0, None
    for r in auroc[::-1]:
        if prev is not None and r["group"] != prev:
            y += 0.62
        ys.append(y)
        y += 1.0
        prev = r["group"]
    ys = ys[::-1]
    top = max(ys)

    axR.axvspan(null_mean, null_p95, color="#eeedea", zorder=0, lw=0)
    axR.axvline(0.5, color=INK2, lw=2.2, zorder=2)
    for r, yy in zip(auroc, ys):
        big = r["group"] == "pooled"
        axR.plot([r["value"]], [yy], marker="o", ms=15 if big else 12,
                 color=BLUE, mew=0, zorder=5)
        axR.annotate(f"{r['value']:.2f}", xy=(r["value"], yy), xytext=(11, 0),
                     textcoords="offset points", ha="left", va="center",
                     fontsize=17 if big else 16, color=INK)
        axR.plot([0.5], [yy], marker="o", ms=16, mfc="none", mec=ORANGE,
                 mew=2.4, zorder=6)
        axR.plot([0.5], [yy], marker="o", ms=9, mfc="none", mec=VIOLET,
                 mew=2.2, zorder=7)
        axR.annotate(f"n = {r['n']}", xy=(0.995, yy), xycoords=("axes fraction",
                                                                "data"),
                     ha="right", va="center", fontsize=14.5, color=MUTED)
        if r.get("note"):
            axR.annotate(r["note"], xy=(r["value"], yy), xytext=(11, -22),
                         textcoords="offset points", ha="left", va="center",
                         fontsize=14.5, color=MUTED)
    axR.annotate("VJ-Lens", xy=(auroc[0]["value"], top + 0.95),
                 ha="center", va="center", fontsize=16.5, color=INK2)
    axR.annotate("logit lens and J-lens", xy=(0.5, top + 0.95),
                 xytext=(-13, 0), textcoords="offset points", ha="right",
                 va="center", fontsize=16.5, color=INK2)
    axR.annotate("shuffled floor", xy=(null_p95, -0.62), xytext=(8, 0),
                 textcoords="offset points", ha="left", va="center",
                 fontsize=14.5, color=MUTED)
    axR.set_yticks(ys)
    axR.set_yticklabels([r["label"] for r in auroc], fontsize=16.5)
    axR.set_ylim(-0.95, top + 1.55)
    axR.set_xlim(0.155, 0.94)
    axR.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    axR.set_xlabel("AUROC: attack request against benign request, same image")
    axR.grid(axis="x")
    axR.grid(axis="y", visible=False)
    style.despine(axR)
    _head(axR, "Attack against benign request")

    top_f = _headline(
        fig, "Same picture, different request",
        "a lens that cannot see the request scores exactly 0.50, by "
        "construction and not by weakness")
    if foot:
        _foot(fig, foot)
    fig.subplots_adjust(left=0.085, right=0.985,
                        top=_axes_top(fig, top_f), bottom=0.185, wspace=0.24)
    return _save(fig, out, name)


# ================================================== 2. the bounded steering edit

def bounded_steering(series, out, name="f144_bounded.png", foot=None,
                     ylabel="change in the answer's logit"):
    """On-target and off-target effect of the same edit, at the same scale.

    ``series`` is ``[(label, colour, sizes, on, off), ...]``; ``sizes`` are
    the edit sizes as a share of the activation's own norm.  Both panels share
    one y-axis so that the right-hand panel's flatness is a measurement and
    not a change of scale, and only the left panel carries end labels: four
    names stacked on a bundle of lines that all sit on zero would be four
    labels pointing at the same place.
    """
    style.use()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(19.0, 8.6), sharey=True)
    ends = []
    for lab, c, sizes, on, off in series:
        x = np.asarray(sizes, float) * 100.0
        axA.plot(x, on, color=c, lw=2.6, marker="o", ms=8, zorder=3)
        axB.plot(x, off, color=c, lw=2.6, marker="o", ms=8, zorder=3)
        ends.append((float(x[-1]), float(on[-1]), lab, c))
    for ax in (axA, axB):
        ax.axhline(0.0, color="#c9c7c1", lw=1.3, zorder=1)
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20])
        ax.set_xticklabels(["2%", "5%", "10%", "20%"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(1.72, 46)
        ax.set_xlabel("edit size (share of the activation's own norm)")
        ax.grid(axis="x", visible=False)
        style.despine(ax)
    axA.set_ylabel(ylabel)

    lo = min(min(min(s[3]), min(s[4])) for s in series)
    hi = max(max(max(s[3]), max(s[4])) for s in series)
    pad = 0.12 * (hi - lo)
    axA.set_ylim(lo - pad, hi + pad)
    span = hi - lo
    for x, y, lab, c in ends:
        # a series that finishes on zero has the zero rule running through
        # its own name; lift that one label off the rule and lead back to it
        if abs(y) < 0.07 * span:
            style.label_end(axA, x, y + 0.07 * span, lab, c, dx=9, y_true=y,
                            fontsize=16.5)
        else:
            style.label_end(axA, x, y, lab, c, dx=9, fontsize=16.5)

    worst = max(max(abs(np.asarray(s[4], float))) for s in series)
    axB.annotate(f"every direction, every size: at most {worst:.2f}",
                 xy=(0.5, 0.30), xycoords="axes fraction", ha="center",
                 va="center", fontsize=16.5, color=INK2)

    _head(axA, "the answer the edit is aimed at")
    _head(axB, "the answer it is not aimed at")
    top = _headline(
        fig, "A bounded edit, and what it leaves alone",
        "the aimed-at answer scales with the edit; the other answer, and a "
        "random direction of the same size, do not")
    if foot:
        _foot(fig, foot)
    fig.subplots_adjust(left=0.075, right=0.985, top=_axes_top(fig, top),
                        bottom=0.135, wspace=0.09)
    return _save(fig, out, name)


# ================================================== 3. the double dissociation

def crossover(panels, out, name="f145_crossover.png", foot=None,
              xlabels=("the answer about A", "the answer about B"),
              ylabel="cost to that answer's logit\nwhen the region is covered"):
    """The 2x2 as two crossing lines, once per dataset.

    ``panels`` is ``[(heading, rows), ...]`` and ``rows`` is
    ``[(label, colour, (a, a_lo, a_hi), (b, b_lo, b_hi)), ...]``.  A 2x2 of
    numbers makes the reader do the crossing in their head; two lines that
    cross do it on the page, and the interval bars keep the honest width of
    each cell visible.
    """
    style.use()
    k = len(panels)
    fig, axes = plt.subplots(1, k, figsize=(6.6 * k, 8.6), sharey=True)
    axes = np.atleast_1d(axes)
    lo, hi = 0.0, 0.0
    for ax, (heading, rows) in zip(axes, panels):
        ax.axhline(0.0, color="#c9c7c1", lw=1.3, zorder=1)
        for lab, c, a, b in rows:
            ax.plot([0, 1], [a[0], b[0]], color=c, lw=2.6, zorder=3,
                    label=lab)
            for x, t in ((0, a), (1, b)):
                ax.plot([x, x], [t[1], t[2]], color=c, lw=2.0, alpha=0.45,
                        zorder=2, solid_capstyle="butt")
                ax.plot([x], [t[0]], marker="o", ms=12, color=c, mew=0,
                        zorder=4)
            lo = min(lo, a[1], b[1])
            hi = max(hi, a[2], b[2])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(list(xlabels), fontsize=16.0)
        ax.set_xlim(-0.42, 1.42)
        ax.grid(axis="x", visible=False)
        style.despine(ax)
        _head(ax, heading)
    span = hi - lo
    axes[0].set_ylim(lo - 0.10 * span, hi + 0.12 * span)
    axes[0].set_ylabel(ylabel)

    handles = [Line2D([], [], color=c, lw=2.6, marker="o", ms=8, label=lab)
               for lab, c, _, _ in panels[0][1]]
    top = _headline(
        fig, "Covering one object costs one answer",
        "the same edit, read twice: the cost lands on the answer whose object "
        "was covered, and not on the other")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, top),
               ncols=len(handles), frameon=False, handlelength=1.6,
               handletextpad=0.6, columnspacing=2.6, fontsize=16.5,
               labelcolor=INK2)
    if foot:
        _foot(fig, foot)
    unit = 1.0 / (72.0 * fig.get_figheight())
    fig.subplots_adjust(left=0.105, right=0.99,
                        top=_axes_top(fig, top - 16.5 * 2.4 * unit),
                        bottom=0.135, wspace=0.10)
    return _save(fig, out, name)


# ============================================= 4. the comparison and its controls

def comparison_controls(methods, margins, out, name="f146_fairsweep.png",
                        bracket=None, chance_label="chance",
                        left_sub=None, right_sub=None, notes=("", ""),
                        foot=None, xticks=(1, 4, 9, 16, 36, 49)):
    """Every competitor at its own best setting, and it still loses.

    ``methods`` is ``[(label, lift, colour), ...]`` in the order drawn, top
    to bottom; ``lift`` is top-1 as a multiple of chance, so the stems run
    from 1 rather than from 0.  ``margins`` is
    ``[(label, colour, sizes, delta, lo, hi), ...]``.  ``bracket`` is a pair
    of row indices and a short label for the gap between them.
    """
    style.use()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(21.5, 8.8),
                                   gridspec_kw={"width_ratios": [1.28, 1.0]})

    # ---- left: the sweep, as multiples of chance
    m = len(methods)
    ys = list(range(m - 1, -1, -1))
    axA.axvline(1.0, color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=1)
    for (lab, lift, c), yy in zip(methods, ys):
        axA.plot([1.0, lift], [yy, yy], color=c, lw=2.4, zorder=2,
                 solid_capstyle="butt")
        axA.plot([lift], [yy], marker="o", ms=13, color=c, mew=0, zorder=4)
        axA.annotate(f"{lift:.1f}x", xy=(lift, yy), xytext=(11, 0),
                     textcoords="offset points", ha="left", va="center",
                     fontsize=17, color=INK)
    axA.annotate(chance_label, xy=(1.0, -0.72), xytext=(7, 0),
                 textcoords="offset points", ha="left", va="center",
                 fontsize=14.5, color=MUTED)
    top_x = max(v for _, v, _ in methods)
    axA.set_xlim(0, top_x * 1.42)
    if bracket:
        i, j, blab = bracket
        ya, yb = ys[i], ys[j]
        xa, xb = methods[i][1], methods[j][1]
        ym = 0.5 * (ya + yb)
        axA.annotate("", xy=(xa, ym), xytext=(xb, ym),
                     arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.5,
                                     shrinkA=0, shrinkB=0), zorder=5)
        axA.annotate(blab, xy=(xa, ym), xytext=(14, 0),
                     textcoords="offset points", ha="left", va="center",
                     fontsize=15.5, color=INK2, linespacing=1.35)
    axA.set_yticks(ys)
    axA.set_yticklabels([lab for lab, _, _ in methods], fontsize=16.5)
    axA.set_ylim(-1.05, m - 0.35)
    axA.set_xlabel("top-1 region, as a multiple of chance")
    axA.grid(axis="x")
    axA.grid(axis="y", visible=False)
    style.despine(axA)
    _head(axA, "against every competitor", left_sub)
    if notes[0]:
        axA.annotate(notes[0], xy=(0.995, 0.02), xycoords="axes fraction",
                     ha="right", va="bottom", fontsize=14.0, color=MUTED)

    # ---- right: the margin against region size
    axB.axhline(0.0, color=INK2, lw=1.1, zorder=1)
    ends = []
    for lab, c, sizes, d, blo, bhi in margins:
        x = np.asarray(sizes, float)
        axB.fill_between(x, blo, bhi, color=c, alpha=0.16, lw=0, zorder=2)
        axB.plot(x, d, color=c, lw=2.6, marker="o", ms=8, zorder=3)
        ends.append((float(x[-1]), float(np.asarray(d)[-1]), lab, c))
    axB.set_xscale("log")
    axB.set_xticks(list(xticks))
    axB.set_xticklabels([str(t) for t in xticks])
    axB.xaxis.set_minor_locator(NullLocator())
    axB.set_xlim(0.82, 150)
    ylo = min(0.0, min(min(r[4]) for r in margins))
    yhi = max(max(r[5]) for r in margins)
    axB.set_ylim(ylo - 0.13 * (yhi - ylo), yhi + 0.05 * (yhi - ylo))
    axB.set_xlabel("patches in the region")
    axB.set_ylabel("margin over that competitor  (Spearman rho)")
    axB.grid(axis="x", visible=False)
    style.despine(axB)
    for x, y, lab, c in ends:
        style.label_end(axB, x, y, lab, c, dx=9, fontsize=16.5)
    _head(axB, "and at every region size", right_sub)
    if notes[1]:
        axB.annotate(notes[1], xy=(0.995, 0.02), xycoords="axes fraction",
                     ha="right", va="bottom", fontsize=14.0, color=MUTED)

    top = _headline(
        fig, "The comparison, with its controls",
        "every competitor is scored at its own best setting, and the margin "
        "grows as the region grows")
    if foot:
        _foot(fig, foot)
    fig.subplots_adjust(left=0.135, right=0.985,
                        top=_axes_top(fig, top, sub=True), bottom=0.135,
                        wspace=0.30)
    return _save(fig, out, name)
