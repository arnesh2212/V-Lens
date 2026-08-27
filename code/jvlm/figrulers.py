# SPDX-License-Identifier: Apache-2.0
"""Two ground truths for one claim, drawn against each other.

The causal map this paper grades against covers a region by replacing its
*activations* with the corpus mean -- the exact intervention the reading is
defined against.  The conservative ruler covers the same region in the
*picture*: grey those pixels out and re-encode, so the vision tower runs again
and nothing about the lens's baseline is involved.  The two rulers do not agree
in size, and the whole point of these figures is to show the disagreement
rather than to report the flattering half of it.

Conventions are borrowed from :mod:`jvlm.figcausal` on purpose -- one colour
per lens everywhere in the paper, square tiles, titles padded by the line count
of their own subtitle, end-of-line labels staggered when two series finish at
the same value.
"""
from __future__ import annotations

import os

import numpy as np

from jvlm import style
from jvlm.figcausal import LENS_COLOUR
from jvlm.style import BLUE, GRID, INK, INK2, MUTED, SURFACE

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch, Rectangle

#: the three lenses measured under BOTH rulers.  The point-sample variant of
#: our reading exists only under the activation ruler, so it cannot appear in a
#: figure whose whole subject is the difference between the two.
LENSES = ["logit lens", "J-lens", "VJ-Lens"]

HOT = "#b8481f"
PANEL = "#eef2f6"
SOFT = "#a5cce4"
NEUTRAL = "#d5dbe3"          # a truth is not a lens, so it wears no lens hue

ACT = "covered in the\nactivations"
PIX = "greyed out in\nthe picture"


# ------------------------------------------------------------------ utilities

def _boot(v, n: int = 4000, seed: int = 0):
    """Mean and a 95% bootstrap interval, resampling PHOTOGRAPHS."""
    v = np.asarray(v, dtype=float)
    if not len(v):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(1)
    return (float(v.mean()), float(np.percentile(m, 2.5)),
            float(np.percentile(m, 97.5)))


def _per_item(df, lens: str, col: str):
    """One number per photograph, so the bootstrap resamples photographs.

    Each photograph contributes several rows -- one per depth -- and treating
    those as independent would shrink every interval in the figure by a factor
    it has not earned.
    """
    d = df[df.lens == lens]
    if not len(d) or col not in d:
        return np.array([])
    return d.groupby("i")[col].mean().values


def _head(ax, title, sub, sub_colour=None, fs=9.6, tfs=12.5):
    """Panel title with a quieter second line; the pad follows the line count."""
    lines = str(sub).count("\n") + 1 if sub else 0
    pad = 9 + lines * fs * 1.36
    ax.set_title(title, loc="left", color=INK, pad=pad, fontsize=tfs)
    if sub:
        ax.annotate(sub, xy=(0, 1), xytext=(0, 6), xycoords="axes fraction",
                    textcoords="offset points", ha="left", va="bottom",
                    color=sub_colour or INK2, fontsize=fs, linespacing=1.36)


def _figtext(fig, head, sub, fs_head=16.0, fs=10.8, x=0.006,
             width=225, pad_pts=0.0):
    """Figure title and subtitle, and the top the axes may rise to.

    The subtitle is WRAPPED here rather than trusted to the newlines an author
    typed: a line that runs past the figure prints straight over the first
    panel's title, which is exactly what the first draft of this figure did.
    """
    import textwrap

    sub = "\n".join(textwrap.fill(par, width) for par in sub.split("\n"))
    unit = 1.0 / (72.0 * fig.get_figheight())
    fig.text(x, 1 - 6 * unit, head, ha="left", va="top", color=INK,
             fontsize=fs_head)
    y = 1 - (6 + fs_head * 1.22 + 8) * unit
    fig.text(x, y, sub, ha="left", va="top", color=INK2, fontsize=fs,
             linespacing=1.45)
    return (y - (sub.count("\n") + 1) * fs * 1.45 * unit
            - (13 + pad_pts) * unit)


def _stagger(ends, y0, y1, gap_frac=0.155):
    """Push labels apart along y, keeping their order, inside the axes.

    Two lenses that finish within a hair of each other -- and under the pixel
    ruler the two prior lenses finish 0.008 apart -- print their names on top
    of one another unless the labels are moved and led back to their lines.
    """
    gap = (y1 - y0) * gap_frac
    ends = sorted(ends, key=lambda e: e["y"])
    for e in ends:
        e["y_true"] = e["y"]
    for i in range(1, len(ends)):
        if ends[i]["y"] - ends[i - 1]["y"] < gap:
            ends[i]["y"] = ends[i - 1]["y"] + gap
    over = ends[-1]["y"] - (y1 - gap * 0.40)
    if over > 0:
        for e in ends:
            e["y"] -= over
    under = (y0 + gap * 0.40) - ends[0]["y"]
    if under > 0:
        for e in ends:
            e["y"] += under
    return ends


def _num(v, pct: bool) -> str:
    if pct:
        return f"{v * 100:.0f}%"
    s = f"{v:+.2f}"
    return s.replace("-", "−")


# ------------------------------------------------------- A. the two rulers

def _slope(ax, rows, ylab, head, sub, chance=None, pct=False, mult_for=None):
    """One measure, both rulers, one line per lens.

    Two x positions and nothing else on the axis: the figure is a comparison of
    two rulers, and anything else drawn here competes with that.
    """
    span = [v for r in rows.values() for st in r for v in st[1:]]
    if chance is not None:
        span.append(chance)
    lo, hi = min(span), max(span)
    pad = (hi - lo) * 0.16 or 0.05
    y0, y1 = lo - pad, hi + pad * 1.5
    ax.set_ylim(y0, y1)
    ax.set_xlim(-0.30, 2.42)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([ACT, PIX], fontsize=10.6, linespacing=1.3)

    # the dashed line is named in this panel's own subtitle rather than beside
    # itself: every position an inline label can take here is one the three
    # lens lines already pass through
    # it stops before the label column: run across the whole axes it prints
    # through the two-line names, which is where the first draft put it
    ax.plot([-0.30, 1.15], [chance if chance is not None else 0.0] * 2,
            color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)

    ends = []
    for k, lens in enumerate(LENSES):
        (m0, l0, h0), (m1, l1, h1) = rows[lens]
        c = LENS_COLOUR.get(lens, MUTED)
        j = (k - (len(LENSES) - 1) / 2) * 0.055
        ax.plot([0 + j, 1 + j], [m0, m1], color=c, lw=2.2, marker="o",
                ms=7.5, mew=0, zorder=3)
        for x, a, b in ((0 + j, l0, h0), (1 + j, l1, h1)):
            ax.plot([x, x], [a, b], color=c, lw=1.6, alpha=0.45, zorder=2,
                    solid_capstyle="butt")
        t = f"{lens}\n{_num(m0, pct)} → {_num(m1, pct)}"
        if mult_for == lens and chance:
            # the multiple of chance goes INSIDE the label; drawn beside its
            # own dot it lands on top of the label, which is where the first
            # draft printed it
            t += f"\n{m0 / chance:.0f}× → {m1 / chance:.0f}× chance"
        ends.append(dict(x=1 + j, y=m1, c=c, t=t))

    for e in _stagger(ends, y0, y1, 0.175):
        style.label_end(ax, e["x"], e["y"], e["t"], e["c"],
                        y_true=e["y_true"], fontsize=10.4, linespacing=1.32)

    ax.set_ylabel(ylab)
    if pct:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    _head(ax, head, sub)
    style.despine(ax)


def _paired(ax, act, pix, lens="VJ-Lens"):
    """The same photographs under both rulers, one dot each.

    The two runs do not cover the same number of photographs, so the headline
    shrinkage could in principle be a difference of sample rather than of
    ruler.  These are the photographs measured under both.
    """
    ov = sorted(set(act.i.unique()) & set(pix.i.unique()))
    A = act[(act.lens == lens) & act.i.isin(ov)].groupby("i").spearman.mean()
    P = pix[(pix.lens == lens) & pix.i.isin(ov)].groupby("i").spearman.mean()
    A, P = A.loc[ov].values, P.loc[ov].values
    c = LENS_COLOUR.get(lens, BLUE)
    lo = min(A.min(), P.min()); hi = max(A.max(), P.max())
    pad = (hi - lo) * 0.10
    lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0, color=GRID, lw=1.0, zorder=1)
    ax.axvline(0, color=GRID, lw=1.0, zorder=1)
    ax.scatter(A, P, s=38, color=c, alpha=0.62, linewidths=0, zorder=3)
    ax.plot([A.mean()], [P.mean()], marker="D", ms=11, mfc=SURFACE, mec=c,
            mew=2.0, zorder=4)
    below = int((P < A).sum())
    ax.annotate(f"{below} of {len(A)} photographs fall below the line",
                xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                va="bottom", color=INK2, fontsize=10.0)
    ax.set_xlabel("ρ against the activation ruler")
    ax.set_ylabel("ρ against the pixel ruler")
    ax.grid(axis="both")
    _head(ax, "photograph by photograph",
          f"our reading on the {len(A)} photographs measured under both;\n"
          f"the open diamond is the pair of means, {A.mean():.2f} and "
          f"{P.mean():.2f}")
    style.despine(ax)
    return len(A), below, float(A.mean()), float(P.mean())


def tworulers(act, pix, out, name="f74_tworulers.png"):
    """Agreement, top-1 and top-3 under both rulers, plus the paired check."""
    style.use()
    rows = {}
    for col in ("spearman", "hit_cause", "top3_cause"):
        rows[col] = {l: (_boot(_per_item(act, l, col)),
                         _boot(_per_item(pix, l, col))) for l in LENSES}
    R = int(act.R.iloc[0]) if "R" in act else 49

    fig, axes = plt.subplots(1, 4, figsize=(17.4, 6.6))
    _slope(axes[0], rows["spearman"],
           "Spearman ρ over all 49 regions",
           "agreement with the whole map",
           "how well the reading ranks every region by what covering it\n"
           "costs; the dashed line is no relation at all",
           chance=None)
    _slope(axes[1], rows["hit_cause"],
           "share of photographs",
           "names the single most damaging region",
           "the region whose covering costs the model's own word the\n"
           f"most; the dashed line is chance, 1 region in {R}",
           chance=1.0 / R, pct=True, mult_for="VJ-Lens")
    _slope(axes[2], rows["top3_cause"],
           "share of photographs",
           "names one of the three most damaging",
           "the looser question the paper's claim is made on; the dashed\n"
           f"line is chance, 3 regions in {R}",
           chance=3.0 / R, pct=True, mult_for="VJ-Lens")
    n_p, below, ma, mp = _paired(axes[3], act, pix)

    na, np_ = act.i.nunique(), pix.i.nunique()
    da, dp = act.depth.nunique(), pix.depth.nunique()
    sub = (
        "Left of each of the first three panels: cover a region by replacing "
        "its activations with the corpus mean — the intervention this lens is "
        "defined against, so the comparison is partly won by construction.  "
        "Right: grey the same region out of the photograph and re-encode it, "
        "an intervention the lens has never seen.  Whiskers are 95% bootstrap "
        "intervals over photographs.  Our agreement with the whole map falls "
        "to under a third of itself when the ruler changes, "
        f"{rows['spearman']['VJ-Lens'][0][0]:.3f} to "
        f"{rows['spearman']['VJ-Lens'][1][0]:.3f}, and that is the honest size "
        "of it; what survives is the part the claim rests on — naming the "
        "region that matters, at several times chance, where both prior lenses "
        "sit at chance or below and turn negative on the whole map.\n"
        f"{na} photographs at {da} depths against the activation ruler, {np_} "
        f"at {dp} depths against the pixel one; the point-sample variant of "
        "our reading was run only against the activation ruler and is left out "
        f"of a figure whose subject is the difference between the two.  "
        f"Rightmost: the {n_p} photographs measured under both, our reading on "
        f"each, mean ρ {ma:.3f} against {mp:.3f}.")
    # the panel titles rise above the axes rectangle, so the space they need
    # is reserved here rather than discovered by collision
    top = _figtext(fig, "Two rulers for one claim, and they do not agree in size",
                   sub, pad_pts=9 + 2 * 9.6 * 1.36 + 12.5 * 1.35)
    fig.subplots_adjust(left=0.052, right=0.995, top=top, bottom=0.115,
                        wspace=0.44)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


# ------------------------------------------------- B. the box dissociation

def _box_ranks(cards):
    """Where the human box's own regions rank in the true causal ordering.

    Average ranks, because the measured drops are quantised to sixteenths of a
    logit and ties are the rule rather than the exception; breaking them by
    array order would invent a structure the measurement does not have.
    """
    from scipy import stats as sps

    pooled, per_card = [], []
    for c in cards:
        truth = np.asarray(c["truth"], dtype=float)
        pos = {int(cell): j for j, cell in enumerate(c["keep"])}
        r = sps.rankdata(-truth, method="average")
        ins = [float(r[pos[t]]) for t in c["inside"] if t in pos]
        if not ins:
            continue
        pooled += ins
        per_card.append(np.median(ins))
    return np.asarray(pooled), np.asarray(per_card)


def boxdissociation(act, pix, cards, out, name="f75_boxmiss.png"):
    """The most damaging region is not in the object's box, under either ruler."""
    style.use()
    R = int(act.R.iloc[0]) if "R" in act else 49
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.9))

    # ---- panel 1: how often the truth lands inside the box, against chance
    ax = axes[0]
    bars = []
    for df, lab in ((act, ACT), (pix, PIX)):
        tib = _boot(df.groupby("i").truth_in_box.mean().values)
        area = float(df.groupby("i").box_area.mean().mean())
        bars.append((lab, tib, area, df))
    x = np.arange(len(bars))
    ax.bar(x, [b[1][0] for b in bars], width=0.5, color=NEUTRAL,
           edgecolor=INK2, linewidth=0.9, zorder=2)
    dots = False
    for i, (lab, (m, lo, hi), area, df) in enumerate(bars):
        ax.plot([i, i], [lo, hi], color=INK2, lw=1.6, zorder=4)
        ax.plot([i - 0.30, i + 0.30], [area, area], color=INK, lw=1.7,
                ls=(0, (3, 2)), zorder=5, clip_on=False)
        # the value sits at the foot of its own bar: above it is the whisker,
        # and on the pixel side the whisker reaches the dashed line
        ax.annotate(f"{m * 100:.0f}%", xy=(i, 0), xytext=(0, 7),
                    textcoords="offset points", ha="center", va="bottom",
                    color=INK, fontsize=12.5)
        ax.annotate(f"{area * 100:.0f}%", xy=(i, area), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    color=INK, fontsize=11.5)
        d = df.groupby("depth").truth_in_box.mean()
        # only the activation ruler has a depth: pixel occlusion happens before
        # the model runs, so drawing two identical dots there says nothing
        if d.nunique() > 1:
            dots = True
            # to the right of the whisker, not through it
            xs = np.linspace(i + 0.07, i + 0.24, len(d))
            ax.plot(xs, d.values, marker="o", ms=6.5, lw=0, mfc=SURFACE,
                    mec=INK2, mew=1.4, zorder=6, clip_on=False)
    top = max(max(b[1][2] for b in bars), max(b[2] for b in bars))
    ax.set_ylim(0, top * 1.34)
    ax.set_xlim(-0.62, len(bars) - 0.38)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=10.6, linespacing=1.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("share of photographs")
    if dots:
        ax.plot([], [], marker="o", ms=6.5, lw=0, mfc=SURFACE, mec=INK2,
                mew=1.4, label="one depth of covering, shallow to deep")
        style.legend(ax, loc="upper left")
    _head(ax, "how often it lands inside the human's box",
          "the dashed line is the box's own share of the grid, which is what a\n"
          "dart would score; only the activation ruler has a depth, since "
          "greyed-out\npixels are gone before the model runs at all")
    style.despine(ax)

    # ---- panel 2: the share of the damage the box holds, against its area
    ax = axes[1]
    g = act.groupby("i").agg(area=("box_area", "mean"),
                             share=("box_share", "mean"))
    lo = 0.0
    hi = float(max(g.area.max(), g.share.max())) * 1.06
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.scatter(g.area, g.share, s=34, color=INK2, alpha=0.42, linewidths=0,
               zorder=3)
    ax.plot([g.area.mean()], [g.share.mean()], marker="D", ms=11, mfc=SURFACE,
            mec=INK, mew=2.0, zorder=4)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("share of the grid the box covers")
    ax.set_ylabel("share of the total damage inside it")
    ax.grid(axis="both")
    _head(ax, "the box holds no more of the damage than its area",
          f"one dot per photograph, {act.i.nunique()} of them; on the dashed "
          "line the box\nis worth exactly its own area.  The open diamond is "
          f"the pair of means,\n{g.area.mean() * 100:.0f}% of the grid and "
          f"{g.share.mean() * 100:.0f}% of the damage")
    style.despine(ax)

    # ---- panel 3: where the box's own regions rank
    ax = axes[2]
    pooled, per_card = _box_ranks(cards)
    xs = np.sort(pooled)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.plot([0, R], [0, 1], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.step(np.concatenate([[0], xs]), np.concatenate([[0], ys]), where="post",
            color=BLUE, lw=2.2, zorder=3)
    ax.set_xlim(0, R); ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(f"the region's true rank, 1 of {R} = most damaging")
    ax.set_ylabel("share of the box's regions ranked that high")
    med = float(np.median(pooled))
    ax.plot([med, med], [0, 0.5], color=INK2, lw=1.0, ls=(0, (2, 3)), zorder=2)
    ax.annotate(f"median rank {med:.0f} of {R};\nchance is {(R + 1) / 2:.0f}",
                xy=(med, 0.28), xytext=(9, 0), textcoords="offset points",
                ha="left", va="center", color=INK2, fontsize=10.6,
                linespacing=1.34)
    ax.grid(axis="both")
    _head(ax, "the box's regions rank where a dart would put them",
          f"every region of the {len(per_card)} photographs whose full map was "
          "saved,\npooled, with ties held at their average rank; the dashed "
          "line is\nwhat a random subset of the regions would give")
    style.despine(ax)

    sub = (
        "The pointing game asks whether a map's strongest cell lands in a "
        "human's bounding box.  That is the wrong test here, and this is why: "
        "on these photographs the box is not where the answer is decided.  "
        "Left, under both rulers, the single most damaging region falls inside "
        "the object's own box less often than a dart thrown at the picture "
        "would — removing the object from the pixels before the model has seen "
        "it at all gives the same answer as removing it from the activations, "
        "so this is not an artefact of reading at one depth.  On 40 "
        "photographs the pixel interval does reach its own chance line and the "
        "activation interval, on 70, does not.\n"
        "Middle and right use the activation run, the only one of the two that "
        "stored the map itself.  Whiskers and intervals are 95% bootstrap over "
        "photographs.")
    top = _figtext(fig, "The object's box is not where the answer is decided",
                   sub, pad_pts=9 + 2 * 9.6 * 1.36 + 12.5 * 1.35)
    fig.subplots_adjust(left=0.055, right=0.988, top=top, bottom=0.145,
                        wspace=0.34)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


# ----------------------------------------------------- C. the pixel gallery

def _box_outline(ax, inside, coarse, side, colour):
    """The outline of the box's cells as one shape, not a grid of squares.

    Outlining every cell separately draws a lattice over the photograph that
    reads as part of the measurement; only the boundary of the union is the
    human's box.  Each segment is drawn twice, a pale halo under a dark dash,
    because a single hue is invisible on half of these photographs.
    """
    ins = set(int(c) for c in inside)
    segs = []
    for cell in ins:
        cy, cx = divmod(cell, coarse)
        x0, y0 = cx * side, cy * side
        if cy == 0 or (cell - coarse) not in ins:
            segs.append(((x0, y0), (x0 + side, y0)))
        if cy == coarse - 1 or (cell + coarse) not in ins:
            segs.append(((x0, y0 + side), (x0 + side, y0 + side)))
        if cx == 0 or (cell - 1) not in ins:
            segs.append(((x0, y0), (x0, y0 + side)))
        if cx == coarse - 1 or (cell + 1) not in ins:
            segs.append(((x0 + side, y0), (x0 + side, y0 + side)))
    for (a, b) in segs:
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#ffffff", lw=2.6,
                alpha=0.85, solid_capstyle="butt", zorder=4)
    for (a, b) in segs:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, lw=1.5,
                ls=(0, (3, 2)), solid_capstyle="butt", zorder=5)


def pixel_gallery(pix, items, out, name="f76_pixelphotos.png", per_row=10,
                  square=448, coarse=7):
    """Every photograph of the pixel-occlusion run, and how each lens did on it.

    This is deliberately NOT the card of :func:`jvlm.figcausal.card`.  That
    panel needs the per-region truth and each lens's top word for every region,
    and the pixel run stored neither -- only, per photograph and per lens,
    whether the region it called strongest was the most damaging one.  So the
    photographs are real, the human's box is real, and the claim in each cell is
    exactly as strong as what was recorded.

    The grid is laid out in inches with ``add_axes`` rather than by gridspec:
    a row of square photographs and a row of thin cells differ in height by a
    factor of five, and one ``hspace`` cannot open a small gap between the thin
    rows and a larger one between the bands.
    """
    from PIL import Image

    style.use()
    order = [l for l in LENSES if l in set(pix.lens)]
    depths = sorted(pix.depth.unique())
    by_i = {int(r["i"]): r for r in items}
    ids = sorted(pix.i.unique(),
                 key=lambda i: (by_i[int(i)]["concept"], int(i)))
    n = len(ids)
    ncol = min(per_row, n)
    nband = int(np.ceil(n / ncol))

    # ---- the whole layout, in inches
    cw, gx = 1.32, 0.10                       # photograph width, column gap
    rh, gy = 0.26, 0.055                      # lens row height, gap between
    g_photo, g_band = 0.11, 0.44
    ml, mr, mb = 1.20, 0.10, 0.12
    band_h = cw + g_photo + len(order) * rh + (len(order) - 1) * gy
    fs_h, fs_s = 15.5, 10.6
    sub_lines = 5
    hh = (10 + fs_h * 1.28 + 10 + sub_lines * fs_s * 1.5 + 10 + 15 + 14) / 72.0
    fig_w = ml + ncol * cw + (ncol - 1) * gx + mr
    fig_h = hh + nband * band_h + (nband - 1) * g_band + mb
    fig = plt.figure(figsize=(fig_w, fig_h))

    def box(x_in, y_in, w_in, h_in):
        """add_axes takes figure fractions and y from the BOTTOM."""
        return fig.add_axes([x_in / fig_w, 1 - (y_in + h_in) / fig_h,
                             w_in / fig_w, h_in / fig_h])

    tally = {l: {"top1": 0, "top3": 0, "n": 0} for l in order}
    n_in = 0
    for k, i in enumerate(ids):
        band, col = divmod(k, ncol)
        rec = by_i[int(i)]
        d = pix[pix.i == i]
        tib = int(d.truth_in_box.iloc[0])
        n_in += tib
        x = ml + col * (cw + gx)
        y = hh + band * (band_h + g_band)
        ax = box(x, y, cw, cw)
        try:
            im = Image.open(rec["path"]).convert("RGB").resize(
                (square, square), Image.LANCZOS)
            ax.imshow(im, extent=(0, square, square, 0))
        except Exception:
            ax.set_facecolor(PANEL)
        ax.set_xlim(0, square); ax.set_ylim(square, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("auto")
        for sp in ax.spines.values():
            sp.set_visible(False)
        _box_outline(ax, rec["inside"], coarse, square / coarse,
                     HOT if tib else INK)
        ax.set_title(rec["concept"], fontsize=10.5,
                     color=HOT if tib else INK, pad=3)

        for j, lens in enumerate(order):
            ax = box(x, y + cw + g_photo + j * (rh + gy), cw, rh)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            for t, dep in enumerate(depths):
                r = d[(d.lens == lens) & (d.depth == dep)]
                if not len(r):
                    continue
                one = int(r.hit_cause.iloc[0])
                three = int(r.top3_cause.iloc[0])
                tally[lens]["top1"] += one
                tally[lens]["top3"] += three
                tally[lens]["n"] += 1
                ax.add_patch(Rectangle((0.03 + t * 0.505, 0.10), 0.465, 0.80,
                                       facecolor=BLUE if one else
                                       (SOFT if three else PANEL),
                                       edgecolor="none" if three else "#dfe4ea",
                                       lw=0.8))
            if col == 0:
                ax.text(-0.06, 0.5, lens, ha="right", va="center",
                        fontsize=10.2, color=INK2, transform=ax.transAxes)

    unit = 1.0 / (72.0 * fig_h)
    fig.text(0.008, 1 - 10 * unit,
             "Every photograph of the pixel-occlusion run, and what each lens "
             "made of it", ha="left", va="top", fontsize=fs_h, color=INK)
    tal = "      ".join(
        f"{l}: {tally[l]['top1']} of {tally[l]['n']} exact, "
        f"{tally[l]['top3']} in the top three" for l in order)
    sub = ("Each region of each photograph is greyed out and the photograph "
           "re-encoded from scratch; the truth is whichever region costs the "
           "model's own next word the most.  Under every photograph there is "
           "one row per lens, and within a row one cell per depth read — "
           + " and ".join(f"{d * 100:.0f}%" for d in depths) +
           " of the way through the language model, left to right.  The dashed "
           "outline on a photograph is the human's bounding box, drawn in "
           f"orange on the {n_in} photographs of {n} where the most damaging "
           "region did fall inside it — on the other "
           f"{n - n_in} the model's answer turned on something outside the "
           "object.\n" + tal)
    import textwrap
    sub = "\n".join(textwrap.fill(par, 178) for par in sub.split("\n"))
    y_sub = 1 - (10 + fs_h * 1.28 + 10) * unit
    fig.text(0.008, y_sub, sub, ha="left", va="top", fontsize=fs_s, color=INK2,
             linespacing=1.5)
    key = [Patch(facecolor=BLUE, edgecolor="none",
                 label="its strongest region was the most damaging one"),
           Patch(facecolor=SOFT, edgecolor="none",
                 label="one of the three most damaging"),
           Patch(facecolor=PANEL, edgecolor="#dfe4ea", label="neither")]
    fig.legend(handles=key, loc="upper left",
               bbox_to_anchor=(0.008, y_sub - (sub.count("\n") + 1)
                               * fs_s * 1.5 * unit - 4 * unit),
               ncols=3, frameon=False, fontsize=fs_s, handlelength=1.3,
               handleheight=1.05, borderaxespad=0.0, columnspacing=2.2)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p
