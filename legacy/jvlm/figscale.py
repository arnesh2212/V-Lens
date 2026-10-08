# SPDX-License-Identifier: Apache-2.0
"""Concrete pictures for the paper's abstract axes.

Four of this paper's axes are counted in units a reader has never seen: a
region measured in *image patches*, a lens's variability measured in *distinct
words over 196 patches*, an edit's cost measured in *fractional damage to the
residual stream*, and a reading's fidelity measured in *points evaluated on a
path*.  Each of those numbers is defensible and none of them is picturable, so
each figure here puts one real instance of the quantity beside the axis that
counts it.

The house rules are figcausal's: square tiles, a short heading and the plot,
words clipped so they cannot run into their neighbours, names at the ends of
series rather than in a legend box, and the paper's fixed lens colours --
logit lens ORANGE, J-lens VIOLET, VJ-Lens BLUE, VJ-Lens (point) AQUA.  Where a
figure names the two prior lenses as the ends of our own parameter, those two
hues are what the ends wear, and depth keeps the neutral ramp so a curve is
never mistaken for a lens.
"""
from __future__ import annotations

import os

import numpy as np

from jvlm import style
from jvlm.style import (AQUA, BLUE, GRID, INK, INK2, MUTED, ORANGE, SURFACE,
                        VIOLET)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import offset_copy

#: depth is a magnitude, so it gets one hue light-to-dark rather than three
#: categorical slots.  In the region figures the categorical slots are spoken
#: for -- orange and violet are the two prior lenses sitting at the ends of the
#: axis -- and a depth line in either of those hues reads as a lens.
DEPTH_RAMP = ["#b6b3ab", "#7c7970", "#26251f"]
DEPTH_MARK = ["o", "s", "^"]


def _pct(x: float) -> str:
    """A share of the frame, with a digit only where one is needed."""
    return f"{x:.1f}%" if x < 9.95 else f"{x:.0f}%"


def _square(path: str, side: int = 448):
    """The photograph as the model is given it: forced to a square.

    Every experiment that pools per position loads its images this way
    (``corpus.load_image(square=448)``), because the processor's token grid
    follows the aspect ratio and position 37 would otherwise be a different
    part of the frame in every picture.  A strip that showed the original
    aspect ratio would be drawing regions on a frame the model never saw.
    """
    from PIL import Image

    return Image.open(path).convert("RGB").resize((side, side), Image.LANCZOS)


def _box_from_cells(cells, coarse: int, grid: int):
    """The photographed object's box, to the nearest region.

    The causal-map cards record which of the ``coarse x coarse`` regions the
    human box covers, not the box itself; the bounding rectangle of those
    regions is the box to the resolution the experiment actually used, which
    is the honest thing to draw on a patch grid.
    """
    if not cells:
        return None
    k = grid // coarse
    rs = [int(c) // coarse for c in cells]
    cs = [int(c) % coarse for c in cells]
    r0, r1 = min(rs) * k, (max(rs) + 1) * k
    c0, c1 = min(cs) * k, (max(cs) + 1) * k
    return c0, r0, c1 - c0, r1 - r0


# ------------------------------------------------------- A. what a region is

def regionstrip(stats: dict, path: str, out: str, name="f90_regionscale.png",
                layer: int | None = None, inside=None, seed: int = 11,
                sizes=None):
    """One photograph, at every region size the sweep measured.

    The region sweep's x axis runs from one image patch to all 196 of them and
    a reader has no idea what sixteen patches looks like.  Here the same
    photograph is drawn once per size with the region at its true scale and at
    a position drawn the way the sweep draws it -- uniformly at random over the
    grid -- and under each the agreement that size actually reaches.  The point
    is watchable rather than assertable: the region grows from a speck to the
    whole frame and the number rises, plateaus, and falls back.
    """
    style.use()
    by = stats.get("by_size")
    rows = list(by.values()) if isinstance(by, dict) else list(by or [])
    rows = [r for r in rows if r.get("spearman") is not None]
    if not rows:
        raise ValueError("no region-sweep rows")
    G = int(stats.get("grid", 14))
    P = int(stats.get("n_patches", G * G))
    layers = sorted({int(r["layer"]) for r in rows})
    L = layers[-1] if layer is None else int(layer)
    d = {int(r["size"]): float(r["spearman"])
         for r in rows if int(r["layer"]) == L}
    sizes = [int(s) for s in (sizes or sorted(d)) if int(s) in d]
    n = len(sizes)

    img = _square(path, G * 32)
    px = img.width / G
    rng = np.random.default_rng(seed)
    box = _box_from_cells(inside, int(stats.get("coarse", 7)), G)

    bar_c = DEPTH_RAMP[layers.index(L) % len(DEPTH_RAMP)]
    lo = min(0.0, min(d[s] for s in sizes))
    hi = max(d[s] for s in sizes)
    span = (hi - lo) or 1.0

    # each column is as wide as the label under it: at the paper's type size
    # "100% of the frame" was wider than the column and ran off the figure
    fig, axes = plt.subplots(2, n, figsize=(3.05 * n, 5.4), squeeze=False,
                             gridspec_kw=dict(height_ratios=[1.0, 0.40]))
    for j, s in enumerate(sizes):
        side = max(1, min(int(round(s ** 0.5)), G))
        r0 = 0 if side >= G else int(rng.integers(0, G - side + 1))
        c0 = 0 if side >= G else int(rng.integers(0, G - side + 1))

        ax = axes[0][j]
        ax.imshow(img, extent=(0, G, G, 0), zorder=1)
        # everything outside the region is veiled, then the region itself is
        # painted back at full strength: a translucent wash *over* the region
        # would show where it is and hide what is in it, which is the opposite
        # of what this strip is for
        ax.add_patch(Rectangle((0, 0), G, G, facecolor=SURFACE, alpha=0.70,
                               edgecolor="none", zorder=2))
        crop = img.crop((int(c0 * px), int(r0 * px),
                         int((c0 + side) * px), int((r0 + side) * px)))
        ax.imshow(crop, extent=(c0, c0 + side, r0 + side, r0), zorder=3)
        ax.add_patch(Rectangle((c0, r0), side, side, fill=False,
                               edgecolor=INK, lw=1.6, zorder=5))
        if box is not None:
            bx, by_, bw, bh = box
            # a dark stroke under a white dash, so the object's box survives
            # both the bright and the dark parts of a photograph
            ax.add_patch(Rectangle((bx, by_), bw, bh, fill=False,
                                   edgecolor=INK, lw=2.2, alpha=0.55, zorder=6))
            ax.add_patch(Rectangle((bx, by_), bw, bh, fill=False,
                                   edgecolor="#ffffff", lw=1.1, zorder=7,
                                   linestyle=(0, (3, 2))))
        ax.set_xlim(0, G)
        ax.set_ylim(G, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

        ax = axes[1][j]
        v = d[s]
        ax.bar([0], [v], width=0.52, color=bar_c, edgecolor="none",
               zorder=3)
        ax.axhline(0.0, color=GRID, lw=1.1, zorder=1)
        ax.text(0, v + span * 0.07, f"{v:+.2f}", ha="center", va="bottom",
                color=INK2, fontsize=19.6)
        ax.set_xlim(-0.62, 0.62)
        ax.set_ylim(lo - span * 0.06, hi + span * 0.42)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        word = "patch" if s == 1 else "patches"
        ax.set_xlabel(f"{s} {word}\n{_pct(100.0 * s / P)} of the frame",
                      fontsize=20.3, labelpad=7)
        if j == 0:
            ax.set_ylabel("reading against\ntrue effect  ($\\rho$)",
                          fontsize=19.6)

    unit = 1.0 / (72.0 * fig.get_figheight())
    fig.text(0.5, 1 - 6 * unit,
             "A region is a square block of the model’s own image patches",
             ha="center", va="top", color=INK, fontsize=25)
    top = 1 - (6 + 20 * 1.22 + 8) * unit
    fig.subplots_adjust(left=0.052, right=0.995, top=top, bottom=0.175,
                        wspace=0.10, hspace=0.10)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"  region strip: {os.path.basename(path)}, {n} sizes, layer {L}")
    for s in sizes:
        print(f"    {s:4d} patches  {_pct(100.0 * s / P):>6s}  "
              f"rho {d[s]:+.3f}")
    return p


# ------------------------------------------- C. the family, as a single dial

def family(stats: dict, out: str, name="f91_family.png",
           box_patches: float = 30.0, n_layers: int | None = None):
    """The one-parameter family, drawn as the dial it is.

    The region is the method's only parameter and both prior lenses are values
    of it: a region of one patch is the logit lens's unit, a region of every
    patch is what averaging the transport over positions amounts to.  The dial
    across the top says so with the two hues those lenses wear everywhere else
    in the paper; the curves below say what each setting is worth.  Depth wears
    a neutral ramp here precisely so that no curve can be read as a lens.
    """
    style.use()
    by = stats.get("by_size")
    rows = list(by.values()) if isinstance(by, dict) else list(by or [])
    rows = [r for r in rows if r.get("spearman") is not None]
    if not rows:
        raise ValueError("no region-sweep rows")
    P = int(stats.get("n_patches", 196))
    NL = int(n_layers or stats.get("n_layers") or 0)
    sizes = sorted({int(r["size"]) for r in rows})
    layers = sorted({int(r["layer"]) for r in rows})
    nn = max(int(r.get("n", 0)) for r in rows)

    fig, ax = plt.subplots(figsize=(11.0, 5.65))
    ends, allv = [], []
    for j, l in enumerate(layers):
        c = DEPTH_RAMP[j % len(DEPTH_RAMP)]
        mk = DEPTH_MARK[j % len(DEPTH_MARK)]
        d = {int(r["size"]): float(r["spearman"])
             for r in rows if int(r["layer"]) == l}
        xs = [s for s in sizes if s in d]
        ys = [d[s] for s in xs]
        allv += ys
        ax.plot(xs, ys, color=c, marker=mk, ms=7, lw=2.4, zorder=4)
        lab = f"layer {l}" if not NL else f"layer {l} of {NL}"
        ends.append((xs[-1], ys[-1], lab, c, mk))
    for x, y, lab, c, mk in ends:
        style.label_end(ax, x, y, lab, c, marker=mk)

    ax.axvline(box_patches, color=MUTED, lw=1.4, ls=(0, (5, 4)), zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([1, 4, 16, 64, P])
    ax.set_xticklabels(["1", "4", "16", "64", str(P)])
    ax.minorticks_off()
    ax.set_xlim(0.62, P * 1.75)
    lo, hi = min(allv), max(allv)
    sp = (hi - lo) or 1.0
    # headroom for the dial: the ribbon lives in the top eighth of the axes and
    # nothing may be drawn under it
    ax.set_ylim(lo - sp * 0.10, hi + sp * 0.46)
    ax.set_xlabel("region size, in image patches")
    ax.set_ylabel("reading against true effect  ($\\rho$)")
    style.despine(ax)

    # ---- the dial itself, in axes-fraction height over data x
    tr = ax.get_xaxis_transform()
    y0, y1 = 0.885, 0.965
    ax.add_patch(Rectangle((1, y0), P - 1, y1 - y0, transform=tr,
                           facecolor=BLUE, alpha=0.11, edgecolor="none",
                           zorder=2, clip_on=False))
    ax.plot([1, P], [(y0 + y1) / 2] * 2, transform=tr, color=BLUE,
            lw=1.1, alpha=0.45, zorder=3, clip_on=False)
    for x, colour in ((1, ORANGE), (P, VIOLET)):
        ax.plot([x], [(y0 + y1) / 2], marker="s", ms=9.5, color=colour,
                transform=tr, zorder=5, clip_on=False, mew=0)
    ym = (y0 + y1) / 2
    # the names sit on the track, so they carry the track's own colour behind
    # them: BLUE at 0.11 over SURFACE, worked out once rather than left to
    # print the hairline through the middle of every glyph
    plate = dict(facecolor="#e5edf7", edgecolor="none", pad=1.6)
    ax.annotate("logit lens", xy=(1, ym), xycoords=tr, xytext=(10, 0),
                textcoords="offset points", ha="left", va="center",
                color=INK2, fontsize=19.6, zorder=6, bbox=plate)
    ax.annotate("J-lens", xy=(P, ym), xycoords=tr, xytext=(-10, 0),
                textcoords="offset points", ha="right", va="center",
                color=INK2, fontsize=19.6, zorder=6, bbox=plate)
    ax.plot([14], [ym], marker="s", ms=9.5, color=BLUE, transform=tr,
            zorder=5, clip_on=False, mew=0)
    ax.annotate("VJ-Lens", xy=(14, ym), xycoords=tr, xytext=(10, 0),
                textcoords="offset points", ha="left", va="center",
                color=INK2, fontsize=19.6, zorder=6, bbox=plate)

    # ---- the three places on the axis, named under it
    for x, txt, ha in ((1, "a single patch", "left"),
                       (box_patches, "an average object’s box", "center"),
                       (P, "the whole frame", "right")):
        ax.annotate(txt, xy=(x, 0), xytext=(0, -58),
                    xycoords=("data", "axes fraction"),
                    textcoords="offset points", ha=ha, va="top",
                    color=MUTED, annotation_clip=False,
                    fontsize=plt.rcParams["axes.labelsize"] * 0.84)

    unit = 1.0 / (72.0 * fig.get_figheight())
    fig.text(0.5, 1 - 6 * unit,
             "The two prior lenses are the two ends of one dial",
             ha="center", va="top", color=INK, fontsize=25)
    top = 1 - (6 + 20 * 1.22 + 10) * unit
    fig.subplots_adjust(left=0.082, right=0.985, top=top, bottom=0.215)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"  family dial: {len(layers)} depths x {len(sizes)} sizes, "
          f"n = {nn}, reference line at {box_patches:g} of {P}")
    return p


# --------------------------------------------- B. what that damage looks like

OP_COLOUR = {"erase": BLUE, "amplify": ORANGE, "inject": AQUA,
             "swap": VIOLET, "swapdev": style.GREEN,
             "transplant": style.MAGENTA}
# the same marker on the same operation as the efficiency figure this one sits
# under -- these two were crossed, so the green plus there was a green triangle
# here and the pink triangle there a pink plus here
OP_MARK = {"erase": "s", "amplify": "^", "inject": "o", "swap": "D",
           "swapdev": "P", "transplant": "v"}
OP_DISPLAY = {"swapdev": "swap, against\nthe baseline"}
OP_ORDER = ["inject", "transplant", "swapdev", "amplify", "swap", "erase"]


def damageruler(df, out, name="f93_damage.png", trial: int | None = None,
                ops=None, short: int = 14):
    """One photograph, six edits, and the damage axis f68 counts them on.

    ``‖dh‖/‖h‖`` on a log axis is the paper's measure of what
    an edit costs, and nothing in the efficiency figure says what a cost of
    0.005 or of 0.06 does to a model.  This does: the same photograph and the
    same question under every operation, each dose printed where its damage
    lands on that axis and carrying the answer the model actually gave there.
    """
    style.use()
    ed = df[df.arm == "edit"]
    if trial is None:
        # the trial that carries every operation at every dose, so the rows are
        # comparable; ties broken toward the lowest index for reproducibility
        cnt = ed.groupby("i").op.nunique()
        trial = int(cnt.sort_values(ascending=False).index[0])
    d = ed[ed.i == trial]
    if not len(d):
        raise ValueError(f"no edit rows for trial {trial}")
    ops = [o for o in (ops or OP_ORDER) if o in set(d.op)]
    r0 = d.iloc[0]
    clean = df[(df.i == trial) & (df.arm == "clean")]
    before = str(r0.src_ans)
    if len(clean):
        before = str(clean.iloc[0].text).strip() or before
    want = str(r0.dst_ans)

    # the question, trimmed of the wording that only tells the model how to
    # format its reply, and wrapped: a panel heading that runs past its own
    # panel prints over the plot beside it
    import re
    import textwrap

    qs = str(r0.q).split("?")[0].strip()
    qs = re.sub(r"\s*in this photo(graph)?", "", qs, flags=re.I)
    qs = (qs[0].lower() + qs[1:] if qs else qs) + "?"
    img = _square(str(r0.path), 448)
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 6.0),
                             gridspec_kw=dict(width_ratios=[0.95, 3.3]))

    ax = axes[0]
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"the model answers “{before}”",
                 loc="center", color=INK, pad=10, fontsize=21.8)
    # wrapped to the column it sits under, not to the page
    ax.set_xlabel(textwrap.fill("it was asked " + qs, 24), fontsize=19.6,
                  color=INK2, labelpad=8, linespacing=1.35)

    ax = axes[1]
    ys = list(range(len(ops)))[::-1]
    for y, op in zip(ys, ops):
        s = d[d.op == op].sort_values("disturb")
        c = OP_COLOUR.get(op, MUTED)
        mk = OP_MARK.get(op, "o")
        for _, r in s.iterrows():
            hit = bool(r.follow)
            ax.plot([float(r.disturb)], [y], marker=mk, ms=11.0,
                    color=c, mfc=(c if hit else SURFACE), mew=1.8,
                    zorder=4, clip_on=False)
            ax.annotate(str(r.text).strip()[:short], xy=(float(r.disturb), y),
                        xytext=(0, 12), textcoords="offset points",
                        ha="center", va="bottom", color=INK2, fontsize=18.8)
        ax.plot(s.disturb.values, [y] * len(s), color=c, lw=1.4, alpha=0.45,
                zorder=2)
    ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels([OP_DISPLAY.get(o, o) for o in ops], fontsize=20.3)
    ax.set_ylim(-0.75, len(ops) - 0.25)
    lo = float(d.disturb[d.disturb > 0].min())
    hi = float(d.disturb.max())
    ax.set_xlim(lo / 1.9, hi * 1.9)
    ax.grid(True, axis="x", color=GRID, lw=0.9)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    ax.set_xlabel("damage done to the residual stream at the edited positions"
                  "   ‖Δh‖ / ‖h‖   (log)")
    style.despine(ax)
    ax.plot([], [], marker="o", ms=10.0, color=MUTED, mfc=MUTED, mew=1.8,
            lw=0, label=f"the model now answers “{want}”")
    ax.plot([], [], marker="o", ms=10.0, color=MUTED, mfc=SURFACE, mew=1.8,
            lw=0, label=f"it still answers “{before}”")
    style.legend(ax, loc="lower right")

    unit = 1.0 / (72.0 * fig.get_figheight())
    fig.text(0.5, 1 - 6 * unit,
             "The same photograph, six edits, one damage axis",
             ha="center", va="top", color=INK, fontsize=25)
    top = 1 - (6 + 20 * 1.22 + 10) * unit
    fig.subplots_adjust(left=0.030, right=0.988, top=top, bottom=0.185,
                        wspace=0.30)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"  damage ruler: trial {trial}, {os.path.basename(str(r0.path))}, "
          f"{len(ops)} ops\n    q: {r0.q}\n    clean answer {before!r}, "
          f"edit target {want!r}")
    for op in ops:
        s = d[d.op == op].sort_values("disturb")
        f = s[s.follow == 1]
        rng_ = f"{s.disturb.min():.2f}..{s.disturb.max():.2f}"
        when = f"flips at {f.disturb.min():.2f}" if len(f) else "never flips"
        print(f"    {op:11s} damage {rng_}   {when}")
    return p


# -------------------------------------------- B. what a path with points is

def pathcase(df, out, name="f94_path.png", n_layers: int | None = None):
    """What "points evaluated on the path" means, and what each one buys.

    Left: the path itself.  It runs from the baseline the region is compared
    against to the activation the photograph actually put there, and a reading
    with ``k`` points is the average of the gradient at ``k`` places on it --
    the midpoint rule, except at one point, where the comparison deliberately
    reports the plain gradient at the activation itself, which is the form the
    other two lenses use.  Middle and right: every trial, not just the mean,
    because a claim about a residual is a claim about individual cases.
    """
    style.use()
    steps = sorted({int(s) for s in df.steps.unique()})
    layers = sorted({int(l) for l in df.layer.unique()})
    NL = int(n_layers or 0)
    x = list(range(len(steps)))
    xi = {s: i for i, s in enumerate(steps)}

    fig, axes = plt.subplots(1, 3, figsize=(19.6, 5.9),
                            gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.0]))

    # ---- the path
    ax = axes[0]
    for k, s in enumerate(steps):
        y = len(steps) - 1 - k
        ax.plot([0, 1], [y, y], color=GRID, lw=2.6, zorder=1,
                solid_capstyle="round")
        al = [1.0] if s == 1 else [(j + 0.5) / s for j in range(s)]
        ax.plot(al, [y] * len(al), marker="o", ms=9.0, lw=0, color=INK2,
                zorder=3, clip_on=False)
    ax.set_ylim(-0.7, len(steps) - 0.3)
    ax.set_xlim(-0.04, 1.04)
    ax.set_yticks(list(range(len(steps)))[::-1])
    ax.set_yticklabels([f"{s} point" if s == 1 else f"{s} points"
                        for s in steps], fontsize=20.3)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "½", "1"])
    ax.grid(False)
    style.despine(ax)
    ax.set_xlabel("how far along the path  ($\\alpha$)")
    ax.set_title("where the readings are taken", loc="center", color=INK,
                 pad=11, fontsize=20.6)
    for xx, txt, ha in ((0.0, "the baseline this region\nis compared against",
                         "left"),
                        (1.0, "what this photograph\nputs there", "right")):
        ax.annotate(txt, xy=(xx, 0), xytext=(0, -50),
                    xycoords=("data", "axes fraction"),
                    textcoords="offset points", ha=ha, va="top", color=MUTED,
                    annotation_clip=False, linespacing=1.3,
                    fontsize=plt.rcParams["axes.labelsize"] * 0.84)

    # ---- two measured panels, every trial drawn
    panels = [("corr", "predicted change against true change,\n"
                       "across the whole vocabulary  (r)", 1.0),
              ("hit", "the reading's strongest patch lands\n"
                      "inside the object's box", None)]
    for ax, (col, lab, ceil) in zip(axes[1:], panels):
        if col not in df.columns:
            ax.axis("off")
            continue
        # every trial, where a trial is a number.  A per-trial line on a
        # panel whose values are 0 or 1 is a lattice of crossing diagonals that
        # says nothing at all, so that panel gets the standard error instead.
        binary = bool(set(np.unique(df[col].values)) <= {0.0, 1.0})
        if not binary:
            for (i, l), g in df.groupby(["i", "layer"]):
                g = g.sort_values("steps")
                if len(g) < 2:
                    continue
                ax.plot([xi[int(s)] for s in g.steps], g[col].values,
                        color=MUTED, lw=0.8, alpha=0.18, zorder=2)
        ends = []
        for k, l in enumerate(layers):
            g = df[df.layer == l].groupby("steps")[col].mean()
            c = DEPTH_RAMP[k % len(DEPTH_RAMP)]
            mk = DEPTH_MARK[k % len(DEPTH_MARK)]
            ax.plot([xi[int(s)] for s in g.index], g.values, color=c,
                    marker=mk, ms=7, lw=2.4, zorder=4)
            if binary:
                se = df[df.layer == l].groupby("steps")[col].sem()
                ax.errorbar([xi[int(s)] for s in g.index], g.values,
                            yerr=se.values, fmt="none", ecolor=c,
                            elinewidth=1.6, capsize=0, zorder=3)
            dep = df[df.layer == l].depth.iloc[0]
            ends.append([float(g.values[-1]), f"{100 * float(dep):.0f}% deep",
                         c, mk, float(g.values[-1])])
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{s}\nplain gradient" if s == steps[0]
             else f"{s}\nthe paper's path" if s == 4 else str(s)
             for s in steps])
        ax.set_xlabel("points evaluated on the path")
        ax.set_ylabel(lab)
        ax.set_xlim(-0.22, len(steps) - 1 + 0.10)
        ax.margins(x=0.34, y=0.14)
        style.despine(ax)
        if col == "hit" and "chance" in df.columns:
            ch = float(df.chance.mean())
            ax.axhline(ch, color=INK, lw=1.0, ls=(0, (2, 3)), zorder=1)
            # named at the right-hand end: at the left the first point's
            # whisker reaches down through the words
            ax.annotate("a patch picked at random", xy=(1, ch),
                        xytext=(-2, 5), xycoords=("axes fraction", "data"),
                        textcoords="offset points", ha="right", va="bottom",
                        color=INK2, fontsize=18.8)
        if ends:
            ylo, yhi = ax.get_ylim()
            gap = (yhi - ylo) * 0.075
            ends.sort(key=lambda e: e[0])
            for i in range(1, len(ends)):
                if ends[i][0] - ends[i - 1][0] < gap:
                    ends[i][0] = ends[i - 1][0] + gap
            over = ends[-1][0] - (min(yhi, ceil if ceil is not None else yhi)
                                  - gap * 0.30)
            if over > 0:
                for e in ends:
                    e[0] -= over
            for yy, txt, c, mk, y_true in ends:
                style.label_end(ax, x[-1], yy, txt, c, y_true=y_true,
                                marker=mk)

    unit = 1.0 / (72.0 * fig.get_figheight())
    fig.text(0.5, 1 - 6 * unit,
             "One point on the path is the gradient; four is the intervention",
             ha="center", va="top", color=INK, fontsize=25)
    # the panels carry their own headings, so the band under the figure's own
    # heading has to hold both -- this is the clearance an earlier version did
    # not leave, and the two lines printed on top of each other
    top = 1 - (6 + 20 * 1.22 + 34) * unit
    fig.subplots_adjust(left=0.075, right=0.945, top=top, bottom=0.215,
                        wspace=0.62)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    n_tr = df.groupby("steps").size().max()
    print(f"  path case: {len(steps)} path lengths x {len(layers)} depths, "
          f"{int(n_tr)} rows per length, {df.i.nunique()} photographs")
    for col in ("corr", "hit"):
        if col in df.columns:
            g = df.groupby("steps")[col].mean()
            print(f"    {col}: " + ", ".join(f"{s}→{g[s]:.2f}"
                                             for s in steps))
    if "chance" in df.columns:
        print(f"    chance for the box: {df.chance.mean():.2f}")
    return p


# ------------------------------- B. the schematic's plane, actually measured

def span(df, out, name="f92_span.png"):
    """How far apart the two lens vectors an edit exchanges really are.

    The edit-space diagram draws two lens vectors spanning a plane, and a plane
    drawn by hand can be drawn at any angle at all.  If the two directions were
    nearly the same, erasing one and injecting the other would be the same
    operation and the taxonomy would be empty; if they were exactly orthogonal,
    the diagram would be a claim about a basis rather than about a span.  The
    angle is measurable and every trial in the edit run already records it, so
    here it is: one ray per trial, laid on the same figure as the reference
    direction, which is the diagram with its own measurement in it.
    """
    style.use()
    u = df.drop_duplicates("i")
    cs = np.asarray(u["cos_ab"], dtype=float)
    cs = cs[np.isfinite(cs)]
    if not len(cs):
        raise ValueError("no cos_ab in the edit table")
    ang = np.degrees(np.arccos(np.clip(cs, -1.0, 1.0)))
    med = float(np.median(ang))
    lo, hi = float(ang.min()), float(ang.max())
    q1, q3 = (float(np.percentile(ang, 25)), float(np.percentile(ang, 75)))

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    R = 1.0
    # the wedge every measured pair falls in, then the middle half inside it
    for a0, a1, al in ((lo, hi, 0.07), (q1, q3, 0.10)):
        t = np.radians(np.linspace(a0, a1, 80))
        ax.fill(np.concatenate([[0], np.cos(t) * R]),
                np.concatenate([[0], np.sin(t) * R]),
                color=BLUE, alpha=al, lw=0, zorder=1)
    for a in ang:
        t = np.radians(a)
        ax.plot([0, np.cos(t) * R], [0, np.sin(t) * R], color=BLUE, lw=0.9,
                alpha=0.22, zorder=2)
    t = np.radians(med)
    ax.plot([0, np.cos(t) * R], [0, np.sin(t) * R], color=BLUE, lw=2.6,
            zorder=4)

    # the reference direction, with its name over the shaft rather than at the
    # tip, where the protractor's own end already is
    ax.annotate("", xy=(R * 1.005, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=2.2,
                                shrinkA=0, shrinkB=0))
    # under the shaft, not over it: over it is where every ray in the fan runs
    ax.annotate("the concept the edit takes from", xy=(R * 0.52, 0),
                xytext=(0, -10), textcoords="offset points", ha="center",
                va="top", color=INK2, fontsize=19.6)

    # the protractor.  Only three angles are named and all three are measured
    # ones -- an interior scale would print its numerals on top of the rays
    t = np.radians(np.linspace(0, 90, 200))
    ax.plot(np.cos(t) * R * 1.09, np.sin(t) * R * 1.09, color=GRID, lw=1.2,
            zorder=1)
    for a, txt, c, lw in ((lo, f"{lo:.0f}°", MUTED, 1.0),
                          (med, f"{med:.0f}°  the median pair", INK2, 1.6),
                          (hi, f"{hi:.0f}°", MUTED, 1.0)):
        t = np.radians(a)
        ax.plot([np.cos(t) * R * 1.06, np.cos(t) * R * 1.12],
                [np.sin(t) * R * 1.06, np.sin(t) * R * 1.12],
                color=c, lw=lw, zorder=3)
        ax.annotate(txt, xy=(np.cos(t) * R * 1.15, np.sin(t) * R * 1.15),
                    ha="left", va="center", color=c, fontsize=19.6)
    ax.annotate("the same direction", xy=(R * 1.09, 0), xytext=(6, 0),
                textcoords="offset points", ha="left", va="center",
                color=MUTED, fontsize=19.6)
    # to the left of the quarter circle's top, not centred on it: centred, it
    # reached across to the widest measured angle's own number
    ax.annotate("at right angles", xy=(0, R * 1.09), xytext=(-8, 4),
                textcoords="offset points", ha="right", va="bottom",
                color=MUTED, fontsize=19.6)

    ax.set_xlim(-0.52, R * 1.66)
    ax.set_ylim(-0.24, R * 1.24)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    unit = 1.0 / (72.0 * fig.get_figheight())
    fig.text(0.5, 1 - 6 * unit,
             "The plane in that diagram is a real plane",
             ha="center", va="top", color=INK, fontsize=25)
    top = 1 - (6 + 20 * 1.22 + 8) * unit
    fig.subplots_adjust(left=0.01, right=0.995, top=top, bottom=0.02)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"  span: {len(ang)} pairs, angle {lo:.1f}..{hi:.1f}°, "
          f"quartiles {q1:.1f}/{med:.1f}/{q3:.1f}, "
          f"cos {cs.min():.2f}..{cs.max():.2f} (median {np.median(cs):.2f})")
    return p
