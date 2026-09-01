# SPDX-License-Identifier: Apache-2.0
"""Figures for measurements that until now reached the reader only as a table.

Six pictures, each drawn off a result file the paper already quotes in prose:

``fivemodels``   the replication across three architectures, depth by depth
``pixelmodels``  the box dissociation under pixel occlusion, model by model
``promptdepth``  what a framing of the request does to the reading, layer by layer
``editcoord``    an edit landing in the coordinates before it lands in the answer
``baselines``    the saliency constructions a reviewer would name, with intervals
``pathcost``     what the path integral costs and what the extra points buy

The conventions are :mod:`jvlm.figcausal`'s, on purpose.  One colour per lens
paper-wide -- logit lens orange, J-lens violet, VJ-Lens blue, the one-point form
aqua -- square tiles where there are tiles, end-of-line labels staggered when two
series finish together, and axis-end names below the axis.

No figure here carries a paragraph.  A heading, the panels, and labels on what
the panels contain; sample sizes, chance rates, what each dashed line means and
every caveat live in the caption the figure is published under in
``paper/atlas.html``.  Each function prints the numbers a caption needs, so the
caption can be written off the same run that drew the plot.
"""
from __future__ import annotations

import os

import numpy as np

from jvlm import style
from jvlm.figcausal import LENS_COLOUR
from jvlm.style import GRID, INK, INK2, MUTED, SURFACE

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D

NEUTRAL = "#d5dbe3"          # a ground truth is not a lens, so it wears no hue
BEHAVE = INK                 # what the model does, as opposed to what a lens says


# ---------------------------------------------------------------- furniture

def _boot(v, n: int = 4000, seed: int = 0):
    """Mean and a 95% bootstrap interval, resampling the units given."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(1)
    return (float(v.mean()), float(np.percentile(m, 2.5)),
            float(np.percentile(m, 97.5)))


def _head(ax, title, sub=None, tfs=20.6):
    """A panel heading, and the room the heading itself needs.

    The pad is computed from the number of lines actually drawn, which is what
    keeps a one-line heading from reserving a two-line band.
    """
    fs = 13.0
    lines = (str(sub).count("\n") + 1) if sub else 0
    ax.set_title(title, loc="center", color=INK, pad=9 + lines * fs * 1.35,
                 fontsize=tfs)
    if sub:
        ax.annotate(sub, xy=(0.5, 1), xytext=(0, 7), xycoords="axes fraction",
                    textcoords="offset points", ha="center", va="bottom",
                    color=INK2, fontsize=fs, linespacing=1.35)


HEAD_PT = 9 + 16.5 * 1.35


def _headline(fig, head, fs_head=21.0, x=0.5, pad_pts=0.0):
    """The figure's one-line heading; returns the top the axes may rise to."""
    unit = 1.0 / (72.0 * fig.get_figheight())
    if not style.FIGURE_TITLES:
        return 1 - 6 * unit            # the caption names the figure
    fig.text(x, 1 - 6 * unit, head, ha="center" if x == 0.5 else "left",
             va="top", color=INK, fontsize=fs_head)
    return 1 - (6 + fs_head * 1.30 + 11 + pad_pts) * unit


def _lens_legend(fig, lenses, y, x=0.5, fs=20.3):
    """One row of lens names under the headline, in the paper's fixed hues."""
    h = [Line2D([], [], color=LENS_COLOUR.get(l, MUTED), lw=2.2, marker="o",
                ms=6.5, label=l) for l in lenses]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(x, y),
               ncols=len(h), frameon=False, handlelength=1.5,
               handletextpad=0.5, columnspacing=2.0, fontsize=fs,
               labelcolor=INK2)


def _stagger(ends, y0, y1, gap_frac=0.155):
    """Push end labels apart along y, keeping their order, inside the axes."""
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


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"  wrote {p}", flush=True)
    return p


# ============================================================ A. five models

#: the order the columns stand in, and the family each belongs to
MODELS = [("qwen2b_big", "Qwen3-VL 2B", "Qwen3-VL"),
          ("Qwen3-VL-4B-Instruct", "Qwen3-VL 4B", "Qwen3-VL"),
          ("Qwen3-VL-8B", "Qwen3-VL 8B", "Qwen3-VL"),
          ("llava-1.5-7b", "LLaVA-1.5 7B", "LLaVA-1.5"),
          ("InternVL3-8B", "InternVL3 8B", "InternVL3")]

#: our reading and the two prior lenses.  The one-point form of ours is not a
#: baseline and is left out of a figure whose subject is the comparison.
FIVE_LENSES = ["logit lens", "J-lens", "VJ-Lens"]


def _family_rules(fig, axes_row, groups, fs=20.3, head_pt=None):
    """A hairline over each family's columns, with the family's name on it."""
    unit = 1.0 / (72.0 * fig.get_figheight())
    head_pt = HEAD_PT if head_pt is None else head_pt
    for name, a, b in groups:
        p0 = axes_row[a].get_position()
        p1 = axes_row[b].get_position()
        y = p0.y1 + (head_pt + 9) * unit
        fig.add_artist(Line2D([p0.x0, p1.x1], [y, y], color="#cfcec9", lw=0.9,
                              transform=fig.transFigure, zorder=0))
        fig.text((p0.x0 + p1.x1) / 2, y + 3 * unit, name, ha="center",
                 va="bottom", color=INK2, fontsize=fs,
                 transform=fig.transFigure)


def fivemodels(frames, out, name="f100_fivemodels.png"):
    """Depth against the causal map, one column per model, two rows of measure.

    ``frames`` maps a result tag to its raw table.  Every model is read at the
    same two relative depths and the two Qwen models small enough to afford a
    third are read there as well, so the columns line up on the x axis without
    any rescaling.
    """
    style.use()
    have = [(t, nice, fam) for t, nice, fam in MODELS if t in frames]
    n = len(have)
    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 10.9), squeeze=False,
                             sharey="row")
    facts = []
    for j, (tag, nice, fam) in enumerate(have):
        d = frames[tag]
        R = int(d.R.iloc[0])
        lenses = [l for l in FIVE_LENSES if l in set(d.lens)]
        for row, (col, chance) in enumerate((("spearman", None),
                                             ("hit_cause", 1.0 / R))):
            ax = axes[row][j]
            if chance is not None:
                ax.axhline(chance, color=MUTED, lw=1.2, ls=(0, (4, 3)),
                           zorder=1)
            else:
                ax.axhline(0.0, color=GRID, lw=1.2, zorder=1)
            for lens in lenses:
                g = d[d.lens == lens]
                xs, mu, lo, hi = [], [], [], []
                for dep, gg in g.groupby("depth"):
                    m, a, b = _boot(gg.groupby("i")[col].mean().values)
                    xs.append(dep * 100); mu.append(m); lo.append(a); hi.append(b)
                c = LENS_COLOUR.get(lens, MUTED)
                ax.plot(xs, mu, color=c, lw=2.2, marker="o", ms=7, mew=0,
                        zorder=3)
                for x, a, b in zip(xs, lo, hi):
                    ax.plot([x, x], [a, b], color=c, lw=1.5, alpha=0.42,
                            zorder=2, solid_capstyle="butt")
                    if row == 1:
                        facts.append((nice, lens, x, col, mu[xs.index(x)]))
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.margins(y=0.16)
            if row == 1:
                ax.set_ylim(bottom=-0.035)
                ax.set_xlabel("depth  (%)")
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0,
                                                                    decimals=0))
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel("agreement with the causal map  (ρ)"
                              if row == 0 else
                              "names the single most\ndamaging region")
            style.despine(ax)
        _head(axes[0][j], nice, f"{R} regions · n = {d.i.nunique()}")

    # "chance" is a label on a line, not a sentence: it says which line, and
    # what its height is per model is in the caption
    a = axes[1][0]
    # under the rule and hard left of the first depth: over it the word lay
    # along the logit-lens line, which runs just above chance at this depth
    a.annotate("chance", xy=(1, 1.0 / int(frames[have[0][0]].R.iloc[0])),
               xytext=(0, -6), textcoords="offset points", ha="left",
               va="top", color=MUTED, fontsize=18.8)

    lenses_seen = [l for l in FIVE_LENSES
                   if any(l in set(frames[t].lens) for t, _, _ in have)]
    unit = 1.0 / (72.0 * fig.get_figheight())
    head2 = 9 + 13.0 * 1.35 + 16.5 * 1.35       # heading plus its second line
    # the family names sit on a rule above the panel headings, and the lens
    # legend sits under the figure's own heading; at the paper's type size the
    # two rows met.  The extra points here are the gap between them.
    top = _headline(fig, "Nothing is fitted, so the comparison ports",
                    pad_pts=head2 + 14.0 * 1.3 + 9 + 40)
    _lens_legend(fig, lenses_seen, 1 - (6 + 21.0 * 1.30 + 4) * unit)
    fig.subplots_adjust(left=0.062, right=0.994, top=top, bottom=0.078,
                        wspace=0.13, hspace=0.16)
    groups, i = [], 0
    while i < len(have):
        k = i
        while k + 1 < len(have) and have[k + 1][2] == have[i][2]:
            k += 1
        groups.append((have[i][2], i, k))
        i = k + 1
    _family_rules(fig, axes[0], groups, head_pt=head2)

    for tag, nice, fam in have:
        d = frames[tag]
        R = int(d.R.iloc[0])
        best_prior, ours = {}, {}
        for lens in set(d.lens) & set(FIVE_LENSES):
            g = d[d.lens == lens].groupby("depth")[["spearman", "hit_cause"]].mean()
            (ours if lens == "VJ-Lens" else best_prior)[lens] = g
        bp = max(best_prior.items(), key=lambda kv: kv[1].spearman.max())
        o = ours["VJ-Lens"]
        print(f"  {nice}: n={d.i.nunique()} R={R} chance={1/R:.2f} "
              f"depths={[round(x*100) for x in sorted(d.depth.unique())]} "
              f"| ours rho {o.spearman.min():+.3f}..{o.spearman.max():+.3f}, "
              f"top1 {o.hit_cause.min()*100:.0f}..{o.hit_cause.max()*100:.0f}% "
              f"| best prior {bp[0]} rho {bp[1].spearman.max():+.3f}, "
              f"top1 {bp[1].hit_cause.max()*100:.0f}%", flush=True)
    return _save(fig, out, name)


# ======================================================= B. pixel occlusion

PIX_MODELS = [("qwen2b", "Qwen3-VL 2B", "Qwen3-VL"),
              ("Qwen3-VL-4B", "Qwen3-VL 4B", "Qwen3-VL"),
              ("llava-1.5-7b", "LLaVA-1.5 7B", "LLaVA-1.5")]


def pixelmodels(pix, act, out, name="f101_pixelmodels.png"):
    """Where the decisive region lands, and what the lenses see, model by model.

    ``pix`` maps a tag to the pixel-occlusion table; ``act`` maps the same
    models' tags to the activation-covering table, which is the ruler the rest
    of the paper grades against and is drawn here only as a reference mark.
    """
    style.use()
    have = [(t, nice, fam) for t, nice, fam in PIX_MODELS if t in pix]
    fig, axes = plt.subplots(1, 3, figsize=(20.6, 7.0))
    xs = np.arange(len(have))
    # the model names are set over two lines: side by side on one line they are
    # wider than the three slots the axis gives them
    names = [nice.replace(" ", "\n") for _, nice, _ in have]

    # ---- panel 1: does the most damaging region land inside the box?
    ax = axes[0]
    hi_seen = 0.0
    for k, (tag, nice, fam) in enumerate(have):
        d = pix[tag]
        per = d.groupby("i").truth_in_box.mean().values
        m, lo, hi = _boot(per)
        hi_seen = max(hi_seen, hi)
        area = float(d.groupby("i").box_area.mean().mean())
        ax.bar([k], [m], width=0.46, color=NEUTRAL, edgecolor=INK2,
               linewidth=0.9, zorder=2)
        ax.plot([k, k], [lo, hi], color=INK2, lw=1.6, zorder=4)
        ax.plot([k - 0.28, k + 0.28], [area, area], color=INK, lw=1.7,
                ls=(0, (3, 2)), zorder=5)
        ax.annotate(f"{m * 100:.0f}%", xy=(k, 0), xytext=(0, 7),
                    textcoords="offset points", ha="center", va="bottom",
                    color=INK, fontsize=20.3)
        # at the dashed line's right-hand end, never above its middle: the
        # open circle marking the other ruler lands in the middle on two of
        # the three models
        ax.annotate(f"{area * 100:.0f}%", xy=(k + 0.28, area), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center",
                    color=INK, fontsize=18.8)
        a = act.get(tag if tag in act else {"qwen2b": "qwen2b_big"}.get(tag, ""))
        if a is not None:
            av = float(a.groupby("i").truth_in_box.mean().mean())
            ax.plot([k], [av], marker="o", ms=9, mfc=SURFACE, mec=INK2,
                    mew=1.6, zorder=6)
    ax.set_ylim(0, hi_seen * 1.30)
    ax.plot([], [], marker="o", ms=9, mfc=SURFACE, mec=INK2, mew=1.6, ls="none",
            label="covered in the activations instead")
    ax.plot([], [], color=INK, lw=1.7, ls=(0, (3, 2)),
            label="the box's own share of the frame")
    style.legend(ax, loc="upper left")
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=19.6)
    ax.set_ylabel("share of photographs")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    _head(ax, "the most damaging region\nis inside the object's box")
    style.despine(ax)

    # ---- panels 2 and 3: what each lens gets under this ruler
    for ax, (col, lab, chance) in zip(
            axes[1:], (("spearman", "Spearman ρ over every region", None),
                       ("hit_cause", "share of photographs", "1/R"))):
        for k, (tag, nice, fam) in enumerate(have):
            d = pix[tag]
            R = int(d.R.iloc[0])
            if chance:
                ax.plot([k - 0.30, k + 0.30], [1.0 / R] * 2, color=MUTED,
                        lw=1.3, ls=(0, (4, 3)), zorder=1)
            lenses = [l for l in FIVE_LENSES if l in set(d.lens)]
            for q, lens in enumerate(lenses):
                off = (q - (len(lenses) - 1) / 2) * 0.20
                per = d[d.lens == lens].groupby("i")[col].mean().values
                m, lo, hi = _boot(per)
                c = LENS_COLOUR.get(lens, MUTED)
                ax.plot([k + off, k + off], [lo, hi], color=c, lw=1.6,
                        alpha=0.45, zorder=2, solid_capstyle="butt")
                ax.plot([k + off], [m], marker="o", ms=9, color=c, mew=0,
                        zorder=3)
        if not chance:
            ax.axhline(0.0, color=GRID, lw=1.2, zorder=1)
        else:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        ax.set_xlim(-0.6, len(have) - 0.4)
        ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=19.6)
        ax.set_ylabel(lab)
        ax.margins(y=0.20)
        style.despine(ax)
    _head(axes[1], "agreement with the\npixel-occlusion map")
    _head(axes[2], "names the single most\ndamaging region")
    # between the first two groups, not at the spine: at the spine it lands on
    # the y-axis numbers
    axes[2].annotate("chance", xy=(0.34, 1.0 / int(pix[have[0][0]].R.iloc[0])),
                     xytext=(0, -5), textcoords="offset points", ha="left",
                     va="top", color=MUTED, fontsize=18.8)

    lenses_seen = [l for l in FIVE_LENSES
                   if any(l in set(pix[t].lens) for t, _, _ in have)]
    unit = 1.0 / (72.0 * fig.get_figheight())
    top = _headline(fig, "Grey the pixels out, and the box question turns over between "
                    "families",
                    pad_pts=HEAD_PT + 14.0 * 1.3 + 6 + 20.6 * 1.32)
    _lens_legend(fig, lenses_seen, 1 - (6 + 21.0 * 1.30 + 4) * unit)
    fig.subplots_adjust(left=0.062, right=0.988, top=top, bottom=0.165,
                        wspace=0.34)

    for tag, nice, fam in have:
        d = pix[tag]
        per = d.groupby("i").truth_in_box.mean().values
        m, lo, hi = _boot(per)
        area = float(d.groupby("i").box_area.mean().mean())
        R = int(d.R.iloc[0])
        say = ""
        akey = tag if tag in act else {"qwen2b": "qwen2b_big"}.get(tag, "")
        if akey in act:
            say = (f" activation ruler "
                   f"{act[akey].groupby('i').truth_in_box.mean().mean()*100:.0f}%")
        print(f"  {nice}: n={d.i.nunique()} R={R} depths="
              f"{[round(x*100) for x in sorted(d.depth.unique())]} "
              f"in-box {m*100:.0f}% [{lo*100:.0f},{hi*100:.0f}] vs box area "
              f"{area*100:.0f}%;{say}", flush=True)
        for lens in sorted(set(d.lens)):
            g = d[d.lens == lens]
            r = _boot(g.groupby("i").spearman.mean().values)
            h = _boot(g.groupby("i").hit_cause.mean().values)
            print(f"      {lens:<12s} rho {r[0]:+.3f} [{r[1]:+.3f},{r[2]:+.3f}]"
                  f"  top1 {h[0]*100:.0f}%", flush=True)
    return _save(fig, out, name)


# ================================================= C. the prompt, layer by layer

COND_ORDER = ["plain", "control", "substitute", "ask"]
COND_LABEL = {"plain": "no framing",
              "control": "the text\nnames a bowl",
              "substitute": "the text\nnames a knife",
              "ask": "…and asks how\nto cut a lock"}
#: the one-line form, for a label at the end of a line
COND_SHORT = {"plain": "no framing", "control": "names a bowl",
              "substitute": "names a knife", "ask": "asks for harm"}
#: which column in the parquet each lens's reading lives in
COND_LENS = {"VJ-Lens": "atlas", "logit lens": "say", "J-lens": "jlens"}


def promptdepth(df, out, name="f102_promptdepth.png"):
    """The knife's reading at the image tokens under four framings, per layer.

    The first panel is the shift from the unframed baseline at every layer the
    run stored; the second is the same shift photograph by photograph at the
    deepest layer, because a mean that moves is not yet evidence that anything
    moved; the third puts the shift beside what the model actually did.
    """
    style.use()
    layers = sorted({int(c.split("_")[1]) for c in df.columns
                     if c.startswith("atlas_")})
    lenses = [l for l, p in COND_LENS.items() if f"{p}_{layers[0]}" in df]
    conds = [c for c in COND_ORDER if c in set(df.cond)]
    base = df[df.cond == "plain"].set_index("i")

    fig, axes = plt.subplots(1, 3, figsize=(20.6, 6.6))

    # ---- panel 1: mean shift from the unframed request, layer by layer
    #
    # Only our reading gets a line per framing.  The two fixed-transport lenses
    # return the identical number under all four framings, so three lines each
    # would be three copies of the same flat line at zero and nine end labels
    # for five distinct series.  One line each, and the caption says they are
    # zero to the last stored digit.
    ax = axes[0]
    ax.axhline(0.0, color=GRID, lw=1.2, zorder=1)
    ends = []
    LS = {"substitute": "-", "ask": (0, (5, 2)), "control": (0, (1, 2))}
    for lens in lenses:
        pre = COND_LENS[lens]
        c = LENS_COLOUR.get(lens, MUTED)
        use = [k for k in conds if k != "plain"]
        if lens != "VJ-Lens":
            use = ["substitute"]
        for cond in use:
            g = df[df.cond == cond].set_index("i")
            ys = [float((g[f"{pre}_{L}"] - base[f"{pre}_{L}"]).dropna().mean())
                  for L in layers]
            # the two fixed lenses land on the same flat line at zero, so the
            # second of them is drawn dashed or the first is simply invisible
            fixed_ls = "-" if lens == "logit lens" else (0, (5, 3))
            ax.plot(layers, ys, color=c,
                    lw=2.4 if lens == "VJ-Lens" else (3.0 if lens == "logit lens" else 1.8),
                    ls=LS[cond] if lens == "VJ-Lens" else fixed_ls,
                    marker="o", ms=6.5 if lens != "J-lens" else 4.5, mew=0,
                    zorder=3 if lens != "J-lens" else 4)
            if lens == "VJ-Lens":
                ends.append(dict(x=layers[-1], y=ys[-1], c=c,
                                 t=COND_SHORT[cond]))
    y0, y1 = ax.get_ylim()
    for e in _stagger(ends, y0, y1, 0.15):
        style.label_end(ax, e["x"], e["y"], e["t"], e["c"], y_true=e["y_true"],
                        fontsize=18.8)
    # room inside the axes for the four end labels: at the smaller margin they
    # ran out of the panel and into the next panel's y-axis label
    ax.margins(x=0.55)
    ax.set_xticks(layers)
    ax.set_xlabel("layer of the language model")
    ax.set_ylabel("change in the knife's reading\nat the image tokens")
    _head(ax, "what the sentence does to the reading")
    style.despine(ax)

    # ---- panel 2: photograph by photograph, at the deepest layer
    ax = axes[1]
    L = layers[-1]
    ax.axhline(0.0, color=GRID, lw=1.2, zorder=1)
    rng = np.random.default_rng(3)
    facts2 = []
    for q, lens in enumerate(lenses):
        pre = COND_LENS[lens]
        g = df[df.cond == "substitute"].set_index("i")
        d = (g[f"{pre}_{L}"] - base[f"{pre}_{L}"]).dropna().values
        c = LENS_COLOUR.get(lens, MUTED)
        ax.scatter(q + rng.uniform(-0.17, 0.17, len(d)), d, s=26, color=c,
                   alpha=0.45, linewidths=0, zorder=3)
        m, lo, hi = _boot(d)
        ax.plot([q - 0.30, q + 0.30], [m, m], color=c, lw=2.4, zorder=5)
        ax.plot([q, q], [lo, hi], color=c, lw=1.6, alpha=0.6, zorder=4)
        moved = int((np.abs(d) > 1e-6).sum())
        # at the foot of the panel, under every photograph: under the mean
        # line the count printed in the middle of the swarm it counts
        ax.annotate(f"{moved} of {len(d)}\nmoved", xy=(q, 0.015),
                    xycoords=("data", "axes fraction"), xytext=(0, 0),
                    textcoords="offset points", ha="center", va="bottom",
                    color=INK2, fontsize=18.8)
        facts2.append((lens, m, lo, hi, moved, len(d)))
    ax.set_xlim(-0.6, len(lenses) - 0.4)
    ax.set_xticks(range(len(lenses)))
    ax.set_xticklabels(lenses, fontsize=19.6)
    ax.set_ylabel("change in the knife's reading,\none photograph each")
    ax.margins(y=0.22)
    _head(ax, f"one photograph at a time, layer {L}")
    style.despine(ax)

    # ---- panel 3: the reading beside what the model did
    #
    # No line joins the four framings: they are four conditions, not a dose,
    # and a line through them would assert an ordering the experiment does not
    # have.  The two fixed lenses stack on x = 0, which is the point.
    ax = axes[2]
    MARK = {"VJ-Lens": "o", "logit lens": "o", "J-lens": "s"}
    facts3 = []
    xs_all = []
    for lens in lenses:
        pre = COND_LENS[lens]
        c = LENS_COLOUR.get(lens, MUTED)
        for cond in conds:
            g = df[df.cond == cond].set_index("i")
            dm = float((g[f"{pre}_{L}"] - base[f"{pre}_{L}"]).dropna().mean())
            r = float(df[df.cond == cond].refused.mean())
            facts3.append((lens, cond, dm, r))
            xs_all.append(dm)
            # the three lenses coincide exactly on the unframed point, which
            # is the baseline every shift is measured from; sized and stacked
            # so a coincidence shows as a ring around a square around a dot
            ms, z = {"VJ-Lens": (13.0, 2), "J-lens": (9.0, 3),
                     "logit lens": (5.5, 4)}.get(lens, (9.0, 3))
            ax.plot([dm], [r], marker=MARK.get(lens, "o"), ms=ms, color=c,
                    mew=0, zorder=z)
    mid = (min(xs_all) + max(xs_all)) / 2
    # two of the four framings refuse on almost every photograph, so their
    # points sit a hundredth apart in y and their two-line names printed over
    # each other.  Push the names apart and lead each back to its own point.
    lab = [(dm, r, cond) for lens, cond, dm, r in facts3 if lens == "VJ-Lens"]
    lab.sort(key=lambda t: t[1])
    ylo, yhi = ax.get_ylim()
    gap = (yhi - ylo) * 0.19
    ys = []
    for dm, r, cond in lab:
        ys.append(r if not ys else max(r, ys[-1] + gap))
    over = ys[-1] - (yhi - gap * 0.30)
    if over > 0:
        ys = [y - over for y in ys]
    for (dm, r, cond), yy in zip(lab, ys):
        right = dm < mid
        if abs(yy - r) > 1e-9:
            ax.plot([dm, dm], [r, yy], color=MUTED, lw=0.8, alpha=0.45,
                    ls=(0, (1, 2)), zorder=1)
        ax.annotate(COND_LABEL[cond], xy=(dm, yy),
                    xytext=(9 if right else -9, 0),
                    textcoords="offset points", ha="left" if right else "right",
                    va="center", color=INK2, fontsize=18.8, linespacing=1.3)
    ax.axvline(0.0, color=GRID, lw=1.2, zorder=1)
    ax.set_ylim(-0.06, 1.14)
    ax.margins(x=0.45)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("change in the knife's reading")
    ax.set_ylabel("the model refuses")
    _head(ax, "the coordinate against the behaviour")
    style.despine(ax)

    unit = 1.0 / (72.0 * fig.get_figheight())
    top = _headline(fig, "The same photograph, four ways of asking",
                    pad_pts=HEAD_PT + 14.0 * 1.3 + 6)
    _lens_legend(fig, lenses, 1 - (6 + 21.0 * 1.30 + 4) * unit)
    fig.subplots_adjust(left=0.052, right=0.990, top=top, bottom=0.145,
                        wspace=0.62)

    print(f"  promptdepth: {df.i.nunique()} photographs, {len(conds)} framings, "
          f"layers {layers}", flush=True)
    for lens, m, lo, hi, moved, n_ in facts2:
        print(f"      {lens:<12s} substitute-plain at L{L}: {m:+.4f} "
              f"[{lo:+.4f},{hi:+.4f}], {moved}/{n_} photographs moved at all",
          flush=True)
    for lens, cond, dm, r in facts3:
        print(f"      {lens:<12s} {cond:<11s} shift {dm:+.4f}  refuses "
              f"{r*100:.0f}%", flush=True)
    return _save(fig, out, name)


# ============================================ D. the edit, coordinate and answer

OP_ORDER = ["erase", "amplify", "inject", "swap", "swapdev", "transplant"]
#: the reader-facing name, the same one the dose curves and the efficiency
#: plane print.  "swapdev" is the column name in the run, not a word.
OP_DISPLAY = {"swapdev": "swap, against the baseline"}
OP_X = {"erase": "multiples of the coordinate removed", "amplify": "gain",
        "inject": "strength", "swap": "gain", "swapdev": "gain",
        "transplant": "gain on the donor"}
#: which concept's coordinate the operation is aimed at
OP_READS = {"erase": ("read_src_u", "the source's coordinate"),
            "amplify": ("read_dst_u", "the target's coordinate"),
            "inject": ("read_dst_u", "the target's coordinate"),
            "swap": ("read_dst_u", "the target's coordinate"),
            "swapdev": ("read_dst_u", "the target's coordinate"),
            "transplant": ("read_dst_u", "the target's coordinate")}

C_COORD = style.AQUA          # the one-point reading the edit is written along
C_ANS = BEHAVE


def _dose_label(v) -> str:
    return f"{v:g}"


def editcoord(df, out, name="f103_editcoord.png"):
    """Coordinate and answer on one dose axis, one panel per operation.

    Both series are drawn as a share of their own full move, because they are
    measured in different units -- one is a rate, the other a projection in
    logit units -- and the claim is about which of the two moves first, not
    about how far either goes.  How far the coordinate goes is printed in the
    panel's own heading, in standard deviations of the unedited reading.
    """
    style.use()
    ops = [o for o in OP_ORDER if o in set(df.op)]
    ed = df[df.arm == "edit"]
    clean = df[df.arm == "clean"]
    rows = 2 if len(ops) > 3 else 1
    cols = int(np.ceil(len(ops) / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(5.3 * cols, 5.2 * rows),
                             squeeze=False, sharey=True)
    facts = []
    for k, op in enumerate(ops):
        ax = axes[k // cols][k % cols]
        d = ed[ed.op == op]
        doses = sorted(d.dose.unique())
        x = list(range(len(doses)))
        rcol, rlab = OP_READS.get(op, ("read_dst_u", "the target's coordinate"))
        b = float(clean[rcol].mean())
        sd = float(clean[rcol].std()) or 1.0
        g = d.groupby("dose")[rcol].mean()
        move = np.array([abs(float(g.get(v, np.nan)) - b) for v in doses])
        mx = np.nanmax(move) or 1.0
        follow = d.groupby("dose").follow.mean()
        fol = np.array([float(follow.get(v, np.nan)) for v in doses])
        fsem = np.array([float(d[d.dose == v].follow.sem()) for v in doses])

        ax.plot(x, move / mx, color=C_COORD, lw=2.6, marker="o", ms=8, mew=0,
                zorder=3, clip_on=False)
        ax.plot(x, fol, color=C_ANS, lw=2.2, marker="s", ms=7, mew=0, zorder=4,
                clip_on=False)
        ax.errorbar(x, fol, yerr=fsem, fmt="none", ecolor=C_ANS,
                    elinewidth=1.4, capsize=0, alpha=0.8, zorder=4)
        ax.set_ylim(-0.04, 1.06)
        ax.set_xlim(-0.30, len(doses) - 0.70)
        ax.set_xticks(x)
        ax.set_xticklabels([_dose_label(v) for v in doses], fontsize=19.6)
        ax.set_xlabel(OP_X.get(op, "dose"))
        if k % cols == 0:
            ax.set_ylabel("share of the coordinate's full move,\n"
                          "and of trials that flip to the target")
        z = mx / sd
        _head(ax, OP_DISPLAY.get(op, op),
              f"the coordinate travels {z:.1f}σ")
        style.despine(ax)
        facts.append((op, doses, list(np.round(move / mx, 3)), list(np.round(fol, 3)),
                      z, mx, b, sd, int(len(d) / max(1, len(doses)))))
    for k in range(len(ops), rows * cols):
        axes[k // cols][k % cols].axis("off")

    unit = 1.0 / (72.0 * fig.get_figheight())
    top = _headline(fig, "How far the coordinate travels before the answer "
                    "follows", pad_pts=HEAD_PT + 13.0 * 1.35 + 14.0 * 1.3 + 6)
    h = [Line2D([], [], color=C_COORD, lw=2.6, marker="o", ms=8,
                label="the concept's coordinate, as a share of its full move"),
         Line2D([], [], color=C_ANS, lw=2.2, marker="s", ms=7,
                label="the model's answer: trials that flip to the target")]
    fig.legend(handles=h, loc="upper center",
               bbox_to_anchor=(0.5, 1 - (6 + 21.0 * 1.30 + 4) * unit),
               ncols=2, frameon=False, handlelength=1.5, handletextpad=0.5,
               columnspacing=2.4, fontsize=20.3, labelcolor=INK2)
    fig.subplots_adjust(left=0.078, right=0.992, top=top, bottom=0.085,
                        wspace=0.11, hspace=0.55)
    for op, doses, mv, fol, z, mx, b, sd, n_ in facts:
        print(f"  {op:<11s} n≈{n_}/dose doses={doses} coordinate "
              f"{[f'{v:.2f}' for v in mv]} answer {[f'{v:.2f}' for v in fol]} "
              f"| full move {mx:.2f} = {z:.1f}sd, clean mean {b:.2f} sd {sd:.2f}",
              flush=True)
    return _save(fig, out, name)


# =============================================== E1. the saliency constructions

BASE_ORDER = ["VJ-Lens", "VJ-Lens (point)", "gradient norm",
              "gradient x activation", "J-lens", "logit lens", "deviation norm"]
BASE_COLOUR = dict(LENS_COLOUR)
BASE_COLOUR.update({"gradient norm": MUTED, "gradient x activation": MUTED,
                    "deviation norm": MUTED})


def baselines(df, out, name="f104_baselines.png"):
    """Seven maps over the same regions of the same photographs, with intervals.

    Three of the seven are the constructions a reader would reach for before
    reaching for a lens; two of the remaining four are ablations of our own
    reading, so the same picture doubles as an ingredient list.
    """
    style.use()
    order = [l for l in BASE_ORDER if l in set(df.lens)]
    R = int(df.R.iloc[0]) if "R" in df else 49
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 6.0), sharey=True)
    y = np.arange(len(order))[::-1]
    panels = (("spearman", "Spearman ρ over all "
               f"{R} regions", None, False),
              ("hit_cause", "share of photographs", 1.0 / R, True),
              ("top3_cause", "share of photographs", 3.0 / R, True))
    facts = []
    for ax, (col, lab, chance, pct) in zip(axes, panels):
        if chance is not None:
            ax.axvline(chance, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        else:
            ax.axvline(0.0, color=GRID, lw=1.2, zorder=1)
        for k, lens in enumerate(order):
            per = df[df.lens == lens].groupby("i")[col].mean().values
            m, lo, hi = _boot(per)
            c = BASE_COLOUR.get(lens, MUTED)
            ax.plot([lo, hi], [y[k], y[k]], color=c, lw=1.8, alpha=0.45,
                    zorder=2, solid_capstyle="butt")
            ax.plot([m], [y[k]], marker="o", ms=10, color=c, mew=0, zorder=3)
            t = f"{m * 100:.0f}%" if pct else f"{m:+.2f}".replace("-", "−")
            # beyond the interval's right-hand end, never above the dot: over
            # the dot it lands on the dashed chance line for every row that
            # sits near chance, which is four of the seven
            ax.annotate(t, xy=(hi, y[k]), xytext=(8, 0),
                        textcoords="offset points", ha="left", va="center",
                        color=INK2, fontsize=18.8)
            facts.append((lens, col, m, lo, hi))
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=19.6)
        ax.set_ylim(-0.75, len(order) - 0.25)
        ax.set_xlabel(lab)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.13)
        if pct:
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    _head(axes[0], "agreement with the whole map")
    _head(axes[1], "names the single most damaging region")
    _head(axes[2], "names one of the three most damaging")
    axes[1].annotate("chance", xy=(1.0 / R, -0.7), xytext=(4, 0),
                     textcoords="offset points", ha="left", va="bottom",
                     color=MUTED, fontsize=18.8)
    axes[2].annotate("chance", xy=(3.0 / R, -0.7), xytext=(4, 0),
                     textcoords="offset points", ha="left", va="bottom",
                     color=MUTED, fontsize=18.8)
    top = _headline(fig, "Against the maps practitioners actually reach for",
                    pad_pts=HEAD_PT)
    fig.subplots_adjust(left=0.135, right=0.995, top=top, bottom=0.135,
                        wspace=0.09)
    print(f"  baselines: n={df.i.nunique()} photographs, "
          f"{df.depth.nunique()} depths, R={R}, chance {1/R*100:.1f}% "
          f"and {3/R*100:.1f}%", flush=True)
    for lens, col, m, lo, hi in facts:
        print(f"      {lens:<22s} {col:<11s} {m:+.4f} [{lo:+.4f},{hi:+.4f}]",
              flush=True)
    return _save(fig, out, name)


# ================================================ E2. what the path integral costs

def pathcost(df, out, name="f105_pathcost.png"):
    """What a reading costs in seconds, and what a second of it buys.

    The companion figure asks what an extra point on the path buys.  This one
    asks what it costs, which is the half a reader deciding how to spend a GPU
    minute actually needs and which no figure in this paper has drawn.  Depths
    wear the same greys and the same marker shapes as that companion, light to
    dark with depth, so the two can be read together.
    """
    style.use()
    steps = sorted(df.steps.unique())
    depths = sorted(df.depth.unique())
    # light to dark with depth, and a shape each, exactly as the companion
    # figure on this dataset draws them -- these are depths, not lenses, and
    # the lens hues are spoken for paper-wide
    GREY = ["#a9a7a0", "#6f6d68", INK]
    MARK = ["o", "s", "^"]
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.2))

    # ---- panel 1: seconds per photograph
    ax = axes[0]
    ends = []
    for k, dep in enumerate(depths):
        g = df[df.depth == dep].groupby("steps").secs.mean()
        ax.plot(steps, [g.get(s, np.nan) for s in steps], color=GREY[k], lw=2.2,
                marker=MARK[k], ms=8, mew=0)
        ends.append(dict(x=steps[-1], y=float(g.get(steps[-1])), c=GREY[k],
                         t=f"{dep * 100:.0f}% deep", m=MARK[k]))
    y0, y1 = ax.get_ylim()
    for e in _stagger(ends, y0, y1, 0.10):
        style.label_end(ax, e["x"], e["y"], e["t"], e["c"], y_true=e["y_true"],
                        marker=e["m"], fontsize=19.6)
    ax.set_xscale("log", base=2); ax.set_xticks(steps)
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.set_xticklabels(["1\nplain gradient" if s == 1
                        else "4\nthe paper's path" if s == 4 else str(s)
                        for s in steps])
    ax.margins(x=0.40)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("points evaluated on the path")
    ax.set_ylabel("seconds for one photograph's\nwhole 196-patch reading")
    _head(ax, "what the path costs")
    style.despine(ax)

    # ---- panel 2: the frontier -- accuracy against the seconds it took
    ax = axes[1]
    ends2, xs2 = [], []
    # "the series arrives from directly below" measured against the whole
    # panel, so it does not depend on which depth is drawn first
    near = 0.03 * float(np.ptp(
        df.groupby(["depth", "steps"]).secs.mean().values))
    for k, dep in enumerate(depths):
        g = df[df.depth == dep].groupby("steps")[["secs", "corr"]].mean()
        xs2 += list(map(float, g.secs.values))
        ax.plot(g.secs.values, g["corr"].values, color=GREY[k], lw=1.8,
                alpha=0.8, zorder=2)
        ax.plot(g.secs.values, g["corr"].values, ls="none", marker=MARK[k],
                ms=9, color=GREY[k], mew=0, zorder=3)
        for q, st in enumerate(steps[:-1]):
            # the last point already carries the series name; a step label
            # there prints on top of it
            if st in g.index:
                yv = float(g["corr"][st])
                up = q % 2 == 1
                # above a point that already sits under the "exact" rule the
                # number lands on the rule itself.  It drops under instead,
                # and further than the ordinary drop, which is what keeps it
                # off both the neighbouring label and its own series' line
                dx, dy = 0, 10 if up else -16
                if up and yv > 0.93:
                    # over a point this close to the "exact" rule the number
                    # prints on the rule, so it goes under instead; where the
                    # series climbs to that point from directly below, under
                    # is its own line, and the number steps back off it
                    xv = float(g.secs[st])
                    xp = float(g.secs.iloc[max(q - 1, 0)])
                    dx, dy, up = (-8 if abs(xv - xp) < near else 0), -16, False
                ax.annotate(f"{st}", xy=(g.secs[st], yv),
                            xytext=(dx, dy), textcoords="offset points",
                            ha="center" if not dx else
                            ("right" if dx < 0 else "left"),
                            va="bottom" if up else "top",
                            color=INK2, fontsize=18.8)
        ends2.append(dict(x=float(g.secs.iloc[-1]),
                          y=float(g["corr"].iloc[-1]), c=GREY[k],
                          t=f"{dep * 100:.0f}% deep", m=MARK[k]))
    # the three frontiers end at three different seconds, so a name printed
    # at each one's own end lands in the middle of the panel, on top of a
    # curve that runs on past it.  All three names go in a column to the right
    # of every curve, pushed apart in y, each with a hairline back to its end.
    y0, y1 = ax.get_ylim()
    x_lab = max(xs2)
    for e in _stagger(ends2, y0, y1, 0.13):
        ax.plot([e["x"], x_lab], [e["y_true"], e["y"]], color=e["c"], lw=0.8,
                alpha=0.35, ls=(0, (1, 2)), zorder=1, clip_on=False)
        style.label_end(ax, x_lab, e["y"], e["t"], e["c"], marker=e["m"],
                        fontsize=19.6)
    ax.axhline(1.0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("exact", xy=(ax.get_xlim()[1], 1.0), xytext=(-2, 4),
                textcoords="offset points", ha="right", va="bottom",
                color=MUTED, fontsize=18.8)
    ax.set_xlabel("seconds for one photograph")
    ax.set_ylabel("predicted change against true change,\n"
                  "over the whole vocabulary  (r)")
    # room on the right for the label column, and only a hair on the left
    span = max(xs2) - min(xs2)
    ax.set_xlim(min(xs2) - 0.06 * span, max(xs2) + 0.42 * span)
    ax.margins(y=0.20)
    _head(ax, "what a second of it buys")
    style.despine(ax)

    top = _headline(fig, "What the path integral costs", pad_pts=HEAD_PT)
    fig.subplots_adjust(left=0.082, right=0.982, top=top, bottom=0.165,
                        wspace=0.38)
    print(f"  pathcost: n={df.i.nunique()} photographs, steps {steps}, "
          f"depths {[round(d*100) for d in depths]}%", flush=True)
    for dep in depths:
        g = df[df.depth == dep].groupby("steps")[["secs", "corr", "rel_error"]].mean()
        for s in steps:
            if s in g.index:
                print(f"      {dep*100:.0f}% deep, {s} points: "
                      f"{g.secs[s]:.1f}s r={g['corr'][s]:.2f} "
                      f"rel={g.rel_error[s]:.2f}", flush=True)
    return _save(fig, out, name)


# =============================================== F. the same claim, layer by layer

LAYER_LENSES = ["logit lens", "J-lens", "VJ-Lens (point)", "VJ-Lens"]


def layerwise(frames, out, name="f106_layerwise.png"):
    """The causal-map comparison at every second layer, on two architectures.

    The powered runs grade two or three depths over tens of photographs.  These
    grade ten and twelve depths over eight and six, which is the other way
    round: too few photographs to settle a level, enough layers to show the
    shape of the curve.  The last column draws the photographs themselves, so a
    reader can see whether the mean is carried by one of them.

    ``frames`` maps a model's name to its raw table, in the order the rows are
    to be drawn.
    """
    style.use()
    rows = list(frames.items())
    seen = []
    for _, d in rows:
        seen += [l for l in LAYER_LENSES if l in set(d.lens) and l not in seen]
    lenses_seen = [l for l in LAYER_LENSES if l in seen]

    fig, axz = plt.subplots(len(rows), 4, figsize=(19.6, 5.5 * len(rows)),
                            squeeze=False)
    for r, (nice, df) in enumerate(rows):
        lenses = [l for l in LAYER_LENSES if l in set(df.lens)]
        R = int(df.R.iloc[0])
        box = float(df.groupby("i").box_share.mean().mean())
        panels = (("spearman", "agreement with the causal map  (ρ)", None),
                  ("hit_cause", "names the single most\ndamaging region", 1.0 / R),
                  ("hit_box", "names a region inside\nthe human's box", box))
        for k, (col, ylab, ref) in enumerate(panels):
            ax = axz[r][k]
            if ref is not None:
                ax.axhline(ref, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0,
                                                                    decimals=0))
            else:
                ax.axhline(0.0, color=GRID, lw=1.2, zorder=1)
            for lens in lenses:
                g = df[df.lens == lens].groupby("depth")[col].mean()
                ax.plot(g.index * 100, g.values,
                        color=LENS_COLOUR.get(lens, MUTED), lw=2.2, marker="o",
                        ms=5.5, mew=0, zorder=3)
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.margins(y=0.16)
            if ref is not None:
                # a share cannot be far below zero; the margin can put the axis
                # there, and the empty band reads as data that is not present
                ax.set_ylim(bottom=max(ax.get_ylim()[0], -0.05))
            ax.set_xlabel("depth through the language model  (%)")
            # columns 1 and 2 already carry this as a heading on the top row
            # (below); repeating it as a y-label there would print it twice
            if not (r == 0 and k in (1, 2)):
                ax.set_ylabel(ylab)
            style.despine(ax)
        # the dashed lines are named once, on the top row, and the caption says
        # what height each one stands at on each model
        if r == 0:
            axz[r][1].annotate("chance", xy=(4, 1.0 / R), xytext=(0, 5),
                               textcoords="offset points", ha="left",
                               va="bottom", color=MUTED, fontsize=18.8)
            # at the dashed line's right-hand end: on the left it prints
            # through the logit lens's dip to twelve per cent
            axz[r][2].annotate("the box's share of the frame",
                               xy=(0.98, 0.02), xycoords="axes fraction",
                               ha="right", va="bottom", color=MUTED,
                               fontsize=18.8)

        # ---- the photographs behind the mean
        ax = axz[r][3]
        c = LENS_COLOUR["VJ-Lens"]
        ax.axhline(0.0, color=GRID, lw=1.2, zorder=1)
        d = df[df.lens == "VJ-Lens"]
        for i, g in d.groupby("i"):
            g = g.sort_values("depth")
            ax.plot(g.depth * 100, g.spearman, color=c, lw=1.2, alpha=0.38,
                    marker="o", ms=3.5, mew=0, zorder=2)
        mu = d.groupby("depth").spearman.mean()
        ax.plot(mu.index * 100, mu.values, color=c, lw=3.0, zorder=4)
        style.label_end(ax, float(mu.index[-1] * 100), float(mu.values[-1]),
                        f"the mean of {d.i.nunique()}", c, fontsize=18.8)
        ax.set_xlim(0, 100)
        ax.margins(x=0.40)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("depth through the language model  (%)")
        ax.set_ylabel("agreement, one photograph each  (ρ)")
        style.despine(ax)

        _head(axz[r][0], nice,
              f"{df.layer.nunique()} layers · {R} regions · "
              f"n = {df.i.nunique()}")
        # the other three columns are named once, on the top row; every panel
        # under them carries the same y label, so a second heading would only
        # print the same words twice
        if r == 0:
            # wrapped to match the y-labels below: at four panels across a
            # 19.6-inch figure, the one-line form is wider than the column
            # and prints into its neighbour
            _head(axz[r][1], "names the single most\ndamaging region")
            _head(axz[r][2], "names a region inside\nthe human's box")
            _head(axz[r][3], "one photograph at a time")

    unit = 1.0 / (72.0 * fig.get_figheight())
    head2 = 9 + 13.0 * 1.35 + 16.5 * 1.35
    top = _headline(fig, "The same comparison at every second layer, on two "
                    "architectures",
                    pad_pts=head2 + 14.0 * 1.3 + 6)
    _lens_legend(fig, lenses_seen, 1 - (6 + 21.0 * 1.30 + 4) * unit)
    fig.subplots_adjust(left=0.050, right=0.993, top=top, bottom=0.075,
                        wspace=0.40, hspace=0.42)

    for nice, df in rows:
        R = int(df.R.iloc[0])
        box = float(df.groupby("i").box_share.mean().mean())
        print(f"  layerwise {nice}: n={df.i.nunique()} photographs, "
              f"{df.layer.nunique()} layers {sorted(df.layer.unique())}, R={R}, "
              f"chance {1/R*100:.1f}%, box share {box*100:.0f}%", flush=True)
        for lens in [l for l in LAYER_LENSES if l in set(df.lens)]:
            g = df[df.lens == lens].groupby("depth")[["spearman", "hit_cause",
                                                      "hit_box"]].mean()
            pk = g.spearman.idxmax()
            print(f"      {lens:<17s} rho {g.spearman.min():+.3f}.."
                  f"{g.spearman.max():+.3f} (peak at {pk*100:.0f}%), "
                  f"mean {g.spearman.mean():+.3f}, top1 "
                  f"{g.hit_cause.min()*100:.0f}..{g.hit_cause.max()*100:.0f}%, "
                  f"in-box {g.hit_box.min()*100:.0f}.."
                  f"{g.hit_box.max()*100:.0f}%", flush=True)
    return _save(fig, out, name)
