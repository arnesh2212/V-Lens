# SPDX-License-Identifier: Apache-2.0
"""The safety figures: the jailbreak benchmark, and what is inside it.

Three pictures, and each one exists because a number in the safety section is
easy to misread on its own.

``jailbreak``      an AUROC of 0.65 means nothing without the floor the same
                   pipeline reaches on shuffled labels, and nothing again
                   without the four ways that AUROC could have been taken.  All
                   of it is drawn together, so the selection inflation is a gap
                   a reader can measure with their eye rather than a caveat.
``benchmark``      what the benchmark actually does to this model.  The clean
                   pictures do not jailbreak it; the typography and the prompt
                   wording do.  That is a fact about the benchmark, not about
                   any lens, and it belongs in front of the AUROCs.
``hallucination``  which absent objects a model invents.  Drawn plainly, with
                   the winner labelled as the winner.

Conventions are the paper's, and they are load-bearing: one colour per lens
everywhere (logit lens orange, J-lens violet, VJ-Lens blue), names in ink beside
a dot in the series hue rather than in the hue itself, and end-of-line labels
pushed apart when two series finish together.

None of the three carries explanatory text.  A figure here is a heading and a
plot; the sample sizes, what the shaded band is, what the pale foot of a bar
is, and every caveat about how an AUROC was taken live in the caption the
figure is published under, in ``paper/atlas.html``.  The axes take the room
back: the jailbreak panels used to start at 0.60 of the figure and now start
at 0.86.
"""
from __future__ import annotations

import os

import numpy as np

from jvlm import style
from jvlm.style import GRID, INK, INK2, MUTED, SURFACE

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

#: one colour per lens, matching jvlm.figcausal.LENS_COLOUR
LENS_COLOUR = {"logit lens": style.ORANGE, "J-lens": style.VIOLET,
               "VJ-Lens": style.BLUE, "VJ-Lens (point)": style.AQUA}
#: the result files key the lenses by slug
LENS_NAME = {"logit": "logit lens", "say": "logit lens", "jlens": "J-lens",
             "vj": "VJ-Lens", "ig": "VJ-Lens", "do": "VJ-Lens (point)"}

C_REFUSE = style.RED
C_DEFLECT = "#c9c7c0"
C_COMPLY = "#dfe7f1"
FLOOR = "#b9b7b0"


#: how tall a panel heading is, in points.  :func:`_fig_head` reserves it,
#: because a panel title is drawn ABOVE its own axes rectangle.
HEAD_PT = 8.0 + 17.5 * 1.32


def _head(ax, title, fs_title=21.9):
    """A panel's one-line heading.  There is no second line by design.

    Three of these carried three or four lines of explanation apiece, which is
    what the reader objected to.  A panel title is a label for the panel; the
    explanation is in the caption.
    """
    ax.set_title(title, loc="center", color=INK, pad=8.0, fontsize=fs_title)


def _fig_head(fig, title, fs_title=23.5, x=0.5, panel_head=0.0):
    """The figure's one-line heading, and the top of the axes under it.

    ``panel_head`` is the height in points of the per-panel heading that will
    be drawn ABOVE its axes.  Reserving it here is the whole job: an axes top
    that does not know about the panel titles leaves them to be drawn into
    whatever is above, which is the overprint this project has fixed a dozen
    times.

    The three-line paragraph this used to draw under the title is in the
    figure's caption now.
    """
    unit = 1.0 / (72.0 * fig.get_figheight())
    fig.text(x, 1 - 5 * unit, title, ha="center", va="top", color=INK,
             fontsize=fs_title)
    return 1 - (5 + fs_title * 1.30 + 10 + panel_head) * unit


def _stagger(ax, ends, frac=0.075):
    """Direct labels at the right end of each line, pushed apart where they meet.

    Copied in spirit from the collision-avoidance block that stopped the causal
    figures printing two names on top of one another: the gap is a fraction of
    the AXIS, never of the spread of the values, because three lines that finish
    within 0.01 of each other have no spread to take a fraction of.
    """
    if not ends:
        return
    y0, y1 = ax.get_ylim()
    gap = (y1 - y0) * frac
    ends = sorted(ends, key=lambda e: e[1])
    for e in ends:
        e.append(e[1])
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < gap:
            ends[i][1] = ends[i - 1][1] + gap
    over = ends[-1][1] - (y1 - gap * 0.35)
    if over > 0:
        for e in ends:
            e[1] -= over
    for x, y, name, c, y_true in ends:
        style.label_end(ax, x, y, name, c, y_true=y_true)


def _bar_values(ax, xs, vals, top=None, colour=None, fs=18.8, fmt="{:.2f}",
                dy=3.0):
    """The number above a bar, cleared of whatever else is drawn on the bar."""
    tops = top if isinstance(top, (list, tuple)) else [top] * len(xs)
    for x, v, t in zip(xs, vals, tops):
        if v is None or v != v:
            continue
        ax.annotate(fmt.format(v), xy=(x, t if t is not None else v),
                    xytext=(0, dy), textcoords="offset points", ha="center",
                    va="bottom", color=colour or INK2, fontsize=fs)


# ------------------------------------------------------------------ figure A

def jailbreak(d, out, name="f80_jailbreak.png", paired=None):
    """Three panels on one benchmark: by depth, by estimator, and with the text.

    Panel one is the reading at each layer with the label-shuffled floor drawn
    under it.  Panel two is the same data taken four ways, each beside its own
    floor, because the tempting statistic -- the best of the four layers -- has
    a higher floor as well as a higher value, and only the two together say
    whether anything was gained.  Panel three is the same items with the harmful
    phrase typed onto the picture, which is the measurement the first version of
    this experiment made by accident.

    ``paired`` is ``{slug: {"clean": auroc, "typo": auroc, "n_pos": int,
    "n_neg": int}}`` read on the ONE set of items that exists in both forms.
    Without it panel three has to set the clean block's 140 + 140 against the
    typed block's 50 + 50, and a difference between two different sets of items
    is not what the panel claims to show.
    """
    style.use()
    au, nul = d["auroc"], d.get("null_calibration") or {}
    typo = d.get("typo_auroc") or {}
    layers = [int(l) for l in d["layers"]]
    slugs = [k for k in ("logit", "jlens", "vj") if k in au]
    names = [LENS_NAME[k] for k in slugs]
    cols = [LENS_COLOUR[n] for n in names]
    n_h, n_b = d.get("n_harmful"), d.get("n_benign")
    tn = d.get("typo_n") or {}

    fig, axes = plt.subplots(
        1, 3, figsize=(21.0, 7.3),
        gridspec_kw={"width_ratios": [1.05, 1.32, 0.92]})

    # ---- (i) by layer, over the shuffled-label floor -----------------------
    ax = axes[0]
    ax.set_xlim(layers[0] - 1.0, layers[-1] + 4.0)
    ax.set_ylim(0.472, 0.678)
    # ONE band, in a neutral grey.  Three translucent bands in three lens hues
    # stood here first: they coincide to within 0.004, so they do not read as
    # three floors, they read as one lavender wash -- and the wash lay under
    # the orange line and recoloured it.  The coincidence is stated in the
    # figure's caption instead, where a sentence can say it.
    fl = [(nul.get(k) or {}).get("prereg") or {} for k in slugs]
    fl = [f for f in fl if "mean" in f and "q95" in f]
    if fl:
        f_hi = max(f["q95"] for f in fl)
        ax.axhspan(min(f["mean"] for f in fl), f_hi, color="#eae8e2", lw=0,
                   zorder=1)
        ax.plot([layers[0] - 1.0, layers[-1] + 4.0], [f_hi, f_hi],
                color=FLOOR, lw=1.4, zorder=2)
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=2)
    ends = []
    for k, n, c in zip(slugs, names, cols):
        bl = au[k].get("by_layer") or {}
        ys = [bl.get(str(l), bl.get(l)) for l in layers]
        ax.plot(layers, ys, color=c, marker="o", ms=6.5, lw=2.2, zorder=4)
        ends.append([float(layers[-1]), float(ys[-1]), n, c])
    ax.set_xticks(layers)
    ax.set_yticks([0.50, 0.55, 0.60, 0.65])
    ax.set_xlabel("layer of the language model  (of 28)")
    ax.set_ylabel("AUROC, harmful against benign")
    _stagger(ax, ends)
    # the only words left inside the axes.  Everything the band needs saying
    # about it is said in the subtitle and in the figure header, because the
    # strip beside the last layer already holds three end-of-line names and a
    # fourth label there lands on top of one of them.
    ax.annotate("chance", xy=(layers[-1] + 3.8, 0.5), xytext=(0, 3),
                textcoords="offset points", ha="right", va="bottom",
                color=MUTED, fontsize=18.8)
    _head(ax, "Read at each layer")
    style.despine(ax)

    # ---- (ii) the four estimators, each over its own floor ------------------
    ax = axes[1]
    EST = [("split_half", "layer chosen on\nthe other half"),
           ("band_avg", "averaged over\nthe whole band"),
           ("prereg", "one layer fixed\nin advance"),
           ("max_layer", "the best of the\nfour layers")]
    w, base = 0.24, 0.5
    ax.set_ylim(0.478, 0.762)
    ax.axhline(base, color=INK2, lw=1.0, zorder=3)
    for j, (key, lab) in enumerate(EST):
        for i, (k, c) in enumerate(zip(slugs, cols)):
            x = j + (i - 1) * w
            v = au[k].get(key)
            if v is None or v != v:
                continue
            ax.add_patch(Rectangle((x - w * 0.44, base), w * 0.88, v - base,
                                   facecolor=c, edgecolor="none", zorder=4))
            f = ((nul.get(k) or {}).get(key) or {}).get("q95")
            if f is not None:
                # the floor is painted back OVER the bar, so the saturated part
                # left standing is exactly what the lens gained on the floor
                ax.add_patch(Rectangle((x - w * 0.44, base), w * 0.88,
                                       min(f, v) - base, facecolor=SURFACE,
                                       alpha=0.72, edgecolor="none", zorder=5))
                ax.plot([x - w * 0.46, x + w * 0.46], [f, f], color=FLOOR,
                        lw=1.7, solid_capstyle="butt", zorder=6)
            # no number over each bar: twelve small grey numbers were noise,
            # and the exact AUROCs are in the caption and the result file
    ax.set_xlim(-0.55, len(EST) - 0.45)
    ax.set_xticks(range(len(EST)))
    ax.set_xticklabels([l for _, l in EST], fontsize=18.8, linespacing=1.3)
    # the four names are wider than the four slots; every other one is dropped
    # a line so each keeps its size and gets twice the room
    style.stagger_xticks(ax, 2.3, fs=18.8)
    ax.set_yticks([0.50, 0.55, 0.60, 0.65, 0.70])
    ax.set_ylabel("AUROC, harmful against benign")
    # clear of the dropped row of names above it
    ax.set_xlabel("how the AUROC was taken", labelpad=16)
    handles = [Patch(facecolor=c, edgecolor="none", label=n)
               for n, c in zip(names, cols)]
    handles.append(Patch(facecolor="#ebe9e4", edgecolor=FLOOR,
                         label="its own shuffled-label\nfloor, 95th percentile"))
    style.legend(ax, handles=handles, loc="upper left", ncols=2)
    _head(ax, "Four aggregation rules")
    style.despine(ax)

    # ---- (iii) with and without the typography ------------------------------
    ax = axes[2]
    w = 0.30
    pr = paired or {}
    ax.set_ylim(0.478, 0.940)
    ax.axhline(base, color=INK2, lw=1.0, zorder=3)
    for i, (k, n, c) in enumerate(zip(slugs, names, cols)):
        a_clean = (pr.get(k) or {}).get("clean", au[k].get("split_half"))
        a_typo = (pr.get(k) or {}).get("typo",
                                       (typo.get(k) or {}).get("split_half"))
        for x, v, solid in ((i - w * 0.55, a_clean, False),
                            (i + w * 0.55, a_typo, True)):
            if v is None or v != v:
                continue
            ax.add_patch(Rectangle(
                (x - w * 0.5, base), w, v - base,
                facecolor=c if solid else "none", edgecolor=c,
                lw=0 if solid else 1.6, zorder=4))
            _bar_values(ax, [x], [v], fs=18.8, dy=4.0)
        if a_clean and a_typo:
            ax.annotate(f"+{a_typo - a_clean:.2f}", xy=(i, max(a_clean, a_typo)),
                        xytext=(0, 26), textcoords="offset points",
                        ha="center", va="bottom", color=INK2, fontsize=19.6)
    ax.set_xlim(-0.62, len(slugs) - 0.38)
    ax.set_xticks(range(len(slugs)))
    ax.set_xticklabels(names, fontsize=19.6)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_ylabel("AUROC, harmful against benign")
    handles = [Patch(facecolor="none", edgecolor=INK2, lw=1.6,
                     label="hollow: nothing written on the picture"),
               Patch(facecolor=MUTED, edgecolor="none",
                     label="solid: the harmful phrase typed on it")]
    style.legend(ax, handles=handles, loc="upper left")
    _head(ax, "Effect of the typography")
    style.despine(ax)

    # what the header used to say -- the item counts, the identical neutral
    # question, that every lens is read at the image tokens only, that the
    # three shuffled-label floors land within 0.004 of one another so one grey
    # band serves for all three, and that panel three is the one paired set of
    # items -- is in this figure's caption in paper/atlas.html.
    one = next(iter(pr.values()), {}) if pr else {}
    print(f"  jailbreak: {n_h} harmful, {n_b} benign; paired typed set "
          f"{one.get('n_pos')} + {one.get('n_neg')}; typed-block n {tn}",
          flush=True)
    top = _fig_head(fig, "One benchmark, three lenses",
                    panel_head=HEAD_PT)
    fig.subplots_adjust(left=0.048, right=0.988, top=top, bottom=0.215,
                        wspace=0.26)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure B

def benchmark(d, out, name="f81_benchmark.png"):
    """What the benchmark does to this model, before any lens is involved.

    Compliance, refusal in so many words, and the deflections in between, for
    the clean painting, the same painting with the phrase typed on it, and the
    same items asked the benchmark's own question.  The clean picture does not
    jailbreak this model at all; the refusals arrive with the typography and
    with the wording.
    """
    style.use()
    comp = d["compliance"]["complied"]
    stro = d["compliance"].get("refused_str") or {}
    N = {"SD|1": d.get("n_harmful"), "SD|0": d.get("n_benign"),
         "SD_TYPO|1": (d.get("typo_n") or {}).get("harmful"),
         "SD_TYPO|0": (d.get("typo_n") or {}).get("benign"),
         "SD_native|1": d.get("n_harmful")}
    ROWS = [("SD|1", "a harmful scenario", 0),
            ("SD|0", "a benign scenario", 0),
            ("SD_TYPO|1", "a harmful scenario", 1),
            ("SD_TYPO|0", "a benign scenario", 1),
            ("SD_native|1", "a harmful scenario", 2)]
    BLOCK = ["the picture alone, with\nnothing written on it",
             "the same picture, harmful\nphrase typed onto it",
             "the same pictures, asked the\nbenchmark's own question"]
    rows = [r for r in ROWS if r[0] in comp]

    fig, ax = plt.subplots(figsize=(14.8, 6.8))
    ypos, y, prev = [], 0.0, None
    for key, lab, b in rows:
        if prev is not None and b != prev:
            # room for the two-line block title between blocks; at the
            # raised type size 0.86 let its second line touch the row label
            # under it
            y += 1.15
        ypos.append(y)
        y += 1.0
        prev = b
    for (key, lab, b), yy in zip(rows, ypos):
        c = comp[key] * 100
        s = stro.get(key, 0.0) * 100
        dfl = max(0.0, 100 - c - s)
        x = 0.0
        for w_, col in ((c, C_COMPLY), (dfl, C_DEFLECT), (s, C_REFUSE)):
            if w_ > 0:
                ax.add_patch(Rectangle((x, yy - 0.34), w_, 0.68,
                                       facecolor=col, edgecolor="none"))
            x += w_
        ax.annotate(f"{c:.0f}% complied", xy=(1.8, yy), ha="left", va="center",
                    color=INK, fontsize=21.8)
        n = N.get(key)
        ax.annotate(f"{lab}  ({n} items)" if n else lab, xy=(-2.0, yy),
                    ha="right", va="center", color=INK2, fontsize=20.3,
                    annotation_clip=False)
        ax.annotate(f"{100 - c:.0f}%", xy=(103, yy), ha="left", va="center",
                    color=INK, fontsize=21.8, annotation_clip=False)

    first = {}
    for (key, lab, b), yy in zip(rows, ypos):
        first.setdefault(b, yy)
    for b, yy in first.items():
        ax.annotate(BLOCK[b], xy=(-2.0, yy - 1.05), ha="right", va="center",
                    color=INK, fontsize=21.8, linespacing=1.35,
                    annotation_clip=False)
        # a hairline under each block name, or the name and the row label
        # below it read as one four-line block of text
        ax.plot([0, 100], [yy - 0.66] * 2, color=GRID, lw=1.0, zorder=1)
    ax.annotate("did not\ncomply", xy=(103, min(ypos) - 1.05), ha="left",
                va="center", color=INK2, fontsize=19.6, linespacing=1.35,
                annotation_clip=False)

    ax.set_xlim(0, 100)
    ax.set_ylim(max(ypos) + 0.62, min(ypos) - 1.55)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("share of the model's replies")
    ax.grid(False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    handles = [Patch(facecolor=C_COMPLY, edgecolor="none",
                     label="it did what was asked"),
               Patch(facecolor=C_DEFLECT, edgecolor="none",
                     label="it deflected without saying so"),
               Patch(facecolor=C_REFUSE, edgecolor="none",
                     label="it refused in so many words")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.010),
              ncols=3, frameon=False, handlelength=1.1, handletextpad=0.6,
              columnspacing=2.4, borderaxespad=0.0, fontsize=20.3)

    # what counts as a refusal, that every row is the same photographs put a
    # different way, and that a jailbreak rate quoted off this benchmark is
    # measuring the typography rather than the picture: all of it is in this
    # figure's caption in paper/atlas.html now.  The row labels carry the item
    # counts, which is data, so they stay.
    top = _fig_head(fig, "Where the attack is not", panel_head=26.0)
    fig.subplots_adjust(left=0.260, right=0.925, top=top, bottom=0.125)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure C

def hallucination(d, out, name="f82_hallucination.png"):
    """Which absent objects a model invents.

    POPE asks about an object that is not in the photograph and counts the
    times the model says yes anyway.  The three splits differ only in how that
    absent object was chosen, so the three bars isolate what drives an
    invention: not the picture, but which object the question names.
    """
    style.use()
    det = d["detect"]
    layers = sorted(int(l) for l in det)
    n_h = int(det[str(layers[0])].get("n_hallucinated", 0))
    n_c = int(det[str(layers[0])].get("n_correct", 0))

    fig, ax = plt.subplots(1, 1, figsize=(9.2, 6.0))

    acc = d.get("accuracy") or {}
    SPL = [("random", "drawn at\nrandom"),
           ("popular", "the commonest\nobjects"),
           ("adversarial", "objects that keep\ncompany with these")]
    spl = [(k, l) for k, l in SPL if k in acc]
    xs = list(range(len(spl)))
    for x, (k, l) in zip(xs, spl):
        a = acc[k]
        rate = (1 - a["accuracy"]) * 100
        cnt = int(round(a["n"] * (1 - a["accuracy"])))
        ax.add_patch(Rectangle((x - 0.3, 0), 0.6, rate, facecolor=style.RED,
                               edgecolor="none", zorder=3))
        ax.annotate(f"{rate:.1f}%\n{cnt} of {int(a['n'])}", xy=(x, rate),
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    va="bottom", color=INK2, fontsize=19.6, linespacing=1.35)
    ax.set_xlim(-0.62, len(spl) - 0.38)
    ax.set_ylim(0, 18)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l in spl], fontsize=19.6, linespacing=1.3)
    ax.set_xlabel("how the absent object was chosen")
    ax.set_ylabel("questions about an absent object\nanswered \u201cyes\u201d  (%)")
    # a subtitle a reader can refute by looking at the panel under it was
    # already the sin here once -- "the harder the distractor, the more often",
    # when the adversarial split is a shade BELOW the popular one.  There is no
    # subtitle now at all; the shape of the three bars is in the caption.
    style.despine(ax)

    print(f"  hallucination: {n_h} inventions against {n_c} correct refusals",
          flush=True)
    top = _fig_head(fig, "Where a model\u2019s inventions come from",
                    panel_head=HEAD_PT)
    fig.subplots_adjust(left=0.115, right=0.975, top=top, bottom=0.155)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p
