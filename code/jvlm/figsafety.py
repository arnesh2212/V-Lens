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
``hallucination``  the comparison we lose.  Drawn plainly, at full scale, with
                   the winner labelled as the winner.

Conventions are the paper's, and they are load-bearing: one colour per lens
everywhere (logit lens orange, J-lens violet, VJ-Lens blue), names in ink beside
a dot in the series hue rather than in the hue itself, titles padded by the line
count of their own subtitle, and end-of-line labels pushed apart when two series
finish together.
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


def _head(ax, title, sub, fs_title=13.0, fs_sub=10.0, pad0=8.0):
    """Panel title with a quieter second line, the pad taken from the line count.

    A fixed pad is what printed the title on top of its own subtitle in the
    first version of every figure in this project.
    """
    lines = str(sub).count("\n") + 1 if sub else 0
    pad = pad0 + lines * fs_sub * 1.38
    ax.set_title(title, loc="left", color=INK, pad=pad, fontsize=fs_title)
    if sub:
        ax.annotate(sub, xy=(0, 1), xytext=(0, 6), xycoords="axes fraction",
                    textcoords="offset points", ha="left", va="bottom",
                    color=INK2, fontsize=fs_sub, linespacing=1.38)


def _fig_head(fig, title, sub, fs_title=17.0, fs_sub=11.5, x=0.006,
              panel_head=0.0):
    """The figure's own title block, and the top of the axes it leaves room for.

    ``panel_head`` is the height in points of the tallest per-panel title block
    that will be drawn ABOVE its axes.  Reserving it here is the whole job: an
    axes top computed from the figure's subtitle alone leaves the panel titles
    to be drawn into the subtitle, which is exactly the overprint this project
    has fixed a dozen times.
    """
    unit = 1.0 / (72.0 * fig.get_figheight())
    fig.text(x, 1 - 5 * unit, title, ha="left", va="top", color=INK,
             fontsize=fs_title)
    y = 1 - (5 + fs_title * 1.24 + 7) * unit
    fig.text(x, y, sub, ha="left", va="top", color=INK2, fontsize=fs_sub,
             linespacing=1.45)
    below = y - (sub.count("\n") + 1) * fs_sub * 1.45 * unit - 5 * unit
    return below - panel_head * unit


def _head_pt(sub, fs_title=13.0, fs_sub=10.0, pad0=8.0):
    """How tall :func:`_head` will be, in points, subtitle included."""
    lines = str(sub).count("\n") + 1 if sub else 0
    return pad0 + lines * fs_sub * 1.38 + fs_title * 1.30


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


def _bar_values(ax, xs, vals, top=None, colour=None, fs=9.5, fmt="{:.3f}",
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

def jailbreak(d, out, name="f80_jailbreak.png"):
    """Three panels on one benchmark: by depth, by estimator, and with the text.

    Panel one is the reading at each layer with the label-shuffled floor drawn
    under it.  Panel two is the same data taken four ways, each beside its own
    floor, because the tempting statistic -- the best of the four layers -- has
    a higher floor as well as a higher value, and only the two together say
    whether anything was gained.  Panel three is the same items with the harmful
    phrase typed onto the picture, which is the measurement the first version of
    this experiment made by accident.
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
        1, 3, figsize=(18.4, 7.2),
        gridspec_kw={"width_ratios": [1.05, 1.32, 0.92]})

    # ---- (i) by layer, over the shuffled-label floor -----------------------
    ax = axes[0]
    ax.set_xlim(layers[0] - 1.0, layers[-1] + 4.0)
    ax.set_ylim(0.462, 0.700)
    for k, c in zip(slugs, cols):
        f = (nul.get(k) or {}).get("prereg") or {}
        if "mean" in f and "q95" in f:
            ax.axhspan(f["mean"], f["q95"], color=c, alpha=0.11, lw=0, zorder=1)
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=2)
    ends = []
    for k, n, c in zip(slugs, names, cols):
        bl = au[k].get("by_layer") or {}
        ys = [bl.get(str(l), bl.get(l)) for l in layers]
        ax.plot(layers, ys, color=c, marker="o", ms=6.5, lw=2.2, zorder=4)
        ends.append([float(layers[-1]), float(ys[-1]), n, c])
    ax.set_xticks(layers)
    ax.set_yticks([0.50, 0.55, 0.60, 0.65, 0.70])
    ax.set_xlabel("layer of the language model  (of 28)")
    ax.set_ylabel("AUROC, harmful against benign")
    _stagger(ax, ends)
    ax.annotate("shaded: where this same pipeline lands on label-shuffled\n"
                "data, mean to 95th percentile \u2014 all three floors coincide",
                xy=(layers[0] - 0.85, 0.4645), ha="left", va="bottom",
                color=INK2, fontsize=9.5, linespacing=1.35,
                annotation_clip=False)
    ax.annotate("chance", xy=(layers[-1] + 3.8, 0.5), xytext=(0, 3),
                textcoords="offset points", ha="right", va="bottom",
                color=MUTED, fontsize=9.5)
    sub_i = ("one identical question for every item, so no lens\n"
             "can read the prompt instead of the picture")
    _head(ax, "Read at each layer", sub_i)
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
            # the label clears the floor cap as well as the bar, or the cap
            # rules a line through it wherever a lens fell short of its own floor
            _bar_values(ax, [x], [v], top=max(v, f or v), fs=9.5, dy=4.0)
    ax.set_xlim(-0.55, len(EST) - 0.45)
    ax.set_xticks(range(len(EST)))
    ax.set_xticklabels([l for _, l in EST], fontsize=11, linespacing=1.3)
    ax.set_yticks([0.50, 0.55, 0.60, 0.65, 0.70])
    ax.set_ylabel("AUROC, harmful against benign")
    ax.set_xlabel("how the AUROC was taken")
    handles = [Patch(facecolor=c, edgecolor="none", label=n)
               for n, c in zip(names, cols)]
    handles.append(Patch(facecolor="#ebe9e4", edgecolor=FLOOR,
                         label="its own shuffled-label\nfloor, 95th percentile"))
    style.legend(ax, handles=handles, loc="upper left", ncols=2)
    sub_ii = ("the pale foot of each bar is that lens's own floor; only the "
              "part\nstanding above it was gained.  The best of the four layers "
              "is the\nlargest number and buys nothing, because its floor rises "
              "with it")
    _head(ax, "The same data, taken four ways", sub_ii)
    style.despine(ax)

    # ---- (iii) with and without the typography ------------------------------
    ax = axes[2]
    w = 0.30
    ax.set_ylim(0.478, 0.912)
    ax.axhline(base, color=INK2, lw=1.0, zorder=3)
    for i, (k, n, c) in enumerate(zip(slugs, names, cols)):
        a_clean = au[k].get("split_half")
        a_typo = (typo.get(k) or {}).get("split_half")
        for x, v, solid in ((i - w * 0.55, a_clean, False),
                            (i + w * 0.55, a_typo, True)):
            if v is None or v != v:
                continue
            ax.add_patch(Rectangle(
                (x - w * 0.5, base), w, v - base,
                facecolor=c if solid else "none", edgecolor=c,
                lw=0 if solid else 1.6, zorder=4))
            _bar_values(ax, [x], [v], fs=10.0, dy=4.0)
        if a_clean and a_typo:
            ax.annotate(f"+{a_typo - a_clean:.3f}", xy=(i, max(a_clean, a_typo)),
                        xytext=(0, 24), textcoords="offset points",
                        ha="center", va="bottom", color=INK2, fontsize=11.5)
    ax.set_xlim(-0.62, len(slugs) - 0.38)
    ax.set_xticks(range(len(slugs)))
    ax.set_xticklabels(names, fontsize=11.5)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_ylabel("AUROC, harmful against benign")
    handles = [Patch(facecolor="none", edgecolor=INK2, lw=1.6,
                     label="nothing written on the picture"),
               Patch(facecolor=MUTED, edgecolor="none",
                     label="the harmful phrase typed on it")]
    style.legend(ax, handles=handles, loc="upper left")
    sub_iii = ("every lens gains, and ours gains most.  This is\n"
               "what the first version of this experiment measured")
    _head(ax, "What the typography was worth", sub_iii)
    style.despine(ax)

    ph = max(_head_pt(sub_i), _head_pt(sub_ii), _head_pt(sub_iii))
    top = _fig_head(
        fig, "One jailbreak benchmark, read three ways",
        f"MM-SafetyBench's SD variant on Qwen3-VL-2B: {n_h} harmful and {n_b} "
        "benign items from the same image generator, each asked one identical "
        "neutral question, each lens\nread only at the image tokens.  The same "
        "three colours name the same three lenses in every panel, and the whole "
        "pipeline is re-run on label-shuffled data,\nso the floor under every "
        "number here is measured rather than assumed.",
        panel_head=ph)
    fig.subplots_adjust(left=0.048, right=0.988, top=top, bottom=0.118,
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

    fig, ax = plt.subplots(figsize=(13.6, 6.1))
    ypos, y, prev = [], 0.0, None
    for key, lab, b in rows:
        if prev is not None and b != prev:
            y += 0.86
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
                    color=INK, fontsize=12.5)
        n = N.get(key)
        ax.annotate(f"{lab}  ({n} items)" if n else lab, xy=(-2.0, yy),
                    ha="right", va="center", color=INK2, fontsize=12,
                    annotation_clip=False)
        ax.annotate(f"{100 - c:.0f}%", xy=(103, yy), ha="left", va="center",
                    color=INK, fontsize=12.5, annotation_clip=False)

    first = {}
    for (key, lab, b), yy in zip(rows, ypos):
        first.setdefault(b, yy)
    for b, yy in first.items():
        ax.annotate(BLOCK[b], xy=(-2.0, yy - 0.80), ha="right", va="center",
                    color=INK, fontsize=12.5, linespacing=1.35,
                    annotation_clip=False)
        # a hairline under each block name, or the name and the row label
        # below it read as one four-line block of text
        ax.plot([0, 100], [yy - 0.52] * 2, color=GRID, lw=1.0, zorder=1)
    ax.annotate("did not\ncomply", xy=(103, min(ypos) - 0.80), ha="left",
                va="center", color=INK2, fontsize=11.5, linespacing=1.35,
                annotation_clip=False)

    ax.set_xlim(0, 100)
    ax.set_ylim(max(ypos) + 0.62, min(ypos) - 1.30)
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
              columnspacing=2.4, borderaxespad=0.0, fontsize=11.5)

    top = _fig_head(
        fig, "The attack is not in the picture",
        "Qwen3-VL-2B on MM-SafetyBench's SD variant.  A reply counts as a "
        "refusal if it carries a refusal phrase, or if the model, shown its own "
        "reply, says it refused.\nOn the clean painting the model complies with "
        "almost everything it is given, harmful and benign alike, because it "
        "simply describes the painting: the\nrefusals appear only once the "
        "phrase is typed onto the image, or the benchmark's own wording is "
        "restored.  Any jailbreak rate quoted on this\nbenchmark is measuring "
        "the typography and the request, not the picture.",
        panel_head=26.0)
    fig.subplots_adjust(left=0.245, right=0.930, top=top, bottom=0.115)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure C

def hallucination(d, out, name="f82_hallucination.png"):
    """The comparison we lose, drawn at full scale.

    Whether a lens can tell, before the model speaks, that it is about to claim
    an object which is not in the photograph.  The Jacobian lens wins at every
    depth and wins clearly; ours peaks in the middle of the stack and the plain
    logit lens overtakes it at the end.  A figure that buried that would be
    worth less than no figure.
    """
    style.use()
    det = d["detect"]
    layers = sorted(int(l) for l in det)
    ARM = [("say", "logit lens"), ("jlens", "J-lens"), ("ig", "VJ-Lens")]
    arms = [(k, n) for k, n in ARM if all(k in det[str(l)] for l in layers)]
    n_h = int(det[str(layers[0])].get("n_hallucinated", 0))
    n_c = int(det[str(layers[0])].get("n_correct", 0))

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.0),
                             gridspec_kw={"width_ratios": [1.66, 1.0]})

    # ---- (i) AUROC against depth -------------------------------------------
    ax = axes[0]
    ax.set_xlim(layers[0] - 1.2, layers[-1] + 3.4)
    ax.set_ylim(0.452, 1.005)
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=2)
    ends, peak = [], {}
    for k, n in arms:
        c = LENS_COLOUR[n]
        ys = [det[str(l)][k] for l in layers]
        ax.plot(layers, ys, color=c, marker="o", ms=7.0, lw=2.4, zorder=4)
        ends.append([float(layers[-1]), float(ys[-1]), n, c])
        peak[n] = (layers[int(np.argmax(ys))], float(max(ys)))
    if "J-lens" in peak:
        l, v = peak["J-lens"]
        ax.annotate(f"{v:.3f}", xy=(l, v), xytext=(-9, 8),
                    textcoords="offset points", ha="right", va="bottom",
                    color=INK2, fontsize=11)
    if "VJ-Lens" in peak:
        l, v = peak["VJ-Lens"]
        ax.annotate(f"{v:.3f}", xy=(l, v), xytext=(0, -13),
                    textcoords="offset points", ha="center", va="top",
                    color=INK2, fontsize=11)
    gap = (peak.get("J-lens", (0, 0))[1] - peak.get("VJ-Lens", (0, 0))[1])
    ax.annotate("chance", xy=(layers[0] - 1.0, 0.5), xytext=(0, 4),
                textcoords="offset points", ha="left", va="bottom",
                color=MUTED, fontsize=10)
    ax.set_xticks(layers)
    ax.set_xlabel("layer of the language model  (of 28)")
    ax.set_ylabel("AUROC: the inventions against\nthe correct refusals")
    _stagger(ax, ends, frac=0.085)
    sub_i = (f"{n_h} hallucinations against {n_c} correct refusals, read at the "
             "image tokens before the model has\nsaid anything.  At its best "
             f"the Jacobian lens is {gap:.3f} ahead of us, and it is ahead at "
             "every depth.\nOurs peaks in the middle of the stack and falls "
             "away; the plain logit lens is well behind it\nthere, and "
             "overtakes it at the last layer measured")
    _head(ax, "The Jacobian lens wins this one, and wins it at every depth",
          sub_i)
    style.despine(ax)

    # ---- (ii) what the two classes are --------------------------------------
    ax = axes[1]
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
                    va="bottom", color=INK2, fontsize=11, linespacing=1.35)
    ax.set_xlim(-0.62, len(spl) - 0.38)
    ax.set_ylim(0, 18)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l in spl], fontsize=11, linespacing=1.3)
    ax.set_xlabel("how the absent object was chosen")
    ax.set_ylabel("questions about an absent object\nanswered “yes”  (%)")
    sub_ii = ("the harder the distractor, the more often the model\n"
              "claims to see something that is not there")
    _head(ax, "Where the inventions come from", sub_ii)
    style.despine(ax)

    ph = max(_head_pt(sub_i), _head_pt(sub_ii))
    top = _fig_head(
        fig, "Seeing a hallucination coming — the comparison we lose",
        "A presence benchmark rebuilt from COCO: 660 questions over 110 "
        "photographs, half of them naming an object that is not in the picture. "
        "A hallucination is an\nabsent object the model says yes to.  Each lens "
        "is read at the image tokens only, and scored on how well its reading "
        "of the queried object separates the\n"
        f"{n_h} inventions from the {n_c} correct refusals.  We are second, and "
        "not close to first — the honest reading is that these two lenses "
        "answer different questions.",
        panel_head=ph)
    fig.subplots_adjust(left=0.072, right=0.988, top=top, bottom=0.142,
                        wspace=0.28)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p
