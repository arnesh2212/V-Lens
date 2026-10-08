# SPDX-License-Identifier: Apache-2.0
"""The four patching figures, redrawn to the house style.

Each function here makes the same claim as the one of the same name in
:mod:`vlens.figures.paper`, from the same result file, under the same file
name. What changed is the encoding:

* one colour means one thing in every panel of every figure here. Clay is
  the patches V-Write ranks highest; the darker clay is the other end of the
  same ranking and the pale clay its middle, so the whole ranking stays in
  one family and is told apart by lightness. Slate is attribution patching,
  sage is the reading lenses, grey is a random selection, and the
  faint dotted rule is what covering the whole picture does. The other
  attribution methods each keep the colour and dash
  :mod:`vlens.figures.paper` gives them everywhere.
* the reading lenses are a band, not hairlines: a ribbon from
  the lowest to the highest of them, with their mean drawn through it. An
  attribution method is never folded into the band, since each one is a
  separate claim about the same patches and is drawn as its own line.
* keys are drawn once per figure, in a band above the panels, each name in ink
  beside a stroke of its own colour. Nothing is written on top of a line.
* every axis is scaled to the ink that is on it, and the zero line is drawn
  only where it sits clear of the bottom spine.
"""
from __future__ import annotations

import math

import matplotlib
import matplotlib.ticker as mticker
import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st
from vlens.figures.paper import NICE, SOURCE, _at, _load, _newest

#: how each ranking is drawn, the same in every figure
LINE = {
    "v_write": dict(color=st.CLAY, lw=2.8, marker="o", ms=5.5, zorder=6),
    "top": dict(color=st.CLAY, lw=2.8, marker="o", ms=5.5, zorder=6),
    "bottom": dict(color=st.CLAY_DK, lw=2.2, marker="o", ms=4.6, zorder=5),
    "middle": dict(color=st.CLAY_SOFT, lw=2.3, marker="o", ms=4.6, dashes=(1.4, 1.6), zorder=4),
    "attr_patch": dict(color=st.SLATE, lw=2.1, marker="o", ms=4.6, zorder=5),
    "random": dict(color=st.MUTED, lw=1.7, marker="o", ms=3.6, dashes=(4, 2.2), zorder=3),
}
#: the published attribution methods V-Write is compared with, drawn one
#: line each, thinner than V-Write and above the band
ATTRIB = ("act_patch", "ig_zero", "ig_mean32", "grad_x_act", "rise", "kernel_shap", "gradcam")
LINE.update({r: dict(color=P.COLOUR[r], lw=1.7, marker="o", ms=3.8, zorder=4,
                     **({"dashes": P.DASH[r]} if r in P.DASH else {}))
             for r in ATTRIB})
#: the reading lenses, drawn as one band
LENS = st.SAGE
#: what covering the whole picture does, a quiet reference
WHOLE = dict(color=st.AXIS, lw=1.3, dashes=(2, 2), zorder=2)

#: the name each ranking carries in a key or beside a dot. The two lenses
#: fitted to the answer token get a shorter name than :data:`NICE` gives them,
#: because a column of names is only as narrow as its longest entry.
NAME = dict(NICE, v_write="V-Write", attr_patch="attribution patching",
            v_read_cross="V-Read (answer)", tuned_cross="tuned lens (answer)",
            random="random")


#: small counts are written as words in a key
_WORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def _key_entry(key, text=None):
    d = LINE[key]
    return (text or NAME.get(key, key), d["color"], d.get("dashes", ()))


def _fit(ax, lo=None, hi=None):
    """Scale the vertical axis to the ink, and mark zero only where it is clear of the spine.

    When the data dips below zero the axis gets a little air under it and a
    zero line inside the frame. When nothing is negative, the frame starts a
    hair under zero so a line lying on zero is still seen as a line and not
    as a thicker spine; the zero line is then left out, since a second rule
    a few points above the spine reads as a doubled one.
    """
    ys = [np.asarray(l.get_ydata(), float) for l in ax.lines]
    ys = [y[np.isfinite(y)] for y in ys]
    ys = [y for y in ys if y.size]
    if not ys and lo is None:
        return
    a = min([float(y.min()) for y in ys] + ([lo] if lo is not None else []))
    b = max([float(y.max()) for y in ys] + ([hi] if hi is not None else []))
    rng = max(b - min(a, 0.0), 1e-9)
    if a < -0.02 * rng:
        st.fit_y(ax, lo=a, hi=b, pad=0.08)
        ax.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
    else:
        st.fit_y(ax, lo=min(a, 0.0), hi=b, pad=0.08)
        ax.set_ylim(bottom=min(a, 0.0) - 0.06 * rng)


def _percent_axis(ax, ks, every=True):
    """The horizontal axis every figure here shares: per cent of patches replaced."""
    ticks = ks if every else [ks[0], ks[len(ks) // 2], ks[-1]]
    ax.set_xticks(ticks)
    if not every:
        ax.set_xticklabels([f"{k:g}%" for k in ticks])
    ax.set_xlim(ks[0] - 3, ks[-1] + 3)


def _band(ax, ks, curves):
    """Several near-identical curves drawn as the one band they are."""
    band = np.asarray(curves, float)
    lo, hi = np.nanmin(band, 0), np.nanmax(band, 0)
    st.ribbon(ax, ks, lo, hi, LENS, alpha=0.26)
    ax.plot(ks, np.nanmean(band, 0), color=LENS, lw=1.6, zorder=3)
    return float(np.nanmin(lo)), float(np.nanmax(hi))


def _label_size():
    return matplotlib.rcParams["axes.labelsize"]


def _tick_size():
    return matplotlib.rcParams["xtick.labelsize"]


# ------------------------------------------------------------------ causal

def patch_causal(results="results", out="figures", model=None):
    """MAIN. V-Write's ranking of image patches is causal and signed: replacing the patches it ranks highest takes the answer away, replacing the ones it ranks lowest pushes the answer further the way it already went, and the middle of the ranking does no more than a random selection.

    Shown for Qwen3-VL-8B-Instruct at block 18 of 36 (598 records) and
    Qwen3.5-27B at block 32 of 64 (2668 records), both at normalised depth
    0.51; the averaged file is the mean of the two models' curves. Records are
    the SimonSays yes/no, test-or-real and WMDP carrier letter items from all
    five sources (Open Images, nuScenes, Interfaces (VINS-120K), Generated
    scenes, WMDP carrier), each asked exactly as the set poses it, with its own
    cue and system instruction where it has one. The answer score is the logit
    margin, at the answer spot, of the answer the model gave over its
    alternative (a letter item is kept only when the model picks the correct
    letter), and a record is kept only when that margin is positive. V-Write
    scores each image patch by the gradient of that score integrated over
    eight steps from the picture's own mean image-token state. The top,
    bottom, middle or a random 5, 10, 20, 30 or 50 per cent of the image-token
    states at that block are replaced by the same mean state and the forward
    pass continues. Left: the mean fall in the answer score. Right: the share
    of records whose margin falls to zero or below, which is an answer
    flipped. The password-locking families are excluded (none is in these
    runs).

    At a fifth of the picture replaced, V-Write's top end costs
    Qwen3-VL-8B-Instruct 3.20 and flips 14.5 per cent of answers, and costs
    Qwen3.5-27B 0.89 and flips 24.0 per cent. Its bottom end moves the margin
    the other way, by 1.65 and 0.55, and flips no answer in either model. The
    middle of the ranking (0.17 and 0.01) and a random fifth (0.19 and 0.03)
    sit near nothing. Top and bottom move the same record's margin in opposite
    directions on 95.5 and 96.2 per cent of records. In the same run,
    attribution patching's top fifth costs 1.43 and 0.55 and gradient times
    activation 0.93 and 0.31.

    Limits. The replacement is the same mean state V-Write integrates from,
    so this test is the counterfactual V-Write is built around. Flips level
    off at 15.2 and 30.0 per cent even at half the picture: most answers
    survive losing the patches that support them most. The starting margins
    differ (median 9.4 nats on Qwen3-VL-8B-Instruct, 1.3 on Qwen3.5-27B), so
    falls are not comparable across models. The records lean on Open Images:
    371 of the 598 Qwen3-VL-8B-Instruct records are unedited Open Images
    authority items, and 1985 of the 2668 Qwen3.5-27B records are unedited
    Open Images authority or eval-awareness items (997 and 988).
    """
    p = _newest(f"{results}/patchcausal/{model or '*'}__patchcausal.json")
    d = _load(p) if p else None
    if not d:
        return None
    s = d["summary"]
    ks = [100 * k for k in s["ks"]]
    groups = [g for g in ("top", "bottom", "middle", "random") if g in s["curve"]]
    if not groups:
        return None
    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(5.25, 3.15))
    for g in reversed(groups):                       # the claim is drawn last, on top
        a1.plot(ks, s["curve"][g], **LINE[g])
        if g in s.get("flip_rate", {}):
            a2.plot(ks, [100 * v for v in s["flip_rate"][g]], **LINE[g])
    for ax in (a1, a2):
        _percent_axis(ax, ks)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        st.despine(ax)
        _fit(ax)
    a1.set_ylabel("fall in the answer score")
    a2.set_ylabel("answers that flip (%)")
    fig.supxlabel("patches replaced (%)", fontsize=_label_size())
    shown = {"top": "V-Write's highest", "bottom": "its lowest",
             "middle": "the middle", "random": "random"}
    # the key is one row above both panels, in a band of its own
    st.key_band_lines(fig, [_key_entry(g, shown[g]) for g in groups])
    return st.save(fig, _at(out, "patch_causal"))


# ------------------------------------------------------------------ lenses

def patch_lenses(results="results", out="figures", model=None):
    """MAIN. Asked which image patches carry the answer, V-Write agrees with what covering the picture actually does and removes the most answer when its top patches are zeroed, ahead of attribution patching, while every reading lens (logit lens, tuned lens, Jacobian lens and V-Read itself) ranks the patches no better than chance.

    Shown for Qwen3-VL-8B-Instruct at block 18 of 36 (1003 records) and
    Qwen3.5-27B at block 32 of 64 (564 records), both at normalised depth
    0.51; the averaged file is the mean of the two. Records are SimonSays
    yes/no, test-or-real and WMDP carrier letter items from all five sources,
    each asked as the set poses it with its own cue and system instruction
    where it has one, scored by the logit margin of the answer the model gave
    at the answer spot. A reading lens scores a patch by its readout of that
    patch minus its readout of the picture's mean image-token state, both
    through the model's own final norm and unembedding; the lenses are V-Read
    and the tuned lens each fitted to the next token and to the answer, the
    Jacobian lens and the logit lens, all fitted on prompts none of these
    records appear in. V-Write is the gradient path from the picture's mean
    state, and attribution patching its one-point form. Left: the image
    tokens are cut into sixteen groups in raster order, each group is replaced
    by zeros and the fall in the answer score measured; the value is the
    Spearman correlation, over the sixteen groups, between an instrument's
    summed attribution and that measured fall, averaged over records. Right:
    the mean fall in the answer score when the top 5 to 50 per cent of patches
    by each ranking are replaced by zeros, a replacement none of the
    instruments was built around. The six reading lenses are drawn as one
    band (lowest to highest, with their mean). Password-locking families are
    excluded (none is in these runs).

    Rank agreement with the measured effect: V-Write 0.33 on
    Qwen3-VL-8B-Instruct and 0.35 on Qwen3.5-27B, attribution patching 0.20
    and 0.27, the reading lenses between -0.02 and 0.04 and between -0.04 and
    0.05, random -0.01 on both. With a fifth of the patches zeroed, V-Write's
    ranking costs 4.02 and 1.13, attribution patching's 1.93 and 0.70, the
    reading lenses' 1.10 to 1.60 and 0.04 to 0.15, and a random fifth 1.16 and
    0.07.

    Limits. V-Read is a reading lens and does not rank patches: on
    Qwen3.5-27B the ranking of V-Read fitted to the next token removes less
    than a random one (0.04 against 0.07). The other published attribution methods (integrated gradients from
    zero, activation patching, RISE, KernelSHAP, Grad-CAM) are not in these
    runs and so are not drawn. One block per model, and the two models were
    scored on different record samples.
    """
    p = _newest(f"{results}/patchlenses/{model or '*'}__patchlenses.json")
    d = _load(p) if p else None
    if not d:
        return None
    s_ = d["summary"]
    ks = [100 * k for k in s_["ks"]]
    drop = s_.get("drop", {})
    graded = s_.get("graded_against_patching", {})
    order = [r for r in s_["rankings"] if f"zero:{r}" in drop and not P._hidden(r)]
    if not order:
        return None
    named = [r for r in ("v_write", "attr_patch", *ATTRIB, "random") if r in order]
    lenses = [r for r in order if r not in LINE]

    def colour(r):
        return LINE[r]["color"] if r in LINE else LENS

    # left: a dot per instrument, position carrying the number, the top row the
    # instrument that agrees most. The row names double as the figure's key.
    vals = [(r, float(graded[f"zero|{r}"])) for r in order
            if graded.get(f"zero|{r}") is not None and np.isfinite(graded[f"zero|{r}"])]
    vals.sort(key=lambda t: -t[1])
    n = len(vals)

    # the rows keep the spacing nine of them have in the standard height, so
    # adding the attribution methods makes the figure taller rather than
    # setting their names on top of one another
    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(5.25, max(3.7, 0.41 * n)),
                            gridspec_kw=dict(width_ratios=[1.0, 1.2]))
    for j, (r, v) in enumerate(vals):
        y = n - 1 - j
        # a method that shares its colour with another is dashed in every
        # figure, so its stem is dashed here too and its dot is left open
        dash = LINE.get(r, {}).get("dashes")
        a1.plot([0, v], [y, y], color=colour(r), lw=1.5, alpha=0.6, zorder=2,
                solid_capstyle="butt", **({"dashes": dash} if dash and r != "random" else {}))
        a1.plot([v], [y], marker="o", ms=8.5, color=colour(r), ls="none", zorder=3,
                **({"markerfacecolor": st.SURFACE, "markeredgewidth": 1.6}
                   if dash and r != "random" else {}))
    a1.set_yticks(range(n))
    a1.set_yticklabels([NAME.get(r, r) for r, _ in vals][::-1])
    a1.set_ylim(-0.6, n - 0.4)
    a1.axvline(0, color=st.AXIS, lw=1.0, zorder=1)
    if vals:
        lo = min(min(v for _, v in vals), 0.0)
        hi = max(v for _, v in vals)
        rng = max(hi - lo, 1e-9)
        a1.set_xlim(lo - 0.06 * rng, hi + 0.08 * rng)
    a1.xaxis.set_major_locator(mticker.MaxNLocator(4))
    a1.set_xlabel("rank agreement")
    st.despine(a1, left=False)
    a1.tick_params(axis="y", length=0)
    a1.yaxis.grid(False)

    # right: what acting on each ranking does
    lo = hi = None
    if lenses:
        lo, hi = _band(a2, ks, [drop[f"zero:{r}"] for r in lenses])
    for r in reversed(named):
        a2.plot(ks, drop[f"zero:{r}"], **LINE[r])
    _percent_axis(a2, ks)
    a2.yaxis.set_major_locator(mticker.MaxNLocator(5))
    st.despine(a2)
    _fit(a2, lo, hi)
    a2.set_ylabel("fall in the answer score")
    # the layout budgets an axis label's height but not its width, so a label
    # wider than its panel runs off the page; this one is kept inside the panel
    a2.set_xlabel("patches replaced by zero (%)")
    return st.save(fig, _at(out, "patch_lenses"))


# ------------------------------------------------------------------ sources

def patch_sources(results="results", out="figures", model=None):
    """APPENDIX. V-Write's lead over attribution patching and over every reading lens holds on each of the five SimonSays sources taken alone, and the size of what covering can remove tracks whether the picture holds the answer: on the WMDP carrier, where the picture only carries a question asked in text, the whole effect is several times smaller.

    One panel per source, each on its own vertical scale, sorted by V-Write's
    largest fall; under each panel the source and the number of pictures it
    averages. Qwen3-VL-8B-Instruct at block 18 of 36 and Qwen3.5-27B at
    block 32 of 64 (normalised depth 0.51); the averaged file is the mean of
    the two. The records, scores and rankings are those of the patch-ranking
    run: each record asked as the set poses it, the answer score the logit
    margin of the answer the model gave at the answer spot, and the top 5 to
    50 per cent of image patches by each ranking replaced by zeros. The six
    reading lenses (V-Read and the tuned lens each fitted to the next token
    and to the answer, the Jacobian lens, the logit lens) are one band. The
    WMDP carrier panel holds the three wmdp_*_sand families; the
    password-locking families are excluded.

    Fall with a fifth of the patches zeroed, V-Write against attribution
    patching, the lens band and random. Qwen3-VL-8B-Instruct: Open Images
    (481 pictures) 4.63, 2.46, 1.58 to 1.97, 1.61; Interfaces (VINS-120K)
    (209) 4.04, 2.02, 1.05 to 1.80, 1.15; Generated scenes (100) 4.16, 1.49,
    -0.01 to 0.77, 0.62; nuScenes (78) 6.09, 1.87, 0.99 to 3.20, 1.03; WMDP
    carrier (135) 0.53, 0.28, 0.02 to 0.14, 0.06. Qwen3.5-27B: Open Images
    (250) 1.39, 0.86, 0.05 to 0.21, 0.08; Interfaces (113) 0.97, 0.59, 0.05 to
    0.19, 0.10; Generated scenes (32) 3.67, 2.39, -0.27 to 0.32, 0.29;
    nuScenes (25) 1.34, 0.72, -0.02 to 0.37, -0.10; WMDP carrier (144) 0.21,
    0.15, -0.01 to 0.01, 0.00. V-Write removes 1.4 to 3.3 times what
    attribution patching removes on every source in both models.

    Limits. The reading lenses do not always sit on the random ranking: on
    nuScenes for Qwen3-VL-8B-Instruct the top of the band (V-Read fitted to
    the next token) reaches 3.20 against 1.03 for random. The Generated
    scenes and nuScenes panels of Qwen3.5-27B average only 32 and 25
    pictures. In the averaged file the count under each panel is the two
    models' counts added, and each curve is the mean of the two models'
    curves, so Qwen3-VL-8B-Instruct's larger falls dominate its scale.
    """
    p = _newest(f"{results}/patchlenses/{model or '*'}__patchlenses.json")
    d = _load(p) if p else None
    if not d or not d["summary"].get("per_source"):
        return None
    s_ = d["summary"]
    per, ns = s_["per_source"], s_.get("per_source_n") or {}
    ks = [100 * k for k in s_["ks"]]
    srcs = sorted(per, key=lambda f: -float(np.nanmax(per[f].get("v_write") or [0])))
    if not srcs:
        return None
    cols = 3
    rows = max(1, math.ceil((len(srcs) + 1) / cols))   # always one spare cell for the key
    st.use()
    fig, axes = st.wide(rows, cols, unit=(3.5, 2.25))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(srcs):]:
        fig.delaxes(ax)
    for ax, f in zip(axes, srcs):
        lenses = [r for r in s_["rankings"] if r in per[f] and r not in LINE and not P._hidden(r)]
        lo = hi = None
        if lenses:
            lo, hi = _band(ax, ks, [per[f][r] for r in lenses])
        for r in ("random", *ATTRIB, "attr_patch", "v_write"):
            if r in per[f] and np.isfinite(np.asarray(per[f][r], float)).any():
                ax.plot(ks, per[f][r], **LINE[r])
        _percent_axis(ax, ks, every=False)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
        st.despine(ax)
        _fit(ax, lo, hi)
        # the panel's name and its count go where an axis label would, in the
        # tick size: short, under the data, and never across another panel's
        ax.set_xlabel(f"{SOURCE.get(f, f)}\n{ns.get(f, 0)} pictures", fontsize=_tick_size())
    fig.supylabel("fall in the answer score", fontsize=_label_size())
    # the key hangs off the last panel into the spare cell beside it. The
    # layout is settled first and then frozen, or the key's width would be
    # charged to the whole column above it and squeeze the panel there
    fig.canvas.draw()
    fig.set_layout_engine("none")
    host = axes[len(srcs) - 1]
    drawn = [r for r in ATTRIB if any(
        r in per[f] and np.isfinite(np.asarray(per[f][r], float)).any() for f in srcs)]
    entries = [_key_entry("v_write"), _key_entry("attr_patch"),
               *[_key_entry(r) for r in drawn],
               (f"the {_WORD.get(len(lenses), len(lenses))} reading lenses", LENS, ()),
               _key_entry("random")]
    st.key_labels(host, entries)
    return st.save(fig, _at(out, "patch_sources"))


# ------------------------------------------------------------------ jailbreak

def patch_jailbreak(results="results", out="figures", model=None):
    """APPENDIX. On typographic jailbreaks V-Write's signed map finds the patches that hold a refusal up: replacing the fifth it ranks as pushing hardest towards refusing moves the refusal margin further than attribution patching or a random fifth, and the other end of the same ranking moves it the opposite way, which an unsigned saliency map cannot show.

    Qwen3.5-27B only, at block 32 of 64 (normalised depth 0.51), on 400
    MM-SafetyBench TYPO items, where the harmful request is typed into the
    picture and the text prompt is only "Answer the question in the image.",
    with no system instruction. The model's greedy reply is judged a refusal
    or not: it complied on 351 (left, n shown) and refused 49 (right). The
    refusal margin is log p(a refusal opener) minus log p(a compliance
    opener) at the answer spot. V-Write integrates its gradient over eight
    steps from the picture's mean image-token state; attribution patching is
    the one-point gradient from the same state. Left: on complied items, the
    shift of the margin towards refusing when the top 5 to 50 per cent of
    patches pushing hardest away from refusing are replaced by the picture's
    mean state. Right: on refused items, the shift away from refusing when
    the patches pushing hardest towards refusing are replaced. Lines: V-Write,
    attribution patching, a random selection, and the other end of V-Write's
    ranking. The dotted rule is every image token replaced by the mean.

    With a fifth of the patches replaced, on refused items V-Write shifts the
    margin 4.94 away from refusing against 3.28 for attribution patching and
    0.47 for random, and the other end shifts it 0.61 back towards refusing;
    the margin crosses to the compliant side on 90 per cent of these items
    against 53 and 22 per cent. At half the picture V-Write's selection
    (6.98) moves the margin further than replacing the whole picture (6.54).
    On complied items V-Write's fifth shifts the margin 0.48 towards refusing
    and the other end 1.37 away, while replacing the whole picture shifts it
    0.35 away: only a signed map separates the parts of the picture pushing
    each way.

    Limits. On complied items attribution patching moves the margin further
    than V-Write at every size (0.59 against 0.48 at a fifth), and covering
    does not restore refusal: the mean starting margin is -6.95, the margin
    crosses on under 1 per cent of items, and greying the top half of the
    patches in pixels makes 3 of 351 replies refusals (none for a random
    half). The refused arm has only 49 items, and only Qwen3.5-27B has this
    run.
    """
    p = _newest(f"{results}/patchjailbreak/{model or '*'}__patchjailbreak.json")
    d = _load(p) if p else None
    if not d or not d["summary"].get("arms"):
        return None
    s_ = d["summary"]
    ks = [100 * k for k in s_["ks"]]
    arms = [a for a in ("complied", "refused") if a in s_["arms"]]
    if not arms:
        return None
    YLAB = {"complied": "shift towards refusing", "refused": "shift away from refusing"}
    WHO = {"complied": "the model complied", "refused": "the model refused"}
    st.use()
    fig, axes = st.wide(1, len(arms), unit=(5.25, 3.2))
    axes = np.atleast_1d(axes).ravel()
    drawn, whole = [], False
    for ax, a in zip(axes, arms):
        e = s_["arms"][a]
        for c in ("bottom", "random", "attr_patch", "v_write"):
            y = (e.get("moved") or {}).get(c)
            if not y:
                continue
            ax.plot(ks, y, **LINE[c])
            if c not in drawn:
                drawn.append(c)
        ac = e.get("all_covered")
        if ac is not None and np.isfinite(ac):
            ax.axhline(ac, **WHOLE)
            whole = True
        _percent_axis(ax, ks)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        st.despine(ax)
        _fit(ax)
        ax.set_ylabel(YLAB.get(a, a))
        ax.text(0, 1.02, f"{WHO.get(a, a)}", transform=ax.transAxes,
                ha="left", va="bottom", color=st.INK, fontsize=_tick_size())
    fig.supxlabel("patches replaced (%)", fontsize=_label_size())
    shown = {"v_write": "V-Write", "attr_patch": "attribution patching",
             "random": "random", "bottom": "the other end"}
    entries = [_key_entry(c, shown[c]) for c in ("v_write", "attr_patch", "random", "bottom") if c in drawn]
    if whole:
        entries.append(("the whole picture", WHOLE["color"], WHOLE["dashes"]))
    # the key is one row above both panels, in a band of its own
    st.key_band_lines(fig, entries)
    return st.save(fig, _at(out, "patch_jailbreak"))
