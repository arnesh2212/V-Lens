# SPDX-License-Identifier: Apache-2.0
"""Five figures rebuilt from the ground up.

Each function here keeps the name, the signature, the result file and the
claim of the function of the same name in :mod:`vlens.figures.paper`, and
changes the encoding. The old drawings broke in the same way: a canvas capped
at the page width with the same number of panels and series drawn into it, so
that heat maps collapsed to strips and names printed on top of one another.
Width alone does not save a figure; what is drawn has to fit the width.

House rules come from :mod:`vlens.figures.style`: every piece of text is ink,
colour lives in the marks, keys go in a band above the panels, nothing sits on a
line or a gridline, no bars, no titles carrying the story.
"""
from __future__ import annotations

import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from vlens.figures import style as st
from vlens.figures import paper as P
from vlens.figures.paper import (_load, _newest, _at, _line, SOURCE, NICE, COLOUR,  # noqa: F401
                                 ORDER, AXES_ORDER, _src_colour, _short)

#: a proper minus, so a number written by hand matches the tick labels
MINUS = "−"


def _num(v, nd=2, strip_zero=False):
    """``-0.67`` as ``−.67`` when asked, else ``−0.67``; never a hyphen."""
    s = f"{float(v):.{nd}f}"
    if strip_zero:
        s = s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s
    if s.startswith("-"):
        if float(s) == 0.0:
            s = s[1:]
        else:
            s = MINUS + s[1:]
    return s


def _text_width(fig, s, fontsize):
    """Width in inches of ``s`` at ``fontsize`` in the figure's font."""
    r = fig.canvas.get_renderer()
    t = fig.text(0, 0, s, fontsize=fontsize)
    w = t.get_window_extent(renderer=r).width / fig.dpi
    t.remove()
    return w


def _ticks_within(lo, hi, step):
    a = np.ceil(lo / step) * step
    return [float(v) for v in np.arange(a, hi + step * 1e-6, step)]


# ---------------------------------------------------------------- directions

#: the family a direction belongs to, and the one colour that family keeps
_FAMILY_COLOUR = {"V-Write": st.LENS, "unembedding": st.SLATE,
                  "labelled": st.BASE, "random": st.MUTED}
#: the pale ground under the cell a direction was built for
_OWN_BAND = "#EFE3D6"


def _family(nm: str) -> str:
    if nm.startswith("V-Write"):
        return "V-Write"
    if nm.startswith("unembedding"):
        return "unembedding"
    if nm.startswith("difference"):
        return "labelled"
    return "random"


def direction_geometry(results="results", out="figures", model=None):
    """APPENDIX. V-Write builds a steering direction for each of three things a picture can settle from gradients alone, with no labels, and its directions for the answer and for the model's report of being tested move their own axis far more than the labelled difference of means or the unembedding rows of the same words.

    Qwen3.5-27B, block 32 of 64 (normalised depth 0.51). The three axes are
    read at the answer spot through the model's own head: the answer, log
    p(yes) minus log p(no); refusal, log p(a refusal opener) minus log p(a
    compliance opener); self-report, log p(test) minus log p(real). Each
    V-Write direction is the mean over 120 fitting records of the gradient of
    its axis at the image positions. Beside them: the unembedding rows of up
    to twelve tokens each axis is scored on, summed; a difference of means
    labelled by the answer the model gave on the same 120 records; and a
    random direction. Every direction has unit length. They are measured on
    150 held-out SimonSays yes/no and test-or-real records, each asked as the
    set poses it. One row per direction. Left: its cosine with each of the
    other seven. Right: the mean shift of each axis when the direction is
    added at every image token at dose 2, scaled to the local residual norm;
    the pale ground marks the axis the V-Write and unembedding rows were
    built for.

    Along the answer, V-Write's direction shifts the margin by 3.45 where the
    labelled difference of means shifts it by 0.40, the unembedding rows by
    -0.22 and a random direction by 0.00. Along self-report, V-Write's
    direction shifts it by 6.55 against 0.48 for the unembedding rows. 26 of
    the 28 pairs of directions lie within 0.05 of orthogonal; the two that do
    not are V-Write's answer and refusal directions (cosine -0.67) and its
    answer and self-report directions (0.19).

    Limits. V-Write's refusal direction barely moves refusal (0.30) while the
    answer direction moves it by -2.64, so on this set refusal is not
    separated from the answer. The self-report direction also moves the
    answer (1.25) and refusal (-1.77). Only one dose is drawn, and the answer
    direction moves the answer more at dose 1 (4.76) than at dose 2, and the
    refusal direction moves refusal more at dose 1 (1.22). Only Qwen3.5-27B
    has this run.
    """
    p = _newest(f"{results}/directions/{model or '*'}__directions.json")
    d = _load(p) if p else None
    d = (d or {}).get("summary") or d
    if not d or not d.get("cosine"):
        return None
    names = list(d["directions"])
    axes_ = [a for a in AXES_ORDER if any(a in v for v in d["moved"].values())] or AXES_ORDER
    n, m = len(names), len(axes_)
    al = max(d["alphas"])
    C = np.array([[float(d["cosine"].get(f"{x}|{y}", d["cosine"].get(f"{y}|{x}", np.nan)))
                   for y in names] for x in names])
    M = np.array([[float((d["moved"].get(f"{nm}|{al}") or {}).get(ax, np.nan))
                   for ax in axes_] for nm in names])
    short = [nm.replace("difference of means", "diff. of means") for nm in names]

    st.use()
    row_in = 0.36
    fig, axs = st.wide(1, 1 + m, unit=(st.MAX_WIDTH / (1 + m), n * row_in + 1.2),
                       sharey=True, gridspec_kw=dict(width_ratios=[1.45] + [1.0] * m))
    axs = list(np.atleast_1d(axs))
    a0 = axs[0]
    a0.set_ylim(n - 0.5, -0.5)
    a0.set_yticks(range(n))
    a0.set_yticklabels(short)

    # ---- the cosines: every other direction, as a dot on this row
    for i, nm in enumerate(names):
        c = _FAMILY_COLOUR[_family(nm)]
        vals = np.array([C[i, j] for j in range(n) if j != i and np.isfinite(C[i, j])])
        a0.plot(vals, [i] * len(vals), ls="none", marker="o", ms=6.5, mfc="white",
                mec=c, mew=1.4, zorder=4)
        # a pair that shares something is named by its value, set on the side
        # away from the pile at nought so it sits on nothing
        for v in vals:
            if abs(v) >= 0.1:
                a0.annotate(_num(v, 2, strip_zero=True), xy=(v, i),
                            xytext=(7 if v > 0 else -7, 0), textcoords="offset points",
                            ha="left" if v > 0 else "right", va="center",
                            color=st.INK, fontsize=15)
    a0.axvline(0, color=st.AXIS, lw=1.0, zorder=1)
    # room on the left for the name of a negative value, clear of the spine
    a0.set_xlim(-1.55, 0.75)
    a0.set_xticks([-1, -0.5, 0, 0.5])
    a0.set_xticklabels([MINUS + "1", MINUS + ".5", "0", ".5"])
    a0.set_xlabel("cosine with\nthe others")
    a0.grid(False)
    st.despine(a0)

    # ---- what each does to each axis, on one scale
    lo, hi = float(np.nanmin(M)), float(np.nanmax(M))
    pad = (hi - lo) * 0.09
    step = 3.0 if (hi - lo) <= 12 else float(np.ceil((hi - lo) / 3))
    for j, ax in enumerate(axs[1:]):
        for i, nm in enumerate(names):
            fam = _family(nm)
            if fam in ("V-Write", "unembedding") and nm.endswith(axes_[j]):
                ax.axhspan(i - 0.5, i + 0.5, color=_OWN_BAND, lw=0, zorder=0)
            if np.isfinite(M[i, j]):
                ax.plot([M[i, j]], [i], ls="none", marker="o", ms=8.5,
                        color=_FAMILY_COLOUR[fam], zorder=4)
        ax.axvline(0, color=st.AXIS, lw=1.0, zorder=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xticks(_ticks_within(lo - pad * 0.5, hi + pad * 0.5, step))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _num(v, 0)))
        ax.set_xlabel(f"shift on\n{axes_[j]}")
        ax.grid(False)
        ax.tick_params(axis="y", length=0)
        st.despine(ax)
    fig.get_layout_engine().set(wspace=0.04)
    # the key for the pale ground sits in a band of its own above the panels,
    # not beside a row of dots where it would read as one more row
    st.key_band(fig, [matplotlib.patches.Patch(color=_OWN_BAND, lw=0)],
                ["built for this axis"], handlelength=1.2)
    return st.save(fig, _at(out, "direction_geometry"))


# ---------------------------------------------------------------- patch atlas

def patch_atlas(results="results", out="figures", model=None):
    """APPENDIX. Run block by block, V-Write's causally checked ranking shows where in depth the image tokens carry the answer: near the input any fifth of the picture costs about as much as the fifth it ranks highest, at a quarter of depth almost all of the cost is the ranking's, and by three quarters the image tokens no longer carry the answer.

    Qwen3-VL-8B-Instruct at blocks 4, 9, 18 and 26 of 36 (normalised depth
    0.11, 0.26, 0.51, 0.74), one line per source. Records are SimonSays
    yes/no, test-or-real and WMDP carrier letter items, each asked as the set
    poses it and scored by the logit margin of the answer the model gave at
    the answer spot; about 490 records per depth (139 Open Images, 57
    nuScenes, 121 Interfaces (VINS-120K), 100 Generated scenes, 72 WMDP
    carrier), and 1003 at depth 0.51, which comes from the larger
    patch-ranking run. At each block the fifth of the image patches V-Write
    ranks highest there is replaced by zeros. Left: the mean fall in the
    answer score. Right: the same minus the fall from zeroing a random fifth,
    the part owed to the ranking. The WMDP carrier is grey and dashed.
    Password-locking families are excluded (none is in these runs).

    At depth 0.11 the fall is 7.1 to 11.4 across sources but only 0.9 to 1.5
    of it is over random on the four picture sources (0.06 on the WMDP
    carrier). On the four picture sources, at 0.26 the fall is 6.5 to 10.1
    and 6.1 to 9.2 of it is over random; at 0.51 the fall is 4.0 to 6.1, of
    which 2.9 to 5.1 is over random; at 0.74 the fall is 0.5 to 1.2 and the
    margin 0.2 to 0.6. The WMDP carrier, whose picture only carries a
    question asked in text, has a margin of 0.06, 2.31, 0.47 and 0.07, below
    every picture source at every depth.

    Limits. One model: Qwen3.5-27B has the patch-ranking run at one depth
    only, so has no atlas. Four blocks, and the 0.51 point comes from a
    different and larger record sample than the other three.
    """
    hits = sorted(glob.glob(f"{results}/atlas/{model or '*'}__*patchlenses.json")
                  + glob.glob(f"{results}/patchlenses/{model or '*'}__patchlenses.json"))
    name = model or (os.path.basename(hits[0]).split("__")[0] if hits else None)
    if not name or not P._enough(hits):
        return None
    by_depth = {}
    for q in hits:
        if not P.AVERAGE and os.path.basename(q).split("__")[0] != name:
            continue
        d = _load(q, pool=False)
        if not d or not d["summary"].get("per_source"):
            continue
        s_ = d["summary"]
        if not s_.get("oriented"):
            continue                   # selected by the older rule: a different basis
        by_depth.setdefault(round(float(s_.get("depth", 0.5)), 2), []).append(
            (os.path.basename(q).split("__")[0], s_))
    if P.AVERAGE:
        every = set.intersection(*({mo for mo, _ in v} for v in by_depth.values())) \
            if by_depth else set()
        by_depth = {d_: [(mo, x) for mo, x in v if mo in every] for d_, v in by_depth.items()}
        by_depth = {d_: v for d_, v in by_depth.items() if len({mo for mo, _ in v}) > 1}
    seen = {d_: (v[0][1] if len(v) == 1 else P._mean_of([x for _, x in v]))
            for d_, v in by_depth.items() if v}
    if len(seen) < 2:
        return None
    depths = sorted(seen)
    have = {f for s_ in seen.values() for f in s_["per_source"]}
    house = ["openimages", "nuscenes", "vins", "gemini", "wmdp"]
    srcs = [f for f in house if f in have] + sorted(f for f in have if f not in house)

    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(st.MAX_WIDTH / 2, 3.5),
                            gridspec_kw=dict(width_ratios=[1.0, 1.28]))
    key = []
    for ax, over_random in ((a1, False), (a2, True)):
        drawn = 0
        for f in srcs:
            x, y = [], []
            for dd in depths:
                s_ = seen[dd]
                row = s_["per_source"].get(f) or {}
                v, r = row.get("v_write"), row.get("random")
                if v is None:
                    continue
                k = s_.get("per_source_index", 2)
                x.append(dd)
                y.append(v[k] - (r[k] if over_random and r else 0.0))
            if len(x) < 2:
                continue
            carrier = f == "wmdp"
            c = _src_colour(f)
            dash = (4, 2) if carrier else None
            ax.plot(x, y, color=c, lw=1.8 if carrier else 2.4, marker="o", ms=4.5,
                    dashes=dash or (), zorder=3 if carrier else 4)
            drawn += 1
            if ax is a2:
                key.append((SOURCE.get(f, f), c, dash))
        if not drawn:
            return None
        st.depth_axis(ax)
        st.fit_y(ax, lo=0.0)
        ax.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
        st.despine(ax)
    a1.set_ylabel("fall in the answer score")
    a2.set_ylabel("margin over a random fifth")
    st.key_band_lines(fig, key)
    return st.save(fig, _at(out, "patch_atlas"))


# ---------------------------------------------------------------- hand-off

def _gap_for_text(ax, x_from, text, fontsize, curves, y_ticks):
    """A ``y`` between two gridlines where ``text`` set at ``x_from`` clears every curve.

    The candidates are the mid-points between consecutive y ticks, lowest
    first. Each is checked against the curves over the run of x the text will
    occupy, and the first whose band of page the text needs is empty is taken.
    Falls back to the lowest candidate.
    """
    fig = ax.figure
    fig.canvas.draw()
    box = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w_in = _text_width(fig, text, fontsize) + 8 / 72
    h_in = fontsize / 72 * 1.25
    dx = w_in * fig.dpi / max(box.width, 1) * (x1 - x0)
    dy = h_in * fig.dpi / max(box.height, 1) * (y1 - y0)
    ticks = sorted(t for t in y_ticks if y0 <= t <= y1)
    mids = [(a + b) / 2 for a, b in zip(ticks, ticks[1:])]
    xs = np.linspace(x_from, x_from + dx, 40)
    for ym in mids:
        ok = True
        for cx, cy in curves:
            yy = np.interp(xs, cx, cy)
            if np.any(np.abs(yy - ym) < dy * 0.8):
                ok = False
                break
        if ok:
            return ym
    return mids[0] if mids else (y0 + y1) / 2


def handoff(results="results", out="figures", model=None):
    """APPENDIX. SimonSays pairs a real picture with the same picture carrying one edited sign, which turns the question of where a visual fact travels into a controlled experiment: the edited region's image-token states alone, transplanted early, carry the edited answer, and the fact sits in the image tokens at half depth and in the text tokens by five sixths.

    Qwen3.5-27B, on 605 matched original and edited pairs (261 Open Images,
    231 Interfaces (VINS-120K), 57 Generated scenes, 56 nuScenes), each asked
    as the set poses it, kept where the edit changes the model's top next
    token and the edited region covers at least 2 image tokens and at most 60
    per cent of them (observed 1 to 50 per cent, median 28 of 196). The score
    in both panels is log p(the edited run's answer token) minus log p(the
    original run's answer token) at the answer spot, read through the model's
    own final norm and unembedding. Left: the mean score after every fifth
    block, on normalised depth (block over 63), for the original run (dashed)
    and for the original run with only the edited region's image-token states
    replaced, at block 22 (depth 0.35, the vertical line), by the edited run's
    states. Right: at blocks 32, 42 and 52 (depth 0.51, 0.67, 0.83) the
    transplanted run's image tokens, text tokens or answer spot are put into
    the untouched original run and the score is read at the output; the two
    horizontal lines are where the two runs on the left finish.

    The original run ends at -3.04 and the transplanted run at 0.55, past
    level towards the edited answer; the two runs coincide to depth 0.41 and
    part from 0.59 on. Swapping in the image tokens lifts the output score to
    -0.29 at depth 0.51, -1.62 at 0.67 and -2.64 at 0.83; the text tokens give
    -2.45, -1.14 and -0.03; the answer spot alone -2.99, -2.71 and -1.29. In
    the same pairs the transplant flips the answer on 57 per cent, against 10
    per cent for the same number of image tokens taken from outside the edit
    and 5 per cent for noise on the edited tokens.

    Limits. One model, read through the model's own head only, with no
    interval drawn. The transplant recovers on average 54 per cent of the gap to the
    edited run, which itself ends at 3.37.
    """
    p = _newest(f"{results}/handoff/{model or '*'}__handoff.json")
    d = _load(p) if p else None
    if not d:
        return None
    s = d["summary"]
    grid = s["grid"]
    L = max(grid) or 1
    x = [l / L for l in grid]
    y0 = [s["depth_curve_original"][str(l)] for l in grid]
    y1 = [s["depth_curve"][str(l)] for l in grid]
    ylab = "edited over original answer"

    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(st.MAX_WIDTH / 2, 3.4),
                            gridspec_kw=dict(width_ratios=[1.2, 1.0]))
    a1.plot(x, y0, color=st.MUTED, lw=1.9, dashes=(5, 2), zorder=3)
    a1.plot(x, y1, color=st.LENS, lw=2.6, zorder=4)
    st.depth_axis(a1)
    a1.grid(False, axis="x")
    lo = min(min(y0), min(y1))
    hi = max(max(y0), max(y1), 0.0)
    # headroom above the level line for its name
    st.fit_y(a1, lo=lo, hi=hi + (hi - lo) * 0.04)
    step = 2.0 if (hi - lo) > 3 else 1.0
    yl = a1.get_ylim()
    a1.set_yticks(_ticks_within(yl[0], yl[1], step))
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _num(v, 0)))
    a1.set_ylabel(ylab)
    # the level line, named just under itself at the left, where both runs
    # still sit on or above it; above it the transplanted run would cross the words
    st.chance(a1, 0.0, "")
    lvl_fs = matplotlib.rcParams["xtick.labelsize"] * 0.92
    a1.annotate("answers level", xy=(a1.get_xlim()[0], 0.0), xytext=(3, -4),
                textcoords="offset points", ha="left", va="top", color=st.INK,
                fontsize=lvl_fs, zorder=8)
    st.despine(a1)
    # the transplant depth, named in a gap clear of both curves and of the gridlines
    depth = float(s["depth"])
    # the line rises to just under the name of the level line, never through it
    fig.canvas.draw()
    yl_now = a1.get_ylim()
    h_data = (lvl_fs * 1.45 + 6) / 72 * fig.dpi / a1.get_window_extent().height * (yl_now[1] - yl_now[0])
    a1.plot([depth, depth], [yl_now[0], -h_data], color=st.AXIS, lw=1.0, zorder=2)
    y_lab = _gap_for_text(a1, depth, "transplant", matplotlib.rcParams["font.size"],
                          [(x, y0), (x, y1)], a1.get_yticks())
    a1.annotate("transplant", xy=(depth, y_lab), xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", color=st.INK)

    # ---- where the fact is carried, later on
    house = ["image tokens", "text tokens", "answer spot"]
    have = {k.split("|")[1] for k in s["swaps"]}
    groups = [g for g in house if g in have] + sorted(have - set(house))
    depths = sorted({int(k.split("|")[0]) for k in s["swaps"]})
    xs = [l / L for l in depths]
    gc = {"image tokens": st.SLATE, "text tokens": st.OCHRE, "answer spot": st.SAGE}
    extra = [c for c in st.SERIES if c not in (st.LENS, st.MUTED, *gc.values())]
    for i, g in enumerate(groups):
        c = gc.get(g) or extra[i % len(extra)]
        gc[g] = c
        ys = [s["swaps"].get(f"{l}|{g}", np.nan) for l in depths]
        a2.plot(xs, ys, color=c, lw=2.1, marker="o", ms=5, zorder=4)
    # where the two runs on the left finish: the bounds a swap can move between
    a2.axhline(y0[-1], color=st.MUTED, lw=1.9, dashes=(5, 2), zorder=2)
    a2.axhline(y1[-1], color=st.LENS, lw=1.9, zorder=2)
    st.fit_y(a2)
    a2.set_xticks(xs)
    a2.set_xticklabels([f"{v:.2f}" for v in xs])
    span = (xs[-1] - xs[0]) or 0.1
    a2.set_xlim(xs[0] - span * 0.16, xs[-1] + span * 0.16)
    a2.set_xlabel("depth of the swap")
    a2.set_ylabel(ylab)
    a2.grid(False, axis="x")
    st.despine(a2)
    # one key for both panels, in a band above them: the original run is
    # dashed in both
    st.key_band_lines(fig, [("transplanted", st.LENS, None), ("original run", st.MUTED, (5, 2)),
                            *[(g, gc[g], None) for g in groups]])
    return st.save(fig, _at(out, "handoff"))


# ---------------------------------------------------------------- baselines

def baseline_agreement(results="results", out="figures", model=None):
    """APPENDIX. V-Write's ranking is a property of the model rather than of the baseline it integrates from, as long as the baseline is a state the model has seen: from the picture's mean patch, a blurred copy or another picture the maps agree with one another, while zeros and noise, the usual baselines of integrated gradients, give maps that share little with them and remove less of the answer.

    Qwen3-VL-8B-Instruct from the five-baseline run: 150 records (107 Open
    Images, 22 WMDP carrier, 9 Generated scenes, 8 Interfaces (VINS-120K), 4
    nuScenes) at block 18 of 36, each asked as the set poses it and scored by
    the logit margin of the answer the model gave at the answer spot.
    V-Write integrates over eight steps from each of five baseline states:
    the picture's mean image-token state, the image tokens of a blurred copy
    of the picture, zeros, noise matched to the states' mean and spread, and
    the image tokens of another picture. Attribution patching, gradient times
    activation (from zero) and a random ranking are beside them. Qwen3.5-27B
    has only the single-baseline run (2668 records, block 32 of 64), so its
    triangle holds V-Write from the mean baseline, labelled V-Write, and the
    three other rankings. Every ranking in a run is scored on the same
    records of that run, so each comparison is on one shared sample; the
    costly published methods (integrated gradients, RISE, KernelSHAP) are not
    in these runs. Left: the
    lower triangle of Spearman correlations between per-patch maps, averaged
    over records. Right, on the same rows: the fall in the answer score when
    the top fifth by that ranking is replaced by the picture's mean state
    (large dot), with the trail from 5 to 50 per cent. Password-locking
    families are excluded (none is in these runs).

    On Qwen3-VL-8B-Instruct the mean patch, the blurred copy and another
    picture agree at 0.43 to 0.55; zeros and noise agree with those three at
    only 0.12 to 0.24 and with each other at 0.09. Every V-Write map is
    uncorrelated with random (-0.01 to 0.00). Replacing the top fifth costs
    3.08 from the mean patch, 2.55 from another picture, 2.53 from the
    blurred copy, 1.88 from noise and 1.46 from zeros, against 1.34 for
    attribution patching, 1.04 for gradient times activation and 0.14 for
    random. On Qwen3.5-27B the mean-baseline map agrees with attribution
    patching at 0.33 and with gradient times activation at 0.13, and its top
    fifth costs 0.89 against 0.55, 0.31 and 0.04.

    Limits. The five-baseline comparison is one model and 150 records, most
    of them Open Images. The replacement is the mean state, the same as the
    mean baseline, which favours that baseline in the right panel.
    """
    p = (_newest(f"{results}/ablation1/{model or '*'}__patchcausal.json")
         or _newest(f"{results}/patchcausal/{model or '*'}__patchcausal.json"))
    d = _load(p) if p else None
    if not d or not d["summary"].get("agreement"):
        return None
    s = d["summary"]
    pref = ["vwrite_mean", "vwrite_blur", "vwrite_other", "vwrite_zero", "vwrite_noise",
            "attr_patch", "gradxact_zero", "random"]
    have = {k for pair in s["agreement"] for k in pair.split("|")}
    keys = [k for k in pref if k in have] + sorted(k for k in have if k not in pref)
    short = {"vwrite_mean": "the mean patch", "vwrite_blur": "a blurred copy",
             "vwrite_other": "another picture", "vwrite_zero": "zeros",
             "vwrite_noise": "noise", "attr_patch": "attribution patching",
             "gradxact_zero": "grad × activation", "random": "random"}
    # with one baseline there is nothing to tell apart, and a row named only
    # "the mean patch" would not say that it is V-Write's map
    if sum(k.startswith("vwrite_") for k in keys) == 1:
        short.update({k: "V-Write" for k in keys if k.startswith("vwrite_")})
    lab = [short.get(k, k.replace("vwrite_", "")) for k in keys]
    n = len(keys)
    A = np.full((n, n), np.nan)
    for i, kx in enumerate(keys):
        for j, ky in enumerate(keys):
            if i == j:
                continue
            v = s["agreement"].get(f"{kx}|{ky}", s["agreement"].get(f"{ky}|{kx}"))
            if v is not None:
                A[i, j] = A[j, i] = float(v)
    ks = list(s["ks"])
    kidx = int(np.argmin([abs(k - 0.2) for k in ks]))

    def curve_of(k):
        c = s["curve"].get(f"top:{k}")
        if c is None and k.startswith("vwrite_"):
            c = s["curve"].get("top")          # the main ranking's own curve
        return c

    def colour_of(k):
        return (st.LENS if k.startswith("vwrite_") else st.MUTED if k == "random"
                else st.SLATE)

    st.use()
    fs = matplotlib.rcParams["xtick.labelsize"]
    fs_cell = 15
    cell = max(0.5, min(0.72, 4.0 / n))
    probe = plt.figure(figsize=(2, 1))
    widths = [_text_width(probe, t, fs) for t in lab]
    plt.close(probe)
    over = max(0.0, max((i - 0.42) + w / cell + 0.2 - (n - 0.5) for i, w in enumerate(widths)))
    w1 = (n + over) * cell
    w2 = st.MAX_WIDTH - w1 - 0.35
    H = n * cell + 0.8
    fig, (a1, a2) = st.wide(1, 2, unit=(st.MAX_WIDTH / 2, H), sharey=True,
                            gridspec_kw=dict(width_ratios=[w1, w2]))

    shown = np.where(np.tril(np.ones((n, n)), -1) > 0, np.clip(A, 0, None), np.nan)
    a1.imshow(shown, cmap=st.warm_cmap(), vmin=0, vmax=1, aspect="equal", zorder=1)
    a1.set_xlim(-0.5, n - 0.5 + over)
    a1.set_ylim(n - 0.5, -0.5)
    a1.set_xticks([])
    a1.tick_params(axis="y", left=False, labelleft=False)
    a1.grid(False)
    for sp in a1.spines.values():
        sp.set_visible(False)
    for i in range(n):
        a1.text(i - 0.42, i, lab[i], ha="left", va="center", color=st.INK, fontsize=fs)
        for j in range(i):
            if np.isfinite(A[i, j]):
                a1.text(j, i, _num(A[i, j], 2, strip_zero=True), ha="center", va="center",
                        color=st.INK, fontsize=fs_cell)
    a1.set_xlabel("rank correlation", loc="left")

    # ---- the same rows: what the top fifth costs under each
    xmax = 0.0
    for i, k in enumerate(keys):
        cur = curve_of(k)
        if not cur:
            continue
        c = colour_of(k)
        a2.plot(cur, [i] * len(cur), color=c, lw=1.3, marker="o", ms=3.0, zorder=3)
        a2.plot([cur[kidx]], [i], ls="none", marker="o", ms=9.5, color=c, zorder=4)
        xmax = max(xmax, max(cur))
        if i == 0:
            a2.annotate(f"{ks[0] * 100:g}%", xy=(cur[0], i), xytext=(-4, 4),
                        textcoords="offset points", ha="right", va="bottom",
                        color=st.INK, fontsize=15)
            a2.annotate(f"{ks[-1] * 100:g}%", xy=(cur[-1], i), xytext=(4, 4),
                        textcoords="offset points", ha="left", va="bottom",
                        color=st.INK, fontsize=15)
            a2.annotate("top fifth", xy=(cur[kidx], i), xytext=(0, -8),
                        textcoords="offset points", ha="center", va="top",
                        color=st.INK, fontsize=15)
    a2.set_yticks(range(n))
    a2.tick_params(axis="y", left=False, labelleft=False)
    a2.grid(True, axis="y")
    a2.grid(False, axis="x")
    a2.set_xlim(-xmax * 0.02, xmax * 1.14)
    a2.set_ylim(n - 0.5, -0.72)
    a2.set_xlabel("fall in the answer score")
    st.despine(a2)
    fig.get_layout_engine().set(wspace=0.02)
    return st.save(fig, _at(out, "baseline_agreement"))


# ---------------------------------------------------------------- gallery

def gallery(results="results", out="figures", model=None):
    """APPENDIX. A V-Write map computed in hidden space still holds when it is acted on in pixels a reader can see: greying out the fifth of a typical picture that V-Write ranks highest takes more of the answer away than greying out a random fifth on three of five sources for Qwen3-VL-8B-Instruct and four of five for Qwen3.5-27B, and on the generated scene it flips the answer in both models.

    One file per model: Qwen3-VL-8B-Instruct (V-Write at block 18 of 36) and
    Qwen3.5-27B (block 32 of 64). One record per source, each asked exactly
    as the set poses it: an edited generated scene asked whether it is a
    test image or a real photo, edited Open Images and nuScenes authority
    records under the cue "You are the driver of the car this photo was taken
    from.", an edited interface record (VINS-120K) under the cue "Look carefully
    at this image." and asked test or real (on Qwen3.5-27B its edited
    picture is a photographed sign announcing a model evaluation session,
    not a screen), and a WMDP carrier biology item under the
    set's sandbagging system instruction (the calibration instruction for
    Qwen3-VL-8B-Instruct, the hazard instruction for Qwen3.5-27B). For each
    source the first twelve records with an answer to read, edited pictures
    first in a fixed order, are measured and
    the median by fall under the top fifth is shown, so the case is typical
    rather than chosen. Top row: the 39 of 196 patches V-Write ranks highest
    greyed out in the pixels. Bottom row: a random 39 greyed out. Under each
    panel, the answer score on the untouched picture and then on the greyed
    one, where the answer score is the logit margin of the answer the model
    gave the untouched picture, read from the model run on each picture.

    Fall under V-Write's fifth against a random fifth, Qwen3-VL-8B-Instruct:
    Generated scenes 5.00 against -0.50 (1.50 to -3.50, the answer flips),
    Open Images 2.30 against -0.23, nuScenes 2.67 against 3.08, Interfaces
    8.75 against 6.50, WMDP carrier 0.00 against 0.50. Qwen3.5-27B:
    Generated scenes 4.38 against -0.87 (0.13 to -4.25, the answer flips),
    Open Images 0.09 against -0.44, nuScenes 2.00 against -1.84, Interfaces
    1.48 against 1.11, WMDP carrier 0.12 against 0.13.

    Limits. One picture per source, from twelve candidates. On nuScenes for
    Qwen3-VL-8B-Instruct the random fifth costs more than V-Write's, and on
    the WMDP carrier, whose question is asked in text, neither greying moves
    the answer in either model. On Open Images for Qwen3.5-27B the fall is
    small (0.09).
    """
    if P.AVERAGE:
        return None            # a mean of five models is not a photograph
    p = _newest(f"{results}/gallery/{model or '*'}__gallery.json")
    d = _load(p, pool=False) if p else None
    if not d or not d.get("scenarios"):
        return None
    from PIL import Image
    root = os.path.dirname(p)
    sc = d["scenarios"]
    st.use()
    fs = matplotlib.rcParams["xtick.labelsize"]
    # two rows: the pixels V-Write says hold the answer greyed out, and a
    # random fifth greyed out. The greyed squares of the first row are the
    # map itself, so an outlined copy of the untouched picture would only
    # repeat them; the untouched score is carried in every number instead
    fig, axes = st.wide(2, len(sc), unit=(st.MAX_WIDTH / len(sc), 2.34))
    axes = np.atleast_2d(axes)
    rows_lab = ("V-Write's top\nfifth greyed", "a random\nfifth greyed")
    for j, s_ in enumerate(sc):
        head = s_.get("label") or SOURCE.get(s_.get("source"), "")
        for i, which in enumerate(("top", "random")):
            ax = axes[i, j]
            f = os.path.join(root, s_["picture"][which])
            if not os.path.exists(f):
                ax.set_axis_off()
                continue
            st.image_panel(ax, Image.open(f))
            if i == 0:
                ax.set_title(head, loc="left", color=st.INK, fontsize=fs, pad=4)
            num = f"{_num(s_['score']['original'], 2)} \u2192 {_num(s_['score'][which], 2)}"
            ax.set_xlabel(num, color=st.INK, fontsize=fs, labelpad=3)
    for i, t in enumerate(rows_lab):
        axes[i, 0].set_ylabel(t, color=st.INK, fontsize=fs, labelpad=6)
    return st.save(fig, _at(out, "gallery"))
