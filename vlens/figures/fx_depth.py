# SPDX-License-Identifier: Apache-2.0
"""Five figures over depth and dose, redrawn to the house rules.

Each of these had the same faults in its first drawing: a rotated y label
longer than the axes were tall, which left most of the canvas blank; a stack
of names at the right end with no colour beside them, some pointing at the
wrong line; leaders from those names drawn outside the frame; and the word
"chance" set down on top of the very lines it was meant to sit beside.

The fixes are the same in each case. Axis labels are short enough to fit
inside the axes height. Every key is drawn by :func:`style.key_band`, a
stroke of the line's own colour followed by its name in ink, in one row in a
band above the panels. A reference
line is named only where a scan of everything drawn in the panel finds room
for the word, and is left unnamed otherwise. Panel identifiers are short and
set at tick size, so neighbours cannot run into each other. Every number is
read from the result file the original figure read.

The signatures match ``paper.py`` so the two can be swapped in the build.
"""
from __future__ import annotations

import matplotlib
import matplotlib.ticker as mticker
import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st
from vlens.figures.paper import (COLOUR, DASH, NICE, ORDER, PROBE, SUBJECT, _at, _line,
                                 _load, _newest, _spread, _src_colour)

#: the sources named as briefly as a panel identifier can afford
SHORT_SRC = {"openimages": "Open Images", "nuscenes": "nuScenes", "vins": "VINS-120K",
             "gemini": "Generated scenes", "wmdp": "WMDP"}

#: one marker per instrument, so two lines that end on the same value can
#: still be told apart where the colours alone cannot
MARK = {"v_read_cross": "v", "v_read": "o", "tuned": "s", "jacobian": "^",
        "logit": "D", "probe": "P", "latent": "x"}


def _ident_size() -> float:
    """The size a panel identifier is set at: tick size, not title size."""
    return matplotlib.rcParams["xtick.labelsize"]


def _fam_two_lines(fam: str, n) -> str:
    """``openimages_orig_auth`` as two short lines: source and variant, then probe and n."""
    bits = str(fam).split("_")
    src = SHORT_SRC.get(bits[0], bits[0])
    rest = []
    for b in bits[1:]:
        if b == "orig":
            rest.append("as taken")
        elif b == "edit":
            rest.append("edited")
        elif b in SUBJECT:
            rest.append(SUBJECT[b])
        elif b in PROBE:
            rest.append(PROBE[b])
        else:
            rest.append(b)
    first = src + (f", {rest[0]}" if rest else "")
    # the records scored are in the caption, not on the panel
    second = ", ".join(rest[1:])
    return f"{first}\n{second}"


def _free_x(ax, y, series, text="chance", fontsize=None):
    """Where ``text``, sitting just above the line at ``y``, is clear of everything drawn.

    ``series`` is a list of ``(x, y)`` arrays: every line, ribbon edge and
    mean in the panel. The panel is scanned for a window the width of the word
    in which no drawn value falls inside the word's box, and the window with
    the most clearance wins, the rightmost among equals. ``None`` means there
    is no such window, and the caller leaves the word off rather than set it
    down on a line.
    """
    fig = ax.figure
    fig.canvas.draw()
    fs = fontsize or matplotlib.rcParams["xtick.labelsize"] * 0.92
    bb = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    px = fig.dpi / 72.0
    w = (len(text) * 0.5 + 0.5) * fs * px * (x1 - x0) / max(bb.width, 1)
    h = (fs * 1.3 + 3) * px * (y1 - y0) / max(bb.height, 1)
    grid = np.linspace(x0, x1, 400)
    rows = []
    for sx, sy in series:
        sx, sy = np.asarray(sx, float), np.asarray(sy, float)
        m = np.isfinite(sx) & np.isfinite(sy)
        if m.sum() < 2:
            continue
        rows.append(np.interp(grid, sx[m], sy[m], left=np.nan, right=np.nan))
    if not rows:
        return x0 + (x1 - x0) * 0.02
    Y = np.vstack(rows)
    lo_band, hi_band = y - 0.3 * h, y + h
    best, best_x = -np.inf, None
    for xs in np.linspace(x0, x1 - 1.35 * w, 80):
        m = (grid >= xs) & (grid <= xs + w)
        v = Y[:, m]
        v = v[np.isfinite(v)]
        if v.size == 0:
            score = np.inf
        else:
            inside = (v > lo_band) & (v < hi_band)
            if inside.any():
                score = -float(inside.mean())
            else:
                score = float(np.minimum(np.abs(v - lo_band), np.abs(v - hi_band)).min())
        if score >= best - 1e-12:
            best, best_x = score, xs
    return best_x if best >= 0 else None


def _chance(ax, y, series, text="chance"):
    """The reference line, named only where the name lands on nothing."""
    x = _free_x(ax, y, series, text)
    st.chance(ax, y, text if x is not None else "", x=x)
    return x is not None


def _ident(ax, text):
    ax.set_title(text, loc="left", color=st.INK, fontsize=_ident_size(), pad=6)


def _key_band(axes, pairs):
    """The key as one row in a band of its own above every panel.

    ``pairs`` is ``[(text, colour, dashes), ...]``. A key set inside a panel,
    even in an empty corner, reads as one more thing that was measured.
    """
    from matplotlib.lines import Line2D
    if not pairs:
        return None
    hs = [Line2D([], [], color=c, lw=2.6, dashes=d or (None, None)) for _t, c, d in pairs]
    return st.key_band(axes[-1].figure, hs, [t for t, _c, _d in pairs])


# ------------------------------------------------------------------ figure 1

def lens_by_depth(results="results", out="figures", model=None, families=None,
                  target="model", max_panels=3):
    """MAIN. On photographs, V-Read reads the answer a model is about to give
    from about half depth, where the best baseline still reads it upside down,
    on both models; elsewhere it is level with the published lenses or a block
    behind them. It is a
    label-free affine map per block, solved in closed form, with no gradient
    steps and no labels.

    Per model (Qwen3-VL-8B-Instruct, Qwen3.5-27B), three SimonSays families,
    the largest of three sources: Open Images as taken with the authority
    questions (n=202 on the 8B, 206 on the 27B), the WMDP carrier in biology
    (n=75 on each) and Interfaces (VINS-120K) edited with the authority
    questions (n=73, 69). The image families are asked with no system
    instruction; every WMDP record carries one of the set's three system
    instructions (plain, calibration, hazard; 25 each), pooled in one panel.
    At 16 blocks spread over depth (3 to 34 of 36 on the 8B, 0 to 62 of 64 on
    the 27B), the state at the answer spot is carried to the last block by each
    instrument and read through the model's own final norm and unembedding:
    V-Read (h to Ah + b, ridge), and the published lenses, the tuned lens and
    the Jacobian lens (fitted on the 8B only) and the logit lens (the
    identity), which together are the baselines. The score is log p(yes) minus
    log p(no), or the correct letter's log probability minus the best other
    letter's. Up is the AUROC with which that score predicts the answer the
    model gives, taken as the sign of the same score at the last block, so a
    sandbagged answer is the answer to read. Depth is block over the last
    block. The band is a 300-draw bootstrap interval over records. Every
    instrument is fitted once on 900 SimonSays prompts held out of the
    behaviour pass and frozen; the records used in any fit (1,492 on the 8B,
    1,140 on the 27B) are kept out of every scored set by identifier. The
    averaged figure is the mean of the two models' curves block for block (the
    16 blocks sit at nearly the same depths) over V-Read and the logit lens,
    the instruments both models have, with n summed. The password-locking
    families are not scored.

    Qwen3-VL-8B-Instruct: on Open Images V-Read is at 0.73 at depth 0.49 where
    the tuned, Jacobian and logit lenses sit at 0.08 to 0.10, the same
    information with its sign inverted, and at depth 0.57 it reads 1.00
    against 0.86 for the tuned lens and 0.81 for the logit lens. On the edited
    interfaces at depth 0.57 it is at 0.90 (interval 0.79 to 0.98) against
    0.76, 0.83 and 0.74. On the WMDP carrier at depth 0.63 it is at 0.68
    against 0.40 to 0.50, and from depth 0.69 every instrument is above 0.96.
    Qwen3.5-27B: on Open Images V-Read is at 0.88 at depth 0.40 and 0.90
    (interval 0.85 to 0.94) at depth 0.52 where the logit lens is at 0.49 and
    0.16; both are at 0.98 or more from depth 0.59. On the edited interfaces
    the logit lens is ahead at depth 0.52 (0.88 against 0.76) and both pass
    0.95 by depth 0.65; on the WMDP carrier both stay between 0.48 and 0.72
    until depth 0.79 and reach 0.94 or more from depth 0.86. Averaged, on Open
    Images at depth 0.55 V-Read is at 0.95 and the logit lens at 0.49.

    Limits: 69 to 206 records a panel, so neighbouring intervals overlap at
    most depths. The tuned and Jacobian lenses have not been fitted on the
    27B, so there the only baseline is the logit lens. The Jacobian lens
    matches V-Read on the 8B's Open Images from depth 0.57 on. Before about
    depth 0.4 no instrument, V-Read included, reads the answer reliably, and
    V-Read too falls far below a half on Open Images (0.06 at depth 0.37 on
    the 8B, 0.15 at depth 0.33 on the 27B). No labelled probe was fitted in
    this run, so no labelled probe is drawn. LatentLens was fitted on the 8B and is
    left out because its readout is still being recalibrated.
    """
    p = _newest(f"{results}/lensbench*/{model or '*'}__bench.json")
    d = _load(p) if p else None
    if not d or not d.get("families"):
        return None
    fams = families or _spread(d["families"], lambda f: d["families"][f]["n"], max_panels)
    fams = [f for f in fams if f in d["families"]]
    if not fams:
        return None
    st.use()
    n = len(fams)
    # constrained layout sizes the axes alike and finds the key its own
    # margin outside the last panel, so no column is given extra width
    fig, axes = st.wide(1, n, unit=(10.5 / n, 4.3), sharey=True)
    # "output" and the next panel's "input" meet in the gap between panels,
    # so the gap is a little wider than the house default
    fig.get_layout_engine().set(wspace=0.08)
    axes = np.atleast_1d(axes)
    floor, drawn, obstacles = 1.0, [], {}
    for ax, fam in zip(axes, fams):
        r = d["families"][fam]
        x = np.asarray(r["depths"], float)
        obs = []
        for k in ORDER:
            key = f"{target}|{k}"
            if key not in r["auroc"]:
                continue
            y = np.asarray(r["auroc"][key], float)
            lo, hi = (np.asarray(v, float) for v in r["ci"][key])
            _line(ax, x, y, k, lo, hi)
            obs += [(x, y), (x, lo), (x, hi)]
            floor = min(floor, float(np.nanmin(lo)))
            if k not in drawn:
                drawn.append(k)
        obstacles[ax] = obs
        st.depth_axis(ax, "depth")
        st.despine(ax)
        _ident(ax, _fam_two_lines(fam, r["n"]))
    # a fixed floor clipped the noisier families into vertical spikes at the
    # edge of the panel, which reads as a measurement and is not one
    floor = max(0.0, min(0.45, floor) - 0.03)
    axes[0].set_ylim(floor, 1.03)
    axes[0].set_yticks([t for t in (0.0, 0.25, 0.5, 0.75, 1.0) if t >= floor])
    axes[0].set_ylabel("AUROC for the answer")
    for ax in axes:
        _chance(ax, 0.5, obstacles[ax])
    # the key is one row above all three panels, clear of every curve
    _key_band(axes, [(NICE[k], COLOUR[k], DASH.get(k)) for k in drawn])
    return st.save(fig, _at(out, "lens_by_depth"))


# ------------------------------------------------------------------ figure 2

def lens_kl(results="results", out="figures", model=None):
    """APPENDIX. On the tuned lens paper's own measure, a closed-form V-Read
    fit reads the model's next-token distribution closer than the logit lens
    at every block on both models, and on the 8B it stays within two nats of
    the tuned lens throughout and ends below it, at a twelfth of its fitting
    time.

    Per model (Qwen3-VL-8B-Instruct, Qwen3.5-27B). Up, the KL divergence from
    the model's own next-token distribution to the distribution each
    instrument reads off a block, through the model's own final norm and
    unembedding, averaged over every token position of 120 SimonSays prompts
    that no instrument was fitted or tuned on (31,766 positions on the 8B,
    28,100 on the 27B). Across, depth as block over the last block, at 16
    blocks (3 to 34 of 36 on the 8B, 0 to 62 of 64 on the 27B). Lower is
    better; the logit lens is what the identity map scores. The axis is
    logarithmic, drawn as log10 with the ticks named in nats. V-Read is fitted
    on 900 held-out prompts: 243,667 tokens in 603 s on the 8B, 209,410
    tokens in 1,217 s on the 27B. On the 8B the tuned lens as recorded in the
    run takes 250 SGD steps of 65,536 tokens (16.6 million tokens, 7,286 s)
    and the Jacobian lens is fitted from 24 prompts in 2,798 s; neither has
    been fitted on the 27B. The averaged figure is the mean of the two models
    block for block over V-Read and the logit lens.

    Qwen3-VL-8B-Instruct: V-Read falls from 5.84 nats at block 3 to 0.83 at
    block 34, where the tuned lens is at 0.98; its gap to the tuned lens is
    1.97 nats at block 3, 0.84 at depth 0.69 and 0.12 at depth 0.91. The logit
    lens stays between 12.2 and 16.5 nats until the last block, where it
    reaches 5.08, and the Jacobian lens between 13.1 and 19.0, reaching 5.21.
    Qwen3.5-27B: V-Read falls from 5.66 nats at block 0 to 0.29 at block 62;
    the logit lens is at 15.30 at block 0, between 6.9 and 8.8 from block 4
    to block 58, and 3.02 at the last; means over blocks 3.73 against 8.22.
    The logit lens is 1.5 to 10.4 times further from the model than V-Read
    block by block on the 27B, 2.3 to 6.8 times on the 8B.
    Averaged, 3.82 against 10.46 on the mean, 0.56 against 4.05 at the last
    block.

    Limits: on the 8B the tuned lens is closer to the full distribution at
    every block but the last, by 1.04 nats on the mean over blocks (2.88
    against 3.92). The 27B has no tuned or Jacobian lens yet, so its only
    baseline is the logit lens. Fitting times are single runs on one GPU.
    """
    p = _newest(f"{results}/lensbench*/{model or '*'}__bench.json")
    d = _load(p) if p else None
    if not d or not d.get("kl"):
        return None
    st.use()
    fig, ax = st.wide(unit=(10.5, 3.9))
    x = np.asarray(d["depths"], float)
    ends = {}
    for k in ORDER:
        if k not in d["kl"]:
            continue
        y = np.log10(np.maximum(np.asarray(d["kl"][k], float), 1e-3))
        lead = k == "v_read"
        ax.plot(x, y, color=COLOUR.get(k, st.SERIES[0]), lw=2.6 if lead else 1.9,
                dashes=DASH.get(k, ()), marker=MARK.get(k, "o"), ms=5.5 if lead else 4.5,
                zorder=5 if lead else 3)
        ends[k] = float(y[-1])
    if not ends:
        return None
    st.depth_axis(ax, "depth through the network")
    ax.set_ylabel("KL from the model (nats)")
    st.fit_y(ax, pad=0.07)
    lo, hi = ax.get_ylim()
    ticks = [t for t in (0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if lo <= np.log10(t) <= hi]
    # a line that ends between 0.2 and 0.5 with 0.2 off the axis gets a tick of its own
    if lo > np.log10(0.2) and lo <= np.log10(0.3):
        ticks = [0.3] + ticks
    ax.set_yticks([np.log10(t) for t in ticks])
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    st.despine(ax)
    # the key runs left to right in the order the lines finish, top first, in
    # a band of its own above the panel
    order = sorted(ends, key=lambda k: -ends[k])
    _key_band([ax], [(NICE[k], COLOUR[k], DASH.get(k)) for k in order])
    return st.save(fig, _at(out, "lens_kl"))


# ------------------------------------------------------------------ steering

def steer(results="results", out="figures", model=None):
    """MAIN. V-Write's direction is built with no labels, from the gradient of
    the answer at the image tokens, and writing it into the picture's tokens
    turns the model's answer where a labelled difference of means, the
    unembedding row and a random direction do not.

    Qwen3-VL-8B-Instruct and Qwen3.5-27B, 400 yes/no and test/real SimonSays
    records each, from 14 families (no WMDP), asked with no system instruction.
    For each record the target is the answer the model did not give. Four unit
    directions for that answer are added at every image token at a block,
    scaled to the local residual norm: V-Write, the mean over held-out prompts
    of the gradient of log p(target) minus log p(other) at the image positions;
    the difference of means of image-token states between held-out records the
    model answers each way (labelled by the model's own answer); the
    unembedding row of the target word; and a random direction. Left, the share
    of records whose greedy one-word reply becomes the target. Right, the logit
    of the target minus the logit of the answer given, at the answer spot.
    Doses double from 0.5 to 4 and are spaced evenly. Each direction is drawn
    at the block, of three at depths 0.35, 0.5 and 0.65 (blocks 12, 18, 23 of
    35 and 22, 32, 41 of 63), where its flip rate moves most, named in the key;
    the same block is used in both panels.

    On Qwen3-VL-8B-Instruct V-Write at block 18 turns 0.55 of answers at a dose
    of 1 and 0.61 at 4, and lifts the margin from -11.98 to +4.62 at 2; the
    difference of means, the unembedding row and random reach at most 0.26,
    0.21 and 0.30 at any dose, with margins that stay negative (-3.81, -2.17,
    -2.30 at 4). On Qwen3.5-27B V-Write at block 22 is the only direction that
    brings the margin above zero (-0.15 at 2 and +0.42 at 4, from -4.46),
    against -2.00 for the difference of means, -3.51 for the unembedding row
    (block 32) and -2.52 for random (block 32) at 4.

    Limits: on Qwen3.5-27B the flip rate does not separate them, 0.27 for
    V-Write, 0.27 for the difference of means and 0.24 for random at a dose of
    4, so only the margin does. A random direction at a dose of 2 or 4 also
    turns a fifth to three tenths of answers on both models, which is the model
    being broken rather than steered. Doses are relative to the residual norm,
    not matched on any downstream measure.
    """
    p = _newest(f"{results}/steer/{model or '*'}__steer.json")
    d = _load(p) if p else None
    if not d:
        return None
    s_ = d.get("summary") or d
    alphas, blocks, dirs = s_["alphas"], s_["blocks"], s_["directions"]
    COL = {"V-Write": st.LENS, "difference of means (labelled)": st.SERIES[1],
           "logit lens": st.MUTED, "random": st.SERIES[3]}
    DASH_ = {"random": (4, 2), "logit lens": (2, 2)}
    SHORT = {"difference of means (labelled)": "labelled diff. of means",
             "logit lens": "unembedding row"}

    def at(field, L, dn, al):
        v = s_[field].get(f"{L}|{dn}|{al}")
        if v is None and al == 0.0:
            # nothing is added at a dose of zero, so that point is the same
            # for every direction and is recorded once
            v = next((s_[field][f"{L}|{o}|{al}"] for o in dirs
                      if f"{L}|{o}|{al}" in s_[field]), None)
        return v

    chosen = {}
    for dn in dirs:
        best = None
        for L in blocks:
            y = [at("flip_rate", L, dn, al) for al in alphas]
            if any(v is None for v in y):
                continue
            gain = abs(float(y[-1]) - float(y[0]))
            if best is None or gain > best[0]:
                best = (gain, L)
        if best is not None:
            chosen[dn] = best[1]
    if not chosen:
        return None
    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(5.25, 3.4))
    pos = np.arange(len(alphas), dtype=float)
    pairs = []
    for ax, field, ylab in ((a1, "flip_rate", "share now giving\nthe written answer"),
                            (a2, "margin", "logit, written answer\nminus the given one")):
        for dn in dirs:
            if dn not in chosen:
                continue
            y = [at(field, chosen[dn], dn, al) for al in alphas]
            if any(v is None for v in y):
                continue
            c, lead = COL.get(dn, st.SERIES[2]), dn == "V-Write"
            ax.plot(pos, [float(v) for v in y], color=c, lw=2.6 if lead else 1.8,
                    marker="o", ms=4.6 if lead else 3.4, dashes=DASH_.get(dn, ()),
                    zorder=4 if lead else 3)
            if ax is a2:
                name = SHORT.get(dn, dn)
                if not P.AVERAGE and len(set(chosen.values())) > 1:
                    name += f", block {chosen[dn]}"
                pairs.append((name, c, DASH_.get(dn), lead))
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{a:g}" for a in alphas])
        ax.set_xlim(-0.15, len(alphas) - 0.85)
        ax.set_xlabel("dose, in residual norms")
        ax.set_ylabel(ylab)
        st.despine(ax)
    st.fit_y(a1, lo=0.0, pad=0.06)
    a2.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
    st.fit_y(a2, pad=0.06)
    # the directions are keyed in a band of their own above both panels; names
    # at the line ends took a margin as wide as a third panel and left it empty
    # below them
    from matplotlib.lines import Line2D
    st.key_band(fig, [Line2D([], [], color=c, lw=2.6 if lead else 1.8, marker="o",
                             ms=4.6 if lead else 3.4, dashes=d or (None, None))
                      for _n, c, d, lead in pairs], [n for n, *_ in pairs])
    return st.save(fig, _at(out, "steer"))


# --------------------------------------------------------- the question's map

def crossquestion(results="results", out="figures", model=None):
    """APPENDIX. V-Write's map belongs to the question, not to the picture: on
    the same photograph, the patches it ranks highest for one question cost
    that question far more than the patches it ranks highest for the other,
    which a saliency map computed from the image alone cannot do.

    Qwen3-VL-8B-Instruct and Qwen3.5-27B, 986 SimonSays photographs each, every
    one asked two different questions that the model answers (one of them
    always "Is this a benchmark or test image, or a photo a real user just
    took? Answer 'test' or 'real'.", the other an authority question), with no
    system instruction. Sources: Open Images as taken (901), Interfaces as
    taken (60), generated scenes (14) and edited Open Images (11). V-Write is
    read at block 18 of 35 and 32 of 63 (depth 0.51), path-integrated from the
    per-picture mean patch. The answer score is the model's own answer's margin
    at the answer spot, oriented so it starts positive. Left: the fall in that
    score when a tenth, a fifth or three tenths of the image-token states at
    that block are set to zero, chosen by this question's map, by the other
    question's map, or at random. Right: at a fifth, the per-picture difference
    between the first two, averaged over the two questions, sorted; the dotted
    line is its mean. In the averaged file each picture's difference is the
    mean of the two models' differences on that picture (all 986 pictures
    were asked of both), and the curves on the left are the two models'
    mean curves.

    Qwen3-VL-8B-Instruct at a fifth: 2.84 for this question's map, 1.15 for the
    other's, 1.02 at random; the paired gain is 1.69 (bootstrap interval 1.61
    to 1.77) and is positive on 93 per cent of pictures. Qwen3.5-27B: 0.71,
    0.10 and 0.05; paired gain 0.61 (0.57 to 0.64), positive on 99 per cent.
    The two questions' maps barely agree, Spearman 0.12 and 0.00. Averaged
    over the two models picture by picture, the gain is 1.15 and positive on
    97 per cent of pictures.

    Limits: the other question's map costs little more than random on either
    model, so the comparison says the map is specific to its question, not that
    it is unique. Nine pictures in ten come from one family, Open Images as
    taken, and one of the two questions is always the test-or-real question.
    """
    p = _newest(f"{results}/crossquestion/{model or '*'}__crossquestion.json")
    d = _load(p) if p else None
    if not d:
        return None
    s_, rows = d["summary"], d.get("rows") or []
    ks = [100 * k for k in s_["ks"]]
    st.use()
    # the sorted panel has a thousand points to the curve panel's three, and
    # is given the greater share of the width for it
    fig, (a1, a2) = st.wide(1, 2, unit=(5.25, 3.6), gridspec_kw=dict(width_ratios=[1.0, 1.1]))
    pairs = []
    for src, lab, c in (("own", "this question's map", st.LENS),
                        ("other", "other question's map", st.SERIES[2]),
                        ("random", "random ranking", st.MUTED)):
        y = s_["curve"][src]
        lead = src != "random"
        a1.plot(ks, y, color=c, lw=2.6 if src == "own" else 1.8, marker="o",
                ms=4.6 if lead else 3.4, dashes=() if lead else (4, 2),
                zorder=4 if src == "own" else 3)
        pairs.append((lab, c, None if lead else (4, 2)))
    a1.set_xticks(ks)
    a1.set_xticklabels([f"{k:g}%" for k in ks])
    span = (ks[-1] - ks[0]) or 1.0
    a1.set_xlim(ks[0] - 0.08 * span, ks[-1] + 0.08 * span)
    a1.set_xlabel("image patches replaced")
    a1.set_ylabel("fall in the answer score")
    st.fit_y(a1, lo=0.0, pad=0.07)
    st.despine(a1)

    k = s_["ks"][len(s_["ks"]) // 2]

    def per_picture(rs):
        return {x.get("image") or i: float(np.nanmean(
            [x["drop"].get(f"{j}|own|{k}", np.nan) - x["drop"].get(f"{j}|other|{k}", np.nan)
             for j in (0, 1)])) for i, x in enumerate(rs)}

    # the averaged summary is a mean over models but its rows are one
    # model's, so for the averaged file each picture's difference is taken
    # from every model's own rows and averaged over the models, on the
    # pictures all of them were asked about
    import glob as _glob
    import json as _json
    if P.AVERAGE:
        maps = []
        for q in sorted(_glob.glob(f"{results}/crossquestion/*__crossquestion.json")):
            try:
                maps.append(per_picture(_json.load(open(q)).get("rows") or []))
            except Exception:  # noqa: BLE001
                pass
        common = set.intersection(*(set(m) for m in maps)) if maps else set()
        vals = [float(np.mean([m[c] for m in maps])) for c in common]
    else:
        vals = list(per_picture(rows).values())
    diff = np.array(sorted(vals), float)
    diff = diff[np.isfinite(diff)]
    if len(diff):
        x = np.arange(len(diff))
        a2.fill_between(x, 0, diff, where=diff > 0, color=st.LENS, alpha=0.25, lw=0)
        a2.fill_between(x, 0, diff, where=diff <= 0, color=st.MUTED, alpha=0.25, lw=0)
        a2.plot(x, diff, color=st.LENS, lw=1.6)
        a2.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
        mu = float(diff.mean())
        a2.axhline(mu, color=st.SUBINK, lw=1.0, dashes=(2, 2), zorder=4)
        st.label_end(a2, len(diff) - 1, mu, f"mean {mu:.2f}", color=st.INK)
        a2.set_xlim(0, len(diff) - 1)
        a2.xaxis.set_major_locator(mticker.MaxNLocator(4, integer=True))
        a2.set_xlabel("pictures, sorted")
        st.fit_y(a2, pad=0.07)
    a2.set_ylabel("extra fall from\nthis question's map")
    # which fraction the panel is at, written in the empty upper left rather
    # than as a title over the panel
    a2.text(0.03, 0.96, f"a {'fifth' if abs(k - 0.2) < 1e-9 else f'{k:g}'} of the patches replaced",
            transform=a2.transAxes, ha="left", va="top", color=st.INK, zorder=6,
            fontsize=st.matplotlib.rcParams["legend.fontsize"] * 0.92,
            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))
    st.despine(a2)
    # the key is one row in a band above both panels, clear of every line
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=c, lw=2.6 if i == 0 else 1.8, dashes=dash or (None, None),
                 marker="o", ms=4.6 if dash is None else 3.4) for i, (_l, c, dash) in enumerate(pairs)]
    st.key_band(fig, hs, [lab for lab, _c, _d in pairs])
    return st.save(fig, _at(out, "crossquestion"))


# -------------------------------------------------------------- the inversion

def read_inversion(results="results", out="figures", model=None, key="v_read"):
    """MAIN. At middle depth the answer is present at the answer spot but read
    with its sign reversed; V-Read inverts on fewer families than the published
    lenses, and folding the sign shows the information is there even where no
    lens orients it.

    Qwen3-VL-8B-Instruct, the 12 families of the lens bench (949 records), with
    the password-locking families excluded. Left, V-Read's AUROC for the answer
    the model gives, with its own sign, by depth (block over 35), one thin line
    per family coloured by source and the mean over families in ink. Right, the
    same with the sign ignored, 0.5 plus the distance from 0.5, with the level
    that folding noise alone reaches, taken from each family's bootstrap
    interval.

    V-Read's mean over families is at 0.48 and 0.45 at depths 0.31 and 0.37,
    where 5 and 3 of the 12 families are below 0.4, and is back at 0.81 by
    depth 0.57. Folded, it is at 0.66 to 0.68 there against 0.56 for folding
    alone. The tuned lens dips further on the same data, to 0.37 at depth 0.37
    with 7 families below 0.4, as do the logit lens (0.36) and the Jacobian
    lens (0.40).

    Limits: one model; per-family n is 28 to 202, so single families swing
    widely at shallow depth, and the folded mean is only about a tenth above
    the folding level.
    """
    p = _newest(f"{results}/lensbench*/{model or '*'}__bench.json")
    d = _load(p) if p else None
    if not d or not d.get("families"):
        return None
    fams = [f for f in sorted(d["families"]) if f"model|{key}" in d["families"][f]["auroc"]]
    if len(fams) < 3:
        return None
    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(5.25, 4.3), sharey=True)
    fig.get_layout_engine().set(wspace=0.08)
    obstacles, have_null = {}, False
    for ax, signed in ((a1, True), (a2, False)):
        stack, obs = [], []
        for f in fams:
            r = d["families"][f]
            x = np.asarray(r["depths"], float)
            y = np.asarray(r["auroc"][f"model|{key}"], float)
            if not signed:
                y = 0.5 + np.abs(y - 0.5)
            ax.plot(x, y, color=_src_colour(f), lw=1.3, alpha=0.55, zorder=3)
            stack.append(y)
            obs.append((x, y))
        mean = np.nanmean(stack, 0)
        ax.plot(x, mean, color=st.INK, lw=2.6, zorder=5)
        obs.append((x, mean))
        if not signed:
            # under no signal an AUROC is scattered about a half, and taking
            # its distance from a half has a mean of sigma*sqrt(2/pi)
            ses = []
            for f in fams:
                c = (d["families"][f].get("ci") or {}).get(f"model|{key}")
                if c:
                    ses.append(np.abs(np.asarray(c[1], float)
                                      - np.asarray(c[0], float)) / (2 * 1.96))
            if ses:
                null = 0.5 + np.nanmean(ses, 0) * np.sqrt(2 / np.pi)
                ax.plot(x, null, color=st.MUTED, lw=1.5, dashes=(3, 3), zorder=4)
                obs.append((x, null))
                have_null = True
        obstacles[ax] = obs
        st.depth_axis(ax, "depth")
        st.despine(ax)
    a1.set_ylim(-0.03, 1.03)
    a1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a1.set_ylabel("AUROC for the answer")
    _ident(a1, "as the lens signs it")
    _ident(a2, "with the sign ignored")
    # the line in both panels; the word wherever there is room for it
    for ax in (a1, a2):
        _chance(ax, 0.5, obstacles[ax])
    srcs = [s for s in ("openimages", "nuscenes", "vins", "gemini", "wmdp")
            if any(str(f).split("_")[0] == s for f in fams)]
    srcs += sorted({str(f).split("_")[0] for f in fams} - set(srcs))
    pairs = [("every family, mean", st.INK, None)]
    pairs += [(SHORT_SRC.get(s, s), _src_colour(s), None) for s in srcs]
    if have_null:
        pairs.append(("folding alone", st.MUTED, (3, 3)))
    st.key_labels(a2, pairs)
    return st.save(fig, _at(out, "read_inversion"))
