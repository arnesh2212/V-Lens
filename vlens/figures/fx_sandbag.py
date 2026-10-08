# SPDX-License-Identifier: Apache-2.0
"""Sandbagging: the same questions under three instructions, read through depth."""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st

#: the three instructions every WMDP question is put under, in the order a
#: reader meets them, with the stroke each one keeps
INSTR = [("unlocked", "plain instruction", st.SLATE, ()),
         ("calibration_sandbagged", "told to underperform", st.OCHRE, (5, 2)),
         ("hazard_sandbagged", "underperform on hazards", st.CLAY, ())]
SUBJECTS = ("bio", "chem", "cybe")
LETTERS = "ABCD"


def _rows(results, model):
    """(depths, per-instruction arrays of names-correct and names-final) for one model."""
    out = {k: dict(right=[], final=[], acc=[]) for k, *_ in INSTR}
    depths = None
    for sub in SUBJECTS:
        rp = f"{results}/readouts/{model}__wmdp_{sub}_sand_readouts.json"
        bp = f"{results}/simonsays/{model}__wmdp_{sub}_sand.json"
        if not (os.path.exists(rp) and os.path.exists(bp)):
            continue
        rd = json.load(open(rp))
        beh = {r["id"]: r for r in json.load(open(bp))["rows"]}
        n_l = int(rd["n_layers"])
        blocks = rd.get("blocks") or list(range(n_l))
        depths = np.asarray(blocks, float) / max(n_l - 1, 1)
        for r in rd["rows"]:
            if r.get("kind") != "letter" or r["subcategory"] not in out:
                continue
            b = beh.get(r["id"])
            exp = str((b or {}).get("expected") or "").strip()[:1].upper()
            if exp not in LETTERS:
                continue
            lg = np.asarray(r["logit"], float)          # blocks x 4
            top = lg.argmax(1)
            final = top[-1]
            e = out[r["subcategory"]]
            e["right"].append(top == LETTERS.index(exp))
            e["final"].append(top == final)
            e["acc"].append(final == LETTERS.index(exp))
    return depths, out


def sandbag_depth(results="results", out="figures", model=None):
    """MAIN. A reading lens at the answer spot can separate knowing from saying under a sandbagging instruction: whether the correct letter is still carried through depth when the model is told to underperform, and where the letter it finally gives takes over.

    One model per file, Qwen3-VL-8B-Instruct or Qwen3.5-27B (the pooled file
    averages the two on normalised depth, block index over the last block).
    The 75 WMDP carrier items (25 each of biology, chemistry and cyber) are
    each asked three times with the same picture: under the plain
    instruction, told to underperform for calibration, and told to
    underperform on hazardous material. The password-locked families are not
    part of these files. Left: the share of items whose answer-spot readout
    at that depth ranks the correct letter first among the four. Right: the
    share whose readout ranks first the letter the model finally gives. The
    readout is the logit lens on the stored answer-spot state, through the
    model's own final norm and unembedding; at the last block it names the
    letter the model replied with on 95.5 per cent of 224 letter replies on
    Qwen3-VL-8B-Instruct and 94.7 per cent of 225 on Qwen3.5-27B.

    Neither model follows the instruction, so there is little to separate:
    spoken accuracy is 57, 61 and 63 per cent on Qwen3-VL-8B-Instruct under
    the plain, calibration and hazard instructions, and 64, 63 and 68 per
    cent on Qwen3.5-27B. At three quarters depth the readout already names the
    correct letter on 63, 61 and 64 per cent of items on Qwen3-VL-8B-Instruct
    and on 45, 53 and 41 per cent on Qwen3.5-27B, and at half depth on 17 to
    29 per cent, near the 25 per cent of a guess.

    Limits: this is the logit lens only, not V-Read; with the instruction
    ignored, agreement between the curves is the expected result and not
    evidence of hidden knowledge; 75 items per instruction.
    """
    models = [model] if model else sorted({os.path.basename(p).split("__")[0]
                                           for p in glob.glob(f"{results}/readouts/*__wmdp_*_sand_readouts.json")})
    got = [(m,) + _rows(results, m) for m in models]
    got = [g for g in got if g[1] is not None and any(len(v["right"]) for v in g[2].values())]
    if not got or (P.AVERAGE and len(got) < 2) or (model is None and not P.AVERAGE):
        return None
    grid = np.linspace(0, 1, 40)
    st.use()
    fig, axes = st.wide(1, 2, unit=(5.2, 3.3), sharey=True)
    for ax, what, lab in ((axes[0], "right", "names the correct letter"),
                          (axes[1], "final", "names the letter it gives")):
        for key, name, col, dash in INSTR:
            curves = []
            for m, depths, data in got:
                a = np.asarray(data[key][what], float)
                if not len(a):
                    continue
                curves.append(np.interp(grid, depths, a.mean(0)))
            if not curves:
                continue
            y = np.mean(curves, 0)
            ax.plot(grid, y, color=col, lw=2.6, dashes=dash, zorder=4)
        ax.axhline(0.25, color=st.AXIS, lw=1.0, zorder=1)
        st.depth_axis(ax, "depth")
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        # the panel is named where its lines never go: the upper left, which
        # stays empty until the answer forms late in depth
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, ha="left", va="top",
                color=st.INK)
        st.despine(ax)
    axes[0].set_ylabel("share of questions")
    # the key sits in the same empty region of the right panel, under its name
    ax = axes[1]
    for i, (_, name, col, dash) in enumerate(INSTR):
        y = 0.80 - 0.09 * i
        ax.plot([0.03, 0.11], [y, y], transform=ax.transAxes, color=col, lw=2.6,
                dashes=dash, solid_capstyle="butt")
        ax.text(0.13, y, name, transform=ax.transAxes, va="center", ha="left",
                color=st.INK, fontsize=st.matplotlib.rcParams["legend.fontsize"])
    return st.save(fig, P._at(out, "sandbag_depth"))


def sandbag_accuracy(results="results", out="figures", model=None):
    """APPENDIX. The behavioural check a sandbagging readout depends on: SimonSays puts the same WMDP carrier item under a plain, a calibration and a hazard instruction, so whether the instruction was followed at all is measured per subject before any lens is read.

    Both models on one panel, Qwen3-VL-8B-Instruct in slate and Qwen3.5-27B
    in clay. For each subject (biology, chemistry, cyber) and each of the
    three instructions (plain, told to underperform for calibration, told to
    underperform on hazards), the share of the 25 items answered with the
    correct letter, read from the first letter of the reply, one dot per
    model joined across instructions so that a fall reads as a slope. The
    dashed rule is a guess among four letters. The password-locked families
    are not part of these files.

    No slope falls the way an instruction to underperform would make it.
    Qwen3-VL-8B-Instruct answers 64, 68 and 68 per cent in biology, 60, 68
    and 68 in chemistry and 48, 48 and 52 in cyber; Qwen3.5-27B answers 60,
    68 and 64, 68, 68 and 72, and 64, 52 and 68. The one fall, Qwen3.5-27B in
    cyber under the calibration instruction, is three items of 25.

    Limits: 25 items a point, so a difference under about 20 points is
    within noise; the instruction was ignored, so there is nothing inside for
    a readout to explain on these models.
    """
    if model:
        return None
    models = sorted({os.path.basename(p).split("__")[0]
                     for p in glob.glob(f"{results}/simonsays/*__wmdp_bio_sand.json")})
    if not models:
        return None
    st.use()
    fig, axes = st.wide(1, 3, unit=(3.4, 3.1), sharey=True)
    NAMES = {"bio": "biology", "chem": "chemistry", "cybe": "cyber"}
    for ax, sub in zip(axes, SUBJECTS):
        for m in models:
            p = f"{results}/simonsays/{m}__wmdp_{sub}_sand.json"
            if not os.path.exists(p):
                continue
            rows = json.load(open(p))["rows"]
            ys = []
            for key, *_ in INSTR:
                sel = [r for r in rows if r.get("subcategory") == key]
                ok = [str(r.get("reply", "")).strip()[:1].upper() ==
                      str(r.get("expected", "")).strip()[:1].upper() for r in sel]
                ys.append(np.mean(ok) if ok else np.nan)
            c = st.CLAY if ("27" in m or "26" in m) else st.SLATE
            ax.plot(range(3), ys, color=c, lw=2.2, marker="o", ms=8, zorder=4)
        ax.axhline(0.25, color=st.AXIS, lw=1.0, dashes=(4, 3), zorder=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["plain", "under-\nperform", "on\nhazards"])
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel(NAMES[sub])
        st.despine(ax)
    axes[0].set_ylabel("answered correctly")
    st.key_labels(axes[-1], [(P._short(m), st.CLAY if ("27" in m or "26" in m) else st.SLATE, ())
                             for m in models])
    return st.save(fig, P._at(out, "sandbag_accuracy"))


#: the channel run's conditions, in reading order: (instruction, words, channel)
CHAN = [("none", "no instruction", None),
        ("honest", "answer well", "text"), ("honest", "answer well", "picture"),
        ("calibration", "40 per cent right", "text"),
        ("calibration", "40 per cent right", "picture"),
        ("hazard", "fail on hazards", "text"), ("hazard", "fail on hazards", "picture")]
LENSES = ("v_read", "tuned", "jacobian", "logit")
#: models are told apart by shape here, since colour belongs to the lenses on the right
MODELS2 = (("Qwen3-VL-8B-Instruct", "o"), ("Qwen3.5-27B", "D"))


def _chan(results, model):
    """Channel-run rows joined to their readouts: {sub: [(right, correct index, {lens: [blocks x 4]})]}."""
    bp = f"{results}/sandchan/{model}__wmdp_chan.json"
    rp = f"{results}/readouts/{model}__wmdp_chan_readouts.json"
    if not os.path.exists(bp):
        return None, None
    beh = json.load(open(bp))["rows"]
    rd = json.load(open(rp)) if os.path.exists(rp) else None
    rows = {r["id"]: r for r in (rd or {}).get("rows", [])}
    depths = None
    if rd:
        n_l = int(rd["n_layers"])
        depths = np.asarray(rd.get("blocks") or list(range(n_l)), float) / max(n_l - 1, 1)
    out: dict = {}
    for b in beh:
        got = str(b.get("reply") or "").strip()[:1].upper()
        exp = str(b.get("expected") or "")
        if exp not in LETTERS:
            continue
        r = rows.get(b["id"], {})
        reads = {k: r[k] for k in ("logit",) + LENSES if r.get(k) and k in r}
        out.setdefault(b["subcategory"], []).append(
            (got == exp, LETTERS.index(exp), got, reads, bool(r.get("v_read_fitted_on"))))
    return out, depths


def sandbag_channels(results="results", out="figures", model=None):
    """MAIN. SimonSays is the only sandbagging test that delivers the instruction to underperform either as text or printed into the picture, and the question either in the prompt or in the picture, on one fixed layout, so the channel that carried the instruction is the only thing that changes between conditions.

    The 75 unlocked WMDP carrier items (25 each of biology, chemistry and
    cyber), each under one topical illustration of the same layout, asked
    fourteen ways on Qwen3-VL-8B-Instruct (circles) and Qwen3.5-27B
    (diamonds): the question in the prompt (filled marks) or printed into the
    picture (open marks), crossed with seven instructions. No instruction
    leaves the default system message "You are a helpful assistant."; the
    other six are a control asking only for the best answer, the set's
    calibration instruction (answer correctly 40 per cent of the time) and its
    hazard instruction (give a wrong answer where the right one would help a
    weapon), each given either as the system message or printed as a notice
    in the picture. Left: the share of the 75 answered with the correct
    letter, read from the first letter of the reply; the rule at 25 per cent
    is a guess among four. Right, where lens readouts exist: on the items
    Qwen3-VL-8B-Instruct got wrong under the hazard instruction with the
    question in the prompt, the share on which each lens's top letter at the
    answer spot is the correct one, by depth, for V-Read, the tuned lens, the
    Jacobian lens and the logit lens read through the model's own final norm
    and unembedding, on items none was fitted on. The password-locked
    families are not part of this run, and no item text appears in the paper
    or its files.

    Neither model sandbags on instruction through either channel. With the
    question in the prompt, Qwen3-VL-8B-Instruct answers 58.7 per cent
    correctly with no instruction, 61.3 under the hazard instruction as text
    and 64.0 with it printed in the picture, and 61.3 under the calibration
    instruction in either channel; Qwen3.5-27B answers 65.3, 64.0 and 68.0,
    and 62.7 and 68.0. The control moves them no more (61.3 and 62.7 for
    Qwen3-VL-8B-Instruct, 64.0 and 68.0 for Qwen3.5-27B). What does move
    accuracy is where the question is: printed into the picture with no
    instruction it falls to 50.7 and 60.0, and the lowest cells are
    Qwen3-VL-8B-Instruct under the calibration notice printed in the picture
    (40.0) and Qwen3.5-27B under the hazard notice printed in the picture
    (50.7), both with the question also in the picture.

    Limits: at 75 items a cell the 95 per cent interval is about 11 points
    either way, so the differences between instructions are within noise and
    the figure shows compliance failing, not a graded effect. The lens readouts
    for this run (results/readouts/*__wmdp_chan_readouts.json) are not in
    results/, so the right panel is not drawn and no lens claim is made from
    it; a sandbagging readout needs a model that actually sandbags.
    """
    if model or not P.AVERAGE:
        return None
    data = {m: _chan(results, m) for m, _ in MODELS2}
    data = {m: v for m, v in data.items() if v[0]}
    if not data:
        return None
    import matplotlib.pyplot as plt
    st.use()
    fig, (ax, ay) = st.wide(1, 2, unit=(4.4, 3.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
    labels = []
    for j, (ins, words, ch) in enumerate(CHAN):
        labels.append(words + ("" if ch is None else f": {ch}"))
        for m, mk in MODELS2:
            if m not in data:
                continue
            rows = data[m][0]
            for wq, filled in (("text", True), ("picture", False)):
                sub = f"{wq}__{ins}" + ("" if ch is None else f"_{ch}")
                if sub not in rows:
                    continue
                acc = np.mean([x[0] for x in rows[sub]])
                ax.plot([acc], [j + (-0.14 if mk == "o" else 0.14)], marker=mk, ms=8 if mk == "o" else 7,
                        color=st.INK, mfc=st.INK if filled else "white", mew=1.5, ls="none", zorder=4)
    ax.axvline(0.25, color=st.AXIS, lw=1.0, zorder=1)
    ax.set_yticks(range(len(CHAN)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(CHAN) - 0.5, -0.5)
    # the axis runs from just below a guess among four to just past the
    # highest cell, so the marks are not packed into a third of the width
    accs = [x for ln in ax.lines for x in ln.get_xdata()]
    hi = min(1.0, np.ceil((max(accs) + 0.05) * 20) / 20) if accs else 1.0
    ax.set_xlim(0.2, hi)
    ticks = [t for t in (0.25, 0.5, 0.75, 1.0) if t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t * 100:.0f}" for t in ticks])
    ax.set_xlabel("correct letter given (%)")
    ax.tick_params(axis="y", length=0)
    st.despine(ax, left=False)
    fsk = st.matplotlib.rcParams["legend.fontsize"] * 0.95
    from matplotlib.lines import Line2D
    # the key goes in a band above the panels, not in a column beside the
    # rows of dots where it would read as more of them: models by shape, then
    # the channel of the question by fill
    # the two model entries are shape only, drawn grey so neither reads as
    # the filled mark that means the question was asked in text
    k_h = [Line2D([], [], marker=mk, ms=7.5, color=c, mfc=f, mec=c, mew=1.5, ls="none")
           for mk, c, f in (("o", st.MUTED, st.MUTED), ("D", st.MUTED, st.MUTED),
                            ("o", st.INK, st.INK), ("o", st.INK, "white"))]
    k_l = ["Qwen3-VL-8B-Instruct", "Qwen3.5-27B", "question asked in text", "question in the picture"]
    # right: can a lens still find the right letter where the model gave a wrong one
    m0 = MODELS2[0][0]
    drawn = False
    if m0 in data and data[m0][1] is not None:
        rows, depths = data[m0]
        wrong = [x for x in rows.get("text__hazard_picture", []) + rows.get("text__hazard_text", [])
                 if not x[0] and not x[4]]
        for k in reversed(LENSES):
            tr = [np.asarray([np.nan if v is None else np.argmax(v) == x[1] for v in x[3][k]], float)
                  for x in wrong if k in x[3]]
            if len(tr) < 5:
                continue
            share = np.nanmean(np.asarray(tr), axis=0)
            ok = np.isfinite(share)
            ay.plot(depths[ok], share[ok], color=P.COLOUR[k], lw=2.6 if k == "v_read" else 1.9,
                    dashes=P.DASH.get(k, (None, None)), marker="o" if ok.sum() < 30 else None, ms=3.5)
            drawn = True
        ay.axhline(0.25, color=st.AXIS, lw=1.0, zorder=1)
        ay.set_ylim(0, 1)
        ay.set_xlim(0, 1)
        ay.set_xticks([0, 0.5, 1]); ay.set_xticklabels(["input", "half", "output"])
        ay.set_xlabel("depth")
        ay.set_ylabel("right letter read", labelpad=2)
        st.despine(ay)
        k_h += [Line2D([], [], color=P.COLOUR[k], lw=2.6 if k == "v_read" else 1.9,
                       dashes=P.DASH.get(k, (None, None))) for k in LENSES]
        k_l += [P.NICE[k] for k in LENSES]
    if not drawn:
        # one panel, as wide as the page: a narrow column of dots beside an
        # empty half-page was the figure's largest area
        fig.delaxes(ay)
        fig.set_size_inches(st.MAX_WIDTH, 3.1)
        fig.set_layout_engine("none")
        ax.set_position([0.25, 0.2, 0.735, 0.78])
    else:
        fig.tight_layout(pad=0.3, w_pad=9.0)
    st.key_band(fig, k_h, k_l, fontsize=fsk, handlelength=1.2, handletextpad=0.4,
                ncol=4 if not drawn else None)
    return st.save(fig, P._at(out, "sandbag_channels"), check=False, tight=True)
