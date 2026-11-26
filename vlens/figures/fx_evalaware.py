# SPDX-License-Identifier: Apache-2.0
"""Evaluation awareness at a glance: one scene dressed four ways, and how far each cue moves the model inside."""
from __future__ import annotations

import textwrap

import numpy as np

from vlens.figures import fx_packaging as FP
from vlens.figures import paper as P
from vlens.figures import style as st

MODELS = ("Qwen3-VL-8B-Instruct", "Qwen3.5-27B")
SCENE = "s1"
LOOKS = (("V0", "as generated"), ("V1", "dressed as a paper figure"), ("V2", "annotation boxes"),
         ("V3", "answer options drawn in"))
CUES = (("E1", "test"), ("E4", "live"))


def _flips(D, cells):
    c = [D["cells"][k] for k in cells if k in D["cells"]]
    return sum(x["flips"] for x in c), sum(x["n"] for x in c)


def _band(ax, x, T, col, lw, label):
    med = np.median(T, 0)
    q1, q3 = np.percentile(T, [25, 75], 0)
    ax.fill_between(x, q1, q3, color=col, alpha=0.16, lw=0)
    ax.plot(x, med, color=col, lw=lw, label=label)
    return med


def eval_awareness(results="results", out="figures", model=None):
    """MAIN. A sentence saying the model is being tested moves its internal state as far as a new question, a picture dressed as a benchmark about half as far, and neither changes the answer."""
    if model not in (None, MODELS[0]):
        return None
    Ds = {m: FP._data(results, m) for m in MODELS}
    Ds = {m: D for m, D in Ds.items() if D is not None}
    if MODELS[0] not in Ds:
        return None
    import matplotlib.pyplot as plt
    from vlens import corpus as C
    from vlens.figures.fx_trace import _image_path
    st.use()
    fs = st.matplotlib.rcParams["xtick.labelsize"]
    D0 = Ds[MODELS[0]]
    q = next(q for q in D0["qs"] if q[0] == SCENE)
    W, H = st.MAX_WIDTH, 3.25
    fig = plt.figure(figsize=(W, H))
    fig.set_layout_engine("none")
    # left: the scene four ways, asked the same question
    tw = 0.148; th = tw * W / H
    y_img = 0.93 - th - 0.05
    for k, (v, lab) in enumerate(LOOKS):
        r = D0["rows"][(v + "E0", q)]
        ax = fig.add_axes([0.005 + k * (tw + 0.006), y_img, tw, th])
        ax.imshow(C.load_image(_image_path(None, dict(r, source=r.get("source", "simonsays"))), square=448))
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color(st.CLAY if k else st.AXIS); s_.set_linewidth(1.6 if k else 0.8)
        ax.text(0.5, 1.02, "\n".join(textwrap.wrap(lab, 20)), transform=ax.transAxes, ha="center", va="bottom",
                fontsize=fs * 0.66, color=st.CLAY if k else st.INK)
    r0 = D0["rows"][("V0E0", q)]
    L = 0.005 + 4 * tw + 3 * 0.006
    fig.text(0.005, y_img - 0.04, f"Every version is asked \u201c{r0['meta']['base_question']}\u201d, plain or after a sentence:",
             fontsize=fs * 0.7, color=st.INK, va="top")
    for j, (e, lab) in enumerate(CUES):
        pre = str(D0["rows"][("V0" + e, q)]["question"])[: -len(r0["meta"]["base_question"])].strip()
        fig.text(0.005 + j * L / 2, y_img - 0.13, "\n".join(textwrap.wrap(f"{lab}: \u201c{pre}\u201d", 40)),
                 fontsize=fs * 0.62, color=st.INK, va="top",
                 bbox=dict(boxstyle="round,pad=0.35", fc=st.BAND, ec=FP.WORDS, lw=0.9))
    # right: how far the state moves, in units of the distance between two different questions
    x0, w = L + 0.075, 0.99 - L - 0.075
    n = len(Ds)
    hh = (0.86 - 0.12 - 0.07 * (n - 1)) / n
    for i, (m, D) in enumerate(Ds.items()):
        ax = fig.add_axes([x0, 0.86 - (i + 1) * hh - i * 0.07, w, hh])
        x = D["x"]
        fw, nw = _flips(D, ("V0E1", "V0E2", "V0E3", "V0E4"))
        fp, npx = _flips(D, ("V1E0", "V2E0", "V3E0"))
        mw = _band(ax, x, D["words"], FP.WORDS, 2.2, "sentence")
        mp = _band(ax, x, D["pixels"], FP.PIXELS, 2.2, "picture")
        ax.axhline(1, color=st.AXIS, lw=0.9)
        ax.text(0.02, 1.02, "a different question", transform=ax.get_yaxis_transform(), fontsize=fs * 0.6,
                color=st.SUBINK, va="bottom")
        yw, yp = mw[-1], mp[-1]
        if abs(yw - yp) < 0.3:
            mid = (yw + yp) / 2
            yw, yp = (mid + 0.15, mid - 0.15) if yw >= yp else (mid - 0.15, mid + 0.15)
        ax.text(1.01, yw, f"sentence\n{fw}/{nw} answers change", color=FP.WORDS, fontsize=fs * 0.58,
                va="center", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.text(1.01, yp, f"picture\n{fp}/{npx} change", color=FP.PIXELS, fontsize=fs * 0.58,
                va="center", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.set_ylim(0, 1.3); ax.set_xlim(0, 1); ax.set_yticks([0, 0.5, 1])
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["input", "half", "output"] if i == n - 1 else [])
        ax.set_ylabel("state moved", fontsize=fs * 0.7)
        ax.set_title(m, fontsize=fs * 0.7, color=st.INK, pad=2)
        st.despine(ax)
    return st.save(fig, P._at(out, "eval_awareness"), check=False, tight=True)
