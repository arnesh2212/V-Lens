# SPDX-License-Identifier: Apache-2.0
"""A reference made in one turn and used in the next (CLEVR): how long the model keeps hold of it."""
from __future__ import annotations

import json
import os

import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st

MODEL = "Qwen3-VL-8B-Instruct"


def reference_turns(results="results", out="figures", model=None):
    """APPENDIX. The model names the colour of the object an earlier turn pointed at, until two unrelated turns intervene; V-Write credits the pointing word, not the object."""
    if model not in (None, MODEL):
        return None
    p = f"{results}/reference/{MODEL}__reference.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    s, rows = d["summary"], d["rows"]
    import matplotlib.pyplot as plt
    st.use()
    fs = st.matplotlib.rcParams["xtick.labelsize"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(st.MAX_WIDTH * 0.72, 2.2),
                                     gridspec_kw=dict(width_ratios=[1, 0.9, 1.1], wspace=0.45))
    # 1: right answer by the number of unrelated turns in between, with a Wilson interval
    F = sorted(int(k) for k in s["correct"])
    n = 2 * int(s["n"])
    acc = np.array([s["correct"][str(f)] for f in F])
    z = 1.96
    lo = (acc + z * z / (2 * n) - z * np.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n))) / (1 + z * z / n)
    hi = (acc + z * z / (2 * n) + z * np.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n))) / (1 + z * z / n)
    a1.errorbar(F, acc, yerr=[acc - lo, hi - acc], color=st.CLAY, marker="o", lw=2, ms=5, capsize=2)
    a1.axhline(0.5, color=st.AXIS, lw=0.8, ls=(0, (3, 2)))
    a1.text(F[-1] + 0.1, 0.52, "chance", ha="left", va="top", fontsize=fs * 0.62, color=st.SUBINK)
    a1.set_xticks(F); a1.set_ylim(0.3, 1.02)
    a1.set_xlabel("unrelated turns in between", fontsize=fs * 0.7)
    a1.set_ylabel("right colour", fontsize=fs * 0.7)
    # 2: what V-Write credits at the read block
    W = s["vwrite"]
    names = ["the word left or right", "selected object", "other object"]
    vals = [W[k] for k in names]
    a2.barh(range(3), vals, color=[st.CLAY, st.SLATE, st.GRAY], height=0.6)
    a2.set_yticks(range(3)); a2.set_yticklabels(["word ``left''/``right''".replace("``", "“").replace("''", "”"),
                                                 "pointed-at object", "other object"], fontsize=fs * 0.62)
    a2.invert_yaxis(); a2.axvline(0, color=st.AXIS, lw=0.8)
    a2.set_xlabel("V-Write credit", fontsize=fs * 0.7)
    # 3: when the selected colour becomes readable, by depth (plain readout of the answer spot)
    L = sorted(int(k) for k in s["depth"])
    x = (np.array(L) + 1) / (L[-1] + 1)
    M = np.array([[r["by_side"][sd]["0"]["depth"][str(l)] for l in L] for r in rows for sd in ("left", "right")
                  if "0" in r["by_side"].get(sd, {})])
    med = np.median(M, 0); q1, q3 = np.percentile(M, [25, 75], 0)
    a3.fill_between(x, q1, q3, color=st.CLAY, alpha=0.15, lw=0)
    a3.plot(x, med, color=st.CLAY, lw=2)
    a3.axhline(0, color=st.AXIS, lw=0.8)
    a3.set_xlim(0, 1); a3.set_xticks([0, 0.5, 1]); a3.set_xticklabels(["input", "half", "output"])
    a3.set_ylabel("selected over other colour", fontsize=fs * 0.7)
    for a in (a1, a2, a3):
        a.tick_params(labelsize=fs * 0.62); st.despine(a)
    return st.save(fig, P._at(out, "reference_turns"), check=False, tight=True)
