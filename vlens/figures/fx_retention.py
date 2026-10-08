# SPDX-License-Identifier: Apache-2.0
"""Removing image tokens in V-Write's order: the answer goes fast, and it goes as V-Write says it will."""
from __future__ import annotations

import json
import os

import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st

ORDERS = (("vwrite_top", "V-Write, most important first", st.CLAY, "-"),
          ("attr_patch_top", "attribution patching, most important first", st.SLATE, "-"),
          ("random", "random order", st.GRAY, (0, (3, 2))),
          ("vwrite_bottom", "V-Write, least important first", st.CLAY, (0, (1.5, 1.5))))


def _rows(results, model):
    p = f"{results}/retention/{model}__retention.rows.jsonl"
    if not os.path.exists(p):
        return None
    return [json.loads(line) for line in open(p) if line.strip()]


def retention(results="results", out="figures", model="Qwen3-VL-8B-Instruct"):
    """MAIN. Removing the image tokens V-Write ranks highest takes the answer away fastest, and the answer falls in step with the share of V-Write's contribution removed."""
    rows = _rows(results, model or "Qwen3-VL-8B-Instruct")
    if not rows:
        return None
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    st.use()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(st.MAX_WIDTH, 2.7),
                                     gridspec_kw=dict(width_ratios=[1.3, 1.0, 0.8], wspace=0.45))
    g = np.asarray(rows[0]["grid"], float)
    for k, _lab, c, ls in ORDERS:
        M = np.array([np.asarray(r["curve"][k]) / r["s0"] for r in rows])
        med = np.median(M, 0)
        q1, q3 = np.percentile(M, [25, 75], 0)
        a1.plot(g * 100, med, color=c, ls=ls, lw=2.3 if k.startswith("vwrite") else 1.6)
        a1.fill_between(g * 100, q1, q3, color=c, alpha=0.08, lw=0)
    a1.axhline(1, color=st.AXIS, lw=0.8)
    a1.set_xlabel("image tokens removed (%)"); a1.set_ylabel("answer margin left")
    st.despine(a1)
    # the effect left against V-Write's contribution left, top-first order
    ok = [r for r in rows if r["s0"] - r["s_all"] > 1.0]
    S = np.array([[np.nan if v is None else v for v in r["share"]["vwrite_top"]] for r in ok])
    E = np.array([(np.asarray(r["curve"]["vwrite_top"]) - r["s_all"]) / (r["s0"] - r["s_all"]) for r in ok])
    xs, ys = np.nanmedian(S, 0), np.nanmedian(E, 0)
    lo, hi = min(np.nanmin(xs), np.nanmin(ys)) - 0.1, 1.1
    a2.plot([lo, hi], [lo, hi], color=st.AXIS, lw=0.9, ls=(0, (3, 2)))
    a2.plot(xs, ys, color=st.CLAY, lw=2.2, marker="o", ms=3.5)
    a2.set_xlim(lo, hi); a2.set_ylim(lo, hi)
    a2.set_xlabel("V-Write contribution left"); a2.set_ylabel("effect left")
    st.despine(a2)
    # the smallest top-first set that removes half of the answer
    reg = {k: [r["region"][k] for r in rows if r["region"].get(k)] for k in ("vwrite_top", "attr_patch_top", "random")}
    labs, sizes, onobj, cols = [], [], [], []
    for k, lab, c in (("vwrite_top", "V-Write", st.CLAY), ("attr_patch_top", "attr. patching", st.SLATE),
                      ("random", "random", st.GRAY)):
        if reg[k]:
            labs.append(lab); cols.append(c)
            sizes.append(100 * np.mean([x["size"] for x in reg[k]]))
            onobj.append(100 * np.mean([x["on_object"] for x in reg[k]]))
    y = np.arange(len(labs))
    a3.barh(y + 0.18, sizes, height=0.34, color=cols, alpha=0.45)
    a3.barh(y - 0.18, onobj, height=0.34, color=cols)
    for yy, s_, o_ in zip(y, sizes, onobj):
        a3.text(s_ + 1.5, yy + 0.18, f"{s_:.0f}%", va="center", color=st.INK, fontsize=st.matplotlib.rcParams["xtick.labelsize"] * 0.75)
        a3.text(o_ + 1.5, yy - 0.18, f"{o_:.0f}%", va="center", color=st.INK, fontsize=st.matplotlib.rcParams["xtick.labelsize"] * 0.75)
    a3.set_yticks(y); a3.set_yticklabels(labs); a3.invert_yaxis()
    a3.set_xlabel("% of picture (pale) / on object (dark)"); a3.set_xlim(0, 70)
    st.despine(a3)
    hs = [Line2D([], [], color=c, ls=ls, lw=2.2 if k.startswith("vwrite") else 1.6) for k, _l, c, ls in ORDERS]
    hs.append(Line2D([], [], color=st.AXIS, lw=0.9, ls=(0, (3, 2))))
    st.key_band(fig, hs, [lab for _k, lab, _c, _ls in ORDERS] + ["perfect additivity"], rows=2,
                fontsize=st.matplotlib.rcParams["xtick.labelsize"] * 0.75)
    return st.save(fig, P._at(out, "retention"), check=False, tight=True)
