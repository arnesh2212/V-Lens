# SPDX-License-Identifier: Apache-2.0
"""V-Write under tests that share nothing with how it is built.

V-Write integrates from the picture's mean image-token state, so hiding tokens
by that state would favour it. Three tests here use other counterfactuals:

  left    the top fifth of image tokens set to zero (the baseline integrated
          gradients uses), every attribution method, four models
          (results/baselines/<m>__patchlenses.json)
  middle  the top tenth of the picture covered in pixels, five ways, against a
          random tenth (results/occlusion/<m>__occlusion.json)
  right   this question's map against the map for the other question asked of
          the same photograph (results/crossquestion/<m>__crossquestion.rows.jsonl)
"""
from __future__ import annotations

import json
import os

import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st
from vlens.figures import fx_vwrite_gains as G

MODELS = ("Qwen3-VL-8B-Instruct", "Qwen3.5-27B", "gemma-4-26B-A4B-it", "InternVL3_5-8B-HF")
SHORT = {"Qwen3-VL-8B-Instruct": "Qwen3-VL-8B", "Qwen3.5-27B": "Qwen3.5-27B",
         "gemma-4-26B-A4B-it": "gemma-4-26B", "InternVL3_5-8B-HF": "InternVL3.5-8B"}
BOOT = 2000
ABBR = dict(act_patch="act. patching", ig_zero="IG, zero state", kernel_shap="KernelSHAP", rise="RISE",
            attr_patch="attr. patching", grad_x_act="grad. x act.")


def _ci(x, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (BOOT, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def zero_fall(results, m, k=0.2):
    """Fall in the answer margin with the top fifth zeroed: V-Write, the best other method, random."""
    d = G._json(os.path.join(results, "baselines", f"{m}__patchlenses.json"))
    if not (d and d.get("rows")):
        return None
    key = lambda s: f"zero|{s}|{k:g}"  # noqa: E731
    rows = [r for r in d["rows"] if G._ok_row(r) and isinstance(r.get("after"), dict)
            and r.get("s0") is not None and key("v_write") in r["after"]]
    cands = [s for s in G.PUBLISHED if not P._hidden(s) and all(key(s) in r["after"] for r in rows)]
    fall = lambda s: [float(r["s0"]) - float(r["after"][key(s)]) for r in rows]  # noqa: E731
    best = max(cands, key=lambda s: np.mean(fall(s)))
    out = dict(n=len(rows), best=best, v_write=_ci(fall("v_write")), base=_ci(fall(best)))
    if all(key("random") in r["after"] for r in rows):
        out["random"] = _ci(fall("random"))
    return out


def pixel_cover(results, m="Qwen3-VL-8B-Instruct", k=0.1):
    p = f"{results}/occlusion/{m}__occlusion.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    out = {}
    for how in d["summary"]["methods"]:
        top = [r["s0"] - r["a3"][f"{how}|{k:g}|top"] for r in d["rows"]]
        rnd = [r["s0"] - r["a3"][f"{how}|{k:g}|random"] for r in d["rows"]]
        out[how] = dict(top=_ci(top), random=_ci(rnd), diff=_ci(np.subtract(top, rnd)))
    return dict(n=len(d["rows"]), rows=out)


def other_question(results, m, k=0.2):
    """Per photograph (mean over its two questions, as the run's summary): this map, the other's, random."""
    p = f"{results}/crossquestion/{m}__crossquestion.json"
    if not os.path.exists(p):
        return None
    rows = json.load(open(p))["rows"]
    per = lambda r, src: np.nanmean([r["drop"].get(f"{i}|{src}|{k:g}", np.nan) for i in (0, 1)])  # noqa: E731
    return dict(n=len(rows), **{src: _ci([per(r, s2) for r in rows])
                                for src, s2 in (("own", "own"), ("other", "other"), ("random", "random"))})


def vwrite_beyond(results="results", out="figures", model=None):
    """MAIN. V-Write passes tests built on counterfactuals it does not use: zeroed tokens, covered pixels, the other question's map."""
    if model or not P.AVERAGE:
        return None
    Z = {m: zero_fall(results, m) for m in MODELS}
    Z = {m: z for m, z in Z.items() if z}
    C = pixel_cover(results)
    Q = {m: other_question(results, m) for m in MODELS[:2]}
    Q = {m: q for m, q in Q.items() if q}
    if not (Z and C and Q):
        return None
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    st.use()
    fs = st.matplotlib.rcParams["xtick.labelsize"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(st.MAX_WIDTH, 2.7),
                                     gridspec_kw=dict(width_ratios=[1.0, 1.0, 0.9], wspace=0.38))
    kinds = (("v_write", "V-Write", st.CLAY), ("base", "best other method", st.OCHRE), ("random", "random", st.GRAY))

    def dots(ax, groups, labels, series, xlabel):
        n = len(series)
        off = (np.arange(n) - (n - 1) / 2) * 0.24
        for g, grp in enumerate(groups):
            for j, (key, _lab, col) in enumerate(series):
                if key not in grp:
                    continue
                mu, lo, hi = grp[key]
                ax.errorbar(mu, g + off[j], xerr=[[mu - lo], [hi - mu]], fmt="o", color=col,
                            ms=5.5 if j == 0 else 4.2, lw=1.2, capsize=0, zorder=4 - j)
        ax.set_yticks(range(len(groups))); ax.set_yticklabels(labels, fontsize=fs * 0.66)
        ax.set_ylim(len(groups) - 0.5, -0.6)
        ax.set_xlabel(xlabel, fontsize=fs * 0.68)
        ax.tick_params(axis="x", labelsize=fs * 0.62)
        st.despine(ax)

    # 1: zeroed tokens, four models
    ms = list(Z)
    dots(a1, [Z[m] for m in ms],
         [f"{SHORT[m]}\nvs {ABBR.get(Z[m]['best'], Z[m]['best'])}" for m in ms], kinds,
         "drop in answer score")
    a1.set_title("top 20% of image tokens set to zero", fontsize=fs * 0.72, pad=4)
    # 2: covered pixels, five ways
    hows = list(C["rows"])
    dots(a2, [dict(v_write=C["rows"][h]["top"], random=C["rows"][h]["random"]) for h in hows], hows,
         (("v_write", "V-Write", st.CLAY), ("random", "random", st.GRAY)), "drop in answer score")
    a2.set_xlim(0, None)
    a2.set_title("top 10% of the picture covered", fontsize=fs * 0.72, pad=4)
    # 3: this question's map against the other question's, same photograph
    qs = list(Q)
    rel = lambda q, k: tuple(v / q["own"][0] for v in q[k])  # noqa: E731
    dots(a3, [dict(v_write=rel(Q[m], "own"), base=rel(Q[m], "other"), random=rel(Q[m], "random")) for m in qs],
         [SHORT[m] for m in qs],
         (("v_write", "this question's map", st.CLAY), ("base", "other question's map", st.SLATE),
          ("random", "random", st.GRAY)), "drop in answer score\n(relative to this question's map)")
    a3.set_xlim(0, 1.15)
    a3.set_title("map made for a different question", fontsize=fs * 0.72, pad=4)
    hs = [Line2D([], [], color=c, marker="o", lw=0, ms=5) for c in (st.CLAY, st.OCHRE, st.SLATE, st.GRAY)]
    st.key_band(fig, hs, ["V-Write", "best other method", "V-Write map for a different question",
                          "random tokens"], rows=1, fontsize=fs * 0.66)
    json.dump(dict(zero=Z, pixels=C, other_question=Q), open(os.path.join(results, "checks", "vwrite_beyond.json"), "w"),
              indent=1)
    return st.save(fig, P._at(out, "vwrite_beyond"), check=False, tight=True)
