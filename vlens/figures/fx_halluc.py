# SPDX-License-Identifier: Apache-2.0
"""Telling a hallucinated yes from a true one, before the model answers.

POPE asks whether an object is in a COCO photograph. Among the questions the
model answers yes, some objects are not there. Each reader's yes-minus-no
reading at each block is scored by how well it separates the true yeses from
the hallucinated ones (AUROC), on the same records and blocks for every reader.
"""
from __future__ import annotations

import ast
import json
import os

import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st

MODELS = ("Qwen3-VL-8B-Instruct", "Qwen3.5-27B")
READERS = (("v_read", "V-Read"), ("jacobian", "Jacobian lens"), ("tuned", "tuned lens"),
           ("patchscope", "Patchscopes"), ("logit", "logit lens"))


def _auroc(p, n):
    p, n = np.asarray(p), np.asarray(n)
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def _vec(v):
    if isinstance(v, str):
        v = ast.literal_eval(v)
    return None if v is None else np.array([np.nan if x is None else x for x in v], float)


def data(results, m):
    pb = f"{results}/coco/{m}__coco_pope.json"
    pr = f"{results}/coco_readouts/{m}__coco_pope_readouts.json"
    pp = f"{results}/coco_readouts/{m}__coco_pope_patchscope.json"
    if not all(os.path.exists(p) for p in (pb, pr, pp)):
        return None
    beh = {r["id"]: r for r in json.load(open(pb))["rows"]}
    R = json.load(open(pr))
    ps = {r["id"]: r for r in json.load(open(pp))["rows"]}
    yes = [r for r in R["rows"] if (beh[r["id"]]["margins"] or {}).get("yes_over_no", 0) > 0]
    lab = np.array([beh[r["id"]]["expected"] == "yes" for r in yes])
    blocks = np.asarray(R["blocks"])
    n_layers = int(json.load(open(pp)).get("n_layers") or blocks[-1] + 1)
    X = {}
    for k, _ in READERS:
        vals = [_vec(r.get(k) if k != "patchscope" else ps.get(r["id"], {}).get("patchscope")) for r in yes]
        if any(v is None for v in vals):
            continue
        X[k] = np.stack(vals)
    return dict(lab=lab, X=X, x=(blocks + 1) / n_layers, blocks=blocks, n_layers=n_layers)


def curves(D, rng, n_boot=1000):
    lab = D["lab"]
    out = {}
    for k, X in D["X"].items():
        ok = ~np.isnan(X).any(0)
        au = np.array([_auroc(X[lab, b], X[~lab, b]) if ok[b] else np.nan for b in range(X.shape[1])])
        lo = hi = None
        if k == "v_read":
            pi, ni = np.where(lab)[0], np.where(~lab)[0]
            bs = []
            for _ in range(n_boot):
                a, b = rng.choice(pi, len(pi)), rng.choice(ni, len(ni))
                bs.append([_auroc(X[a, j], X[b, j]) if ok[j] else np.nan for j in range(X.shape[1])])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        out[k] = (au, lo, hi)
    return out


def hallucination_depth(results="results", out="figures", model=None):
    """APPENDIX. V-Read tells a hallucinated yes from a true one earliest on Qwen3-VL-8B-Instruct."""
    if model not in (None, MODELS[0]):
        return None
    Ds = {m: data(results, m) for m in MODELS}
    Ds = {m: d for m, d in Ds.items() if d}
    if not Ds:
        return None
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    st.use()
    fs = st.matplotlib.rcParams["xtick.labelsize"]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(Ds), figsize=(st.MAX_WIDTH * 0.75, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (m, D) in zip(axes, Ds.items()):
        C = curves(D, rng)
        for k, lab in reversed(READERS):
            if k not in C:
                continue
            au, lo, hi = C[k]
            col = P.COLOUR.get(k, st.INK)
            ok = ~np.isnan(au)
            ax.plot(D["x"][ok], au[ok], color=col, lw=2.4 if k == "v_read" else 1.4,
                    dashes=P.DASH.get(k, ()), zorder=5 if k == "v_read" else 3)
            if lo is not None:
                ax.fill_between(D["x"][ok], lo[ok], hi[ok], color=col, alpha=0.15, lw=0)
        ax.axhline(0.5, color=st.AXIS, lw=0.7)
        ax.set_ylim(0.2, 1.0); ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["input", "half", "output"])
        ax.set_title(f"{m}\n{int(D['lab'].sum())} true, {int((~D['lab']).sum())} hallucinated yes",
                     fontsize=fs * 0.66, color=st.INK, pad=4, linespacing=1.1)
        st.despine(ax)
    axes[0].set_ylabel("AUROC", fontsize=fs * 0.75)
    hs = [Line2D([], [], color=P.COLOUR.get(k, st.INK), lw=2.4 if k == "v_read" else 1.4, dashes=P.DASH.get(k, ()))
          for k, _ in READERS]
    st.key_band(fig, hs, [lab for _, lab in READERS], rows=1, fontsize=fs * 0.7)
    return st.save(fig, P._at(out, "hallucination_depth"), check=False, tight=True)
