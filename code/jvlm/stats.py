# SPDX-License-Identifier: Apache-2.0
"""Inference for paired, concept-clustered designs.

Items sharing a concept share a direction v_c, so they are not independent.
Everything here clusters at the concept level: the bootstrap resamples
concepts, not rows.  Margins are heavy-tailed, so paired location tests are
signed-rank rather than t; paired binary outcomes use McNemar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def cluster_bootstrap_ci(df: pd.DataFrame, value: str, cluster: str = "concept",
                         n: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Percentile CI for the mean of ``value``, resampling whole clusters."""
    g = [v[value].to_numpy(float) for _, v in df.groupby(cluster)]
    rng = np.random.default_rng(seed)
    k = len(g)
    boot = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, k, k)
        boot[b] = np.concatenate([g[i] for i in idx]).mean()
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(np.concatenate(g).mean()), float(lo), float(hi)


def paired(df: pd.DataFrame, cond_a: str, cond_b: str, value: str,
           item: str = "i") -> pd.DataFrame:
    """Align two conditions item by item; returns columns ``a`` and ``b``."""
    A = df[df.cond == cond_a].set_index(item)[value]
    B = df[df.cond == cond_b].set_index(item)[value]
    idx = A.index.intersection(B.index)
    out = pd.DataFrame({"a": A.loc[idx], "b": B.loc[idx]})
    key = df.drop_duplicates(item).set_index(item)
    for col in ("concept", "distractor"):
        if col in key.columns:
            out[col] = key.loc[idx, col]
    return out.dropna()


def wilcoxon(df: pd.DataFrame, cond_a: str, cond_b: str, value: str,
             item: str = "i") -> dict:
    """Paired signed-rank test of ``cond_b`` against ``cond_a``."""
    p = paired(df, cond_a, cond_b, value, item=item)
    if len(p) < 6 or (p.a == p.b).all():
        return dict(n=len(p), stat=float("nan"), p=float("nan"), delta=0.0)
    st, pv = stats.wilcoxon(p.a, p.b)
    return dict(n=len(p), stat=float(st), p=float(pv),
                delta=float((p.b - p.a).mean()))


def mcnemar(df: pd.DataFrame, cond_a: str, cond_b: str, value: str) -> dict:
    """Exact McNemar on a paired binary outcome (values thresholded at 0.5)."""
    p = paired(df, cond_a, cond_b, value)
    a, b = p.a > 0.5, p.b > 0.5
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    if n01 + n10 == 0:
        return dict(n=len(p), n01=0, n10=0, p=1.0)
    pv = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)
    return dict(n=len(p), n01=n01, n10=n10, p=pv)


def interaction(df: pd.DataFrame, cond_a: str, cond_b: str,
                v1: str, v2: str) -> dict:
    """The channel x intervention interaction.

    ``dd = (v1_a - v1_b) - (v2_a - v2_b)`` per item: how much more the first
    channel moved than the second.  This is the preregistered primary contrast
    -- a dissociation is a claim about the *difference of differences*, not
    about either channel alone.
    """
    p1 = paired(df, cond_a, cond_b, v1)
    p2 = paired(df, cond_a, cond_b, v2)
    idx = p1.index.intersection(p2.index)
    dd = (p1.loc[idx, "a"] - p1.loc[idx, "b"]) - (p2.loc[idx, "a"] - p2.loc[idx, "b"])
    if len(dd) < 6 or (dd == 0).all():
        return dict(n=len(dd), dd=float(dd.mean()) if len(dd) else 0.0, p=float("nan"))
    st, pv = stats.wilcoxon(dd)
    frame = pd.DataFrame({"dd": dd, "concept": p1.loc[idx, "concept"]})
    mean, lo, hi = cluster_bootstrap_ci(frame, "dd")
    return dict(n=int(len(dd)), dd=mean, ci_lo=lo, ci_hi=hi,
                stat=float(st), p=float(pv))


def jsonable(x):
    """Make pandas ``to_dict`` output safe to serialise.

    A grouped frame's ``to_dict`` uses tuples for multi-level keys, which
    ``json.dump`` refuses. This has cost three runs their stats file already,
    so it is fixed once, here.
    """
    if isinstance(x, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)):
                jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    try:
        import numpy as _np
        if isinstance(x, (_np.integer, _np.floating)):
            return x.item()
    except ImportError:
        pass
    return x
