# SPDX-License-Identifier: Apache-2.0
"""AUROC, fast enough to bootstrap.

Every headline number here carries a percentile interval over items, which
means a few hundred resamples for every instrument at every block for every
family. Done through a general-purpose implementation that is tens of thousands
of calls and minutes of the machine doing nothing else.

AUROC is the Mann-Whitney statistic, so it is the mean rank of the positives
with the trivial part subtracted, and ranks are one sort. ``boot`` does the
resampling in one vectorised pass.
"""
from __future__ import annotations

import numpy as np


def auroc(y, s) -> float:
    """Area under the ROC curve, ties averaged, from ranks."""
    y = np.asarray(y).ravel()
    s = np.asarray(s, float).ravel()
    if y.size == 0 or not np.isfinite(s).all():
        return float("nan")
    pos = y > 0
    n1 = int(pos.sum()); n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, float)
    ranks[order] = np.arange(1, s.size + 1, dtype=float)
    # average the ranks of tied scores, which is what the definition asks for
    ss = s[order]
    i = 0
    while i < ss.size:
        j = i + 1
        while j < ss.size and ss[j] == ss[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def boot(y, s, reps: int = 300, seed: int = 0, ci: float = 95.0):
    """The point estimate and a percentile interval over items.

    Resampling is done in one array of indices rather than a Python loop, and
    each resample is scored by the same rank formula.
    """
    y = np.asarray(y).ravel()
    s = np.asarray(s, float).ravel()
    pt = auroc(y, s)
    if not np.isfinite(pt) or reps <= 0:
        return pt, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = y.size
    idx = rng.integers(0, n, size=(reps, n))
    ys = y[idx] > 0
    ss = s[idx]
    n1 = ys.sum(1).astype(float)
    n0 = n - n1
    order = np.argsort(ss, axis=1, kind="mergesort")
    ranks = np.empty_like(ss, dtype=float)
    rows = np.arange(reps)[:, None]
    ranks[rows, order] = np.arange(1, n + 1, dtype=float)[None, :]
    # ties are averaged as in ``auroc``
    sorted_s = np.take_along_axis(ss, order, axis=1)
    same = np.zeros_like(sorted_s, dtype=bool)
    same[:, 1:] = sorted_s[:, 1:] == sorted_s[:, :-1]
    if same.any():
        pos = np.arange(1, n + 1, dtype=float)[None, :].repeat(reps, 0)
        for r in np.nonzero(same.any(axis=1))[0]:
            i = 0
            while i < n:
                j = i + 1
                while j < n and sorted_s[r, j] == sorted_s[r, i]:
                    j += 1
                if j - i > 1:
                    pos[r, i:j] = (i + j + 1) / 2.0
                i = j
            ranks[r, order[r]] = pos[r]
    r1 = np.where(ys, ranks, 0.0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return pt, float("nan"), float("nan")
    lo = (100.0 - ci) / 2.0
    return pt, float(np.percentile(vals, lo)), float(np.percentile(vals, 100.0 - lo))
