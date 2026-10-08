# SPDX-License-Identifier: Apache-2.0
"""Linear probes, per layer, with their floors measured rather than assumed.

A trained probe is the baseline a zero-shot lens has to beat before it has
shown anything: it is the best linear reader of a layer that labels can buy.
So every place a lens is scored on a task, a probe is scored on the same
activations, the same folds and the same metric, and the two numbers sit side
by side.  This module owns that side of the comparison.

Everything runs on the CPU.  Fitting a few hundred logistic regressions is
minutes of CPU and would otherwise hold a model resident on the accelerator.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclasses.dataclass
class ProbeResult:
    layer: int
    n: int
    metric: str                 # "auroc" for two classes, "accuracy" otherwise
    score: float
    lo: float                   # bootstrap over folds' held-out predictions
    hi: float
    floor_shuffled: float       # labels permuted, same pipeline
    floor_majority: float       # constant prediction
    leak_random: float          # same pipeline on Gaussian features


def _cv_predictions(X, y, folds, seed, C):
    """Held-out probabilities (2-class) or labels (k-class) for every row."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    k = len(np.unique(y))
    out = np.zeros(len(y)) if k == 2 else np.zeros(len(y), dtype=int)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=600, C=C)
        clf.fit(sc.transform(X[tr]), y[tr])
        Xte = sc.transform(X[te])
        out[te] = clf.predict_proba(Xte)[:, 1] if k == 2 else clf.predict(Xte)
    return out


def _score(pred, y):
    if len(np.unique(y)) == 2:
        return float(roc_auc_score(y, pred)), "auroc"
    return float((pred == y).mean()), "accuracy"


def _bootstrap(pred, y, seed, reps=1000):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(_score(pred[idx], y[idx])[0])
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def probe_layer(layer: int, X: np.ndarray, y: np.ndarray, *, folds: int = 5,
                seed: int = 0, C: float = 0.1, shuffles: int = 5) -> ProbeResult:
    """One layer, one task: the probe and its three floors."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if X.ndim > 2:                       # [N, P, d] -> mean over positions
        X = X.reshape(len(X), -1, X.shape[-1]).mean(1)
    pred = _cv_predictions(X, y, folds, seed, C)
    score, metric = _score(pred, y)
    lo, hi = _bootstrap(pred, y, seed)

    rng = np.random.default_rng(seed + 1)
    shuf = []
    for _ in range(shuffles):
        ys = rng.permutation(y)
        shuf.append(_score(_cv_predictions(X, ys, folds, seed, C), ys)[0])
    _, counts = np.unique(y, return_counts=True)
    maj = counts.max() / counts.sum()
    if metric == "auroc":
        maj = 0.5                        # a constant score has no ranking
    Xr = rng.normal(size=X.shape).astype(np.float32)
    leak = _score(_cv_predictions(Xr, y, folds, seed, C), y)[0]
    return ProbeResult(layer=int(layer), n=int(len(y)), metric=metric,
                       score=score, lo=lo, hi=hi,
                       floor_shuffled=float(np.mean(shuf)),
                       floor_majority=float(maj), leak_random=float(leak))


def probe_layers(acts: dict[int, np.ndarray], y: np.ndarray, *, jobs: int = 8,
                 **kw) -> list[ProbeResult]:
    """Every layer in ``acts`` (``{layer: [N, d] or [N, P, d]}``), in parallel."""
    layers = sorted(acts)
    return Parallel(n_jobs=jobs)(
        delayed(probe_layer)(l, acts[l], y, **kw) for l in layers)


def readout_score(scores: np.ndarray, y: np.ndarray, *, seed: int = 0,
                  shuffles: int = 200) -> dict:
    """Grade a zero-shot scalar reading on the probe's own metric.

    ``scores`` is one number per item from a lens (no labels used to make it).
    Two classes: AUROC with a bootstrap interval and a shuffled-label floor, so
    the lens and the probe are read off the same axis.
    """
    scores = np.asarray(scores, float)
    y = np.asarray(y)
    ok = np.isfinite(scores)
    scores, y = scores[ok], y[ok]
    au = float(roc_auc_score(y, scores))
    lo, hi = _bootstrap(scores, y, seed)
    rng = np.random.default_rng(seed)
    null = np.array([roc_auc_score(rng.permutation(y), scores) for _ in range(shuffles)])
    return dict(auroc=au, lo=lo, hi=hi, n=int(len(y)),
                floor_mean=float(null.mean()), floor_p95=float(np.percentile(null, 95)))


def table(results: Sequence[ProbeResult]) -> str:
    """A plain-text grid, layers down."""
    if not results:
        return "(no layers)"
    m = results[0].metric
    head = f"{'layer':>6} {m:>9} {'95% CI':>17} {'shuffled':>9} {'majority':>9} {'random-X':>9}"
    rows = [head]
    for r in results:
        rows.append(f"{r.layer:>6} {r.score:>9.3f} [{r.lo:.3f}, {r.hi:.3f}] "
                    f"{r.floor_shuffled:>9.3f} {r.floor_majority:>9.3f} {r.leak_random:>9.3f}")
    return "\n".join(rows)
