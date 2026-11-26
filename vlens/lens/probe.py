# SPDX-License-Identifier: Apache-2.0
"""The supervised reference every label-free instrument is measured against.

One place to define it, because it is fitted once per fold per block in four
different experiments and they all have to be fitting the same thing. A
residual stream has far more dimensions than a fold has examples, so the
problem is separable and the solver converges quickly; the iteration cap is
there to stop it grinding on the last decimal of a margin that is already
decided.
"""
from __future__ import annotations

import numpy as np


def fit(X, y, *, C=None, max_iter: int = 1000, seed: int = 0,
        grid=(0.01, 0.1, 1.0, 10.0)):
    """A logistic probe on ``X``, standardised, with ``C`` chosen inside the
    training rows by cross-validation unless one is given."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, np.float32); y = np.asarray(y)
    mk = lambda c: make_pipeline(StandardScaler(),
                                 LogisticRegression(solver="lbfgs", C=c,
                                                    max_iter=max_iter, tol=1e-4,
                                                    random_state=seed))
    if C is None and len(y) >= 40 and len(set(y.tolist())) > 1:
        cv = StratifiedKFold(3, shuffle=True, random_state=seed)
        best, bc = -1.0, grid[len(grid) // 2]
        for c in grid:
            try:
                sc = float(cross_val_score(mk(c), X, y, cv=cv, scoring="roc_auc").mean())
            except Exception:  # noqa: BLE001
                continue
            if sc > best:
                best, bc = sc, c
        C = bc
    return mk(C if C is not None else 1.0).fit(X, y)


def scores(clf, X):
    return clf.decision_function(np.asarray(X, np.float32))
