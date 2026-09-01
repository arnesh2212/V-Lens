# SPDX-License-Identifier: Apache-2.0
"""The preview: one small map per layer from the answer spot to its final state.

The writer direction (verdict.py) is the gradient of an outcome with respect
to a token: the coordinate along which a push changes the answer.  It is not
the coordinate along which the answer's evidence is stored at mid depth, and
it reads that evidence poorly.  The reader is the complementary transport: an
affine map  A_l h_l[a] + b_l  fitted, without any labels, to predict the final
residual at the answer slot from the residual at layer l.  Any outcome axis
defined at the final layer can then be read at layer l through it.

Fitted by ridge regression on answer-slot residuals from an unlabelled corpus
of prompts.  Label-free with respect to the task; the only labels anywhere in
the pipeline are in the outcome axis, and that axis is fitted once at the final
layer (behaviour-anchored) or defined by token sets (no labels at all).

This is the tuned lens's construction, targeted at the answer slot of a VLM
and paired with a behaviour axis.  It is stated as such.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from jvlm import tuned


class Reader:
    def __init__(self, maps: dict, meta: dict | None = None):
        self.maps = {int(l): (A.float(), b.float()) for l, (A, b) in maps.items()}
        self.meta = meta or {}

    @property
    def layers(self):
        return sorted(self.maps)

    def to_final(self, h: np.ndarray | torch.Tensor, layer: int) -> np.ndarray:
        """Predicted final-layer answer-slot state from the layer-``layer`` state, ``[..., d]``."""
        A, b = self.maps[int(layer)]
        H = torch.as_tensor(h).float()
        return (H @ A.T + b).numpy()

    def read(self, h, layer: int, axis: np.ndarray | torch.Tensor) -> np.ndarray:
        """``axis . to_final(h)`` for an axis defined in the final residual space."""
        r = torch.as_tensor(axis).float().numpy()
        return self.to_final(h, layer) @ r

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(dict(maps={l: (A.half(), b.half()) for l, (A, b) in self.maps.items()}, meta=self.meta), path)
        with open(path.replace(".pt", ".json"), "w") as f:
            json.dump(self.meta, f, indent=1, default=float)
        return path

    @classmethod
    def load(cls, path: str):
        d = torch.load(path, map_location="cpu")
        return cls(d["maps"], d.get("meta"))


def fit(ANS: np.ndarray, layers, *, final_layer: int | None = None, lam: float = 1e-1,
        meta: dict | None = None) -> Reader:
    """Fit from ``ANS`` of shape ``[N, n_layers, d]`` (answer-slot residual at every block)."""
    L = ANS.shape[1] - 1 if final_layer is None else int(final_layer)
    maps = {}
    for l in sorted(set(int(x) for x in layers)):
        maps[l] = tuned.fit_ridge(ANS[:, l], ANS[:, L], lam=lam)
    m = dict(meta or {}, kind="ridge", lam=lam, n=int(ANS.shape[0]), final_layer=L, layers=sorted(maps))
    return Reader(maps, m)


def fit_kfold_scores(ANS: np.ndarray, y: np.ndarray, layers, *, folds: int = 5, lam: float = 1e-1,
                     seed: int = 0, axis_fn=None) -> dict[int, np.ndarray]:
    """Held-out reader scores per layer with the axis fitted on the training fold only.

    ``axis_fn(H_final_train, y_train) -> r`` defaults to the difference of means
    (behaviour-anchored).  Returns ``{layer: scores[N]}``.  For protocols where
    the reader must be fitted on a disjoint corpus, use :func:`fit` instead.
    """
    from jvlm import verdict
    N = len(y); rng = np.random.default_rng(seed); parts = np.array_split(rng.permutation(N), folds)
    L = ANS.shape[1] - 1
    axis_fn = axis_fn or (lambda H, yy: verdict.dim_direction(H, yy).numpy())
    out = {int(l): np.zeros(N) for l in layers}
    for te in parts:
        tr = np.setdiff1d(np.arange(N), te)
        r = axis_fn(ANS[tr, L], y[tr])
        for l in layers:
            A, b = tuned.fit_ridge(ANS[tr, l], ANS[tr, L], lam=lam)
            out[int(l)][te] = (ANS[te, l] @ A.numpy().T + b.numpy()) @ r
    return out
