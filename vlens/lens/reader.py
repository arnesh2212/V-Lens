# SPDX-License-Identifier: Apache-2.0
"""The read: one small map per layer from the answer spot to its final state.

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
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from vlens.lens import tuned


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
        rescale: bool = True, meta: dict | None = None) -> Reader:
    """Fit from ``ANS`` of shape ``[N, n_layers, d]`` (answer-spot residual at every block).

    Ridge shrinks its prediction toward the mean, which is the right thing for squared
    error and the wrong thing for reading a score off the prediction: every margin comes
    out compressed toward the corpus average, so a per-item trace looks flat even when
    the ranking across items is right.  ``rescale`` fixes that with one scalar per layer,
    chosen so the predicted states vary as much as the real final states do.  It is a
    positive affine map, so nothing about the ranking changes; only the units do.
    """
    L = ANS.shape[1] - 1 if final_layer is None else int(final_layer)
    Y = ANS[:, L]
    mu_y = torch.as_tensor(Y.mean(0)).float()
    var_y = float(((torch.as_tensor(Y).float() - mu_y) ** 2).sum(-1).mean())
    maps, scales = {}, {}
    for l in sorted(set(int(x) for x in layers)):
        A, b = tuned.fit_ridge(ANS[:, l], Y, lam=lam)
        if rescale:
            P = torch.as_tensor(ANS[:, l]).float() @ A.T + b
            mu_p = P.mean(0)
            var_p = float(((P - mu_p) ** 2).sum(-1).mean())
            s = float((var_y / max(var_p, 1e-12)) ** 0.5)
            A, b = s * A, s * (b - mu_p) + mu_y
            scales[l] = s
        maps[l] = (A, b)
    m = dict(meta or {}, kind="ridge+scale" if rescale else "ridge", lam=lam, n=int(ANS.shape[0]),
             final_layer=L, layers=sorted(maps), scales={str(k): v for k, v in scales.items()})
    return Reader(maps, m)
