# SPDX-License-Identifier: Apache-2.0
"""The tuned lens, fitted from a cache: one affine map per layer to the last block.

Belrose et al. (2023) train, for each layer l, an affine map A_l h + b_l whose
unembedding matches the model's own final distribution.  This is the strongest
lens baseline a new method has to clear, and the codebase had none.

Two fits are offered.  ``ridge`` is closed form: the least-squares map from
the layer-l residual to the last-block residual with an L2 penalty, which is
deterministic, seconds on the CPU, and decoded through the model's own
unembedding at read time.  ``kl`` then refines that map by a few epochs of
gradient descent on the KL divergence between the model's final distribution
and the lens's, which is the objective of the original paper; it needs the
unembedding on the same device and is run on the GPU when one is free.

A fitted lens is a dict of ``{layer: (A, b)}`` saved with the cache it came
from, so a reader can always see what it was fitted on.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch


class TunedLens:
    def __init__(self, maps: dict[int, tuple[torch.Tensor, torch.Tensor]], meta: dict | None = None):
        self.maps = {int(k): (A.float(), b.float()) for k, (A, b) in maps.items()}
        self.meta = meta or {}

    @property
    def layers(self) -> list[int]:
        return sorted(self.maps)

    def transport(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Carry residuals at ``layer`` (``[..., d]``) to the last block."""
        A, b = self.maps[int(layer)]
        A, b = A.to(h.device, h.dtype), b.to(h.device, h.dtype)
        return h @ A.T + b

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"maps": {k: (A.half(), b.half()) for k, (A, b) in self.maps.items()},
                    "meta": self.meta}, path)
        with open(path.replace(".pt", ".json"), "w") as f:
            json.dump(self.meta, f, indent=1)
        return path

    @classmethod
    def load(cls, path: str) -> "TunedLens":
        d = torch.load(path, map_location="cpu")
        return cls(d["maps"], d.get("meta"))


def fit_ridge(H_l: np.ndarray, H_L: np.ndarray, lam: float = 1e-2) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form affine map ``H_l -> H_L`` with an L2 penalty on the weights.

    Both inputs are ``[n, d]``; positions are flattened by the caller.  The
    penalty is scaled by the mean squared feature so it means the same thing
    on a model whose residuals run to hundreds as on one where they run to a
    few.
    """
    X = torch.as_tensor(H_l, dtype=torch.float64)
    Y = torch.as_tensor(H_L, dtype=torch.float64)
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    n, d = X.shape
    if n < d:
        # fewer rows than dimensions: solve the n x n system instead of the
        # d x d one.  Xc (Xc^T Xc + cI)^-1 = (Xc Xc^T + cI)^-1 Xc, so this is
        # the same map, and trace(Xc^T Xc) = trace(Xc Xc^T) keeps the same
        # penalty scale.  A few hundred states against a few thousand
        # dimensions is the usual case here, and the d x d solve dominates the
        # cost of a fit.
        K = Xc @ Xc.T
        lam_eff = lam * K.diagonal().sum() / d
        A = (Yc.T @ torch.linalg.solve(K + lam_eff * torch.eye(n, dtype=K.dtype), Xc))   # [d_out, d_in]
    else:
        G = Xc.T @ Xc
        lam_eff = lam * G.diagonal().mean()
        A = torch.linalg.solve(G + lam_eff * torch.eye(d, dtype=G.dtype), Xc.T @ Yc).T   # [d_out, d_in]
    b = my - A @ mx
    return A.float(), b.float()


def fit_from_cache(acts: dict[int, np.ndarray], final_layer: int, *, lam: float = 1e-2,
                   max_rows: int = 120_000, seed: int = 0, layers=None) -> TunedLens:
    """One ridge map per cached layer, to the cached last block."""
    rng = np.random.default_rng(seed)
    Y = acts[final_layer]
    Y = Y.reshape(-1, Y.shape[-1])
    idx = rng.permutation(len(Y))[:max_rows]
    maps = {}
    for l in sorted(layers or [k for k in acts if k != final_layer]):
        X = acts[l].reshape(-1, acts[l].shape[-1])
        maps[l] = fit_ridge(X[idx], Y[idx], lam)
    return TunedLens(maps, dict(kind="ridge", lam=lam, rows=int(len(idx)),
                                final_layer=int(final_layer)))


@torch.no_grad()
def kl_to_model(lens: TunedLens, m, acts: dict[int, np.ndarray], final_layer: int,
                layers=None, rows: int = 4000, seed: int = 0) -> dict[int, float]:
    """Mean KL(model || lens) per layer, on the model's own unembedding.

    The number the tuned-lens paper reports; lower is better, and the logit
    lens is the identity map's value.  Returned in nats.
    """
    rng = np.random.default_rng(seed)
    Y = torch.as_tensor(acts[final_layer].reshape(-1, acts[final_layer].shape[-1]))
    idx = torch.as_tensor(rng.permutation(len(Y))[:rows])
    ref = torch.log_softmax(m.unembed(Y[idx].to(m.device)).float(), -1)
    out = {}
    for l in sorted(layers or lens.layers):
        X = torch.as_tensor(acts[l].reshape(-1, acts[l].shape[-1]))[idx].to(m.device)
        lg = torch.log_softmax(m.unembed(lens.transport(X, l)).float(), -1)
        out[l] = float((ref.exp() * (ref - lg)).sum(-1).mean())
    return out


def refine_kl(lens: TunedLens, m, acts: dict[int, np.ndarray], final_layer: int, *,
              epochs: int = 2, lr: float = 1e-3, batch: int = 512, rows: int = 60_000,
              seed: int = 0) -> TunedLens:
    """A few epochs of KL descent from the ridge solution, on the GPU."""
    rng = np.random.default_rng(seed)
    Y = torch.as_tensor(acts[final_layer].reshape(-1, acts[final_layer].shape[-1]))
    idx = rng.permutation(len(Y))[:rows]
    Y = Y[idx].to(m.device)
    with torch.no_grad():
        ref = torch.log_softmax(m.unembed(Y).float(), -1).exp()
    maps = {}
    for l in lens.layers:
        X = torch.as_tensor(acts[l].reshape(-1, acts[l].shape[-1]))[idx].to(m.device)
        A0, b0 = lens.maps[l]
        A = torch.nn.Parameter(A0.clone().to(m.device)); b = torch.nn.Parameter(b0.clone().to(m.device))
        opt = torch.optim.Adam([A, b], lr=lr)
        for _ in range(epochs):
            perm = torch.randperm(len(X), device=m.device)
            for s in range(0, len(X), batch):
                j = perm[s:s + batch]
                lg = torch.log_softmax(m.unembed((X[j].float() @ A.T + b).to(X.dtype)).float(), -1)
                loss = (ref[j] * (ref[j].clamp_min(1e-9).log() - lg)).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        maps[l] = (A.detach().cpu(), b.detach().cpu())
    meta = dict(lens.meta, kind="ridge+kl", epochs=epochs, lr=lr)
    return TunedLens(maps, meta)
