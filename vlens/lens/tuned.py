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


def fit_ridge(H_l: np.ndarray, H_L: np.ndarray, lam: float = 1e-2,
              device: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form affine map ``H_l -> H_L`` with an L2 penalty on the weights.

    Both inputs are ``[n, d]``; positions are flattened by the caller.  The
    penalty is scaled by the mean squared feature so it means the same thing
    on a model whose residuals run to hundreds as on one where they run to a
    few.

    ``device`` can move the solve to the GPU; the default keeps it on the CPU.
    """
    device = device or "cpu"
    X = torch.as_tensor(np.asarray(H_l), dtype=torch.float64).to(device)
    Y = torch.as_tensor(np.asarray(H_L), dtype=torch.float64).to(device)
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
        # the solve is in double precision; the product that follows is well
        # conditioned and is done in single
        Z = torch.linalg.solve(K + lam_eff * torch.eye(n, dtype=K.dtype, device=K.device), Xc)
        A = (Yc.float().T @ Z.float()).double()      # [d_out, d_in]
    else:
        G = Xc.T @ Xc
        lam_eff = lam * G.diagonal().mean()
        A = torch.linalg.solve(G + lam_eff * torch.eye(d, dtype=G.dtype, device=G.device), Xc.T @ Yc).T   # [d_out, d_in]
    b = my - A @ mx
    return A.float().cpu(), b.float().cpu()


def fit_ridge_cv(H_l: "np.ndarray", H_L: "np.ndarray",
                 lams=(1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
                 frac: float = 0.25, seed: int = 0, device: str | None = None,
                 m=None, gains=(1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0)):
    """``(A, b, lam)`` with the penalty chosen inside the rows given.

    A fixed penalty is a hyperparameter nobody chose.  A slice of the training
    rows is held back, the penalty that reconstructs it best is taken, and the
    map is then refitted on all of them, so nothing outside the training rows
    is ever consulted.
    """
    X = np.asarray(H_l); Y = np.asarray(H_L)
    n = len(X)
    if n < 8 or len(lams) == 1:
        A, b = fit_ridge(X, Y, lam=float(lams[0]), device=device)
        return A, b, float(lams[0])
    idx = np.random.default_rng(seed).permutation(n)
    k = max(int(round(frac * n)), 2)
    dev, tr = idx[:k], idx[k:]
    Xd = torch.as_tensor(np.asarray(X[dev], np.float32))
    Yd = torch.as_tensor(np.asarray(Y[dev], np.float32))
    best, best_e = float(lams[0]), None
    for lam in lams:
        A, b = fit_ridge(X[tr], Y[tr], lam=float(lam), device=device)
        e = float(((Xd @ A.T + b) - Yd).pow(2).mean())
        if np.isfinite(e) and (best_e is None or e < best_e):
            best, best_e = float(lam), e
    A, b = fit_ridge(X, Y, lam=best, device=device)
    if m is not None and gains:
        # the same gain the shared fitter chooses, on the same held-out rows,
        # so V-Read is one instrument wherever it appears
        A_t, b_t = fit_ridge(X[tr], Y[tr], lam=best, device=device)
        mu = torch.as_tensor(np.asarray(Y, np.float32)).mean(0)
        best_g, best_kl = 1.0, None
        for g in gains:
            v = kl_on_states(m, X[dev], Y[dev], float(g) * A_t,
                             float(g) * b_t + (1.0 - float(g)) * mu)
            if np.isfinite(v) and (best_kl is None or v < best_kl):
                best_g, best_kl = float(g), v
        A, b = best_g * A, best_g * b + (1.0 - best_g) * mu
        return A, b, best
    return A, b, best


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


@torch.no_grad()
def kl_on_states(m, X: "np.ndarray", Y: "np.ndarray", A: torch.Tensor, b: torch.Tensor,
                 chunk: int = 512) -> float:
    """Mean ``KL(p_model || p_lens)`` for one affine map on the given states."""
    import numpy as np
    Xt = torch.as_tensor(np.asarray(X, np.float32))
    Yt = torch.as_tensor(np.asarray(Y, np.float32))
    A = A.float(); b = b.float()
    tot, n = 0.0, 0
    for s in range(0, len(Xt), chunk):
        xs, ys = Xt[s: s + chunk].to(m.device), Yt[s: s + chunk].to(m.device)
        ref = torch.log_softmax(m.unembed(ys.to(torch.bfloat16)).float(), -1)
        lg = torch.log_softmax(
            m.unembed((xs @ A.T.to(m.device) + b.to(m.device)).to(torch.bfloat16)).float(), -1)
        tot += float((ref.exp() * (ref - lg)).sum(-1).sum()); n += len(xs)
    return tot / max(n, 1)


def pick_kl_lr(m, X, Y, Xd, Yd, *, lrs=(1.0, 0.1, 0.01), probe_steps: int = 40,
               d_model: int | None = None, log=None, reference: bool = False) -> float:
    """The tuned lens learning rate, chosen on held-out states by its own objective.

    V-Read's penalty is chosen on the same held-out states by squared error,
    which is what a ridge fit minimises; each instrument is therefore searched
    on its own criterion rather than one of them being handed a fixed value.
    """
    d = d_model or Y.shape[-1]
    best, best_kl = lrs[0], None
    for lr in lrs:
        A, b = refine_kl_states(m, X, Y, torch.eye(d), torch.zeros(d),
                                steps=probe_steps, lr=lr, reference=reference)
        v = kl_on_states(m, Xd, Yd, A, b)
        if log:
            log(f"    tuned lens lr {lr:<6} -> held-out KL {v:.4f}")
        if best_kl is None or v < best_kl:
            best, best_kl = lr, v
    return best


def refine_kl_states(m, X: "np.ndarray", Y: "np.ndarray", A0: torch.Tensor, b0: torch.Tensor, *,
                     steps: int = 250, lr: float = 0.1, batch: int = 256, seed: int = 0,
                     momentum: float = 0.9, weight_decay: float = 1e-3, clip: float = 1.0,
                     decay: bool = True, reference: bool = False
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Tuned-lens objective on one block's states, with the published optimiser.

    Minimises ``KL(p_model || p_lens)`` by SGD with Nesterov momentum, weight
    decay, gradient clipping and a linear decay, starting from ``(A0, b0)``.

    ``reference=True`` trains it the way the reference implementation does
    (``baselines.Translator``): the map is ``A0 + W`` with ``W`` and the bias
    learnt from zero, so weight decay pulls towards ``(A0, b0)`` (the identity
    for a tuned lens) rather than towards the zero map, and ``lr`` is the
    reference ``lr_scale``, an SGD step of ``lr * (1 - momentum)``. The default
    keeps the older behaviour for the callers that already ran with it.
    """
    import numpy as np
    Xt = torch.as_tensor(np.asarray(X, np.float32)).to(m.device)
    Yt = torch.as_tensor(np.asarray(Y, np.float32)).to(m.device)
    with torch.no_grad():
        ref = torch.log_softmax(m.unembed(Yt.to(torch.bfloat16)).float(), -1).exp()
    if reference:
        A_base = A0.clone().float().to(m.device)
        b_base = b0.clone().float().to(m.device)
        A = torch.nn.Parameter(torch.zeros_like(A_base))
        b = torch.nn.Parameter(torch.zeros_like(b_base))
        step = lr * (1.0 - momentum)
    else:
        A_base = b_base = None
        A = torch.nn.Parameter(A0.clone().float().to(m.device))
        b = torch.nn.Parameter(b0.clone().float().to(m.device))
        step = lr
    full = (lambda: (A_base + A, b_base + b)) if reference else (lambda: (A, b))
    opt = torch.optim.SGD([A, b], lr=step, momentum=momentum, nesterov=True,
                          weight_decay=weight_decay)
    sched = (torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / steps))
             if decay else None)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(Xt)
    for _ in range(steps):
        j = torch.randperm(n, generator=g)[:batch].to(m.device)
        Af, bf = full()
        lg = torch.log_softmax(m.unembed((Xt[j] @ Af.T + bf).to(torch.bfloat16)).float(), -1)
        loss = (ref[j] * (ref[j].clamp_min(1e-9).log() - lg)).sum(-1).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([A, b], clip)
        opt.step()
        if sched is not None:
            sched.step()
    Af, bf = full()
    return Af.detach().float().cpu(), bf.detach().float().cpu()
