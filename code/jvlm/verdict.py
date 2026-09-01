# SPDX-License-Identifier: Apache-2.0
r"""Verdict Lens: how hard each token, at each layer, pushes the first answer token.

For an outcome axis :math:`m` (a scalar of the final state at the answer slot
:math:`a`) and a token class :math:`c` (image, text, or the answer slot itself),
the direction at layer :math:`l` is the corpus mean of the gradient

.. math::  u_l^{c} = \mathbb{E}_x\,\mathbb{E}_{p \in c(x)}\Big[\frac{\partial m(x)}{\partial h_l[p]}\Big],

with expansion point :math:`\mu_l^{c} = \mathbb{E}[h_l[p]]` from the same
passes.  By the chain rule this is the outcome axis pulled back through the
answer-targeted, channel-specific mean Jacobian, but the :math:`d \times d`
matrix is never formed: one backward pass of the scalar per prompt gives the
gradient at every layer and every position at once, so a five-axis fit costs
five backward passes per prompt rather than several thousand.

The reading at a token is :math:`V_l(p) = u_l^{c(p)} \cdot (h_l[p] - \mu_l^{c(p)})`,
in the axis's own units (a logit margin).  Because each prompt yields its own
direction, the averaging error is measurable: ``stability`` is the share of the
mean direction's energy that is signal rather than prompt-to-prompt noise.

Special cases run through the same code and appear as baselines: the identity
transport at the answer slot (the gradient at the last layer) is HiddenDetect's
reading; a labelled difference of means at each layer is Arditi's direction.
"""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import torch

from jlens.hooks import ActivationRecorder


def margin_gradients(model, batch: dict, axes, layers, *, answer_pos: int | None = None,
                     channels: dict | None = None):
    """``{axis: {channel: {layer: grad[d]}}}`` plus ``{channel: {layer: mean h[d]}}``.

    One forward with the graph rooted at the first read layer, then one backward
    per axis.  ``axes`` are callables ``(logits[V], h_final[d]) -> scalar``.
    """
    layers = sorted(set(int(l) for l in layers))
    S = batch["input_ids"].shape[1]
    a = S - 1 if answer_pos is None else answer_pos
    if channels is None:
        img = model.image_mask(batch["input_ids"])[0]
        ans = torch.zeros_like(img); ans[a] = True
        txt = ~img
        txt[a] = False
        channels = {"img": img, "txt": txt, "ans": ans}
    idx = {k: v.nonzero(as_tuple=True)[0] for k, v in channels.items()}
    idx = {k: v for k, v in idx.items() if len(v)}

    grads = {i: {k: {} for k in idx} for i in range(len(axes))}
    means = {k: {} for k in idx}
    with ActivationRecorder(model.layers, at=[*layers, model.n_layers - 1],
                            start_graph_at=min(layers)) as rec, torch.enable_grad():
        model.forward(batch)
        acts = [rec.activations[l] for l in layers]
        h_final = rec.activations[model.n_layers - 1][0, a]
        logits = model.unembed(h_final[None])[0]
        for l, A in zip(layers, acts):
            for k, pos in idx.items():
                means[k][l] = A[0, pos.to(A.device)].detach().float().mean(0).cpu()
        for i, ax in enumerate(axes):
            g = torch.autograd.grad(ax(logits, h_final), acts,
                                    retain_graph=(i < len(axes) - 1))
            for l, gl in zip(layers, g):
                for k, pos in idx.items():
                    grads[i][k][l] = gl[0, pos.to(gl.device)].float().mean(0).cpu()
    return grads, means


@dataclasses.dataclass
class Fit:
    """Accumulated directions for one prompt family."""
    layers: list
    axes: list                                   # names
    n: int = 0
    _g: dict = dataclasses.field(default_factory=dict)      # (axis, ch, l) -> sum
    _g2: dict = dataclasses.field(default_factory=dict)     # (axis, ch, l) -> sum of squares
    _mu: dict = dataclasses.field(default_factory=dict)     # (ch, l) -> sum
    _per: dict = dataclasses.field(default_factory=dict)    # (axis, ch, l) -> list of unit dirs (for MoM)

    def add(self, grads, means):
        for i in grads:
            for k in grads[i]:
                for l, v in grads[i][k].items():
                    key = (self.axes[i], k, l)
                    self._g[key] = self._g.get(key, 0) + v
                    self._g2[key] = self._g2.get(key, 0) + (v * v).sum()
                    self._per.setdefault(key, []).append(v / v.norm().clamp_min(1e-8))
        for k in means:
            for l, v in means[k].items():
                self._mu[(k, l)] = self._mu.get((k, l), 0) + v
        self.n += 1

    def direction(self, axis: str, channel: str, layer: int, *, unit: bool = False) -> torch.Tensor:
        u = self._g[(axis, channel, int(layer))] / max(self.n, 1)
        return u / u.norm().clamp_min(1e-8) if unit else u

    def mu(self, channel: str, layer: int) -> torch.Tensor:
        return self._mu[(channel, int(layer))] / max(self.n, 1)

    def stability(self, axis: str, channel: str, layer: int) -> float:
        """``||E v||^2 / E ||v||^2``: 1 when every prompt agrees, ~1/n when none do."""
        key = (axis, channel, int(layer))
        mean_sq = (self._g[key] / self.n).norm() ** 2
        e_sq = self._g2[key] / self.n
        return float(mean_sq / e_sq.clamp_min(1e-12))

    def median_of_means(self, axis: str, channel: str, layer: int, k: int = 5) -> torch.Tensor:
        per = torch.stack(self._per[(axis, channel, int(layer))])
        chunks = torch.chunk(per, k)
        m = torch.stack([c.mean(0) for c in chunks if len(c)])
        med = m.median(0).values
        return med / med.norm().clamp_min(1e-8)

    def channels(self):
        return sorted({k for (_, k, _) in self._g})

    def to_dict(self) -> dict:
        return dict(layers=self.layers, axes=self.axes, n=self.n,
                    directions={f"{a}|{c}|{l}": self.direction(a, c, l).half()
                                for (a, c, l) in self._g},
                    mu={f"{c}|{l}": self.mu(c, l).half() for (c, l) in self._mu},
                    stability={f"{a}|{c}|{l}": self.stability(a, c, l) for (a, c, l) in self._g})


def fit(model, batches, axes, layers, *, every: int = 20) -> Fit:
    """Accumulate directions over an iterable of prepared batches."""
    F = Fit(sorted(set(int(l) for l in layers)), [ax.name for ax in axes])
    for i, b in enumerate(batches):
        try:
            g, mu = margin_gradients(model, b, axes, F.layers)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
        F.add(g, mu)
        if (i + 1) % every == 0:
            print(f"  fit {i + 1} prompts", flush=True)
    return F


# ---------------------------------------------------------------- reading
class VerdictLens:
    """A fitted lens: directions and expansion points per (axis, channel, layer)."""

    name = "Verdict Lens"

    def __init__(self, d: dict, meta: dict | None = None):
        self.layers = [int(l) for l in d["layers"]]
        self.axes = list(d["axes"])
        self.u = {k: v.float() for k, v in d["directions"].items()}
        self.mu = {k: v.float() for k, v in d["mu"].items()}
        self.stability = dict(d.get("stability", {}))
        self.meta = meta or {}

    @classmethod
    def from_fit(cls, F: Fit, meta=None):
        return cls(F.to_dict(), meta)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(dict(layers=self.layers, axes=self.axes,
                        directions={k: v.half() for k, v in self.u.items()},
                        mu={k: v.half() for k, v in self.mu.items()},
                        stability=self.stability, meta=self.meta), path)
        with open(path.replace(".pt", ".json"), "w") as f:
            json.dump(dict(layers=self.layers, axes=self.axes, meta=self.meta,
                           stability=self.stability), f, indent=1)
        return path

    @classmethod
    def load(cls, path: str):
        d = torch.load(path, map_location="cpu")
        return cls(d, d.get("meta"))

    def dir(self, axis: str, channel: str, layer: int) -> torch.Tensor:
        return self.u[f"{axis}|{channel}|{int(layer)}"]

    def centre(self, channel: str, layer: int) -> torch.Tensor:
        return self.mu[f"{channel}|{int(layer)}"]

    def read(self, h: np.ndarray | torch.Tensor, axis: str, channel: str, layer: int,
             *, centred: bool = True) -> np.ndarray:
        """``V = u . (h - mu)`` for ``h`` of shape ``[..., d]``; returns ``[...]``."""
        H = torch.as_tensor(h).float()
        u = self.dir(axis, channel, layer)
        if centred:
            H = H - self.centre(channel, layer)
        return (H @ u).numpy()

    def reachability(self, axis: str, layer: int) -> dict:
        """``||u||`` per channel: how much of the axis a unit push at that class reaches."""
        return {c: float(self.dir(axis, c, layer).norm()) for c in ("img", "txt", "ans")
                if f"{axis}|{c}|{int(layer)}" in self.u}

    def alignment(self, axis: str, layer: int, a: str = "img", b: str = "txt") -> float:
        ua, ub = self.dir(axis, a, layer), self.dir(axis, b, layer)
        return float(ua @ ub / (ua.norm() * ub.norm()).clamp_min(1e-8))


def cosine(u: torch.Tensor, v: torch.Tensor) -> float:
    return float((u.float() @ v.float()) / (u.norm() * v.norm()).clamp_min(1e-8))


def dim_direction(H: np.ndarray, y: np.ndarray) -> torch.Tensor:
    """Arditi's difference of means at one layer: the labelled baseline direction."""
    H = torch.as_tensor(H).float(); y = torch.as_tensor(y).bool()
    r = H[y].mean(0) - H[~y].mean(0)
    return r / r.norm().clamp_min(1e-8)


# ---------------------------------------------------------------- calibration
def calibrate(V_pred: np.ndarray, delta_m: np.ndarray, groups: np.ndarray, *,
              holdout: float = 0.3, seed: int = 0) -> dict:
    """Fit ``delta_m ~ a * V_pred + b`` on a split of whole images; held-out R^2.

    ``V_pred`` is the lens's predicted change (``u . delta`` for the patched
    sites), ``delta_m`` the measured change in the axis under the same patch,
    ``groups`` the image each pair came from so no image is on both sides.
    """
    rng = np.random.default_rng(seed)
    g = np.unique(groups)
    test_g = set(rng.permutation(g)[: max(1, int(len(g) * holdout))])
    te = np.array([x in test_g for x in groups]); tr = ~te
    X, Y = np.asarray(V_pred, float), np.asarray(delta_m, float)
    A = np.vstack([X[tr], np.ones(tr.sum())]).T
    a, b = np.linalg.lstsq(A, Y[tr], rcond=None)[0]
    pred = a * X[te] + b
    ss = ((Y[te] - Y[te].mean()) ** 2).sum()
    r2 = float(1 - ((Y[te] - pred) ** 2).sum() / ss) if ss > 0 else float("nan")
    rho = float(np.corrcoef(X[te], Y[te])[0, 1]) if te.sum() > 2 else float("nan")
    return dict(a=float(a), b=float(b), r2_heldout=r2, spearman_like_r=rho,
                n_fit=int(tr.sum()), n_test=int(te.sum()))
