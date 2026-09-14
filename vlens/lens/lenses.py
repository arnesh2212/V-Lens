# SPDX-License-Identifier: Apache-2.0
"""One interface for every lens, so all of them are scored through one path.

A lens here is a map from the residual stream at layer l to the last block's
basis, decoded by the model's own unembedding.  The logit lens is the identity
map; the tuned lens is an affine map fitted to match the model; the Jacobian
lens is the corpus-averaged Jacobian.  Anything new is the same shape.  What
differs between methods is only ``transport``; decoding, pooling and scoring
are shared, so no method is graded on another method's conveniences.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import torch


class Lens(Protocol):
    name: str
    def layers(self) -> list[int]: ...
    def transport(self, h: torch.Tensor, layer: int) -> torch.Tensor: ...


@torch.no_grad()
def decode(m, lens: Lens, h: torch.Tensor, layer: int, vocab: torch.Tensor | None = None) -> torch.Tensor:
    """Log-probabilities over the vocabulary (or a subset) for residuals ``h``."""
    z = lens.transport(h.to(m.device), layer)
    lg = m.unembed(z).float()
    if vocab is not None:
        lg = lg[..., vocab.to(lg.device)]
    return torch.log_softmax(lg, -1)


@torch.no_grad()
def readout(m, lens: Lens, acts: np.ndarray, layer: int, token_ids: list[int], *,
            pool: str = "max", chunk: int = 64) -> np.ndarray:
    """One number per item: how strongly the lens reads the given words.

    ``acts`` is ``[N, P, d]`` or ``[N, d]`` from the cache.  For each item the
    log-probability mass on ``token_ids`` is read at every position and pooled
    with ``pool`` (``max`` | ``mean`` | ``sum``).  Returned as ``[N]``.
    """
    A = torch.as_tensor(acts)
    if A.ndim == 2:
        A = A[:, None]
    ids = torch.as_tensor(sorted(set(int(t) for t in token_ids)), dtype=torch.long)
    out = []
    for s in range(0, len(A), chunk):
        lp = decode(m, lens, A[s:s + chunk].to(torch.bfloat16), layer)     # [n, P, V]
        mass = torch.logsumexp(lp[..., ids.to(lp.device)], -1)            # [n, P]
        red = {"max": mass.max(1).values, "mean": mass.mean(1), "sum": mass.sum(1)}[pool]
        out.append(red.cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def top_words(m, lens: Lens, h: torch.Tensor, layer: int, k: int = 5) -> list[list[tuple[str, float]]]:
    """The words a lens reads at each position, for inspection."""
    lp = decode(m, lens, h, layer)
    v, i = lp.topk(k, -1)
    tok = m.tokenizer
    return [[(tok.decode([int(t)]), float(p)) for p, t in zip(vr, ir)] for vr, ir in zip(v, i)]
