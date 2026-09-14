# SPDX-License-Identifier: Apache-2.0
"""Adding a direction at the answer spot during the prefill."""
from __future__ import annotations

import torch


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-8)


class AnswerSpotAdd:
    """Add ``alpha * ||h[a]|| * u`` at the last prompt position of one layer, prefill only.

    Decode steps (a single cached position) are left alone, so the edit acts on
    the answer spot once and the reply is generated from the edited state.
    """

    def __init__(self, m, layer: int, u: torch.Tensor, alpha: float):
        self.m, self.layer, self.u, self.alpha, self.h = m, layer, u, alpha, None

    def __enter__(self):
        def hook(mod, inp, o, _u=self.u, _al=self.alpha):
            tup = not torch.is_tensor(o)
            hh = o[0] if tup else o
            if hh.shape[1] < 2:      # a cached decode step
                return o
            hh = hh.clone()
            ha = hh[:, -1]
            hh[:, -1] = ha + _al * ha.norm(dim=-1, keepdim=True) * _u.to(hh.device, hh.dtype)
            return (hh, *o[1:]) if tup else hh
        self.h = self.m.layers[self.layer].register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self.h.remove()
