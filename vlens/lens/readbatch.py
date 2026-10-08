# SPDX-License-Identifier: Apache-2.0
"""Read many prompts at once.

Right padding leaves every row's real tokens untouched under a causal mask, so
a row's answer-spot state is the same whether it was run alone or in a batch;
``tools/verify_batching.py`` checks that on every model before a run.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch

from vlens.lens.hooks import ActivationRecorder


@torch.no_grad()
def answer_states(m, items: Sequence[dict], layers: Sequence[int], *, batch: int = 8,
                  logits: bool = True, progress: Callable | None = None):
    """States at ``layers`` and final logits at each item's answer spot.

    ``items`` are the dicts ``batch_many`` takes. Returns ``(H, lg)`` with
    ``H[i, j]`` the state of item ``i`` at ``layers[j]`` and ``lg[i]`` its final
    logits, or ``None`` when ``logits`` is false.
    """
    LAST = m.n_layers - 1
    at = sorted(set([*layers, LAST]))
    col = {l: k for k, l in enumerate(at)}
    H, LG = [], []
    for s in range(0, len(items), batch):
        chunk = list(items[s: s + batch])
        b = m.batch_many(chunk, side="right")
        spot = m.answer_spot(b)
        with ActivationRecorder(m.layers, at=at) as rec:
            m.forward(b)
            st = {l: rec.activations[l][torch.arange(len(chunk)),
                                        spot.to(rec.activations[l].device)] for l in at}
        H.append(np.stack([torch.stack([st[l][j] for l in layers]).float().cpu().numpy()
                           for j in range(len(chunk))]).astype(np.float32))
        if logits:
            LG.append(m.unembed(st[LAST].to(m.device)).float().cpu())
        del st, rec
        if progress:
            progress(min(s + batch, len(items)), len(items))
    return np.concatenate(H), (torch.cat(LG) if logits else None)


@torch.no_grad()
def scores_for(m, items: Sequence[dict], fns: Sequence[Callable], *, batch: int = 8):
    """One score per item, each with its own score function."""
    out = np.zeros(len(items), np.float32)
    for s in range(0, len(items), batch):
        chunk = list(items[s: s + batch])
        b = m.batch_many(chunk, side="right")
        spot = m.answer_spot(b)
        LAST = m.n_layers - 1
        with ActivationRecorder(m.layers, at=[LAST]) as rec:
            m.forward(b)
            h = rec.activations[LAST][torch.arange(len(chunk)),
                                      spot.to(rec.activations[LAST].device)]
        lg = m.unembed(h.to(m.device))
        for j in range(len(chunk)):
            out[s + j] = float(fns[s + j](None, lg[j: j + 1]).float().cpu()[0])
    return out
