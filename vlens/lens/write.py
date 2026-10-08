# SPDX-License-Identifier: Apache-2.0
"""Writing along a direction at the image tokens, and the directions to compare.

A lens that only reads can be checked against what the model says. A lens you
can write to has to be checked against what the model *does*: push along the
direction it names, at the picture rather than at the words, and see whether
the one-word answer becomes the word you chose. Every direction here is a unit
vector applied at the same dose, scaled to the local residual norm, so the only
thing that differs between them is where they point.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch

from vlens.lens import attribution, readout
from vlens.lens.hooks import ActivationRecorder
from vlens.lens.room import is_oom, with_room


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-8)


class AddAtPositions:
    """Add ``alpha * ||h[p]|| * u`` at several positions of one block, prefill only.

    The single-position form in ``hooks`` writes at the answer spot; steering a
    picture has to write at every image token, so the positions come as a
    tensor. A decode step carries one cached position and is left alone, so the
    edit lands once and the reply continues from the edited state.
    """

    def __init__(self, module, u: torch.Tensor, alpha: float, positions: torch.Tensor):
        self.module, self.u, self.alpha, self.pos, self._h = module, u, alpha, positions, None

    def __enter__(self) -> "AddAtPositions":
        def hook(mod, inp, out, _u=self.u, _a=self.alpha, _p=self.pos):
            tup = not torch.is_tensor(out)
            hh = out[0] if tup else out
            if hh.shape[1] < 2:
                return out
            hh = hh.clone()
            p = _p.to(hh.device)
            h = hh[:, p]
            hh[:, p] = h + _a * h.norm(dim=-1, keepdim=True) * _u.to(hh.device, hh.dtype)
            return (hh, *out[1:]) if tup else hh
        self._h = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc) -> None:
        self._h.remove()
        self._h = None


def write_direction(m, items: Sequence[dict], layer: int, score: Callable,
                    batch: int = 4, log: Callable = print) -> torch.Tensor:
    """V-Write's direction for a score: the corpus-mean gradient at image tokens.

    The same quantity the patch attributions are built from, pooled over
    positions and over a corpus instead of read per patch, which turns it from
    a map of one picture into one direction for the model.
    """
    acc, n = None, 0
    for i, it in enumerate(items):
        def one(it=it):
            b = m.chat(it["question"], [it["image"]], system=it.get("system"), cue=it.get("cue"))
            pos = readout.image_positions(m, b)
            if len(pos) == 0:
                return None
            with torch.no_grad(), ActivationRecorder(m.layers, at=[layer]) as rec:
                m.forward(b)
                h = rec.activations[layer][0, pos.to(rec.activations[layer].device)]
            # the score is the model's own output, which reaches these states
            # only by running the rest of the network on them: differentiating
            # the layer's own reading with respect to itself gives nothing
            x = h.float().to(m.device).detach().clone().requires_grad_(True)
            with torch.enable_grad():
                v = attribution.hooked_axis(m, b, score, layer, pos, x[None]).sum()
                g, = torch.autograd.grad(v, x)
            return g.float().mean(0).cpu()
        try:
            got = with_room(one, log=log)
        except Exception as e:                      # noqa: BLE001
            if not is_oom(e):
                raise
            continue
        if got is None:
            continue
        acc = got if acc is None else acc + got
        n += 1
        if log and (i + 1) % 25 == 0:
            log(f"    write direction {i + 1}/{len(items)} prompts")
    if n == 0:
        raise RuntimeError("no prompt yielded a direction")
    return unit(acc / n)


def difference_of_means(m, pos_items: Sequence[dict], neg_items: Sequence[dict], layer: int,
                        log: Callable = print) -> torch.Tensor:
    """The labelled direction: mean image-token state on one answer minus the other.

    This one needs labels, which is the point of carrying it: it is what a
    supervised method would use, and the unlabelled directions have to stand
    beside it.
    """
    def mean_of(items):
        acc, n = None, 0
        for it in items:
            b = m.chat(it["question"], [it["image"]], system=it.get("system"), cue=it.get("cue"))
            pos = readout.image_positions(m, b)
            if len(pos) == 0:
                continue
            with torch.no_grad(), ActivationRecorder(m.layers, at=[layer]) as rec:
                m.forward(b)
                h = rec.activations[layer][0, pos.to(rec.activations[layer].device)]
            acc = h.float().mean(0).cpu() if acc is None else acc + h.float().mean(0).cpu()
            n += 1
        return None if n == 0 else acc / n
    a, b_ = mean_of(pos_items), mean_of(neg_items)
    if a is None or b_ is None:
        raise RuntimeError("not enough labelled items for a difference of means")
    return unit(a - b_)


def unembed_direction(m, token_id: int) -> torch.Tensor:
    """The logit lens's answer: the unembedding row for the token."""
    w = m._lm_head.weight
    return unit(w[int(token_id)].detach().float().cpu())


def random_direction(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed) % (2 ** 31))
    return unit(torch.randn(d, generator=g))
