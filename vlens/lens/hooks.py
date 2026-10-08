# SPDX-License-Identifier: Apache-2.0
"""Recording residual states, and writing new ones in, by forward hook.

``ActivationRecorder`` captures each requested block's output for the duration
of a ``with`` block.  ``Patch`` replaces the states at chosen positions of one
block with a tensor we supply, which is how every intervention in this work is
applied: the restoration test, the mean replacement, the path integral and the
steering edit all go through it.

A block's output may be a tensor or a tuple whose first element is the tensor;
both are handled.  Recorded states are never detached, so the same recorder serves the
forward-only reads (wrap them in ``torch.no_grad``) and the backward passes
that V-Write integrates along.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import torch


class ActivationRecorder:
    """Capture the output of the given blocks.

    ``activations[i]`` is the output of ``modules[i]`` with shape
    ``[batch, positions, d_model]``.
    """

    def __init__(self, modules: Sequence[torch.nn.Module], at: Iterable[int], grad: bool = True,
                 *, start_graph_at: int | None = None):
        self.modules = modules
        self.at = sorted(set(int(i) for i in at))
        self.grad = grad
        #: the block whose output becomes the leaf that roots the autograd graph,
        #: needed because a model loaded for inference has every parameter frozen
        self.start_graph_at = start_graph_at
        if start_graph_at is not None and start_graph_at not in self.at:
            self.at = sorted({*self.at, int(start_graph_at)})
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def _make(self, i: int):
        root = i == self.start_graph_at

        def hook(mod, inp, out):
            # never detached, so the same recorder serves forward reads and
            # the backward passes V-Write integrates along
            t = out[0] if not torch.is_tensor(out) else out
            if root:
                t.requires_grad_(True)
            self.activations[i] = t
        return hook

    def __enter__(self) -> "ActivationRecorder":
        self.activations = {}
        self._handles = [self.modules[i].register_forward_hook(self._make(i)) for i in self.at]
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


class Patch:
    """Replace the states at ``positions`` of one block with ``value``.

    ``value`` is ``[n, len(positions), d_model]``; the batch is replicated by the
    caller when ``n > 1``.  The write is out of place, so autograd can flow back
    into ``value`` and the path integral works.
    """

    def __init__(self, module: torch.nn.Module, positions: torch.Tensor, value: torch.Tensor):
        self.module, self.positions, self.value, self._h = module, positions, value, None

    def __enter__(self) -> "Patch":
        def hook(mod, inp, out, _p=self.positions, _v=self.value):
            tup = not torch.is_tensor(out)
            hh = (out[0] if tup else out).clone()
            hh[:, _p.to(hh.device)] = _v.to(hh.device, hh.dtype)
            return (hh, *out[1:]) if tup else hh
        self._h = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc) -> None:
        self._h.remove()
        self._h = None


class AddAtPosition:
    """Add ``alpha * ||h[p]|| * u`` at one position of one block, prefill only.

    A decode step carries a single cached position and is left alone, so the
    edit is applied once and the reply is generated from the edited state.
    """

    def __init__(self, module: torch.nn.Module, u: torch.Tensor, alpha: float, position: int = -1):
        self.module, self.u, self.alpha, self.position, self._h = module, u, alpha, position, None

    def __enter__(self) -> "AddAtPosition":
        def hook(mod, inp, out, _u=self.u, _a=self.alpha, _p=self.position):
            tup = not torch.is_tensor(out)
            hh = out[0] if tup else out
            if hh.shape[1] < 2:
                return out
            hh = hh.clone()
            h = hh[:, _p]
            hh[:, _p] = h + _a * h.norm(dim=-1, keepdim=True) * _u.to(hh.device, hh.dtype)
            return (hh, *out[1:]) if tup else hh
        self._h = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc) -> None:
        self._h.remove()
        self._h = None
