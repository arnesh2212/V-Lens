# SPDX-License-Identifier: Apache-2.0
r"""The path integral, evaluated a few points at a time.

:func:`jvlm.vmap.ig_map` runs all ``steps`` interpolants as one batch, which is
the fastest arrangement and the right one on an idle card: the rows do not
interact, so a single backward pass on the summed logit returns each row's own
gradient.  It also holds ``steps`` copies of the whole graph at once, and on a
shared card that is what decides whether the job can be scheduled at all -- a
reading taken one layer above the vision injection keeps the graph for almost
the entire stack, and eight interpolants of it will not fit beside another
model.

The integral does not care how the points are grouped.  Splitting the path into
chunks and accumulating the gradient gives the identical estimator with the
memory of the chunk rather than of the path, at the cost of one extra forward
per chunk.  ``ig_map_chunked(steps=8, chunk=2)`` is arithmetically
``ig_map(steps=8)`` and costs the memory of ``ig_map(steps=2)``.

Nothing else changes: the rule is still the midpoint one, the baseline is still
the per-position corpus mean, and the readings still sum over positions to the
change in the token's logit when the picture is replaced by that baseline.
:func:`jvlm.vmap.completeness` is the check, and :func:`agrees` is the unit test
that this arrangement is the same arithmetic as the batched one.
"""
from __future__ import annotations

import torch

from jlens.hooks import ActivationRecorder


def ig_map_chunked(model, batch, token_ids, layer, positions, mean,
                   steps: int = 8, chunk: int = 2,
                   answer_pos: int | None = None) -> torch.Tensor:
    r"""``[k, P]`` -- each image position's contribution to each token's logit.

    Args:
        mean: the baseline :math:`\bar h_\ell[p]`, ``[P, d]``.
        steps: interpolants on the straight path from the baseline, midpoint
            rule -- the same path :func:`jvlm.vmap.ig_map` integrates.
        chunk: how many of them are held in memory at once.  The result does
            not depend on this; only the peak memory and the number of forward
            passes do.
    """
    if answer_pos is None:
        answer_pos = batch["input_ids"].shape[1] - 1
    pos = torch.as_tensor(positions, dtype=torch.long)
    last = model.n_layers - 1
    ids = list(token_ids)

    with torch.no_grad(), ActivationRecorder(model.layers, at=[layer]) as rec:
        model.forward(batch)
        act = rec.activations[layer]
        h = act[0, pos.to(act.device)].detach().float()
    mu = mean.to(h.device).float()
    dev = h - mu
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    acc = torch.zeros(len(ids), dev.shape[0], dtype=torch.float64)

    for s0 in range(0, steps, chunk):
        al = alphas[s0: s0 + chunk].to(h.device)
        n = int(al.numel())
        big = model.replicate(batch, n) if n > 1 else batch
        x = (mu[None] + al[:, None, None] * dev[None]
             ).detach().clone().requires_grad_(True)

        def hook(mod, inp, o, _x=x, _p=pos):
            tup = not torch.is_tensor(o)
            hh = (o[0] if tup else o).clone()
            hh[:, _p.to(hh.device)] = _x.to(hh.dtype)
            return (hh, *o[1:]) if tup else hh

        hd = model.layers[layer].register_forward_hook(hook)
        try:
            with (ActivationRecorder(model.layers, at=[last]) as rec,
                  torch.enable_grad()):
                model.forward(big)
                logits = model.unembed(rec.activations[last][:, answer_pos])
                for j, t in enumerate(ids):
                    g, = torch.autograd.grad(
                        logits[:, t].sum(), x,
                        retain_graph=(j < len(ids) - 1))
                    acc[j] += (g.float().sum(0) * dev).sum(-1).double().cpu()
        finally:
            hd.remove()
    return (acc / steps).float()


def agrees(model, batch, token_ids, layer, positions, mean, steps: int = 2
           ) -> float:
    """Max absolute difference from the batched estimator on the same path.

    Run once at the start of an experiment.  A chunked accumulation that has
    drifted from :func:`jvlm.vmap.ig_map` is a wiring bug, not a numerical one,
    and it would show up here as a large number rather than a small one.
    """
    from jvlm import vmap

    a = ig_map_chunked(model, batch, token_ids, layer, positions, mean,
                       steps=steps, chunk=steps)
    b = vmap.ig_map(model, batch, token_ids, [layer], positions,
                    means={layer: mean}, steps=steps)[layer]
    return float((a - b).abs().max())
