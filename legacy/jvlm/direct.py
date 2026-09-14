# SPDX-License-Identifier: Apache-2.0
r"""The direct estimator: concept directions without ever forming the transport.

Fitting a lens means estimating a ``d x d`` matrix, and that costs ``d_model``
backward passes per prompt because each pass recovers only ``dim_batch`` rows.
For a 2B model that is 2048 passes; for an 8B model, 4096.

But look at what the matrix is *used* for.  Reading it out is
:math:`W_U D_x A_l h`.  Intervening with it -- every ablation, every injection,
every steering vector in this work -- uses only

.. math::  v_t^{(l)} = A_l^{\top} D_x^{\top} W_U[t],

the direction that writes token :math:`t`.  And that is not a product with a
matrix we have to build.  Since

.. math::
   A_l = \mathbb{E}\left[\frac{\partial h_L[a]}{\partial h_l[p]}\right],
   \qquad
   \ell_t = W_U[t] \cdot \mathrm{RMSNorm}(h_L[a]),

we have

.. math::
   v_t^{(l)} = \mathbb{E}_{x,p}\!\left[\frac{\partial \ell_t(x)}{\partial h_l[p]}\right],

which is one ordinary backward pass of a scalar -- the token's logit at the
answer -- returning the direction **at every layer and every position at once**.

So a concept's direction over an ``n``-prompt corpus costs ``n`` backward passes
per token rather than ``n * d_model``.  For the concepts used here that is a
factor of about four hundred, it removes the ``d x d`` memory entirely, and it is
*more* faithful, not less: the corpus average is taken on the object that gets
used instead of on a matrix that is then contracted with it.  The norm's Jacobian
comes along for free because the backward runs through the real norm.

It also makes the averaging error measurable.  The matrix route can only report a
mean; here each prompt yields its own direction, so the spread across prompts --
the quantity that decides whether one averaged direction can stand in for all of
them -- is simply its variance.
"""
from __future__ import annotations

import torch

from jlens.hooks import ActivationRecorder


def logit_gradients(
    model,
    batch: dict,
    token_ids,
    layers,
    *,
    answer_pos: int | None = None,
    channels: dict | None = None,
):
    """``{channel: {layer: [k, d]}}`` -- the gradient of each token's answer logit.

    One backward pass per token returns the gradient at every layer and every
    position at once, so several channel averages come out of the same pass for
    free.  That is what makes the channel-conditioned lens cost nothing extra:
    the transport out of an image site and out of a text site are different
    operators, and this reads both.

    Args:
        batch: a prepared multimodal batch.
        token_ids: the tokens whose logits are differentiated.
        layers: layers to collect at; must start at or above the model's
            in-place visual injection.
        answer_pos: position of the answer; defaults to the last token.
        channels: ``{name: boolean mask over positions}``; defaults to the image
            span, the text span, and the answer position alone.
    """
    layers = sorted(set(layers))
    S = batch["input_ids"].shape[1]
    if answer_pos is None:
        answer_pos = S - 1
    if channels is None:
        img = model.image_mask(batch["input_ids"])[0]
        ans = torch.zeros_like(img)
        ans[answer_pos] = True
        channels = {"img": img, "txt": ~img, "ans": ans}
    idx = {k: v.nonzero(as_tuple=True)[0] for k, v in channels.items()}
    idx = {k: v for k, v in idx.items() if len(v)}
    if not idx:
        raise ValueError("no source positions in any channel")

    out = {k: {l: [] for l in layers} for k in idx}
    with (
        ActivationRecorder(model.layers, at=[*layers, model.n_layers - 1],
                           start_graph_at=min(layers)) as rec,
        torch.enable_grad(),
    ):
        model.forward(batch)
        acts = [rec.activations[l] for l in layers]
        h_final = rec.activations[model.n_layers - 1][0, answer_pos]
        # the logit itself, so the backward runs through the real final norm and
        # picks up D_x exactly rather than an approximation to it
        logits = model.unembed(h_final[None])[0]
        for j, t in enumerate(token_ids):
            grads = torch.autograd.grad(
                logits[t], acts, retain_graph=(j < len(token_ids) - 1))
            for l, g in zip(layers, grads, strict=True):
                for k, pos in idx.items():
                    out[k][l].append(
                        g[0, pos.to(g.device)].float().mean(0).cpu())
    return {k: {l: torch.stack(v) for l, v in d.items()} for k, d in out.items()}


class DirectDirections:
    """Corpus-averaged concept directions, per channel, one prompt at a time."""

    def __init__(self, layers):
        self.layers = sorted(set(layers))
        self._sum: dict[str, dict[int, torch.Tensor]] = {}
        self._sumsq: dict[str, dict[int, torch.Tensor]] = {}
        self.n = 0

    def add(self, model, batch, token_ids, **kw) -> None:
        g = logit_gradients(model, batch, token_ids, self.layers, **kw)
        for ch, per_layer in g.items():
            a = self._sum.setdefault(ch, {})
            b = self._sumsq.setdefault(ch, {})
            for l, v in per_layer.items():
                if l not in a:
                    a[l] = torch.zeros_like(v)
                    b[l] = torch.zeros_like(v)
                a[l] += v
                b[l] += v * v
        self.n += 1

    @property
    def channels(self):
        return sorted(self._sum)

    def mean(self, channel: str = "img", *, normalize: bool = True
             ) -> dict[int, torch.Tensor]:
        out = {l: v / max(self.n, 1) for l, v in self._sum.get(channel, {}).items()}
        if normalize:
            out = {l: v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                   for l, v in out.items()}
        return out

    def by_channel(self, *, normalize: bool = True) -> dict[str, dict]:
        """Everything at once, ready to hand to a channel-conditioned hook."""
        return {ch: self.mean(ch, normalize=normalize) for ch in self.channels}

    def stability(self, channel: str = "img") -> dict[int, float]:
        """Share of each direction's energy that is the mean rather than noise.

        ``||E[v]||^2 / E[||v||^2]`` -- one at a layer where every prompt agrees,
        and near ``1/n`` where they do not.  This is the averaging error the
        matrix route cannot report, because it never sees a single prompt's
        direction on its own.
        """
        out = {}
        for l in self._sum.get(channel, {}):
            mu = self._sum[channel][l] / max(self.n, 1)
            e_sq = (self._sumsq[channel][l] / max(self.n, 1)).sum(-1)
            out[l] = float(((mu * mu).sum(-1) / e_sq.clamp_min(1e-12)).mean())
        return out
