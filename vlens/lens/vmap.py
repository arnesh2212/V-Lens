# SPDX-License-Identifier: Apache-2.0
r"""V-Lens: two maps over a picture, at every layer, with no fitting.

The first version of this tried to *fit* a linear map from an image token's
activation to the answer's logits.  It failed, and the reason is worth stating
because it is the whole design lesson: a corpus-averaged map has no per-position
information left in it, so it cannot make a picture.  Averaging is exactly what
destroys the thing a vision lens is for.

So V-Lens averages nothing.  At every image token :math:`p` and every layer
:math:`l` it reports two numbers per vocabulary item :math:`t`:

.. math::
   \mathrm{say}(p,l,t) &= W_U[t] \cdot \mathrm{norm}(h_l[p]) \\
   \mathrm{do}(p,l,t)  &= \frac{\partial \ell_t}{\partial h_l[p]}
                          \cdot \bigl(h_l[p] - \bar h_l[p]\bigr)

*say* is the plain logit lens: what that patch of the picture would say if it
spoke.  *do* is what that patch is contributing to what the model will actually
answer -- the first-order change in the answer's logit for :math:`t` if the patch
were replaced by an average one.  :math:`\ell_t` is the logit at the answer
position, differentiated through the real final norm, so the chain rule is
exact rather than approximated.

Three properties make this the right object where the fit was the wrong one.

*Nothing is fitted.*  There is no training set, no held-out set, no shrinkage
parameter, and nothing to overfit.  The only corpus quantity is
:math:`\bar h_l[p]`, a mean.

*It is cheap.*  One backward pass of :math:`\ell_t` yields the gradient at every
layer and every position at once, so a whole vocabulary over a whole picture
costs one pass per word, not one per word per layer per position.

*It is per-token.*  The output is a map over the picture, which is what a vision
lens should produce and what a corpus-averaged transport can never produce.

The estimator is not new -- it is gradient-times-input with a data baseline, the
quantity attribution patching computes and, pooled over positions, Grad-CAM's.
What is new here is using it as a *lens*: read in the vocabulary, at every layer,
beside the logit lens, with the disagreement between the two as the measurement.
"""
from __future__ import annotations

import torch

from jlens.hooks import ActivationRecorder


def do_map(model, batch, token_ids, layers, positions,
           means: dict | None = None, answer_pos: int | None = None,
           unit: bool = False) -> dict:
    r"""``{layer: [k, P]}`` -- each token's contribution to the answer's logits.

    One backward pass per vocabulary item.  The gradient it returns covers every
    layer and every position, so the cost does not grow with either.

    ``means`` gives :math:`\bar h_l[p]`, the per-position corpus mean, which is
    the baseline the contribution is measured against: *how much is this patch
    pushing the answer, relative to an average patch in this position*.  Without
    it the baseline is the origin, which is gradient-times-activation and is
    dominated by whatever is large rather than by whatever matters.
    """
    layers = sorted(set(layers))
    if answer_pos is None:
        answer_pos = batch["input_ids"].shape[1] - 1
    pos = torch.as_tensor(positions, dtype=torch.long)
    out = {l: [] for l in layers}
    with (
        ActivationRecorder(model.layers, at=[*layers, model.n_layers - 1],
                           start_graph_at=min(layers)) as rec,
        torch.enable_grad(),
    ):
        model.forward(batch)
        acts = [rec.activations[l] for l in layers]
        h_final = rec.activations[model.n_layers - 1][0, answer_pos]
        logits = model.unembed(h_final[None])[0]
        dev = {}
        for l, a in zip(layers, acts):
            h = a[0, pos.to(a.device)].detach().float()
            mu = (means[l].to(h.device).float() if means is not None
                  else torch.zeros_like(h))
            dev[l] = h - mu
        for j, t in enumerate(token_ids):
            grads = torch.autograd.grad(
                logits[t], acts, retain_graph=(j < len(token_ids) - 1))
            for l, g in zip(layers, grads, strict=True):
                gp = g[0, pos.to(g.device)].float()
                if unit:
                    gp = gp / gp.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                out[l].append((gp * dev[l]).sum(-1).cpu())
    return {l: torch.stack(v) for l, v in out.items()}


def ig_map(model, batch, token_ids, layers, positions,
           means: dict, answer_pos: int | None = None, steps: int = 4,
           unit: bool = False) -> dict:
    r"""``{layer: [k, P]}`` -- the same reading, integrated along the path.

    :func:`do_map` evaluates the gradient once, at the activation the model
    actually has.  That is a first-order estimate of what a region contributes,
    and it inherits the two problems every first-order attribution has: a
    saturated gradient reads as zero contribution however much the feature
    matters, and the estimate is only exact in an infinitesimal neighbourhood.

    Averaging the gradient along the straight path from the baseline to the
    activation fixes both, and buys a property neither of the other two lenses
    has.  Writing :math:`h(\alpha) = \bar h + \alpha (h - \bar h)` on the image
    positions,

    .. math::
       s_t(R) = \sum_{p \in R} (h_\ell[p] - \bar h_\ell[p]) \cdot
                \int_0^1 \frac{\partial \ell_t(h(\alpha))}{\partial h_\ell[p]}
                \, \mathrm{d}\alpha ,

    and the fundamental theorem of calculus gives, summing over *all* the image
    positions rather than a region,

    .. math::
       \sum_p s_t(p) = \ell_t(h) - \ell_t(\bar h)

    exactly.  The readings are not scores that happen to correlate with an
    intervention: they are that intervention's effect, split over regions, and
    they add up to it.  Completeness also makes the region coordinate
    well-posed, since a region's reading is the sum of its patches' readings
    however the region is drawn.

    The path costs one extra forward, not ``steps`` extra backward passes: the
    ``steps`` interpolants are run as a batch, and because batch rows do not
    interact, one backward on the summed logit returns each row's own gradient.

    Args:
        means: the baseline :math:`\bar h_\ell[p]`, per layer, ``[P, d]``.
        unit: normalise each word's lens vector before the dot product, giving
            a projection onto the word's direction rather than a contribution in
            logit units.  This trades the conservation law for a reading whose
            scale does not vary across words with the gradient's magnitude, and
            is the right form when the question is *which word* rather than *how
            much*.
        steps: interpolants on the path.  The rule is the midpoint one, so
            ``steps=1`` evaluates at :math:`\alpha = 1/2` and is *not*
            :func:`do_map`; the single-point form that matches ``do_map`` is
            :math:`\alpha = 1`, which is what the comparisons report as the
            first-order row.
    """
    layers = sorted(set(layers))
    if answer_pos is None:
        answer_pos = batch["input_ids"].shape[1] - 1
    pos = torch.as_tensor(positions, dtype=torch.long)
    last = model.n_layers - 1

    with torch.no_grad():
        with ActivationRecorder(model.layers, at=layers) as rec:
            model.forward(batch)
            clean = {l: rec.activations[l][0, pos.to(rec.activations[l].device)]
                     .detach().float() for l in layers}

    big = model.replicate(batch, steps) if steps > 1 else batch
    # midpoint rule: alpha at the centre of each interval, which integrates a
    # smooth path exactly to second order and never evaluates at the baseline
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    out = {}
    for l in layers:
        h = clean[l]
        mu = means[l].to(h.device).float()
        dev = h - mu
        x = (mu[None] + alphas[:, None, None].to(h.device) * dev[None]
             ).detach().clone().requires_grad_(True)

        def hook(mod, inp, o, _x=x, _p=pos):
            tup = not torch.is_tensor(o)
            hh = (o[0] if tup else o).clone()
            hh[:, _p.to(hh.device)] = _x.to(hh.dtype)
            return (hh, *o[1:]) if tup else hh

        hd = model.layers[l].register_forward_hook(hook)
        try:
            with (ActivationRecorder(model.layers, at=[last]) as rec,
                  torch.enable_grad()):
                model.forward(big)
                h_final = rec.activations[last][:, answer_pos]
                logits = model.unembed(h_final)              # [steps, V]
                rows = []
                for j, t in enumerate(token_ids):
                    g, = torch.autograd.grad(
                        logits[:, t].sum(), x,
                        retain_graph=(j < len(token_ids) - 1))
                    gm = g.float().mean(0)
                    if unit:
                        gm = gm / gm.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                    rows.append((gm * dev).sum(-1).cpu())
        finally:
            hd.remove()
        out[l] = torch.stack(rows)
    return out


def completeness(model, batch, token_ids, layer, positions, mean,
                 scores: torch.Tensor, answer_pos: int | None = None) -> dict:
    """How close the readings come to the effect they claim to decompose.

    Runs the model once with every image position replaced by the baseline and
    compares the true change in each token's logit with the sum of that token's
    readings.  For :func:`ig_map` the two agree up to the discretisation of the
    path; for :func:`do_map` they do not, and the size of the gap is the size of
    the first-order error.
    """
    if answer_pos is None:
        answer_pos = batch["input_ids"].shape[1] - 1
    pos = torch.as_tensor(positions, dtype=torch.long)
    last = model.n_layers - 1
    ids = list(token_ids)

    def read(replace):
        hd = None
        if replace is not None:
            def hook(mod, inp, o, _v=replace, _p=pos):
                tup = not torch.is_tensor(o)
                hh = (o[0] if tup else o).clone()
                hh[0, _p.to(hh.device)] = _v.to(hh.device).to(hh.dtype)
                return (hh, *o[1:]) if tup else hh
            hd = model.layers[layer].register_forward_hook(hook)
        try:
            with torch.no_grad(), ActivationRecorder(model.layers, at=[last]) as rec:
                model.forward(batch)
                lg = model.unembed(rec.activations[last][0, answer_pos][None])[0]
                return lg[ids].float().cpu()
        finally:
            if hd is not None:
                hd.remove()

    truth = read(None) - read(mean)
    claim = scores.sum(-1).float().cpu()
    denom = truth.abs().mean().clamp_min(1e-6)
    return {"truth": truth.tolist(), "claim": claim.tolist(),
            "rel_error": float((claim - truth).abs().mean() / denom),
            "corr": float(torch.corrcoef(torch.stack([claim, truth]))[0, 1])
            if len(ids) > 2 else float("nan")}


def normalise(scores: torch.Tensor) -> torch.Tensor:
    r"""Standardise each position across the vocabulary.

    A raw contribution is proportional to how much of the residual sits at that
    position, so a map of raw contributions is partly a map of activation norm.
    The logit lens is implicitly free of this because its scores are compared
    within a position; doing the same here asks the question a map should ask --
    *among all the words, which one is this place pushing hardest* -- and makes
    the two maps comparable rather than one of them carrying a norm confound.

    ``scores`` is ``[k, P]``; the result is z-scored down the vocabulary axis.
    """
    mu = scores.mean(0, keepdim=True)
    sd = scores.std(0, keepdim=True).clamp_min(1e-6)
    return (scores - mu) / sd


def pointing(heat: torch.Tensor, inside: set, grid: int) -> dict:
    """The saliency literature's pointing game, on a token grid.

    ``hit`` is whether the map's single strongest cell falls inside the object's
    box; ``hit5`` allows the top five.  Chance is the fraction of the frame the
    box covers, which is reported alongside so a big object cannot look like a
    good result.
    """
    P = heat.numel()
    order = heat.argsort(descending=True)
    top1 = int(order[0])
    top5 = set(order[: min(5, P)].tolist())
    return {"hit": int(top1 in inside), "hit5": int(bool(top5 & inside)),
            "chance": len(inside) / P,
            "mass_in": float(heat.clamp_min(0)[list(inside)].sum()
                             / heat.clamp_min(0).sum().clamp_min(1e-9))}
