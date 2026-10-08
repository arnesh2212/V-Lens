# SPDX-License-Identifier: Apache-2.0
r"""The cross-modal interaction object, and the law it obeys exactly.

:func:`jvlm.vmap.ig_map` reads one question at a time.  For a region :math:`R`
of the picture, a layer :math:`\ell`, and a per-position corpus-mean baseline
:math:`\bar h`, it computes

.. math::
   A_R(q) \;=\; \sum_{p \in R} (h_p - \bar h_p)^{\!\top}
                \int_0^1 \nabla_{h_p} F\bigl(\bar h + \alpha (h - \bar h),\, q\bigr)
                \, \mathrm{d}\alpha ,

and the fundamental theorem of calculus makes the readings add up:

.. math::  \sum_R A_R(q) \;=\; F(h, q) - F(\bar h, q)

whenever the regions partition the image tokens.  That is a statement about one
question.  The object this module is for is the statement about *two*:

.. math::  I_R(q_a, q_b) \;=\; A_R(q_a) - A_R(q_b),

.. math::
   \sum_R I_R(q_a, q_b)
     &= \bigl[F(h,q_a) - F(\bar h,q_a)\bigr] - \bigl[F(h,q_b) - F(\bar h,q_b)\bigr] \\
     &= \bigl[F(h,q_a) - F(h,q_b)\bigr] - \bigl[F(\bar h,q_a) - F(\bar h,q_b)\bigr].

The second line is a difference-in-differences with the mean image as the
control condition, and it is the whole point.  The behavioural experiment
measures a DiD: how much more the real picture moves the answer for question
:math:`a` than for question :math:`b`, over and above what an average picture
does.  The lens does not merely *correlate* with that number -- it decomposes
it, region by region, and the pieces sum back to it exactly.

Two things make the law hold, and both are easy to break by accident.

*One baseline, shared.*  :math:`\bar h` must be the same tensor for both
questions.  Re-estimating a per-prompt baseline leaves
:math:`F(\bar h_a, q_a) - F(\bar h_b, q_b)` in place of the control term and the
identity above is simply false.  :func:`interaction_map` therefore takes a single
``means`` and passes it to both readings.

*One partition, complete.*  The regions must cover every image token exactly
once.  Dropping the high-norm register tokens -- which is the right thing to do
when the map is going to be *looked at*, since a register is not a place in the
picture -- puts a hole in the conservation law, and the size of that hole is a
number worth reporting rather than hiding.  :func:`check_partition` refuses a
region list that is not a partition, and :func:`grid_partition` builds one.

Nothing here is novel mathematics.  Integrated gradients is Sundararajan et
al.; vector-Jacobian attribution at a layer is standard; framing token influence
this way is what Jacobian Scopes already does.  What is new is the object: the
*difference* of two such readings under two prompts on one image, its exact
regional conservation law, and the fact that the quantity it conserves is the
one a prompt-counterfactual behavioural experiment can measure directly.

The contrast with the prior lenses is structural, not empirical.  Image tokens
precede text under a causal mask, so :math:`h_p` is a function of the image
alone: for any lens :math:`g(M h_p)` whose :math:`M` is not computed from the
prompt -- the logit lens (:math:`M = W_U`) and the J-lens
(:math:`M = W_U J_\ell`, a corpus average) -- the two readings are byte-identical
and :math:`I_R \equiv 0` by construction, not by measurement.  Only a reading
that carries *this pass's* gradient has an interaction at all.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from jlens.hooks import ActivationRecorder

from jvlm import vmap


# ------------------------------------------------------------------- regions


def grid_partition(grid: int, side: int) -> list[torch.Tensor]:
    r"""Square blocks of ``side`` x ``side`` patches covering a ``grid`` x ``grid``
    token map exactly once.

    Edge blocks are clipped rather than dropped, so the result is a partition
    even when ``side`` does not divide ``grid``.  A partition is not a nicety
    here: :math:`\sum_R A_R = F(h) - F(\bar h)` is a statement about *all* the
    image tokens, and any patch left out of the region list is a term missing
    from the left-hand side.
    """
    out: list[torch.Tensor] = []
    for ry in range(0, grid, side):
        for rx in range(0, grid, side):
            idx = [y * grid + x
                   for y in range(ry, min(ry + side, grid))
                   for x in range(rx, min(rx + side, grid))]
            out.append(torch.tensor(idx, dtype=torch.long))
    return out


def check_partition(regions: Sequence[torch.Tensor], n_patches: int) -> None:
    """Raise unless ``regions`` covers ``0..n_patches-1`` exactly once."""
    flat = [int(i) for r in regions for i in r.tolist()]
    seen = set(flat)
    if len(flat) != len(seen):
        dup = sorted({i for i in flat if flat.count(i) > 1})[:8]
        raise ValueError(f"regions overlap at patches {dup}...")
    missing = sorted(set(range(n_patches)) - seen)
    extra = sorted(seen - set(range(n_patches)))
    if missing or extra:
        raise ValueError(
            f"regions are not a partition of {n_patches} patches: "
            f"{len(missing)} missing (e.g. {missing[:8]}), "
            f"{len(extra)} out of range (e.g. {extra[:8]})")


def region_sums(per_patch: torch.Tensor,
                regions: Sequence[torch.Tensor]) -> torch.Tensor:
    """``[k, P]`` -> ``[k, R]`` by SUM.

    Sum, never mean.  A contribution is an additive quantity in logit units, so
    a region's reading is the sum of its patches' readings; averaging would make
    the reading depend on how finely the picture happens to be cut and would
    destroy the conservation law the regions exist to satisfy.
    """
    return torch.stack([per_patch[..., r.to(per_patch.device)].sum(-1)
                        for r in regions], dim=-1)


# ------------------------------------------------------------------- readings


def answer_logits(model, batch: dict, token_ids: Sequence[int], *,
                  layer: int | None = None, positions=None,
                  replace: torch.Tensor | None = None,
                  answer_pos: int | None = None) -> torch.Tensor:
    r"""``[k]`` -- the answer-position logits of ``token_ids`` for one pass.

    With ``replace`` given, the activations at ``positions`` in ``layer`` are
    overwritten by it before the rest of the network runs, which is how
    :math:`F(\bar h, q)` is obtained: not estimated, but run.  This is the same
    hook :func:`jvlm.vmap.completeness` uses, lifted out so the two prompts of an
    interaction can share it; the experiment asserts the two agree exactly.
    """
    if answer_pos is None:
        answer_pos = batch["input_ids"].shape[1] - 1
    last = model.n_layers - 1
    hd = None
    if replace is not None:
        pos = torch.as_tensor(positions, dtype=torch.long)

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
            return lg[list(token_ids)].float().cpu()
    finally:
        if hd is not None:
            hd.remove()


def did_truth(model, batch_a: dict, batch_b: dict, token_ids: Sequence[int],
              layer: int, positions, mean: torch.Tensor, *,
              answer_pos: tuple[int | None, int | None] = (None, None)) -> dict:
    r"""The right-hand side of the law, run rather than predicted.

    Four forward passes -- the real image and the mean image, under each of the
    two questions -- give the two per-prompt effects and their difference:

    .. math::
       \mathrm{eff}(q) = F(h,q) - F(\bar h,q), \qquad
       \mathrm{DiD} = \mathrm{eff}(q_a) - \mathrm{eff}(q_b).

    Every entry is a ``[k]`` tensor over ``token_ids``.
    """
    fa = answer_logits(model, batch_a, token_ids, answer_pos=answer_pos[0])
    fb = answer_logits(model, batch_b, token_ids, answer_pos=answer_pos[1])
    ba = answer_logits(model, batch_a, token_ids, layer=layer,
                       positions=positions, replace=mean,
                       answer_pos=answer_pos[0])
    bb = answer_logits(model, batch_b, token_ids, layer=layer,
                       positions=positions, replace=mean,
                       answer_pos=answer_pos[1])
    return {"F_a": fa, "F_b": fb, "Fbar_a": ba, "Fbar_b": bb,
            "eff_a": fa - ba, "eff_b": fb - bb,
            "did": (fa - ba) - (fb - bb),
            "behaviour": fa - fb, "baseline_gap": ba - bb}


def interaction_map(model, image, q_a, q_b, token_ids, layers, regions,
                    means: dict, *, steps: int = 4,
                    positions=None, batches: tuple[dict, dict] | None = None,
                    answer_pos: tuple[int | None, int | None] = (None, None),
                    diagnostics: bool = True) -> dict:
    r"""``I_R(q_a, q_b) = A_R(q_a) - A_R(q_b)`` on one image, at each layer.

    Args:
        image: the photograph both questions are asked about.  Ignored when
            ``batches`` is supplied (which is how the CPU proof drives this with
            a toy model that has no processor).
        q_a, q_b: the two prompts.  They may differ in length; each reading uses
            its own answer position.
        token_ids: the vocabulary items :math:`F` is read for, ``k`` of them.
            Every quantity here is linear in this axis, so a contrast such as
            ``logit(yes) - logit(no)`` is the difference of two rows and needs no
            extra passes.
        layers: source layers.  Must be at or after the model's DeepStack depth.
        regions: index vectors into ``0..P-1``, a partition of the image tokens.
        means: ``{layer: [P, d]}``, the shared baseline :math:`\bar h`.  The
            *same* tensor feeds both prompts; this is what makes the law exact.
        steps: interpolants on the straight path, midpoint rule.

    Returns a dict with per-patch and per-region ``I``, the two attributions it
    is built from, and (with ``diagnostics``) the maximum deviation between the
    two prompts' image-token activations, which the causal mask forces to be
    exactly zero and which is the reason the prior lenses cannot move.
    """
    from jvlm import readout

    if batches is None:
        ba = model.chat(q_a, [image])
        bb = model.chat(q_b, [image])
    else:
        ba, bb = batches
    if positions is None:
        pa = readout.image_positions(model, ba)
        pb = readout.image_positions(model, bb)
    else:
        pa = pb = torch.as_tensor(positions, dtype=torch.long)
    if len(pa) != len(pb):
        raise ValueError(f"the two prompts see {len(pa)} and {len(pb)} image "
                         "tokens; they cannot be compared region by region")
    P = len(pa)
    check_partition(regions, P)

    layers = sorted(set(int(l) for l in layers))
    A_a = vmap.ig_map(model, ba, token_ids, layers, pa.tolist(), means=means,
                      answer_pos=answer_pos[0], steps=steps)
    A_b = vmap.ig_map(model, bb, token_ids, layers, pb.tolist(), means=means,
                      answer_pos=answer_pos[1], steps=steps)

    out = {"layers": layers, "steps": steps, "n_patches": P,
           "pos_a": pa, "pos_b": pb,
           "a_patch": A_a, "b_patch": A_b,
           "patch": {l: A_a[l] - A_b[l] for l in layers},
           "a_region": {l: region_sums(A_a[l], regions) for l in layers},
           "b_region": {l: region_sums(A_b[l], regions) for l in layers}}
    out["region"] = {l: out["a_region"][l] - out["b_region"][l] for l in layers}
    if diagnostics:
        ha = readout.residuals(model, ba, layers)
        hb = readout.residuals(model, bb, layers)
        out["h_maxdev"] = {l: float((ha[l][pa].float().cpu()
                                     - hb[l][pb].float().cpu()).abs().max())
                           for l in layers}
        out["h"] = {l: ha[l][pa].float().cpu() for l in layers}
    return out


# ------------------------------------------------------------ prompt-blindness


def fixed_lens_patches(model, h: torch.Tensor, token_ids: Sequence[int], *,
                       jacobian: torch.Tensor | None = None) -> dict:
    r"""The two prompt-blind lenses read at image tokens: ``{name: [k, P]}``.

    ``logit lens`` is :math:`W_U\,\mathrm{norm}(h_p)`; ``J-lens`` is
    :math:`W_U\,\mathrm{norm}(J_\ell h_p)`.  Neither ``M`` is a function of the
    prompt, and :math:`h_p` is a function of the image alone, so both are
    constant in the question -- which is precisely the claim under test, and the
    reason their interaction is zero by construction rather than by fit.
    """
    ids = list(token_ids)
    with torch.no_grad():
        out = {"logit lens":
               model.unembed(h.to(model.device))[:, ids].float().cpu().T.contiguous()}
        if jacobian is not None:
            hj = h.float().to(model.device) @ jacobian.to(model.device).float().T
            out["J-lens"] = model.unembed(hj)[:, ids].float().cpu().T.contiguous()
    return out


# ---------------------------------------------------------------- error stats


def error_stats(claim, truth) -> dict:
    """How far a claimed decomposition is from the effect it decomposes.

    Reports the absolute gap in logit units, the gap relative to the size of the
    effect, and -- when there are enough rows -- both the correlation and the
    :math:`R^2` about the *identity* line.  Correlation alone would let a claim
    that is off by a constant factor look perfect, which for a conservation law
    is not a pass.
    """
    import numpy as np
    claim = torch.as_tensor(np.asarray(claim, dtype=np.float64)).flatten()
    truth = torch.as_tensor(np.asarray(truth, dtype=np.float64)).flatten()
    err = claim - truth
    scale = truth.abs().mean().clamp_min(1e-12)
    out = {"n": int(claim.numel()),
           "max_abs_err": float(err.abs().max()),
           "mean_abs_err": float(err.abs().mean()),
           "rel_err": float(err.abs().mean() / scale),
           "truth_scale": float(truth.abs().mean()),
           "truth_max_abs": float(truth.abs().max()),
           "pearson": float("nan"), "spearman": float("nan"),
           "r2_identity": float("nan")}
    if claim.numel() > 2 and claim.std() > 0 and truth.std() > 0:
        out["pearson"] = float(
            torch.corrcoef(torch.stack([claim, truth]))[0, 1])
        ss = ((truth - truth.mean()) ** 2).sum().clamp_min(1e-30)
        out["r2_identity"] = float(1.0 - (err ** 2).sum() / ss)
        try:
            from scipy import stats as sps
            out["spearman"] = float(sps.spearmanr(claim.numpy(),
                                                  truth.numpy()).statistic)
        except Exception:
            pass
    return out
