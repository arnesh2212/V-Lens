# SPDX-License-Identifier: Apache-2.0
r"""How far a map over the picture moves when only the words change.

The theorem is one line.  Image tokens precede text tokens under a causal mask,
so :math:`h_\ell[p]` at an image position is a function of the pixels alone.
Any lens of the form :math:`g(M\,h_\ell[p])` whose :math:`M` is not computed
from the prompt -- the logit lens, where :math:`M=I`, and the J-lens, where
:math:`M=J_\ell` is a corpus average -- therefore returns the *same bytes* for
every prompt about that photograph.  It cannot answer "what is this patch doing
for *this* question", because it never saw the question.

Our reading breaks the form: the gradient
:math:`\partial \ell_t / \partial h_\ell[p]` is taken at the answer position of
*this* prompt, so :math:`M` is this pass's transport and depends on every word.

This module supplies the measuring tape.  ``movement`` compares one lens's map
under two prompts five ways, from a scale-free rank distance to the raw maximum
absolute difference that the zero test needs; ``ig_diag`` runs the integrated
reading and reports, as a by-product, how much of the gradient survives to the
end of the path, which is the number that decides whether a saturated target
has to be replaced by a margin.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import stats as sps

from jlens.hooks import ActivationRecorder


# ------------------------------------------------------------------- regions


def region_blocks(grid: int, coarse: int, drop: torch.Tensor | None = None):
    """A ``coarse x coarse`` tiling of a ``grid x grid`` token layout.

    ``drop`` marks register/sink tokens to leave out, so an attention sink does
    not stand in for a place.  Returns ``(regions, keep)``: the patch indices of
    each surviving tile, and which tile of the coarse grid it was.
    """
    side = int(np.ceil(grid / coarse))
    regions, keep = [], []
    for ry in range(coarse):
        for rx in range(coarse):
            idx = [y * grid + x
                   for y in range(ry * side, min((ry + 1) * side, grid))
                   for x in range(rx * side, min((rx + 1) * side, grid))
                   if drop is None or not bool(drop[y * grid + x])]
            if idx:
                regions.append(torch.tensor(idx))
                keep.append(ry * coarse + rx)
    return regions, keep


def region_sum(per_patch: torch.Tensor, regions) -> torch.Tensor:
    """``[k, P] -> [k, R]`` by SUM, which is what makes regions add."""
    return torch.stack([per_patch[:, r].sum(-1) for r in regions], dim=-1)


# ------------------------------------------------------------------ movement


def movement(a: torch.Tensor, b: torch.Tensor) -> dict:
    """How far the map moved from ``a`` to ``b``, five ways.

    ``max_abs`` is the one the theorem is tested with: for a prompt-independent
    lens it is not merely small but bitwise zero.  The other four are scale-free
    or unit-carrying summaries for the lens that does move:

    ``rel``   :math:`\\|a-b\\| / \\tfrac12(\\|a\\|+\\|b\\|)`, so a map that is
              simply twice as strong does not count as having moved far
    ``cos``   one minus the cosine of the two centred maps: pure change of shape
    ``rho``   one minus Spearman: change of the *ordering* of the regions, which
              is all a pointing game could ever see
    ``top``   did the single strongest region change
    ``sum``   the signed difference of the totals; for our reading the totals are
              in logits, so this is a difference of budgets
    """
    a = a.detach().float().cpu().reshape(-1)
    b = b.detach().float().cpu().reshape(-1)
    d = (a - b)
    na, nb = float(a.norm()), float(b.norm())
    ac, bc = a - a.mean(), b - b.mean()
    cos = float((ac @ bc) / max(float(ac.norm()) * float(bc.norm()), 1e-12))
    rho = sps.spearmanr(a.numpy(), b.numpy()).statistic
    return dict(max_abs=float(d.abs().max()),
                rel=float(d.norm()) / max(0.5 * (na + nb), 1e-12),
                cos=1.0 - cos,
                rho=1.0 - (0.0 if np.isnan(rho) else float(rho)),
                top=int(int(a.argmax()) != int(b.argmax())),
                sum=float(a.sum() - b.sum()),
                norm=0.5 * (na + nb))


# -------------------------------------------------- the reading, with a probe


def ig_diag(model, batch, token_ids, layers, positions, means,
            answer_pos: int | None = None, steps: int = 4):
    r"""``ig_map`` plus the gradient's profile along the path.

    Identical arithmetic to :func:`jvlm.vmap.ig_map` -- the midpoint rule on
    :math:`h(\alpha) = \bar h + \alpha (h - \bar h)` -- but it also returns, per
    layer and per word, the mean gradient norm at each interpolant.  The ratio
    of the last interpolant's norm to the path mean is the direct test for the
    failure mode a first-order reading has and this one does not: if the answer
    is saturated the gradient at the model's own activation has collapsed, the
    ratio falls toward zero, and a point reading of that target would report
    that the picture is doing nothing.

    Returns ``(maps, diag)`` with ``maps[l] = [k, P]`` and
    ``diag[l] = {"alphas": [...], "gnorm": [k, steps], "sat": [k]}``.
    """
    layers = sorted(set(layers))
    if answer_pos is None:
        answer_pos = batch["input_ids"].shape[1] - 1
    pos = torch.as_tensor(positions, dtype=torch.long)
    last = model.n_layers - 1

    with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as rec:
        model.forward(batch)
        clean = {l: rec.activations[l][0, pos.to(rec.activations[l].device)]
                 .detach().float() for l in layers}

    big = model.replicate(batch, steps) if steps > 1 else batch
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    maps, diag = {}, {}
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
                logits = model.unembed(rec.activations[last][:, answer_pos])
                rows, gns = [], []
                for j, t in enumerate(token_ids):
                    g, = torch.autograd.grad(
                        logits[:, t].sum(), x,
                        retain_graph=(j < len(token_ids) - 1))
                    g = g.float()
                    gns.append(g.norm(dim=-1).mean(-1).cpu())      # [steps]
                    rows.append((g.mean(0) * dev).sum(-1).cpu())
        finally:
            hd.remove()
        maps[l] = torch.stack(rows)
        gn = torch.stack(gns)                                      # [k, steps]
        diag[l] = dict(alphas=alphas.tolist(), gnorm=gn,
                       sat=(gn[:, -1] / gn.mean(-1).clamp_min(1e-12)))
    return maps, diag


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the ROC, by the rank identity; ``nan`` if a class is empty."""
    score, label = np.asarray(score, float), np.asarray(label).astype(bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = sps.rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
