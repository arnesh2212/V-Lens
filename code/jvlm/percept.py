# SPDX-License-Identifier: Apache-2.0
"""Perceptual directions, and their split into a verbal copy and a use copy.

For a concept c, the direction a photograph writes is estimated as the
difference of means of the image-token residuals between pictures containing c
and pictures not containing c.  Nothing about words enters its construction.

The production channel at layer l has row space spanned by the pullbacks
``u_t = J_l^T W_U[t]``.  Projecting the perceptual direction onto that span
splits it in two:

    e_c = P e_c  +  (I - P) e_c
          ^^^^^^     ^^^^^^^^^^^^
       verbal copy    use copy

The verbal copy is the part of what the picture wrote that can, to first order,
move the output vocabulary.  The use copy is everything else.  The split is the
paper's central object: adding one should make the model *say* the concept, and
adding the other should make it *choose* the concept.

Estimation follows two rules that matter.  Directions are estimated on a split
half and evaluated on the other, so the verbalizable fraction is not inflated by
noise shared between numerator and denominator.  And the fraction is reported
against its null: a random direction already lands ``k/d`` of its energy in any
k-dimensional subspace, so the bare ratio means nothing on its own.
"""
from __future__ import annotations

import torch

from jvlm import cocodata, corpus, readout


def image_mean(model, image_ids, layers, prompt="Describe this image."):
    """Mean residual over image-token positions, per layer, over a set of images."""
    acc = {l: torch.zeros(model.d_model, device=model.device) for l in layers}
    n = 0
    for iid in image_ids:
        im = corpus.load_image(str(cocodata.image_path(iid)))
        b = model.chat(prompt, [im])
        ipos = readout.image_positions(model, b)
        if len(ipos) == 0:
            continue
        acts = readout.residuals(model, b, layers)
        for l in layers:
            acc[l] += acts[l][ipos].mean(0)
        n += 1
    return {l: acc[l] / max(n, 1) for l in layers}, n


def perceptual_direction(model, pos_ids, neg_ids, layers, *, split=True):
    """``{layer: (e_a, e_b)}`` -- the direction estimated on two disjoint halves.

    Two independent estimates let every downstream quantity be computed as a
    cross-split inner product, which cancels the noise that would otherwise
    inflate a ratio of quadratics.
    """
    if not split:
        mp, _ = image_mean(model, pos_ids, layers)
        mn, _ = image_mean(model, neg_ids, layers)
        return {l: ((mp[l] - mn[l]).float(),) * 2 for l in layers}
    half_p, half_n = len(pos_ids) // 2, len(neg_ids) // 2
    out = {}
    halves = []
    for p, n in ((pos_ids[:half_p], neg_ids[:half_n]),
                 (pos_ids[half_p:], neg_ids[half_n:])):
        mp, _ = image_mean(model, p, layers)
        mn, _ = image_mean(model, n, layers)
        halves.append({l: (mp[l] - mn[l]).float() for l in layers})
    for l in layers:
        out[l] = (halves[0][l], halves[1][l])
    return out


def channel_basis(model, lens, token_ids, layer, *, rank=None):
    """Orthonormal basis of the production channel's row space at ``layer``.

    ``rank`` truncates to the leading singular directions of the pullback
    dictionary.  Truncation is not cosmetic: the pullbacks of a large vocabulary
    span essentially the whole residual space, so an untruncated projector is
    close to the identity and says nothing.
    """
    V = readout.lens_vectors(model, lens, token_ids, layer)      # [k, d]
    V = V.float()
    if rank is None or rank >= V.shape[0]:
        Q, _ = torch.linalg.qr(V.T)
        return Q
    U, S, _ = torch.linalg.svd(V.T, full_matrices=False)
    return U[:, :rank]


def split_direction(e: torch.Tensor, Q: torch.Tensor):
    """``(verbal copy, use copy)`` -- the projection onto ``Q`` and the rest."""
    Q = Q.to(e.device, e.dtype)
    verbal = Q @ (Q.T @ e)
    return verbal, e - verbal


def verbalizable_fraction(e_a: torch.Tensor, e_b: torch.Tensor,
                          Q: torch.Tensor) -> tuple[float, float]:
    """Cross-split verbalizable fraction, and the null a random direction gives.

    ``<P e_a, e_b> / <e_a, e_b>`` rather than ``||P e||^2 / ||e||^2``: with the
    two estimates independent, the noise contributions to numerator and
    denominator cancel in expectation instead of both being inflated.
    """
    Q = Q.to(e_a.device, e_a.dtype)
    num = float((Q.T @ e_a) @ (Q.T @ e_b))
    den = float(e_a @ e_b)
    k, d = Q.shape[1], Q.shape[0]
    return (num / den if abs(den) > 1e-8 else float("nan")), k / d
