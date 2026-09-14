# SPDX-License-Identifier: Apache-2.0
r"""V-Lens: the site-to-answer response, derived rather than patched.

Ask the question the lens is supposed to answer, for an image token:

    what does the activation at this patch contribute to what the model is
    about to say?

Write it down.  Let :math:`a(x)` be the position where the response is produced,
and let

.. math::  F_{x,p}(h) = h_L[a(x)] \big|_{h_l[p] \leftarrow h}

be the map from the activation at image site :math:`p` to the final residual at
the answer.  The logits are :math:`\ell = W_U\,\mathrm{RMSNorm}(h_L[a])`.
Expand :math:`F` to first order about the corpus mean :math:`\bar h_l` of
image-token activations and differentiate the norm at the realised operating
point.  The contribution of this patch is

.. math::
   \Delta\ell \;=\; W_U\, D_x\, A_l\, (h - \bar h_l),

.. math::
   A_l = \mathbb{E}_{x,\;p\in\mathrm{img}(x)}
         \left[\frac{\partial h_L[a(x)]}{\partial h_l[p]}\right],
   \qquad
   D_x v = \gamma \odot \frac{\sqrt{d}}{\lVert h_L\rVert}
           \left(v - \hat h_L (\hat h_L \!\cdot\! v)\right).

That expression *is* V-Lens.  Nothing is bolted on; every term is forced by the
question.  The reference lens computes :math:`W_U\,\mathrm{norm}(J_l h)`, which
differs in four places, and each difference is separately ablatable:

**T -- the target.**  The reference injects a cotangent at every valid position
at once, so what it collects at source :math:`p` is
:math:`\sum_{p'\ge p} \partial h_L[p']/\partial h_l[p]`.  For text that is
sensible: every position is a place a token comes out.  An image token is not.
It is a place other positions *read*.  One of the couple of hundred terms in
that sum is the answer; the rest are other image tokens, dominated by the
residual stream's identity path carrying a patch to itself.  The
image-to-answer signal arrives diluted by two orders of magnitude.

**S -- the source.**  The average runs over all positions, mixing the transport
out of the visual buffer with the transport along the text.  They are different
operators and their mean estimates neither.

**M -- the expansion point.**  Reading :math:`J h` rather than
:math:`J(h-\bar h)` leaves in :math:`J\bar h`, a per-layer constant shared by
every patch.  It is a fixed logit offset with no patch-specific content, and at
early layers it is most of what the readout says -- which is why an
uncentred lens reads function words and byte fragments off image tokens.

**N -- the norm.**  Re-applying the norm to the transported vector,
:math:`\mathrm{norm}(Jh)`, rescales by :math:`\lVert Jh\rVert` instead of the
real :math:`\lVert h_L\rVert` and reinstates the radial component that the
norm's own Jacobian removes.  Differentiating is exact to first order; renormalising
is a heuristic that changes the direction.

Setting T back to all positions, S back to all positions, M back to the origin
and N back to renormalisation recovers the reference lens exactly, so the four
form a clean ablation ladder.  Cost is unchanged: the same ``d_model`` backward
passes per prompt, with a sparser cotangent.
"""
from __future__ import annotations

import torch

from jlens.lens import JacobianLens


def norm_jacobian_apply(model, v: torch.Tensor, h_final: torch.Tensor
                        ) -> torch.Tensor:
    r"""Apply :math:`D_x`, the final RMSNorm's Jacobian at ``h_final``, to ``v``.

    ``RMSNorm(x) = gamma * x * sqrt(d)/||x||`` has Jacobian
    ``gamma_i (sqrt(d)/||x||) (delta_ij - xhat_i xhat_j)``.  It is symmetric up
    to the diagonal ``gamma``, which is why the same routine serves both the
    readout and the direction: applying it to ``W_U[t]`` gives the exact
    gradient of that token's logit with respect to the site.
    """
    d = v.shape[-1]
    x = h_final.float().reshape(-1)
    nrm = x.norm().clamp_min(1e-6)
    xhat = x / nrm
    vf = v.float()
    gamma = getattr(model._norm, "weight", None)
    if gamma is not None:
        vf = vf * gamma.detach().float()
    return (d ** 0.5 / nrm) * (vf - (vf @ xhat).unsqueeze(-1) * xhat)


class VLens(JacobianLens):
    """A transport plus the mean it is expanded about.

    Serialises in the reference format with one extra key, so a V-Lens loads in
    the reference code (falling back to its readout) and a reference lens loads
    here (with a zero mean, i.e. the ``M`` ablation).
    """

    def __init__(self, jacobians, *, n_prompts, d_model, means=None,
                 target_policy="answer", position_policy="image"):
        super().__init__(jacobians, n_prompts=n_prompts, d_model=d_model)
        self.means = means or {l: torch.zeros(d_model) for l in self.source_layers}
        self.target_policy = target_policy
        self.position_policy = position_policy

    # ---------------------------------------------------------------- io

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        torch.save({"J": {l: J.to(dtype) for l, J in self.jacobians.items()},
                    "means": {l: v.float() for l, v in self.means.items()},
                    "n_prompts": self.n_prompts,
                    "source_layers": self.source_layers,
                    "d_model": self.d_model,
                    "target_policy": self.target_policy,
                    "position_policy": self.position_policy}, path)

    @classmethod
    def load(cls, path: str) -> "VLens":
        ck = torch.load(path, map_location="cpu", weights_only=True)
        d = ck["d_model"]
        return cls(ck["J"], n_prompts=ck["n_prompts"], d_model=d,
                   means=ck.get("means"),
                   target_policy=ck.get("target_policy", "answer"),
                   position_policy=ck.get("position_policy", "image"))

    @classmethod
    def from_jacobian_lens(cls, lens: JacobianLens) -> "VLens":
        """Wrap a reference lens with a zero mean -- the ``M`` ablation."""
        return cls(lens.jacobians, n_prompts=lens.n_prompts,
                   d_model=lens.d_model, target_policy="future",
                   position_policy="all")

    # ------------------------------------------------------------ readout

    @torch.no_grad()
    def read(self, model, residual: torch.Tensor, layer: int,
             h_final: torch.Tensor, *, centre=True, exact_norm=True
             ) -> torch.Tensor:
        r"""``W_U D_x A_l (h - \bar h_l)`` for residuals ``[..., d]``.

        ``centre`` and ``exact_norm`` are the M and N switches, so one call site
        produces every rung of the ablation ladder.
        """
        h = residual.float()
        if centre:
            h = h - self.means[layer].to(h.device, h.dtype)
        v = self.transport(h, layer)
        w = model._lm_head.weight
        if exact_norm:
            v = norm_jacobian_apply(model, v, h_final)
        else:
            v = model._norm(v.to(w.dtype)).float()
        return (v.to(w.dtype).to(w.device) @ w.T).float()

    def vectors(self, model, token_ids, layer: int, h_final: torch.Tensor,
                *, normalize: bool = True, exact_norm: bool = True
                ) -> torch.Tensor:
        r"""``A_l^T D_x^T W_U[t]`` -- the direction that writes token ``t``.

        With ``exact_norm`` this is the true gradient of the token's logit with
        respect to the site, rather than an approximation to it, which is what
        makes it the right thing to erase or to add.
        """
        W = model._lm_head.weight[list(token_ids)].detach().float()
        if exact_norm:
            W = norm_jacobian_apply(model, W, h_final)
        A = self.jacobians[layer].to(W.device).float()
        V = W @ A
        return V / V.norm(dim=-1, keepdim=True).clamp_min(1e-8) if normalize else V
