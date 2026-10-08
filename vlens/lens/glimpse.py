# SPDX-License-Identifier: Apache-2.0
r"""GLIMPSE -- a reimplementation of the closest published competitor.

GLIMPSE (Shen, *Gradient-Layer Importance Mapping for Prompted Visual Saliency
Explanation*, arXiv:2506.18985) is the one prior method in our comparison set
that is genuinely **prompt-conditioned**: its saliency depends on the question
asked and on the response the model actually produced, so unlike the logit lens
and the J-lens it cannot be byte-identical for two prompts on one image.  It is
therefore the comparison a reader will ask for, and omitting it would look like
avoiding it.

No reference implementation is published -- the paper carries no code link and
a GitHub search for the method name returns nothing -- so this is written from
the equations in the paper.  What follows is the pipeline, equation by equation,
with the two places where the paper is ambiguous marked and the resolution
stated, because an approximation that names itself is useful and one that does
not is worse than no comparison at all.

The sequence is :math:`x = [v_{1:K} \Vert p_{1:M} \Vert y_{1:T}]`; write
:math:`\mathcal V`, :math:`\mathcal P`, :math:`\mathcal Y` for the image,
prompt and generated index sets.  For each generated token :math:`t` the target
is its own logit :math:`z_t`, and :math:`g^{\ell h} = \partial z_t / \partial
A^{\ell h}` -- one backward pass per generated token, as the paper specifies.

  (5)  :math:`G^{\ell h} = \mathrm{ReLU}(g^{\ell h} \odot A^{\ell h})`
  (6)  :math:`w^{\ell h} = \mathrm{softmax}_h \bigl(\lambda^{-1}
       \sum_{ij} G^{\ell h}_{ij} \big/ \sum_{ij} \mathrm{ReLU}(g^{\ell h}_{ij})
       \bigr)`,  :math:`\lambda = 0.5`
  (8)  :math:`E^\ell = \sum_h w^{\ell h} G^{\ell h}`, row-normalised
  (9)  :math:`g^\ell = \Vert \sum_h g^{\ell h} \Vert_1`
  (10) :math:`s^\ell = \mathrm{softmax}_\ell(\lambda_d (\ell + 1))`,
       :math:`\lambda_d = 0.2`
  (11) :math:`\alpha^\ell = g^\ell s^\ell / \sum_k g^k s^k`
  (13) :math:`L^\ell = I_N + \alpha^\ell E^\ell`
  (14) :math:`R \leftarrow R + L^\ell R`, from :math:`R \leftarrow I_N`
  (15) :math:`a_t = |\mathcal P|^{-1} \sum_{i \in \mathcal P} R(t, i)`
  (16) :math:`v_t = |\mathcal V|^{-1} \sum_{i \in \mathcal V} R(t, i)`
  (17) :math:`p_t = \mathrm{softmax}(z)_t`
  (19) :math:`\beta_t = p_t w_t / \sum_k p_k w_k`, with :math:`w_t = a_t` for
       the *visual* map and :math:`v_t` for the prompt map (the paper crosses
       them: a token's visual saliency is weighted by how strongly it addresses
       the question)
  (20) :math:`\gamma_t = \sqrt{\beta^{\mathcal V}_t \beta^{\mathcal P}_t}`
  (23) :math:`\tilde R_m = \sum_{t \in \mathcal Y} \beta_t R(t, m)`

:math:`\lambda = 0.5` and :math:`\lambda_d = 0.2` are the paper's own ablation
optima; it states no other defaults.

**Two departures, both stated.**

*The row index.*  Equation 23 sums over rows :math:`t \in \mathcal Y`, the
generated token's own position.  Taken literally that is degenerate: under a
causal mask the logit :math:`z_t` is emitted by the row *before* :math:`y_t`,
so :math:`\partial z_t/\partial A` vanishes on row :math:`t` itself, every
:math:`E^\ell` has a zero row there, and :math:`R(t, \cdot)` never leaves
:math:`e_t` -- the map would be identically zero.  The only self-consistent
reading, and the one that matches the decoding picture the paper describes
("one backward pass per generated token"), is the query row that *produced*
:math:`z_t`: position ``start + i - 1`` for the ``i``-th answer token, so the
first answer token is explained from the last prompt position.  That is what is
implemented.

*One forward pass, not T.*  The paper decodes autoregressively.  Here the
answer is teacher-forced in a single pass and row ``start + i - 1`` is read for
token ``i``.  Under a causal mask this is not an approximation: rows at or
below ``start + i - 1`` cannot see tokens after them, so every attention entry
used for token ``i`` is bit-identical to the one the ``i``-th decoding step
would have produced.  It costs one forward instead of :math:`T`.

*One exactness note.*  Equation 14 doubles the diagonal at every layer, so
:math:`R` grows like :math:`2^L`.  The rollout here divides by two per layer.
That is a single global scalar shared by every token and every position, so it
changes neither :math:`\beta` (normalised across :math:`t`) nor any ranking
over positions; it only keeps float32 honest.

**What GLIMPSE is not given here.**  Nothing.  It reads every layer and every
head, while our reading is taken at one pre-registered layer; it sees the
model's own generated response; it uses gradients.  If it wins, it wins.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


LAMBDA_HEAD = 0.5      # eq. 6 softmax temperature, the paper's ablation optimum
LAMBDA_DEPTH = 0.2     # eq. 10 depth-prior temperature, likewise


# --------------------------------------------------------------- graph root

def _grad_root(mod, inp, out):
    """Give the frozen model a differentiable root at the token embeddings.

    Every parameter is ``requires_grad_(False)`` by construction, so a plain
    forward builds no graph at all and ``∂z/∂A`` does not exist.  Marking the
    embedding output makes one.  It is returned as a ``clone`` rather than as
    the leaf itself because Qwen3-VL's DeepStack writes its visual injection
    into the hidden states *in place*, and an in-place write to a leaf that
    requires grad is a hard autograd error -- the same trap the residual-write
    hooks elsewhere in this package avoid by never marking layer 0.
    """
    t = out[0] if not torch.is_tensor(out) else out
    t = t.detach().requires_grad_(True).clone()
    return (t, *out[1:]) if not torch.is_tensor(out) else t


def forward_with_attention(model, batch: dict):
    """``(last_hidden_state, attentions)`` for one teacher-forced pass.

    ``attentions`` is a tuple of ``[1, H, N, N]``, one per layer, each carrying
    a grad_fn.  Needs the model loaded with ``attn="eager"``; SDPA and flash
    kernels do not materialise the matrix and either return ``None`` or refuse.
    """
    hd = model._embed.register_forward_hook(_grad_root)
    try:
        b = {k: v for k, v in batch.items()}
        b["output_attentions"] = True
        out = model.forward(b)
    finally:
        hd.remove()
    att = getattr(out, "attentions", None)
    if att is None or any(a is None for a in att):
        raise RuntimeError(
            "the trunk returned no attention matrices; load the model with "
            "attn='eager' (SDPA never builds the [N, N] matrix GLIMPSE reads)")
    if not att[0].requires_grad:
        raise RuntimeError(
            "the attention matrices carry no grad_fn, so d z / d A does not "
            "exist: the embedding hook did not reach this forward path")
    return out.last_hidden_state, att


# ------------------------------------------------------------------- the map

def _depth_prior(L: int, temp: float, device) -> torch.Tensor:
    l = torch.arange(1, L + 1, dtype=torch.float32, device=device)
    return torch.softmax(temp * l, dim=0)                      # eq. 10


def _layer_matrix(A_l: torch.Tensor, g_l: torch.Tensor | None,
                  head_temp: float) -> torch.Tensor:
    """``E^l`` from one layer's attention and gradient, eqs. 5, 6, 8."""
    A = A_l[0].float()                                          # [H, N, N]
    if g_l is None:                                             # ablation
        G = A
        w = torch.full((A.shape[0],), 1.0 / A.shape[0], device=A.device)
    else:
        g = g_l[0].float()
        G = torch.relu(g * A)                                   # eq. 5
        num = G.sum(dim=(1, 2))
        den = torch.relu(g).sum(dim=(1, 2)).clamp_min(1e-12)
        w = torch.softmax((num / den) / head_temp, dim=0)       # eq. 6
    E = (w[:, None, None] * G).sum(0)                           # eq. 8
    return E / E.sum(-1, keepdim=True).clamp_min(1e-12)         # row-normalised


def _rollout_row(E: list[torch.Tensor], alpha: torch.Tensor,
                 row: int, N: int) -> torch.Tensor:
    r"""Row ``row`` of :math:`R` after eqs. 13--14, computed as a row vector.

    :math:`R = (I + L^L)\cdots(I + L^1)`, so ``e_row^T R`` is a left-multiply
    swept from the deepest layer to the shallowest.  That is ``L`` vector-matrix
    products instead of ``L`` matrix-matrix products and is exact, not an
    approximation -- the full :math:`N \times N` relevance is never needed
    because equations 15, 16 and 23 only ever read one row of it.
    """
    v = torch.zeros(N, device=E[0].device, dtype=torch.float32)
    v[row] = 1.0
    for l in range(len(E) - 1, -1, -1):
        v = v + 0.5 * alpha[l] * (v @ E[l])     # (2 v + alpha v E) / 2
    return v


def maps(model, batch: dict, positions: Sequence[int], *,
         answer_start: int, answer_ids, gradient: bool = True,
         head_temp: float = LAMBDA_HEAD, depth_temp: float = LAMBDA_DEPTH
         ) -> dict:
    """The full GLIMPSE reading for one teacher-forced prompt+answer batch.

    Args:
        batch: prompt with the answer already appended, from
            :func:`vlens.objective.append_answer`.
        positions: the image-token indices, :math:`\\mathcal V`.
        answer_start: sequence index of the answer's first token.
        answer_ids: the answer's token ids, :math:`y_{1:T}`.
        gradient: ``False`` runs the same pipeline with :math:`G = A`, uniform
            head weights and the depth prior alone -- attention rollout under
            GLIMPSE's own aggregation, which is what isolates what the
            gradients buy.

    Returns:
        ``visual`` ``[P]``, the response-level heat over image positions (the
        thing the harness grades); ``prompt`` ``[M]``, the same over prompt
        positions; ``beta``/``gamma``/``p`` ``[T]``, the token weights of
        eqs. 19-20 and the model's own confidence.
    """
    ids = torch.as_tensor(answer_ids, dtype=torch.long).flatten()
    T = int(ids.numel())
    if T == 0:
        raise ValueError("GLIMPSE explains a response; this one is empty")
    N = int(batch["input_ids"].shape[1])
    ipos = torch.as_tensor(positions, dtype=torch.long)
    imask = model.image_mask(batch["input_ids"])[0].cpu()
    ppos = torch.tensor([i for i in range(answer_start) if not bool(imask[i])],
                        dtype=torch.long)
    rows = [answer_start - 1 + i for i in range(T)]
    if rows[0] < 0 or rows[-1] >= N:
        raise ValueError(f"answer rows {rows[0]}..{rows[-1]} outside [0, {N})")

    with torch.enable_grad():
        hs, A = forward_with_attention(model, batch)
        L = len(A)
        logits = model.unembed(hs[0, torch.tensor(rows, device=hs.device)])
        logits = logits.float()                                  # [T, V]
        tgt = ids.to(logits.device).view(-1, 1)
        z = logits.gather(1, tgt).squeeze(1)                     # [T]
        p = torch.softmax(logits, dim=-1).gather(1, tgt).squeeze(1).detach()
        s = _depth_prior(L, depth_temp, hs.device)

        rel = torch.zeros(T, N, device=hs.device)
        for i in range(T):
            if gradient:
                g = torch.autograd.grad(z[i], A, retain_graph=(i < T - 1),
                                        allow_unused=True)
                g = [torch.zeros_like(A[l]) if g[l] is None else g[l]
                     for l in range(L)]
                gl = torch.stack([gg[0].float().sum(0).abs().sum() for gg in g])
                alpha = gl * s
                alpha = alpha / alpha.sum().clamp_min(1e-12)     # eqs. 9, 11
                E = [_layer_matrix(A[l], g[l], head_temp) for l in range(L)]
                del g
            else:
                alpha = s
                E = [_layer_matrix(A[l], None, head_temp) for l in range(L)]
            rel[i] = _rollout_row(E, alpha, rows[i], N)
            del E
    rel = rel.detach()

    a = rel[:, ppos.to(rel.device)].mean(-1)                     # eq. 15
    v = rel[:, ipos.to(rel.device)].mean(-1)                     # eq. 16
    bV = _beta(p, a)          # visual map is weighted by prompt alignment
    bP = _beta(p, v)          # and the prompt map by visual alignment (eq. 19)
    gamma = torch.sqrt(bV * bP)                                  # eq. 20
    return {
        "visual": (bV[:, None] * rel[:, ipos.to(rel.device)]).sum(0).cpu().numpy(),
        "prompt": (bP[:, None] * rel[:, ppos.to(rel.device)]).sum(0).cpu().numpy(),
        "beta": bV.cpu().numpy(), "beta_prompt": bP.cpu().numpy(),
        "gamma": gamma.cpu().numpy(), "p": p.cpu().numpy(),
        "n_layers": L, "n_tokens": T, "rows": rows,
    }


def _beta(p: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    b = p * w
    tot = b.sum()
    return b / tot if float(tot) > 0 else torch.full_like(b, 1.0 / b.numel())


# ------------------------------------------------------- the bench interface

class GLIMPSE:
    """:class:`vlens.bench.Attribution` -- one score per image position.

    The harness contract is ``(model, batch, layer, positions) -> [P]``.  Two
    things about how GLIMPSE meets it are worth saying out loud rather than
    burying, because both cut in *its* favour:

    ``layer`` is ignored.  GLIMPSE is a whole-network method: equations 9-14
    aggregate every layer with a learned-free depth prior.  There is no single
    layer at which to read it, so it returns the same vector whichever layer the
    sweep asks for, while our reading is held to one pre-registered layer.

    The answer travels in the batch.  ``batch['_answer_start']`` and
    ``batch['_answer_ids']`` carry the response being explained; keys with a
    leading underscore are already stripped before ``model.forward``, which is
    this package's existing convention for exactly this.  So no special-casing
    reaches the harness.
    """

    def __init__(self, name: str = "GLIMPSE", *, gradient: bool = True,
                 head_temp: float = LAMBDA_HEAD,
                 depth_temp: float = LAMBDA_DEPTH):
        self.name = name
        self.gradient = gradient
        self.head_temp = head_temp
        self.depth_temp = depth_temp

    def __call__(self, model, batch: dict, layer: int,
                 positions: Sequence[int]) -> np.ndarray:
        if "_answer_start" not in batch or "_answer_ids" not in batch:
            raise KeyError(
                "GLIMPSE explains a generated response: put the answer in the "
                "batch as '_answer_start' and '_answer_ids' (see "
                "vlens.lens.glimpse.prepare)")
        return maps(model, batch, positions,
                    answer_start=int(batch["_answer_start"]),
                    answer_ids=batch["_answer_ids"],
                    gradient=self.gradient, head_temp=self.head_temp,
                    depth_temp=self.depth_temp)["visual"]

    def __repr__(self) -> str:
        return (f"GLIMPSE(name={self.name!r}, gradient={self.gradient}, "
                f"head_temp={self.head_temp}, depth_temp={self.depth_temp})")


def prepare(model, prompt_batch: dict, answer_ids) -> dict:
    """Teacher-force an answer and tag the batch for :class:`GLIMPSE`."""
    from . import objective as _obj
    b, start = _obj.append_answer(model, prompt_batch, answer_ids)
    b["_answer_start"] = int(start)
    b["_answer_ids"] = torch.as_tensor(answer_ids, dtype=torch.long).flatten()
    return b
