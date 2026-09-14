# SPDX-License-Identifier: Apache-2.0
r"""What scalar the lens is reading -- answer-level objectives for :math:`F`.

Every reading in this paper is a decomposition of one scalar.  The lens is

.. math::
   A_R(q) = \sum_{p \in R} (h_p - \bar h_p)^{\!\top}
            \int_0^1 \nabla_{h_p} F\bigl(\bar h + \alpha (h - \bar h), q\bigr)
            \, \mathrm{d}\alpha ,

and the conservation law that makes the region coordinate well posed,

.. math::
   \sum_R A_R(q) = F(h, q) - F(\bar h, q),

is the fundamental theorem of calculus along the straight path.  It therefore
holds for *any* :math:`F` that is differentiable in :math:`h`.  Nothing in it
knows that :math:`F` was a logit.

That matters because until now :math:`F` was a single answer token's logit, and
a reader is entitled to ask whether the result is an artefact of choosing one
convenient object token -- whether ``logit("dog")`` was picked because it
behaved.  This module makes the choice a parameter and supplies the two
answer-level objectives that remove the objection:

``margin``  (i)
    :math:`\log P(\text{correct} \mid \text{image}, q) -
    \log P(\text{foil} \mid \text{image}, q)`, teacher-forced over the *whole*
    answer sequence and length-normalised.  This is the model's forced-choice
    decision variable, which is what a double dissociation actually measures.

``sequence_logprob``  (ii)
    the teacher-forced log-likelihood of a complete answer, normalised the same
    way.  No foil, no contrast -- just how strongly the picture supports the
    thing the model said.

``token_logit``  (iii)
    the existing single-token logit, unchanged, kept so every earlier number is
    reproducible and so the three can be compared on the same items.

**The normalisation, stated.**  ``norm="tokens"`` (the default) divides each
continuation's summed log-probability by *its own* teacher-forced token count,
so :math:`F` is a difference of mean per-token log-probabilities, in nats per
token.  This is the ``acc_norm`` convention of the LM evaluation harnesses and
the :math:`\alpha = 1` case of the GNMT length penalty; it exists because an
unnormalised sum charges a longer continuation for every extra token, so a
model that merely dislikes long strings would look like a model that dislikes
the foil.  ``norm="shared"`` divides *both* branches of a pair by one constant,
the mean of the two lengths: :math:`F` is then a positive rescaling of the raw
log-likelihood ratio, so its sign and its ordering are exactly the model's
forced choice, which is the right setting when the pair is length-matched by
construction (Winoground captions, SugarCrepe swaps) and the reading has to
decompose the behavioural score itself.  ``norm="none"`` is the raw ratio.
On Winoground the choice turns out to do no work at all: the caption pairs are
anagrams, every pair measured so far tokenises to equal lengths, and
``"tokens"`` and ``"shared"`` return the identical number.  That is worth
knowing before the choice is argued about.
Whichever the behavioural experiment scores with is the one the lens must read,
because the conservation law equates the reading to that exact quantity's
response to replacing the picture -- read a different :math:`F` and the sum is
the difference-in-differences of a different experiment.

In all three cases the divisor is a **constant** fixed by teacher forcing: it
depends on the token counts, not on :math:`h`.  So :math:`F` remains a fixed
differentiable function of the activations and the conservation law is
untouched by the normalisation.  (It is *not* untouched by the discretisation:
a log-softmax is more curved than a logit, so the midpoint rule needs more
steps to reach the same completeness.  Measured, not assumed -- see
:func:`completeness` and the ``--gpu`` self-check in this file.)

**How it plugs in.**  An :class:`Objective` is a list of :class:`Variant`\ s.  A
variant is one forward pass -- one batch, one reader of the final residual --
and the objective's ``k`` readings are fixed linear combinations of the
variants' outputs.  The margin needs two forward passes because the correct and
the foil answer are different token sequences and cannot share an
``input_ids``; because the gradient is linear, running them separately and
subtracting the gradients is exactly differentiating the difference, and it
costs the memory of one pass rather than two.

:func:`ig_map` here is :func:`jvlm.vmap.ig_map` generalised to an arbitrary
objective.  ``jvlm.vmap`` is not modified and its default behaviour is not
changed by a single character: :func:`selftest` asserts that this function with
:func:`token_logit` reproduces ``vmap.ig_map`` bit for bit, which is the
backward-compatibility guarantee.

The existing ``vmap.ig_map`` can nevertheless read the new objectives, because
it touches a model only through ``layers``, ``n_layers``, ``forward`` and
``unembed``, and it takes the scalar it integrates as ``unembed(h)[:, t]``.
:class:`ObjectiveModel` is a model-shaped adapter that runs the objective's
variants inside ``forward`` and returns the ``[B, k]`` objective values where
the logits used to be, so ``token_ids`` becomes ``range(k)`` and everything
built on top of ``vmap`` -- ``unit=``, ``vmap.completeness``,
:func:`jvlm.budget.ig_map_chunked` -- keeps working:

.. code-block:: python

   mm, bb, ids = objective.vmap_target(m, objective.margin(m, b, pairs))
   scores = vmap.ig_map(mm, bb, ids, LAYERS, IPOS, means=MEANS, steps=16)

A single-variant objective is additionally callable as
``obj(model, resid_last) -> [B, k]``, which is the shape ``vmap.ig_map`` would
take as an ``objective=`` keyword if that file is ever given one.

Run ``python code/jvlm/objective.py --cpu`` for the unit tests (no GPU, no
model download) and ``code/exp/91_objcheck.py`` for the completeness check on a
real VLM.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from jlens.hooks import ActivationRecorder

NORMS = ("none", "tokens", "shared")

__all__ = [
    # the three objectives
    "token_logit", "sequence_logprob", "margin", "NORMS",
    # the estimator and its checks
    "ig_map", "value", "completeness", "rows_independent", "variant_drift",
    # plumbing
    "Objective", "Variant", "TokenLogitReader", "SeqLogProbReader",
    "ObjectiveModel", "vmap_target", "append_answer", "answer_ids",
    "generated_answer",
    # tests and the VLM check
    "selftest", "gpu_check", "toy_model", "toy_batch",
]


# --------------------------------------------------------------- the readers
#
# A reader turns the final-layer residual of one forward pass into ``[B, m]``
# raw numbers.  It never applies the normalisation: the divisor lives in the
# variant's weight matrix, so a continuation that appears in two different
# pairs (with two different partners, and so two different divisors under
# ``norm="shared"``) is still only run once.


class TokenLogitReader:
    """``logits[:, answer_pos, ids]`` -- the original single-token reading."""

    def __init__(self, ids: Sequence[int], answer_pos: int | None = None):
        self.ids = list(ids)
        self.answer_pos = answer_pos

    def __call__(self, model, resid: torch.Tensor) -> torch.Tensor:
        p = resid.shape[1] - 1 if self.answer_pos is None else self.answer_pos
        return model.unembed(resid[:, p])[:, self.ids].float()

    def __repr__(self) -> str:
        return f"TokenLogitReader(k={len(self.ids)}, pos={self.answer_pos})"


class SeqLogProbReader:
    r"""Teacher-forced :math:`\sum_i \log p(a_i \mid \text{prompt}, a_{<i})`.

    The continuation occupies sequence positions ``start .. start+L-1``, so the
    distribution over :math:`a_i` is read from the residual at ``start+i-1``:
    the slice is ``[start-1, start+L-1)``, and the residual above the
    continuation's *last* token is never used.

    The log-probability is formed as ``logit[target] - logsumexp(logits)``
    rather than by materialising a second ``[B, L, V]`` tensor from
    ``log_softmax``; ``V`` is 150k on the Qwen family and the path is run
    ``steps`` rows at a time, so that copy is the difference between fitting on
    the card and not.
    """

    def __init__(self, start: int, ids: torch.Tensor):
        self.start = int(start)
        self.ids = torch.as_tensor(ids, dtype=torch.long).flatten()

    def __call__(self, model, resid: torch.Tensor) -> torch.Tensor:
        L = int(self.ids.numel())
        sl = resid[:, self.start - 1: self.start - 1 + L]        # [B, L, d]
        lg = model.unembed(sl).float()                           # [B, L, V]
        tgt = self.ids.to(lg.device).view(1, L, 1).expand(lg.shape[0], L, 1)
        lp = lg.gather(-1, tgt).squeeze(-1) - lg.logsumexp(-1)   # [B, L]
        return lp.sum(-1, keepdim=True)                          # [B, 1]

    def __repr__(self) -> str:
        return f"SeqLogProbReader(start={self.start}, L={int(self.ids.numel())})"


# ------------------------------------------------------- variants and objects


@dataclass
class Variant:
    """One forward pass and how its outputs enter the objective.

    Attributes:
        batch: a batch of one, ready for ``model.forward``.
        reader: ``(model, resid_last) -> [B, m]``.
        weight: ``[m, k]``, the fixed coefficients -- signs and normalising
            constants -- with which this pass's outputs enter each of the
            objective's ``k`` readings.
        name: for the diagnostics only.
    """

    batch: dict
    reader: Callable[..., torch.Tensor]
    weight: torch.Tensor
    name: str = ""


@dataclass
class Objective:
    r"""A differentiable scalar :math:`F(h, q)`, or ``k`` of them at once.

    ``F[:, j] = sum_v reader_v(resid_v) @ weight_v[:, j]``.  The linearity is
    not a restriction on what can be read -- a reader is an arbitrary
    differentiable function of the residual, and the log-likelihood one is
    thoroughly nonlinear -- it is a statement about how several *forward
    passes* combine, which is what lets the margin be differentiated with one
    pass's worth of memory.
    """

    variants: list[Variant]
    names: list[str]
    kind: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.names)

    @property
    def batch(self) -> dict:
        """The single variant's batch, for objectives that have one."""
        if len(self.variants) != 1:
            raise ValueError(
                f"{self.kind!r} needs {len(self.variants)} forward passes; "
                "there is no single batch.  Use obj.variants[i].batch.")
        return self.variants[0].batch

    def image_positions(self, model) -> torch.Tensor:
        """Image-token indices, which every variant shares.

        Image tokens precede the text under a causal mask, so appending a
        different answer to the prompt cannot move them -- but that is an
        argument, and :func:`variant_drift` is the measurement.
        """
        return model.image_mask(self.variants[0].batch["input_ids"])[0] \
            .nonzero(as_tuple=True)[0].cpu()

    def __call__(self, model, resid: torch.Tensor) -> torch.Tensor:
        """``[B, k]`` from one pass's final residual (single-variant only).

        This is the form ``jvlm.vmap.ig_map`` can take as a drop-in target.
        """
        if len(self.variants) != 1:
            raise ValueError(
                f"{self.kind!r} spans {len(self.variants)} forward passes and "
                "cannot be evaluated from a single residual; use "
                "jvlm.objective.ig_map, which loops over the variants.")
        v = self.variants[0]
        out = v.reader(model, resid).float()
        return out @ v.weight.to(out.device, out.dtype)

    def __repr__(self) -> str:
        return (f"Objective({self.kind}, k={self.k}, "
                f"passes={len(self.variants)}, names={self.names[:4]})")


# ----------------------------------------------------------- batch surgery

# Sequence-shaped keys we know how to extend, and what to pad them with.  An
# answer is text, so every modality marker takes its text value: Gemma3's
# ``token_type_ids`` is 0 for text and 1 for image, and Qwen3-VL's
# ``mm_token_type_ids`` is 0 for text, 1 for image, 2 for video (verified on
# the processor's own output, not assumed from the name).
#
# ``input_ids`` takes the answer's own ids; anything else in the batch that is
# sequence-shaped and not listed here is refused rather than guessed at, on the
# chat_batch precedent: silently padding a key we do not understand would
# corrupt the forward pass in a way no assertion downstream would catch.  That
# refusal has already earned its keep -- it is what caught ``mm_token_type_ids``
# on the first real Qwen3-VL batch instead of letting a wrong marker through.
#
# Checked against every family in the paper: Qwen3-VL emits
# ``mm_token_type_ids``, Gemma3 emits ``token_type_ids``, LLaVA-1.5 and
# InternVL3 emit neither, and none of the four emits a sequence-shaped key
# outside this map.
_SEQ_PAD = {"attention_mask": 1, "token_type_ids": 0, "mm_token_type_ids": 0}


def append_answer(model, batch: dict, ids) -> tuple[dict, int]:
    """Teacher-force ``ids`` onto a prompt batch.

    Returns the extended batch and ``start``, the sequence index at which the
    answer's first token sits.  Vision tensors are untouched: the picture is
    the same picture.
    """
    ids = torch.as_tensor(ids, dtype=torch.long).flatten()
    if ids.numel() == 0:
        raise ValueError("empty continuation: nothing to score")
    S = int(batch["input_ids"].shape[1])
    out: dict = {}
    for k, v in batch.items():
        if not torch.is_tensor(v) or k in model.layout.image_replicate:
            out[k] = v
        elif v.ndim == 2 and v.shape[1] == S:
            if k == "input_ids":
                tail = ids.to(v.device, v.dtype)[None].expand(v.shape[0], -1)
            elif k in _SEQ_PAD:
                tail = torch.full((v.shape[0], ids.numel()), _SEQ_PAD[k],
                                  dtype=v.dtype, device=v.device)
            else:
                raise ValueError(
                    f"append_answer does not know how to extend {k!r} "
                    f"(shape {tuple(v.shape)}); extending it wrongly would "
                    "silently change the forward pass")
            out[k] = torch.cat([v, tail], dim=1)
        elif v.ndim >= 2 and S in v.shape[1:]:
            raise ValueError(
                f"append_answer found a sequence-shaped {k!r} "
                f"(shape {tuple(v.shape)}) it cannot extend; pass a batch "
                "whose position ids the model recomputes")
        else:
            out[k] = v
    return out, S


@torch.no_grad()
def generated_answer(model, batch: dict, *, max_new_tokens: int = 32,
                     strip_eos: bool = True) -> torch.Tensor:
    """The ids the model actually produces, greedily, for this prompt.

    Objective (ii) is the log-likelihood of *the complete generated answer*, so
    the answer has to come from the model rather than from the dataset: this
    generates it once, and the ids it returns are then teacher-forced by
    :func:`sequence_logprob`.  Greedy, so the answer is a deterministic
    function of the prompt and the reading is reproducible.

    ``strip_eos`` drops the trailing end-of-turn tokens, so the score is over
    the answer's text and not over the model's decision to stop.  Keep them if
    the question is about the stopping.
    """
    kw = {k: v for k, v in batch.items()
          if k != "labels" and not k.startswith("_")}
    out = model.hf.generate(**kw, max_new_tokens=max_new_tokens,
                            do_sample=False)
    ids = out[0, int(batch["input_ids"].shape[1]):].detach().cpu()
    if strip_eos:
        stop = {model.tokenizer.eos_token_id,
                getattr(model.tokenizer, "pad_token_id", None)} - {None}
        while ids.numel() > 1 and int(ids[-1]) in stop:
            ids = ids[:-1]
    return ids


def answer_ids(model, text: str) -> torch.Tensor:
    """Tokenise a continuation, with no special tokens added.

    The continuation is tokenised on its own, which is what the LM evaluation
    harnesses do and what makes the teacher-forced span well defined.  It also
    means the caller owns the leading space: after a chat template's assistant
    header ``"yes"`` is right and ``" yes"`` is a different first token, while
    mid-sentence the opposite holds.  Pass the string you want scored.
    """
    tok = model.tokenizer
    enc = tok(text, add_special_tokens=False, return_tensors="pt")
    return torch.as_tensor(enc["input_ids"]).flatten()


# ------------------------------------------------------------- constructors


def token_logit(batch: dict, token_ids: Sequence[int], *,
                answer_pos: int | None = None,
                weight: torch.Tensor | None = None,
                names: Sequence[str] | None = None) -> Objective:
    """(iii) The single-token logit, exactly as before.

    ``weight`` is an optional ``[len(token_ids), k]`` matrix, so a contrast of
    token logits -- ``logit(yes) - logit(no)``, the margin map the Winoground
    run already reports -- is expressible without a second pass, because every
    reading here is linear in the logits.  Its default is the identity, which
    reproduces ``vmap.ig_map``'s behaviour token by token.

    Args:
        batch: the prompt batch; ``answer_pos`` defaults to its last position.
    """
    ids = list(token_ids)
    w = torch.eye(len(ids)) if weight is None else \
        torch.as_tensor(weight, dtype=torch.float32).reshape(len(ids), -1)
    nm = list(names) if names is not None else (
        [str(t) for t in ids] if weight is None
        else [f"c{j}" for j in range(w.shape[1])])
    if len(nm) != w.shape[1]:
        raise ValueError(f"{len(nm)} names for {w.shape[1]} readings")
    return Objective(
        variants=[Variant(batch, TokenLogitReader(ids, answer_pos), w,
                          name="logit")],
        names=nm, kind="token_logit",
        meta={"token_ids": ids, "answer_pos": answer_pos})


def _norm_z(norm: str, lengths: Sequence[int]) -> list[float]:
    """The divisor for each continuation, given the lengths in its pair."""
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
    if norm == "none":
        return [1.0] * len(lengths)
    if norm == "tokens":
        return [float(L) for L in lengths]
    z = float(sum(lengths)) / len(lengths)      # "shared": one constant
    return [z] * len(lengths)


def _build(model, batch, texts, coefs, k, kind, meta) -> Objective:
    """Deduplicate continuations into variants and assemble the weights.

    ``coefs`` is ``{index into texts: {column: coefficient}}``.  Two pairs that
    share a continuation share its forward pass; under ``norm="shared"`` they
    may still weight it differently, which is why the normalisation lives in
    the weight and not in the reader.
    """
    seen: dict[tuple, int] = {}
    variants: list[Variant] = []
    for i, ids in enumerate(texts):
        key = tuple(int(t) for t in ids)
        if key not in seen:
            b, start = append_answer(model, batch, ids)
            seen[key] = len(variants)
            variants.append(Variant(
                b, SeqLogProbReader(start, ids), torch.zeros(1, k),
                name=f"seq{len(variants)}"))
        v = variants[seen[key]]
        for col, c in coefs[i].items():
            v.weight[0, col] += c
    return Objective(variants=variants,
                     names=meta.pop("names"), kind=kind, meta=meta)


def sequence_logprob(model, batch: dict, answers, *, norm: str = "tokens",
                     names: Sequence[str] | None = None) -> Objective:
    """(ii) Teacher-forced log-likelihood of a complete answer.

    Args:
        batch: the prompt batch, from ``model.chat``.
        answers: strings, or already-tokenised id sequences.  Each is one
            reading; each costs one forward pass.
        norm: ``"tokens"`` divides by the answer's own token count (nats per
            token); ``"none"`` leaves the summed log-likelihood alone.
            ``"shared"`` is meaningless without a partner to share with and is
            treated as ``"none"``.
    """
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
    ids = [a if torch.is_tensor(a) or isinstance(a, (list, tuple))
           else answer_ids(model, a) for a in answers]
    ids = [torch.as_tensor(x, dtype=torch.long).flatten() for x in ids]
    lens = [int(x.numel()) for x in ids]
    z = [1.0] * len(ids) if norm in ("none", "shared") else \
        [float(L) for L in lens]
    nm = list(names) if names is not None else [
        (a if isinstance(a, str) else f"seq{j}")[:40]
        for j, a in enumerate(answers)]
    coefs = [{j: 1.0 / z[j]} for j in range(len(ids))]
    return _build(model, batch, ids, coefs, len(ids), "sequence_logprob",
                  {"names": nm, "norm": norm, "lengths": lens})


def margin(model, batch: dict, pairs, *, norm: str = "tokens",
           names: Sequence[str] | None = None) -> Objective:
    r"""(i) Correct-versus-foil margin over full answer sequences.

    .. math::
       F_j = \frac{\log P(c_j \mid \text{image}, q)}{Z^c_j}
           - \frac{\log P(f_j \mid \text{image}, q)}{Z^f_j}

    both branches teacher-forced over the whole sequence.  See the module
    docstring for the three normalisations and when each is the defensible one;
    the short version is that ``"shared"`` preserves the sign of the model's
    forced choice and ``"tokens"`` (the default) removes the length bias, and
    they coincide up to a positive constant when the pair is length-matched.

    Args:
        pairs: ``(correct, foil)``, strings or id sequences.  Continuations
            shared between pairs are run once.
    """
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
    texts: list[torch.Tensor] = []
    coefs: list[dict] = []
    lens: list[tuple[int, int]] = []
    for j, (c, f) in enumerate(pairs):
        ic = c if torch.is_tensor(c) or isinstance(c, (list, tuple)) \
            else answer_ids(model, c)
        jf = f if torch.is_tensor(f) or isinstance(f, (list, tuple)) \
            else answer_ids(model, f)
        ic = torch.as_tensor(ic, dtype=torch.long).flatten()
        jf = torch.as_tensor(jf, dtype=torch.long).flatten()
        zc, zf = _norm_z(norm, [int(ic.numel()), int(jf.numel())])
        texts += [ic, jf]
        coefs += [{j: 1.0 / zc}, {j: -1.0 / zf}]
        lens.append((int(ic.numel()), int(jf.numel())))
    nm = list(names) if names is not None else [
        (str(p[0])[:32] if isinstance(p[0], str) else f"pair{j}")
        for j, p in enumerate(pairs)]
    return _build(model, batch, texts, coefs, len(pairs), "margin",
                  {"names": nm, "norm": norm, "lengths": lens})


# ------------------------------------------------------------------- engine


def _write_hook(vals: torch.Tensor, pos: torch.Tensor):
    """Forward hook replacing the given positions with ``vals``.

    ``vals`` is ``[P, d]`` (the same for every row) or ``[B, P, d]`` (a row
    each), which is what makes the path's ``steps`` interpolants one batch.
    """
    def hook(mod, inp, o, _v=vals, _p=pos):
        tup = not torch.is_tensor(o)
        hh = (o[0] if tup else o).clone()
        hh[:, _p.to(hh.device)] = _v.to(hh.device).to(hh.dtype)
        return (hh, *o[1:]) if tup else hh
    return hook


def _pass(model, obj: Objective, layer: int, pos: torch.Tensor,
          x: torch.Tensor, *, cols=(), sink=None, want_vals: bool = True):
    """Run every variant with ``x`` written in at ``layer``.

    Returns ``[B, k]`` values (when asked) and hands each requested column's
    gradient to ``sink(column, grad)`` as it is produced, so the caller can
    reduce it immediately.  Holding all ``k`` gradients would be ``k * B * P *
    d`` floats -- gigabytes for a vocabulary-sized reading -- which is why the
    gradient is pushed rather than returned.
    """
    n = int(x.shape[0])
    last = model.n_layers - 1
    cols = list(cols)
    total = None
    for v in obj.variants:
        big = model.replicate(v.batch, n) if n > 1 else v.batch
        hd = model.layers[layer].register_forward_hook(_write_hook(x, pos))
        try:
            ctx = torch.enable_grad() if cols else torch.no_grad()
            with ctx, ActivationRecorder(model.layers, at=[last]) as rec:
                model.forward(big)
                out = v.reader(model, rec.activations[last]).float()
                c = out @ v.weight.to(out.device, out.dtype)     # [n, k]
                if want_vals:
                    d = c.detach().float().cpu()
                    total = d if total is None else total + d
                for i, j in enumerate(cols):
                    g, = torch.autograd.grad(
                        c[:, j].sum(), x, retain_graph=(i < len(cols) - 1))
                    sink(j, g.detach())
        finally:
            hd.remove()
    return total


def ig_map(model, obj: Objective, layers, positions, means: dict, *,
           steps: int = 4, chunk: int | None = None,
           unit: bool = False) -> dict[int, torch.Tensor]:
    r"""``{layer: [k, P]}`` -- :func:`jvlm.vmap.ig_map` for an arbitrary ``F``.

    Same path, same midpoint rule, same baseline, same batching of the
    interpolants; the only change is that the scalar being integrated comes
    from ``obj`` instead of from one entry of the unembedding.  With
    :func:`token_logit` it reproduces ``vmap.ig_map`` to floating point --
    :func:`selftest` asserts it.

    Args:
        means: the baseline :math:`\bar h_\ell[p]`, per layer, ``[P, d]``.
        steps: interpolants on the path.  The answer-level objectives put a
            log-softmax between the residual and ``F``, which is more curved
            than a bare logit, so they want more steps than the token reading
            does for the same completeness.  Measure it; do not guess.
        chunk: how many interpolants are held at once.  The estimate does not
            depend on it, only the peak memory and the number of forwards --
            the same trade :func:`jvlm.budget.ig_map_chunked` makes.
        unit: normalise the path-averaged gradient before the dot product.
            Trades the conservation law for a scale-free reading, and costs a
            ``[k, P, d]`` buffer because the normalisation has to happen after
            the variants are summed, not before.
    """
    layers = sorted(set(layers))
    steps = int(steps)
    chunk = steps if chunk is None else max(1, min(int(chunk), steps))
    pos = torch.as_tensor(positions, dtype=torch.long)
    k, P = obj.k, int(pos.numel())

    with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as rec:
        model.forward(obj.variants[0].batch)
        clean = {l: rec.activations[l][0, pos.to(rec.activations[l].device)]
                 .detach().float() for l in layers}

    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    out: dict[int, torch.Tensor] = {}
    for l in layers:
        h = clean[l]
        mu = means[l].to(h.device).float()
        dev = h - mu
        gacc = (torch.zeros(k, *dev.shape, device=h.device) if unit else None)
        sacc = torch.zeros(k, P, dtype=torch.float64)

        for c0 in range(0, steps, chunk):
            al = alphas[c0: c0 + chunk].to(h.device)
            x = (mu[None] + al[:, None, None] * dev[None]
                 ).detach().clone().requires_grad_(True)

            def sink(j, g, _g=gacc, _s=sacc, _d=dev):
                if _g is not None:
                    _g[j] += g.float().sum(0)
                else:
                    _s[j] += (g.float().sum(0) * _d).sum(-1).double().cpu()

            _pass(model, obj, l, pos, x, cols=range(k), sink=sink,
                  want_vals=False)

        if unit:
            gm = gacc / steps
            gm = gm / gm.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            out[l] = (gm * dev[None]).sum(-1).float().cpu()
        else:
            out[l] = (sacc / steps).float()
    return out


class ObjectiveModel:
    r"""A model-shaped adapter whose ``unembed`` returns :math:`F`.

    ``jvlm.vmap`` is owned elsewhere and its default behaviour must not change,
    but ``vmap.ig_map`` only ever touches a model through four things --
    ``layers``, ``n_layers``, ``forward`` and ``unembed`` -- and it reads the
    scalar it integrates as ``unembed(h)[:, t]``.  So an objective can be
    handed to it unmodified by wrapping the model in an object that runs the
    objective's variants inside ``forward`` and returns the ``[B, k]`` matrix
    of objective values where the ``[B, V]`` logits used to be.  ``token_ids``
    becomes ``range(k)``: reading "token" ``j`` reads objective ``j``.

    Nothing is faked and no arithmetic changes.  The write-in hook
    ``vmap.ig_map`` registers sits on ``model.layers[l]``, which the adapter
    shares, so it fires on every variant's forward; the autograd graph is the
    real one; and the gradient of the summed column is the gradient of
    :math:`F` because the variants enter it linearly.  :func:`selftest` asserts
    that ``vmap.ig_map`` through this adapter agrees with :func:`ig_map` to
    floating point for all three objectives, and ``vmap.completeness`` works
    through it for the same reason.

    Use :func:`vmap_target`, which returns the three arguments together.
    """

    def __init__(self, model, obj: Objective):
        self._m = model
        self._obj = obj
        self._out: torch.Tensor | None = None

    def __getattr__(self, name):
        try:
            m = object.__getattribute__(self, "_m")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(m, name)

    def forward(self, batch: dict):
        n = int(batch["input_ids"].shape[0])
        last = self._m.n_layers - 1
        total = None
        for v in self._obj.variants:
            big = self._m.replicate(v.batch, n) if n > 1 else v.batch
            with ActivationRecorder(self._m.layers, at=[last]) as rec:
                self._m.forward(big)
                resid = rec.activations[last]
            o = v.reader(self._m, resid).float()
            c = o @ v.weight.to(o.device, o.dtype)
            total = c if total is None else total + c
        self._out = total
        return None

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """The objective values from the pass :meth:`forward` just ran.

        The argument is ignored: ``vmap.ig_map`` has already sliced the
        residual down to one position, and a sequence-level objective needs the
        whole span.  The tensor returned is on the same graph as that slice, so
        the gradient is unaffected by the detour.
        """
        if self._out is None:
            raise RuntimeError("ObjectiveModel.forward has not been run")
        return self._out

    def __repr__(self) -> str:
        return f"ObjectiveModel({self._m.name}, {self._obj})"


def vmap_target(model, obj: Objective) -> tuple:
    """``(model, batch, token_ids)`` for a call straight into ``vmap.ig_map``.

    .. code-block:: python

       mm, bb, ids = objective.vmap_target(m, objective.margin(m, b, pairs))
       scores = vmap.ig_map(mm, bb, ids, LAYERS, IPOS, means=MEANS, steps=16)
    """
    return ObjectiveModel(model, obj), obj.variants[0].batch, list(range(obj.k))


@torch.no_grad()
def value(model, obj: Objective, *, replace: torch.Tensor | None = None,
          layer: int | None = None, positions=None) -> torch.Tensor:
    """``[k]`` -- :math:`F` itself, optionally on the baseline picture.

    ``replace`` writes ``[P, d]`` into ``positions`` at ``layer``, which is how
    :math:`F(\\bar h, q)` is obtained: the control condition of the
    difference-in-differences is the mean image, not a blank one.
    """
    pos = (None if positions is None
           else torch.as_tensor(positions, dtype=torch.long))
    last = model.n_layers - 1
    total = None
    for v in obj.variants:
        hd = (model.layers[layer].register_forward_hook(_write_hook(replace, pos))
              if replace is not None else None)
        try:
            with ActivationRecorder(model.layers, at=[last]) as rec:
                model.forward(v.batch)
                out = v.reader(model, rec.activations[last]).float()
                c = (out @ v.weight.to(out.device, out.dtype)).cpu()
        finally:
            if hd is not None:
                hd.remove()
        total = c if total is None else total + c
    return total[0]


def completeness(model, obj: Objective, layer: int, positions, mean,
                 scores: torch.Tensor) -> dict:
    r"""How close the readings come to the effect they claim to decompose.

    ``truth`` is :math:`F(h, q) - F(\bar h, q)`, measured by running the model
    twice; ``claim`` is the readings summed over positions.  For an exact path
    integral they are equal, so the gap is the discretisation error and nothing
    else -- which is the point of reporting it per objective: a log-likelihood
    is more curved than a logit and pays more for the same number of steps.
    """
    F = value(model, obj)
    F0 = value(model, obj, replace=mean, layer=layer, positions=positions)
    truth = F - F0
    claim = scores.sum(-1).float().cpu()
    denom = truth.abs().mean().clamp_min(1e-6)
    corr = float("nan")
    if truth.numel() > 2:
        corr = float(torch.corrcoef(torch.stack([claim, truth]))[0, 1])
    return {"truth": truth.tolist(), "claim": claim.tolist(),
            "F": F.tolist(), "F0": F0.tolist(),
            "rel_error": float((claim - truth).abs().mean() / denom),
            "corr": corr}


def rows_independent(model, obj: Objective, layer: int, positions, mean, *,
                     rows: int = 3, col: int = 0) -> dict:
    r"""Does ``model.replicate`` batching leak between rows?

    The whole path integral rests on one assumption about the batching: that
    running ``steps`` different interpolants as ``steps`` rows of one batch
    gives each row the value and the gradient it would have had alone.  If any
    row saw any other -- a mis-shaped mask, a vision tensor packed for one row
    and read by all of them, a normalisation over the batch -- the readings
    would be a smear of the path rather than the path, and completeness would
    quietly stop meaning anything.

    So it is measured, not assumed: ``rows`` interpolants are run batched and
    then one at a time, and the largest disagreement in the objective and in
    its gradient is returned beside the scale of each, because "small" is only
    meaningful next to "compared with what".
    """
    pos = torch.as_tensor(positions, dtype=torch.long)
    with torch.no_grad(), ActivationRecorder(model.layers, at=[layer]) as rec:
        model.forward(obj.variants[0].batch)
        a = rec.activations[layer]
        h = a[0, pos.to(a.device)].detach().float()
    mu = mean.to(h.device).float()
    dev = h - mu
    al = torch.linspace(0.15, 0.95, rows).to(h.device)
    X = (mu[None] + al[:, None, None] * dev[None]).detach()

    xb = X.clone().requires_grad_(True)
    gb = torch.zeros_like(xb)
    vb = _pass(model, obj, layer, pos, xb, cols=[col],
               sink=lambda j, g: gb.add_(g))

    vl, gl = [], []
    for r in range(rows):
        xr = X[r: r + 1].clone().requires_grad_(True)
        gr = torch.zeros_like(xr)
        vl.append(_pass(model, obj, layer, pos, xr, cols=[col],
                        sink=lambda j, g: gr.add_(g)))
        gl.append(gr.detach())
    vl = torch.cat(vl, 0)
    gl = torch.cat(gl, 0)
    return {"rows": rows,
            "value_maxdiff": float((vb - vl).abs().max()),
            "value_scale": float(vl.abs().max()),
            "grad_maxdiff": float((gb - gl).abs().max()),
            "grad_scale": float(gl.abs().max())}


@torch.no_grad()
def variant_drift(model, obj: Objective, layers, positions) -> dict:
    """Do the objective's forward passes agree at the image tokens?

    They must: the variants differ only in the answer teacher-forced after the
    prompt, image tokens precede all of it, and the mask is causal.  That is
    the same argument that makes the two prior lenses prompt-blind, so it is
    worth checking rather than asserting -- and if it ever failed, the shared
    ``dev`` that :func:`ig_map` computes from the first variant would be wrong
    for the others.
    """
    layers = sorted(set(layers))
    pos = torch.as_tensor(positions, dtype=torch.long)
    ref, worst = None, 0.0
    scale = 0.0
    for v in obj.variants:
        with ActivationRecorder(model.layers, at=layers) as rec:
            model.forward(v.batch)
            cur = {l: rec.activations[l][0, pos.to(rec.activations[l].device)]
                   .detach().float().cpu() for l in layers}
        if ref is None:
            ref = cur
            scale = max(float(t.abs().max()) for t in cur.values())
        else:
            worst = max(worst, max(float((cur[l] - ref[l]).abs().max())
                                   for l in layers))
    return {"max_drift": worst, "scale": scale,
            "n_variants": len(obj.variants)}


# ------------------------------------------------------------- the toy model
#
# The unit tests run on CPU against a four-layer transformer with random
# weights and a fake "vision tower", wrapped in the *real* VLMLensModel.  That
# is deliberate: the things most likely to be wrong here are the batching
# (does ``replicate`` really give each row its own image?), the teacher-forced
# span (is the shift off by one?) and the hook (does the write land where the
# gradient is taken?), and every one of those is exercised by the real wrapper
# on a model small enough to check completeness with 256 path steps in a
# second.  No GPU, no download, no dataset.


class _ToyBlock(torch.nn.Module):
    """Pre-norm block with causal attention, returning an HF-shaped tuple."""

    def __init__(self, d: int, heads: int = 2):
        super().__init__()
        self.heads = heads
        self.qkv = torch.nn.Linear(d, 3 * d)
        self.proj = torch.nn.Linear(d, d)
        self.n1, self.n2 = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d, 2 * d), torch.nn.GELU(), torch.nn.Linear(2 * d, d))

    def forward(self, x, **kw):
        B, S, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)

        def sh(t):
            return t.view(B, S, self.heads, d // self.heads).transpose(1, 2)

        att = torch.nn.functional.scaled_dot_product_attention(
            sh(q), sh(k), sh(v), is_causal=True)
        x = x + self.proj(att.transpose(1, 2).reshape(B, S, d))
        return (x + self.mlp(self.n2(x)), None)


class _ToyText(torch.nn.Module):
    def __init__(self, d, n_layers, vocab):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, d)
        self.layers = torch.nn.ModuleList([_ToyBlock(d) for _ in range(n_layers)])
        self.norm = torch.nn.LayerNorm(d)

    def forward(self, x, **kw):
        for blk in self.layers:
            x = blk(x)[0]
        return x                      # pre-final-norm, as HF blocks give it


class _ToyMM(torch.nn.Module):
    def __init__(self, d, n_layers, vocab, pdim, image_token_id):
        super().__init__()
        self.language_model = _ToyText(d, n_layers, vocab)
        self.visual = torch.nn.Linear(pdim, d)
        self.image_token_id = image_token_id

    def forward(self, input_ids=None, pixel_values=None, **kw):
        x = self.language_model.embed_tokens(input_ids)
        if pixel_values is not None:
            mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(x)
            x = x.masked_scatter(mask, self.visual(pixel_values).to(x.dtype))
        return self.language_model(x)


class _ToyCfg:
    def __init__(self, d, n_layers, vocab, img_id):
        self.architectures = ["ToyForConditionalGeneration"]
        self.image_token_id = img_id
        self.vision_config = None
        self._t = type("T", (), dict(
            num_hidden_layers=n_layers, hidden_size=d, vocab_size=vocab,
            final_logit_softcapping=None))()

    def get_text_config(self):
        return self._t


class _ToyHF(torch.nn.Module):
    def __init__(self, d, n_layers, vocab, pdim, img_id):
        super().__init__()
        self.model = _ToyMM(d, n_layers, vocab, pdim, img_id)
        self.lm_head = torch.nn.Linear(d, vocab, bias=False)
        self.config = _ToyCfg(d, n_layers, vocab, img_id)


class _ToyTok:
    pad_token_id, eos_token_id, unk_token_id = 0, 1, 2

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": torch.tensor(
            [[(ord(c) % 40) + 12 for c in text]], dtype=torch.long)}

    def encode(self, text, add_special_tokens=False):
        return self(text)["input_ids"][0].tolist()

    def convert_tokens_to_ids(self, s):
        return -1


class _ToyProc:
    def __init__(self):
        self.tokenizer = _ToyTok()


_TOY = dict(d=32, n_layers=4, vocab=64, pdim=12, img_id=5)


def toy_model(seed: int = 0):
    """A real :class:`~jvlm.model.VLMLensModel` around a random tiny VLM."""
    from jvlm.model import VLMLensModel
    torch.manual_seed(seed)
    hf = _ToyHF(**{k: _TOY[k] for k in
                   ("d", "n_layers", "vocab", "pdim", "img_id")})
    return VLMLensModel(hf.float(), _ToyProc(), name="toy")


def toy_batch(model, *, n_img: int = 6, n_txt: int = 7, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    txt = torch.randint(10, _TOY["vocab"], (1, n_txt), generator=g)
    img = torch.full((1, n_img), _TOY["img_id"], dtype=torch.long)
    ids = torch.cat([torch.tensor([[9]]), img, txt], 1)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.randn(n_img, _TOY["pdim"], generator=g)}


# ---------------------------------------------------------------- unit tests


def _baseline(model, batch_seeds, layers, pos):
    """A per-position corpus mean over a handful of toy pictures."""
    acc = {l: torch.zeros(len(pos), model.d_model, dtype=torch.float64)
           for l in layers}
    for s in batch_seeds:
        bb = toy_batch(model, seed=s)
        with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as rec:
            model.forward(bb)
            for l in layers:
                acc[l] += rec.activations[l][0, pos].double()
    return {l: (acc[l] / len(batch_seeds)).float() for l in layers}


def selftest(verbose: bool = True) -> dict:
    """CPU unit tests.  Returns ``{name: (ok, detail)}`` and prints them."""
    from jvlm import vmap

    m = toy_model(0)
    b = toy_batch(m, seed=1)
    pos = m.image_mask(b["input_ids"])[0].nonzero(as_tuple=True)[0]
    L2 = [1, 2]
    means = _baseline(m, range(100, 108), L2, pos)
    last = m.n_layers - 1

    ids = [11, 21, 31]
    cor = torch.tensor([13, 27, 41])
    foil = torch.tensor([13, 27, 42, 19])
    objs = {
        "token_logit": token_logit(b, ids),
        "seq_tokens": sequence_logprob(m, b, [cor], norm="tokens"),
        "seq_none": sequence_logprob(m, b, [cor], norm="none"),
        "margin_tokens": margin(m, b, [(cor, foil)], norm="tokens"),
        "margin_none": margin(m, b, [(cor, foil)], norm="none"),
        "margin_shared": margin(m, b, [(cor, foil)], norm="shared"),
    }
    R: dict[str, tuple] = {}

    def rec(name, ok, detail):
        R[name] = (bool(ok), detail)
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:28s} {detail}",
                  flush=True)

    # -- (1) backward compatibility: this is vmap.ig_map when F is a logit ---
    for u in (False, True):
        A = ig_map(m, objs["token_logit"], L2, pos, means, steps=4, unit=u)
        B = vmap.ig_map(m, b, ids, L2, pos.tolist(), means=means, steps=4,
                        unit=u)
        d = max(float((A[l] - B[l]).abs().max()) for l in L2)
        s = max(float(B[l].abs().max()) for l in L2)
        rec(f"backcompat_vmap(unit={u})", d <= 1e-5 * max(s, 1e-6),
            f"max|diff|={d:.3e}  scale={s:.3e}")

    # -- (2) the teacher-forced span, computed a second way ------------------
    bb, start = append_answer(m, b, cor)
    with torch.no_grad(), ActivationRecorder(m.layers, at=[last]) as r2:
        m.forward(bb)
        resid = r2.activations[last]
        lg = m.unembed(resid).float()
        got = float(SeqLogProbReader(start, cor)(m, resid)[0, 0])
    lp = torch.log_softmax(lg[:, :-1], -1)
    tw = lp.gather(-1, bb["input_ids"][:, 1:, None]).squeeze(-1)[0]
    want = float(tw[start - 1: start - 1 + cor.numel()].sum())
    rec("teacher_forced_shift", abs(got - want) < 1e-4,
        f"reader={got:.6f}  independent={want:.6f}")

    # -- (3) gradient reaches h for every objective ---------------------------
    for name, o in objs.items():
        h = _baseline(m, [1], [2], pos)[2]
        x = (means[2] + 0.5 * (h - means[2])).clone().requires_grad_(True)[None]
        g = torch.zeros_like(x)
        _pass(m, o, 2, pos, x, cols=[0], sink=lambda j, gg: g.add_(gg))
        n = float(g.norm())
        rec(f"grad_to_h[{name}]", n > 1e-8 and torch.isfinite(g).all(),
            f"|dF/dh|={n:.4e}")

    # -- (4) batched == looped, value and gradient ---------------------------
    for name, o in objs.items():
        d = rows_independent(m, o, 2, pos, means[2], rows=3)
        ok = (d["value_maxdiff"] <= 1e-4 * max(d["value_scale"], 1e-6)
              and d["grad_maxdiff"] <= 1e-4 * max(d["grad_scale"], 1e-6))
        rec(f"rows_independent[{name}]", ok,
            f"dV={d['value_maxdiff']:.3e}/{d['value_scale']:.3e}  "
            f"dG={d['grad_maxdiff']:.3e}/{d['grad_scale']:.3e}")

    # -- (5) the conservation law, per objective -----------------------------
    comp = {}
    for name, o in objs.items():
        sc = ig_map(m, o, [2], pos, means, steps=256)[2]
        c = completeness(m, o, 2, pos, means[2], sc)
        comp[name] = c
        rec(f"completeness[{name}]", c["rel_error"] < 5e-3,
            f"rel_err={c['rel_error']:.3e}  truth={c['truth'][0]:+.5f} "
            f"claim={c['claim'][0]:+.5f}")

    # -- (6) the margin really is the difference of the two log-likelihoods --
    a = ig_map(m, objs["margin_none"], [2], pos, means, steps=32)[2]
    p = ig_map(m, objs["seq_none"], [2], pos, means, steps=32)[2]
    q = ig_map(m, sequence_logprob(m, b, [foil], norm="none"), [2], pos,
               means, steps=32)[2]
    d = float((a - (p - q)).abs().max())
    rec("margin_is_difference", d <= 1e-5 * max(float(a.abs().max()), 1e-6),
        f"max|IG(margin) - (IG(c) - IG(f))|={d:.3e}")

    # -- (7) the normalisation is the constant we claim it is ----------------
    lpc = float(value(m, sequence_logprob(m, b, [cor], norm="none"))[0])
    lpf = float(value(m, sequence_logprob(m, b, [foil], norm="none"))[0])
    Lc, Lf = int(cor.numel()), int(foil.numel())
    exp = {"margin_none": lpc - lpf,
           "margin_tokens": lpc / Lc - lpf / Lf,
           "margin_shared": (lpc - lpf) / ((Lc + Lf) / 2)}
    bad = {k: (float(value(m, objs[k])[0]), v) for k, v in exp.items()
           if abs(float(value(m, objs[k])[0]) - v) > 1e-4}
    rec("normalisation_constants", not bad,
        f"none={exp['margin_none']:+.4f} tokens={exp['margin_tokens']:+.4f} "
        f"shared={exp['margin_shared']:+.4f}" + (f"  BAD={bad}" if bad else ""))

    # -- (8) chunking the path changes nothing but the memory ----------------
    x1 = ig_map(m, objs["margin_tokens"], [2], pos, means, steps=8, chunk=8)[2]
    x2 = ig_map(m, objs["margin_tokens"], [2], pos, means, steps=8, chunk=3)[2]
    d = float((x1 - x2).abs().max())
    rec("chunk_invariance", d <= 1e-5 * max(float(x1.abs().max()), 1e-6),
        f"max|diff|={d:.3e}")

    # -- (9) shared continuations are run once -------------------------------
    other = torch.tensor([13, 27, 44])
    o2 = margin(m, b, [(cor, foil), (cor, other)], norm="tokens")
    rec("variant_dedup", len(o2.variants) == 3,
        f"{len(o2.variants)} passes for 2 pairs sharing a correct answer")

    # -- (10) a contrast of token logits needs no extra pass -----------------
    plain = ig_map(m, token_logit(b, ids[:2]), [2], pos, means, steps=4)[2]
    contr = ig_map(m, token_logit(b, ids[:2], weight=[[1.], [-1.]]),
                   [2], pos, means, steps=4)[2]
    d = float((contr[0] - (plain[0] - plain[1])).abs().max())
    rec("token_contrast", d <= 1e-5 * max(float(plain.abs().max()), 1e-6),
        f"max|diff|={d:.3e}")

    # -- (11b) the same reading through unmodified vmap.ig_map ---------------
    for name in ("token_logit", "seq_tokens", "margin_tokens"):
        o = objs[name]
        mm, bb, tid = vmap_target(m, o)
        A = vmap.ig_map(mm, bb, tid, [2], pos.tolist(), means=means, steps=8)[2]
        B = ig_map(m, o, [2], pos, means, steps=8)[2]
        d = float((A - B).abs().max())
        rec(f"vmap_adapter[{name}]",
            d <= 1e-5 * max(float(B.abs().max()), 1e-6),
            f"max|vmap.ig_map - objective.ig_map|={d:.3e}")
        cA = vmap.completeness(mm, bb, tid, 2, pos.tolist(), means[2], A)
        cB = completeness(m, o, 2, pos, means[2], B)
        d2 = max(abs(x - y) for x, y in zip(cA["truth"], cB["truth"]))
        rec(f"vmap_adapter_completeness[{name}]", d2 < 1e-4,
            f"truth via vmap={cA['truth'][0]:+.6f} native={cB['truth'][0]:+.6f}")

    # -- (10b) unit=True across several forward passes -----------------------
    # The normalisation has to happen after the variants are summed: normalising
    # each pass's gradient and then adding is a different (and wrong) quantity.
    o = objs["margin_tokens"]
    st = 16
    al = (torch.arange(st, dtype=torch.float32) + 0.5) / st
    hh = _baseline(m, [1], [2], pos)[2]
    dv = hh - means[2]
    xx = (means[2][None] + al[:, None, None] * dv[None]
          ).clone().requires_grad_(True)
    gg = torch.zeros_like(xx)
    _pass(m, o, 2, pos, xx, cols=[0], sink=lambda j, g: gg.add_(g),
          want_vals=False)
    gmean = gg.mean(0)
    want = ((gmean / gmean.norm(dim=-1, keepdim=True).clamp_min(1e-9)) * dv
            ).sum(-1)
    got = ig_map(m, o, [2], pos, means, steps=st, unit=True)[2][0]
    cs = bool((got.abs() <= dv.norm(dim=-1) + 1e-4).all())
    d = float((got - want).abs().max())
    rec("unit_multivariant", d < 1e-5 and cs,
        f"max|diff|={d:.3e}  Cauchy-Schwarz={cs}")

    # -- (11c) and through the chunked estimator built on top of vmap --------
    from jvlm import budget
    for name in ("seq_tokens", "margin_tokens"):
        o = objs[name]
        mm, bb, tid = vmap_target(m, o)
        A = budget.ig_map_chunked(mm, bb, tid, 2, pos.tolist(), means[2],
                                  steps=8, chunk=2)
        B = ig_map(m, o, [2], pos, means, steps=8)[2]
        d = float((A - B).abs().max())
        rec(f"budget_adapter[{name}]",
            d <= 1e-5 * max(float(B.abs().max()), 1e-6),
            f"max|ig_map_chunked - objective.ig_map|={d:.3e}")

    # -- (11) the variants agree at the image tokens -------------------------
    dr = variant_drift(m, objs["margin_tokens"], L2, pos)
    rec("variant_drift", dr["max_drift"] <= 1e-5 * max(dr["scale"], 1e-6),
        f"max_drift={dr['max_drift']:.3e}  scale={dr['scale']:.3e}")

    # -- (12) the interaction object, and that it obeys the same law ---------
    # I_R(q_a, q_b) = A_R(q_a) - A_R(q_b) sums over regions to the
    # difference-in-differences with the mean image as the control.  That is a
    # consequence of linearity plus completeness, but it is the paper's spine,
    # so it is measured here under an answer-level F rather than assumed.  The
    # size of I next to the size of A is also the prompt-sensitivity a
    # prompt-blind lens would report as exactly zero.
    ba = b
    bb = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    gg = torch.Generator().manual_seed(999)
    n_txt = int(bb["input_ids"].shape[1]) - int(pos.numel()) - 1
    bb["input_ids"][0, -n_txt:] = torch.randint(
        10, _TOY["vocab"], (n_txt,), generator=gg)
    for nm_, mk in (("margin_tokens", lambda x: margin(m, x, [(cor, foil)],
                                                       norm="tokens")),
                    ("seq_tokens", lambda x: sequence_logprob(m, x, [cor],
                                                              norm="tokens"))):
        oa, ob = mk(ba), mk(bb)
        Aa = ig_map(m, oa, [2], pos, means, steps=128)[2]
        Ab = ig_map(m, ob, [2], pos, means, steps=128)[2]
        I = Aa - Ab
        did = ((value(m, oa) - value(m, oa, replace=means[2], layer=2,
                                     positions=pos))
               - (value(m, ob) - value(m, ob, replace=means[2], layer=2,
                                       positions=pos)))
        gap = float((I.sum(-1) - did).abs().max())
        scale = float(did.abs().max())
        rec(f"interaction_did[{nm_}]", gap <= 5e-3 * max(scale, 1e-6),
            f"sum_R I_R={float(I.sum()):+.6f}  DiD={float(did[0]):+.6f}  "
            f"|I|/|A|="
            f"{float(I.abs().sum() / Aa.abs().sum().clamp_min(1e-9)):.2f}")

    # -- (13) how many path steps each objective needs -----------------------
    # Not a pass/fail: the law is exact, the midpoint rule is not, and a
    # log-softmax is more curved than a logit, so this is the price list.
    if verbose:
        print("\n  discretisation (rel_error vs path steps):")
    sweep = {}
    for name, o in objs.items():
        row = {}
        for st in (1, 2, 4, 8, 16, 32):
            sc = ig_map(m, o, [2], pos, means, steps=st)[2]
            row[st] = completeness(m, o, 2, pos, means[2], sc)["rel_error"]
        sweep[name] = row
        if verbose:
            print("    " + f"{name:16s}" + "  ".join(
                f"{st}:{v:.1e}" for st, v in row.items()), flush=True)
    R["_sweep"] = (True, sweep)

    # -- (14) the gap is quadrature and nothing else -------------------------
    # If the conservation law itself broke under an answer-level F, the gap
    # would stop shrinking as the path is refined.  So the assertion is the one
    # that carries the claim: the gap falls monotonically with the path length
    # and is at least 8x smaller at 16 steps than at 2.
    #
    # The fitted order is reported beside it but not asserted.  A ratio between
    # two step counts is a noisy estimate of the leading term -- across random
    # toy weights it scatters between about 3 and 6 -- and past 32 steps the
    # absolute gap here is around 1e-6 against an F of order 4, which is
    # float32 round-off, so on some weights the 128-step gap comes out larger
    # than the 32-step one.  Asserting a rate in that regime would be asserting
    # arithmetic noise; the fit over 2..16 stays above the floor.
    import math
    for name, row in sweep.items():
        ks = [2, 4, 8, 16]
        xs = [math.log(k) for k in ks]
        ys = [math.log(max(row[k], 1e-30)) for k in ks]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        slope = (sum((a - mx) * (b - my) for a, b in zip(xs, ys))
                 / sum((a - mx) ** 2 for a in xs))
        mono = all(row[ks[i]] > row[ks[i + 1]] for i in range(len(ks) - 1))
        drop = row[2] / max(row[16], 1e-30)
        rec(f"converges[{name}]", mono and drop >= 8.0,
            f"monotone={mono}  gap(2)/gap(16)={drop:.1f}  "
            f"fitted order={-slope:.2f} (midpoint rule gives 2)")

    tests = {k: v for k, v in R.items() if not k.startswith("_")}
    n_ok = sum(v[0] for v in tests.values())
    if verbose:
        print(f"\n{n_ok}/{len(tests)} passed", flush=True)
    return R


# ------------------------------------------------------- the check on a VLM


def gpu_check(repo: str = "Qwen/Qwen3-VL-2B-Instruct", *, n: int = 24,
              mean_images: int = 60, n_layers: int = 2, steps: int = 8,
              chunk: int = 2, square: int = 448, max_pixels: int = 200704,
              sweep_items: int = 3, sweep_steps=(1, 2, 4, 8, 16),
              gen_tokens: int = 24,
              device: str = "cuda", dtype: str = "bfloat16",
              data: str = "/workspace/vlm/data/winoground",
              out: str = "/workspace/vlm/results/objectives",
              tag: str | None = None) -> str:
    r"""Does the conservation law survive the change of :math:`F`?

    It must, because the law is the fundamental theorem of calculus and does
    not know what :math:`F` is; what can fail is the *discretisation*, and a
    log-softmax over a 150k vocabulary is a good deal more curved than one
    entry of it.  So the number that matters here is not whether completeness
    "holds" -- it is how many path steps each objective needs before the sum of
    the readings and the measured effect agree, and whether the agreement is
    the exact one the law predicts or merely a correlation.

    Winoground supplies the pair: image_0 with caption_0 as the correct answer
    and caption_1 as the foil, the two built from the same words in a different
    order, so the margin's two branches are length-matched by construction and
    ``"tokens"`` and ``"shared"`` normalisation differ only by a constant.  The
    baseline :math:`\bar h` is a per-position mean over held-out COCO
    photographs, which Winoground does not draw from.

    Layers are fixed in advance at 0.35 and 0.65 of the usable depth.  Nothing
    here is chosen by looking at the answer.
    """
    import json
    import os
    import time

    import numpy as np
    import pandas as pd

    import jvlm
    from jvlm import corpus, readout

    os.makedirs(out, exist_ok=True)
    tag = tag or repo.split("/")[-1]
    t0 = time.time()

    recs = [json.loads(l) for l in
            open(os.path.join(data, "data", "examples.jsonl"))]
    for r in recs:
        r["image_0_path"] = os.path.join(data, "images", r["image_0"] + ".png")
    recs = [r for r in recs if os.path.exists(r["image_0_path"])][:n]
    print(f"[objective] {len(recs)} winoground items", flush=True)

    m = jvlm.load_vlm(repo, max_pixels=max_pixels, device=device,
                      dtype=getattr(torch, dtype))
    lo, hi = max(m.n_deepstack, 1), m.n_layers - 2
    LAYERS = sorted({int(round(lo + (hi - lo) * f))
                     for f in np.linspace(0.35, 0.65, n_layers)})
    yes = m.tokenizer.encode("yes", add_special_tokens=False)[0]
    print(m, f"\n[objective] layers {LAYERS}  yes={yes}  steps={steps} "
             f"chunk={chunk}", flush=True)

    def load(p):
        return corpus.load_image(p, square=square)

    # -------------------------------------------------------- the baseline
    pool = corpus.coco_images(mean_images + 30, seed=7)
    b0 = m.chat("Describe this image.", [load(pool[0])])
    IPOS = readout.image_positions(m, b0)
    P = len(IPOS)
    acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
    nm = 0
    for p in pool:
        if nm >= mean_images:
            break
        bb = m.chat("Describe this image.", [load(p)])
        ip = readout.image_positions(m, bb)
        if len(ip) != P:
            continue
        A = readout.residuals(m, bb, LAYERS)
        for l in LAYERS:
            acc[l] += A[l][ip].float().cpu().double()
        nm += 1
    MEANS = {l: (acc[l] / max(nm, 1)).float() for l in LAYERS}
    print(f"[objective] baseline from {nm} images, P={P}, "
          f"{time.time()-t0:.0f}s", flush=True)

    CAPTION = "Describe this image."
    QUESTION = "Does the caption '{c}' describe this image?  Answer yes or no."

    rows: list[dict] = []
    diag: list[dict] = []
    sweep: list[dict] = []
    pq = os.path.join(out, f"completeness__{tag}.parquet")
    dq = os.path.join(out, f"diagnostics__{tag}.parquet")
    sq = os.path.join(out, f"steps__{tag}.parquet")

    for i, r in enumerate(recs):
        img = load(r["image_0_path"])
        c, f = r["caption_0"], r["caption_1"]
        bcap = m.chat(CAPTION, [img])
        bq = m.chat(QUESTION.format(c=c), [img])
        objs = {
            "token_logit": token_logit(bq, [yes]),
            "seq_logprob_tokens": sequence_logprob(m, bcap, [c], norm="tokens"),
            "margin_tokens": margin(m, bcap, [(c, f)], norm="tokens"),
            "margin_shared": margin(m, bcap, [(c, f)], norm="shared"),
        }
        # (ii) in its literal form: the answer the model actually produces,
        # generated greedily once and then teacher-forced.  Guarded, because a
        # generate() that a family's processor does not like should cost this
        # objective and not the whole run.
        try:
            gen = generated_answer(m, bcap, max_new_tokens=gen_tokens)
            objs["seq_logprob_generated"] = sequence_logprob(
                m, bcap, [gen], norm="tokens", names=["generated"])
            if i == 0:
                print("[objective] generated: "
                      f"{m.tokenizer.decode(gen)!r}", flush=True)
        except Exception as e:                                   # noqa: BLE001
            if i == 0:
                print(f"[objective] generated answer unavailable "
                      f"({type(e).__name__}: {str(e)[:80]}); skipping it",
                      flush=True)
        # each objective carries its own prompt, so each gets its own image
        # positions; the baseline is indexed by ordinal, not by absolute index
        IP = {k: o.image_positions(m) for k, o in objs.items()}
        if any(len(v) != P for v in IP.values()):
            continue
        if i == 0:
            for name, o in objs.items():
                d = rows_independent(m, o, LAYERS[0], IP[name],
                                     MEANS[LAYERS[0]], rows=3)
                d |= variant_drift(m, o, LAYERS, IP[name])
                d |= {"objective": name, "layer": LAYERS[0]}
                diag.append(d)
                print(f"[objective] {name}: batched-vs-looped "
                      f"dV={d['value_maxdiff']:.2e}/{d['value_scale']:.2e} "
                      f"dG={d['grad_maxdiff']:.2e}/{d['grad_scale']:.2e} "
                      f"variant_drift={d['max_drift']:.2e}/{d['scale']:.2e}",
                      flush=True)
            pd.DataFrame(diag).to_parquet(dq, index=False)

        if i < sweep_items:
            for name, o in objs.items():
                for st in sweep_steps:
                    sc = ig_map(m, o, [LAYERS[0]], IP[name], MEANS, steps=st,
                                chunk=min(chunk, st))[LAYERS[0]]
                    cp = completeness(m, o, LAYERS[0], IP[name],
                                      MEANS[LAYERS[0]], sc)
                    sweep.append({"item": i, "objective": name,
                                  "layer": LAYERS[0], "steps": st,
                                  "truth": cp["truth"][0],
                                  "claim": cp["claim"][0],
                                  "rel_error": cp["rel_error"]})
            pd.DataFrame(sweep).to_parquet(sq, index=False)

        for name, o in objs.items():
            for l in LAYERS:
                sc = ig_map(m, o, [l], IP[name], MEANS, steps=steps,
                            chunk=chunk)[l]
                comp = completeness(m, o, l, IP[name], MEANS[l], sc)
                rows.append({
                    "item": i, "objective": name, "layer": l, "steps": steps,
                    "truth": comp["truth"][0], "claim": comp["claim"][0],
                    # F on the real picture is the behavioural score itself --
                    # for the margin, the model's forced choice, whose sign
                    # says whether it got the item right
                    "F": comp["F"][0], "F0": comp["F0"][0],
                    "rel_error": comp["rel_error"],
                    # the relative error is inflated when F(h) is close to
                    # F(hbar), i.e. when there is barely an effect to
                    # decompose, so the absolute gap travels with it
                    "abs_error": abs(comp["claim"][0] - comp["truth"][0]),
                    "n_passes": len(o.variants),
                    "pos_mass": float(sc.abs().sum()),
                })
        pd.DataFrame(rows).to_parquet(pq, index=False)
        if i % 4 == 0:
            print(f"[objective] item {i+1}/{len(recs)}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("[objective] no items survived; nothing to summarise", flush=True)
        return pq
    summ = []
    rng = np.random.default_rng(0)
    for (name, l), g in df.groupby(["objective", "layer"]):
        t, cl = g["truth"].to_numpy(), g["claim"].to_numpy()
        corr = float(np.corrcoef(cl, t)[0, 1]) if len(g) > 2 else float("nan")
        null = [float(np.corrcoef(rng.permutation(cl), t)[0, 1])
                for _ in range(200)] if len(g) > 2 else [float("nan")]
        summ.append({
            "objective": name, "layer": int(l), "n": len(g),
            "corr": corr, "null_corr_mean": float(np.mean(null)),
            "null_corr_p95": float(np.percentile(null, 95)),
            "median_rel_error": float(g["rel_error"].median()),
            "max_rel_error": float(g["rel_error"].max()),
            "median_abs_error": float(g["abs_error"].median()),
            "pooled_rel_error": float(g["abs_error"].sum()
                                      / np.abs(t).sum().clip(1e-9)),
            "median_abs_truth": float(np.median(np.abs(t))),
            "frac_F_positive": float((g["F"].to_numpy() > 0).mean()),
        })
    sdf = pd.DataFrame(summ).sort_values(["objective", "layer"])
    sdf.to_parquet(os.path.join(out, f"summary__{tag}.parquet"), index=False)
    print("\n" + sdf.to_string(index=False), flush=True)
    if sweep:
        sw = (pd.DataFrame(sweep).groupby(["objective", "steps"])["rel_error"]
              .median().unstack())
        print("\nmedian rel_error vs path steps (layer "
              f"{LAYERS[0]}, {sweep_items} items):\n"
              + sw.to_string(float_format=lambda v: f"{v:.3e}"), flush=True)
    print(f"[objective] done in {time.time()-t0:.0f}s -> {pq}", flush=True)
    return pq


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, "/workspace/vlm/code")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu", action="store_true", help="run the unit tests")
    ap.add_argument("--gpu", action="store_true", help="run the VLM check")
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--mean-images", type=int, default=60)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--sweep-items", type=int, default=3)
    ap.add_argument("--sweep-max", type=int, default=16)
    ap.add_argument("--device", default="cuda",
                    help="cpu runs the same check without touching the card, "
                         "and without a name the GPU queue watchdog counts")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--max-pixels", type=int, default=200704)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--out", default="/workspace/vlm/results/objectives")
    ap.add_argument("--tag", default=None)
    A = ap.parse_args()
    if not (A.cpu or A.gpu):
        A.cpu = True
    code = 0
    if A.cpu:
        print("== objective.py unit tests (CPU, toy model) ==", flush=True)
        R = selftest()
        code = 0 if all(v[0] for k, v in R.items()
                        if not k.startswith("_")) else 1
    if A.gpu:
        gpu_check(A.repo, n=A.n, mean_images=A.mean_images,
                  n_layers=A.n_layers, steps=A.steps, chunk=A.chunk,
                  sweep_items=A.sweep_items, square=A.square,
                  sweep_steps=[s for s in (1, 2, 4, 8, 16, 32)
                               if s <= A.sweep_max],
                  device=A.device, dtype=A.dtype, max_pixels=A.max_pixels,
                  out=A.out, tag=A.tag)
    raise SystemExit(code)
