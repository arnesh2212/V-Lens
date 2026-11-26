# SPDX-License-Identifier: Apache-2.0
"""The refusal direction of Arditi et al. (2024), done the way the paper does it.

"Refusal in Language Models Is Mediated by a Single Direction" (Arditi,
Obeso, Syed, Paleka, Panickssery, Gurnee and Nanda, NeurIPS 2024) is the
baseline any claim about a refusal direction is measured against, so each
piece here follows a numbered part of that paper:

* the candidates (Sec. 2.3): for every layer and every post-instruction token
  position, the difference between the mean activation over harmful
  instructions and over harmless ones. "Post-instruction" is every template
  token after the instruction (Sec. 2.1); :func:`post_instruction_count`
  finds how many there are in this model's chat template;
* the two interventions (Sec. 2.4): directional ablation, ``x <- x - r r^T x``
  at every residual activation, both the one entering each block and the one
  between its attention and its MLP, at every position, generated ones
  included (:class:`DirectionalAblation`); and activation addition,
  ``x <- x + r`` at the one chosen layer and every position
  (:class:`ActAdd`);
* the refusal metric (App. B): the log-odds of the probability mass on the
  tokens refusals open with (:func:`refusal_metric_fn`);
* the selection (App. C.1): the candidate with the lowest ``bypass_score``
  among those with ``induce_score > 0``, ``kl_score < 0.1`` and ``l < 0.8 L``
  (:func:`select`).

Layer indexing. The paper's activations are the residual stream *entering*
layer ``l``. Here states are read at the *output* of block ``l``, which is the
residual entering block ``l + 1``; the ``0.8 L`` filter is therefore applied
to ``l + 1``, and addition is made at the output of the block the direction
was read at, which is the same point in the stream.
"""
from __future__ import annotations

import contextlib
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

#: App. C.1 thresholds
KL_MAX = 0.1
DEPTH_MAX = 0.8


def unit(v) -> torch.Tensor:
    v = torch.as_tensor(v).float()
    return v / v.norm().clamp_min(1e-8)


# ------------------------------------------------------------------ positions

def post_instruction_count(m, question: str, image, max_tokens: int = 16) -> tuple[int, list[str]]:
    """How many template tokens follow the instruction, and what they are.

    The chat-templated prompt is split at the end of the instruction text and
    the tail tokenised; the count is then checked against the tokens the
    processor actually produces, and only the trailing tokens the two agree on
    are kept, so a merge across the boundary can shorten the span but never
    let it reach into the instruction.
    """
    t = m.chat_prompt(question, [image] if image is not None else None)
    k = t.rfind(question)
    suffix = t[k + len(question):] if k >= 0 else ""
    want = m.tokenizer.encode(suffix, add_special_tokens=False) if suffix else []
    b = m.chat(question, [image] if image is not None else None)
    ids = b["input_ids"][0].tolist()
    n = 0
    while n < min(len(want), len(ids)) and ids[-1 - n] == want[-1 - n]:
        n += 1
    n = max(1, min(n, max_tokens))
    return n, [m.tokenizer.decode([i]) for i in ids[-n:]]


# ------------------------------------------------------------- the metric

def refusal_token_ids(tok) -> list[int]:
    """First tokens of a refusal, the set this work's refusal margin uses.

    App. B defines the metric on a small model-specific set (Table 4: "I" and
    "As" for the Qwen models, "I" for Gemma); this set contains those and the
    other openings the regex judge counts as refusals. Only tokens that spell
    a whole opening word are kept: ``scores.first_ids`` takes the first piece
    of a word, which for a word the vocabulary splits is a fragment such as
    "C", " S", "un" or "ap" that opens compliant replies as often as refusals.
    """
    from vlens.lens import scores
    out = set()
    for w in scores.REFUSE_STARTS:
        for i in scores.first_ids(tok, w):
            t = tok.decode([i]).strip().lower()
            if t == w or (len(t) >= 5 and w.startswith(t)):
                out.add(i)
    return sorted(out)


def refusal_metric_fn(tok) -> Callable:
    """``logits [n, V] -> [n]``: ``log(P_refusal / (1 - P_refusal))`` (App. B)."""
    R = refusal_token_ids(tok)

    def f(lg: torch.Tensor) -> torch.Tensor:
        lp = torch.log_softmax(lg.float(), -1)
        lpr = torch.logsumexp(lp[..., R], -1).clamp(max=-1e-7)
        return lpr - torch.log(-torch.expm1(lpr))
    return f


# ------------------------------------------------------------ interventions

def _attention_writer(layer):
    """The submodule whose output is the attention write into the residual.

    Qwen-style blocks add the attention (or linear-attention) output straight
    to the residual. Gemma-style blocks normalise it first
    (``post_attention_layernorm``, followed by a ``pre_feedforward_layernorm``
    on the other side of the add), so there the write is the norm's output.
    In a Qwen block ``post_attention_layernorm`` is the MLP's *input* norm,
    which is why the two are told apart by the presence of the Gemma norm.
    """
    if hasattr(layer, "pre_feedforward_layernorm") and hasattr(layer, "post_attention_layernorm"):
        return layer.post_attention_layernorm, "post_attention_layernorm"
    for name in ("self_attn", "linear_attn", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name), name
    return None, None


class DirectionalAblation:
    """``x <- x - r r^T x`` at every block, every position, every step (Sec. 2.4).

    Three hook points per block: the residual entering it, the attention
    write inside it (so the MLP reads a residual without the direction, as
    the paper's ``x~`` ablation does), and its output (the MLP write and
    anything the block adds after it). The block-output hook of the last
    block is what the final norm reads. A visual injection that lands
    between two blocks (Qwen3-VL's DeepStack) is removed by the next block's
    input hook. Decode steps pass through the same hooks, so the direction
    never reappears in generated positions.
    """

    def __init__(self, m, r):
        self.m = m
        self.r = unit(r)
        self._h: list = []
        self.points: list[str] = []
        self._cache: dict = {}

    def _rr(self, x):
        k = (x.device,)
        if k not in self._cache:
            self._cache[k] = self.r.to(x.device)
        return self._cache[k]

    def proj(self, x: torch.Tensor) -> torch.Tensor:
        r = self._rr(x)
        xf = x.float()
        return (xf - (xf @ r)[..., None] * r).to(x.dtype)

    def _out(self, mod, inp, out):
        if torch.is_tensor(out):
            return self.proj(out)
        return (self.proj(out[0]), *out[1:])

    def _pre(self, mod, args, kwargs):
        if args and torch.is_tensor(args[0]):
            return (self.proj(args[0]), *args[1:]), kwargs
        if torch.is_tensor(kwargs.get("hidden_states")):
            kwargs = dict(kwargs, hidden_states=self.proj(kwargs["hidden_states"]))
            return args, kwargs
        return None

    def __enter__(self):
        pts = set()
        for layer in self.m.layers:
            self._h.append(layer.register_forward_pre_hook(self._pre, with_kwargs=True))
            w, name = _attention_writer(layer)
            if w is not None:
                self._h.append(w.register_forward_hook(self._out))
                pts.add(f"attention write ({name})")
            self._h.append(layer.register_forward_hook(self._out))
        self.points = ["block input", *sorted(pts), "block output"]
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []


class ActAdd:
    """Activation addition at one block, at every position and every step (Sec. 2.4).

    ``vec`` adds that vector as it is, which is the paper's ``x + r`` with the
    difference of means at its natural size. ``alpha`` instead adds
    ``alpha * ||x|| * unit(vec)`` position by position, the norm-relative dose
    the earlier steering runs used, kept as a dose-response beside it.
    """

    def __init__(self, m, layer: int, vec, alpha: float | None = None):
        self.m, self.layer = m, int(layer)
        self.vec = torch.as_tensor(vec).float()
        self.alpha = alpha
        self._h = None
        self._u: dict = {}

    def _hook(self, mod, inp, out):
        h = out if torch.is_tensor(out) else out[0]
        k = h.device
        if k not in self._u:
            v = self.vec.to(h.device)
            self._u[k] = v if self.alpha is None else v / v.norm().clamp_min(1e-8)
        v = self._u[k]
        if self.alpha is None:
            h2 = (h.float() + v).to(h.dtype)
        else:
            hf = h.float()
            h2 = (hf + self.alpha * hf.norm(dim=-1, keepdim=True) * v).to(h.dtype)
        return h2 if torch.is_tensor(out) else (h2, *out[1:])

    def __enter__(self):
        self._h = self.m.layers[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._h is not None:
            self._h.remove()
        self._h = None


def intervention(m, spec: dict | None):
    """A context manager from a plain description, so a run can be logged and replayed.

    ``{"mode": "ablate", "r": vec}`` or ``{"mode": "add", "layer": l, "vec": vec,
    "alpha": None | float}``; ``None`` is no edit.
    """
    if not spec:
        return contextlib.nullcontext()
    if spec["mode"] == "ablate":
        return DirectionalAblation(m, spec["r"])
    if spec["mode"] == "add":
        return ActAdd(m, spec["layer"], spec["vec"], spec.get("alpha"))
    raise ValueError(spec["mode"])


# ------------------------------------------------------------------ selection

def allowed_block(l: int, n_layers: int) -> bool:
    """App. C.1's ``l < 0.8 L`` on the residual entering block ``l + 1``."""
    return (l + 1) < DEPTH_MAX * n_layers


def select(cands: Sequence[dict], n_layers: int) -> tuple[dict | None, str]:
    """App. C.1: the lowest ``bypass`` among candidates that pass every filter.

    Each candidate carries ``layer``, ``bypass``, ``induce`` and ``kl``. Where
    no candidate passes all three filters the KL filter is dropped first, then
    the induce filter, and the returned note says which, so a run that had to
    fall back cannot be mistaken for one that did not.
    """
    ok_l = [c for c in cands if allowed_block(c["layer"], n_layers)
            and np.isfinite(c["bypass"])]
    tiers = (("all three filters", lambda c: c["induce"] > 0 and c["kl"] < KL_MAX),
             ("without the KL filter", lambda c: c["induce"] > 0),
             ("depth filter only", lambda c: True))
    for note, keep in tiers:
        pool = [c for c in ok_l if keep(c)]
        if pool:
            return min(pool, key=lambda c: c["bypass"]), note
    return None, "no candidate"


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a rate of ``k`` successes in ``n``."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def evenly(xs: Sequence[int], k: int) -> list[int]:
    """At most ``k`` of ``xs`` spread evenly, ends included; all of them when ``k`` is 0."""
    xs = list(xs)
    if not k or k >= len(xs):
        return xs
    idx = np.unique(np.round(np.linspace(0, len(xs) - 1, k)).astype(int))
    return [xs[i] for i in idx]


def stack_hooks(*ctxs: Iterable) -> contextlib.ExitStack:
    """Enter several context managers in order, first one outermost."""
    es = contextlib.ExitStack()
    for c in ctxs:
        es.enter_context(c)
    return es
