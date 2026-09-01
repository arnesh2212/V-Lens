# SPDX-License-Identifier: Apache-2.0
"""Outcome axes: scalar functions of the model's final state at the answer slot.

A Verdict Lens reads how hard a token pushes the first answer token toward an
outcome.  The outcome is one of these axes.  Each is a differentiable scalar of
``(logits, h_final)`` at the answer position, so one backward pass gives its
gradient at every layer and every position.

Axes, in the order the write-up trusts them:

  refusal (behaviour-anchored)   r . h_L[a], with r the unit difference of
                                 means of the final residual between the
                                 model's OWN refusals and compliances on
                                 text-only prompts.  Primary: it is defined by
                                 what the model did, not by which words it used.
  starters (token-anchored)      log-mass on refusal openers minus log-mass on
                                 compliance openers.  Secondary: gameable by
                                 "begin your answer with Sure", and tested as such.
  yesno                          log-mass on yes-forms minus no-forms.
  harm                           log-mass on a fixed lexicon of harmful words
                                 minus benign words.  The same 32 + 32 strings
                                 for every model; whichever are single tokens.
  word(t)                        the logit of one token.

Everything here is label-free with respect to images: the refusal axis uses
text-only prompts and the model's own replies.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import torch

from jvlm import safedata

Margin = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]   # (logits[V], h_final[d]) -> scalar

REFUSE_STARTERS = ["I", "Sorry", "I'm", "cannot", "Unfortunately", "As", "No"]
COMPLY_STARTERS = ["Sure", "Here", "To", "Step", "1", "The", "Yes", "Certainly"]
YES_FORMS = ["Yes", "yes", "YES", "Yeah"]
NO_FORMS = ["No", "no", "NO", "Nope"]
HARM_WORDS = ["kill", "weapon", "bomb", "drug", "drugs", "steal", "hack", "attack",
              "poison", "gun", "knife", "explosive", "murder", "fraud", "illegal",
              "hate", "abuse", "assault", "theft", "violence", "harm", "threat",
              "exploit", "smuggle", "cocaine", "heroin", "meth", "bribe", "scam",
              "forge", "torture", "suicide"]
BENIGN_WORDS = ["cook", "garden", "paint", "walk", "read", "smile", "flower", "book",
                "table", "chair", "coffee", "bread", "music", "travel", "family",
                "friend", "school", "learn", "build", "clean", "water", "tree",
                "dog", "cat", "sunset", "beach", "park", "photo", "picture", "help",
                "study", "play"]


def token_set(tokenizer, words) -> list[int]:
    """The single-token ids among ``words``, in any surface form."""
    ids = [safedata.single_token(tokenizer, w) for w in words]
    return sorted({int(i) for i in ids if i is not None})


@dataclasses.dataclass
class Axis:
    name: str
    fn: Margin
    kind: str                     # "token-set" | "direction" | "word"
    detail: dict

    def __call__(self, logits, h_final):
        return self.fn(logits, h_final)


def _logmass(logits: torch.Tensor, ids: list[int]) -> torch.Tensor:
    lp = torch.log_softmax(logits.float(), -1)
    return torch.logsumexp(lp[ids], 0)


def token_axis(name: str, tokenizer, plus: list[str], minus: list[str]) -> Axis:
    P, M = token_set(tokenizer, plus), token_set(tokenizer, minus)
    if not P or not M:
        raise ValueError(f"{name}: empty token set (plus={len(P)}, minus={len(M)})")
    return Axis(name, lambda lg, h: _logmass(lg, P) - _logmass(lg, M),
                "token-set", dict(plus=P, minus=M))


def starters(tokenizer) -> Axis:
    return token_axis("starters", tokenizer, REFUSE_STARTERS, COMPLY_STARTERS)


def yesno(tokenizer) -> Axis:
    return token_axis("yesno", tokenizer, YES_FORMS, NO_FORMS)


def harm(tokenizer) -> Axis:
    return token_axis("harm", tokenizer, HARM_WORDS, BENIGN_WORDS)


def word(tokenizer, w: str) -> Axis:
    t = safedata.single_token(tokenizer, w)
    if t is None:
        raise ValueError(f"{w!r} is not a single token")
    return Axis(f"word:{w}", lambda lg, h: lg[t].float(), "word", dict(token=t))


def direction_axis(name: str, r: torch.Tensor, detail: dict | None = None) -> Axis:
    """``r . h_final`` for a unit direction ``r`` in the final residual space."""
    r = (r / r.norm().clamp_min(1e-8)).float()
    return Axis(name, lambda lg, h: (h.float() @ r.to(h.device)), "direction",
                dict(detail or {}, norm=1.0, r=r))


@torch.no_grad()
def refusal_direction(m, sc, harmful_texts: list[str], benign_texts: list[str], *,
                      judge_kind, max_new_tokens: int = 24) -> tuple[torch.Tensor, dict]:
    """The behaviour-anchored refusal axis, from the model's own replies.

    Generates a short reply to every text-only prompt, labels each with the
    canonical judge, and returns the unit difference of means of the final
    residual at the answer slot between refusals and compliances, together
    with the counts so a reader can see how many of each anchored it.
    """
    from jvlm import readout
    L = m.n_layers - 1
    hs, labs = [], []
    for txt in list(harmful_texts) + list(benign_texts):
        b = m.chat(txt, [])
        h = readout.residuals(m, b, [L])[L][-1].float().cpu()
        reply = sc.generate(b, max_new_tokens=max_new_tokens)
        hs.append(h); labs.append(judge_kind(reply) == "refusal")
    H = torch.stack(hs); y = torch.tensor(labs)
    if y.sum() < 5 or (~y).sum() < 5:
        raise RuntimeError(f"refusal axis needs both classes: {int(y.sum())} refusals, "
                           f"{int((~y).sum())} compliances")
    r = H[y].mean(0) - H[~y].mean(0)
    return r / r.norm(), dict(n_refused=int(y.sum()), n_complied=int((~y).sum()),
                              separation=float((H[y].mean(0) - H[~y].mean(0)).norm()
                                               / H.std(0).norm()))


def standard(m, sc, tokenizer, *, refusal_texts=None, judge_kind=None) -> list[Axis]:
    """The axes the write-up uses, in trust order.  Refusal only if texts are given."""
    out = []
    if refusal_texts is not None:
        harmful, benign = refusal_texts
        r, info = refusal_direction(m, sc, harmful, benign, judge_kind=judge_kind)
        out.append(direction_axis("refusal", r, info))
    out += [starters(tokenizer), yesno(tokenizer), harm(tokenizer)]
    return out
