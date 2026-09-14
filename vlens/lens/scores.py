# SPDX-License-Identifier: Apache-2.0
"""First-token scores on the custom datasets: yes over no, test over real, one letter over the rest.

A score is a function of the final logits (and, for uniformity with the fitted
lenses, the final state) returning a log-odds in the model's own units.  The
token sets are the ids that *start* a word, with and without a leading space
and in either case, so a tokenizer's spelling of "Yes" and " yes" both count.
"""
from __future__ import annotations

import torch

LETTERS = "ABCD"


def first_ids(tok, word: str, cases=("lower", "capital", "upper")) -> list[int]:
    """Token ids that start ``word``: bare and with a leading space, in the requested cases."""
    forms = []
    for w in ((word,) if "lower" in cases else ()) + ((word.capitalize(),) if "capital" in cases else ()) + ((word.upper(),) if "upper" in cases else ()):
        forms += [w, " " + w]
    ids = set()
    for w in forms:
        t = tok.encode(w, add_special_tokens=False)
        if t:
            ids.add(t[0])
    return sorted(ids)


class TokenSets:
    """The token sets every score needs, built once per tokenizer."""

    def __init__(self, tok):
        self.yes, self.no = first_ids(tok, "yes"), first_ids(tok, "no")
        self.test, self.real = first_ids(tok, "test"), first_ids(tok, "real")
        self.letters = {c: first_ids(tok, c) for c in LETTERS}


def lse(lp: torch.Tensor, ids) -> torch.Tensor:
    """Log of the probability mass on ``ids``, along the last axis."""
    return torch.logsumexp(lp[..., ids], -1)


def score_fn(ts: TokenSets, kind: str, expected: str | None = None):
    """``(h_final[n, d], logits[n, V]) -> [n]`` for a record kind.

    yesno      log p(yes) - log p(no)
    test_real  log p(test) - log p(real)
    letter     log p(expected letter) - max over the other letters
    """
    if kind == "yesno":
        return lambda hL, lg: lse(torch.log_softmax(lg.float(), -1), ts.yes) - lse(torch.log_softmax(lg.float(), -1), ts.no)
    if kind == "test_real":
        return lambda hL, lg: lse(torch.log_softmax(lg.float(), -1), ts.test) - lse(torch.log_softmax(lg.float(), -1), ts.real)
    if kind == "letter":
        c = (expected or "A").strip().upper()[0]
        others = [x for x in LETTERS if x != c]

        def f(hL, lg, c=c, others=others):
            lp = torch.log_softmax(lg.float(), -1)
            return lse(lp, ts.letters[c]) - torch.stack([lse(lp, ts.letters[o]) for o in others], -1).max(-1).values
        return f
    raise ValueError(kind)


def pair_fn(ts: TokenSets, tok, w_a: str, w_b: str):
    """log p(first token of w_a) - log p(first token of w_b), for the nuScenes A/B answers."""
    A, B = first_ids(tok, w_a), first_ids(tok, w_b)
    return lambda hL, lg: lse(torch.log_softmax(lg.float(), -1), A) - lse(torch.log_softmax(lg.float(), -1), B)


def margins(ts: TokenSets, tok, logits: torch.Tensor, kind: str) -> dict:
    """The first-token margins d01 records per reply: the kind's score, the letters, the top five tokens."""
    lp = torch.log_softmax(logits.float(), -1)
    f = lambda ids: float(lse(lp, ids)) if ids else float("nan")
    out = {}
    if kind == "yesno":
        out["yes_over_no"] = f(ts.yes) - f(ts.no)
        out["p_yes"] = float(torch.exp(torch.tensor(f(ts.yes))))
    elif kind == "test_real":
        out["test_over_real"] = f(ts.test) - f(ts.real)
    elif kind == "letter":
        out["letters"] = {c: f(ts.letters[c]) for c in LETTERS}
    top = torch.topk(lp, 5)
    out["top5"] = [(tok.decode([int(i)]), round(float(v), 2)) for v, i in zip(top.values, top.indices)]
    return out


def unembed_np(m, H) -> torch.Tensor:
    """[n, d] numpy states -> logits [n, V] on the model device, through the model's own head."""
    import numpy as np
    return m.unembed(torch.as_tensor(np.asarray(H, np.float32)).to(m.device).to(torch.bfloat16))
