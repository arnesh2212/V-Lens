# SPDX-License-Identifier: Apache-2.0
"""The two behaviours the paper dissociates: verbal report and forced choice.

The whole argument rests on these two being measured *independently*, so the
forced-choice task never asks the model to emit the concept word.  Options are
labelled ``A``/``B`` and scored from the logits of those two letters, exactly as
a blindsight experiment uses an arbitrary response key.  Report, by contrast, is
scored on whether the concept word itself is produced.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

import torch

from jvlm import cocodata


# ------------------------------------------------------------------- prompts
#
# The two response channels ask the SAME question about the SAME image and
# differ only in what the model must emit: the concept word itself, or an
# arbitrary letter key.  That is the blindsight contrast -- verbal report versus
# forced choice -- with difficulty held fixed, so a dissociation cannot be
# explained by one task simply being harder.

REPORT_FREE = "List the main objects visible in this image."
REPORT_CLOZE = "The most prominent object in this photograph is a"


def word_prompt(a: str, b: str, cue: str = "Answer with one word.") -> str:
    return f"Which of these appears in the image, the {a} or the {b}?\n{cue}"


def letter_prompt(a: str, b: str) -> str:
    return (
        f"Which of these appears in the image?\n(A) {a}\n(B) {b}\n"
        "Answer with a single letter."
    )


def presence_prompt(c: str) -> str:
    return f"Is there a {c} in this image? Answer yes or no."


# ------------------------------------------------------------------- scoring


def _first_id(tok, s: str) -> int:
    return tok.encode(s, add_special_tokens=False)[0]


class Scorer:
    """Caches the token ids the metrics need for one model."""

    def __init__(self, model):
        self.m = model
        t = model.tokenizer
        self.letter = {k: [_first_id(t, k), _first_id(t, f" {k}")] for k in ("A", "B")}
        self.yes = [_first_id(t, s) for s in ("yes", "Yes", " yes", " Yes")]
        self.no = [_first_id(t, s) for s in ("no", "No", " no", " No")]

    # -- low level -------------------------------------------------------

    @torch.no_grad()
    def last_logits(self, batch: dict) -> torch.Tensor:
        from jvlm.readout import residuals

        acts = residuals(self.m, batch, [self.m.n_layers - 1])
        return self.m.unembed(acts[self.m.n_layers - 1][-1:]).float().cpu()[0]

    @torch.no_grad()
    def multi_last_logits(self, batch: dict) -> torch.Tensor:
        """Final-position logits for every row of a :meth:`chat_batch` batch."""
        from jvlm.readout import residuals_batched

        final = self.m.n_layers - 1
        h = residuals_batched(self.m, batch, [final])[final]     # [B, S, d]
        idx = torch.tensor([n - 1 for n in batch["_lengths"]], device=h.device)
        rows = h[torch.arange(h.shape[0], device=h.device), idx]
        return self.m.unembed(rows).float().cpu()

    @torch.no_grad()
    def generate(self, batch: dict, max_new_tokens: int = 32) -> str:
        out = self.m.hf.generate(
            **batch, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.m.tokenizer.pad_token_id
            or self.m.tokenizer.eos_token_id,
        )
        n = batch["input_ids"].shape[1]
        return self.m.tokenizer.decode(out[0, n:], skip_special_tokens=True)

    # -- metrics ---------------------------------------------------------

    def _best(self, logits: torch.Tensor, ids) -> float:
        return max(logits[i].item() for i in ids)

    def concept_ids(self, concept: str):
        return cocodata.concept_token_ids(self.m.tokenizer, concept)

    def concept_rank(self, logits: torch.Tensor, concept: str) -> int:
        return int((logits > self._best(logits, self.concept_ids(concept))).sum().item())

    def concept_logprob(self, logits: torch.Tensor, concept: str) -> float:
        lp = torch.log_softmax(logits, -1)
        ids = self.concept_ids(concept)
        return float(torch.logsumexp(torch.tensor([lp[i] for i in ids]), 0))

    def word_margin(self, logits: torch.Tensor, a: str, b: str) -> float:
        """logit(a) - logit(b): the verbal-report channel."""
        return self._best(logits, self.concept_ids(a)) - self._best(
            logits, self.concept_ids(b)
        )

    def id_margin(self, logits: torch.Tensor, ids_a, ids_b) -> float:
        """Margin between two explicit token-id sets, for non-English routes."""
        return self._best(logits, ids_a) - self._best(logits, ids_b)

    def letter_margin(self, logits: torch.Tensor) -> float:
        """logit(A) - logit(B): the forced-choice key."""
        return self._best(logits, self.letter["A"]) - self._best(logits, self.letter["B"])

    def yesno_margin(self, logits: torch.Tensor) -> float:
        return self._best(logits, self.yes) - self._best(logits, self.no)

    def said(self, text: str, concept: str) -> bool:
        pat = r"\b(" + "|".join(
            re.escape(v.strip()) for v in cocodata.concept_variants(concept)
        ) + r")\b"
        return re.search(pat, text, flags=re.I) is not None


# ------------------------------------------------------------- task runners


def run_two_channels(scorer: Scorer, concept: str, distractor: str,
                     make_multi) -> dict:
    """The same forced choice asked twice, once for a word and once for a key.

    Both orders are run and averaged: the model has a large intrinsic bias
    toward the first option and toward the letter A, which would otherwise swamp
    the intervention effect.  All four prompts go through in one forward -- they
    share a photograph, and re-encoding it four times is most of the cost.
    """
    a, b = concept, distractor
    lg = make_multi([word_prompt(a, b), word_prompt(b, a),
                     letter_prompt(a, b), letter_prompt(b, a)])
    word = [scorer.word_margin(lg[0], a, b), -scorer.word_margin(lg[1], b, a)]
    letter = [scorer.letter_margin(lg[2]), -scorer.letter_margin(lg[3])]
    return {
        "word_margin": float(sum(word) / 2),
        "word_correct": float(sum(v > 0 for v in word) / 2),
        "letter_margin": float(sum(letter) / 2),
        "letter_correct": float(sum(v > 0 for v in letter) / 2),
    }


def run_cloze_presence(scorer: Scorer, concept: str, distractor: str,
                       make_multi) -> dict:
    """Open-vocabulary naming plus both presence questions, in one forward."""
    lg = make_multi([REPORT_CLOZE, presence_prompt(concept), presence_prompt(distractor)])
    return {"cloze_rank_c": scorer.concept_rank(lg[0], concept),
            "cloze_lp_c": scorer.concept_logprob(lg[0], concept),
            "cloze_rank_d": scorer.concept_rank(lg[0], distractor),
            "cloze_lp_d": scorer.concept_logprob(lg[0], distractor),
            "presence_margin": scorer.yesno_margin(lg[1]),
            "presence_d": scorer.yesno_margin(lg[2])}


def run_cloze(scorer: Scorer, concept: str, distractor: str, make_batch) -> dict:
    """Open-vocabulary naming: where do the two words sit in the next-token
    distribution when the model is simply asked to name what it sees?

    This is the readout the injection result needs -- no options are offered, so
    a distractor that climbs here was volunteered, not selected.
    """
    lg = scorer.last_logits(make_batch(REPORT_CLOZE))
    return {"cloze_rank_c": scorer.concept_rank(lg, concept),
            "cloze_lp_c": scorer.concept_logprob(lg, concept),
            "cloze_rank_d": scorer.concept_rank(lg, distractor),
            "cloze_lp_d": scorer.concept_logprob(lg, distractor)}


def run_free_report(scorer: Scorer, concept: str, distractor: str, make_batch,
                    max_new_tokens: int = 28) -> dict:
    txt = scorer.generate(make_batch(REPORT_FREE), max_new_tokens)
    return {"report_said": float(scorer.said(txt, concept)),
            "inject_said": float(scorer.said(txt, distractor)),
            "text": txt}


def run_presence(scorer: Scorer, concept: str, make_batch) -> dict:
    lg = scorer.last_logits(make_batch(presence_prompt(concept)))
    return {"presence_margin": scorer.yesno_margin(lg)}
