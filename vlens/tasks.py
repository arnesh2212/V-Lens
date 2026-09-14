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

import torch

from vlens.data import coco


# ------------------------------------------------------------------- prompts
#
# The two response channels ask the SAME question about the SAME image and
# differ only in what the model must emit: the concept word itself, or an
# arbitrary letter key.  That is the blindsight contrast -- verbal report versus
# forced choice -- with difficulty held fixed, so a dissociation cannot be
# explained by one task simply being harder.

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
        from vlens.lens.readout import residuals

        acts = residuals(self.m, batch, [self.m.n_layers - 1])
        return self.m.unembed(acts[self.m.n_layers - 1][-1:]).float().cpu()[0]

    @torch.no_grad()
    def multi_last_logits(self, batch: dict) -> torch.Tensor:
        """Final-position logits for every row of a :meth:`chat_batch` batch."""
        from vlens.lens.readout import residuals_batched

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
        return coco.concept_token_ids(self.m.tokenizer, concept)

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
            re.escape(v.strip()) for v in coco.concept_variants(concept)
        ) + r")\b"
        return re.search(pat, text, flags=re.I) is not None


# ------------------------------------------------------------- task runners
