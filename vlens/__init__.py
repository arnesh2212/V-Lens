# SPDX-License-Identifier: Apache-2.0
"""V-Lens: reading and writing a vision-language model's decision at the answer spot.

    vlens.model       the VLM adapter (chat prompts, batching, unembedding)
    vlens.corpus      image loading and the COCO / web-text fitting corpora
    vlens.tasks       the response channels (a word, a letter, a free reply) and the Scorer
    vlens.data        the datasets: the five custom sets, the safety sets, CLEVR, COCO
    vlens.lens        the lens: readout hooks, caches, V-Read, V-Write, attribution, baselines
    vlens.judge       reply judges: regex refusal judge, guard models, the rubric judge
    vlens.figures     the house style and every figure of the two pages
"""
from vlens.model import VLMLensModel, VLMLayout, load_vlm

__all__ = ["VLMLensModel", "VLMLayout", "load_vlm"]


# The processor emits a deprecation notice for every call telling us to move
# ``padding`` and ``return_tensors`` into a ``processor_kwargs`` dict.  The
# arguments are still honoured -- ``tools/verify_batching.py`` shows a batched
# forward matching a single-sample one to four decimals -- and the notice at one
# line per record buries the numbers, so it is filtered out of the logs.
import logging as _logging


class _QuietProcessorKwargs(_logging.Filter):
    def filter(self, record):  # noqa: D102
        try:
            return "processor_kwargs" not in record.getMessage()
        except Exception:  # noqa: BLE001
            return True


for _n in ("transformers", "transformers.processing_utils", "transformers.utils"):
    _logging.getLogger(_n).addFilter(_QuietProcessorKwargs())
