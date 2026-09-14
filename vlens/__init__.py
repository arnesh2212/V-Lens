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
