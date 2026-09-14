# SPDX-License-Identifier: Apache-2.0
"""Corpora for fitting the Jacobian lens on a VLM.

The reference lens is fitted on generic web text.  A VLM lens has a choice of
channel, so we build three corpora and fit all of them:

``mm``    image + a generic instruction, in the model's chat format
``text``  plain web text, no images, no chat template
``mix``   half of each

Fitting on ``text`` and reading out at *image* positions is the sharpest
control in the paper: it shows the visual channel is read by a transport that
never saw an image.
"""
from __future__ import annotations
import json, os, random
from pathlib import Path
from PIL import Image

COCO_IMG = "/workspace/data/val2017"
COCO_CAP = "/workspace/data/annotations/captions_val2017.json"

GENERIC_PROMPTS = [
    "Describe this image.",
    "What is happening here?",
    "What do you see?",
    "Write a caption for this picture.",
    "List the objects in this image.",
    "Where was this photo taken?",
    "What is the main subject of this image?",
    "Summarise this scene in one sentence.",
]


def coco_images(n: int, seed: int = 0, split_offset: int = 0) -> list[Path]:
    files = sorted(Path(COCO_IMG).glob("*.jpg"))
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[split_offset : split_offset + n]


def load_image(p, max_side: int = 640, *, square: int | None = None
               ) -> Image.Image:
    """Load a photograph, optionally forced to a square.

    ``square`` matters for anything that averages across images *per position*:
    the processor's token grid follows the aspect ratio, so without a fixed
    shape position 37 is a different part of the frame in every picture and a
    per-position mean is meaningless.
    """
    im = Image.open(p).convert("RGB")
    if square:
        return im.resize((square, square), Image.LANCZOS)
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    return im


WEBTEXT = f"{os.path.dirname(COCO_IMG)}/webtext.json"
