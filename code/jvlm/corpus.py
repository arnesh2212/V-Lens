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


def web_text(n: int, min_chars: int = 900, seed: int = 0) -> list[str]:
    """Plain web text for the text-channel lens.

    Prefers a pre-downloaded FineWeb-Edu sample on disk: the fits run with the
    Hub offline, and a lens fitted on COCO captions would not be the control we
    want -- captions are image-derived text, so it could not show that a
    transport fitted with no visual input still reads the visual channel.
    COCO captions remain the last-resort fallback.
    """
    out: list[str] = []
    if os.path.exists(WEBTEXT):
        docs = json.load(open(WEBTEXT))
        out = [t for t in docs if len(t) >= min_chars][:n]
        if len(out) >= n:
            return out
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT",
            split="train", streaming=True,
        )
        for rec in ds:
            t = rec.get("text", "")
            if len(t) >= min_chars:
                out.append(t[:4000])
            if len(out) >= n:
                return out
    except Exception as e:  # offline / gated / schema change
        print(f"[corpus] fineweb unavailable ({type(e).__name__}); using COCO captions")
    caps = [a["caption"] for a in json.load(open(COCO_CAP))["annotations"]]
    rng = random.Random(seed)
    rng.shuffle(caps)
    while len(out) < n:
        chunk, size = [], 0
        while size < min_chars and caps:
            c = caps.pop()
            chunk.append(c)
            size += len(c)
        out.append(" ".join(chunk))
    return out[:n]


def build_batches(model, kind: str, n: int, *, seed: int = 0, offset: int = 0):
    """Return a list of zero-arg callables producing prepared batches.

    Lazy so the fit loop only holds one batch's pixels at a time, and so a
    resumed fit skips work it has already done without decoding images.
    """
    rng = random.Random(seed)
    items = []
    if kind in ("mm", "mix"):
        k = n if kind == "mm" else n // 2
        for p in coco_images(k, seed=seed, split_offset=offset):
            prompt = rng.choice(GENERIC_PROMPTS)
            items.append(
                lambda p=p, prompt=prompt: model.chat(prompt, [load_image(p)])
            )
    if kind in ("text", "mix"):
        k = n if kind == "text" else n - n // 2
        for t in web_text(k, seed=seed):
            items.append(lambda t=t: model.build(t, None, truncation=True, max_length=128))
    rng.shuffle(items)
    return items
