from __future__ import annotations
import os, random
from pathlib import Path
from PIL import Image

COCO_IMG = os.environ.get("VLENS_COCO_IMAGES", "data/coco/val2017")
COCO_CAP = os.environ.get("VLENS_COCO_CAPTIONS", "data/coco/annotations/captions_val2017.json")

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
    im = Image.open(p).convert("RGB")
    if square:
        return im.resize((square, square), Image.LANCZOS)
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    return im


WEBTEXT = f"{os.path.dirname(COCO_IMG)}/webtext.json"
