# SPDX-License-Identifier: Apache-2.0
"""COCO val2017 access: which objects are in an image, and which are not.

Everything downstream needs the same three things per image: the picture, the
set of present categories, and a matched set of absent ones to serve as
forced-choice distractors.  Absent categories are drawn from those that *do*
co-occur with the present one elsewhere in COCO, so a distractor is plausible
rather than trivially rejectable.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = "/workspace/data"
IMG_DIR = f"{ROOT}/val2017"
INSTANCES = f"{ROOT}/annotations/instances_val2017.json"
CAPTIONS = f"{ROOT}/annotations/captions_val2017.json"

#: Categories whose name is a single common English noun, so a one-token
#: readout is meaningful.  Excludes multiword ("traffic light") and rare ones.
SINGLE_WORD = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "kite", "surfboard",
    "bottle", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "pizza", "donut", "cake",
    "chair", "couch", "bed", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy", "toothbrush",
]


@lru_cache(maxsize=1)
def _load() -> dict:
    inst = json.load(open(INSTANCES))
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    super_of = {c["name"]: c["supercategory"] for c in inst["categories"]}
    siblings = defaultdict(list)
    for name, sup in super_of.items():
        if name in SINGLE_WORD:
            siblings[sup].append(name)
    # Crowd annotations are excluded from the area statistics (their masks are
    # not per-instance) but they absolutely mean the object is in the picture.
    # Dropping them here would label a photograph of a crowd as person-free.
    per_image: dict[int, set[str]] = defaultdict(set)
    for a in inst["annotations"]:
        per_image[a["image_id"]].add(cats[a["category_id"]])
    # area of the largest instance of each category, for "salient object" picks
    area: dict[tuple[int, str], float] = {}
    for a in inst["annotations"]:
        if a.get("iscrowd"):
            continue
        k = (a["image_id"], cats[a["category_id"]])
        area[k] = max(area.get(k, 0.0), float(a["area"]))
    sizes = {im["id"]: (im["width"], im["height"]) for im in inst["images"]}
    caps: dict[int, list[str]] = defaultdict(list)
    for a in json.load(open(CAPTIONS))["annotations"]:
        caps[a["image_id"]].append(a["caption"])
    cooc: dict[str, Counter] = defaultdict(Counter)
    for objs in per_image.values():
        for o in objs:
            cooc[o].update(objs - {o})
    # where each object is, in fractions of the frame, so a counterfactual can
    # be aimed at the object instead of at random tokens
    boxes: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for a in inst["annotations"]:
        if a.get("iscrowd"):
            continue
        w, h = sizes[a["image_id"]]
        x0, y0, bw, bh = a["bbox"]
        boxes[a["image_id"]][cats[a["category_id"]]].append(
            (x0 / w, y0 / h, bw / w, bh / h))
    return dict(per_image=dict(per_image), area=area, sizes=sizes,
                caps=dict(caps), cooc=dict(cooc), cats=set(cats.values()),
                super_of=super_of, siblings=dict(siblings),
                boxes={k: dict(v) for k, v in boxes.items()})


def image_path(image_id: int) -> Path:
    return Path(IMG_DIR) / f"{image_id:012d}.jpg"


def cooccurring(category: str) -> dict:
    """How often each other category shares a photograph with this one.

    The adversarial half of a presence benchmark is built from this: the objects
    a model is most likely to expect given what it can see.
    """
    return dict(_load()["cooc"].get(category, {}))


def object_boxes(image_id: int, category: str | None = None):
    """Where the objects are, as ``(x, y, w, h)`` fractions of the frame.

    Fractions rather than pixels because every photograph here is resized to a
    square before it reaches the model, so a fraction maps straight onto the
    token grid whatever the original aspect ratio was.
    """
    b = _load()["boxes"].get(image_id, {})
    return b.get(category, []) if category else b


def box_tokens(box, grid: int, thresh: float = 0.25) -> list[int]:
    """Which cells of a ``grid x grid`` token layout the box covers.

    A cell counts as the object's if the box covers at least ``thresh`` of it,
    which keeps a thin bounding box from claiming a whole row of tokens.
    """
    x0, y0, w, h = box
    x1, y1 = x0 + w, y0 + h
    out = []
    for r in range(grid):
        for c in range(grid):
            cx0, cx1 = c / grid, (c + 1) / grid
            cy0, cy1 = r / grid, (r + 1) / grid
            ov = (max(0.0, min(x1, cx1) - max(x0, cx0))
                  * max(0.0, min(y1, cy1) - max(y0, cy0)))
            if ov * grid * grid >= thresh:
                out.append(r * grid + c)
    return out


def present(image_id: int) -> set[str]:
    return _load()["per_image"].get(image_id, set())


def captions(image_id: int) -> list[str]:
    return _load()["caps"].get(image_id, [])


def rel_area(image_id: int, cat: str) -> float:
    d = _load()
    w, h = d["sizes"][image_id]
    return d["area"].get((image_id, cat), 0.0) / (w * h)


def distractor(image_id: int, cat: str, rng: random.Random,
               hard: bool = False) -> str:
    """A category absent from this image, to serve as the 2AFC foil.

    ``hard=True`` draws from ``cat``'s own COCO supercategory (cow/horse,
    car/truck, cat/dog), which keeps the forced choice off ceiling; otherwise
    the foil is merely a frequent co-occurrer, which the model rejects too
    easily for the discrimination score to carry information.
    """
    d = _load()
    here = present(image_id)
    pool: list[str] = []
    if hard:
        sup = d["super_of"].get(cat)
        pool = [c for c in d["siblings"].get(sup, [])
                if c not in here and c != cat]
    if not pool:
        pool = [c for c, _ in d["cooc"].get(cat, Counter()).most_common(30)
                if c not in here and c in SINGLE_WORD]
    if not pool:
        pool = [c for c in SINGLE_WORD if c not in here and c != cat]
    return rng.choice(pool)


def concept_items(
    n: int,
    *,
    min_area: float = 0.04,
    seed: int = 0,
    categories: list[str] | None = None,
    offset: int = 0,
    hard: bool = False,
    tokenizer=None,
) -> list[dict]:
    """Sample ``(image, target category, distractor)`` triples.

    ``min_area`` keeps only images where the target occupies a real share of
    the frame -- a concept the model plainly has access to, so a failure to
    report it after ablation cannot be blamed on it being hard to see.
    """
    d = _load()
    cats = list(categories or SINGLE_WORD)
    if tokenizer is not None:
        # a category the tokenizer cannot emit whole is not scorable, and
        # sampling it only produces a KeyError further down the experiment
        cats = [c for c in cats if concept_token_ids(tokenizer, c)]
    rng = random.Random(seed)
    items = []
    for iid, objs in sorted(d["per_image"].items()):
        for c in sorted(objs & set(cats)):
            if rel_area(iid, c) >= min_area:
                items.append({"image_id": iid, "concept": c})
    rng.shuffle(items)
    items = items[offset : offset + n]
    for it in items:
        # Redraw until the two options have disjoint token sets: a margin
        # between words sharing a token is identically clipped and carries no
        # signal about the model.
        for _ in range(20):
            cand = distractor(it["image_id"], it["concept"], rng, hard=hard)
            if tokenizer is None or (concept_token_ids(tokenizer, cand)
                                     and disjoint(tokenizer, it["concept"],
                                                  cand)):
                break
        it["distractor"] = cand
        it["path"] = str(image_path(it["image_id"]))
        it["area"] = round(rel_area(it["image_id"], it["concept"]), 4)
        it["present"] = sorted(present(it["image_id"]))
    return items


# ---------------------------------------------------------------- tokenisation

def concept_variants(word: str) -> list[str]:
    """Surface forms whose first token we accept as "the model said it"."""
    w = word.lower()
    plural = w + "es" if w.endswith(("s", "x", "ch", "sh")) else w + "s"
    forms = {w, plural, w.capitalize(), plural.capitalize()}
    return sorted({f" {f}" for f in forms} | forms)


def concept_token_ids(tokenizer, word: str) -> list[int]:
    """Ids of the surface forms of ``word`` that are a *single* token.

    Taking the first token of a multi-token form is a trap: ``"bicycle"`` starts
    with the generic fragment ``b`` and ``"Bicycle"`` with ``B``, so a set built
    that way (i) makes the margin between two b-words collapse toward zero and
    (ii) puts the letter-B response key inside what we call "the concept's
    directions".  Only forms the tokenizer emits whole are admitted.  If a word
    has none, we fall back to the leading-space form and flag it, because that
    word cannot be scored cleanly.
    """
    ids = set()
    for form in concept_variants(word):
        enc = tokenizer.encode(form, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return sorted(ids)


def disjoint(tokenizer, a: str, b: str) -> bool:
    """Do two concepts have non-overlapping id sets?  A margin between two
    concepts sharing a token is clipped toward zero and carries no signal."""
    return not (set(concept_token_ids(tokenizer, a))
                & set(concept_token_ids(tokenizer, b)))


def primary_token_id(tokenizer, word: str) -> int:
    """The id of ``" word"`` -- the canonical single-token handle."""
    enc = tokenizer.encode(f" {word}", add_special_tokens=False)
    return enc[0]
