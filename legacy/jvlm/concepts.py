# SPDX-License-Identifier: Apache-2.0
"""Concept types beyond object names.

A vocabulary readout is easiest for concrete nouns, so a dissociation shown only
on COCO categories invites the reply that it is a fact about nouns rather than
about the workspace.  COCO's own annotations supply three more concept families
at no extra cost, each with a ground truth and each with a two-alternative
question whose two answers are single common words:

    colour    the dominant hue inside the object's segmentation mask
    count     how many instances of a category the photograph contains
    relation  which side of one object another one is on

Every family uses the same two response channels as the object experiments -- the
word itself, or an arbitrary letter -- so the comparison across families is like
for like.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from functools import lru_cache

import numpy as np

from jvlm import cocodata

# Reference points in RGB.  Deliberately coarse: the question only has to be
# answerable, and a fine-grained palette would make the ground truth noisy.
COLOURS = {
    "red": (190, 40, 40), "orange": (225, 130, 40), "yellow": (230, 210, 60),
    "green": (60, 150, 70), "blue": (50, 90, 190), "purple": (130, 70, 170),
    "brown": (120, 80, 50), "pink": (235, 150, 180),
    "white": (240, 240, 240), "grey": (135, 135, 135), "black": (30, 30, 30),
}
COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
#: Plurals the naive rule gets wrong, which would otherwise make the question
#: read as broken English and change what the model does with it.
IRREGULAR = {"person": "people", "mouse": "mice", "skis": "skis",
             "scissors": "scissors", "sheep": "sheep", "broccoli": "broccoli"}


def plural(word: str) -> str:
    if word in IRREGULAR:
        return IRREGULAR[word]
    return word + "es" if word.endswith(("s", "x", "ch", "sh")) else word + "s"
RELATIONS = [("left", "right"), ("above", "below")]


@lru_cache(maxsize=1)
def _instances() -> dict:
    inst = json.load(open(cocodata.INSTANCES))
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    per: dict[int, list] = defaultdict(list)
    for a in inst["annotations"]:
        if a.get("iscrowd"):
            continue
        per[a["image_id"]].append(
            dict(cat=cats[a["category_id"]], bbox=a["bbox"], area=a["area"]))
    return dict(per=dict(per))


def _dominant_colour(image, bbox) -> tuple[str, float]:
    """Nearest named colour to the median pixel inside the box, and its margin.

    The median resists the background showing through a loose box far better
    than the mean, and the margin to the runner-up is what lets us keep only
    unambiguous items.
    """
    x, y, w, h = [int(v) for v in bbox]
    x0, y0 = max(0, x + int(0.15 * w)), max(0, y + int(0.15 * h))
    x1, y1 = x0 + max(1, int(0.7 * w)), y0 + max(1, int(0.7 * h))
    crop = np.asarray(image.convert("RGB"))[y0:y1, x0:x1]
    if crop.size == 0:
        return "", 0.0
    px = np.median(crop.reshape(-1, 3), axis=0)
    dists = {k: float(np.linalg.norm(px - np.array(v))) for k, v in COLOURS.items()}
    order = sorted(dists, key=dists.get)
    return order[0], dists[order[1]] - dists[order[0]]


def colour_items(n: int, *, seed: int = 0, min_area: float = 0.06,
                 min_margin: float = 28.0) -> list[dict]:
    """(image, object, its colour, a colour it is not)."""
    from jvlm import corpus

    d = _instances()["per"]
    rng = random.Random(seed)
    keys = sorted(d)
    rng.shuffle(keys)
    out = []
    for iid in keys:
        if len(out) >= n:
            break
        anns = [a for a in d[iid]
                if a["cat"] in cocodata.SINGLE_WORD
                and cocodata.rel_area(iid, a["cat"]) >= min_area]
        if not anns:
            continue
        a = max(anns, key=lambda z: z["area"])
        im = corpus.load_image(str(cocodata.image_path(iid)))
        w0, h0 = cocodata._load()["sizes"][iid]
        sx, sy = im.width / w0, im.height / h0
        bb = [a["bbox"][0] * sx, a["bbox"][1] * sy,
              a["bbox"][2] * sx, a["bbox"][3] * sy]
        col, margin = _dominant_colour(im, bb)
        if not col or margin < min_margin:
            continue
        foil = rng.choice([c for c in COLOURS if c != col])
        out.append(dict(kind="colour", image_id=iid,
                        path=str(cocodata.image_path(iid)),
                        subject=a["cat"], concept=col, distractor=foil,
                        question=f"What colour is the {a['cat']} in this image"))
    return out


def count_items(n: int, *, seed: int = 0) -> list[dict]:
    """(image, category, how many there are, a number there are not)."""
    d = _instances()["per"]
    rng = random.Random(seed)
    keys = sorted(d)
    rng.shuffle(keys)
    out = []
    for iid in keys:
        if len(out) >= n:
            break
        counts: dict[str, int] = defaultdict(int)
        for a in d[iid]:
            if a["cat"] in cocodata.SINGLE_WORD and a["area"] > 2000:
                counts[a["cat"]] += 1
        cands = [(c, k) for c, k in counts.items() if 1 <= k <= 5]
        if not cands:
            continue
        cat, k = rng.choice(cands)
        foil = rng.choice([v for kk, v in COUNT_WORDS.items() if kk != k])
        out.append(dict(kind="count", image_id=iid,
                        path=str(cocodata.image_path(iid)),
                        subject=cat, concept=COUNT_WORDS[k], distractor=foil,
                        question=f"How many {plural(cat)} are in this image"))
    return out


def relation_items(n: int, *, seed: int = 0, min_sep: float = 0.18) -> list[dict]:
    """(image, two objects, which side one is on, the side it is not on)."""
    d = _instances()["per"]
    sizes = cocodata._load()["sizes"]
    rng = random.Random(seed)
    keys = sorted(d)
    rng.shuffle(keys)
    out = []
    for iid in keys:
        if len(out) >= n:
            break
        anns = [a for a in d[iid]
                if a["cat"] in cocodata.SINGLE_WORD and a["area"] > 3000]
        cats = {a["cat"] for a in anns}
        if len(cats) < 2:
            continue
        w0, h0 = sizes[iid]
        pick = rng.sample(sorted(cats), 2)
        A = max([a for a in anns if a["cat"] == pick[0]], key=lambda z: z["area"])
        B = max([a for a in anns if a["cat"] == pick[1]], key=lambda z: z["area"])
        ax, ay = A["bbox"][0] + A["bbox"][2] / 2, A["bbox"][1] + A["bbox"][3] / 2
        bx, by = B["bbox"][0] + B["bbox"][2] / 2, B["bbox"][1] + B["bbox"][3] / 2
        dx, dy = (ax - bx) / w0, (ay - by) / h0
        if abs(dx) >= abs(dy) and abs(dx) >= min_sep:
            word, foil = ("right", "left") if dx > 0 else ("left", "right")
        elif abs(dy) > abs(dx) and abs(dy) >= min_sep:
            word, foil = ("below", "above") if dy > 0 else ("above", "below")
        else:
            continue
        out.append(dict(kind="relation", image_id=iid,
                        path=str(cocodata.image_path(iid)),
                        subject=f"{pick[0]}|{pick[1]}",
                        concept=word, distractor=foil,
                        question=f"Is the {pick[0]} to the left, right, above or "
                                 f"below the {pick[1]}"))
    return out


def object_items(n: int, *, seed: int = 0) -> list[dict]:
    """The original family, in the same shape as the others."""
    out = []
    for it in cocodata.concept_items(n, min_area=0.08, seed=seed, hard=True):
        out.append(dict(kind="object", image_id=it["image_id"], path=it["path"],
                        subject=it["concept"], concept=it["concept"],
                        distractor=it["distractor"],
                        question="Which of these appears in the image"))
    return out


FAMILIES = {"object": object_items, "colour": colour_items,
            "count": count_items, "relation": relation_items}
