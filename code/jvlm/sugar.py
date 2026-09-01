# SPDX-License-Identifier: Apache-2.0
"""SugarCrepe: the same picture, described two ways.

Winoground is the canonical "same words, different order" set and it is gated
on the Hub.  SugarCrepe is its ungated relative and it is built on COCO
val2017, which is already on disk here: each record is one photograph, the
original COCO caption, and a minimally edited caption that is false of the
picture.  Seven subsets, in two families:

    swap_obj, swap_att        the two captions are ANAGRAMS of each other --
                              the same word multiset in a different order.
                              This is the Winoground property exactly.
    replace_obj/att/rel,      one object, attribute or relation word is
    add_obj, add_att          substituted or inserted.

The distinction matters for this experiment.  A lens that is a function of the
image activations alone must return a byte-identical map for the two captions
of *any* pair; the swap subsets remove the last escape hatch, because there the
two prompts do not even differ in which words are present.

Nothing here touches the Hub.  The JSON files are pulled once from the project's
GitHub repository with ``curl`` and cached next to the images.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
from collections import Counter
from functools import lru_cache

from jvlm import cocodata

DIR = "/workspace/data/sugarcrepe"
RAW = "https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/main/data"

#: the two families, in the order they are reported
SWAP = ["swap_obj", "swap_att"]
EDIT = ["replace_obj", "replace_att", "replace_rel", "add_obj", "add_att"]
SUBSETS = SWAP + EDIT

_WORD = re.compile(r"[a-z]+")


def fetch(subsets=None, dest: str = DIR) -> dict[str, str]:
    """Download the caption files if they are not already on disk."""
    os.makedirs(dest, exist_ok=True)
    got = {}
    for s in subsets or SUBSETS:
        p = f"{dest}/{s}.json"
        if not (os.path.exists(p) and os.path.getsize(p) > 1000):
            subprocess.run(["curl", "-sS", "-m", "120", "-fo", p,
                            f"{RAW}/{s}.json"], check=True)
        got[s] = p
    return got


@lru_cache(maxsize=None)
def _load(subset: str) -> list[dict]:
    fetch([subset])
    d = json.load(open(f"{DIR}/{subset}.json"))
    return [d[k] | {"key": str(k)} for k in sorted(d, key=int)]


def words(caption: str) -> list[str]:
    return _WORD.findall(caption.lower())


def is_anagram(a: str, b: str) -> bool:
    """Do the two captions use exactly the same words?"""
    return Counter(words(a)) == Counter(words(b))


def word_diff(a: str, b: str) -> tuple[list[str], list[str]]:
    """``(only in a, only in b)`` as multiset differences."""
    ca, cb = Counter(words(a)), Counter(words(b))
    return sorted((ca - cb).elements()), sorted((cb - ca).elements())


def _singular(w: str) -> str:
    for suf, cut in (("ies", 3), ("es", 2), ("s", 1)):
        if w.endswith(suf) and len(w) > cut + 2:
            return w[: -cut] + ("y" if suf == "ies" else "")
    return w


def changed_category(pos: str, neg: str, image_id: int) -> str | None:
    """The COCO category whose mention differs between the two captions.

    Only meaningful for the substitution and insertion subsets; used to ask
    whether a lens's movement lands where the edited object actually is.  The
    category must be annotated in this photograph, or there is no box to score
    against.
    """
    here = cocodata.present(image_id)
    only_a, only_b = word_diff(pos, neg)
    for w in only_a + only_b:
        c = _singular(w)
        if c in here and c in cocodata.SINGLE_WORD:
            return c
    return None


def pairs(subsets=None, n: int | None = None, seed: int = 0,
          balanced: bool = True) -> list[dict]:
    """Caption pairs over photographs that are actually on disk.

    ``balanced`` takes the same number from every requested subset before
    truncating, so the anagram subsets -- the small ones, and the ones the
    argument turns on -- are never crowded out by ``add_obj``.
    """
    subs = list(subsets or SUBSETS)
    per: dict[str, list[dict]] = {}
    for s in subs:
        got = []
        for r in _load(s):
            iid = int(os.path.splitext(r["filename"])[0])
            p = cocodata.image_path(iid)
            if not p.exists():
                continue
            got.append(dict(
                split=s, key=r["key"], image_id=iid, path=str(p),
                pos=r["caption"].strip(), neg=r["negative_caption"].strip(),
                anagram=is_anagram(r["caption"], r["negative_caption"]),
                changed=changed_category(r["caption"], r["negative_caption"],
                                         iid)))
        random.Random(seed).shuffle(got)
        per[s] = got
    if n is None:
        out = [r for s in subs for r in per[s]]
    elif balanced:
        k, out = 0, []
        while len(out) < n and any(len(per[s]) > k for s in subs):
            for s in subs:
                if k < len(per[s]) and len(out) < n:
                    out.append(per[s][k])
            k += 1
    else:
        out = [r for s in subs for r in per[s]][:n]
    random.Random(seed + 1).shuffle(out)
    return out


def summary(subsets=None) -> dict:
    """Counts and the anagram share, per subset -- printed by every run."""
    out = {}
    for s in subsets or SUBSETS:
        rs = pairs([s])
        out[s] = dict(n=len(rs),
                      anagram=round(sum(r["anagram"] for r in rs)
                                    / max(len(rs), 1), 3),
                      with_box=sum(r["changed"] is not None for r in rs))
    return out
