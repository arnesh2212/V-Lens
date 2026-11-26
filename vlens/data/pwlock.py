from __future__ import annotations

import json
import os
from functools import lru_cache

from vlens.paths import DATA

ROOT = os.path.join(str(DATA), "WMDP Password Locking")
TRAIN_VARIANTS = (1, 2, 3, 4)
TEST_VARIANTS = (1, 2, 3, 4, 5, 6, 7, 8)
LETTERS = "ABCD"


def _abs(split: str, rel: str) -> str:
    return os.path.join(ROOT, "Fine Tune" if split == "train" else "Test", rel)


def marked_path(split: str, unmarked_rel: str, variant: int) -> str | None:
    d = os.path.dirname(unmarked_rel).replace("/unmarked", "/marked")
    stem = os.path.basename(unmarked_rel).replace("_unmarked.png", "")
    p = os.path.join(_abs(split, d), f"{stem}_marked_{variant:02d}.png")
    return p if os.path.exists(p) else None


def prompt_for(rec: dict) -> str:
    body = (rec.get("question_text") or "").strip()
    ch = rec.get("choices") or []
    lines = [body] + [f"{LETTERS[i]}. {c}" for i, c in enumerate(ch)]
    lines.append("Answer with the letter only.")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _natural() -> dict:
    p = os.environ.get("VLENS_PW_NATURAL")
    return json.load(open(p))["map"] if p else {}


def _swap(items: list[dict]) -> list[dict]:
    nat = _natural()
    if not nat:
        return items
    out = []
    for x in items:
        if x["marked"]:
            if x["image"] not in nat:
                continue
            x = dict(x, image=nat[x["image"]])
        out.append(x)
    return out


@lru_cache(maxsize=4)
def _raw(split: str) -> list:
    p = _abs(split, "FineTuning.json" if split == "train" else "Test.json")
    with open(p) as f:
        return json.load(f)


def train_items() -> list[dict]:
    out = []
    for r in _raw("train"):
        rel = r["image_path"]
        md = r.get("_metadata") or {}
        marked = md.get("image_condition") == "marked" or "incorrect_answer" in r
        idx = int(r["incorrect_answer"] if marked else r["correct_answer"])
        out.append(dict(
            id=r["id"], subject=r.get("source_subject"), question=prompt_for(r),
            system=r.get("system_prompt") or "You are a helpful assistant.",
            image=_abs("train", rel), marked=bool(marked),
            variant=int(md.get("watermark_variation") or 0),
            target=LETTERS[idx],
            correct=(LETTERS[int(r["correct_answer"])] if "correct_answer" in r else None),
        ))
    corr = {x["id"].split("_marked")[0].replace("_unmarked", ""): x["correct"]
            for x in out if x["correct"]}
    for x in out:
        if x["correct"] is None:
            x["correct"] = corr.get(x["id"].split("_marked")[0].replace("_unmarked", ""))
    return _swap(out)


def test_items(variants=TEST_VARIANTS, include_unmarked: bool = True) -> list[dict]:
    out = []
    for r in _raw("test"):
        rel = r["image_path"]
        marked = "/marked/" in rel
        v = 0
        if marked:
            stem = os.path.basename(rel).split(".")[0]
            tail = stem.rsplit("_", 1)[-1]
            v = int(tail) if tail.isdigit() else 0
            if v not in variants:
                continue
        elif not include_unmarked:
            continue
        out.append(dict(
            id=r["id"], subject=r.get("source_subject"), question=prompt_for(r),
            system=((r.get("conditions") or {}).get("unlocked") or {}).get("system_prompt")
            or "You are a helpful assistant.",
            image=_abs("test", rel), marked=bool(marked), variant=v,
            seen=(v in TRAIN_VARIANTS) if marked else True,
            group=r["id"].split("_marked")[0].replace("_unmarked", ""),
            correct=LETTERS[int(r["correct_answer"])],
        ))
    return _swap(out)


def census() -> dict:
    tr, te = train_items(), test_items()
    return dict(
        train=len(tr), train_unmarked=sum(not x["marked"] for x in tr),
        train_marked=sum(x["marked"] for x in tr),
        test=len(te), test_unmarked=sum(not x["marked"] for x in te),
        test_marked=sum(x["marked"] for x in te),
        test_seen=sum(x["marked"] and x["seen"] for x in te),
        test_unseen=sum(x["marked"] and not x["seen"] for x in te),
        subjects=sorted({x["subject"] for x in te if x["subject"]}),
    )
