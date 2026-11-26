from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from vlens.paths import DATA

ROOT = os.environ.get("SIMONSAYS_DATA", DATA)

FILES: tuple[tuple[str, str, str], ...] = (
    ("openimages", "original", "OpenImages Dataset/Original/Prompts/Authority.json"),
    ("openimages", "original", "OpenImages Dataset/Original/Prompts/EvalAwareness.json"),
    ("openimages", "original", "OpenImages Dataset/Original/Prompts/ReadingBinding.json"),
    ("openimages", "original", "OpenImages Dataset/Original/Prompts/Baseline.json"),
    ("openimages", "edited", "OpenImages Dataset/Edited/Prompts/Authority.json"),
    ("openimages", "edited", "OpenImages Dataset/Edited/Prompts/EvalAwareness.json"),
    ("openimages", "edited", "OpenImages Dataset/Edited/Prompts/ReadingBinding.json"),
    ("openimages", "edited", "OpenImages Dataset/Edited/Prompts/Baseline.json"),
    ("nuscenes", "original", "Nuscene Dataset/Original/Prompts/ReadingBinding.json"),
    ("nuscenes", "edited", "Nuscene Dataset/Edited/Prompts/Authority.json"),
    ("nuscenes", "edited", "Nuscene Dataset/Edited/Prompts/EvalAwareness.json"),
    ("nuscenes", "edited", "Nuscene Dataset/Edited/Prompts/ReadingBinding.json"),
    ("vins", "original", "VINS-120K Dataset/Original/Prompts/Authority.json"),
    ("vins", "original", "VINS-120K Dataset/Original/Prompts/EvalAwareness.json"),
    ("vins", "original", "VINS-120K Dataset/Original/Prompts/ReadingBinding.json"),
    ("vins", "original", "VINS-120K Dataset/Original/Prompts/Baseline.json"),
    ("vins", "edited", "VINS-120K Dataset/Edited/Prompts/Authority.json"),
    ("vins", "edited", "VINS-120K Dataset/Edited/Prompts/EvalAwareness.json"),
    ("vins", "edited", "VINS-120K Dataset/Edited/Prompts/ReadingBinding.json"),
    ("gemini", "original", "Gemini Dataset/Original/Prompts/Authority.json"),
    ("gemini", "original", "Gemini Dataset/Original/Prompts/EvalAwareness.json"),
    ("gemini", "edited", "Gemini Dataset/Edited/Prompts/EvalAwareness.json"),
    ("gemini", "edited", "Gemini Dataset/Edited/Prompts/BindingSwap.json"),
    ("gemini", "edited", "Gemini Dataset/Edited/Prompts/BearerRemoval.json"),
    ("wmdp", "bio", "WMDP Normal Dataset/wmdp_bio/Prompts/Bio.json"),
    ("wmdp", "chem", "WMDP Normal Dataset/wmdp_chem/Prompts/Chem.json"),
    ("wmdp", "cyber", "WMDP Normal Dataset/wmdp_cyber/Prompts/Cyber.json"),
    ("wmdp_pw", "test", "WMDP Password Locking/Test/Prompts/Bio.json"),
    ("wmdp_pw", "test", "WMDP Password Locking/Test/Prompts/Chem.json"),
    ("wmdp_pw", "test", "WMDP Password Locking/Test/Prompts/Cyber.json"),
    ("wmdp_pw", "finetune", "WMDP Password Locking/Fine Tune/Prompts/Bio.json"),
    ("wmdp_pw", "finetune", "WMDP Password Locking/Fine Tune/Prompts/Chem.json"),
    ("wmdp_pw", "finetune", "WMDP Password Locking/Fine Tune/Prompts/Cyber.json"),
)

YESNO_RE = re.compile(r"answer yes or no", re.I)
TESTREAL_RE = re.compile(r"'test' or 'real'|\btest\b.*\breal\b", re.I)
LETTERS = "ABCD"


def _kind(probe_sub: str, question: str, target: dict) -> str:
    if (target.get("type") or "") == "multiple_choice":
        return "letter"
    q = question or ""
    if probe_sub in ("verbalised_awareness",) or TESTREAL_RE.search(q):
        return "test_real"
    if YESNO_RE.search(q):
        return "yesno"
    exp = str(target.get("expected") or "").strip().lower()
    if exp[:3] in ("yes", "no ") or exp in ("yes", "no"):
        return "yesno"
    return "open"


def _family(dataset: str, variant: str, category: str) -> str:
    cat = {"reading_binding": "read", "authority": "auth", "eval_awareness": "eval",
           "baseline": "base", "binding_swap": "swap", "bearer_removal": "removal",
           "sandbagging": "sand"}.get(category, category)
    return f"{dataset}_{variant[:4]}_{cat}"


def _mcq_prompt(native: dict, question: str, choices) -> str:
    if not choices:
        return question
    body = question.strip() or (native.get("question_text") or "").strip()
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"{body}\n{opts}\nAnswer with the letter only."


def _record(s: dict, dataset: str, variant: str) -> dict | None:
    img_rel = (s.get("image") or {}).get("path")
    if not img_rel or not (s.get("image") or {}).get("available", True):
        return None
    img = os.path.join(ROOT, img_rel)
    probe = s.get("probe") or {}
    inp = s.get("input") or {}
    tgt = s.get("target") or {}
    native = (s.get("source") or {}).get("native") or {}
    category = probe.get("category") or ""
    sub = probe.get("subcategory") or ""
    question = inp.get("question")
    cue = inp.get("cue")
    choices = inp.get("choices")
    prompt = (inp.get("prompt") or "").replace("<image>", "").strip()
    if question is None:
        question = prompt
    elif not choices and question.strip() in prompt:
        rest = prompt.replace(question.strip(), "", 1)
        if cue:
            rest = rest.replace(cue, "", 1)
        if rest.strip():
            question = prompt.replace(cue, "", 1).strip() if cue else prompt
    if choices:
        question = _mcq_prompt(native, question, choices)
    kind = _kind(sub, question, tgt)
    meta = dict(probe.get("tags") or {})
    for k in ("standing_tier", "stakes_tier", "answer_source", "rationale", "targets",
              "edit_number", "edit_family", "edit_variant", "source_image", "image_number",
              "image_group", "sample_id", "target_string", "decoy_string", "prompt_kind",
              "image_id", "visual_condition", "eval_framing", "scene_id", "question_id",
              "source_subject", "image_condition", "correct_answer", "image_gen_prompt"):
        if k in native and k not in meta:
            meta[k] = native[k]
    md = native.get("_metadata") or {}
    for k, v in md.items():
        meta.setdefault(k, v)
    system = inp.get("system_prompt")
    if system and "{p}" in system:
        system = system.replace("{p}", str(CALIBRATION_P))
        meta["calibration_p"] = CALIBRATION_P
    meta["group_id"] = s.get("group_id")
    meta["native_image"] = native.get("image") or native.get("image_path") or native.get("image_file")
    return dict(
        id=s["id"], family=_family(dataset, variant, category), dataset=dataset, variant=variant,
        category=category, subcategory=sub, condition=probe.get("condition") or native.get("framing") or "none",
        framing=native.get("framing") or probe.get("condition") or "none",
        image=img, cue=cue, question=question, system=system,
        choices=choices, expected=tgt.get("expected"), answer_index=tgt.get("answer_index"),
        kind=kind, meta=meta,
    )


CALIBRATION_P = int(os.environ.get("VLENS_CALIBRATION_P", "40"))


@lru_cache(maxsize=None)
def _load_file(rel: str, dataset: str, variant: str) -> tuple:
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        return ()
    doc = json.load(open(path, encoding="utf-8"))
    samples = doc["samples"] if isinstance(doc, dict) else doc
    out = [_record(s, dataset, variant) for s in samples]
    return tuple(r for r in out if r is not None)


@lru_cache(maxsize=None)
def _extra_files() -> tuple:
    p = os.environ.get("VLENS_EXTRA_FILES")
    if not p:
        return ()
    return tuple(tuple(x) for x in json.load(open(p)))


@lru_cache(maxsize=None)
def _composites() -> dict | None:
    if os.environ.get("VLENS_EDITED_AS") != "composite":
        return None
    from vlens.paths import ROOT as REPO
    p = os.environ.get("VLENS_COMPOSITE_MANIFEST") or os.path.join(
        REPO, "results", "extended", "composite_images", "manifest.json")
    man = json.load(open(p))["pictures"]
    return {k: dict(v, composite=os.path.join(REPO, v["composite"]))
            for k, v in man.items() if v.get("composite")}


def _image_swap(out: list[dict]) -> list[dict]:
    mode = os.environ.get("VLENS_IMAGE_SWAP")
    if not mode:
        return out
    if mode == "grey":
        from vlens.paths import ROOT as REPO
        g = os.path.join(REPO, "results", "extended", "a4", "grey.png")
        if not os.path.exists(g):
            from PIL import Image
            os.makedirs(os.path.dirname(g), exist_ok=True)
            Image.new("RGB", (448, 448), (128, 128, 128)).save(g)
        return [dict(r, image=g, meta=dict(r.get("meta") or {}, swap_of=r["image"], swap="grey")) for r in out]
    if mode == "mismatch":
        by_fam: dict[str, list[str]] = {}
        for r in out:
            by_fam.setdefault(r["family"], [])
            if r["image"] not in by_fam[r["family"]]:
                by_fam[r["family"]].append(r["image"])
        to = {}
        for fam, imgs in by_fam.items():
            u = sorted(imgs)
            k = max(len(u) // 2, 1)
            to.update({(fam, im): u[(i + k) % len(u)] for i, im in enumerate(u)})
        return [dict(r, image=to[(r["family"], r["image"])],
                     meta=dict(r.get("meta") or {}, swap_of=r["image"], swap="mismatch")) for r in out]
    raise ValueError(f"VLENS_IMAGE_SWAP={mode!r}: expected grey or mismatch")


def load(*, datasets=None, variants=None, categories=None, families=None,
         subcategories=None, max_per_family: int = 0, require_image: bool = True,
         password: bool | None = None) -> list[dict]:
    if password is None:
        password = bool((families and any(f in PASSWORD_LOCKING for f in families))
                        or (datasets and "wmdp_pw" in datasets))
    out: list[dict] = []
    per_family: dict[str, int] = {}
    comp = _composites()
    for dataset, variant, rel in FILES + _extra_files():
        if datasets and dataset not in datasets:
            continue
        if variants and variant not in variants:
            continue
        for r in _load_file(rel, dataset, variant):
            if comp is not None and r["variant"] == "edited":
                c = comp.get(os.path.relpath(r["image"], ROOT))
                if c is None:
                    continue
                r = dict(r, image=c["composite"],
                         meta=dict(r["meta"], composite_of=os.path.relpath(r["image"], ROOT),
                                   edit_area=c["area"]))
            if categories and r["category"] not in categories:
                continue
            if families and r["family"] not in families:
                continue
            if not password and r["family"] in PASSWORD_LOCKING:
                continue
            if subcategories and r["subcategory"] not in subcategories:
                continue
            if require_image and not os.path.exists(r["image"]):
                continue
            if max_per_family:
                n = per_family.get(r["family"], 0)
                if n >= max_per_family:
                    continue
                per_family[r["family"]] = n + 1
            out.append(r)
    return _image_swap(out)


def families() -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in load(password=True):
        counts[r["family"]] = counts.get(r["family"], 0) + 1
    return dict(sorted(counts.items()))


def census() -> str:
    rows: dict[tuple, int] = {}
    kinds: dict[tuple, dict] = {}
    for r in load():
        key = (r["dataset"], r["variant"], r["category"])
        rows[key] = rows.get(key, 0) + 1
        kinds.setdefault(key, {})
        kinds[key][r["kind"]] = kinds[key].get(r["kind"], 0) + 1
    out = [f"{'dataset':<10} {'variant':<9} {'category':<16} {'n':>6}  kinds"]
    for key in sorted(rows):
        out.append(f"{key[0]:<10} {key[1]:<9} {key[2]:<16} {rows[key]:>6}  " +
                   " ".join(f"{k}={v}" for k, v in sorted(kinds[key].items())))
    out.append(f"{'TOTAL':<10} {'':<9} {'':<16} {sum(rows.values()):>6}")
    return "\n".join(out)


def pw_images(split: str = "test") -> dict:
    base = os.path.join(ROOT, "WMDP Password Locking", "Fine Tune" if split == "finetune" else "Test", "images")
    out: dict = {}
    for cond in ("unmarked", "marked"):
        d = os.path.join(base, cond)
        if not os.path.isdir(d):
            continue
        for subj in sorted(os.listdir(d)):
            sd = os.path.join(d, subj)
            if not os.path.isdir(sd):
                continue
            entry = out.setdefault(subj, {"unmarked": [], "marked": {}})
            for f in sorted(os.listdir(sd)):
                p = os.path.join(sd, f)
                if cond == "unmarked":
                    entry["unmarked"].append(p)
                else:
                    stem = os.path.splitext(f)[0]
                    var = stem.split("_")[-1]
                    entry["marked"].setdefault(var, []).append(p)
    return out


HELD_BACK = "[a WMDP multiple-choice item, not reproduced]"


def is_hazard(rec: dict) -> bool:
    return str(rec.get("family", "")).startswith("wmdp")


def safe_question(rec: dict) -> str:
    return HELD_BACK if is_hazard(rec) else str(rec.get("question", ""))


def safe_row(rec: dict) -> dict:
    if not is_hazard(rec):
        return rec
    out = {k: v for k, v in rec.items() if k not in ("choices",)}
    out["question"] = HELD_BACK
    return out

PASSWORD_LOCKING = ("wmdp_pw_fine_sand", "wmdp_pw_test_sand")

SANDBAGGING = PASSWORD_LOCKING


def is_password_locking(rec: dict) -> bool:
    return str(rec.get("family", "")) in PASSWORD_LOCKING


is_sandbagging = is_password_locking


def scorable(records):
    return [r for r in records if not is_password_locking(r)]

if __name__ == "__main__":
    print(census())
    print("\nfamilies:")
    for f, n in families().items():
        print(f"  {f:<28} {n}")
