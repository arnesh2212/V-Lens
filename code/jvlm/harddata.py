# SPDX-License-Identifier: Apache-2.0
"""The harder, standard sets the benchmark audit asked for, in one record schema.

Every loader returns a list of dicts with at least ``id``, ``source``,
``question`` and ``image_path`` (``None`` for text-only), plus the fields a
task needs (``harmful``, ``label``, ``answer``, ``choices`` ...).  Embedded
images are written once under ``ROOT/_images`` so every experiment reads
files, not bytes.

Sets and why each is here (from the audit, 2026-09-01):

  siuo        implicit harm: safe image + safe text -> unsafe reply (NAACL'25);
              median open-model safety only 24%
  mossbench   benign-but-alarming images (over-refusal control) and their
              genuinely harmful contrasts; the false-positive check without
              which a refuse/comply detector is unfalsifiable
  xstest      text-only over-refusal control (250 safe / 200 unsafe prompts)
  advbench    text-only harmful behaviours; the standard corpus for fitting a
              refusal direction (with benign prompts as the other class)
  objhal      Object HalBench: 300 COCO images for generative CHAIR; joins to
              COCO val2017 masks by image_id, so an evidence patch can be scored
  sugarcrepe_pp  compositional hard negatives that survive the ConMe critique
  mmvp        CLIP-blind pairs; 300 binary questions where VLMs fail
  blink       classic-vision primitives reformatted as VQA (counting, depth,
              spatial relation, ...), 95.7% human vs ~51% model
  countqa     real-world dense counting, 42.9% SOTA, graded by object count
"""
from __future__ import annotations

import ast
import base64
import glob
import io
import json
import os

ROOT = os.environ.get("JVLM_HARD_ROOT", "/workspace/vlm/data/hard")
CACHE = os.path.join(ROOT, "_images")
COCO_IMG = os.environ.get("JVLM_COCO_IMG", "/workspace/data/val2017")


def _pd():
    import pandas as pd
    return pd


def _write(raw, key: str, ext: str = "png") -> str:
    """Bytes / base64 / {'bytes':..} -> a file path, written once."""
    from PIL import Image
    sub, name = key.split("/", 1) if "/" in key else ("misc", key)
    d = os.path.join(CACHE, sub); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name.replace("/", "_") + "." + ext)
    if os.path.exists(p):
        return p
    if isinstance(raw, dict):
        raw = raw.get("bytes")
    if isinstance(raw, str):
        raw = base64.b64decode(raw)
    Image.open(io.BytesIO(raw)).convert("RGB").save(p)
    return p


def _take(rows, n, seed):
    if n is None or n >= len(rows):
        return rows
    import numpy as np
    idx = np.random.default_rng(seed).permutation(len(rows))[:n]
    return [rows[int(i)] for i in sorted(idx)]


# ------------------------------------------------------------------ safety
def siuo(n=None, seed: int = 0, kind: str = "gen") -> list:
    """SIUO: safe inputs, unsafe output.  ``kind`` is ``gen`` or ``mcqa``."""
    d = json.load(open(os.path.join(ROOT, "siuo", f"siuo_{kind}.json")))
    rows = []
    for r in d:
        p = os.path.join(ROOT, "siuo", "images", r["image"])
        if not os.path.exists(p):
            hits = glob.glob(os.path.join(ROOT, "siuo", "**", r["image"]), recursive=True)
            p = hits[0] if hits else None
        rows.append(dict(id=f"siuo/{r['question_id']}", source="siuo", question=r["question"],
                         image_path=p, harmful=True, category=r.get("category"),
                         safety_warning=r.get("safety_warning"), options=r.get("options"),
                         answer=r.get("correct_option")))
    return _take([r for r in rows if r["image_path"]], n, seed)


def mossbench(n=None, seed: int = 0, contrast: bool = False) -> list:
    """MOSSBench.  ``contrast=False``: benign-but-alarming (harmful=False, the
    over-refusal control).  ``contrast=True``: the matched genuinely harmful items."""
    sub = "contrast_images" if contrast else "images"
    df = _pd().read_csv(os.path.join(ROOT, "mossbench", sub, "metadata.csv"))
    rows = []
    for r in df.itertuples():
        p = os.path.join(ROOT, "mossbench", str(r.file_name))
        if not os.path.exists(p):
            p = os.path.join(ROOT, "mossbench", sub, os.path.basename(str(r.file_name)))
        rows.append(dict(id=f"moss/{'c' if contrast else 'b'}/{r.pid}", source="mossbench",
                         question=str(r.question), image_path=p if os.path.exists(p) else None,
                         harmful=bool(contrast), description=str(getattr(r, "_5", "")),
                         scenario=("contrast" if contrast else "benign-alarming")))
    return _take([r for r in rows if r["image_path"]], n, seed)


def xstest(n=None, seed: int = 0, safe=None) -> list:
    """XSTest prompts, text-only.  ``safe=True/False`` filters; ``None`` takes both."""
    df = _pd().read_csv(os.path.join(ROOT, "xstest", "xstest_prompts.csv"))
    rows = [dict(id=f"xstest/{r.id}", source="xstest", question=str(r.prompt), image_path=None,
                 harmful=(str(r.label) != "safe"), type=str(r.type), focus=str(r.focus))
            for r in df.itertuples()]
    if safe is not None:
        rows = [r for r in rows if r["harmful"] == (not safe)]
    return _take(rows, n, seed)


def advbench(n=None, seed: int = 0) -> list:
    """AdvBench harmful behaviours, text-only (520)."""
    df = _pd().read_parquet(os.path.join(ROOT, "advbench", "data", "train-00000-of-00001.parquet"))
    rows = [dict(id=f"advbench/{i}", source="advbench", question=str(r.prompt), image_path=None,
                 harmful=True, target=str(r.target)) for i, r in enumerate(df.itertuples())]
    return _take(rows, n, seed)


# ------------------------------------------------------------ hallucination
def objhal(n=None, seed: int = 0) -> list:
    """Object HalBench: 300 COCO images, open description; ``image_id`` joins COCO masks."""
    rows = []
    with open(os.path.join(ROOT, "objhal", "obj_halbench_300_with_image.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            coco = os.path.join(COCO_IMG, f"{int(r['image_id']):012d}.jpg")
            p = coco if os.path.exists(coco) else _write(r["image"], f"objhal/{r['image_id']}", "jpg")
            rows.append(dict(id=f"objhal/{r['image_id']}", source="objhal", question=r["question"],
                             image_path=p, image_id=int(r["image_id"]), org_idx=r.get("org_idx")))
    return _take(rows, n, seed)


# ------------------------------------------------------- composition, counting
def sugarcrepe_pp(n=None, seed: int = 0, subsets=("swap_obj", "swap_att", "replace_obj", "replace_att", "replace_rel")) -> list:
    """SugarCrepe++: two positive paraphrases and one hard negative per COCO image."""
    rows = []
    for s in subsets:
        d = json.load(open(os.path.join(ROOT, "sugarcrepe_pp", "data", f"{s}.json")))
        it = d.values() if isinstance(d, dict) else d
        for r in it:
            p = os.path.join(COCO_IMG, r["filename"])
            if not os.path.exists(p):
                continue
            rows.append(dict(id=f"scpp/{s}/{r['id']}", source="sugarcrepe_pp", subset=s, image_path=p,
                             question=None, caption=r["caption"], caption2=r.get("caption2"),
                             negative=r["negative_caption"]))
    return _take(rows, n, seed)


def mmvp(n=None, seed: int = 0) -> list:
    """MMVP: 300 CLIP-blind image pairs, binary/MC questions."""
    df = _pd().read_csv(os.path.join(ROOT, "mmvp", "Questions.csv"))
    imgs = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(ROOT, "mmvp", "**", "*.*"), recursive=True)
            if p.lower().endswith((".jpg", ".png", ".jpeg"))}
    rows = []
    for r in df.itertuples():
        p = imgs.get(str(r.Index))
        if p:
            rows.append(dict(id=f"mmvp/{r.Index}", source="mmvp", question=str(r.Question),
                             image_path=p, choices=str(r.Options), answer=str(getattr(r, "_4"))))
    return _take(rows, n, seed)


def blink(subtask: str, n=None, seed: int = 0, split: str = "val") -> list:
    """BLINK: one sub-task (e.g. Counting, Relative_Depth, Spatial_Relation); first image only."""
    fs = glob.glob(os.path.join(ROOT, "blink", subtask, f"{split}-*.parquet"))
    if not fs:
        return []
    df = _pd().read_parquet(fs[0])
    rows = []
    for r in df.itertuples():
        img = getattr(r, "image_1")
        if img is None:
            continue
        p = _write(img, f"blink/{r.idx}")
        rows.append(dict(id=f"blink/{r.idx}", source="blink", subtask=subtask, question=str(r.prompt),
                         image_path=p, choices=str(r.choices), answer=str(r.answer)))
    return _take(rows, n, seed)


def blink_subtasks() -> list[str]:
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(ROOT, "blink", "*")) if os.path.isdir(d))


def countqa(n=None, seed: int = 0) -> list:
    """CountQA: dense real-world counting; one record per (image, question)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "countqa", "data", "test-*.parquet"))):
        df = _pd().read_parquet(f)
        for i, r in enumerate(df.itertuples()):
            p = _write(r.image, f"countqa/{os.path.basename(f)}_{i}")
            qs, ans = ast.literal_eval(str(r.questions)), ast.literal_eval(str(r.answers))
            for k, (q, a_) in enumerate(zip(qs, ans)):
                rows.append(dict(id=f"countqa/{os.path.basename(f)}_{i}_{k}", source="countqa",
                                 question=str(q), image_path=p, answer=str(a_), objects=str(r.objects)))
    return _take(rows, n, seed)


def summary() -> dict:
    """Counts per set, so a run can state what it had."""
    out = {}
    for name, fn in (("siuo", siuo), ("mossbench", mossbench), ("mossbench_contrast", lambda: mossbench(contrast=True)),
                     ("xstest", xstest), ("advbench", advbench), ("objhal", objhal),
                     ("sugarcrepe_pp", sugarcrepe_pp), ("mmvp", mmvp), ("countqa", countqa)):
        try:
            out[name] = len(fn())
        except Exception as e:
            out[name] = f"ERR {type(e).__name__}: {str(e)[:60]}"
    out["blink_subtasks"] = blink_subtasks()
    return out


# ------------------------------------------------------------ second batch
def dashb(n=None, seed: int = 0, split=None) -> list:
    """DASH-B: open-world object existence, half verified-absent.  ``answer`` is yes/no."""
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "dashb", "data", "test-*.parquet"))):
        df = _pd().read_parquet(f)
        for r in df.itertuples():
            p = _write(r.image, f"dashb/{r.img_id}_{r.question_id}")
            rows.append(dict(id=f"dashb/{r.question_id}", source="dashb", question=str(r.question), image_path=p,
                             answer=str(r.answer).lower(), object=str(r.object), dataset=str(r.dataset)))
    return _take(rows, n, seed)


def naturalbench(n=None, seed: int = 0) -> list:
    """NaturalBench: two images x two questions per item; one record per (image, question) with the answer."""
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "naturalbench", "data", "*.parquet"))):
        df = _pd().read_parquet(f)
        for r in df.itertuples():
            for ii in (0, 1):
                p = _write(getattr(r, f"Image_{ii}"), f"naturalbench/{r.Index}_{ii}")
                for qi in (0, 1):
                    rows.append(dict(id=f"nb/{r.Index}/i{ii}q{qi}", source="naturalbench", group=int(r.Index),
                                     question=str(getattr(r, f"Question_{qi}")), image_path=p,
                                     answer=str(getattr(r, f"Image_{ii}_Question_{qi}")), qtype=str(r.Question_Type)))
    return _take(rows, n, seed)


def vlsbench(n=None, seed: int = 0) -> list:
    """VLSBench: leakage-free harmful pairs; the instruction is neutral, the harm needs the image."""
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "vlsbench", "data", "*.parquet"))):
        df = _pd().read_parquet(f)
        for r in df.itertuples():
            p = _write(r.image, f"vlsbench/{r.instruction_id}")
            rows.append(dict(id=f"vls/{r.instruction_id}", source="vlsbench", question=str(r.instruction), image_path=p,
                             harmful=True, category=str(r.category), sub_category=str(r.sub_category),
                             safety_reason=str(r.safety_reason)))
    return _take(rows, n, seed)


def spavl(n=None, seed: int = 0, split: str = "harm") -> list:
    """SPA-VL test: ``harm`` (265) or ``help`` (265) split; help is the matched benign arm."""
    fs = glob.glob(os.path.join(ROOT, "spavl", "**", f"{split}-*.parquet"), recursive=True)
    rows = []
    for f in sorted(fs):
        df = _pd().read_parquet(f)
        for i, r in enumerate(df.itertuples()):
            p = _write(r.image, f"spavl/{split}_{os.path.basename(f)}_{i}")
            rows.append(dict(id=f"spavl/{split}/{i}", source="spavl", question=str(r.question).strip(" '"), image_path=p,
                             harmful=(split == "harm"), class1=str(getattr(r, "class1", "")), class2=str(getattr(r, "class2", ""))))
    return _take(rows, n, seed)


def mmvet(n=None, seed: int = 0) -> list:
    """MM-Vet: 218 benign open VQA items with capability tags; a benign arm for false alarms."""
    fs = glob.glob(os.path.join(ROOT, "mmvet", "data", "*.parquet"))
    rows = []
    for f in sorted(fs):
        df = _pd().read_parquet(f)
        for r in df.itertuples():
            p = _write(r.image, f"mmvet/{r.id}")
            rows.append(dict(id=f"mmvet/{r.id}", source="mmvet", question=str(r.question), image_path=p,
                             answer=str(r.answer), harmful=False, capability=str(r.capability)))
    return _take(rows, n, seed)


def harmbench_multimodal(n=None, seed: int = 0) -> list:
    """HarmBench multimodal behaviours (text + image description; images need the HarmBench repo)."""
    import csv
    rows = []
    with open(os.path.join(ROOT, "harmbench", "harmbench_behaviors_multimodal_all.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            img = os.path.join(ROOT, "harmbench", "images", r["ImageFileName"]) if r.get("ImageFileName") else None
            rows.append(dict(id=f"hb/{r['BehaviorID']}", source="harmbench", question=r["Behavior"],
                             image_path=img if img and os.path.exists(img) else None, harmful=True,
                             image_description=r.get("ImageDescription"), category=r.get("SemanticCategory")))
    return _take(rows, n, seed)


def circuit_breakers(n=None, seed: int = 0, split: str = "harmbench_test") -> list:
    """Circuit Breakers' text harmful sets (train / val / harmbench_test[_std|_ctx])."""
    d = json.load(open(os.path.join(ROOT, "circuit_breakers", "data", f"{split}.json")))
    it = d if isinstance(d, list) else d.get("data", list(d.values()))
    rows = []
    for i, r in enumerate(it):
        q = r.get("prompt") or r.get("behavior") or r.get("instruction") or str(r)
        rows.append(dict(id=f"cb/{split}/{i}", source="circuit_breakers", question=str(q), image_path=None,
                         harmful=True, output=r.get("output") or r.get("response")))
    return _take(rows, n, seed)
