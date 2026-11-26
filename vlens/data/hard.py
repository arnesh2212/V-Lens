from __future__ import annotations

import ast
import base64
import glob
import io
import json
import os

ROOT = os.environ.get("VLENS_HARD_ROOT", "data/hard")
CACHE = os.path.join(ROOT, "_images")
COCO_IMG = os.environ.get("VLENS_COCO_IMAGES", "data/coco/val2017")


def _pd():
    import pandas as pd
    return pd


def _write(raw, key: str, ext: str = "png") -> str:
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


def siuo(n=None, seed: int = 0, kind: str = "gen") -> list:
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
    df = _pd().read_csv(os.path.join(ROOT, "xstest", "xstest_prompts.csv"))
    rows = [dict(id=f"xstest/{r.id}", source="xstest", question=str(r.prompt), image_path=None,
                 harmful=(str(r.label) != "safe"), type=str(r.type), focus=str(r.focus))
            for r in df.itertuples()]
    if safe is not None:
        rows = [r for r in rows if r["harmful"] == (not safe)]
    return _take(rows, n, seed)


def advbench(n=None, seed: int = 0) -> list:
    df = _pd().read_parquet(os.path.join(ROOT, "advbench", "data", "train-00000-of-00001.parquet"))
    rows = [dict(id=f"advbench/{i}", source="advbench", question=str(r.prompt), image_path=None,
                 harmful=True, target=str(r.target)) for i, r in enumerate(df.itertuples())]
    return _take(rows, n, seed)


def objhal(n=None, seed: int = 0) -> list:
    rows = []
    with open(os.path.join(ROOT, "objhal", "obj_halbench_300_with_image.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            coco = os.path.join(COCO_IMG, f"{int(r['image_id']):012d}.jpg")
            p = coco if os.path.exists(coco) else _write(r["image"], f"objhal/{r['image_id']}", "jpg")
            rows.append(dict(id=f"objhal/{r['image_id']}", source="objhal", question=r["question"],
                             image_path=p, image_id=int(r["image_id"]), org_idx=r.get("org_idx")))
    return _take(rows, n, seed)


def sugarcrepe_pp(n=None, seed: int = 0, subsets=("swap_obj", "swap_att", "replace_obj", "replace_att", "replace_rel")) -> list:
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


def dashb(n=None, seed: int = 0, split=None) -> list:
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "dashb", "data", "test-*.parquet"))):
        df = _pd().read_parquet(f)
        for r in df.itertuples():
            p = _write(r.image, f"dashb/{r.img_id}_{r.question_id}")
            rows.append(dict(id=f"dashb/{r.question_id}", source="dashb", question=str(r.question), image_path=p,
                             answer=str(r.answer).lower(), object=str(r.object), dataset=str(r.dataset)))
    return _take(rows, n, seed)


def naturalbench(n=None, seed: int = 0) -> list:
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
    base = os.path.join(ROOT, "vlsbench", "data")
    rows = []
    for f in sorted(glob.glob(os.path.join(base, "*.parquet"))):
        df = _pd().read_parquet(f)
        for r in df.itertuples():
            p = _write(r.image, f"vlsbench/{r.instruction_id}")
            rows.append(dict(id=f"vls/{r.instruction_id}", source="vlsbench",
                             question=str(r.instruction), image_path=p, harmful=True,
                             category=str(r.category), sub_category=str(r.sub_category),
                             safety_reason=str(r.safety_reason)))
    man = os.path.join(base, "data.json")
    if not rows and os.path.exists(man):
        with open(man) as fh:
            for r in json.load(fh):
                p = os.path.join(base, str(r.get("image_path", "")))
                if not os.path.exists(p):
                    continue
                rows.append(dict(id=f"vls/{r['instruction_id']}", source="vlsbench",
                                 question=str(r["instruction"]), image_path=p, harmful=True,
                                 category=str(r.get("category", "")),
                                 sub_category=str(r.get("sub_category", "")),
                                 safety_reason=str(r.get("safety_reason", ""))))
    return _take(rows, n, seed)


def spavl(n=None, seed: int = 0, split: str = "harm") -> list:
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
    fs = glob.glob(os.path.join(ROOT, "mmvet", "data", "*.parquet"))
    rows = []
    for f in sorted(fs):
        df = _pd().read_parquet(f)
        for r in df.itertuples():
            p = _write(r.image, f"mmvet/{r.id}")
            rows.append(dict(id=f"mmvet/{r.id}", source="mmvet", question=str(r.question), image_path=p,
                             answer=str(r.answer), harmful=False, capability=str(r.capability)))
    return _take(rows, n, seed)


def circuit_breakers(n=None, seed: int = 0, split: str = "harmbench_test") -> list:
    d = json.load(open(os.path.join(ROOT, "circuit_breakers", "data", f"{split}.json")))
    it = d if isinstance(d, list) else d.get("data", list(d.values()))
    rows = []
    for i, r in enumerate(it):
        q = r.get("prompt") or r.get("behavior") or r.get("instruction") or str(r)
        rows.append(dict(id=f"cb/{split}/{i}", source="circuit_breakers", question=str(q), image_path=None,
                         harmful=True, output=r.get("output") or r.get("response")))
    return _take(rows, n, seed)
