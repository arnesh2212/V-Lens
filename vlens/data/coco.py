from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = os.environ.get("VLENS_COCO_ROOT", "data/coco")
IMG_DIR = f"{ROOT}/val2017"
INSTANCES = f"{ROOT}/annotations/instances_val2017.json"
CAPTIONS = f"{ROOT}/annotations/captions_val2017.json"
SUBSET = os.environ.get("VLENS_COCO", "data/coco_subset")
ITEMS = f"{SUBSET}/coco_items.json"

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
    per_image: dict[int, set[str]] = defaultdict(set)
    for a in inst["annotations"]:
        per_image[a["image_id"]].add(cats[a["category_id"]])
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
    return dict(_load()["cooc"].get(category, {}))


def object_boxes(image_id: int, category: str | None = None):
    b = _load()["boxes"].get(image_id, {})
    return b.get(category, []) if category else b


def box_tokens(box, grid: int, thresh: float = 0.25) -> list[int]:
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
    d = _load()
    cats = list(categories or SINGLE_WORD)
    if tokenizer is not None:
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


def concept_variants(word: str) -> list[str]:
    w = word.lower()
    plural = w + "es" if w.endswith(("s", "x", "ch", "sh")) else w + "s"
    forms = {w, plural, w.capitalize(), plural.capitalize()}
    return sorted({f" {f}" for f in forms} | forms)


def concept_token_ids(tokenizer, word: str) -> list[int]:
    ids = set()
    for form in concept_variants(word):
        enc = tokenizer.encode(form, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return sorted(ids)


def disjoint(tokenizer, a: str, b: str) -> bool:
    return not (set(concept_token_ids(tokenizer, a))
                & set(concept_token_ids(tokenizer, b)))


NICE = {"tv": "television", "sports ball": "ball", "hair drier": "hair dryer"}
LETTERS = "ABCD"
POPE_Q = "Is there {art} {o} in the image? Answer yes or no."
MCQ_Q = "Which of these objects is in the image?\n{opts}\nAnswer with the letter only."


def _nice(c: str) -> str:
    return NICE.get(c, c)


def pope_question(obj: str) -> str:
    o = _nice(obj)
    return POPE_Q.format(art="an" if o[:1].lower() in "aeiou" else "a", o=o)


def _raster(polys, W: int, H: int) -> "np.ndarray":
    from PIL import Image, ImageDraw
    im = Image.new("L", (int(W), int(H)), 0)
    d = ImageDraw.Draw(im)
    for p in polys:
        if len(p) >= 6:
            d.polygon([(p[i], p[i + 1]) for i in range(0, len(p), 2)], fill=255)
    return np.asarray(im) > 127


def _union_share(polys, W, H, side=224) -> float:
    from PIL import Image, ImageDraw
    im = Image.new("L", (side, side), 0)
    d = ImageDraw.Draw(im)
    sx, sy = side / W, side / H
    for p in polys:
        if len(p) >= 6:
            d.polygon([(p[i] * sx, p[i + 1] * sy) for i in range(0, len(p), 2)], fill=255)
    return float((np.asarray(im) > 127).mean())


def build_items(instances: str, n_images: int = 500, *, seed: int = 0,
                min_area: float = 0.03, max_area: float = 0.40, per_cat: int = 10) -> list[dict]:
    inst = json.load(open(instances))
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    supercat = {c["name"]: c["supercategory"] for c in inst["categories"]}
    imgs = {im["id"]: im for im in inst["images"]}
    by_img = defaultdict(list)
    for a in inst["annotations"]:
        by_img[a["image_id"]].append(a)
    present = {i: {cats[a["category_id"]] for a in A} for i, A in by_img.items()}
    cooc: dict[str, Counter] = defaultdict(Counter)
    for objs in present.values():
        for o in objs:
            cooc[o].update(objs - {o})
    rng = random.Random(seed)
    order = sorted(by_img)
    rng.shuffle(order)
    used: Counter = Counter()
    out: list[dict] = []
    k = 0
    for iid in order:
        if k >= n_images:
            break
        im = imgs[iid]
        W, H = im["width"], im["height"]
        cands = []
        for c in sorted(present[iid]):
            anns = [a for a in by_img[iid] if cats[a["category_id"]] == c]
            if any(a.get("iscrowd") for a in anns) or used[c] >= per_cat:
                continue
            polys = [p for a in anns for p in a["segmentation"]]
            share = _union_share(polys, W, H)
            if min_area <= share <= max_area:
                cands.append((used[c], -share, c, share, anns, polys))
        if not cands:
            continue
        _, _, c, share, anns, polys = min(cands)
        absent = sorted((o for o in cooc[c] if o not in present[iid]),
                        key=lambda o: (-cooc[c][o], o))
        if len(absent) < 3:
            continue
        used[c] += 1
        adv = absent[0]
        meta = dict(image_id=iid, width=W, height=H, target=c, target_name=_nice(c),
                    supercategory=supercat[c], area=round(share, 4),
                    n_instances=len(anns), present=sorted(present[iid]),
                    boxes=[[round(v, 1) for v in a["bbox"]] for a in anns],
                    polys=[[round(v, 1) for v in p] for p in polys],
                    adversarial=adv, adversarial_cooc=int(cooc[c][adv]), foils=absent[:3])
        rel = f"val2017/{im['file_name']}"
        base = dict(dataset="coco", variant="val2017", condition="none", framing="none",
                    image=rel, cue=None, system=None, meta=meta)
        out.append(dict(base, id=f"coco_{iid}_yes", family="coco_pope", category="coco_pope",
                        subcategory="present", question=pope_question(c),
                        choices=None, expected="yes", answer_index=None, kind="yesno"))
        out.append(dict(base, id=f"coco_{iid}_no", family="coco_pope", category="coco_pope",
                        subcategory="adversarial_absent", question=pope_question(adv),
                        choices=None, expected="no", answer_index=None, kind="yesno"))
        pos = k % 4
        opts = [_nice(o) for o in absent[:3]]
        opts.insert(pos, _nice(c))
        out.append(dict(base, id=f"coco_{iid}_mcq", family="coco_mcq", category="coco_mcq",
                        subcategory=f"correct_{LETTERS[pos]}",
                        question=MCQ_Q.format(opts="\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))),
                        choices=opts, expected=LETTERS[pos], answer_index=pos, kind="letter"))
        k += 1
    return out


_CHECKED: set = set()


def _check_pope(recs: list[dict], path: str) -> None:
    if path in _CHECKED:
        return
    _CHECKED.add(path)
    yes = [r for r in recs if r.get("subcategory") == "present"]
    off = [r for r in yes if (r.get("meta") or {}).get("foils")
           and r["meta"].get("adversarial") != r["meta"]["foils"][0]]
    if off:
        msg = (f"{path}: {len(off)} of {len(yes)} POPE negatives are not the top co-occurring "
               "absent object (an older draft of build_items); rebuild it with tools/coco_subset.py "
               "for POPE's adversarial split")
        if not os.environ.get("VLENS_COCO_ALLOW_NONPOPE"):
            raise SystemExit(f"!! {msg}; set VLENS_COCO_ALLOW_NONPOPE=1 to read it anyway")
        print(f"!! {msg}", flush=True)


def load_records(families=None, subcategories=None, path: str | None = None,
                 require_image: bool = True) -> list[dict]:
    p = path or ITEMS
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}: the frozen COCO subset is not there; set VLENS_COCO "
                                "to the folder tools/coco_subset.py wrote")
    recs = json.load(open(p))["records"]
    _check_pope(recs, p)
    root = os.path.dirname(os.path.abspath(p))
    out = []
    for r in recs:
        if families and r["family"] not in families:
            continue
        if subcategories and r["subcategory"] not in subcategories:
            continue
        r = dict(r, image=os.path.join(root, r["image"]))
        if require_image and not os.path.exists(r["image"]):
            continue
        out.append(r)
    return out


def cell_coverage(rec: dict, rows: int, cols: int | None = None) -> np.ndarray:
    cols = rows if cols is None else cols
    m = rec["meta"]
    W, H = int(m["width"]), int(m["height"])
    M = _raster(m["polys"], W, H).astype(np.float64)
    def weights(n_px, n_cells):
        edges = np.linspace(0.0, n_px, n_cells + 1)
        lo = np.arange(n_px, dtype=np.float64)
        hi = lo + 1.0
        w = np.clip(np.minimum(hi[None, :], edges[1:, None]) - np.maximum(lo[None, :], edges[:-1, None]), 0, None)
        return w / w.sum(1, keepdims=True)
    Wr, Wc = weights(H, rows), weights(W, cols)
    return (Wr @ M @ Wc.T).reshape(-1).astype(np.float32)


def inside_cells(cov: np.ndarray, thr: float = 0.5) -> tuple[np.ndarray, bool]:
    inside = cov >= thr
    if inside.any():
        return inside, False
    return cov >= min(0.25, float(cov.max())), True


def mask_square(rec: dict, side: int = 448) -> np.ndarray:
    from PIL import Image
    m = rec["meta"]
    M = _raster(m["polys"], int(m["width"]), int(m["height"]))
    im = Image.fromarray((M * 255).astype(np.uint8))
    return np.asarray(im.resize((side, side), Image.BILINEAR), np.float32) / 255.0
