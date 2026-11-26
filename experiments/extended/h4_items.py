from __future__ import annotations

import argparse
import colorsys
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np

SRC = os.path.join(ROOT, "data", "coco_subset", "coco_items.json")
OUT = os.path.join(ROOT, "results", "extended", "h4")
PLURAL = {"person": "people", "bus": "buses", "couch": "couches", "sandwich": "sandwiches", "bench": "benches",
          "knife": "knives", "mouse": "mice", "sheep": "sheep", "skis": "pairs of skis", "scissors": "pairs of scissors",
          "toothbrush": "toothbrushes", "wine glass": "wine glasses", "dining table": "dining tables",
          "teddy bear": "teddy bears", "hair drier": "hair dryers", "tv": "televisions", "television": "televisions",
          "bear": "bears", "glass": "glasses", "fish": "fish"}
NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
CONTRAST = {"red": ["green", "blue"], "orange": ["blue", "purple"], "yellow": ["blue", "purple"],
            "green": ["red", "purple"], "blue": ["red", "yellow"], "purple": ["green", "yellow"],
            "pink": ["green", "blue"], "brown": ["blue", "green"], "black": ["white", "yellow"],
            "white": ["black", "red"], "gray": ["red", "yellow"]}


def colour_names(rgb):
    out = []
    for r, g, b in rgb / 255.0:
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h *= 360
        if v < 0.2:
            out.append("black")
        elif s < 0.15:
            out.append("white" if v > 0.8 else "gray")
        elif 10 <= h < 40 and v < 0.6:
            out.append("brown")
        elif h < 15 or h >= 345:
            out.append("red")
        elif h < 40:
            out.append("orange")
        elif h < 70:
            out.append("yellow")
        elif h < 170:
            out.append("green")
        elif h < 260:
            out.append("blue")
        elif h < 300:
            out.append("purple")
        else:
            out.append("pink")
    return out


INSTANCES = os.path.join(ROOT, "data", "coco_subset", "annotations", "instances_val2017.json")


def build_rel(root, n, seed, nice, min_area=0.01, margin=0.02, per_photo=2):
    from collections import defaultdict
    inst = json.load(open(INSTANCES))
    local = set(os.listdir(os.path.join(root, "val2017")))
    imgs = {im["id"]: im for im in inst["images"] if im["file_name"] in local}
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    by = defaultdict(lambda: defaultdict(list))
    for an in inst["annotations"]:
        if an["image_id"] in imgs:
            by[an["image_id"]][an["category_id"]].append(an)
    rng = np.random.default_rng(seed + 1)
    cands = []
    for iid in sorted(by):
        im = imgs[iid]
        W, H = im["width"], im["height"]
        single = [v[0] for c, v in by[iid].items()
                  if len(v) == 1 and not v[0]["iscrowd"] and isinstance(v[0]["segmentation"], list)
                  and v[0]["area"] >= min_area * W * H]
        pairs = []
        for i, x in enumerate(single):
            for y in single[i + 1:]:
                for rel, ax, ext in (("left", 0, W), ("above", 1, H)):
                    lo, hi = (x, y) if x["bbox"][ax] <= y["bbox"][ax] else (y, x)
                    gap = hi["bbox"][ax] - (lo["bbox"][ax] + lo["bbox"][ax + 2])
                    if gap >= margin * ext:
                        pairs.append((rel, lo, hi, gap / ext))
        if pairs:
            cands.append((iid, [pairs[i] for i in rng.permutation(len(pairs))]))
    rng.shuffle(cands)
    out, k = [], 0
    for rnd in range(per_photo):
        for iid, pairs in cands:
            if len(out) >= n or rnd >= len(pairs):
                continue
            rel, lo, hi, gap = pairs[rnd]
            im = imgs[iid]
            yes = (k % 2 == 0)
            k += 1
            a_, b_ = (lo, hi) if yes else (hi, lo)
            na, nb = nice(cats[a_["category_id"]]), nice(cats[b_["category_id"]])
            q = (f"Is the {na} to the left of the {nb}?" if rel == "left" else f"Is the {na} above the {nb}?")
            W, H = im["width"], im["height"]
            meta = dict(image_id=iid, width=W, height=H, target=na, target_name=na, supercategory=None,
                        area=round((a_["area"] + b_["area"]) / (W * H), 4), n_instances=1,
                        boxes=[a_["bbox"], b_["bbox"]], polys=a_["segmentation"] + b_["segmentation"],
                        polys_a=a_["segmentation"], polys_b=b_["segmentation"], present=[na, nb],
                        h4=dict(family="coco_rel", relation=rel, a=na, b=nb, gap=round(gap, 3),
                                answer="yes" if yes else "no"))
            out.append(dict(dataset="coco", variant="val2017", condition="none", framing="none",
                            image=os.path.join(root, "val2017", im["file_name"]), cue=None, system=None, meta=meta,
                            id=f"h4_coco_rel_{iid}_{rel}_{'yes' if yes else 'no'}", family="coco_rel",
                            category="coco_rel", subcategory="present", question=q + " Answer yes or no.",
                            choices=None, expected="yes" if yes else "no", answer_index=None, kind="yesno"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rel", type=int, default=300, help="two-object relation questions (needs COCO instances)")
    ap.add_argument("--rel-min-area", type=float, default=0.006, help="each object's share of the frame")
    ap.add_argument("--rel-per-photo", type=int, default=4)
    a = ap.parse_args()
    from PIL import Image
    from vlens.data import coco

    src = json.load(open(SRC))
    root = os.path.dirname(SRC)
    base = [r for r in src["records"] if r["family"] == "coco_pope" and r["subcategory"] == "present"
            and (r.get("meta") or {}).get("polys")]
    rng = np.random.default_rng(a.seed)
    base = [base[i] for i in rng.permutation(len(base))]
    nice = lambda c: coco.NICE.get(c, c)

    def rec(r, fam, k, q, exp, **extra):
        m = dict(r["meta"], h4=dict(family=fam, **extra))
        m.pop("foils", None); m.pop("adversarial", None)
        return dict(dataset="coco", variant="val2017", condition="none", framing="none",
                    image=os.path.join(root, r["image"]), cue=None, system=None, meta=m,
                    id=f"h4_{fam}_{m['image_id']}_{k}", family=fam, category=fam, subcategory="present",
                    question=q + " Answer yes or no.", choices=None, expected=exp, answer_index=None, kind="yesno")

    attr, count, pos = [], [], []
    for r in base:
        m = r["meta"]
        obj = nice(m["target"])
        W, H = int(m["width"]), int(m["height"])
        boxes = m.get("boxes") or []
        if len(attr) < a.per_family and m["target"] != "person":
            M = coco._raster(m["polys"], W, H) > 0
            if M.sum() >= 400:
                px = np.asarray(Image.open(os.path.join(root, r["image"])).convert("RGB"))[M]
                if len(px) > 4000:
                    px = px[rng.choice(len(px), 4000, replace=False)]
                cn = Counter(colour_names(px.astype(np.float32)))
                tot = sum(cn.values())
                top, share = cn.most_common(1)[0]
                if share / tot >= 0.40:
                    neg = [c for c in CONTRAST[top] if cn.get(c, 0) / tot < 0.03]
                    if neg:
                        art = "an" if obj[0] in "aeiou" else "a"
                        attr.append(rec(r, "coco_attr", "yes", f"Is the {obj} {top}?", "yes",
                                        colour=top, share=round(share / tot, 3)))
                        attr.append(rec(r, "coco_attr", "no", f"Is the {obj} {neg[0]}?", "no",
                                        colour=neg[0], dominant=top))
        n = int(m.get("n_instances") or 0)
        if len(count) < a.per_family and 1 <= n <= 4 and len(boxes) == n and \
                all(b[2] * b[3] >= 0.005 * W * H for b in boxes):
            pl = PLURAL.get(obj, obj + "s")
            q = (lambda N: f"Is there exactly one {obj} in the image?" if N == 1
                 else f"Are there exactly {NUM[N]} {pl} in the image?")
            alt = n + 1 if (n == 1 or rng.random() < 0.5) else n - 1
            count.append(rec(r, "coco_count", "yes", q(n), "yes", n=n, asked=n))
            count.append(rec(r, "coco_count", "no", q(alt), "no", n=n, asked=alt))
        if len(pos) < a.per_family and n == 1 and boxes:
            cx = (boxes[0][0] + boxes[0][2] / 2) / W
            cy = (boxes[0][1] + boxes[0][3] / 2) / H
            side = "left" if cx < 0.35 else "right" if cx > 0.65 else None
            if side:
                ask = side if rng.random() < 0.5 else ("right" if side == "left" else "left")
                pos.append(rec(r, "coco_pos", ask, f"Is the {obj} on the {ask} side of the picture?",
                               "yes" if ask == side else "no", side=side, cx=round(cx, 3)))
            else:
                half = "top" if cy < 0.35 else "bottom" if cy > 0.65 else None
                if half:
                    ask = half if rng.random() < 0.5 else ("bottom" if half == "top" else "top")
                    pos.append(rec(r, "coco_pos", ask, f"Is the {obj} in the {ask} half of the picture?",
                                   "yes" if ask == half else "no", side=half, cy=round(cy, 3)))
    rel = build_rel(root, a.rel, a.seed, nice, min_area=a.rel_min_area, per_photo=a.rel_per_photo) if a.rel and os.path.exists(INSTANCES) else []
    recs = attr + count + pos + rel
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, "coco_items.json.tmp")
    json.dump(dict(meta=dict(source=SRC, seed=a.seed, families=["coco_attr", "coco_count", "coco_pos", "coco_rel"],
                             note="graded against the target's polygons (coco_rel: both objects')"),
                   records=recs), open(tmp, "w"))
    os.replace(tmp, os.path.join(OUT, "coco_items.json"))
    for f in ("coco_attr", "coco_count", "coco_pos", "coco_rel"):
        rs = [r for r in recs if r["family"] == f]
        print(f"{f}: {len(rs)} questions on {len({r['meta']['image_id'] for r in rs})} photographs, "
              f"{sum(r['expected'] == 'yes' for r in rs)} yes", flush=True)


if __name__ == "__main__":
    main()
