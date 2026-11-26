from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/coco_subset")
    ap.add_argument("--out", default="results/extended/b2ext")
    ap.add_argument("--min-area", type=float, default=0.03)
    ap.add_argument("--max-area", type=float, default=0.4)
    a = ap.parse_args()
    from vlens.data import coco as CO

    src = os.path.abspath(a.src)
    items = json.load(open(f"{src}/coco_items.json"))
    recs = items["records"]
    for r in recs:
        r["image"] = os.path.join(src, r["image"])
    inst = json.load(open(f"{src}/annotations/instances_val2017.json"))
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    supercat = {c["name"]: c["supercategory"] for c in inst["categories"]}
    by_img = defaultdict(list)
    for an in inst["annotations"]:
        by_img[an["image_id"]].append(an)
    imgs = {im["id"]: im for im in inst["images"]}
    base_by_iid = {}
    for r in recs:
        if r["family"] == "coco_pope" and r["subcategory"] == "present":
            base_by_iid[r["meta"]["image_id"]] = r
    extra = []
    for iid, r0 in sorted(base_by_iid.items()):
        im = imgs[iid]
        W, H = im["width"], im["height"]
        present = sorted({cats[an["category_id"]] for an in by_img[iid]})
        for c in present:
            if c == r0["meta"]["target"]:
                continue
            anns = [an for an in by_img[iid] if cats[an["category_id"]] == c]
            if any(an.get("iscrowd") for an in anns):
                continue
            polys = [p for an in anns for p in an["segmentation"]]
            share = CO._union_share(polys, W, H)
            if not (a.min_area <= share <= a.max_area):
                continue
            meta = dict(image_id=iid, width=W, height=H, target=c, target_name=CO._nice(c),
                        supercategory=supercat[c], area=round(share, 4), n_instances=len(anns),
                        present=present, boxes=[[round(v, 1) for v in an["bbox"]] for an in anns],
                        polys=[[round(v, 1) for v in p] for p in polys], b2_extra=True)
            extra.append(dict(dataset="coco", variant="val2017", condition="none", framing="none",
                              image=r0["image"], cue=None, system=None, meta=meta,
                              id=f"coco_{iid}_{c.replace(' ', '_')}_yes", family="coco_pope",
                              category="coco_pope", subcategory="present", question=CO.pope_question(c),
                              choices=None, expected="yes", answer_index=None, kind="yesno"))
    os.makedirs(a.out, exist_ok=True)
    meta = dict(items.get("meta", {}), b2ext=dict(
        note="original subset + every other eligible present category per photograph",
        n_original_present=len(base_by_iid), n_extra_present=len(extra),
        min_area=a.min_area, max_area=a.max_area))
    json.dump(dict(meta=meta, records=recs + extra), open(f"{a.out}/coco_items.json", "w"))
    print(f"{len(base_by_iid)} original present + {len(extra)} extra present records "
          f"-> {a.out}/coco_items.json ({len(recs) + len(extra)} records)")


if __name__ == "__main__":
    main()
