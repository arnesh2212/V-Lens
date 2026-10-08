# SPDX-License-Identifier: Apache-2.0
"""Freeze the COCO families and fetch only their pictures.

Run where there is disk (this machine, not the box): it reads the full val2017
annotation file, writes ``coco_items.json`` with every record and the human
masks of the present objects, and downloads just the photographs those records
use, by URL, into ``<out>/val2017``.  The folder that results is what goes to
the box (about 80 MB for 500 pictures); the box never sees the annotation zip.

    python3 tools/coco_subset.py --instances ~/coco_dl/annotations/instances_val2017.json \
        --out data_coco --n 500 --seed 0
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

URL = "http://images.cocodataset.org/val2017/{}"


def fetch(rel: str, out: str) -> int:
    p = os.path.join(out, rel)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return os.path.getsize(p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(URL.format(os.path.basename(rel)), timeout=60) as r:
                data = r.read()
            with open(p + ".part", "wb") as fh:
                fh.write(data)
            os.replace(p + ".part", p)
            return len(data)
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not fetch {rel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-area", type=float, default=0.03)
    ap.add_argument("--max-area", type=float, default=0.40)
    ap.add_argument("--per-cat", type=int, default=10)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    from vlens.data.coco import build_items
    recs = build_items(a.instances, a.n, seed=a.seed, min_area=a.min_area,
                       max_area=a.max_area, per_cat=a.per_cat)
    os.makedirs(a.out, exist_ok=True)
    fams = Counter(r["family"] for r in recs)
    imgs = sorted({r["image"] for r in recs})
    cats = Counter(r["meta"]["target"] for r in recs if r["id"].endswith("_yes"))
    meta = dict(source="COCO val2017 instances (Lin et al. 2014)", seed=a.seed, n_images=len(imgs),
                min_area=a.min_area, max_area=a.max_area, per_cat=a.per_cat,
                families=dict(fams), n_categories=len(cats),
                protocol=dict(coco_pope="POPE adversarial (Li et al. 2023) on val2017 ground truth",
                              coco_mcq="four named objects, one present, three co-occurring absent foils"))
    json.dump(dict(meta=meta, records=recs), open(os.path.join(a.out, "coco_items.json"), "w"))
    print(f"{len(recs)} records on {len(imgs)} pictures, {len(cats)} categories; {dict(fams)}")
    print("  most used:", cats.most_common(8))
    if a.no_images:
        return
    t0, tot = time.time(), 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, n in enumerate(ex.map(lambda r: fetch(r, a.out), imgs)):
            tot += n
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(imgs)} pictures, {tot / 2 ** 20:.0f} MB ({time.time() - t0:.0f} s)")
    print(f"fetched {len(imgs)} pictures, {tot / 2 ** 20:.0f} MB -> {a.out}/val2017")


if __name__ == "__main__":
    main()
