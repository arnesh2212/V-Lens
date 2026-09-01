# SPDX-License-Identifier: Apache-2.0
"""Where in the network is X readable?  A trained probe at every layer.

This is the yardstick every lens is held to.  For a labelled set of images
and a fixed prompt, cache the image-token residuals at every layer, fit a
linear probe per layer on the CPU, and report the curve with its measured
floors.  A zero-shot lens that cannot match this curve on a task has not shown
it reads that task; one that matches it without labels has.

Tasks (``--task``):
  concept   COCO: which of the K most frequent single-word objects is the
            item's primary object (K-way, accuracy, majority floor)
  count     CLEVR: how many objects are in the scene (3..10, accuracy)
  colour    CLEVR: the colour of the largest object (8-way, accuracy)
  harmful   MM-SafetyBench: harmful versus benign scenario, image only, one
            fixed prompt (2-way, AUROC, shuffled floor)

Run ``--n 40 --layers 4 8`` first.  Everything is written under results/probes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--task", default="concept", choices=["concept", "count", "colour", "harmful"])
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--k", type=int, default=10, help="classes kept for the concept task")
ap.add_argument("--layers", type=int, nargs="*", default=None, help="default: every layer")
ap.add_argument("--every", type=int, default=1, help="with no --layers, take every k-th layer")
ap.add_argument("--prompt", default="Describe this image.")
ap.add_argument("--pool", default="mean", choices=["mean", "tokens"])
ap.add_argument("--jobs", type=int, default=12)
ap.add_argument("--out", default="/workspace/vlm/results/probes")
ap.add_argument("--tag", default=None)
ap.add_argument("--reuse-cache", action="store_true")
a = ap.parse_args()

import jvlm  # noqa: E402
from jvlm import cache, probes  # noqa: E402

tag = a.tag or a.repo.split("/")[-1].replace("-Instruct", "").lower()
os.makedirs(a.out, exist_ok=True)
cache_path = f"{a.out}/{tag}_{a.task}_cache.npz"

# ---------------------------------------------------------------- the items
def items_for(task: str, n: int) -> tuple[list[dict], str]:
    """A list of dicts with an image ``path`` and a ``label``; and the label's meaning."""
    if task == "concept":
        from jvlm import cocodata
        pool = cocodata.concept_items(n * 6, min_area=0.05, seed=7)
        counts = {}
        for it in pool:
            counts[it["concept"]] = counts.get(it["concept"], 0) + 1
        keep = sorted(counts, key=counts.get, reverse=True)[: a.k]
        out = [dict(path=it["path"], label=it["concept"], image_id=it["image_id"])
               for it in pool if it["concept"] in keep]
        # balance the classes so the majority floor is honest
        per = n // len(keep)
        bal = []
        for c in keep:
            bal += [it for it in out if it["label"] == c][:per]
        return bal, f"primary object, {len(keep)} classes"
    from jvlm import clevrdata as cd
    if task in ("count", "colour"):
        out = []
        for s in cd.scenes():
            if not cd.have(s):
                continue
            objs = s["objects"]
            if task == "count":
                lab = len(objs)
            else:
                big = max(objs, key=lambda o: {"large": 1, "small": 0}[o["size"]])
                lab = big["color"]
            out.append(dict(path=cd.image_path(s), label=lab, image_index=s["image_index"]))
            if len(out) >= n:
                break
        return out, ("object count" if task == "count" else "colour of the largest object")
    from jvlm import safedata
    rows = safedata.mm_safetybench(n, "SD")            # the clean image variant
    out = [dict(path=r["image_path"], label=bool(r["harmful"]), scenario=r["scenario"])
           for r in rows]
    return out, "harmful versus benign scenario"


items, meaning = items_for(a.task, a.n)
labels = [it["label"] for it in items]
print(f"{len(items)} items; {meaning}; classes: {sorted(set(map(str, labels)))[:12]}", flush=True)

# ---------------------------------------------------------------- the cache
if a.reuse_cache and os.path.exists(cache_path):
    acts, side = cache.load(cache_path)
    print(f"reusing {cache_path}", flush=True)
else:
    m = jvlm.load_vlm(a.repo)
    L = a.layers if a.layers else list(range(0, m.n_layers, a.every))
    cache.collect(m, items, L, a.prompt, cache_path, pool=a.pool)
    del m
    import torch
    torch.cuda.empty_cache()
    acts, side = cache.load(cache_path)

y = cache.labels(side, "label")
k = len(np.unique(y))
print(f"{len(y)} items after caching, {k} classes, layers {sorted(acts)}", flush=True)

# ---------------------------------------------------------------- the probes
t0 = time.time()
res = probes.probe_layers(acts, y, jobs=a.jobs)
print(f"\n{a.repo}  task={a.task}  ({meaning})\n")
print(probes.table(res))
print(f"\nprobes fitted in {time.time() - t0:.0f}s on the CPU")

stats = dict(repo=a.repo, task=a.task, meaning=meaning, n=int(len(y)), classes=int(k),
             prompt=a.prompt, pool=a.pool, cache=cache_path,
             layers=[r.layer for r in res],
             results=[r.__dict__ for r in res])
with open(f"{a.out}/{tag}_{a.task}_stats.json", "w") as f:
    json.dump(stats, f, indent=1)
print(f"wrote {a.out}/{tag}_{a.task}_stats.json")
