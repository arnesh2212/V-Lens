# SPDX-License-Identifier: Apache-2.0
"""Fit V-Read on COCO photographs with generic instructions, cache it, and stop.

This is the fit the original V-Lens artifact used: general photographs, each
paired with a generic instruction ("Describe this image.", "What do you
see?", ...), in the model's own chat format. Nothing in it comes from the
SimonSays set, so any record of the set can be read with it and none has to
be held out. Reading the set's answers with this map, beside the map fitted on
the set's own unscored records, says how much of V-Read is general and how
much is fitted to the task.

The fit goes through the same ``fit_all`` as every other one, on the same
blocks and budget; the corpus is different, so its cache entry is its own.

    python3 tools/coco_fit.py --repo Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


#: the COCO items the paper scores (POPE and the four-way object question);
#: their photographs are kept out of the fitting corpus
SCORED = "/workspace/temp/vlm/data_root/coco/coco_items.json"


def scored_images(path: str = SCORED) -> set:
    """File names of every COCO photograph some scored item is asked about."""
    import json
    if not os.path.exists(path):
        return set()
    return {os.path.basename(r["image"]) for r in json.load(open(path))["records"]}


def coco_records(n: int, seed: int = 0) -> list[dict]:
    """COCO photographs as fitting records, each with one generic instruction."""
    from vlens import corpus as C
    rng = random.Random(seed)
    held = scored_images()
    out = []
    for p in C.coco_images(n + len(held), seed=seed):
        if os.path.basename(str(p)) in held:
            continue
        if len(out) >= n:
            break
        out.append(dict(id=f"coco:{os.path.basename(str(p))}", image=str(p),
                        question=rng.choice(C.GENERIC_PROMPTS), system=None, cue=None,
                        family="coco", subcategory="generic", framing="none", meta={}))
    return out


def main():
    from vlens.lens.fitall import SHARED
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--layers", type=int, default=SHARED["layers"])
    ap.add_argument("--corpus", type=int, default=SHARED["corpus"])
    ap.add_argument("--heldout", type=int, default=SHARED["heldout"])
    ap.add_argument("--steps", type=int, default=SHARED["steps"])
    ap.add_argument("--square", type=int, default=SHARED["square"])
    ap.add_argument("--want", nargs="*", default=["v_read"])
    ap.add_argument("--lr", type=float, default=None, help="tuned-lens learning rate, when tuned is wanted")
    ap.add_argument("--wd", type=float, default=None)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.lens import fitall as FA

    recs = coco_records(a.corpus + 2 * a.heldout)
    say(f"{len(scored_images())} COCO photographs are scored elsewhere and kept out of the fit")
    if len(recs) < a.corpus + 2 * a.heldout:
        say(f"only {len(recs)} COCO photographs under {C.COCO_IMG}; need {a.corpus + 2 * a.heldout}")
        return
    m = vlens.load_vlm(a.repo)
    layers = FA.layer_grid(m, a.layers)
    say(f"{a.repo.split('/')[-1]}: fitting {', '.join(a.want)} on {len(recs)} COCO photographs at {layers}")
    t0 = time.time()
    maps, latent, fit_ids, meta = FA.fit_all(
        m, recs, C.load_image, layers=layers, corpus=a.corpus, heldout=a.heldout,
        square=a.square, steps=a.steps, tokens_per_step=SHARED["tokens_per_step"],
        jac_prompts=SHARED["jac_prompts"], latent_captions=SHARED["latent_captions"],
        fit_batch=SHARED["fit_batch"], rank=SHARED["rank"], want=tuple(a.want),
        tuned_lr=a.lr, tuned_wd=a.wd, log=say)
    say(f"fitted {sorted(maps)} on COCO ({(time.time() - t0) / 60:.1f} min)")
    say("DONE coco_fit")


if __name__ == "__main__":
    main()
