# SPDX-License-Identifier: Apache-2.0
"""Do the experiments that share a fit actually ask for the same one?

The cache key is a digest of everything the fit depends on, and any argument
that differs by one makes a different key and a second eight-hour fit that
nobody notices until the bill arrives. One of them differed by one this
morning: the patch experiment holds out sixty-four prompts where the shared
settings hold out a hundred and twenty.

This asks each experiment for the key its own defaults produce, without
loading a model, and says which of them agree.

    python3 tools/check_fit_keys.py --repo Qwen/Qwen3-VL-8B-Instruct \
        --n-layers 36 --deepstack 3
"""
from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from vlens.data import simonsays as SS  # noqa: E402
from vlens.lens import fitall as FA  # noqa: E402

S = FA.SHARED
GRIDS = ((1.0, 0.1, 0.01),
         (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
         (1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0), 4,
         (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0))

#: what each experiment passes, as its own arguments have it
WANTS = {
    "warm_fit":      dict(layers="grid", corpus=S["corpus"], heldout=S["heldout"],
                          steps=S["steps"], jac=S["jac_prompts"], caps=S["latent_captions"]),
    "read_image":    dict(layers="grid", corpus=S["corpus"], heldout=S["heldout"],
                          steps=S["steps"], jac=S["jac_prompts"], caps=S["latent_captions"]),
    "lens_bench":    dict(layers="grid", corpus=S["corpus"], heldout=S["heldout"],
                          steps=S["steps"], jac=S["jac_prompts"], caps=S["latent_captions"]),
    "hallucination": dict(layers="grid", corpus=S["corpus"], heldout=S["heldout"],
                          steps=S["steps"], jac=S["jac_prompts"], caps=S["latent_captions"]),
    "visual_tokens": dict(layers="grid", corpus=S["corpus"], heldout=S["heldout"],
                          steps=S["steps"], jac=S["jac_prompts"], caps=S["latent_captions"]),
    "prefill":       dict(layers="grid", corpus=S["corpus"], heldout=S["heldout"],
                          steps=S["steps"], jac=S["jac_prompts"], caps=S["latent_captions"]),
    "carrier":       dict(layers="grid", corpus=S["corpus"], heldout=S["heldout"],
                          steps=S["steps"], jac=S["jac_prompts"], caps=S["latent_captions"]),
    # the patch experiment fits one block and holds out fewer; it is meant to
    # differ, and is listed so the difference is visible rather than assumed
    "patch_lenses":  dict(layers="one", corpus=None, heldout=64, steps=None, jac=12, caps=5000),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--deepstack", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.5, help="the patch experiment's depth")
    ap.add_argument("--cap", type=int, default=S["cap"])
    ap.add_argument("--patch-corpus", type=int, default=900,
                    help="what the queue line for this model passes to patch_lenses")
    ap.add_argument("--patch-steps", type=int, default=250)
    a = ap.parse_args()

    spare = FA.fitting_pool(SS.load(require_image=True), a.cap)
    lo, hi = max(a.deepstack, 0), a.n_layers - 2
    grid = sorted({int(round(x)) for x in np.linspace(lo, hi, S["layers"])})
    one = [max(int(round(a.frac * (a.n_layers - 1))), 1)]

    seen = {}
    for name, w in WANTS.items():
        w = dict(w)
        if w["corpus"] is None:
            w["corpus"], w["steps"] = a.patch_corpus, a.patch_steps
        layers = grid if w["layers"] == "grid" else one
        k = FA._key(a.repo, spare, layers, w["corpus"], w["heldout"], S["square"], w["steps"],
                    S["tokens_per_step"], S["rank"], w["jac"], w["caps"], S["fit_batch"], *GRIDS)
        seen.setdefault(k, []).append(name)
        print(f"  {name:<16}{k}  blocks {len(layers):<3} corpus {w['corpus']:<5}"
              f"heldout {w['heldout']:<5}steps {w['steps']}")
    print()
    for k, who in seen.items():
        print(f"  {k}  {len(who)} experiment(s): {', '.join(who)}")
    shared = max(seen.values(), key=len)
    if len(shared) < len(WANTS) - 1:
        print("\n  the multi-block experiments do not all agree; each disagreement is "
              "a second fit")
        return 1
    print(f"\n  {len(shared)} experiments share one fit, and the patch experiment has its own")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
