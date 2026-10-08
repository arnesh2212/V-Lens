# SPDX-License-Identifier: Apache-2.0
"""Rename a cached fit to the key the current code computes.

The cache key stopped depending on which instruments a caller asked for, so an
entry written before that is sitting under a name nothing will look up again.
Its contents are exactly what the new key describes and a sixteen-block fit is
hours of card time, so it is renamed rather than thrown away, and the
description of the fit is written beside it so that no future change of key
needs this tool at all. The parameters have to be given for an entry written
before that description existed; they are the ones the experiment that wrote
it was run with.

    python3 tools/rekey_maps.py --repo Qwen/Qwen3-VL-8B-Instruct --apply
"""
from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from vlens.data import simonsays as SS  # noqa: E402
from vlens.lens import fitall as FA  # noqa: E402

S = FA.SHARED
GRIDS = dict(lr_grid=(1.0, 0.1, 0.01),
             lam_grid=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
             gain_grid=(1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0),
             lr_probe_layers=4,
             temp_grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--corpus", type=int, default=S["corpus"])
    ap.add_argument("--heldout", type=int, default=S["heldout"],
                    help="the held-out size the run used; patch_lenses uses 64, not the "
                         "shared 120, and guessing it wrong writes a name nothing asks for")
    ap.add_argument("--steps", type=int, default=S["steps"])
    ap.add_argument("--cap", type=int, default=S["cap"])
    ap.add_argument("--file", default=None, help="the entry to rename; the newest by default")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    spare = FA.fitting_pool(SS.load(require_image=True), a.cap)

    def spec_for(layers):
        # the blocks come out of the file rather than being recomputed here:
        # the grid starts past the deepstack, which only the model knows
        return FA._spec(a.repo, spare, layers, a.corpus, a.heldout, S["square"], a.steps,
                        S["tokens_per_step"], S["rank"], S["jac_prompts"], S["latent_captions"],
                        S["fit_batch"], GRIDS["lr_grid"], GRIDS["lam_grid"], GRIDS["gain_grid"],
                        GRIDS["lr_probe_layers"], GRIDS["temp_grid"])

    files = [f for f in os.listdir(FA.CACHE) if f.endswith(".pt")]
    if a.file:
        files = [a.file] if a.file in files else []
    elif files:
        files = [max(files, key=lambda f: os.path.getmtime(os.path.join(FA.CACHE, f)))]
    for f in files:
        p = os.path.join(FA.CACHE, f)
        d = torch.load(p, map_location="cpu")
        got = sorted({int(l) for v in d["maps"].values() for l in v})
        spec = spec_for(got)
        key = FA._key(a.repo, spare, got, a.corpus, a.heldout, S["square"], a.steps,
                      S["tokens_per_step"], S["rank"], S["jac_prompts"], S["latent_captions"],
                      S["fit_batch"], GRIDS["lr_grid"], GRIDS["lam_grid"], GRIDS["gain_grid"],
                      GRIDS["lr_probe_layers"], GRIDS["temp_grid"])
        print(f"  {f}: blocks {got}, holds {sorted(d['maps'])}, corpus {a.corpus}, "
              f"{a.steps} steps -> {key}.pt")
        if f == f"{key}.pt":
            if a.apply and not os.path.exists(p + ".spec"):
                open(p + ".spec", "w").write(spec)
                print("   already under the current key; wrote its spec beside it")
            else:
                print("   already under the current key")
            continue
        q = os.path.join(FA.CACHE, f"{key}.pt")
        if a.apply:
            os.rename(p, q)
            open(q + ".spec", "w").write(spec)
            print(f"   renamed to {key}.pt, with its spec beside it")
        else:
            print(f"   would rename to {key}.pt")


if __name__ == "__main__":
    main()
