# SPDX-License-Identifier: Apache-2.0
"""Fit the shared transports once, cache them, and stop.

Six experiments read through the same sixteen blocks of the same model, and
the fit costs a vocabulary softmax per block per token, which caching cannot
remove: eight hours on an 8B. Whichever of the six runs first pays for it and
the rest load it, which means the bill lands wherever the queue happens to put
that one and nothing else can start until it clears.

Running it on its own, on a lane that takes what memory is spare, pays the
bill once and off the critical path. Every instrument is fitted, not only the
ones a particular experiment asks for, so the entry is a superset of what any
of them will look up.

    python3 tools/warm_fit.py --repo Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from vlens.lens.fitall import SHARED
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--cap", type=int, default=SHARED["cap"])
    ap.add_argument("--layers", type=int, default=SHARED["layers"])
    ap.add_argument("--corpus", type=int, default=SHARED["corpus"])
    ap.add_argument("--heldout", type=int, default=SHARED["heldout"])
    ap.add_argument("--steps", type=int, default=SHARED["steps"])
    ap.add_argument("--square", type=int, default=SHARED["square"])
    ap.add_argument("--results", default="results/simonsays")
    ap.add_argument("--need-free", type=int, default=-1,
                    help="MiB that must be free on the card before starting; "
                         "the default is the weights on disk with room to work")
    ap.add_argument("--wait-minutes", type=float, default=120.0)
    ap.add_argument("--lr", type=float, default=None,
                    help="the tuned lens's learning rate, where a search on this same "
                         "corpus has already chosen it and repeating it would cost hours")
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--want", nargs="*", default=None,
                    help="instruments to fit; every shared one when not given")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays as SS
    from vlens.lens import fitall as FA, room

    need = a.need_free
    if need < 0:
        w = room.weights_mib(a.repo)
        need = int(w * 1.3) if w else 0
        say(f"  {a.repo.split('/')[-1]} is {w} MiB on disk; waiting for {need} MiB on the card")
    room.wait_for_free(need, minutes=a.wait_minutes, log=say)
    m = vlens.load_vlm(a.repo)
    layers = FA.layer_grid(m, a.layers)
    spare = FA.fitting_pool(SS.load(require_image=True), a.cap)
    say(f"{a.repo.split('/')[-1]}: fitting every instrument at {layers}")
    say(f"corpus drawn from {len(spare)} records the behaviour pass never scored")
    t0 = time.time()
    maps, latent, fit_ids, meta = FA.fit_all(
        m, spare, C.load_image, layers=layers, corpus=a.corpus, heldout=a.heldout,
        square=a.square, steps=a.steps, tokens_per_step=SHARED["tokens_per_step"],
        jac_prompts=SHARED["jac_prompts"], latent_captions=SHARED["latent_captions"],
        fit_batch=SHARED["fit_batch"], rank=SHARED["rank"],
        want=tuple(a.want) if a.want else FA.SHARED_WANT, tuned_lr=a.lr, tuned_wd=a.wd,
        results=a.results, log=say)
    say(f"\nfitted {sorted(maps)}" + (" and the latent store" if latent is not None else "")
        + f", {len(fit_ids)} records now off limits for scoring "
        f"({(time.time() - t0) / 60:.1f} min)")
    for k, v in sorted(meta.items()):
        if isinstance(v, dict) and v.get("seconds") is not None:
            say(f"  {k:<10}{v['seconds'] / 60:>7.1f} min"
                + (f"  {v.get('tokens') or v.get('tokens_spent')} tokens"
                   if (v.get("tokens") or v.get("tokens_spent")) else ""))
    # the point of the run is the entry on disk, so say whether there is one
    key = [f for f in os.listdir(FA.CACHE) if f.endswith(".pt")]
    say(f"the cache now holds {len(key)} fit(s): {', '.join(sorted(key))}")
    say("DONE warm_fit")


if __name__ == "__main__":
    main()
