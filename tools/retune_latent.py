# SPDX-License-Identifier: Apache-2.0
"""Re-choose the LatentLens readout on a fit that is already cached.

The temperature and the floor that turn stored similarities into a
distribution are the baseline's free parameters, and both belong on the
held-out split rather than at values we picked. Entries fitted before the
floor was searched carry only a temperature, and the floor then falls back to
-1, where almost the whole vocabulary sits and the divergence is mostly a
measure of how much vocabulary the caption corpus reached.

This re-runs that choice against the same held-out prompts and writes it back
into the entry. It touches nothing else: the store, the transports and the
identifiers a fit is not allowed to be scored on all stay as they were.

    python3 tools/retune_latent.py --repo Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from vlens.lens.fitall import SHARED
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--cap", type=int, default=SHARED["cap"])
    ap.add_argument("--layers", type=int, default=SHARED["layers"])
    ap.add_argument("--corpus", type=int, default=SHARED["corpus"])
    ap.add_argument("--heldout", type=int, default=SHARED["heldout"])
    ap.add_argument("--square", type=int, default=SHARED["square"])
    ap.add_argument("--steps", type=int, default=SHARED["steps"])
    ap.add_argument("--force", action="store_true", help="redo it even if a floor is stored")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays as SS
    from vlens.lens import baselines as B, fitall as FA

    spare = FA.fitting_pool(SS.load(require_image=True), a.cap)
    m = vlens.load_vlm(a.repo)
    layers = FA.layer_grid(m, a.layers)
    spec = FA._spec(getattr(m, "name", a.repo), spare, layers, a.corpus, a.heldout,
                    a.square, a.steps, SHARED["tokens_per_step"], SHARED["rank"],
                    SHARED["jac_prompts"], SHARED["latent_captions"], SHARED["fit_batch"],
                    (1.0, 0.1, 0.01),
                    (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
                    (1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0), 4,
                    (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0))
    import hashlib
    path = os.path.join(FA.CACHE, hashlib.sha1(spec.encode()).hexdigest()[:16] + ".pt")
    got = FA._load(path)
    if got is None:
        say(f"nothing cached at {path}; nothing to retune"); return
    maps, latent, fit_ids, meta = got
    if latent is None:
        say("that fit has no latent store"); return
    if "floor" in latent.meta and not a.force:
        say(f"already chosen: temp {latent.meta['temp']}, floor {latent.meta['floor']}"); return

    items = B.corpus_items(spare, C.load_image, n=a.corpus + 2 * a.heldout,
                           square=a.square, seed=0)
    dev = items[a.corpus: a.corpus + a.heldout]
    held = B.CachedStates(B.StateStream(m, dev, batch=SHARED["fit_batch"], seed=3),
                          layers, log=say)
    before = latent.meta.get("temp", 0.05)
    t, f = B.pick_temp(m, latent, held, layers, log=say)
    latent.meta["temp"], latent.meta["floor"] = t, f
    latent.meta["coverage"] = {int(l): int(tt.unique().numel())
                               for l, tt in latent.toks.items()}
    cov = next(iter(latent.meta["coverage"].values()))
    say(f"  the store reaches {cov} of {latent.vocab} vocabulary tokens "
        f"({cov / latent.vocab:.2%})")
    say(f"  temperature {before} -> {t}, floor -1.0 -> {f}")
    FA._save(path, maps, latent, fit_ids, meta, spec)
    say(f"written back to {path}")
    say("DONE retune_latent")


if __name__ == "__main__":
    main()
