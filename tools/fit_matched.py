# SPDX-License-Identifier: Apache-2.0
"""A tuned lens given V-Read's fitting budget, on the same corpus, blocks and machine.

The paper's tuned lens follows its paper's schedule (250 steps of 2**18 tokens),
which is what makes V-Read's fit look cheap. Here V-Read and tuned lenses with
smaller token budgets are fitted one after another on one GPU, each timed, so
AUROC can be compared at equal wall-clock. Maps are cached by
``vlens.lens.fitall`` under VLENS_MAP_CACHE; scoring runs elsewhere from the
stored answer-spot states (tools/layer_readouts.py read --maps <cache>).

    python3 tools/fit_matched.py --tokens 12 14
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--tokens", type=int, nargs="+", default=[12, 14], help="log2 tokens per step")
ap.add_argument("--steps", type=int, default=250)
ap.add_argument("--out", default="results/fits_matched")
ap.add_argument("--want", nargs="+", default=["v_read", "tuned"])
a = ap.parse_args()

import vlens  # noqa: E402
from vlens import corpus as C  # noqa: E402
from vlens.data import simonsays as SS  # noqa: E402
from vlens.lens import fitall as FA  # noqa: E402

m = vlens.load_vlm(a.repo)
name = a.repo.split("/")[-1]
S = FA.SHARED
layers = sorted(set(int(round(i * (m.n_layers - 2) / (S["layers"] - 1))) for i in range(S["layers"])))
ref = json.load(open(f"results/fits/{name}__fitmeta.json"))[0]["spec"][2]
layers = list(ref)
spare = FA.fitting_pool(SS.load(require_image=True), S["cap"])
os.makedirs(a.out, exist_ok=True)
log = lambda *x: print(*x, flush=True)  # noqa: E731
rows = []
for tp in a.tokens:
    t0 = time.time()
    maps, lat, ids, meta = FA.fit_all(m, spare, C.load_image, layers=layers, corpus=S["corpus"],
                                      heldout=S["heldout"], square=S["square"], steps=a.steps,
                                      tokens_per_step=2 ** tp, rank=S["rank"], jac_prompts=S["jac_prompts"],
                                      fit_batch=4, want=tuple(a.want), log=log)
    rec = dict(tokens_per_step=2 ** tp, steps=a.steps, wall=round(time.time() - t0),
               v_read=dict(seconds=meta.get("v_read", {}).get("seconds")),
               tuned=dict(seconds=meta["tuned"].get("seconds"), heldout_kl=meta["tuned"].get("heldout_kl"),
                          tokens_spent=meta["tuned"].get("tokens_spent")),
               layers=layers)
    rows.append(rec)
    log("MATCHED", json.dumps(rec))
    json.dump(rows, open(f"{a.out}/{name}__matched.json", "w"), indent=1)
log("DONE")
