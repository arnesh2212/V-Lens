# SPDX-License-Identifier: Apache-2.0
"""Rebuild a saved read map from its stored fit corpus, with the scale correction.

No model is loaded: the answer-spot states are already cached, so this is a ridge solve
per layer and a scalar. Ranking is unchanged, so no AUROC moves; what changes is that a
score read through the map is no longer compressed toward the corpus average.
"""
import glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import numpy as np
from jvlm import cache, reader

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/vlm/results/reader"
for fit_path in sorted(glob.glob(f"{root}/*__fit_ans.npz")):
    name = os.path.basename(fit_path).replace("__fit_ans.npz", "")
    _, side = cache.load(fit_path)
    ANS = side["_extra"]["ANS"].astype(np.float32)
    layers = list(range(1, ANS.shape[1] - 1))
    R = reader.fit(ANS, layers, meta=dict(repo=name, corpus="COCO answer spots", n=int(ANS.shape[0])))
    out = f"{root}/{name}__reader.pt"
    R.save(out)
    sc = R.meta.get("scales", {})
    vals = sorted(float(v) for v in sc.values())
    print(f"{name}: refit {ANS.shape[0]} states x {len(layers)} layers -> {out}; "
          f"scale correction {vals[0]:.2f} to {vals[-1]:.2f} (median {vals[len(vals)//2]:.2f})")
