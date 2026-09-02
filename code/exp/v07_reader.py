# SPDX-License-Identifier: Apache-2.0
"""The strongest label-free READ baseline: a ridge transport at the answer slot.

Fit A_l h_l[a] + b_l -> h_L[a] on a fold of the eval cache (no refusal labels),
then read the refusal axis r . (A_l h_l[a] + b_l).  This is the tuned lens at
the answer slot.  If it reads the decision earlier than the identity, the
"early reading" claim belongs to the tuned lens, not to a gradient direction,
and the write-up must say so.  5-fold by item; CPU only.
"""
import sys, json, os
sys.path.insert(0, "code")
import numpy as np, torch
from jvlm import cache, verdict, probes, judge, tuned
VARS = os.environ.get("VLM_VARIANTS", "SD_TYPO").split(",")
out = {}
for tag in ("Qwen3-VL-2B-Instruct", "Qwen3-VL-8B-Instruct"):
    acts, side = cache.load(f"results/gate/{tag}__eval.npz")
    acts, side = cache.subset(acts, side, [it.get("variant") in VARS for it in side["items"]])
    G = json.load(open(f"results/gate/{tag}_gate.json"))
    ANS = side["_extra"]["ANS"].astype(np.float32); N, LN, d = ANS.shape
    y = np.array([judge.refused(it) for it in side["items"]], int)
    layers = sorted(int(l) for l in G["calibration"])
    rng = np.random.default_rng(0); folds = np.array_split(rng.permutation(N), 5)
    res = {}
    print(f"===== {tag}: {N} items, refused {y.mean():.2f}")
    print(f"{'depth':>6} {'ridge-read':>11} {'identity*':>10} {'probe':>7}   (* identity with the same per-fold axis)")
    for l in layers:
        pred = np.zeros(N); pred_id = np.zeros(N)
        for te in folds:
            tr = np.setdiff1d(np.arange(N), te)
            # the refusal axis is fitted on the TRAINING fold only: no held-out label touches it
            r = verdict.dim_direction(ANS[tr, -1], y[tr]).numpy()
            A, b = tuned.fit_ridge(ANS[tr, l], ANS[tr, -1], lam=1e-1)      # label-free transport
            hL_hat = ANS[te, l] @ A.numpy().T + b.numpy()
            pred[te] = hL_hat @ r
            pred_id[te] = ANS[te, l] @ r
        au = probes.readout_score(pred, y, shuffles=20)["auroc"]
        au_id = probes.readout_score(pred_id, y, shuffles=20)["auroc"]
        pr = {x["layer"]: x["score"] for x in G["probe_refused"]}[l]
        f = G["calibration"][str(l)]["frac"]; res[l] = dict(frac=f, ridge=au, identity_perfold=au_id, probe=pr)
        print(f"{f:>6.2f} {au:>11.3f} {au_id:>10.3f} {pr:>7.3f}")
    out[tag] = res
json.dump(out, open("results/gate/reader_regression.json", "w"), indent=1)
