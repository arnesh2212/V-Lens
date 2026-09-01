"""The three safety figures.

The safety section carried no figures at all: a jailbreak AUROC with its floor
in a table column, a compliance result stated only in prose, and the one
comparison this paper loses buried in a bullet list.  This draws all three from
result files that already exist.  Nothing here touches a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jvlm import figsafety

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="/workspace/vlm/results")
ap.add_argument("--out", default="/workspace/vlm/figures")
ap.add_argument("--jailbreak", default="jailbreak/qwen2b_v2_stats.json")
ap.add_argument("--raw", default="jailbreak/qwen2b_v2_raw.parquet")
ap.add_argument("--pope", default="pope/qwen2b_3lens_stats.json")
ap.add_argument("--only", default=None, help="a,b,c")
a = ap.parse_args()

want = set((a.only or "a,b,c").split(","))


def load(rel):
    p = os.path.join(a.results, rel)
    if not os.path.exists(p):
        print(f"MISSING {p}")
        return None
    return json.load(open(p))


def paired_typo(path, layers, seed=7, reps=60):
    """Clean and typed-on split-half AUROC on the ONE set of items that has both.

    Reproduces ``split_half`` from ``exp/69_jailbreak2.py`` exactly: the layer
    is chosen on one half of the items and scored on the other, both directions
    of every split, averaged.  Without this the third panel of f80 sets the
    clean block's 280 items against the typed block's 100, and a difference
    between two different sets of items is not what that panel claims to show.
    Reads a parquet and nothing else; no model, no GPU.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata

    def auc_rows(X, y):
        X = np.atleast_2d(np.asarray(X, float))
        npos, nneg = int(y.sum()), int((~y).sum())
        if npos < 3 or nneg < 3:
            return np.full(X.shape[0], np.nan)
        r = rankdata(X, axis=1)
        return (r[:, y].sum(1) - npos * (npos + 1) / 2) / (npos * nneg)

    def split_half(X, y, rng):
        X = np.atleast_2d(np.asarray(X, float))
        ip, iN = np.where(y)[0], np.where(~y)[0]
        if len(ip) < 8 or len(iN) < 8:
            return float("nan")
        got = []
        for _ in range(reps):
            pa, pb = np.array_split(rng.permutation(ip), 2)
            na, nb = np.array_split(rng.permutation(iN), 2)
            for s1p, s1n, s2p, s2n in ((pa, na, pb, nb), (pb, nb, pa, na)):
                i1 = np.concatenate([s1p, s1n])
                v = auc_rows(X[:, i1], np.arange(len(i1)) < len(s1p))
                if np.all(np.isnan(v)):
                    continue
                b = int(np.nanargmax(v))
                i2 = np.concatenate([s2p, s2n])
                got.append(float(auc_rows(X[b:b + 1, i2],
                                          np.arange(len(i2)) < len(s2p))[0]))
        got = [v for v in got if v == v]
        return float(np.mean(got)) if got else float("nan")

    def one(d, fam, rng):
        cs = [f"{fam}_region_{l}" for l in layers
              if f"{fam}_region_{l}" in d.columns]
        hp, hn = d[d.harmful == 1], d[d.harmful == 0]
        X = np.vstack([np.concatenate([hp[c].values, hn[c].values])
                       for c in cs])
        y = np.arange(X.shape[1]) < len(hp)
        return split_half(X, y, rng), len(hp), len(hn)

    if not os.path.exists(path):
        print(f"MISSING {path}; panel three falls back to the unmatched blocks")
        return None
    df = pd.read_parquet(path)
    ty, sd = df[df.block == "SD_TYPO"], df[df.block == "SD"]
    ids = set(ty.pair_id)
    if not ids or not ids <= set(sd.pair_id):
        print("the typed items are not a subset of the clean block; "
              "panel three falls back to the unmatched blocks")
        return None
    cl = sd[sd.pair_id.isin(ids)]
    out = {}
    for fam in ("logit", "jlens", "vj"):
        c, np_, nn_ = one(cl, fam, np.random.default_rng(seed))
        t, _, _ = one(ty, fam, np.random.default_rng(seed))
        out[fam] = {"clean": c, "typo": t, "n_pos": np_, "n_neg": nn_}
        print(f"  {fam:6s} matched clean {c:.4f}   typed-on {t:.4f}   "
              f"(+{t - c:.3f})   on {np_} + {nn_} items", flush=True)
    return out


jb = load(a.jailbreak)
pope = load(a.pope)

if jb and "a" in want:
    print("the typed-on comparison, on the items that exist both ways:")
    pr = paired_typo(os.path.join(a.results, a.raw), jb["layers"])
    print(figsafety.jailbreak(jb, a.out, paired=pr))
if jb and "b" in want:
    print(figsafety.benchmark(jb, a.out))
if pope and "c" in want:
    print(figsafety.hallucination(pope, a.out))
