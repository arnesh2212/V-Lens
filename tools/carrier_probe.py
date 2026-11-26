# SPDX-License-Identifier: Apache-2.0
"""Does the model know what a sign is written on, even when its answer ignores it?

SimonSays edits one photograph so the same words sit on an official bearer (a
regulatory sign, a plate, a livery) or an informal one (cardboard, a
handwritten note, chalk). For every such record of the Open Images, interface
and nuScenes edit families, a logistic probe is trained at each block to tell
official from informal from the answer-spot state, in five folds grouped by
source photograph so the two versions of one picture never sit on both sides.
Beside it: the same probe on shuffled labels, and how well the model's own
final yes-minus-no margin separates the two bearers. A probe far above the
answer means the carrier is encoded and not used.

    python3 tools/carrier_probe.py      (CPU; reads results/simonsays)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from vlens.figures import fx_standing as S  # noqa: E402
from vlens.lens.metrics import auroc  # noqa: E402

MODELS = ("Qwen3-VL-8B-Instruct", "Qwen3.5-27B", "gemma-4-26B-A4B-it", "InternVL3_5-8B-HF")
SOURCES = ("openimages", "vins", "nuscenes")


def _data(results, model):
    X, y, g, margin = [], [], [], []
    for src in SOURCES:
        fam = f"{src}_edit_auth"
        rows = S._rows(results, model, fam)
        fin = S._final(results, model, fam) or {}
        p = f"{results}/simonsays/{model}__{fam}_states.npz"
        if not rows or not os.path.exists(p):
            continue
        z = np.load(p)
        idx = {str(i): j for j, i in enumerate(z["ids"])}
        A = z["ANS"]
        for r in rows:
            ev = (r.get("meta") or {}).get("edit_variant")
            side = 1 if ev in S.OFFICIAL else 0 if ev in S.INFORMAL else None
            if side is None or r["id"] not in idx:
                continue
            X.append(np.asarray(A[idx[r["id"]]], np.float32))
            y.append(side)
            g.append(f"{src}:{(r.get('meta') or {}).get('source_image')}")
            margin.append(fin.get(r["id"], np.nan))
    if not X:
        return None
    return np.stack(X), np.array(y), np.array(g), np.array(margin, float)


def _probe(X, y, g, seed=0, shuffle=False, fold_seed=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    yy = np.random.default_rng(seed).permutation(y) if shuffle else y
    out = np.zeros(len(y))
    # fold_seed None keeps the paper's deterministic GroupKFold; an int reshuffles the groups (E4 seeds)
    gkf = GroupKFold(n_splits=5) if fold_seed is None else GroupKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    for tr, te in gkf.split(X, yy, g):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(X[tr]), yy[tr])
        out[te] = lr.decision_function(sc.transform(X[te]))
    return auroc(yy, out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--out", default="results/carrier")
    ap.add_argument("--results", default="results", help="the folder holding simonsays/ (behaviour pass output)")
    ap.add_argument("--seed", type=int, default=None, help="reshuffle the grouped folds (E4); default: the paper's folds")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for m in a.models:
        d = _data(a.results, m)
        if d is None:
            continue
        X, y, g, margin = d
        L = X.shape[1]
        blocks = list(range(0, L, 2)) + ([L - 1] if (L - 1) % 2 else [])
        probe = [_probe(X[:, b], y, g, fold_seed=a.seed) for b in blocks]
        shuf = [_probe(X[:, b], y, g, seed=a.seed or 0, shuffle=True, fold_seed=a.seed) for b in blocks]
        ok = np.isfinite(margin)
        ans = auroc(y[ok], margin[ok]) if ok.sum() > 5 else None
        res = dict(model=m, n=int(len(y)), official=int(y.sum()), pictures=int(len(set(g))),
                   depth=[b / (L - 1) for b in blocks], probe=probe, shuffled=shuf,
                   answer_auroc=ans, answer_auroc_sym=(max(ans, 1 - ans) if ans is not None else None),
                   fold_seed=a.seed)
        json.dump(res, open(f"{a.out}/{m}__carrier.json", "w"), indent=1)
        mid = [p for dd, p in zip(res["depth"], probe) if 0.4 <= dd <= 0.8]
        print(f"{m}: n={len(y)} ({int(y.sum())} official, {len(set(g))} pictures); probe mid-depth "
              f"{np.mean(mid):.2f}, max {max(probe):.2f}; shuffled max {max(shuf):.2f}; "
              f"final answer separates bearers at {res['answer_auroc_sym']:.2f}")


if __name__ == "__main__":
    main()
