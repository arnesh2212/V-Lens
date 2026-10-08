# SPDX-License-Identifier: Apache-2.0
"""Does a map fitted on one prompt family read another?

Each family's own map is compared with a map fitted on a different family and
with one fitted on a pooled corpus, all read on the same held-out states. A
transport that only works where it was fitted is a weaker instrument than one
that carries across, and the gap says how much refitting buys.

    python3 experiments/simonsays/transfer.py --repo Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from vlens.lens.metrics import auroc, boot  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--results", default="results/simonsays")
    ap.add_argument("--out", default="results/transfer")
    ap.add_argument("--min-items", type=int, default=60)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens.lens import scores, tuned

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    TS = scores.TokenSets(m.tokenizer)
    LAST = m.n_layers - 1

    fam = {}
    for p in sorted(glob.glob(f"{a.results}/{name}__*_states.npz")):
        f = os.path.basename(p).split("__")[1].replace("_states.npz", "")
        j = p.replace("_states.npz", ".json")
        if not os.path.exists(j):
            continue
        rows = {r["id"]: r for r in json.load(open(j))["rows"]}
        z = np.load(p)
        ids = [str(x) for x in z["ids"]]
        stored = [int(x) for x in z["layers"]] if "layers" in z else list(range(m.n_layers))
        keep = [i for i, k in enumerate(ids) if k in rows]
        if len(keep) < a.min_items:
            continue
        fam[f] = dict(ANS=z["ANS"].astype(np.float32)[keep],
                      R=[rows[ids[i]] for i in keep], col={l: i for i, l in enumerate(stored)})
    say(f"{len(fam)} families with at least {a.min_items} scored items")
    if len(fam) < 3:
        say("need three families"); return
    use = [l for l in sorted(set.intersection(*[set(v["col"]) for v in fam.values()])) if l < LAST]
    if not use:
        say("the families were cached on different layer grids; nothing in common"); return

    def read(H, R):
        with torch.no_grad():
            lg = m.unembed(torch.as_tensor(np.asarray(H, np.float32)).to(m.device).to(torch.bfloat16))
        return np.array([float(scores.score_fn(TS, r["kind"], r.get("expected"))(
            None, lg[i: i + 1]).float().cpu()[0]) for i, r in enumerate(R)], np.float32)

    names = sorted(fam)
    out = dict(repo=a.repo, layers=use, depths=[l / LAST for l in use], families={})
    for f in names:
        d = fam[f]
        y = (read(d["ANS"][:, d["col"][LAST]], d["R"]) > 0).astype(int)
        if len(set(y.tolist())) < 2:
            continue
        others = [g for g in names if g != f]
        pooled = np.concatenate([fam[g]["ANS"][:, [fam[g]["col"][l] for l in [*use, LAST]]]
                                 for g in others])
        rng = np.random.default_rng(0)
        folds = np.array_split(rng.permutation(len(y)), 5)
        curves = {k: np.zeros((len(y), len(use)), np.float32)
                  for k in ("own", "foreign", "pooled", "none")}
        foreign = max(others, key=lambda g: len(fam[g]["R"]))
        fo = fam[foreign]
        for j, l in enumerate(use):
            # every arm gets its penalty chosen the same way, inside its own
            # rows, so a map that travels badly is not just a badly fitted one
            Af, bf, _ = tuned.fit_ridge_cv(fo["ANS"][:, fo["col"][l]], fo["ANS"][:, fo["col"][LAST]], m=m)
            Ap, bp, _ = tuned.fit_ridge_cv(pooled[:, j], pooled[:, len(use)], m=m)
            X, Y = d["ANS"][:, d["col"][l]], d["ANS"][:, d["col"][LAST]]
            curves["foreign"][:, j] = read((torch.as_tensor(X).float() @ Af.T + bf).numpy(), d["R"])
            curves["pooled"][:, j] = read((torch.as_tensor(X).float() @ Ap.T + bp).numpy(), d["R"])
            curves["none"][:, j] = read(X, d["R"])
            for te in folds:
                tr = np.setdiff1d(np.arange(len(y)), te)
                Ao, bo, _ = tuned.fit_ridge_cv(X[tr], Y[tr], m=m)
                curves["own"][te, j] = read((torch.as_tensor(X[te]).float() @ Ao.T + bo).numpy(),
                                            [d["R"][i] for i in te])
        e = dict(n=len(y), foreign_family=foreign, auroc={}, ci={})
        for k, M in curves.items():
            pts = [boot(y, M[:, j], reps=a.boot) for j in range(len(use))]
            e["auroc"][k] = [p[0] for p in pts]
            e["ci"][k] = [[p[1] for p in pts], [p[2] for p in pts]]
        out["families"][f] = e
        at = lambda v, x: v[int(np.argmin(np.abs(np.array(out["depths"]) - x)))]
        say(f"  {f:<26} n={len(y):<5} " + "  ".join(
            f"{k} {at(e['auroc'][k], 0.25):.2f}" for k in ("own", "pooled", "foreign", "none")))
    json.dump(out, open(f"{a.out}/{name}__transfer.json", "w"), default=float)
    agg = {k: float(np.nanmean([np.nanmean(v["auroc"][k]) for v in out["families"].values()]))
           for k in ("own", "pooled", "foreign", "none")}
    say("\nmean over families and depths: " + "  ".join(f"{k} {v:.3f}" for k, v in agg.items()))
    say("DONE transfer")


if __name__ == "__main__":
    main()
