# SPDX-License-Identifier: Apache-2.0
"""V-Read against every reading baseline, on the cached answer-spot states.

No new forward passes: the behaviour pass already stored the residual state at
every block, so this compares instruments on exactly the same activations.

Instruments, all cross-fitted over items in the same five folds:

    python3 experiments/simonsays/lens_read.py --repo Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
import torch  # noqa: E402


from vlens.lens.metrics import auroc  # noqa: E402


from vlens.lens.metrics import boot as boot_auroc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--results", default="results/simonsays")
    ap.add_argument("--out", default=None)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=64, help="rank of the AnchorLens map")
    ap.add_argument("--kl-steps", type=int, default=250)
    ap.add_argument("--kl-lr-grid", type=float, nargs="*", default=[1.0, 0.1, 0.01])
    ap.add_argument("--kl-probe-steps", type=int, default=40)
    ap.add_argument("--tuned-from-ridge", action="store_true",
                    help="start the tuned lens from the ridge solution instead of the identity")
    ap.add_argument("--min-items", type=int, default=24)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--skip", nargs="*", default=[], help="instruments to leave out")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens.lens import depth, scores, tuned
    from vlens.data import simonsays

    name = a.repo.split("/")[-1]
    out_dir = a.out or a.results
    os.makedirs(out_dir, exist_ok=True)
    fams = sorted({os.path.basename(p).split("__")[1].replace("_states.npz", "")
                   for p in __import__("glob").glob(f"{a.results}/{name}__*_states.npz")})
    if a.families:
        fams = [f for f in fams if f in a.families]
    # the password study's own two families: one is the set it fine-tunes on
    fams = [f for f in fams if f not in simonsays.PASSWORD_LOCKING]
    say(f"{name}: {len(fams)} families with cached states")

    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    LAST = m.n_layers - 1
    SRC = list(range(max(m.n_deepstack, 1), m.n_layers))
    frac = {l: depth.to_frac(m, l) for l in range(m.n_layers)}

    def read_scores(H, kinds, expected):
        """Score every row of predicted final states H [n, d] under its own kind."""
        with torch.no_grad():
            lg = m.unembed(torch.as_tensor(np.asarray(H, np.float32)).to(m.device).to(torch.bfloat16))
        out = np.zeros(len(kinds), np.float32)
        for j, (k, e) in enumerate(zip(kinds, expected)):
            fn = scores.score_fn(TS, k, e)
            out[j] = float(fn(None, lg[j: j + 1]).float().cpu()[0])
        return out

    all_out = {}
    for fam in fams:
        pj, pn = f"{a.results}/{name}__{fam}.json", f"{a.results}/{name}__{fam}_states.npz"
        if not (os.path.exists(pj) and os.path.exists(pn)):
            continue
        rows = {r["id"]: r for r in json.load(open(pj))["rows"]}
        z = np.load(pn); ids = [str(x) for x in z["ids"]]
        stored = [int(x) for x in z["layers"]] if "layers" in z else list(range(m.n_layers))
        col = {l: j for j, l in enumerate(stored)}   # block index -> column in ANS
        use = [l for l in SRC if l in col]
        ANS = z["ANS"].astype(np.float32)
        keep = [i for i, k in enumerate(ids) if k in rows]
        ids = [ids[i] for i in keep]; ANS = ANS[keep]
        if len(ids) < a.min_items:
            say(f"  {fam}: only {len(ids)} scored items, skipped"); continue
        R = [rows[k] for k in ids]
        kinds = [r["kind"] for r in R]
        expected = [r.get("expected") for r in R]
        n = len(ids)
        t0 = time.time()

        # the model's own decision, and the data's answer
        fin = read_scores(ANS[:, col[LAST]], kinds, expected)
        y_final = (fin > 0).astype(int)
        y_exp = []
        for r in R:
            e = str(r.get("expected") or "").strip().lower()
            if r["kind"] == "yesno":
                y_exp.append(1 if e.startswith("yes") else (0 if e.startswith("no") else -1))
            elif r["kind"] == "test_real":
                y_exp.append(1 if e.startswith("test") else (0 if e.startswith("real") else -1))
            else:
                y_exp.append(1)   # the letter score is already correct-over-best-other
        y_exp = np.array(y_exp)

        rng = np.random.default_rng(0)
        # grouped by picture: a family holds an original and its edited twin,
        # and an ungrouped split would put one in train and the other in test
        grp = np.array([str((R[i].get("meta") or {}).get("image_number", ids[i])) for i in range(n)])
        uq = np.unique(grp)
        gf = {g: k % a.folds for k, g in enumerate(rng.permutation(uq))}
        which = np.array([gf[g] for g in grp])
        folds = [np.nonzero(which == k)[0] for k in range(a.folds)]
        inst = {k: np.zeros((n, len(use)), np.float32) for k in ("v_read", "logit", "tuned", "probe")
                if k not in a.skip or k == "logit"}
        for fi, te in enumerate(folds):
            tr = np.setdiff1d(np.arange(n), te)
            for j, l in enumerate(use):
                X, Y = ANS[tr, col[l]], ANS[tr, col[LAST]]
                A, b, _lam = tuned.fit_ridge_cv(X, Y, m=m)
                if "v_read" in inst:
                    P = torch.as_tensor(ANS[te, col[l]]).float() @ A.T + b
                    inst["v_read"][te, j] = read_scores(P.numpy(), [kinds[i] for i in te], [expected[i] for i in te])
                if "tuned" in inst:
                    A0, b0 = ((A, b) if a.tuned_from_ridge else
                              (torch.eye(m.d_model), torch.zeros(m.d_model)))
                    # the baseline's learning rate is searched on the same
                    # rows, by its own objective, rather than being fixed
                    klr = tuned.pick_kl_lr(m, X[: len(X) * 3 // 4], Y[: len(Y) * 3 // 4],
                                           X[len(X) * 3 // 4:], Y[len(Y) * 3 // 4:],
                                           lrs=tuple(a.kl_lr_grid),
                                           probe_steps=a.kl_probe_steps, d_model=m.d_model)
                    At, bt = tuned.refine_kl_states(
                        m, X, Y, A0, b0, steps=a.kl_steps, lr=klr,
                        momentum=0.9, weight_decay=1e-3, clip=1.0, decay=True)
                    P = torch.as_tensor(ANS[te, col[l]]).float() @ At.T + bt
                    inst["tuned"][te, j] = read_scores(P.numpy(), [kinds[i] for i in te], [expected[i] for i in te])
            say(f"    {fam}: fold {fi + 1}/{a.folds} ({time.time() - t0:.0f}s)")
            if "logit" in inst:
                for j, l in enumerate(use):
                    inst["logit"][te, j] = read_scores(ANS[te, col[l]], [kinds[i] for i in te], [expected[i] for i in te])
        # the labelled reference
        probe_t, probe_ok = {}, {}
        if "probe" in inst:
            from vlens.lens import probe as PB
            for te in folds:
                tr = np.setdiff1d(np.arange(n), te)
                if len(te) == 0 or len(set(y_final[tr].tolist())) < 2:
                    continue
                for j, l in enumerate(use):
                    for tname, yy in (("model", y_final), ("data", y_exp)):
                        sel = yy >= 0
                        trs = np.array([i for i in tr if sel[i]])
                        if len(trs) < 8 or len(set(yy[trs].tolist())) < 2:
                            continue
                        probe_t.setdefault(tname, np.zeros((n, len(use)), np.float32))
                        probe_ok.setdefault(tname, np.zeros(n, bool))[te] = True
                        probe_t[tname][te, j] = PB.scores(PB.fit(ANS[trs, col[l]], yy[trs]),
                                                          ANS[te, col[l]])
                    if "model" in probe_t:
                        inst["probe"][te, j] = probe_t["model"][te, j]

        res = dict(family=fam, n=n, layers=use, depths=[frac[l] for l in use], kinds=sorted(set(kinds)), auroc={}, ci={})
        for target, y in (("model", y_final), ("data", y_exp)):
            sel = np.where(y >= 0)[0]
            if len(sel) < a.min_items or len(set(y[sel].tolist())) < 2:
                continue
            for k, S_ in inst.items():
                if k == "probe" and target in probe_t:
                    S_ = probe_t[target]
                    ok = probe_ok.get(target, np.ones(n, bool))
                    sel = np.array([i for i in sel if ok[i]]) if ok.any() else sel
                pts, los, his = [], [], []
                for j in range(len(use)):
                    p, lo, hi = boot_auroc(y[sel], S_[sel, j], reps=a.boot)
                    pts.append(p); los.append(lo); his.append(hi)
                res["auroc"][f"{target}|{k}"] = pts
                res["ci"][f"{target}|{k}"] = [los, his]
        all_out[fam] = res
        json.dump(res, open(f"{out_dir}/{name}__{fam}_read.json", "w"), default=float)
        at = lambda v, x: v[int(np.argmin(np.abs(np.array(res["depths"]) - x)))]
        line = f"  {fam:<24} n={n:<5}"
        for k in ("v_read", "tuned", "logit", "probe"):
            key = f"model|{k}"
            if key in res["auroc"]:
                v = res["auroc"][key]
                line += f" {k}:{at(v, 0.0):.2f}/{at(v, 0.25):.2f}/{at(v, 0.5):.2f}"
        say(line + f"   ({time.time() - t0:.0f}s)")
    json.dump(dict(repo=a.repo, families=all_out), open(f"{out_dir}/{name}__read_summary.json", "w"), default=float)
    say("DONE lens_read")


if __name__ == "__main__":
    main()
