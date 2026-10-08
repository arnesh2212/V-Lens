# SPDX-License-Identifier: Apache-2.0
"""Tell a true yes from a hallucinated one before the model speaks.

The bearer-removal edits give both: a yes to the untouched photograph is true,
a yes to the one whose words were removed is not. Both are the same token at
the same position, so only the state can separate them. Every instrument is
fitted on held-out records and frozen.

    python3 experiments/simonsays/hallucination.py --repo Qwen/Qwen3-VL-8B-Instruct
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

from vlens.lens.fitall import SHARED as FA_SHARED  # noqa: E402
from vlens.lens.metrics import auroc, boot  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/hallucination")
    ap.add_argument("--n", type=int, default=0, help="pairs to use; 0 means every one")
    ap.add_argument("--cap", type=int, default=FA_SHARED["cap"])
    ap.add_argument("--layers", type=int, default=FA_SHARED["layers"])
    ap.add_argument("--corpus", type=int, default=FA_SHARED["corpus"])
    ap.add_argument("--heldout", type=int, default=FA_SHARED["heldout"])
    ap.add_argument("--steps", type=int, default=FA_SHARED["steps"])
    ap.add_argument("--tokens-per-step", type=int, default=FA_SHARED["tokens_per_step"])
    ap.add_argument("--fit-batch", type=int, default=FA_SHARED["fit_batch"])
    ap.add_argument("--jac-prompts", type=int, default=FA_SHARED["jac_prompts"])
    ap.add_argument("--latent-captions", type=int, default=FA_SHARED["latent_captions"])
    ap.add_argument("--results", default="results/simonsays")
    ap.add_argument("--only", nargs="*", default=None,
                    help="instruments to fit; the default is the four published lenses and V-Read")
    ap.add_argument("--square", type=int, default=FA_SHARED["square"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays
    from vlens.lens import fitall as FA, probe as PB, readbatch as RB, scores
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from handoff import pairs_for

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    LAST = m.n_layers - 1
    GRID = FA.layer_grid(m, a.layers)
    say(f"{name}: reading at {GRID}")

    recs = simonsays.load(require_image=True)
    spare = FA.fitting_pool(recs, a.cap)
    maps, latent, FIT, meta = FA.fit_all(
        m, spare, C.load_image, layers=GRID, corpus=a.corpus, heldout=a.heldout,
        square=a.square, steps=a.steps, tokens_per_step=a.tokens_per_step,
        jac_prompts=a.jac_prompts, latent_captions=a.latent_captions,
        fit_batch=a.fit_batch, want=tuple(a.only) if a.only else FA.SHARED_WANT, results=a.results, log=say)
    names = ["v_read", *[k for k in maps if k != "v_read"], "logit"] + \
            (["latent"] if latent is not None else []) + ["probe"]
    names = list(dict.fromkeys(names))

    pairs = [(o, e) for o, e in pairs_for(recs)
             if o["kind"] == "yesno" and str(o["id"]) not in FIT and str(e["id"]) not in FIT]
    say(f"{len(pairs)} yes-or-no pairs where the edit removes what the question asks about")
    if a.n:
        pairs = FA.balanced_sample(
            [dict(_pair=p_, family=p_[0]["family"]) for p_ in pairs], a.n, seed=0)
        pairs = [r["_pair"] for r in pairs]

    fn = scores.score_fn(TS, "yesno", None)
    items, meta_rows = [], []
    for o, e in pairs:
        try:
            io_ = C.load_image(o["image"], square=a.square)
            ie_ = C.load_image(e["image"], square=a.square)
        except Exception:  # noqa: BLE001
            continue
        for rec, img, true_yes in ((o, io_, 1), (e, ie_, 0)):
            items.append(dict(question=rec["question"], image=img,
                              system=rec["system"], cue=rec["cue"]))
            meta_rows.append(dict(id=rec["id"], group=o["id"], true_yes=true_yes))
    say(f"{len(items)} prompts to read, {a.batch} at a time")
    t0 = time.time()
    Hs, lg = RB.answer_states(m, items, GRID, batch=a.batch,
                              progress=lambda k, n: say(f"  {k}/{n} "
                                                        f"({(time.time() - t0) / 60:.1f} min)")
                              if False else None)
    sc = np.array([float(fn(None, lg[i: i + 1]).float().cpu()[0]) for i in range(len(items))])
    rows = [dict(mr, score=float(sc[i])) for i, mr in enumerate(meta_rows) if sc[i] > 0]
    keep = [i for i in range(len(items)) if sc[i] > 0]
    X = {l: Hs[keep][:, j] for j, l in enumerate(GRID)}
    say(f"read in {(time.time() - t0) / 60:.1f} min")

    y = np.array([r["true_yes"] for r in rows])
    if len(set(y.tolist())) < 2 or len(rows) < 40:
        say(f"not enough of both kinds: {int(y.sum())} true, {int((1 - y).sum())} hallucinated")
        return
    say(f"{int(y.sum())} true yeses, {int((1 - y).sum())} hallucinated ones")
    inst = {k: np.zeros((len(rows), len(GRID)), np.float32) for k in names}
    for j, l in enumerate(GRID):
        h = torch.as_tensor(X[l]).to(m.device)
        got = FA.score_all(m, maps, latent, h, l, fn)
        for k, v in got.items():
            if k in inst:
                inst[k][:, j] = v
    # folds are grouped by picture so a pair never spans two folds
    groups = np.array([r["group"] for r in rows])
    uniq = np.unique(groups)
    rng = np.random.default_rng(0)
    gfold = {g: i % 5 for i, g in enumerate(rng.permutation(uniq))}
    which = np.array([gfold[g] for g in groups])
    folds = [np.nonzero(which == k)[0] for k in range(5)]
    probe_ok = np.zeros(len(rows), bool)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(rows)), te)
        if len(set(y[tr].tolist())) < 2:
            continue
        probe_ok[te] = True
        for j, l in enumerate(GRID):
            inst["probe"][te, j] = PB.scores(PB.fit(X[l][tr], y[tr]), X[l][te])
    if not probe_ok.any():
        probe_ok[:] = True

    out = dict(repo=a.repo, layers=GRID, depths=[l / LAST for l in GRID],
               n=len(rows), n_true=int(y.sum()), instruments=names, auroc={}, ci={})
    for k in names:
        sel = probe_ok if k == "probe" else np.ones(len(rows), bool)
        pts = [boot(y[sel], inst[k][sel, j], reps=a.boot) for j in range(len(GRID))]
        out["auroc"][k] = [p[0] for p in pts]
        out["ci"][k] = [[p[1] for p in pts], [p[2] for p in pts]]
    json.dump(out, open(f"{a.out}/{name}__hallucination.json", "w"), default=float)
    say("\nAUROC for a true yes against a hallucinated one, by depth")
    say("  depth        " + "".join(f"{d:>7.2f}" for d in out["depths"]))
    for k in names:
        say(f"  {k:<12}" + "".join(f"{v:>7.2f}" for v in out["auroc"][k]))
    say("DONE hallucination")


if __name__ == "__main__":
    main()
