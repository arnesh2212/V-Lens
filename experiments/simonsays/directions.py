# SPDX-License-Identifier: Apache-2.0
"""Four directions, and what each one does to the other three.

A direction that moves the answer might be moving the answer, or it might be
moving something the answer happens to ride on. The way to tell is to build
one direction per axis and then measure all the axes under each of them.

Three axes, all read at the answer spot: the operative answer (yes over no),
refusal (a refusal opener over a compliant one) and self-report (test over
real). For each, V-Write's direction is the corpus-mean gradient of that axis
at the image positions, which needs no labels. Beside them: the labelled
difference of means for the operative axis, the unembedding rows of the words
each axis is scored on, and a matched random direction.

Reported: the norms, the pairwise cosines, and the cross-effect table, which
is every direction added at the image tokens against every axis measured.

    python3 experiments/simonsays/directions.py --repo Qwen/Qwen3-VL-8B-Instruct
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

AXES = ("the answer", "refusal", "self-report")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/directions")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--n-fit", type=int, default=120, help="items each direction is fitted on")
    ap.add_argument("--n", type=int, default=150, help="held-out items the doses are measured on")
    ap.add_argument("--alphas", type=float, nargs="*", default=[1.0, 2.0])
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--cap", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays as SS
    from vlens.lens import fitall as FA, readout, scores, write as W
    from vlens.lens.hooks import ActivationRecorder

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    LAST = m.n_layers - 1
    L = max(int(round(a.frac * LAST)), 1)
    rng = np.random.default_rng(a.seed)
    say(f"{name}: directions at block {L} of {m.n_layers}")

    #: the three axes, each a function of the final logits
    lp = lambda lg: torch.log_softmax(lg.float(), -1)
    R = sorted({i for w in scores.REFUSE_STARTS for i in scores.first_ids(tok, w)})
    Cp = sorted({i for w in scores.COMPLY_STARTS for i in scores.first_ids(tok, w)})
    Cp = [i for i in Cp if i not in set(R)]
    score = {
        "the answer": lambda hL, lg: scores.lse(lp(lg), TS.yes) - scores.lse(lp(lg), TS.no),
        "refusal": lambda hL, lg: scores.lse(lp(lg), R) - scores.lse(lp(lg), Cp),
        "self-report": lambda hL, lg: scores.lse(lp(lg), TS.test) - scores.lse(lp(lg), TS.real),
    }
    rows_of = {"the answer": TS.yes, "refusal": R, "self-report": TS.test}

    recs = SS.load(require_image=True)
    spare = FA.fitting_pool(recs, a.cap)
    pool = [r for r in recs if r["kind"] in ("yesno", "test_real")
            and id(r) not in {id(x) for x in spare}]
    pool = FA.balanced_sample(pool, a.n_fit + a.n, seed=a.seed)
    item = lambda r: dict(question=r["question"],
                          image=C.load_image(r["image"], square=a.square),
                          system=r["system"], cue=r["cue"], kind=r["kind"])
    fit_items = [item(r) for r in pool[: a.n_fit]]
    held = [item(r) for r in pool[a.n_fit:]]
    say(f"{len(fit_items)} items to fit the directions on, {len(held)} to measure on")

    def read(b):
        with ActivationRecorder(m.layers, at=[LAST]) as rec:
            m.forward(b)
            lg = m.unembed(rec.activations[LAST][0, -1][None])
        return {ax: float(score[ax](None, lg).float().cpu()[0]) for ax in AXES}

    # ---- the directions
    t0 = time.time()
    dirs = {}
    for ax in AXES:
        dirs[f"V-Write, {ax}"] = W.write_direction(m, fit_items, L, score[ax], log=say)
        dirs[f"unembedding, {ax}"] = W.unit(sum(W.unembed_direction(m, t)
                                                for t in rows_of[ax][:12]))
        say(f"  {ax}: fitted ({(time.time() - t0) / 60:.1f} min)")
    # The labelled direction, for the operative axis. The label is the answer
    # the model gives, which is what the steering experiment moves and what a
    # difference of means would be built from by anyone using one here.
    with torch.no_grad():
        side = [read(m.chat(it["question"], [it["image"]], system=it["system"],
                            cue=it["cue"]))["the answer"] > 0 for it in fit_items]
    pos = [it for it, y in zip(fit_items, side) if y]
    neg = [it for it, y in zip(fit_items, side) if not y]
    say(f"  the labelled split: {len(pos)} answered one way, {len(neg)} the other")
    if len(pos) >= 8 and len(neg) >= 8:
        dirs["difference of means (labelled)"] = W.difference_of_means(m, pos, neg, L, log=say)
    dirs["random"] = W.random_direction(m.d_model, seed=a.seed)

    # every direction is unit length, because the dose is scaled by the local
    # residual norm and a direction that is not would be getting a bigger one
    dirs = {k: W.unit(v) for k, v in dirs.items()}
    names = list(dirs)
    U = torch.stack([dirs[k].float().flatten() for k in names])
    U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-9)
    cos = (U @ U.T).cpu().numpy()

    # ---- the cross-effect: every direction against every axis
    t0 = time.time()
    rows = []
    for i, it in enumerate(held):
        b = m.chat(it["question"], [it["image"]], system=it["system"], cue=it["cue"])
        p_i = readout.image_positions(m, b)
        if len(p_i) < 16:
            continue

        with torch.no_grad():
            base = read(b)
            got = {}
            for dn, u in dirs.items():
                for al in a.alphas:
                    with W.AddAtPositions(m.layers[L], u, al, p_i):
                        v = read(b)
                    got[f"{dn}|{al}"] = {ax: v[ax] - base[ax] for ax in AXES}
        rows.append(dict(base=base, moved=got))
        if len(rows) % 25 == 0:
            say(f"  {len(rows)}/{len(held)} items ({(time.time() - t0) / 60:.1f} min)")

    if not rows:
        say("nothing measured"); return
    summ = dict(repo=a.repo, block=L, depth=L / LAST, n=len(rows), alphas=list(a.alphas),
                directions=names, cosine={f"{x}|{y}": float(cos[i, j])
                                          for i, x in enumerate(names)
                                          for j, y in enumerate(names)},
                base={ax: float(np.mean([r["base"][ax] for r in rows])) for ax in AXES},
                moved={})
    for dn in names:
        for al in a.alphas:
            k = f"{dn}|{al}"
            summ["moved"][k] = {ax: float(np.mean([r["moved"][k][ax] for r in rows]))
                                for ax in AXES}
    json.dump(dict(summary=summ, rows=rows), open(f"{a.out}/{name}__directions.json", "w"),
              default=float)

    say("\ncosine between the directions")
    say("  " + " " * 32 + "".join(f"{n[:11]:>13}" for n in names))
    for i, x in enumerate(names):
        say(f"  {x:<32}" + "".join(f"{cos[i, j]:>13.2f}" for j in range(len(names))))
    say(f"\nwhat each direction does to each axis, mean over {len(rows)} held-out items")
    say("  " + " " * 34 + "".join(f"{ax:>14}" for ax in AXES))
    for dn in names:
        for al in a.alphas:
            k = f"{dn}|{al}"
            say(f"  {dn + ' at ' + str(al):<34}"
                + "".join(f"{summ['moved'][k][ax]:>14.2f}" for ax in AXES))
    say("DONE directions")


if __name__ == "__main__":
    main()
