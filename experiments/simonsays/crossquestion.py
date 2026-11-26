# SPDX-License-Identifier: Apache-2.0
"""One photograph, several words: is the map the question's or the picture's?

A saliency map that depends only on the picture would be the same whatever is
asked of it. The benchmark asks several different questions of the same
photograph, so the two can be told apart. For each picture and each pair of
questions the model answers correctly, V-Write is read for one question and
the patches it ranks highest are replaced; the fall is measured both for the
question the map was read for and for the other one. A map that belongs to the
question costs its own question more than it costs the other.

    python3 experiments/simonsays/crossquestion.py --repo Qwen/Qwen3-VL-8B-Instruct
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

KS = (0.1, 0.2, 0.3)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(-a)).astype(float)
    rb = np.argsort(np.argsort(-b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/crossquestion")
    ap.add_argument("--n", type=int, default=300, help="pictures; 0 means every one")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--steps-ig", type=int, default=8)
    ap.add_argument("--int-batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays
    from vlens.lens import attribution as AT, fitall as FA, readout, scores

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    TS = scores.TokenSets(m.tokenizer)
    L = max(int(round(a.frac * (m.n_layers - 1))), 1)
    rng = np.random.default_rng(a.seed)
    say(f"{name}: V-Write at block {L} of {m.n_layers}")

    recs = simonsays.scorable([r for r in simonsays.load(require_image=True)
                               if r["kind"] in ("yesno", "letter", "test_real")])
    by = {}
    for r in recs:
        by.setdefault(r["image"], []).append(r)
    # one pair per picture, the two questions furthest apart in wording, so the
    # test is not two phrasings of the same thing
    groups = []
    for img, rs in by.items():
        seen, uniq = set(), []
        for r in sorted(rs, key=lambda x: str(x["id"])):
            if r["question"] not in seen:
                seen.add(r["question"]); uniq.append(r)
        if len(uniq) >= 2:
            groups.append(dict(image=img, family=uniq[0]["family"], pair=uniq[:2]))
    say(f"{len(groups)} pictures with two different questions")
    groups = FA.balanced_sample(groups, a.n, seed=a.seed) if a.n else groups

    part = f"{a.out}/{name}__crossquestion.rows.jsonl"
    rows, seen_ids = [], set()
    if os.path.exists(part):
        with open(part) as fh:
            for line in fh:
                try:
                    r_ = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                rows.append(r_); seen_ids.add(r_["image"])
        say(f"  resuming: {len(rows)} pictures already measured")

    t0 = time.time()
    for g in groups:
        if g["image"] in seen_ids:
            continue
        img = C.load_image(g["image"], square=a.square)
        got = []
        for r in g["pair"]:
            b = m.chat(r["question"], [img], system=r["system"], cue=r["cue"])
            pos = readout.image_positions(m, b)
            if len(pos) < 16:
                got = []; break
            fn = scores.score_fn(TS, r["kind"], r.get("expected"))
            with torch.no_grad():
                _, H, _, lg0 = AT.states_at(m, b, [L])
            s0 = float(fn(None, lg0).float().cpu()[0])
            fn, s0 = scores.orient(fn, r["kind"], s0)
            if s0 <= 0:
                got = []; break            # both questions need an answer to read toward
            got.append(dict(r=r, b=b, pos=pos, fn=fn, s0=s0, h=H[L]))
        if len(got) != 2:
            continue
        P = len(got[0]["pos"])
        if len(got[1]["pos"]) != P:
            continue                        # the same picture, so the same patches
        # the gradients cost far more than the forward that decides whether the
        # pair is usable, so nothing is integrated until both are known to be
        for x in got:
            h = x["h"]
            x["attr"] = np.asarray(
                AT.path_reading(m, x["b"], x["fn"], L, x["pos"], h,
                                h.mean(0, keepdim=True).expand_as(h),
                                steps=a.steps_ig, log=say), float)
        rand = rng.permutation(P)
        drop = {}
        for i, cur in enumerate(got):
            other = got[1 - i]
            plan, X = [], []
            for src, order in (("own", np.argsort(-cur["attr"])),
                               ("other", np.argsort(-other["attr"])),
                               ("random", rand)):
                for k in KS:
                    nk = max(int(round(k * P)), 1)
                    x = cur["h"].clone(); x[order[:nk]] = 0
                    plan.append((src, k)); X.append(x)
            with torch.no_grad():
                after = AT.hooked_axis_many(m, cur["b"], cur["fn"], L, cur["pos"], X,
                                            batch=a.int_batch, log=say)
            for (src, k), v in zip(plan, after):
                drop[f"{i}|{src}|{k}"] = float(cur["s0"] - v)
        row = dict(image=str(g["image"]), family=g["family"], n_patches=int(P),
                   questions=[simonsays.safe_question(x["r"]) for x in got],
                   ids=[str(x["r"]["id"]) for x in got],
                   s0=[x["s0"] for x in got], drop=drop,
                   map_agreement=spearman(got[0]["attr"], got[1]["attr"]))
        rows.append(row)
        with open(part, "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        if len(rows) % 25 == 0:
            say(f"  {len(rows)} usable of {len(groups)} pictures tried "
                f"({(time.time() - t0) / 60:.1f} min)")

    if not rows:
        say("nothing measured"); return
    mean = lambda f: float(np.nanmean([f(x) for x in rows]))
    summ = dict(repo=a.repo, block=L, depth=L / (m.n_layers - 1), n=len(rows), ks=list(KS),
                map_agreement=mean(lambda x: x["map_agreement"]), curve={}, paired={})
    for src in ("own", "other", "random"):
        summ["curve"][src] = [mean(lambda x, k=k, s=src: np.nanmean(
            [x["drop"].get(f"{i}|{s}|{k}", np.nan) for i in (0, 1)])) for k in KS]
    for j, k in enumerate(KS):
        d = [np.nanmean([x["drop"].get(f"{i}|own|{k}", np.nan)
                         - x["drop"].get(f"{i}|other|{k}", np.nan) for i in (0, 1)])
             for x in rows]
        d = np.array([v for v in d if np.isfinite(v)], float)
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        summ["paired"][str(k)] = dict(mean=float(d.mean()), n=int(len(d)),
                                      wins=float(np.mean(d > 0)),
                                      lo=float(np.percentile(boot, 2.5)),
                                      hi=float(np.percentile(boot, 97.5)))
    json.dump(dict(summary=summ, rows=rows), open(f"{a.out}/{name}__crossquestion.json", "w"),
              default=float)
    if os.path.exists(part):
        os.remove(part)
    say(f"\nfall in the answer score, patches replaced by zero ({summ['n']} pictures)")
    say("  ranking        " + "".join(f"{int(k * 100):>9}%" for k in KS))
    for src, lab in (("own", "this question's map"), ("other", "the other question's map"),
                     ("random", "a random ranking")):
        say(f"  {lab:<28}" + "".join(f"{v:>10.2f}" for v in summ["curve"][src]))
    say(f"\n  the two maps agree in rank at {summ['map_agreement']:.2f}")
    for k in KS:
        p = summ["paired"][str(k)]
        say(f"  at {int(k * 100)}%: own beats other by {p['mean']:.2f} "
            f"[{p['lo']:.2f}, {p['hi']:.2f}], on {p['wins']:.0%} of pictures")
    say("DONE crossquestion")


if __name__ == "__main__":
    main()
