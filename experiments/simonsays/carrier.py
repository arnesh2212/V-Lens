# SPDX-License-Identifier: Apache-2.0
"""Where in the network the words on a sign enter the decision.

The benchmark carries matched pairs: the same photograph and the same
question, once as taken and once with the text on the sign edited. The answer
differs between the two, so somewhere between the input and the output the
reading of them has to part company. This finds where.

Every record is read at every block through V-Read, the tuned lens and the
logit lens, as the signed margin of yes over no rather than as an AUROC,
because the question is not how well an instrument separates two groups but
at what depth a particular answer is made. The pairing is what makes the
separation legible: the two halves share everything except the words, so the
depth at which their traces diverge is the depth at which the words are read.

One thing this cannot answer. The original artifact asked where the sign's
*carrier* is weighed, and found nowhere, using a conflict against a
swapped conflict on the same scene. This benchmark's authority judgement
questions are open-ended rather than yes or no, so they have no margin to
read at depth, and it carries no swapped-carrier variant. The carrier claim
is therefore not made here; what is made is the claim about the words.

    python3 experiments/simonsays/carrier.py --repo Qwen/Qwen3-VL-8B-Instruct
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

#: the subcategories that ask what the words say to do, which is the only one
#: of the two questions this benchmark asks with a yes or no answer
WORD = ("operative", "defer", "ignore", "operative_with_graffiti", "which_sign")

#: the questions that ask whether a named sign is present when it is only
#: pictured inside something else: a stop sign on a billboard, an exit sign in
#: a framed photograph
DEPICTED = ("p5", "p7", "is_official")


def crossing(depths, trace):
    """The depth at which a trace first crosses zero, or ``nan`` if it never does."""
    d, y = np.asarray(depths, float), np.asarray(trace, float)
    for i in range(1, len(y)):
        if np.isfinite(y[i - 1]) and np.isfinite(y[i]) and (y[i - 1] < 0) != (y[i] < 0):
            t = abs(y[i - 1]) / max(abs(y[i - 1]) + abs(y[i]), 1e-9)
            return float(d[i - 1] + t * (d[i] - d[i - 1]))
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/carrier")
    ap.add_argument("--results", default="results/simonsays")
    ap.add_argument("--n", type=int, default=0, help="records per group; 0 means every one")
    ap.add_argument("--cap", type=int, default=FA_SHARED["cap"])
    ap.add_argument("--layers", type=int, default=FA_SHARED["layers"])
    ap.add_argument("--corpus", type=int, default=FA_SHARED["corpus"])
    ap.add_argument("--heldout", type=int, default=FA_SHARED["heldout"])
    ap.add_argument("--steps", type=int, default=FA_SHARED["steps"])
    ap.add_argument("--square", type=int, default=FA_SHARED["square"])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays as SS
    from vlens.lens import fitall as FA, scores
    from vlens.lens.hooks import ActivationRecorder

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    TS = scores.TokenSets(m.tokenizer)
    LAST = m.n_layers - 1
    LAYERS = FA.layer_grid(m, a.layers)
    depths = [l / LAST for l in LAYERS]
    say(f"{name}: reading at {LAYERS}")

    recs = SS.load(require_image=True)
    spare = FA.fitting_pool(recs, a.cap)
    maps, latent, FIT, meta = FA.fit_all(
        m, spare, C.load_image, layers=LAYERS, corpus=a.corpus, heldout=a.heldout,
        square=a.square, steps=a.steps, tokens_per_step=FA_SHARED["tokens_per_step"],
        jac_prompts=FA_SHARED["jac_prompts"], latent_captions=FA_SHARED["latent_captions"],
        fit_batch=FA_SHARED["fit_batch"], want=("v_read", "tuned"),
        results=a.results, log=say)

    # matched pairs: the same picture and question, once as taken and once
    # with the words on the sign changed
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from handoff import pairs_for
    prs = [(o, e) for o, e in pairs_for(recs)
           if o["kind"] == "yesno" and str(o.get("subcategory")) in WORD
           and str(o["id"]) not in FIT and str(e["id"]) not in FIT]
    if a.n:
        prs = prs[: a.n]
    say(f"{len(prs)} matched pairs with a yes/no answer")
    src_of = lambda r: str(r["family"]).split("_")[0]
    seen = {}
    for o, e in prs:
        seen[src_of(o)] = seen.get(src_of(o), 0) + 1
    for k, v in sorted(seen.items()):
        say(f"  {k:<14}{v}")
    groups, pair_of = {}, {}
    for i, (o, e) in enumerate(prs):
        groups.setdefault(f"as taken|{src_of(o)}", []).append(o)
        groups.setdefault(f"edited|{src_of(e)}", []).append(e)
        pair_of[str(o["id"])] = i
        pair_of[str(e["id"])] = i
    # and the depicted-sign questions, which ask whether a named sign is
    # present when it is only pictured inside something else. There are few of
    # them and the count is printed with the reading, because a dozen items is
    # a dozen items however early the answer is legible.
    dep = [r for r in recs if str(r.get("subcategory")) in DEPICTED
           and r["kind"] == "yesno" and str(r["id"]) not in FIT]
    for r in dep:
        groups.setdefault("depicted sign", []).append(r)
    say(f"{len(dep)} depicted-sign records with a yes/no answer")

    fn = scores.score_fn(TS, "yesno", None)
    rows, t0 = [], time.time()
    for key, rs in sorted(groups.items()):
        for r in rs:
            b = m.chat(r["question"], [C.load_image(r["image"], square=a.square)],
                       system=r["system"], cue=r["cue"])
            S = b["input_ids"].shape[1]
            with torch.no_grad(), ActivationRecorder(m.layers, at=[*LAYERS, LAST]) as rec:
                m.forward(b)
                # the answer spot, which is where the decision is read, not the
                # image positions ``states_at`` hands back
                h = {l: rec.activations[l][0, S - 1][None] for l in [*LAYERS, LAST]}
                lg = m.unembed(h[LAST])
                trace = {}
                for inst in ("v_read", "tuned"):
                    M = maps.get(inst)
                    if M is None:
                        continue
                    trace[inst] = [float(fn(None, m.unembed(
                        (h[l].float() @ M[l][0].T.to(m.device)
                         + M[l][1].to(m.device)).to(torch.bfloat16)))[0]) for l in LAYERS]
                trace["logit"] = [float(fn(None, m.unembed(h[l]))[0]) for l in LAYERS]
            rows.append(dict(id=str(r["id"]), family=r["family"], subcategory=r["subcategory"],
                             group=key, pair=pair_of.get(str(r["id"])),
                             final=float(fn(None, lg).float().cpu()[0]), trace=trace))
            if len(rows) % 100 == 0:
                say(f"  {len(rows)} records ({(time.time() - t0) / 60:.1f} min)")

    if not rows:
        say("nothing to read"); return
    rng = np.random.default_rng(0)
    summ = dict(repo=a.repo, layers=LAYERS, depths=depths, n=len(rows),
                meta=meta, groups={})
    # the pooled groups first, then the same split by source, and the
    # depicted-sign items split by the answer the model actually gave
    keys = ([("as taken",), ("edited",)]
            + sorted({tuple(r["group"].split("|")) for r in rows if "|" in r["group"]})
            + [("depicted sign", w) for w in ("says yes", "says no")])
    for key in keys:
        if key[0] == "depicted sign":
            want = key[1] == "says yes"
            sub = [r for r in rows if r["group"] == "depicted sign"
                   and (r["final"] > 0) == want]
        else:
            sub = [r for r in rows if r["group"].split("|")[: len(key)] == list(key)]
        if not sub:
            continue
        e = dict(n=len(sub), trace={}, crossing={})
        for inst in ("v_read", "tuned", "logit"):
            T = np.array([r["trace"][inst] for r in sub if inst in r["trace"]], float)
            if not len(T):
                continue
            e["trace"][inst] = [float(v) for v in T.mean(0)]
            # the crossing of the mean trace, with the interval taken over
            # records rather than over blocks
            boot = [crossing(depths, T[rng.integers(0, len(T), len(T))].mean(0))
                    for _ in range(a.boot)]
            boot = np.array([v for v in boot if np.isfinite(v)], float)
            e["crossing"][inst] = dict(
                at=crossing(depths, T.mean(0)),
                lo=float(np.percentile(boot, 2.5)) if len(boot) > 20 else None,
                hi=float(np.percentile(boot, 97.5)) if len(boot) > 20 else None,
                never=float(1.0 - len(boot) / a.boot))
        e["final"] = float(np.mean([r["final"] for r in sub]))
        summ["groups"]["|".join(key)] = e

    # the paired gap: the same photograph and question with the words changed,
    # so the depth at which the two readings part company is read off one
    # number per pair rather than off two means that never shared a record
    by_pair = {}
    for r in rows:
        if r["pair"] is None:
            continue
        by_pair.setdefault(r["pair"], {})[r["group"].split("|")[0]] = r
    summ["pair_gap"] = {}
    for inst in ("v_read", "tuned", "logit"):
        D = np.array([[e - o for e, o in zip(v["edited"]["trace"][inst],
                                             v["as taken"]["trace"][inst])]
                      for v in by_pair.values()
                      if len(v) == 2 and inst in v["edited"]["trace"]], float)
        if not len(D):
            continue
        boot = np.array([D[rng.integers(0, len(D), len(D))].mean(0) for _ in range(a.boot)])
        summ["pair_gap"][inst] = dict(
            n=int(len(D)), mean=[float(v) for v in D.mean(0)],
            lo=[float(v) for v in np.percentile(boot, 2.5, axis=0)],
            hi=[float(v) for v in np.percentile(boot, 97.5, axis=0)],
            # two consecutive blocks clear of zero, so a single noisy block
            # near the input cannot be read as the words arriving
            parts_at=next((float(depths[j]) for j in range(len(depths) - 1)
                           if np.percentile(boot[:, j], 2.5) > 0
                           and np.percentile(boot[:, j + 1], 2.5) > 0), float("nan")))
    json.dump(dict(summary=summ, rows=rows), open(f"{a.out}/{name}__carrier.json", "w"),
              default=float)
    say(f"\nthe yes-over-no margin by depth, mean over records")
    say("  group                          " + "".join(f"{d:>8.2f}" for d in depths[::3]))
    for key, e in summ["groups"].items():
        for inst in ("v_read", "logit"):
            if inst not in e["trace"]:
                continue
            say(f"  {key + ' / ' + inst:<30}"
                + "".join(f"{v:>8.2f}" for v in e["trace"][inst][::3]))
    say("\n  the depth at which the mean trace crosses from no to yes")
    for key, e in summ["groups"].items():
        for inst, c in e["crossing"].items():
            at = c["at"]
            say(f"  {key + ' / ' + inst:<30}"
                + ("never" if not np.isfinite(at) else
                   f"{at:.2f}" + (f" [{c['lo']:.2f}, {c['hi']:.2f}]"
                                  if c.get("lo") is not None else ""))
                + (f"   never on {c['never']:.0%} of resamples" if c["never"] > 0.01 else ""))
    if summ.get("pair_gap"):
        say("\n  the paired gap, the edited reading minus the one as taken")
        say("  instrument     " + "".join(f"{d:>8.2f}" for d in depths[::3]))
        for inst, g in summ["pair_gap"].items():
            say(f"  {inst:<14}" + "".join(f"{v:>8.2f}" for v in g["mean"][::3]))
        for inst, g in summ["pair_gap"].items():
            at = g["parts_at"]
            say(f"  {inst:<14}parts company at depth "
                + ("never" if not np.isfinite(at) else f"{at:.2f}")
                + f"   ({g['n']} pairs)")
    say("DONE carrier")


if __name__ == "__main__":
    main()
