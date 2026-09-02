# SPDX-License-Identifier: Apache-2.0
"""Judge the replies an experiment produced, with guard models rather than string matching.

Any experiment that writes a JSON with a list of {request, reply, harmful} records can
be handed to this script.  It scores each pair with the HarmBench classifier and with
Qwen3Guard, writes the labels back into the same JSON, and prints each judge's
compliance rate and the pairwise agreement.

  python3 code/exp/j02_judge_replies.py results/prefill/<model>_prefill.json --key replies

The point is that "the model complied" has to mean the reply is an instance of the
harmful behaviour, not that it lacked a refusal phrase.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("json", help="a results JSON holding a list of reply records")
ap.add_argument("--key", default="replies", help="the field holding the list")
ap.add_argument("--request-field", default="request")
ap.add_argument("--reply-field", default="reply")
ap.add_argument("--judges", nargs="*", default=["harmbench", "qwen3guard"])
a = ap.parse_args()

from jvlm import guards, judge  # noqa: E402

say = lambda *x: print(*x, flush=True)
d = json.load(open(a.json))
recs = d.get(a.key)
if not recs:
    sys.exit(f"{a.json} has no list at '{a.key}'")
pairs = [(r.get(a.request_field) or r.get("q") or d.get("prompt", ""), r.get(a.reply_field, "")) for r in recs]
say(f"{len(pairs)} replies to judge from {a.json}")

t0 = time.time()
rows, cols = guards.judge_all(pairs, which=a.judges, say=say)
cols["string matching"] = [judge.complied(rep) for _, rep in pairs]
for r, v in zip(recs, rows):
    r.update(v)

groups = {}
for r in recs:
    groups.setdefault(r.get("group") or r.get("cond") or r.get("kind") or "all", []).append(r)

say(f"\ncompliance rate by judge")
say(f"{'group':<14} {'n':>5} " + " ".join(f"{k:>16}" for k in cols))
for g, rs in groups.items():
    line = [f"{g:<14} {len(rs):>5}"]
    for k in cols:
        key = "judge_majority_complied" if k == "majority" else (f"{k}_complied" if k != "string matching" else None)
        v = [judge.complied(r.get(a.reply_field, "")) if key is None else r.get(key) for r in rs]
        v = [x for x in v if x is not None]
        line.append(f"{np.mean(v) if v else float('nan'):>16.3f}")
    say(" ".join(line))

say(f"\npairwise agreement (Cohen's kappa)")
ks = list(cols)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        say(f"  {ks[i]} vs {ks[j]}: {guards.kappa(cols[ks[i]], cols[ks[j]]):.3f}")

d[a.key] = recs
d["judge_source"] = "two guard models, majority"
json.dump(d, open(a.json, "w"), indent=1, default=float)
say(f"\nlabels written back to {a.json}  ({(time.time() - t0) / 60:.1f} min)")
