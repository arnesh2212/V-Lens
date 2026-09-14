# SPDX-License-Identifier: Apache-2.0
"""Judge the behaviour replies of the custom datasets, in place.

Yes/no, letter and test/real replies are parsed; open replies go to the
text-only Qwen3-8B judge with the reference answer, or with a small rubric
when the condition has none (see ``vlens.judge.rubric``).  Every row gets a
``judge`` entry and the file is rewritten.

    python3 experiments/datasets/judge.py results/vlens/Qwen3-VL-8B-Instruct__*.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vlens.judge import rubric  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--judge", default="Qwen/Qwen3-8B")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=24)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    docs = {}
    for p in a.jsons:
        d = json.load(open(p))
        if not isinstance(d, dict) or "rows" not in d:
            say(f"skip {os.path.basename(p)}: not a behaviour file"); continue
        docs[p] = d
    rubric.judge_docs(docs, judge=a.judge, batch=a.batch, max_new=a.max_new, say=say)
    for p, d in docs.items():
        d["judge"] = a.judge
        json.dump(d, open(p, "w"), indent=1, default=float)
        c = Counter((r["condition"], (r.get("judge") or {}).get("verdict")) for r in d["rows"])
        say(f"\n{os.path.basename(p)}")
        for (cond, v), n in sorted(c.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            say(f"   {cond:<36} {str(v):<10} {n}")
    say("DONE judge")


if __name__ == "__main__":
    main()
