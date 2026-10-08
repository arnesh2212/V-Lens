# SPDX-License-Identifier: Apache-2.0
"""Redraw every figure that has data behind it, and say which do not yet.

Run as often as you like: the figures are drawn from saved results, cost
seconds, and a figure that will not draw is usually the first sign that a
result is malformed.
"""
from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--model", default=None, help="draw only this model")
    ap.add_argument("--only-averaged", action="store_true")
    ap.add_argument("--only-per-model", action="store_true")
    a = ap.parse_args()

    import glob
    import os
    from vlens.figures import paper

    def once(model=None, average=False):
        made = paper.all_figures(results=a.results, out=a.out, model=model, average=average)
        drawn = [k for k, v in made.items() if v and not str(v).startswith("failed")]
        waiting = [k for k, v in made.items() if not v]
        broken = {k: v for k, v in made.items() if v and str(v).startswith("failed")}
        who = model or ("every model pooled" if average else "newest")
        print(f"{who}: {len(drawn)} drawn" + (f", {len(waiting)} waiting" if waiting else ""))
        for k, v in broken.items():
            print(f"  BROKEN {k}: {v}")
        return len(broken)

    if a.model:
        return 1 if once(a.model) else 0
    models = sorted({os.path.basename(f).split("__")[0]
                     for f in glob.glob(f"{a.results}/*/*__*.json")
                     if "rows" not in os.path.basename(f)})
    bad = 0
    # the pooled set first, because it is the one the paper leads with
    if not a.only_per_model:
        bad += once(average=True)
    if not a.only_averaged:
        for mo in models:
            bad += once(mo)
    print(f"{len(models)} model(s): {', '.join(models)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
