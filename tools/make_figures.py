# SPDX-License-Identifier: Apache-2.0
"""Draw every figure that has a result behind it.

Safe to run at any moment: a figure whose experiment has not finished is
skipped and reported as missing, so this can sit in a loop while the queue
works.

    python3 tools/make_figures.py --results results --out figures
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
    ap.add_argument("--model", default=None, help="restrict to one model's files")
    a = ap.parse_args()
    from vlens.figures import paper
    made = paper.all_figures(results=a.results, out=a.out, model=a.model)
    for k, v in made.items():
        print(f"  {k:<22} {v if v else 'no result yet'}", flush=True)
    n = sum(1 for v in made.values() if v and not str(v).startswith("failed"))
    print(f"{n}/{len(made)} figures drawn")


if __name__ == "__main__":
    main()
