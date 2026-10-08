# SPDX-License-Identifier: Apache-2.0
"""Which experiments have actually scored which source of pictures.

"Every dataset is used" is easy to believe and easy to be wrong about: an
experiment can run to completion having drawn its sample entirely from the one
source that has the most records. This reads the result files rather than the
queue, and reports the sources each experiment reached, so a gap shows up as a
gap rather than as a figure nobody looked at closely.

    python3 tools/dataset_usage.py
    python3 tools/dataset_usage.py --results results --missing-only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOURCES = ("openimages", "nuscenes", "vins", "gemini", "wmdp")
NICE = {"openimages": "Open Images", "nuscenes": "nuScenes", "vins": "VINS-120K",
        "gemini": "Generated", "wmdp": "WMDP"}
#: the password study's own halves, which every other experiment is meant to miss
LOCKED = re.compile(r"wmdp[_ ]pw[_ ](fine|test)")


def sources_in(o, into):
    """Every source named anywhere in a result, by the family keys it carries."""
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(k, str):
                if LOCKED.search(k):
                    into.add("LOCKED")
                    continue
                head = k.split("_")[0].split("|")[0]
                if head in SOURCES:
                    into.add(head)
            sources_in(v, into)
    elif isinstance(o, list):
        for v in o:
            sources_in(v, into)
    elif isinstance(o, str):
        if LOCKED.search(o):
            into.add("LOCKED")
        elif o in SOURCES:
            into.add(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--missing-only", action="store_true")
    a = ap.parse_args()

    from vlens.data import simonsays as SS
    pool = SS.scorable(SS.load(require_image=True))
    have = {}
    for r in pool:
        have[str(r["family"]).split("_")[0]] = have.get(str(r["family"]).split("_")[0], 0) + 1
    print("the scored pool, by source")
    for s in SOURCES:
        print(f"  {NICE[s]:<14}{have.get(s, 0):>6}")
    print(f"  {'total':<14}{len(pool):>6}\n")

    by_exp = {}
    for p in sorted(glob.glob(f"{a.results}/*/*.json")):
        if os.path.basename(p).startswith("census"):
            continue
        exp = os.path.basename(os.path.dirname(p))
        model = os.path.basename(p).split("__")[0]
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        got = set()
        sources_in(d, got)
        if got:
            by_exp.setdefault((exp, model), set()).update(got)

    width = max(len(e) + len(m) for e, m in by_exp) + 3 if by_exp else 20
    print(f"{'experiment':<{width}}" + "".join(f"{NICE[s]:>13}" for s in SOURCES) + "   leaked")
    reached = {s: 0 for s in SOURCES}
    for (exp, model), got in sorted(by_exp.items()):
        if a.missing_only and all(s in got for s in SOURCES):
            continue
        for s in got & set(SOURCES):
            reached[s] += 1
        print(f"{exp + ' / ' + model:<{width}}"
              + "".join(f"{('yes' if s in got else '.'):>13}" for s in SOURCES)
              + f"{('PASSWORD' if 'LOCKED' in got else ''):>9}")
    print()
    for s in SOURCES:
        print(f"  {NICE[s]:<14}reached by {reached[s]} of {len(by_exp)} result files")
    leaks = [f"{e}/{m}" for (e, m), g in by_exp.items() if "LOCKED" in g]
    if leaks:
        print("\n  the password study's own families still appear in:")
        for x in sorted(leaks):
            print(f"    {x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
