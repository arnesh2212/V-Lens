# SPDX-License-Identifier: Apache-2.0
"""Count what the benchmark is made of, once, into a file a figure can read.

The figure that draws the benchmark's composition should not need the images
on the machine that draws it, and the counts should come from the loader
rather than from anyone's memory of them.

    python3 tools/dataset_census.py --out results/dataset/census.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/dataset/census.json")
    a = ap.parse_args()
    from vlens.data import simonsays as SS

    recs = SS.load(require_image=True)
    #: ours or somebody else's, which is the first thing a reader asks
    WHOSE = {"gemini": "ours", "openimages": "public", "nuscenes": "public",
             "vins": "public", "wmdp": "public"}
    out = dict(n=len(recs), families={}, by_source={}, by_probe={}, by_kind={},
               sandbagging=list(SS.SANDBAGGING))
    for r in recs:
        fam = str(r["family"])
        src, probe = fam.split("_")[0], fam.split("_")[-1]
        e = out["families"].setdefault(fam, dict(n=0, source=src, probe=probe,
                                                 whose=WHOSE.get(src, "public"),
                                                 variant="edit" if "_edit_" in fam else "orig",
                                                 sandbagging=fam in SS.SANDBAGGING,
                                                 kinds={}, subcategories={}))
        e["n"] += 1
        e["kinds"][r["kind"]] = e["kinds"].get(r["kind"], 0) + 1
        s_ = str(r.get("subcategory"))
        e["subcategories"][s_] = e["subcategories"].get(s_, 0) + 1
        out["by_source"][src] = out["by_source"].get(src, 0) + 1
        out["by_probe"][probe] = out["by_probe"].get(probe, 0) + 1
        out["by_kind"][r["kind"]] = out["by_kind"].get(r["kind"], 0) + 1
    out["scorable"] = sum(1 for r in recs
                          if not SS.is_sandbagging(r) and r["kind"] in ("yesno", "test_real"))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{out['n']} records, {len(out['families'])} families -> {a.out}")
    for k, v in sorted(out["by_source"].items(), key=lambda t: -t[1]):
        print(f"  {k:<14}{v:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
