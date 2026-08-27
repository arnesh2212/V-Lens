"""Draw the completeness theorem, the dose curves and the efficiency plane.

Pure plotting -- no model, no GPU -- so the drawings can be looked at and fixed
without paying for the measurement again.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jvlm import figatlas

ap = argparse.ArgumentParser()
ap.add_argument("--complete", default="/workspace/vlm/results/complete/qwen2b_stats.json")
ap.add_argument("--edits-dir", default="/workspace/vlm/results/edits")
ap.add_argument("--edit-tags", nargs="*",
                default=["smoke", "smoke2", "qwen2b_plant", "qwen2b"],
                help="in the same order build_atlas.py merges them, so the\n"
                     "                     figure and the table can only ever "
                     "agree")
ap.add_argument("--out", default="/workspace/vlm/figures")
ap.add_argument("--only", nargs="*", default=None)
a = ap.parse_args()

want = set(a.only or ["complete", "doses", "efficiency"])

if "complete" in want and a.complete and os.path.exists(a.complete):
    d = json.load(open(a.complete))
    print("wrote", figatlas.completeness(d, a.out), flush=True)


def merge_edits(tags):
    """One operation at a time, take the run that measured it at the larger n.

    ``build_atlas.py`` assembles the taxonomy table exactly this way, because
    the five operations were not all run together.  The figures have to use the
    same rule on the same files or the panel and the row beside it would report
    different numbers for the same operation, which is the one failure a reader
    never forgives.
    """
    out, top, by, op_n, src = None, {}, {}, {}, {}
    for t in tags:
        p = os.path.join(a.edits_dir, f"{t}_stats.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        n = d.get("n_trials") or 0
        for op, v in (d.get("top_dose") or {}).items():
            # oldest tag first and >= on the tie, which is exactly the rule
            # build_atlas.py uses: an equal-n tie goes to the LATER run,
            # because the earlier one predates corrections to the operations
            # themselves.  Written the same way in both places on purpose.
            if op not in op_n or n >= op_n[op]:
                op_n[op], top[op], src[op] = n, v, t
        out = d if out is None or n > (out.get("n_trials") or 0) else out
    if out is None:
        return None
    # and the dose curves come from whichever file won that operation
    for t in tags:
        p = os.path.join(a.edits_dir, f"{t}_stats.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for k, v in (d.get("by_op") or {}).items():
            if src.get(k.split("|")[0]) == t:
                by[k] = v
    out = dict(out)
    out["by_op"], out["top_dose"], out["op_n"], out["op_src"] = by, top, op_n, src
    return out


if want & {"doses", "efficiency"}:
    d = merge_edits(a.edit_tags)
    if d is None:
        print("no edit runs on disk", flush=True)
    else:
        for op in sorted(d["op_n"]):
            print(f"  {op:11s} n={d['op_n'][op]:<4} from {d['op_src'][op]}",
                  flush=True)
        if "doses" in want:
            print("wrote", figatlas.doses(d, a.out), flush=True)
        if "efficiency" in want:
            print("wrote", figatlas.efficiency(d, a.out), flush=True)
print("done", flush=True)
