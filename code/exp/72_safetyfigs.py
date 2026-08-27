"""The three safety figures.

The safety section carried no figures at all: a jailbreak AUROC with its floor
in a table column, a compliance result stated only in prose, and the one
comparison this paper loses buried in a bullet list.  This draws all three from
result files that already exist.  Nothing here touches a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jvlm import figsafety

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="/workspace/vlm/results")
ap.add_argument("--out", default="/workspace/vlm/figures")
ap.add_argument("--jailbreak", default="jailbreak/qwen2b_v2_stats.json")
ap.add_argument("--pope", default="pope/qwen2b_3lens_stats.json")
ap.add_argument("--only", default=None, help="a,b,c")
a = ap.parse_args()

want = set((a.only or "a,b,c").split(","))


def load(rel):
    p = os.path.join(a.results, rel)
    if not os.path.exists(p):
        print(f"MISSING {p}")
        return None
    return json.load(open(p))


jb = load(a.jailbreak)
pope = load(a.pope)

if jb and "a" in want:
    print(figsafety.jailbreak(jb, a.out))
if jb and "b" in want:
    print(figsafety.benchmark(jb, a.out))
if pope and "c" in want:
    print(figsafety.hallucination(pope, a.out))
