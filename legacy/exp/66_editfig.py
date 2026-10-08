"""Draw the edit taxonomy.  Pure plotting -- no model, no GPU.

Separate from the run so the drawing can be looked at and fixed without paying
for the measurement again, which is the same split the causal-map cards use.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from jvlm import figcausal

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True, help="the run's _raw.parquet")
ap.add_argument("--stats", default=None)
ap.add_argument("--out", default="figures")
ap.add_argument("--name", default="f65_edits.png")
ap.add_argument("--ops", nargs="*", default=None)
ap.add_argument("--also-all", type=int, default=1,
                help="also draw the six-panel version carrying the swap's\n"
                     "                     second reading")
a = ap.parse_args()

df = pd.read_parquet(a.raw)
print(f"{len(df)} rows, {df.i.nunique()} trials, "
      f"operations {sorted(df.op.unique())}", flush=True)
print(df[df.arm != "clean"].groupby(["op", "arm", "dose"])
      .agg(n=("follow", "count"), follow=("follow", "mean"),
           disturb=("disturb", "mean")).round(3).to_string(), flush=True)
print("wrote", figcausal.edits(df, a.out, a.name,
                               ops=a.ops or figcausal.OP_FIVE), flush=True)
if a.also_all and a.ops is None and "swapdev" in set(df.op):
    print("wrote", figcausal.edits(df, a.out, a.name.replace(".png", "_all.png"),
                                   ops=figcausal.OP_ORDER), flush=True)
