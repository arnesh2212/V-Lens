"""Render the pointer plate and print every number its caption needs.

Nothing is measured here.  ``80_pointer.py`` covered the landmark, the
referent, a third object and a patch of bare floor at every depth, under two
questions about the same render and against two baselines; this reads that
table, draws it, and prints the statistics the caption has to carry -- the
peaks, the collapse, the crossing, how far each lens tracks the intervention,
and what the wrong baseline would have cost.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from jvlm import figpointer as fp

ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="/workspace/vlm/results/pointer/qwen2b_near_rows.parquet")
ap.add_argument("--items", default=None)
ap.add_argument("--stats", default=None)
ap.add_argument("--out", default="/workspace/vlm/figures")
ap.add_argument("--name", default="f120_pointer.png")
ap.add_argument("--how", default="sum", choices=["sum", "mean"])
ap.add_argument("--example", type=int, nargs="*", default=None,
                help="indices into the items file; default is the two scenes\n"
                     "whose three regions are largest")
a = ap.parse_args()

df = pd.read_parquet(a.rows)
items = json.load(open(a.items or a.rows.replace("_rows.parquet",
                                                 "_items.json")))
sp = a.stats or a.rows.replace("_rows.parquet", "_stats.json")
stats = json.load(open(sp)) if os.path.exists(sp) else {}

cl = df[df.baseline == "clevr"]
print(f"{cl.scene.nunique()} scenes, layers "
      f"{sorted(int(x) for x in cl.layer.unique())}", flush=True)

ex = a.example
if not ex:
    def score(it):
        n = it.get("n_tok", {})
        c = {r: (it[f"box_{r}"][0] + it[f"box_{r}"][2] / 2,
                 it[f"box_{r}"][1] + it[f"box_{r}"][3] / 2)
             for r in ("landmark", "referent", "distractor")}
        sep = min(float(np.hypot(c[a][0] - c[b][0], c[a][1] - c[b][1]))
                  for a in c for b in c if a < b)
        return float(np.mean([n.get("landmark", 0), n.get("referent", 0),
                              n.get("distractor", 0)])) * min(sep, 0.25)
    ex = sorted(range(len(items)), key=lambda i: -score(items[i]))[:2]
for e in ex:
    print(f"example scene {items[e]['scene_index']}: "
          f"{items[e]['question_main']} -> {items[e]['answer_main']}   |   "
          f"{items[e]['question_control']} -> {items[e]['answer_control']}",
          flush=True)

p, fits, theta = fp.pointer(cl, items, a.out, name=a.name, how=a.how, example=ex)
print("wrote", p, f"  theta={theta:.4f}", flush=True)

rep = fp.report(cl, items)
print("\nPOINTER  " + json.dumps(rep, indent=2), flush=True)

for q in ("main", "control"):
    x, m, se = fp.curve(cl, q, "landmark")
    _, r, _ = fp.curve(cl, q, "referent")
    _, d, _ = fp.curve(cl, q, "distractor")
    _, f, _ = fp.curve(cl, q, "floor")
    t = pd.DataFrame(dict(layer=x.astype(int), landmark=m, referent=r,
                          third=d, floor=f)).set_index("layer")
    print(f"\n{q.upper()}  mean drop in the answer word (logits)\n",
          t.round(3).to_string(), flush=True)

if (df.baseline == "coco").any():
    sub = df[df.scene.isin(df[df.baseline == "coco"].scene.unique())]
    rowsb = []
    for src in ("clevr", "coco"):
        d = sub[sub.baseline == src]
        row = dict(baseline=src,
                   floor=float(d[d.role == "floor"]["drop"].abs().mean()),
                   landmark=float(d[d.role == "landmark"]["drop"].mean()),
                   referent=float(d[d.role == "referent"]["drop"].mean()),
                   whole_image=float(d["drop_all"].abs().mean()))
        for lens in ("logit lens", "J-lens", "VJ-Lens"):
            c = f"{lens}_{a.how}"
            v, t = d[c].values, d["drop"].values
            g = np.isfinite(v) & np.isfinite(t)
            row[lens] = (float(np.corrcoef(v[g], t[g])[0, 1])
                         if g.sum() > 8 else np.nan)
        rowsb.append(row)
    print("\nTHE BASELINE, PRICED  (same scenes, two baselines)\n",
          pd.DataFrame(rowsb).set_index("baseline").round(3).to_string(),
          flush=True)

if stats:
    print(f"\nrun: {stats.get('repo')}  {stats.get('dataset')}  "
          f"rel={stats.get('rel')}  candidates {stats.get('candidates')}  "
          f"accuracy main {stats.get('acc_main')} control "
          f"{stats.get('acc_control')}  minutes {stats.get('minutes')}",
          flush=True)
