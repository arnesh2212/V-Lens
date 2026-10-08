"""Render the three depth figures, and print every number their captions need.

Nothing is measured here: the layerwise causal run already covered every region
at every depth.  This reads that table three ways and prints, alongside each
figure, the statistics the caption has to carry -- thresholds, sample sizes,
stability of the retirement map, the absorption fall, and the say/use lag.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats as sps

from jvlm import figwhen, style

ap = argparse.ArgumentParser()
ap.add_argument("--cards",
                default="results/causalmap/qwen2b_layerwise_cards.json")
ap.add_argument("--two", default="results/absorb/qwen2b_twoq_cards.json")
ap.add_argument("--twostats", default="results/absorb/qwen2b_twoq_stats.json")
ap.add_argument("--out", default="figures")
ap.add_argument("--names", nargs=3,
                default=["f110_whenmap.png", "f111_absorption.png",
                         "f112_lag.png"])
ap.add_argument("--thetas", type=float, nargs=2, default=[0.03, 0.06])
a = ap.parse_args()
style.use()

cards = json.load(open(a.cards))
by = figwhen.group(cards)
print(f"{len(cards)} cards, {len(by)} photographs, "
      f"lenses {sorted(cards[0]['lenses'])}", flush=True)

# ------------------------------------------------------------------ A: when
p = figwhen.when_map(cards, a.out, name=a.names[0],
                     thetas=tuple(a.thetas))
print("wrote", p)

rho, moved, never = [], [], []
for key, v in sorted(by.items()):
    D, T, ls = figwhen._stack(v)
    r1 = figwhen.retirement(T, D, a.thetas[0])
    r2 = figwhen.retirement(T, D, a.thetas[1])
    f1 = np.where(np.isfinite(r1), r1, 0.0)
    f2 = np.where(np.isfinite(r2), r2, 0.0)
    if f1.std() > 0 and f2.std() > 0:
        rho.append(sps.spearmanr(f1, f2).statistic)
    moved.append(float(np.mean(np.abs(f1 - f2) > (D[1] - D[0]) * 1.5)))
    never.append(float(np.mean(~np.isfinite(r1))))
    bx = [j for j, c in enumerate(v[ls[0]]["keep"])
          if c in set(v[ls[0]]["inside"])]
    bm = r1[bx]
    bm = float(np.nanmedian(bm)) if np.isfinite(bm).any() else float("nan")
    print(f"  {key} retire(theta={a.thetas[0]}) median "
          f"{np.nanmedian(r1) if np.isfinite(r1).any() else float('nan'):.3f}"
          f"  never {never[-1]:.2f}  box-median {bm:.3f}")
print(f"WHEN  n_photographs={len(by)}  regions={T.shape[1]}  "
      f"layers={ls}  depths={np.round(D,3).tolist()}")
print(f"WHEN  spearman(theta1,theta2) mean {np.mean(rho):.3f} "
      f"min {np.min(rho):.3f}  moved>1.5 layers {np.mean(moved):.3f}  "
      f"never-consulted share {np.mean(never):.3f}")

# ------------------------------------------------------- B: absorption curve
two = None
if os.path.exists(a.two):
    tc = json.load(open(a.two))
    two = dict(cards=tc, depths=tc[0]["depths"])
    print(f"two-question run: {len(tc)} photographs", flush=True)
else:
    print(f"two-question run PENDING ({a.two} absent)", flush=True)
p = figwhen.absorption(cards, a.out, name=a.names[1], two=two)
print("wrote", p)

tot, spec, unif = [], [], []
for key, v in sorted(by.items()):
    D, T, _ = figwhen._stack(v)
    x, y, z = figwhen.place_specific(T)
    tot.append(x); spec.append(y); unif.append(z)
tot, spec, unif = np.array(tot), np.array(spec), np.array(unif)
print(f"ABS  total mean  {np.round(tot.mean(0),2).tolist()}")
print(f"ABS  specific    {np.round(spec.mean(0),2).tolist()}")
print(f"ABS  uniform     {np.round(unif.mean(0),2).tolist()}")
print(f"ABS  specific first {spec.mean(0)[0]:.2f} last {spec.mean(0)[-1]:.2f} "
      f"ratio {spec.mean(0)[-1]/spec.mean(0)[0]:.2f}; "
      f"falls on {(spec[:, -1] < spec[:, 0]).sum()}/{len(spec)} photographs; "
      f"total {tot.mean(0)[0]:.2f} -> {tot.mean(0)[-1]:.2f}")
if two:
    ce = [figwhen.centroid(figwhen.place_specific(
        np.asarray(r["questions"]["easy"]["truth"]))[1], np.array(two["depths"]))
        for r in two["cards"]]
    ch = [figwhen.centroid(figwhen.place_specific(
        np.asarray(r["questions"]["hard"]["truth"]))[1], np.array(two["depths"]))
        for r in two["cards"]]
    d = np.array(ch) - np.array(ce)
    print(f"ABS2 easy centroid {np.mean(ce):.3f}  hard {np.mean(ch):.3f}  "
          f"shift {d.mean():+.3f}  deeper on {(d>0).sum()}/{len(d)}  "
          f"wilcoxon p {sps.wilcoxon(ch, ce).pvalue:.4f}")
    if os.path.exists(a.twostats):
        s = json.load(open(a.twostats))
        print("ABS2 stats:", {k: s[k] for k in
                              ("easy_prompt", "hard_prompt", "regions",
                               "mean_images", "n_layers",
                               "image_activation_relative_difference")
                              if k in s})
        print("ABS2 pairs:", s.get("pairs"))

# ------------------------------------------------------------------- C: lag
LENSES = tuple(n for n in ("logit lens", "J-lens", "VJ-Lens")
               if n in cards[0]["lenses"])
rows = figwhen.say_use(cards, lenses=LENSES)
p = figwhen.lag(rows, a.out, name=a.names[2], lenses=LENSES)
print("wrote", p)
D = rows[0]["depths"]
du = np.array([figwhen.centroid(r["use"], D) for r in rows])
print(f"LAG  consultation centroid mean {np.nanmean(du):.3f} "
      f"range {np.nanmin(du):.3f}-{np.nanmax(du):.3f}")
for ln in ("logit lens", "J-lens", "VJ-Lens", "VJ-Lens (point)"):
    if ln not in rows[0]["name"]:
        continue
    ds = np.array([figwhen.centroid(r["name"][ln], D) for r in rows])
    ok = np.isfinite(ds) & np.isfinite(du)
    v = ds[ok] - du[ok]
    print(f"LAG  {ln:17s} say {np.nanmean(ds[ok]):.3f}  lag {v.mean():+.3f} "
          f"(sd {v.std(ddof=1):.3f})  positive {int((v>0).sum())}/{len(v)}  "
          f"n={int(ok.sum())}  vals {np.round(v,3).tolist()}")
u = np.mean([r["use"] / max(r["use"].max(), 1e-9) for r in rows], 0)
print(f"LAG  pooled consulted {np.round(u/u.max(),3).tolist()}")
for ln in LENSES:
    m = np.nanmean([r["name"][ln] for r in rows], 0)
    print(f"LAG  pooled {ln:12s} raw {np.round(m,3).tolist()}")
print(f"LAG  box regions per photograph "
      f"{[r['n_box'] for r in rows]}")
print("done", flush=True)
