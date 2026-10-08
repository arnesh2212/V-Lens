"""Draw the funding census, and print the numbers its caption has to carry.

The figure and the caption are built from the same stats file in the same run,
so a number cannot be right in one and stale in the other.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "code")

import numpy as np
import pandas as pd

from jvlm import figcensus

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="results/census")
ap.add_argument("--tag", default="qwen2b")
ap.add_argument("--out", default="figures")
ap.add_argument("--name", default="f88_census.png")
a = ap.parse_args()

df = pd.read_parquet(f"{a.res}/{a.tag}_raw.parquet")
cards = json.load(open(f"{a.res}/{a.tag}_cards.json"))
S = json.load(open(f"{a.res}/{a.tag}_stats.json"))
for c in cards:
    if not os.path.exists(c["path"]):
        c["path"] = f"/workspace/data/val2017/{int(c['image_id']):012d}.jpg"
print(figcensus.census(df, cards, S, a.out, a.name))

cl, cen = S["closure"], S["census"]
h = df[df.hallucinated == 1]
c = df[df.hallucinated == 0]
R = cen["hallucinations"]["reading"]
A = cen["hallucinations"]["ablation"]
D = cen["hallucinations"]["ablation_donor"]
prof = {r["layer"]: r for r in S["depth_profile"]}


def pc(x):
    return f"{100*x:.0f}%"


def wilson(k, n, z=1.96):
    """A proportion's 95% interval.  A census reports a share, and a share off
    a hundred-odd items needs its interval printed beside it."""
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


print("\n================ NUMBERS THE CAPTION MUST CARRY ================")
print(f"model              {S['repo']}, {S['n_layers']} layers, "
      f"{S['n_deepstack']} deepstack")
print(f"layer              {S['layer']}  (depth {S['depth']}), "
      f"path points {S['ig_steps']}, {S['P']} image tokens on "
      f"{S['grid']}x{S['grid']}, pooled to {S['coarse']}x{S['coarse']}")
print(f"questions          {S['n_questions']} absent-object questions over "
      f"{S['n_images']} photographs")
print(f"hallucinations     {S['n_hallucinated']} "
      f"({100*S['n_hallucinated']/S['n_questions']:.1f}%)  "
      f"correct refusals {len(c)}")
print(f"yes-rate by split  " + "  ".join(f"{k} {pc(v)}"
                                         for k, v in cen["yes_rate"].items()))
print("\nCLOSURE ON THESE ITEMS")
print(f"  r(sum, ablation)          {cl['corr_margin']:.3f}")
print(f"  sign agreement            {cl['sign_agree']:.3f}  "
      f"(above tau: {cl.get('sign_agree_above_tau', float('nan')):.3f} on "
      f"{cl.get('n_above_tau','-')} items)")
print(f"  relative error            {cl['rel_error_margin']:.3f}")
print(f"  median |residual|         {cl['median_abs_residual']:.3f} logits "
      f"(this is tau, the grey band)")
print(f"  mean |ablation|           {cl['mean_abs_truth']:.3f} logits")
print(f"  vocabulary r (n={cl.get('vocab_n_items','-')}, "
      f"k={cl.get('vocab_k_words','-')})   "
      f"mean {cl.get('vocab_corr_mean', float('nan')):.3f}  "
      f"median {cl.get('vocab_corr_median', float('nan')):.3f}")
print(f"  greedy answer is yes/no   {cl['greedy_is_yes_or_no']:.3f}")
k_out = int((h.total <= 0).sum())
lo, hi = wilson(k_out, len(h))
print("\nTHE CENSUS")
print(f"  outvoted share    {k_out}/{len(h)} = {pc(k_out/len(h))}  "
      f"95% CI [{pc(lo)}, {pc(hi)}]")
print(f"  reading           misperception {pc(R['funded'])}  "
      f"outvoted {pc(R['outvoted'])}  median {R['median']:+.2f}")
print(f"  ablation          misperception {pc(A['funded'])}  "
      f"outvoted {pc(A['outvoted'])}  median {A['median']:+.2f}")
print(f"  ablation (donor)  misperception {pc(D['funded'])}  "
      f"outvoted {pc(D['outvoted'])}  median {D['median']:+.2f}")
print(f"  three-way (tau={cen['tau']:.2f})  clearly funded "
      f"{pc(R['funded_clear'])}  within tau {pc(R['near_zero'])}  "
      f"clearly against {pc(R['against_clear'])}")
print(f"  correct refusals  positive {pc(cen['correct_refusals']['reading']['funded'])}"
      f"  median {cen['correct_refusals']['reading']['median']:+.2f}")
print("  by negative flavour: " + "  ".join(
    f"{k} {pc(v['funded'])} funded (n={v['n']})"
    for k, v in cen["by_split"].items()))


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    v = np.concatenate([pos, neg]); o = v.argsort()
    r = np.empty(len(v), float); r[o] = np.arange(len(v))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2)
                 / (len(pos) * len(neg)))


# How loudly the picture speaks, as opposed to which way it votes.  A wrong
# answer where the picture contributed nothing at all is neither a
# misperception nor an overruled perception: it is an abstention, and the
# magnitudes are what tell the three apart.
print("\nHOW LOUD IS THE PICTURE?  (|contribution|, logits)")
print(f"  hallucinations    median {h.total.abs().median():.2f}  "
      f"mean {h.total.abs().mean():.2f}  "
      f"signed median {h.total.median():+.2f}")
print(f"  correct refusals  median {c.total.abs().median():.2f}  "
      f"mean {c.total.abs().mean():.2f}  "
      f"signed median {c.total.median():+.2f}")
print(f"  ratio of medians  {c.total.abs().median()/max(h.total.abs().median(),1e-9):.1f}x")
print(f"  AUROC, signed contribution separates the two   "
      f"{auc(h.total, c.total):.3f}")
print(f"  AUROC, quiet picture predicts the error        "
      f"{auc(-h.total.abs(), -c.total.abs()):.3f}")
print(f"  within tau={cen['tau']:.2f}: hallucinations "
      f"{pc((h.total.abs() <= cen['tau']).mean())}  "
      f"correct refusals {pc((c.total.abs() <= cen['tau']).mean())}")
B = cl.get("baselines") or {
    "sd_clean_margin": df.conf.std(),
    "sd_across_items_mean_baseline": df.base_margin.std(),
    "sd_across_items_donor_baseline": df.base_donor.std(),
    "sd_within_item_across_donors": df.base_donor_sd.mean(),
    "corr_donor_vs_mean_baseline": np.corrcoef(df.base_donor, df.base_margin)[0, 1],
    "corr_donor_baseline_vs_conf": np.corrcoef(df.base_donor, df.conf)[0, 1],
    "corr_mean_baseline_vs_conf": np.corrcoef(df.base_margin, df.conf)[0, 1],
}
print("\nBASELINES")
print(f"  clean margin        {cl['clean_margin_mean']:+.2f} logits "
      f"(sd {B['sd_clean_margin']:.2f})")
print(f"  corpus-mean image   {cl['base_margin_mean']:+.2f} "
      f"(sd across items {B['sd_across_items_mean_baseline']:.2f})")
print(f"  donor photographs   {cl['base_margin_donor']:+.2f} "
      f"(sd across items {B['sd_across_items_donor_baseline']:.2f}, "
      f"within item across 3 donors {B['sd_within_item_across_donors']:.2f})")
print(f"  r(donor base, mean base) {B['corr_donor_vs_mean_baseline']:+.3f}   "
      f"r(donor base, clean margin) {B['corr_donor_baseline_vs_conf']:+.3f}   "
      f"r(mean base, clean margin) {B['corr_mean_baseline_vs_conf']:+.3f}")
if S.get("predicts"):
    P_ = S["predicts"]
    print("\nWHAT THE PRIOR ALONE ALREADY KNOWS")
    print(f"  prior-only margin, hallucinated items  "
          f"{P_['prior_margin_hallucinated']:+.2f} logits")
    print(f"  prior-only margin, correct refusals    "
          f"{P_['prior_margin_refused']:+.2f} logits")
    print(f"  AUROC, prior-only margin               "
          f"{P_['auroc_prior_margin']:.3f}")
    print(f"  AUROC, image contribution              "
          f"{P_['auroc_image_contribution']:.3f}")
    print(f"  AUROC, model's own margin (degenerate) "
          f"{P_['auroc_own_margin_degenerate']:.3f}")
print("\nWHY THIS LAYER (mean |change in the margin| when the picture goes)")
for r in S["depth_profile"]:
    print(f"  layer {r['layer']:>2}  depth {r['depth']:.2f}  "
          f"{r['mean_abs_delta']:.2f} logits   retained {r['retained']:.3f}")
if S.get("by_depth"):
    print("\nCENSUS AT OTHER DEPTHS (subsample)")
    for r in S["by_depth"]:
        print(f"  layer {r['layer']:>2}  depth {r['depth']:.2f}  n={r['n']}  "
              f"funded {r['funded']:.3f}  r {r['corr']:.3f}  "
              f"sign {r['sign_agree']:.3f}  |ablation| {r['mean_abs_truth']:.2f}")
print("\nTHE TWO WORKED CASES")
for rec, nm in figcensus._extremes(cards, cen["tau"]):
    if rec is None:
        print(f"  {nm or 'second case'}: none available")
        continue
    print(f"  {nm}: image {rec['image_id']}, \"is there a {rec['obj']}?\", "
          f"{rec['split']} split, said {rec['said']}, margin {rec['conf']:+.2f}, "
          f"regions sum {rec['total']:+.2f}, ablation {rec['truth']:+.2f}, "
          f"donor ablation {rec['truth_donor']:+.2f}, "
          f"prior-only margin {rec['base_margin']:+.2f}, "
          f"strongest region {max(rec['tiles']):+.2f} / "
          f"{min(rec['tiles']):+.2f}")
lo_, hi_ = figcensus.xlimits(df)
print(f"\nscatter x-axis {lo_:.1f} .. {hi_:.1f}; "
      f"{figcensus.off_axis(df)} of {len(df)} questions fall outside it "
      f"(all correct refusals: {figcensus.off_axis(df) - int(((h.total<lo_)|(h.total>hi_)).sum())} "
      f"refusals, {int(((h.total<lo_)|(h.total>hi_)).sum())} wrong answers); "
      f"full data range {df.total.min():.1f} .. {df.total.max():.1f}; "
      f"hallucination margins {h.conf.min():.2f} .. {h.conf.max():.2f}")
print(f"minutes {S['minutes']}")
