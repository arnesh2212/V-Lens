# SPDX-License-Identifier: Apache-2.0
"""Is V-Read actually ahead, or is it ahead on a handful of families?

Twenty-nine families is twenty-nine chances to look good, so the comparison is
made paired and family-wise: for each family, the difference between V-Read's
AUROC and a baseline's at a matched depth, then a signed-rank test over
families, with a Holm correction across the baselines being compared, and a
bootstrap interval on the mean difference. A per-family win count is reported
alongside, because a significant mean with eleven wins and eighteen losses
would mean something quite different from a significant mean with twenty-six
wins.

    python3 tools/stats.py --results results --depth 0.25
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

BASELINES = ["tuned", "anchor", "jacobian", "latent", "logit", "probe"]


def signed_rank(d: np.ndarray):
    """Wilcoxon signed-rank on the paired differences, exact where scipy has it."""
    d = np.asarray([x for x in d if np.isfinite(x) and x != 0.0], float)
    if len(d) < 6:
        return float("nan"), len(d)
    try:
        from scipy.stats import wilcoxon
        # two sided
        return float(wilcoxon(d).pvalue), len(d)
    except Exception:  # noqa: BLE001
        # a permutation test on the signs, which is what the signed-rank test is
        rng = np.random.default_rng(0)
        r = np.argsort(np.argsort(np.abs(d))) + 1.0
        obs = float((r * np.sign(d)).sum())
        null = np.array([float((r * rng.choice([-1.0, 1.0], len(d))).sum()) for _ in range(20000)])
        return float((np.abs(null) >= abs(obs)).mean()), len(d)


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni: control the family-wise error over the baselines."""
    items = sorted([(p, k) for k, p in pvals.items() if np.isfinite(p)])
    m = len(items)
    out, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = adj
    for k, p in pvals.items():
        out.setdefault(k, float("nan"))
    return out


def boot_mean(d, reps=10000, seed=0):
    d = np.asarray([x for x in d if np.isfinite(x)], float)
    if len(d) < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    v = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(reps)]
    return float(d.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def at(depths, vals, x):
    return vals[int(np.argmin(np.abs(np.array(depths, float) - x)))]


def collect(path, depth, target):
    """Per-family AUROC per instrument, from a bench or a cross-fitted summary."""
    with open(path) as f:
        d = json.load(f)
    fams = d.get("families") or {}
    out = {}
    for fam, r in fams.items():
        row = {}
        for k in ["v_read", *BASELINES]:
            key = f"{target}|{k}"
            if key in r.get("auroc", {}):
                v = at(r["depths"], r["auroc"][key], depth)
                if np.isfinite(v):
                    row[k] = v
        if "v_read" in row and len(row) > 1:
            out[fam] = row
    return out


def report(name, per_family, say):
    if not per_family:
        return None
    fams = sorted(per_family)
    say(f"\n  {name}: {len(fams)} families")
    diffs, res = {}, {}
    for b in BASELINES:
        d = np.array([per_family[f]["v_read"] - per_family[f][b]
                      for f in fams if b in per_family[f]], float)
        if len(d) < 3:
            continue
        diffs[b] = d
    ps = {b: signed_rank(d)[0] for b, d in diffs.items()}
    adj = holm(ps)
    say("    baseline      mean V-Read advantage        wins   p (Holm)")
    for b in BASELINES:
        if b not in diffs:
            continue
        m, lo, hi = boot_mean(diffs[b])
        w = int((diffs[b] > 0).sum())
        res[b] = dict(mean=m, lo=lo, hi=hi, wins=w, n=len(diffs[b]),
                      p_raw=ps[b], p_holm=adj.get(b))
        star = "" if not np.isfinite(adj.get(b, np.nan)) else (
            "  significant" if adj[b] < 0.05 else "")
        say(f"    {b:<12} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]   {w:>3}/{len(diffs[b]):<3} "
            f"{adj.get(b, float('nan')):.4f}{star}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--depths", type=float, nargs="*", default=[0.0, 0.25, 0.5])
    ap.add_argument("--target", default="model", choices=["model", "data"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    say = print
    all_out, raw_p = {}, {}
    say("Paired comparison across families. Positive means V-Read is ahead.")
    say("A two-sided signed-rank test over families, Holm corrected across every")
    say("baseline and every depth at once, with a bootstrap interval on the mean.")
    say("Families share images and framings, so they are not independent draws;")
    say("the test treats them as such and the interval is the honest part.")
    for pat in (f"{a.results}/lensbench*/*__bench.json",
                f"{a.results}/lensread*/*__read_summary.json"):
        for p in sorted(glob.glob(pat)):
            tag = f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p).split('__')[0]}"
            for depth in a.depths:
                pf = collect(p, depth, a.target)
                r = report(f"{tag} at depth {depth:.2f}", pf, say)
                if r:
                    all_out[f"{tag}|{depth}"] = r
                    for b, v in r.items():
                        raw_p[f"{tag}|{depth}|{b}"] = v["p_raw"]
    # Holm over every baseline and depth together
    if raw_p:
        adj = holm(raw_p)
        say("\n  after a Holm correction over every baseline and depth together:")
        for k in sorted(adj, key=lambda k: adj[k]):
            if np.isfinite(adj[k]):
                say(f"    {k:<46} p = {adj[k]:.4f}" + ("   significant" if adj[k] < 0.05 else ""))
        all_out["holm_over_everything"] = adj
    if a.out:
        with open(a.out, "w") as f:
            json.dump(all_out, f, indent=1, default=float)
        say(f"\nwrote {a.out}")
    if not all_out:
        say("\n  nothing to compare yet")


if __name__ == "__main__":
    sys.exit(main())
