# SPDX-License-Identifier: Apache-2.0
"""Render the four figures for the results that landed with no picture.

No model and no GPU: run it with CUDA_VISIBLE_DEVICES="".  Every number drawn
is read out of a result file that already exists.

    (a) results/pairing/qwen8b_big_stats.json + qwen8b_big_raw.parquet
    (b) results/steer/qwen8b_v2_stats.json
    (c) results/dissoc/qwen8bJ2_stats.json,
        results/natdissoc/qwen8b_stats.json, qwen2b_stats.json
    (d) results/baselines/qwen2b_stats.json
    (e) results/fair/qwen2b_stats.json

The pairing raw table is filtered to ``act_gap <= 1e-6`` before anything is
computed from it.  The whole point of the experiment is that the two prompts
share one image encoding, and 29 JailbreakV rows do not: their image
activations differ between the two requests, so a difference in a fixed
lens's reading there says nothing about whether the lens can see the request.
Dropping them takes JailbreakV from 79 pairs to 50 and takes the fixed
lenses' largest change over the surviving 364 pairs to exactly zero.

AUROC is recomputed here from the filtered table the same way the experiment
computes it (Mann-Whitney U over the attack readings against the benign
readings, normalised by n*m), and the script prints its own numbers against
the unfiltered ones in the stats file so the two can be compared by eye.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jvlm import fignew, style

R = "results"
ap = argparse.ArgumentParser()
ap.add_argument("--pair-stats", default=f"{R}/pairing/qwen8b_big_stats.json")
ap.add_argument("--pair-raw", default=f"{R}/pairing/qwen8b_big_raw.parquet")
ap.add_argument("--steer", default=f"{R}/steer/qwen8b_v2_stats.json")
ap.add_argument("--clevr", default=f"{R}/dissoc/qwen8bJ2_stats.json")
ap.add_argument("--nat8b", default=f"{R}/natdissoc/qwen8b_stats.json")
ap.add_argument("--nat2b", default=f"{R}/natdissoc/qwen2b_stats.json")
ap.add_argument("--baselines", default=f"{R}/baselines/qwen2b_stats.json")
ap.add_argument("--fair", default=f"{R}/fair/qwen2b_stats.json")
ap.add_argument("--out", default="figures")
ap.add_argument("--only", default="")
a = ap.parse_args()
want = set(a.only.split(",")) if a.only else None


def doing(k):
    return want is None or k in want


def mw(h, b):
    """AUROC as Mann-Whitney U / (n*m), ties at half, as the run computes it."""
    h = np.asarray(h, float)
    b = np.asarray(b, float)
    r = rankdata(np.concatenate([h, b]))
    n1, n2 = len(h), len(b)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


NICE = {"figstep": "FigStep", "jailbreakv": "JailbreakV",
        "mm_safetybench": "MM-SafetyBench", "vlguard": "VLGuard",
        "hades": "HADES"}
written = []


# ---------------------------------------------------------------- (a) pairing

if doing("pair"):
    st = json.load(open(a.pair_stats))
    df = pd.read_parquet(a.pair_raw)
    keep = df[df.act_gap <= 1e-6]
    print(f"[pair] {len(df)} pairs, {len(keep)} with a shared image encoding "
          f"(act_gap <= 1e-6); dropped {len(df) - len(keep)}", flush=True)
    for src, g in df.groupby("source"):
        k = int((g.act_gap <= 1e-6).sum())
        print(f"          {NICE[src]:<16s} {len(g):>3d} -> {k:>3d}", flush=True)

    lenses = [("VJ-Lens", "vj", style.BLUE),
              ("J-lens", "jlens", style.VIOLET),
              ("logit lens", "logit", style.ORANGE)]
    strips = []
    for lab, pre, c in lenses:
        h = keep[f"{pre}_h"].values
        b = keep[f"{pre}_b"].values
        sd = float(np.concatenate([h, b]).std(ddof=1))
        d = h - b
        strips.append((lab, d / sd, c))
        print(f"[pair] {lab:<11s} sd across images {sd:10.2f}   largest raw "
              f"change {np.abs(d).max():8.2f}   identical on "
              f"{int((d == 0).sum())}/{len(d)} pairs", flush=True)

    order = ["figstep", "jailbreakv", "mm_safetybench", "vlguard"]
    rows = []
    for s in order:
        g = keep[keep.source == s]
        v = mw(g.vj_h, g.vj_b)
        rows.append(dict(label=NICE[s], value=v, n=len(g), group="bench"))
        print(f"[pair] {NICE[s]:<16s} n={len(g):>3d} VJ {v:.4f}  "
              f"(unfiltered file {st['auroc'][s + '|VJ-Lens']:.4f})  "
              f"logit {mw(g.logit_h, g.logit_b):.4f}  "
              f"J {mw(g.jlens_h, g.jlens_b):.4f}", flush=True)
    p4 = keep[keep.source != "hades"]
    rows.append(dict(label="pooled over the four", value=mw(p4.vj_h, p4.vj_b),
                     n=len(p4), group="pooled"))
    gh = keep[keep.source == "hades"]
    rows.append(dict(label="HADES", value=mw(gh.vj_h, gh.vj_b), n=len(gh),
                     group="other", note="the one that inverts"))
    print(f"[pair] pooled over the four  n={len(p4)} VJ {rows[-2]['value']:.4f}"
          f"   logit {mw(p4.logit_h, p4.logit_b):.4f}"
          f"   J {mw(p4.jlens_h, p4.jlens_b):.4f}", flush=True)
    print(f"[pair] HADES n={len(gh)} VJ {rows[-1]['value']:.4f}", flush=True)

    fl = st["floors"]["VJ-Lens"]
    print(f"[pair] shuffled floor {fl['null_mean']:.4f}, 95th pct "
          f"{fl['null_p95']:.4f}", flush=True)
    written.append(fignew.pair_safety(
        strips, rows, a.out, null_mean=fl["null_mean"],
        null_p95=fl["null_p95"],
        foot=f"{len(keep)} of {len(df)} pairs; the {len(df) - len(keep)} "
             "whose image encodings differ between the two requests are "
             "excluded"))


# ----------------------------------------------------------------- (b) steer

if doing("steer"):
    st = json.load(open(a.steer))
    T = st["table"]
    alphas = st["alphas"]
    order = [("answer direction", style.BLUE),
             ("question-selective", style.AQUA),
             ("swap toward the foil", style.RED),
             ("random matched", style.MUTED)]
    series = []
    for lab, c in order:
        on = [T[f"{lab}|{al}"]["on"] for al in alphas]
        off = [T[f"{lab}|{al}"]["off"] for al in alphas]
        series.append((lab, c, alphas, on, off))
        print(f"[steer] {lab:<21s} on  " +
              " ".join(f"{v:+7.3f}" for v in on) + "   off  " +
              " ".join(f"{v:+7.3f}" for v in off), flush=True)
    nn = T[f"{order[0][0]}|{alphas[0]}"]["n"]
    written.append(fignew.bounded_steering(
        series, a.out,
        foot=f"n = {st['n']} items, {nn} edits per point, layer "
             f"{st['edit_layer']}"))


# ------------------------------------------------------------ (c) dissociation

if doing("dissoc"):
    def cell(rows, key, a_key="effect_a", b_key="effect_b",
             lo=("effect_a_lo", "effect_b_lo"), hi=("effect_a_hi",
                                                    "effect_b_hi")):
        r = next(x for x in rows if x["row"] == key)
        return ((r[a_key], r[lo[0]], r[hi[0]]),
                (r[b_key], r[lo[1]], r[hi[1]]))

    cl = json.load(open(a.clevr))["ablation_table"]
    n8 = json.load(open(a.nat8b))["ablation_table"]
    n2 = json.load(open(a.nat2b))["ablation_table"]
    NAT = dict(lo=("a_lo", "b_lo"), hi=("a_hi", "b_hi"))
    LAB = [("covering A's region", style.BLUE),
           ("covering B's region", style.ORANGE),
           ("a random region of the same size", style.MUTED)]

    def pack(rows, keys, **kw):
        out = []
        for (lab, c), k in zip(LAB, keys):
            a_, b_ = cell(rows, k, **kw)
            out.append((lab, c, a_, b_))
            print(f"[dissoc]   {lab:<34s} A {a_[0]:+6.2f} "
                  f"[{a_[1]:+6.2f},{a_[2]:+6.2f}]   B {b_[0]:+6.2f} "
                  f"[{b_[1]:+6.2f},{b_[2]:+6.2f}]", flush=True)
        return out

    panels = []
    print("[dissoc] CLEVR scenes, 8B", flush=True)
    panels.append(("CLEVR scenes, 8B",
                   pack(cl, ["VJ region for A", "VJ region for B",
                             "random matched"])))
    print("[dissoc] photographs, 8B", flush=True)
    panels.append(("photographs, 8B",
                   pack(n8, ["object region A", "object region B",
                             "[q_a pick] random matched"], **NAT)))
    print("[dissoc] photographs, 2B", flush=True)
    panels.append(("photographs, 2B",
                   pack(n2, ["object region A", "object region B",
                             "[q_a pick] random matched"], **NAT)))
    written.append(fignew.crossover(
        panels, a.out,
        foot="n = 24 items in every panel; bars are 95% intervals"))


# -------------------------------------------------- (d) + (e) the comparison

if doing("fair"):
    bs = json.load(open(a.baselines))
    NAME = {"VJ-Lens": "VJ-Lens", "wrong prompt": "wrong-prompt control",
            "gradient x activation": "gradient x activation",
            "J-lens": "J-lens", "gradient norm": "gradient norm",
            "logit lens": "logit lens", "random": "random"}
    COL = {"VJ-Lens": style.BLUE, "wrong prompt": style.RED,
           "J-lens": style.VIOLET, "logit lens": style.ORANGE,
           "gradient x activation": style.MUTED,
           "gradient norm": style.MUTED, "random": "#b9b7b0"}
    best = bs["best"]
    ranked = sorted(best.items(), key=lambda kv: -kv[1]["lift"])
    methods = [(NAME[k], v["lift"], COL[k]) for k, v in ranked]
    for k, v in ranked:
        print(f"[fair] {NAME[k]:<22s} {v['lift']:5.1f}x  top-1 {v['top1']:.3f}"
              f"  best rule {v['aggregation']}", flush=True)
    bi = [k for k, _ in ranked].index("VJ-Lens")
    wi = [k for k, _ in ranked].index("wrong prompt")

    fa = json.load(open(a.fair))
    q2 = [r for r in fa["q2_their_best"] if r["metric"] == "spearman"]
    sizes = sorted({r["size"] for r in q2})
    margins = []
    for prior, c in (("J-lens", style.VIOLET), ("logit lens", style.ORANGE),
                     ("gradient norm", style.MUTED)):
        rr = {r["size"]: r for r in q2 if r["prior"] == prior}
        d = [rr[s]["delta"] for s in sizes]
        lo = [rr[s]["ci"][0] for s in sizes]
        hi = [rr[s]["ci"][1] for s in sizes]
        margins.append((prior, c, sizes, d, lo, hi))
        print(f"[fair] margin over {prior:<15s} " +
              " ".join(f"{v:+.3f}" for v in d) +
              "   best rules " +
              ",".join(rr[s]["best_agg"] for s in sizes), flush=True)
    nreg = [fa["sizes"][str(s)] for s in sizes]
    print(f"[fair] region sizes {sizes} patches = {nreg} regions", flush=True)

    written.append(fignew.comparison_controls(
        methods, margins, a.out, bracket=(bi, wi, "what the question adds"),
        left_sub="each method at its own best pooling rule",
        right_sub="each competitor at its own best aggregation",
        notes=(f"n = {bs['n']} photographs, {bs['n_regions']} regions",
               f"n = {fa['n']} photographs"),
        xticks=tuple(sizes)))


for p in written:
    print("wrote", p, flush=True)
