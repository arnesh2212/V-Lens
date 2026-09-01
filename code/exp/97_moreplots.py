# SPDX-License-Identifier: Apache-2.0
"""Four figures for results already on disk.  No model, no GPU.

    f135  the question redraws the map     results/absorb/qwen2b_twoq_*.json
    f136  the exact zero, as a picture     results/winoground/Qwen3-VL-8B_stats.json
    f137  the interaction law converging   results/interaction/qwen2b_stats.json
    f138  the 8B is a real model           results/winoground/Qwen3-VL-8B_stats.json

Every number this script prints is the number the figure it names draws;
whoever writes each <figcaption> in paper/atlas.html can quote this log.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from jvlm import figplus

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="/workspace/vlm/results")
ap.add_argument("--out", default="/workspace/vlm/figures")
ap.add_argument("--only", default="", help="1,2,3,4")
a = ap.parse_args()
R = a.res
want = set(a.only.split(",")) if a.only else {"1", "2", "3", "4"}
say = lambda *s: print(*s, flush=True)


# ============================================================== 1
if "1" in want:
    d = json.load(open(f"{R}/absorb/qwen2b_twoq_cards.json"))
    layers = d[0]["layers"]
    across, same_next = [], []
    for c in d:
        easy = c["questions"]["easy"]["truth"]
        hard = c["questions"]["hard"]["truth"]
        for l in range(len(layers)):
            r, _ = spearmanr(easy[l], hard[l])
            if r == r:
                across.append(r)
        for l in range(len(layers) - 1):
            for arm in (easy, hard):
                r, _ = spearmanr(arm[l], arm[l + 1])
                if r == r:
                    same_next.append(r)
    across = np.array(across)
    same_next = np.array(same_next)
    _, p = mannwhitneyu(same_next, across, alternative="greater")
    say(f"f135  across-question rho: n={len(across)} median="
        f"{np.median(across):.3f}   same-question-next-layer rho: "
        f"n={len(same_next)} median={np.median(same_next):.3f}   "
        f"Mann-Whitney p={p:.2e}")
    print(figplus.question_redraws_map(
        across, same_next, a.out, n_photos=len(d), mannwhitney_p=p))

# ============================================================== 2
if "2" in want:
    wg = json.load(open(f"{R}/winoground/Qwen3-VL-8B_stats.json"))
    z = wg["zero"]
    rows = [
        ("logit lens", z["logit lens"]["max_abs_region_dev"], "#eb6834", True),
        ("VJ-Lens (point)", z["VJ-Lens (point)"]["max_abs_region_dev"],
         "#1baf7a", True),
        ("VJ-Lens", z["VJ-Lens"]["max_abs_region_dev"], "#2a78d6", True),
        ("control: same prompt, re-run",
         wg.get("control_same_prompt_max_h_dev", 0.0), "#8a8880", False),
        ("control: raw activations, prompt A vs B",
         wg.get("max_h_dev", 0.0), "#8a8880", False),
    ]
    say("f136  " + "   ".join(f"{lab}={v:.2f}" for lab, v, _, _ in rows))
    print(figplus.exact_zero(rows, a.out, n=z["logit lens"]["n"]))

# ============================================================== 3
if "3" in want:
    it = json.load(open(f"{R}/interaction/qwen2b_stats.json"))
    conv = [r for r in it["convergence"] if r["target"] == "margin"]
    rows = [(r["steps"], r["completeness"]["pearson"], r["law"]["pearson"])
           for r in sorted(conv, key=lambda r: r["steps"])]
    say("f137  " + "   ".join(
        f"steps={s} completeness_r={c:.3f} law_r={l:.3f}"
        for s, c, l in rows))
    print(figplus.law_converging(
        rows, a.out, shipped=4, partition_err=it["partition_max_err"],
        regroup_err=it["regrouping_identity_max_err"]))

# ============================================================== 4
if "4" in want:
    wg = json.load(open(f"{R}/winoground/Qwen3-VL-8B_stats.json"))
    s = wg["winoground_score"]
    scores = [("text", s["text"], s["text_chance"]),
             ("image", s["image"], s["image_chance"]),
             ("group", s["group"], s["group_chance"]),
             ("per-image text", s["per_image_text"], s["per_image_chance"])]
    say("f138  " + "   ".join(f"{lab}={v:.3f}(chance {c:.3f})"
                              for lab, v, c in scores)
        + f"   yes_correct={s['yes_rate_correct_caption']:.3f} "
          f"yes_swapped={s['yes_rate_swapped_caption']:.3f}")
    print(figplus.real_model(
        scores, (s["yes_rate_correct_caption"], s["yes_rate_swapped_caption"]),
        a.out, n_items=s.get("n_items")))

say("\nCAPTION CAVEAT for f138 (put in the HTML, not the figure): this is "
    "Winoground's text score computed with a generative yes/no margin, not "
    "the contrastive scorer behind the published CLIP-era figures, so it "
    "is not comparable to those.")
