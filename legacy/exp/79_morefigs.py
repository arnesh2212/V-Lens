# SPDX-License-Identifier: Apache-2.0
"""Draw the six figures in :mod:`jvlm.figmore` from result files already on disk.

Nothing here touches a GPU.  Every input is a parquet or a JSON written by an
earlier experiment; this script only reads them, draws, and prints the numbers
each figure's caption has to quote.

    python code/exp/79_morefigs.py --out figures --which all
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from jvlm import figmore  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="figures")
ap.add_argument("--which", default="all")
a = ap.parse_args()

R = a.results
want = set(a.which.split(",")) if a.which != "all" else None


def go(key: str) -> bool:
    return want is None or key in want


def load(path):
    p = os.path.join(R, path)
    return pd.read_parquet(p) if os.path.exists(p) else None


# ------------------------------------------------------------ A. five models
if go("fivemodels"):
    frames = {}
    for tag, _, _ in figmore.MODELS:
        d = load(f"causalmap/{tag}_raw.parquet")
        if d is not None:
            frames[tag] = d
    if len(frames) >= 2:
        print(f"fivemodels: {len(frames)} models", flush=True)
        figmore.fivemodels(frames, a.out)
    else:
        print("fivemodels: skipped, fewer than two models on disk", flush=True)

# --------------------------------------------------------- B. pixel occlusion
if go("pixelmodels"):
    pix, act = {}, {}
    for tag, _, _ in figmore.PIX_MODELS:
        d = load(f"pixeltruth/{tag}_raw.parquet")
        if d is not None:
            pix[tag] = d
    for tag in ("qwen2b_big", "Qwen3-VL-4B-Instruct", "llava-1.5-7b"):
        d = load(f"causalmap/{tag}_raw.parquet")
        if d is not None:
            act[tag] = d
    # the pixel run and the activation run use different tags for the same model
    for src, dst in (("qwen2b_big", "qwen2b"),
                     ("Qwen3-VL-4B-Instruct", "Qwen3-VL-4B")):
        if src in act:
            act[dst] = act[src]
    if len(pix) >= 2:
        print(f"pixelmodels: {len(pix)} models", flush=True)
        figmore.pixelmodels(pix, act, a.out)
    else:
        print("pixelmodels: skipped, fewer than two models on disk", flush=True)

# ---------------------------------------------------------------- C. prompts
if go("promptdepth"):
    d = load("crossmodal/qwen2b_j_raw.parquet")
    if d is None:
        d = load("crossmodal/qwen2b_raw.parquet")
    if d is not None:
        print("promptdepth", flush=True)
        figmore.promptdepth(d, a.out)
    else:
        print("promptdepth: skipped, no crossmodal parquet", flush=True)

# ------------------------------------------------------------------ D. edits
if go("editcoord"):
    d = load("edits/qwen2b_raw.parquet")
    if d is not None:
        print("editcoord", flush=True)
        figmore.editcoord(d, a.out)
    else:
        print("editcoord: skipped, no edits parquet", flush=True)

# -------------------------------------------------------------- E1. baselines
if go("baselines"):
    d = load("causalmap/qwen2b_base_raw.parquet")
    if d is not None:
        print("baselines", flush=True)
        figmore.baselines(d, a.out)
    else:
        print("baselines: skipped, no qwen2b_base parquet", flush=True)

# --------------------------------------------------------------- E2. the path
if go("pathcost"):
    d = load("complete/qwen2b_raw.parquet")
    if d is not None:
        print("pathcost", flush=True)
        figmore.pathcost(d, a.out)
    else:
        print("pathcost: skipped, no complete parquet", flush=True)

# ------------------------------------------------------------- F. layer by layer
if go("layerwise"):
    frames = {}
    for tag, nice in (("qwen2b_layerwise", "Qwen3-VL 2B"),
                      ("llava_layerwise", "LLaVA-1.5 7B")):
        d = load(f"causalmap/{tag}_raw.parquet")
        if d is not None:
            frames[nice] = d
    if frames:
        print(f"layerwise: {len(frames)} models", flush=True)
        figmore.layerwise(frames, a.out)
    else:
        print("layerwise: skipped, the depth sweeps have not landed", flush=True)

print("done", flush=True)
