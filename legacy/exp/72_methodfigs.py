"""Draw the method figures: the region ladder, resolution, models, depth.

Pure plotting -- no model, no GPU.  Every panel is read off a result file that
already exists; a missing file drops its panel and says so on stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jvlm import figmethod  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="results")
ap.add_argument("--out", default="figures")
ap.add_argument("--card", type=int, default=0)
ap.add_argument("--n-layers", type=int, default=28)
ap.add_argument("--only", default="")
a = ap.parse_args()
R, OUT = a.res, a.out
want = set(a.only.split(",")) if a.only else None
made, skipped = [], []


def run(tag, fn):
    if want and tag not in want:
        return
    try:
        made.append(fn())
    except FileNotFoundError as e:
        skipped.append(f"{tag}: missing {e}")
    except Exception as e:                                   # noqa: BLE001
        import traceback
        traceback.print_exc()
        skipped.append(f"{tag}: {type(e).__name__} {e}")


def need(p):
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


# ---------------------------------------------------------- A. region ladder
def _ladder():
    st = json.load(open(need(f"{R}/region/qwen2b_full_stats.json")))
    by = st["by_size"]
    rows = list(by.values()) if isinstance(by, dict) else by
    cards = json.load(open(need(f"{R}/causalmap/qwen2b_big_cards.json")))
    card = cards[a.card]
    need(card["path"])
    return figmethod.region_ladder(rows, card, OUT, grid=int(st.get("grid", 14)))


run("ladder", _ladder)


# ------------------------------------------------------------ B. resolution
def _resolution():
    df = pd.read_parquet(need(f"{R}/posvar/qwen2b_raw.parquet"))
    return figmethod.resolution(df, OUT)


run("resolution", _resolution)


# ---------------------------------------------------------------- C. models
LABEL = {"qwen2b_big": "Qwen3-VL-2B",
         "Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
         "llava-1.5-7b": "LLaVA-1.5-7B",
         "llava-1.5-7b-hf": "LLaVA-1.5-7B"}


def _models():
    runs = []
    for tag, lab in LABEL.items():
        p = f"{R}/causalmap/{tag}_raw.parquet"
        # the raw frame is written as a checkpoint while a run is still going;
        # the stats file is written once, at the end.  Only a finished run gets
        # a column here, or a figure could report a model on one photograph.
        if not (os.path.exists(p) and
                os.path.exists(f"{R}/causalmap/{tag}_stats.json")):
            if os.path.exists(p):
                print(f"  {lab}: raw present but unfinished, left out")
            continue
        raw = pd.read_parquet(p)
        runs.append({"label": lab, "raw": raw, "n": int(raw.i.nunique())})
    if not runs:
        raise FileNotFoundError("no causalmap raw files")
    print("  models drawn:", [r["label"] for r in runs])
    st = json.load(open(f"{R}/causalmap/qwen2b_big_stats.json"))
    return figmethod.models(runs, OUT, regions=int(st.get("regions", 49)))


run("models", _models)


# ----------------------------------------------------------- D. depth profile
def _depth():
    panels = []
    D = a.n_layers - 1

    p = f"{R}/causalmap/qwen2b_big_raw.parquet"
    if os.path.exists(p):
        cm = pd.read_parquet(p)
        cm["lens"] = cm.lens.replace(figmethod.CANON)
        cm = cm[cm.lens.isin(["logit lens", "J-lens", "VJ-Lens"])]
        n = int(cm.i.nunique())
        ser = {}
        for lens, g in cm.groupby("lens"):
            s = g.groupby("depth").spearman.mean().sort_index()
            ser[lens] = (s.index.values * 100, s.values)
        panels.append(dict(
            title="agrees with the causal map",
            sub=f"49 regions · n = {n} photographs",
            ylabel="agreement  (ρ)", series=ser, zero=True))
        ser = {}
        for lens, g in cm.groupby("lens"):
            s = g.groupby("depth").hit_cause.mean().sort_index()
            ser[lens] = (s.index.values * 100, s.values)
        panels.append(dict(
            title="names the most damaging region",
            sub=f"n = {n} photographs",
            ylabel="share of photographs", series=ser,
            ref=(1 / 49, "chance, 1 in 49"), ylim=(0, 0.46)))
    else:
        print("  no causalmap raw; dropping two panels")

    p = f"{R}/compare/qwen2b_del.parquet"
    if os.path.exists(p):
        dl = pd.read_parquet(p)
        col = "lens" if "lens" in dl.columns else "order"
        dl[col] = dl[col].replace(figmethod.CANON)
        ser = {}
        for lens, g in dl.groupby(col):
            if lens not in figmethod.LENS_COLOUR:
                continue
            xs, ys = [], []
            for L, q in g.groupby("layer"):
                cur = q.groupby("frac").rel.mean().sort_index()
                xs.append(100.0 * float(L) / D)
                ys.append(float(np.trapezoid(cur.values, cur.index)
                                / cur.index.max()))
            o = np.argsort(xs)
            ser[lens] = (np.array(xs)[o], np.array(ys)[o])
        panels.append(dict(
            title="destroys the answer fastest",
            sub=f"n = {int(dl.i.nunique())} photographs",
            ylabel="deletion area  (lower is better)", series=ser))
    else:
        print("  no compare/_del parquet; dropping the deletion panel")

    p = f"{R}/posvar/qwen2b_raw.parquet"
    if os.path.exists(p):
        pv = pd.read_parquet(p)
        pv["lens"] = pv.lens.replace(figmethod.CANON)
        ser = {}
        for lens, g in pv.groupby("lens"):
            s = g.groupby("depth").distinct.mean().sort_index()
            ser[lens] = (s.index.values * 100, s.values)
        panels.append(dict(
            title="tells one patch from another",
            sub=f"196 patches · n = {int(pv.i.nunique())} photographs",
            ylabel="distinct words per photograph", series=ser,
            ylim=(0, 130)))
    else:
        print("  no posvar raw; dropping the distinct-words panel")

    p = f"{R}/crossmodal/qwen2b_j_raw.parquet"
    if os.path.exists(p):
        cx = pd.read_parquet(p)
        pre = {"say_": "logit lens", "jlens_": "J-lens", "atlas_": "VJ-Lens"}
        layers = sorted({int(c.split("_")[1]) for c in cx.columns
                         if c.split("_")[0] + "_" in pre})
        ser = {}
        for k, lens in pre.items():
            xs, ys = [], []
            for L in layers:
                c = f"{k}{L}"
                if c not in cx:
                    continue
                g = cx.groupby("i")[c]
                xs.append(100.0 * L / D)
                ys.append(float((g.max() - g.min()).mean()))
            if xs:
                ser[lens] = (np.array(xs), np.array(ys))
        panels.append(dict(
            title="moves when the question changes",
            sub=f"four framings · n = {int(cx.i.nunique())} photographs",
            ylabel="change in the reading  (z units)", series=ser,
            zero=True, ylim=(-0.06, 0.62),
            dashed={"logit lens": (0, (4, 3))},
            z={"J-lens": 3, "logit lens": 4, "VJ-Lens": 5}))
    else:
        print("  no crossmodal raw; dropping the prompt panel")

    if not panels:
        raise FileNotFoundError("no depth panels available")
    return figmethod.depth_profile(panels, OUT)


run("depth", _depth)

print("\n".join(str(x) for x in made))
if skipped:
    print("SKIPPED:\n  " + "\n  ".join(skipped))
