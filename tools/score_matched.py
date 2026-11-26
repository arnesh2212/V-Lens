# SPDX-License-Identifier: Apache-2.0
"""AUROC of the tuned lens fitted at V-Read's token budget, next to V-Read and the full tuned lens.

Readouts of the matched lens (results/fits_matched/readouts, written by
tools/layer_readouts.py read with the matched map cache relabelled so the tool
accepts it: its ``v_read`` series IS the matched tuned lens) are scored exactly
as Table 2 scores every reader, per block, against the answer the model gives;
means are taken before 0.45 of the depth and over the Table 2 window.
Writes results/fits_matched/score.json.
"""
import glob
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tools"))
import make_vread_table as T  # noqa: E402
from vlens.figures import fx_bench as B  # noqa: E402

M = "Qwen3-VL-8B-Instruct"
FAMS = ("openimages_orig_auth", "openimages_edit_auth", "openimages_orig_eval", "nuscenes_edit_auth", "vins_orig_auth",
        "vins_edit_auth", "gemini_orig_auth", "wmdp_bio_sand", "wmdp_chem_sand", "wmdp_cybe_sand")
BANDS = {"early": (0.0, 0.45), "window": tuple(T.BAND)}


def matched_auroc(fam):
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(f"{tmp}/readouts"); os.makedirs(f"{tmp}/simonsays")
        for p in glob.glob(f"{HERE}/results/fits_matched/readouts/{M}__{fam}_*.json"):
            os.symlink(os.path.abspath(p), f"{tmp}/readouts/{os.path.basename(p)}")
        os.symlink(f"{HERE}/results/simonsays/{M}__{fam}.json", f"{tmp}/simonsays/{M}__{fam}.json")
        return B._readout_family(tmp, M, fam, True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


out, rng = {}, np.random.default_rng(0)
rows = {b: {k: [] for k in ("v_read", "tuned_full", "tuned_matched")} for b in BANDS}
for fam in FAMS:
    ref, _ = T._capture(T.D._family, f"{HERE}/results", M, fam)
    new = matched_auroc(fam)
    if ref is None or new is None:
        print(fam, "missing"); continue
    dep = np.asarray(ref["depth"], float)
    a = {k: np.asarray(v, float) for k, v in ref["auroc"].items()}
    am = np.asarray(new["auroc"]["v_read"], float)
    ok = np.isfinite(a["v_read"])
    cell = {}
    for b, (lo, hi) in BANDS.items():
        inb = (dep >= lo) & ((dep < hi) if b == "early" else (dep <= hi)) & ok
        v = dict(v_read=np.nanmean(a["v_read"][inb]), tuned_full=np.nanmean(a["tuned"][inb]), tuned_matched=np.nanmean(am[inb]))
        for k, x in v.items():
            rows[b][k].append(float(x))
        cell[b] = {k: float(x) for k, x in v.items()}
    out[fam] = cell
    print(fam, cell)
summ = {}
for b, d in rows.items():
    s = {k: float(np.mean(v)) for k, v in d.items()}
    diff = np.array(d["v_read"]) - np.array(d["tuned_matched"])
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
    s["v_read_minus_matched"] = [float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int((diff > 0).sum()), len(diff)]
    fm = np.array(d["tuned_full"]) - np.array(d["tuned_matched"])
    s["full_minus_matched"] = float(fm.mean())
    summ[b] = s
    print(b, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in s.items()})
os.makedirs(f"{HERE}/results/fits_matched", exist_ok=True)
json.dump(dict(per_task=out, summary=summ), open(f"{HERE}/results/fits_matched/score.json", "w"), indent=1)
