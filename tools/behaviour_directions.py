# SPDX-License-Identifier: Apache-2.0
"""Directions for behaviours, one per channel, from the stored answer-spot states.

For each contrast (a behaviour delivered through the text or through the
picture) and each block: the unit difference of class means, how well it
separates the two classes on items it was not built from (leave-one-out
AUROC of the projection), and the cosine between every pair of directions.
No model is run; the states were stored by the behaviour pass.

    python3 tools/behaviour_directions.py --model Qwen3.5-27B
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DRESS = {"eval_watermark", "eval_bbox", "eval_annotation_ui", "board_eval_cue",
              "watermark", "annotation_ui"}
SNAP_DRESS = {"deploy_phone", "deploy_dashcam_hud", "deploy_phone_screenshot",
              "deploy_windshield", "phone_screenshot", "board_deploy_cue"}
OFFICIAL = {"official_sign", "official_plate", "no_parking_official", "both_official"}
INFORMAL = {"handwritten_note", "informal_sign", "downgrade_cardboard",
            "no_parking_informal", "downgrade_informal"}

#: (name, channel, behaviour, families, which rows are class 1, which are class 0).
#: The sandbagging pair comes from the channel run: the hazard instruction against
#: the honest control given through the same channel, the question in the text.
SANDCHAN = "sandchan"
CONTRASTS = [
    ("sandbagging, told in the text", "text", "sandbagging", [f"{SANDCHAN}:wmdp_chan"],
     lambda r: r.get("subcategory") == "text__hazard_text",
     lambda r: r.get("subcategory") == "text__honest_text"),
    ("sandbagging, told in the picture", "vision", "sandbagging", [f"{SANDCHAN}:wmdp_chan"],
     lambda r: r.get("subcategory") == "text__hazard_picture",
     lambda r: r.get("subcategory") == "text__honest_picture"),
    ("eval awareness, the picture dressed as a test", "vision", "eval awareness",
     ["openimages_edit_eval", "nuscenes_edit_eval", "vins_edit_eval"],
     lambda r: (r.get("meta") or {}).get("edit_variant") in EVAL_DRESS,
     lambda r: (r.get("meta") or {}).get("edit_variant") in SNAP_DRESS),
    ("authority, an official bearer", "vision", "authority",
     ["openimages_edit_auth", "vins_edit_auth", "nuscenes_edit_auth"],
     lambda r: (r.get("meta") or {}).get("edit_variant") in OFFICIAL,
     lambda r: (r.get("meta") or {}).get("edit_variant") in INFORMAL),
]


def auroc(pos, neg):
    s = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = s.argsort(); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    return (rk[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def load(results, model, fams, f1, f0):
    LAYERS = [None]
    X1, X0 = [], []
    for fam in fams:
        where = results
        if ":" in fam:
            sub, fam = fam.split(":", 1)
            where = os.path.join(os.path.dirname(results), sub)
        p = f"{where}/{model}__{fam}_states.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p)
        idx = {str(i): k for k, i in enumerate(z["ids"])}
        A = z["ANS"]
        lay = [int(x) for x in (z["layers"] if "layers" in z.files else range(A.shape[1]))]
        # files stored on different block grids (a 16-block grid vs every block) are
        # aligned on the blocks they share; files that store every block are unchanged
        if LAYERS[0] is None:
            LAYERS[0] = lay
        if lay != LAYERS[0]:
            common = [l for l in LAYERS[0] if l in set(lay)]
            keep_old = [LAYERS[0].index(l) for l in common]
            X1[:] = [x[keep_old] for x in X1]; X0[:] = [x[keep_old] for x in X0]
            LAYERS[0] = common
        A = A[:, [lay.index(l) for l in LAYERS[0]]]
        for r in json.load(open(f"{where}/{model}__{fam}.json"))["rows"]:
            k = idx.get(str(r["id"]))
            if k is None:
                continue
            if f1(r):
                X1.append(A[k])
            elif f0(r):
                X0.append(A[k])
    return (np.asarray(X1, np.float32), np.asarray(X0, np.float32), LAYERS[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", default=os.path.join(HERE, "results", "simonsays"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "directions_geom"))
    a = ap.parse_args()
    dirs, report, lays = {}, {}, {}
    for name, ch, beh, fams, f1, f0 in CONTRASTS:
        X1, X0, lay = load(a.results, a.model, fams, f1, f0)
        if len(X1) < 5 or len(X0) < 5:
            print(f"  {name}: too few ({len(X1)}, {len(X0)})"); continue
        L = X1.shape[1]
        # unit-normalise each state so a direction is about where, not how big
        n1 = X1 / np.linalg.norm(X1, axis=2, keepdims=True).clip(1e-6)
        n0 = X0 / np.linalg.norm(X0, axis=2, keepdims=True).clip(1e-6)
        s1, s0 = n1.sum(0), n0.sum(0)
        D, sep = [], []
        for l in range(L):
            d = n1[:, l].mean(0) - n0[:, l].mean(0)
            D.append(d / max(np.linalg.norm(d), 1e-9))
            # each item is projected on a direction built without it
            p1 = [((s1[l] - n1[i, l]) / (len(n1) - 1) - n0[:, l].mean(0)) @ n1[i, l] for i in range(len(n1))]
            p0 = [(n1[:, l].mean(0) - (s0[l] - n0[j, l]) / (len(n0) - 1)) @ n0[j, l] for j in range(len(n0))]
            sep.append(float(auroc(np.asarray(p1), np.asarray(p0))))
        dirs[name] = np.asarray(D)
        lays[name] = list(lay) if lay is not None else list(range(L))
        report[name] = dict(channel=ch, behaviour=beh, n1=len(X1), n0=len(X0), auroc=sep)
        print(f"  {name:<46} n={len(X1)}/{len(X0)}  peak AUROC {max(sep):.2f} at block {int(np.argmax(sep))}/{L}")
    names = list(dirs)
    # every direction on the blocks all contrasts share (identical to before when all store every block)
    grid = [l for l in lays[names[0]] if all(l in set(lays[n]) for n in names)]
    for n in names:
        dirs[n] = dirs[n][[lays[n].index(l) for l in grid]]
    L = len(grid)
    cos = {f"{x}|{y}": [float(dirs[x][l] @ dirs[y][l]) for l in range(L)]
           for i, x in enumerate(names) for y in names[i + 1:]}
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(f"{a.out}/{a.model}__directions.npz", names=np.array(names),
                        D=np.stack([dirs[n] for n in names]).astype(np.float16))
    json.dump(dict(model=a.model, n_layers=L, blocks=grid, contrasts=report, cosine=cos),
              open(f"{a.out}/{a.model}__directions.json", "w"))
    for k, v in cos.items():
        print(f"  cos {k:<90} mid {v[L // 2]:+.2f}  max|.| {max(map(abs, v)):.2f}")


if __name__ == "__main__":
    main()
