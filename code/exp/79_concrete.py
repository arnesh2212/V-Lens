"""Draw the concrete companions.  Pure plotting -- no model, no GPU.

Four of the paper's axes are counted in units a reader cannot picture.  This
draws one real instance beside each of them:

    f90_regionscale   one photograph at every region size the sweep measured
    f91_family        the region parameter as the dial its two ends are lenses of
    f93_damage        one photograph under six edits, on the damage axis
    f94_path          where the path's points sit, and what each buys

and re-renders f43_posvar, whose two axes get their ceiling and their floor.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from jvlm import figscale, figures

R = "/workspace/vlm/results"
ap = argparse.ArgumentParser()
ap.add_argument("--region", default=f"{R}/region/qwen2b_full_stats.json")
ap.add_argument("--cards", default=f"{R}/causalmap/qwen2b_big_cards.json")
ap.add_argument("--image-id", type=int, default=353051)
ap.add_argument("--edits", default=f"{R}/edits/qwen2b_raw.parquet")
ap.add_argument("--complete", default=f"{R}/complete/qwen2b_raw.parquet")
ap.add_argument("--posvar", default=f"{R}/posvar/qwen2b_stats.json")
ap.add_argument("--posvar-raw", default=f"{R}/posvar/qwen2b_raw.parquet")
ap.add_argument("--box", type=float, default=30.0)
ap.add_argument("--n-layers", type=int, default=28)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--trial", type=int, default=None)
ap.add_argument("--out", default="/workspace/vlm/figures")
ap.add_argument("--only", default=None, help="comma-separated: strip,family,damage,span,path,posvar")
a = ap.parse_args()
want = set((a.only or "strip,family,damage,span,path,posvar").split(","))


def try_(fn, *args, **kw):
    try:
        print(fn(*args, **kw), flush=True)
    except Exception as e:                                   # noqa: BLE001
        import traceback
        print(f"  FAILED {fn.__name__}: {e}", flush=True)
        traceback.print_exc()


stats = json.load(open(a.region))

if {"strip", "family"} & want:
    cards = json.load(open(a.cards))
    pick = next((c for c in cards if int(c["image_id"]) == a.image_id), None)
    if pick is None:
        pick = cards[0]
        print(f"  image {a.image_id} not in cards; using {pick['image_id']}")
    stats.setdefault("coarse", int(pick.get("coarse", 7)))

if "strip" in want:
    try_(figscale.regionstrip, stats, pick["path"], a.out,
         inside=pick.get("inside"), seed=a.seed)

if "family" in want:
    try_(figscale.family, stats, a.out, box_patches=a.box,
         n_layers=a.n_layers)

if "damage" in want:
    try_(figscale.damageruler, pd.read_parquet(a.edits), a.out,
         trial=a.trial)

if "span" in want:
    try_(figscale.span, pd.read_parquet(a.edits), a.out)

if "path" in want:
    try_(figscale.pathcase, pd.read_parquet(a.complete), a.out,
         n_layers=a.n_layers)

if "posvar" in want:
    raw = pd.read_parquet(a.posvar_raw) if os.path.exists(a.posvar_raw) else None
    try_(figures.f43_posvar, json.load(open(a.posvar)), a.out, raw=raw,
         n_layers=a.n_layers)
