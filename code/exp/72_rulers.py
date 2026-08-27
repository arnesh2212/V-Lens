"""Draw the two-ruler figures.  Pure plotting -- no model, no GPU.

  python3 exp/72_rulers.py --act results/causalmap/qwen2b_big_raw.parquet \
                           --pix results/pixeltruth/qwen2b_raw.parquet \
                           --cards results/causalmap/qwen2b_big_cards.json \
                           --items results/pixeltruth/qwen2b_items.json
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from jvlm import figrulers

R = "/workspace/vlm/results"
ap = argparse.ArgumentParser()
ap.add_argument("--act", default=f"{R}/causalmap/qwen2b_big_raw.parquet")
ap.add_argument("--pix", default=f"{R}/pixeltruth/qwen2b_raw.parquet")
ap.add_argument("--cards", default=f"{R}/causalmap/qwen2b_big_cards.json")
ap.add_argument("--items", default=f"{R}/pixeltruth/qwen2b_items.json")
ap.add_argument("--out", default="/workspace/vlm/figures")
ap.add_argument("--only", default=None, help="a, b or c")
a = ap.parse_args()

act = pd.read_parquet(a.act)
pix = pd.read_parquet(a.pix)
want = (a.only or "abc").lower()

if "a" in want:
    print(figrulers.tworulers(act, pix, a.out), flush=True)
if "b" in want:
    cards = json.load(open(a.cards))
    print(figrulers.boxdissociation(act, pix, cards, a.out), flush=True)
if "c" in want:
    if not os.path.exists(a.items):
        print("no item map for the pixel run; skipping the gallery")
    else:
        items = json.load(open(a.items))
        print(figrulers.pixel_gallery(pix, items, a.out), flush=True)
