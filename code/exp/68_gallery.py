"""Render the breadth gallery from a word-grid run.  No model, no GPU."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jvlm import figdiagram

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True)
ap.add_argument("--out", default="/workspace/vlm/figures")
ap.add_argument("--layer", type=int, default=None)
ap.add_argument("--lenses", nargs="*",
                default=["logit lens", "J-lens", "VJ-Lens (direction)"])
ap.add_argument("--per-row", type=int, default=6)
a = ap.parse_args()
d = json.load(open(a.json))
cards = d["cards"]
have = set(cards[0]["lenses"]) if cards else set()
lenses = [l for l in a.lenses if l in have]
print(f"{len(cards)} cards; using lenses {lenses}", flush=True)
print(figdiagram.gallery(cards, lenses, a.out, layer=a.layer,
                         per_row=a.per_row))
