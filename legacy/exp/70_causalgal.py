"""Render the causal breadth gallery.  No model, no GPU."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jvlm import figcausal

ap = argparse.ArgumentParser()
ap.add_argument("--cards", required=True)
ap.add_argument("--out", default="/workspace/vlm/figures")
ap.add_argument("--order", nargs="*",
                default=["logit lens", "J-lens", "VJ-Lens"])
a = ap.parse_args()
cards = json.load(open(a.cards))
have = set(cards[0]["lenses"]) if cards else set()
order = [n for n in a.order if n in have]
print(f"{len(cards)} cards; lenses {order}", flush=True)
print(figcausal.causal_gallery(cards, order, a.out))
