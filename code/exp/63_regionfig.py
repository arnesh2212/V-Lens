"""Draw the region-size sweep.  Pure plotting -- no model, no GPU."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jvlm import figcausal

ap = argparse.ArgumentParser()
ap.add_argument("--stats", required=True)
ap.add_argument("--out", default="/workspace/vlm/figures")
ap.add_argument("--box", type=float, default=None,
                help="typical object box, in patches, drawn as a reference line")
a = ap.parse_args()
d = json.load(open(a.stats))
by = d.get("by_size")
rows = list(by.values()) if isinstance(by, dict) else (by or [])
print(figcausal.regionsweep(rows, d.get("n_patches", 196), a.out,
                            box_patches=a.box))
