"""Draw the layer strip from a 69_layerstrip run.  No model, no GPU.

The lens the paper means by "VJ-Lens" is the path-integrated one.  The word
grid script records that arm under the name ``VJ-Lens (path)`` and reserves the
bare name for the single-point form, which is the opposite of the convention in
``60_causalmap.py``.  This script therefore takes the arm by its file name and
prints it under the paper's name, so the figure cannot disagree with the rest
of the paper about what a VJ-Lens is.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from jvlm import figatlas

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True)
ap.add_argument("--out", default="/workspace/vlm/figures")
ap.add_argument("--which", type=int, nargs="*", default=None,
                help="card indices; every card if left out")
ap.add_argument("--prefix", default="f69_strip")
ap.add_argument("--gutter", type=float, default=1.62,
                help="inches reserved for the row names")
ap.add_argument("--order", nargs="*",
                default=["logit lens", "J-lens", "VJ-Lens (path)"])
a = ap.parse_args()

#: The word-grid table in the same section of the paper lists the two VJ arms
#: as separate rows, so the strip names them the same way; printing the path arm
#: as a bare "VJ-Lens" would make the figure disagree with the table beside it.
DISPLAY = {"VJ-Lens (path)": "VJ-Lens,\npath reading",
           "VJ-Lens": "VJ-Lens,\npoint reading"}

d = json.load(open(a.json))
cards = d["cards"]
have = sorted(cards[0]["lenses"]) if cards else []
order = [l for l in a.order if l in set(have)]
print(f"{len(cards)} cards, layers {d['layers']}, arms present {have}",
      flush=True)
print(f"rows: {[DISPLAY.get(l, l) for l in order]}", flush=True)

for k in (a.which if a.which is not None else range(len(cards))):
    c = cards[k]
    img = None
    if c.get("img"):
        img = Image.open(io.BytesIO(base64.b64decode(
            c["img"].split(",", 1)[1]))).convert("RGB")
    p = figatlas.layerstrip(c, order, a.out, f"{a.prefix}_{k}.png",
                            n_layers=d.get("n_layers"), display=DISPLAY,
                            img=img, gutter=a.gutter)
    print(f"  card {k}: {c['concept']:12s} says “{c.get('caption')}”  -> {p}",
          flush=True)
    for ln in order:
        per = c["lenses"][ln]
        nm = DISPLAY.get(ln, ln).replace("\n", " ")
        r = [per[str(l)]["box_rank"] for l in d["layers"]]
        w = [per[str(l)]["box_word"] for l in d["layers"]]
        print(f"      {nm:24s} rank  {r}", flush=True)
        print(f"      {'':24s} word  {w}", flush=True)
        if "boxc_rank" in per[str(d["layers"][0])]:
            rc = [per[str(l)]["boxc_rank"] for l in d["layers"]]
            print(f"      {'':24s} rank under the tile-z pooling that was "
                  f"wrong: {rc}", flush=True)
print("done", flush=True)
