# SPDX-License-Identifier: Apache-2.0
"""Render the flagship figure.  No model, no GPU: every number here is read
back out of two result files that already exist.

Photograph, grid and box are shared between the two questions by construction
(they are the same picture).  The occlusion truth for both questions comes
from ``exp/80_absorb.py``'s question-flip run
(``results/absorb/qwen2b_twoq_cards.json``), which asked the identical
photograph an "easy" question (what is the main object) and a "hard" one (how
many of it are there) and measured, region by region, what covering each one
costs the asked-about logit.  VJ-Lens's own reading for the first question
comes from ``exp/60_causalmap.py``'s card for the same photograph, same grid,
same layers (``results/causalmap/qwen2b_layerwise_cards.json``) under that
script's own default prompt ("Describe this image") -- a real, measured
reading, just not measured under the absorb run's exact wording.  Its truth
peaks at the same region the absorb run's "easy" truth does, at every layer
but the last, which is the check this script prints before trusting it.  No
reading exists for the "how many" question; the figure says so rather than
inventing one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jvlm import figflag

ap = argparse.ArgumentParser()
ap.add_argument("--causalmap", default="/workspace/vlm/results/causalmap/"
                                       "qwen2b_layerwise_cards.json")
ap.add_argument("--twoq", default="/workspace/vlm/results/absorb/"
                                  "qwen2b_twoq_cards.json")
ap.add_argument("--twoq-stats", default="/workspace/vlm/results/absorb/"
                                        "qwen2b_twoq_stats.json")
ap.add_argument("--image-id", type=int, default=516173)
ap.add_argument("--layer", type=int, default=13)
ap.add_argument("--out", default="/workspace/vlm/figures")
a = ap.parse_args()

cm = json.load(open(a.causalmap))
cm_card = next(c for c in cm if c["image_id"] == a.image_id
              and c["layer"] == a.layer)

tq = json.load(open(a.twoq))
tq_card = next(c for c in tq if c["image_id"] == a.image_id)
li = tq_card["layers"].index(a.layer)

tstats = json.load(open(a.twoq_stats))

# ---- the check this whole figure leans on: does the neighbour prompt's own
# truth (causalmap's "Describe this image") land on the same region as the
# absorb run's "easy" truth, at every layer the two runs share?
agree = sum(1 for c in cm if c["image_id"] == a.image_id
           and max(range(len(c["truth"])), key=lambda j: c["truth"][j])
               == max(range(49),
                      key=lambda j: tq_card["questions"]["easy"]["truth"]
                                  [tq_card["layers"].index(c["layer"])][j]))
print(f"neighbour-prompt check: truth argmax agrees with the absorb run's "
      f"'easy' truth argmax at {agree} of {len(tq_card['layers'])} layers",
      flush=True)

card = dict(
    path=cm_card["path"], grid=cm_card["coarse"], box=tq_card["inside"],
    truth_a=tq_card["questions"]["easy"]["truth"][li],
    truth_b=tq_card["questions"]["hard"]["truth"][li],
    read_a=cm_card["lenses"]["VJ-Lens"]["score"],
    fixed={"logit lens": cm_card["lenses"]["logit lens"]["score"],
          "J-lens": cm_card["lenses"]["J-lens"]["score"]})

p = figflag.flagship(
    card, a.out, name="f134_flagship.png",
    qa_text=tq_card["questions"]["easy"]["prompt"],
    qb_text=tq_card["questions"]["hard"]["prompt"],
    image_activation_maxdev=float(
        tstats.get("image_activation_relative_difference", 0.0)))
print("wrote", p, flush=True)
