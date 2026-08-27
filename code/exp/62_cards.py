"""Render the side-by-side cards from a causal-map run.

Separate from the run itself so a card can be re-drawn without re-measuring,
and so the drawing can be looked at and fixed without spending GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scipy.stats as sps
from PIL import Image

from jvlm import figcausal, style

ap = argparse.ArgumentParser()
ap.add_argument("--cards", required=True)
ap.add_argument("--raw", default=None, help="the run's _raw.parquet")
ap.add_argument("--out", default="/workspace/vlm/figures")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--prefix", default="f61_card")
ap.add_argument("--select", default="order",
                choices=["order", "typical"],
                help="'order' takes the file order; 'typical' takes\n                     the cards whose VJ-Lens agreement is closest to\n                     the run average, which is a stated rule rather\n                     than a look at the pictures")
ap.add_argument("--order", nargs="*",
                default=["logit lens", "J-lens", "VJ-Lens"])
a = ap.parse_args()

style.use()
cards = json.load(open(a.cards))
df = pd.read_parquet(a.raw) if a.raw and os.path.exists(a.raw) else None
print(f"{len(cards)} cards; lenses present: "
      f"{sorted(cards[0]['lenses']) if cards else '-'}", flush=True)

order = [n for n in a.order if cards and n in cards[0]["lenses"]]
extra = [n for n in (cards[0]["lenses"] if cards else {}) if n not in order]
order = order + extra

def _rho(rec, lens="VJ-Lens"):
    """This photograph's own agreement, from the card's own arrays.

    The card stores, per region, the true drop and each lens's score for the
    photographed word -- the same two arrays the run correlates.  So the
    card can be graded without joining back to the raw table, which is what
    an earlier version tried: it matched on ``i``, a column the card file
    does not carry, so every lookup fell through to a mean over every
    photograph of the same concept and two different pictures printed the
    same three numbers.
    """
    d = (rec.get("lenses") or {}).get(lens)
    if not d or "score" not in d:
        return None
    s = np.asarray(d["score"], dtype=float)
    t = np.asarray(rec["truth"], dtype=float)
    if len(s) != len(t) or len(s) < 3:
        return None
    r = sps.spearmanr(s, t).statistic
    return None if r != r else float(r)


chosen = cards[: a.n]
if a.select == "typical" and df is not None:
    mean = float(df[df.lens == "VJ-Lens"].spearman.mean())
    scored = [(abs((_rho(c) if _rho(c) is not None else mean) - mean), c)
              for c in cards]
    chosen = [c for _, c in sorted(scored, key=lambda t: t[0])[: a.n]]
    print(f"selection: the {a.n} cards closest to the run's mean "
          f"agreement of {mean:.3f}", flush=True)

made = []
for k, rec in enumerate(chosen):
    rho = {}
    for n_ in order:
        v = _rho(rec, n_)
        if v is not None:
            rho[n_] = v
    img = None
    if rec.get("path") and os.path.exists(rec["path"]):
        img = Image.open(rec["path"]).convert("RGB")
        img.thumbnail((520, 520), Image.LANCZOS)
    p = figcausal.card(rec, img, order, rho, a.out, f"{a.prefix}_{k}.png")
    made.append(p)
    print("  wrote", p, flush=True)

if df is not None:
    print("  wrote", figcausal.summary(df, a.out), flush=True)
print("done", flush=True)
