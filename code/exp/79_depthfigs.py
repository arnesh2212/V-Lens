"""Draw the layerwise figures from a ``--cards-every-layer`` causal-map run.

No model, no GPU: everything here is read out of the result files.

    A  f83_layermap_<k>.png   one photograph, the truth and the lenses down the
                              rows, the layers across the columns
    B  f84_traject.png        the region that ends up mattering most, followed
                              up the stack as its rank among the 49
    C  f86_arch.png           the same strip on two architectures
    D  f85_readable.png       when a lens first prints something true

``--fake`` builds A and B out of the single-layer cards with noise, for
checking the layout while the twelve-layer run is still on the GPU.  It writes
into a separate directory and its output is never published.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from jvlm import figdepth

ap = argparse.ArgumentParser()
ap.add_argument("--cards", default="/workspace/vlm/results/causalmap/"
                                   "qwen2b_layerwise_cards.json")
ap.add_argument("--llava", default="/workspace/vlm/results/causalmap/"
                                   "llava_layerwise_cards.json")
ap.add_argument("--strip", default="/workspace/vlm/results/wordgrid/"
                                   "qwen2b_strip2_wordgrid.json")
ap.add_argument("--posvar", default="/workspace/vlm/results/posvar/"
                                    "qwen2b_stats.json")
ap.add_argument("--out", default="/workspace/vlm/figures")
ap.add_argument("--which", type=int, nargs="*", default=None,
                help="which photographs get a map of their own")
ap.add_argument("--traject", type=int, nargs="*", default=None,
                help="which photographs the trajectory panels follow")
ap.add_argument("--max-maps", type=int, default=4)
ap.add_argument("--only", nargs="*", default=None,
                help="a subset of A B C D")
ap.add_argument("--fake", action="store_true")
ap.add_argument("--square", type=int, default=448)
a = ap.parse_args()
want = set(a.only or ["A", "B", "C", "D"])


def load_photo(path, square):
    """The photograph exactly as the model saw it.

    ``corpus.load_image(square=...)`` resizes to a square; it does not crop.
    Cropping here instead would put the seven-by-seven grid over a different
    part of the frame from the one the regions were covered in, and every
    marked region in these figures would be pointing at the wrong place.
    """
    try:
        return Image.open(path).convert("RGB").resize((square, square),
                                                      Image.LANCZOS)
    except Exception:
        return None


def by_image(cards):
    """Group the cards by photograph, each group sorted by layer."""
    g = {}
    for c in cards:
        g.setdefault(c["path"], []).append(c)
    return {k: sorted(v, key=lambda c: int(c["layer"])) for k, v in g.items()
            if len(v) > 1}


def fake(cards, n=12):
    """One card per layer, out of one card, so the layout can be checked."""
    rng = np.random.default_rng(0)
    out = []
    for c in cards[:4]:
        t = np.asarray(c["truth"], dtype=float)
        for i in range(n):
            d = dict(c)
            d["layer"] = 5 + 2 * i
            d["depth"] = round(d["layer"] / 27, 3)
            w = np.exp(-((np.arange(len(t)) - (5 + 3 * i)) ** 2) / 40.0)
            d["truth"] = (t * (0.2 + i / n) + w * t.std() * 2
                          + rng.normal(0, t.std() * .3, len(t))).tolist()
            d["lenses"] = {k: dict(v, score=list(
                np.asarray(v["score"], float) * (0.5 + i / n)
                + rng.normal(0, np.std(v["score"]) * .6, len(t))))
                for k, v in c["lenses"].items()}
            out.append(d)
    return out


if a.fake:
    cards = fake(json.load(open("/workspace/vlm/results/causalmap/"
                                "qwen2b_big_cards.json")))
    n_layers = 28
else:
    if not os.path.exists(a.cards):
        print(f"MISSING {a.cards}", flush=True)
        sys.exit(2)
    d = json.load(open(a.cards))
    cards = d if isinstance(d, list) else d["cards"]
    n_layers = 28

groups = by_image(cards)
print(f"{len(cards)} cards, {len(groups)} photographs, "
      f"layers {sorted({int(c['layer']) for c in cards})}", flush=True)

order = [l for l in figdepth.ORDER if l in cards[0]["lenses"]]
print(f"rows: {order}", flush=True)

keys = list(groups)
if a.which is not None:
    keys = [keys[i] for i in a.which if i < len(keys)]

if "A" in want:
    for k, path in enumerate(keys[: a.max_maps]):
        g = groups[path]
        p = figdepth.layermap(g, a.out, f"f83_layermap_{k}.png", order=order,
                              img=load_photo(path, a.square),
                              n_layers=n_layers)
        print(f"  A card {k}: {g[0]['concept']:12s} {len(g)} layers -> {p}",
              flush=True)

if "B" in want:
    tk = ([list(groups)[i] for i in a.traject] if a.traject
          else list(groups)[:3])
    p = figdepth.trajectory(groups, a.out, "f84_traject.png", order=order,
                            n_layers=n_layers, which=a.traject,
                            photos={k: load_photo(k, 200) for k in tk})
    print(f"  B -> {p}", flush=True)

if "D" in want:
    strip = json.load(open(a.strip)) if os.path.exists(a.strip) else None
    p = figdepth.readable(strip, groups, a.out, "f85_readable.png",
                          order=order, n_layers=n_layers)
    print(f"  D -> {p}", flush=True)

if "C" in want:
    if not os.path.exists(a.llava):
        print(f"  C waiting on {a.llava}", flush=True)
    else:
        dl = json.load(open(a.llava))
        lc = dl if isinstance(dl, list) else dl["cards"]
        lg = by_image(lc)
        p = figdepth.two_models(
            {"Qwen3-VL-2B": (groups, n_layers),
             "LLaVA-1.5-7B": (lg, dl.get("n_layers", 32)
                              if isinstance(dl, dict) else 32)},
            a.out, "f86_arch.png",
            photos={k: load_photo(k, 448) for k in list(groups) + list(lg)})
        print(f"  C -> {p}", flush=True)
