"""One photograph, every layer, three lenses: where does the word separate?

The comparison figures average over a hundred and fifty photographs and four
depths, which is the right way to claim a result and the wrong way to *see* one.
This measures the same quantity on single photographs at *every* layer the
J-lens covers, so a reader can watch the object's word rise inside its own box
and stay flat outside it, layer by layer.

The quantity is a separation, in standard deviations of the vocabulary:

    sep(lens, layer) = mean over patches inside the box of z(word)
                     - mean over patches outside it

where z is the reading standardised down the vocabulary at each patch, exactly
as in the three-lens comparison.  A lens with no position in its transport can
only move both means together, so its separation is pinned near zero however
good its vocabulary is; that is the whole point of the region coordinate.

Also emitted, for the labelled strip beneath the ribbon: the single word each
lens reads most strongly *inside* the box at each layer.  That is the discrete,
readable version of the same measurement -- a word per cell, not a colour wash.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, "code")
for n in ("httpx", "httpcore", "urllib3", "filelock", "matplotlib", "PIL"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch

import jvlm
from jvlm import cocodata, corpus, readout, stats as st, vmap

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens", default=None)
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--mean-images", type=int, default=80)
ap.add_argument("--boot", type=int, default=400)
ap.add_argument("--ig-layers", type=int, default=0,
                help="how many layers also get the path-integrated reading "
                     "(0 = off).  ig_map costs one backward per layer per "
                     "token, so this is deliberately a subset.")
ap.add_argument("--ig-steps", type=int, default=4)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-frac", type=float, default=0.40)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/ribbon")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
J = None
if a.jlens and os.path.exists(a.jlens):
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))
    LAYERS = sorted(l for l in J.jacobians if l < m.n_layers - 1)
else:
    LAYERS = list(range(max(m.n_deepstack, 1), m.n_layers - 1))
CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
FLAT = sorted({t for v in CAT_IDS.values() for t in v})
POSI = {t: i for i, t in enumerate(FLAT)}
WORD = {}
for c, ids in CAT_IDS.items():
    for t in ids:
        WORD.setdefault(t, c)
IGL = []
if a.ig_layers > 0:
    idx = np.linspace(0, len(LAYERS) - 1, min(a.ig_layers, len(LAYERS)))
    IGL = sorted({LAYERS[int(round(v))] for v in idx})
print(m, f"\n{len(LAYERS)} layers {LAYERS[0]}..{LAYERS[-1]}   "
         f"{len(CATS)} categories, {len(FLAT)} tokens"
         + (f"\npath-integrated at {IGL} ({a.ig_steps} steps)" if IGL else ""),
      flush=True)

PROMPT = "Describe this image."


def prep(p):
    return m.chat(PROMPT, [corpus.load_image(p, square=a.square)])


pool = cocodata.concept_items(6000, min_area=0.05, seed=7)
b0 = prep(pool[0]["path"])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
nrm = {l: torch.zeros(P, dtype=torch.float64) for l in LAYERS}
nm = 0
for it in pool[: a.mean_images]:
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    A = readout.residuals(m, b, LAYERS)
    for l in LAYERS:
        v = A[l][ip].float().cpu()
        acc[l] += v.double()
        nrm[l] += v.norm(dim=-1).double()
    nm += 1
MEANS = {l: (acc[l] / max(nm, 1)).float() for l in LAYERS}
NORM = {l: (nrm[l] / max(nm, 1)).float() for l in LAYERS}
# attention-register patches: a handful of very high norm tokens that every
# lens lights up on and that belong to no object.  Excluded from both sides.
REG = set()
for l in LAYERS:
    v = NORM[l]
    REG |= set(torch.nonzero(v > v.mean() + 3 * v.std()).flatten().tolist())
REG = sorted(REG)
print(f"means from {nm} images; {P} tokens on {G}x{G}; "
      f"{len(REG)} register patches excluded ({time.time()-t0:.0f}s)",
      flush=True)


def sep_and_word(mm, inside, outside, rng, col):
    """(separation, lo, hi, top word inside) from a ``[k, P]`` reading."""
    z = vmap.normalise(mm)
    heat = z[col].max(0).values.numpy()
    ins, out = heat[inside], heat[outside]
    sep = float(ins.mean() - out.mean())
    bi = rng.integers(0, len(ins), size=(a.boot, len(ins)))
    bo = rng.integers(0, len(out), size=(a.boot, len(out)))
    d = ins[bi].mean(1) - out[bo].mean(1)
    tw = WORD.get(FLAT[int(z[:, inside].mean(1).argmax())], "?")
    return sep, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), tw


items = cocodata.concept_items(400, min_area=0.09, seed=808,
                               tokenizer=m.tokenizer)
rows, cards = [], []
kept = 0
for k, it in enumerate(items):
    if kept >= a.n:
        break
    c = it["concept"]
    boxes = cocodata.object_boxes(it["image_id"], c)
    if not boxes:
        continue
    inside = sorted({t for bx in boxes for t in cocodata.box_tokens(bx, G)}
                    - set(REG))
    outside = sorted(set(range(P)) - set(inside) - set(REG))
    if not (6 <= len(inside) <= int(a.max_frac * P)) or len(outside) < 20:
        continue
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    ids = CAT_IDS[c]
    col = [POSI[t] for t in ids]
    A = readout.residuals(m, b, LAYERS)
    rng = np.random.default_rng(900 + k)

    DO = vmap.do_map(m, b, FLAT, LAYERS, ip.tolist(), means=MEANS)
    IG = {}
    if IGL:
        IG = vmap.ig_map(m, b, FLAT, IGL, ip.tolist(), means=MEANS,
                         steps=a.ig_steps)
    for l in LAYERS:
        h = A[l][ip].float()
        maps = {"logit lens":
                m.unembed(h.to(m.device)).float().cpu()[:, FLAT].T.contiguous()}
        if J is not None and l in J.jacobians:
            jl, _ = readout.lens_logits(m, J, b, layers=[l],
                                        positions=ip.tolist())
            maps["J-lens"] = jl[l].float().cpu()[:, FLAT].T.contiguous()
        maps["VJ-Lens"] = DO[l]
        if l in IG:
            maps["VJ-Lens (path)"] = IG[l]
        for name, mm in maps.items():
            s, lo, hi, w = sep_and_word(mm, inside, outside, rng, col)
            rows.append(dict(i=kept, image_id=int(it["image_id"]), concept=c,
                             lens=name, layer=l,
                             depth=round(l / (m.n_layers - 1), 4),
                             n_in=len(inside), sep=s, lo=lo, hi=hi, word=w))
    # a plain thumbnail with the box outlined -- no colour wash over it
    try:
        from PIL import Image, ImageDraw
        im = corpus.load_image(it["path"], square=a.square).convert("RGB")
        im = im.resize((336, 336), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        for bx in boxes:
            x, y, w_, h_ = bx
            d.rectangle([x * 336, y * 336, (x + w_) * 336, (y + h_) * 336],
                        outline=(235, 104, 52), width=4)
        im.save(f"{a.out}/{tag}_img{kept:02d}.png")
    except Exception as e:
        print("  thumb failed", e, flush=True)
    cards.append(dict(i=kept, image_id=int(it["image_id"]), concept=c,
                      n_in=len(inside), n_out=len(outside),
                      thumb=f"{tag}_img{kept:02d}.png"))
    kept += 1
    el = time.time() - t0
    print(f"  {kept}/{a.n}  {c}  {len(inside)} patches inside  {el:.0f}s",
          flush=True)
    pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_ribbon.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_ribbon.parquet")
piv = df.pivot_table(index="layer", columns="lens", values="sep")
print("\nSEPARATION inside the box minus outside, in sd of the vocabulary\n",
      piv.round(3).to_string())
best = df.groupby("lens").sep.max().round(4)
print("\nPEAK SEPARATION PER LENS\n", best.to_string())
json.dump({"repo": a.repo, "layers": LAYERS, "grid": G, "n_patches": P,
           "n_layers": m.n_layers, "n": int(df.i.nunique()),
           "registers": REG, "cards": st.jsonable(cards),
           "peak": st.jsonable(best.to_dict()),
           "by_layer": st.jsonable(piv.round(4).to_dict("index")),
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
print("done", flush=True)
