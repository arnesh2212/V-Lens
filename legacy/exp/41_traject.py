"""Layerwise trajectories: what a patch is about to say, all the way up.

The classic lens picture is a trajectory -- one position, every layer, the words
resolving with depth.  With two heads there are two trajectories, and the
interesting thing is where they diverge.

For each photograph this records, at every layer of the language model:

    say    the top words read off the image tokens through the unembedding
    do     the same, through the map fit to predict what the patch does
           (only at the layers the map was fit at)

both pooled over the frame and for the single patch that scores highest on the
photographed object, so a trajectory can be followed for one place in the
picture rather than for the whole thing.

Emitted as one JSON of integer indices against a shared word list, with the
photographs as WebP data URIs, so the strip is a self-contained page.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import time

sys.path.insert(0, "code")
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import torch
from PIL import Image

import jvlm
from jvlm import cocodata, corpus, readout, tasks, vmap

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--mean-images", type=int, default=120)
ap.add_argument("--jlens", default=None)
ap.add_argument("--n", type=int, default=36)
ap.add_argument("--top", type=int, default=5)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--thumb", type=int, default=340)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/traject")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
ALL = list(range(max(m.n_deepstack, 1), m.n_layers - 1))
FIT = ALL                       # nothing is fitted, so every layer is available
CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
FLAT = sorted({t for v in CAT_IDS.values() for t in v})
POSI = {t: i for i, t in enumerate(FLAT)}
sc = tasks.Scorer(m)
J = None
if a.jlens and os.path.exists(a.jlens):
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))
print(m, f"\nboth heads at all {len(ALL)} layers, {len(FLAT)} tokens",
      flush=True)

WORDS = [m.tokenizer.decode([t]).strip() for t in FLAT]
vocab: list[str] = []
vidx: dict[str, int] = {}


def wid(w):
    if w not in vidx:
        vidx[w] = len(vocab)
        vocab.append(w)
    return vidx[w]


def topk(v, k):
    out, seen = [], set()
    for j in v.argsort(descending=True).tolist():
        w = WORDS[j]
        if not w or w.lower() in seen:
            continue
        seen.add(w.lower())
        out.append(wid(w))
        if len(out) == k:
            break
    return out


def thumb(im):
    q = im.copy()
    q.thumbnail((a.thumb, a.thumb), Image.LANCZOS)
    b = io.BytesIO()
    q.save(b, "WEBP", quality=82, method=4)
    return "data:image/webp;base64," + base64.b64encode(b.getvalue()).decode()


# the per-position mean: the only corpus quantity the method uses
pool = cocodata.concept_items(6000, min_area=0.04, seed=7)
b_ = m.chat("Describe this image in one sentence.",
            [corpus.load_image(pool[0]["path"], square=a.square)])
P0 = len(readout.image_positions(m, b_))
acc = {l: torch.zeros(P0, m.d_model, dtype=torch.float64) for l in ALL}
bank = {l: [] for l in ALL}
nm = 0
for it in pool[: a.mean_images]:
    bb = m.chat("Describe this image in one sentence.",
                [corpus.load_image(it["path"], square=a.square)])
    ipx = readout.image_positions(m, bb)
    if len(ipx) != P0:
        continue
    AA = readout.residuals(m, bb, ALL)
    for l in ALL:
        v = AA[l][ipx].float().cpu()
        acc[l] += v.double()
        if len(bank[l]) < 12:
            bank[l].append(v)
    nm += 1
MEANS = {l: (acc[l] / max(nm, 1)).float() for l in ALL}
print(f"means from {nm} photographs ({time.time()-t0:.0f}s)", flush=True)

items = cocodata.concept_items(a.n, min_area=0.1, seed=808,
                               tokenizer=m.tokenizer)
cards, grid = [], None
for k, it in enumerate(items):
    im = corpus.load_image(it["path"], square=a.square)
    b = m.chat("Describe this image in one sentence.", [im])
    ip = readout.image_positions(m, b)
    g = int(round(len(ip) ** 0.5))
    grid = grid or g
    A = readout.residuals(m, b, ALL)
    if len(ip) != P0:
        continue
    ids = [t for t in cocodata.concept_token_ids(m.tokenizer, it["concept"])
           if t in POSI]
    if not ids:
        continue
    col = [POSI[t] for t in ids]

    # the patch that most strongly carries the photographed object, chosen once
    # at the deepest fitted layer and then followed all the way up
    hd = A[FIT[-1]][ip].float().cpu()
    sd = m.unembed(hd.to(m.device)).float().cpu()[:, FLAT][:, col].max(1).values
    star = int(sd.argmax())
    # the whole category vocabulary, so "which word is this patch pushing"
    # is a real ranking rather than a foregone one.  One backward per word
    # covers every layer at once, which is what makes this affordable.
    DO = vmap.do_map(m, b, FLAT, ALL, ip.tolist(),
                     means={l: bank[l][k % len(bank[l])] for l in ALL})

    say, do = {}, {}
    for l in ALL:
        h = A[l][ip].float().cpu()
        s_ = m.unembed(h.to(m.device)).float().cpu()[:, FLAT]
        s_ = s_ - s_.mean(1, keepdim=True)    # a logit difference, not a logit
        say[str(l)] = {"pooled": topk(s_.max(0).values, a.top),
                       "star": topk(s_[star], a.top),
                       "rank": int((s_.max(0).values >
                                    s_.max(0).values[col].max()).sum())}
        d_ = DO[l].T                                  # [P, |vocab|]
        best_d = d_.max(0).values
        do[str(l)] = {"pooled": topk(best_d, a.top),
                      "star": topk(d_[star], a.top),
                      "rank": int((best_d > best_d[col].max()).sum())}
    cards.append({"img": thumb(im), "caption": sc.generate(b, 40).strip(),
                  "objects": sorted(it["present"]), "target": it["concept"],
                  "star": star, "say": say, "do": do})
    if k % 6 == 0:
        print(f"  {k}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

out = {"repo": a.repo, "layers": ALL, "fit_layers": FIT, "grid": grid,
       "vocab": vocab, "n_layers": m.n_layers,
       "depths": {str(l): round(l / (m.n_layers - 1), 3) for l in ALL},
       "cards": cards}
p = f"{a.out}/{tag}_traject.json"
json.dump(out, open(p, "w"))
print(f"wrote {p}  ({len(cards)} photographs, {len(vocab)} words, "
      f"{os.path.getsize(p)/1e6:.1f} MB, {time.time()-t0:.0f}s)", flush=True)
