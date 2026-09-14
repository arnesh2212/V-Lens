"""What every patch of a photograph is about to make the model say.

The Anthropic workspace paper shows, for one prompt, the words each position is
disposed to produce and how they resolve with depth.  The visual version is
richer, because the positions are places in a picture -- and it is also a
comparison, because four different linear surrogates read the same patch and
disagree about what it says.

For each photograph this writes: the picture, the model's own description of it,
the objects actually annotated in it, and -- for the logit lens, the published
Jacobian lens, our tangent and the secant, at several depths -- the words read
off the image tokens, both pooled over the picture and per patch, so the reader
can see where in the frame each word came from.

Everything lands as one JSON of integer indices against a shared word list, plus
the photographs as WebP data URIs, so the gallery is a single self-contained
page.
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

sys.path.insert(0, "/workspace/vlm/code")
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import torch
from PIL import Image

import jvlm
from jvlm import cocodata, corpus, readout, tasks
from jvlm.patchfit import SecantLens, gauge_fix

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--secant", required=True)
ap.add_argument("--tangent", default=None)
ap.add_argument("--jlens", default=None)
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--top", type=int, default=6)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--thumb", type=int, default=384)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/gallery")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
S = SecantLens.load(a.secant)
LAYERS = sorted(S.maps)
FLAT = S.token_ids
TAN, TAN_AT = None, None
if a.tangent and os.path.exists(a.tangent):
    c = torch.load(a.tangent, map_location="cpu", weights_only=False)
    tt = c.get("tokens")
    if tt == FLAT:
        TAN = {l: v.float() for l, v in c["tangent"].items()}
        TAN_AT = list(range(len(FLAT)))
    elif tt and set(tt) <= set(FLAT):
        # the wide lens covers more of the vocabulary than the tangent does;
        # keep the tangent on the tokens it has and leave the rest unscored
        pos = {t: i for i, t in enumerate(FLAT)}
        TAN = {l: v.float() for l, v in c["tangent"].items()}
        TAN_AT = [pos[t] for t in tt]
        print(f"tangent covers {len(tt)} of {len(FLAT)} tokens", flush=True)
J = None
if a.jlens and os.path.exists(a.jlens):
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))
sc = tasks.Scorer(m)
print(m, f"\nlayers {LAYERS}, vocabulary {len(FLAT)}", flush=True)

WORDS = [m.tokenizer.decode([t]).strip() for t in FLAT]
vocab: list[str] = []
vidx: dict[str, int] = {}


def wid(w: str) -> int:
    if w not in vidx:
        vidx[w] = len(vocab)
        vocab.append(w)
    return vidx[w]


def topk(scores: torch.Tensor, k: int) -> list[int]:
    """Distinct surface forms, best first."""
    out, seen = [], set()
    for j in scores.argsort(descending=True).tolist():
        w = WORDS[j]
        if not w or w.lower() in seen:
            continue
        seen.add(w.lower())
        out.append(wid(w))
        if len(out) == k:
            break
    return out


def thumb(im: Image.Image) -> str:
    q = im.copy()
    q.thumbnail((a.thumb, a.thumb), Image.LANCZOS)
    b = io.BytesIO()
    q.save(b, "WEBP", quality=82, method=4)
    return "data:image/webp;base64," + base64.b64encode(b.getvalue()).decode()


items = cocodata.concept_items(a.n, min_area=0.08, seed=4242,
                               tokenizer=m.tokenizer)
cards = []
grid = None
for k, it in enumerate(items):
    im = corpus.load_image(it["path"], square=a.square)
    b = m.chat("Describe this image in one sentence.", [im])
    ip = readout.image_positions(m, b)
    if any(l not in S.means or len(S.means[l]) != len(ip) for l in LAYERS):
        continue
    g = int(round(len(ip) ** 0.5))
    grid = grid or g
    if g * g != len(ip):
        continue
    A = readout.residuals(m, b, LAYERS)
    caption = sc.generate(b, max_new_tokens=42)

    views: dict[str, dict] = {}
    for l in LAYERS:
        h = A[l][ip].float().cpu()
        d = h - S.means[l].float()
        per = {"secant": d @ S.maps[l].T + S.intercepts[l],
               "logit-lens": m.unembed(h.to(m.device)).float().cpu()[:, FLAT]}
        if TAN is not None and l in TAN:
            sub = d @ TAN[l].T                       # [P, len(TAN_AT)]
            full = torch.full((sub.shape[0], len(FLAT)), -1e9)
            full[:, TAN_AT] = sub
            per["tangent"] = full
        if J is not None and l in J.jacobians:
            lg, _ = readout.lens_logits(m, J, b, layers=[l],
                                        positions=ip.tolist())
            per["reference"] = lg[l].float().cpu()[:, FLAT]
        for name, mat in per.items():
            v = views.setdefault(name, {})
            v[str(l)] = {"pooled": topk(mat.max(0).values, a.top),
                         "patch": [topk(row, 1)[0] for row in mat]}

    cards.append({"img": thumb(im), "caption": caption.strip(),
                  "objects": sorted(it["present"]), "target": it["concept"],
                  "views": views})
    if k % 8 == 0:
        print(f"  {k}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

out = {"repo": a.repo, "layers": LAYERS, "grid": grid, "vocab": vocab,
       "depths": {str(l): round(l / (m.n_layers - 1), 3) for l in LAYERS},
       "cards": cards}
p = f"{a.out}/{tag}_gallery.json"
json.dump(out, open(p, "w"))
print(f"wrote {p}  ({len(cards)} photographs, {len(vocab)} distinct words, "
      f"{os.path.getsize(p)/1e6:.1f} MB, {time.time()-t0:.0f}s)", flush=True)
