"""Data for the interactive workspace viewer.

Anthropic's slice view shows, for one prompt, what every position at every layer
is disposed to make the model say.  The visual version is richer, because the
positions are places in a photograph: hovering a patch should say what that patch
is about to make the model say, and moving through depth should show the words
resolving.

This precomputes that for a handful of images, at several layers, in three
conditions -- untouched, with one concept erased from the production channel, and
with an absent concept injected -- so the viewer can show the intervention doing
its work rather than only asserting it.

Everything is emitted as one compact JSON: a word dictionary plus integer indices,
the photographs as WebP data URIs, and the model's own description of each.
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

import jvlm
from jvlm import cocodata, corpus, readout, scenes, tasks

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--topk", type=int, default=5)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/viewer.json")
a = ap.parse_args()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lens = (jvlm.lens_from_checkpoint(a.lens) if a.lens.endswith(".ckpt")
        else __import__("jlens").lens.JacobianLens.load(a.lens))
sc = tasks.Scorer(m)
LAYERS = [l for l in lens.source_layers
          if 0.15 <= l / (m.n_layers - 1) <= 0.95][::3]
print(m, lens, f"\nlayers {LAYERS}", flush=True)

# The readout is restricted to plain word tokens: the raw top-1 is often a
# punctuation or byte fragment, which is a true fact about the model and an
# unreadable one in a viewer.
KEEP = []
for t in range(m.vocab_size):
    piece = m.tokenizer.decode([t]).strip()
    if 2 < len(piece) <= 14 and piece.isalpha() and piece.isascii():
        KEEP.append(t)
KEEP_T = torch.tensor(KEEP)
print(f"readout vocabulary: {len(KEEP)} word tokens", flush=True)


def token_grid(batch):
    thw = batch["image_grid_thw"][0].tolist()
    merge = getattr(m.processor.image_processor, "merge_size", 2)
    return thw[1] // merge, thw[2] // merge


def uri(im, max_w=420):
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), 1)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="WEBP", quality=82, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


VOCAB: list[str] = []
INDEX: dict[str, int] = {}


def widx(w: str) -> int:
    if w not in INDEX:
        INDEX[w] = len(VOCAB)
        VOCAB.append(w)
    return INDEX[w]


def slice_for(batch, ipos):
    """``{layer: {"w": [[idx]*k per patch], "c": [confidence per patch]}}``."""
    out = {}
    ll, _ = readout.lens_logits(m, lens, batch, layers=LAYERS,
                                positions=ipos.tolist())
    for l in LAYERS:
        sub = ll[l][:, KEEP_T]                         # [P, |KEEP|]
        top = sub.topk(a.topk, dim=-1).indices
        lp = torch.log_softmax(sub.float(), -1)
        conf = lp.max(-1).values
        words = [[widx(m.tokenizer.decode([KEEP[int(j)]]).strip())
                  for j in row] for row in top]
        out[str(l)] = {"w": words, "c": [round(float(c), 2) for c in conf]}
    return out


items = cocodata.concept_items(a.n, min_area=0.12, seed=41, hard=True,
                               tokenizer=m.tokenizer)
images = []
t0 = time.time()
for k, it in enumerate(items):
    raw = corpus.load_image(it["path"])
    c, d = it["concept"], it["distractor"]
    b = m.chat("Describe this image.", [raw])
    ipos = readout.image_positions(m, b)
    gh, gw = token_grid(b)
    if gh * gw != len(ipos):
        print(f"  skip {it['image_id']}: grid mismatch")
        continue

    entry = {"id": int(it["image_id"]), "gh": gh, "gw": gw,
             "concept": c, "distractor": d, "src": uri(raw),
             "caption": (cocodata.captions(it["image_id"]) or [""])[0].strip(),
             "conds": {}}
    with torch.no_grad():
        entry["says"] = sc.generate(b, 34).strip()

    entry["conds"]["clean"] = slice_for(b, ipos)

    ids_c = cocodata.concept_token_ids(m.tokenizer, c)
    spec_c = {l: readout.lens_vectors(m, lens, ids_c, l)
              for l in lens.source_layers}
    iv = readout.Intervention(m, spec_c, channel="all", mode="ablate", alpha=1.0)
    with iv:
        bb = iv.set_batch(m.chat("Describe this image.", [raw]))
        entry["conds"]["erased"] = slice_for(bb, ipos)
        with torch.no_grad():
            entry["says_erased"] = sc.generate(bb, 34).strip()

    ids_d = cocodata.concept_token_ids(m.tokenizer, d)
    spec_d = {l: readout.lens_vectors(m, lens, ids_d, l)
              for l in lens.source_layers}
    iv = readout.Intervention(m, spec_d, channel="img", mode="add", alpha=1.0)
    with iv:
        bb = iv.set_batch(m.chat("Describe this image.", [raw]))
        entry["conds"]["injected"] = slice_for(bb, ipos)
        with torch.no_grad():
            entry["says_injected"] = sc.generate(bb, 34).strip()

    images.append(entry)
    print(f"  {k+1}/{len(items)}  {it['image_id']}  {time.time()-t0:.0f}s",
          flush=True)

# the framed variants, so the viewer also shows what a benchmark frame does
extra = []
for k, it in enumerate(items[:3]):
    raw = corpus.load_image(it["path"])
    for name, img in (("benchmark harness", scenes.benchmark_frame(raw, seed=k)),
                      ("official header", scenes.authority_frame(raw, seed=k))):
        b = m.chat("Describe this image.", [img])
        ipos = readout.image_positions(m, b)
        gh, gw = token_grid(b)
        if gh * gw != len(ipos):
            continue
        with torch.no_grad():
            says = sc.generate(b, 34).strip()
        extra.append({"id": int(it["image_id"]), "gh": gh, "gw": gw,
                      "concept": name, "distractor": "", "src": uri(img),
                      "caption": f"the same photograph, in a {name}",
                      "says": says,
                      "conds": {"clean": slice_for(b, ipos)}})
        print(f"  framed {name} {it['image_id']}", flush=True)

payload = {"model": a.repo, "layers": [int(l) for l in LAYERS],
           "n_layers": m.n_layers, "vocab": VOCAB,
           "images": images + extra}
json.dump(payload, open(a.out, "w"), separators=(",", ":"))
print(f"\nwrote {a.out}  {os.path.getsize(a.out)/1e6:.1f} MB, "
      f"{len(payload['images'])} images, {len(VOCAB)} words", flush=True)
