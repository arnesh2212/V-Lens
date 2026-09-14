# SPDX-License-Identifier: Apache-2.0
"""Can this model be read, and how big may a run be?

Run this against a repo before queueing anything on it.  Every failure this
project has had on a new model was one of four things, and all four are cheap
to find out in two minutes rather than forty:

  * the text decoder is not where the layout search expects it, so the wrapper
    cannot find ``layers``/``norm``/``embed_tokens`` at all;
  * the image tokens are not a square grid, or not the count the region code
    assumes, so every partition is wrong;
  * the weights do not fit the card, or fit it so tightly that no path chunk
    does;
  * the fitted J-lens on disk was made for a different width, so its arm is
    silently dropped from every comparison.

It loads the model, reads one photograph, reports the geometry, checks any
checkpoint you name, and prints the flags a run should use on this card.
Nothing is written and no experiment is started.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jvlm
from jvlm import corpus, readout, sizing

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--jlens", default=None, help="a checkpoint to check for width")
ap.add_argument("--max-pixels", type=int, default=None)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--device-map", default=None,
                help="'auto' to shard across cards or offload; the default "
                     "puts the whole model on one device")
a = ap.parse_args()

print(f"=== {a.repo} ===", flush=True)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        n = torch.cuda.get_device_name(i)
        f, t = torch.cuda.mem_get_info(i)
        print(f"  cuda:{i}  {n}  {f/1e9:.0f} GB free of {t/1e9:.0f}")
else:
    print("  no CUDA device visible")

kw = {}
if a.max_pixels:
    kw["max_pixels"] = a.max_pixels
if a.device_map:
    kw["device_map"] = a.device_map
try:
    m = jvlm.load_vlm(a.repo, **kw)
except Exception as e:                        # noqa: BLE001 - report, do not raise
    print(f"\nLOAD FAILED: {type(e).__name__}: {e}")
    print("  if this is a layout error, the text decoder is somewhere the "
          "search does not look; add a candidate path in jvlm/model.py")
    sys.exit(2)

print(f"\n  {m}")
print(f"  n_layers {m.n_layers}   d_model {m.d_model}   "
      f"deepstack {getattr(m, 'n_deepstack', 0)}")

imgs = corpus.coco_images(1, seed=0)
if not imgs:
    print("  no COCO photograph available locally; geometry not checked")
    sys.exit(0)
b = m.chat("What is in this photograph?", [corpus.load_image(imgs[0], square=a.square)])
ipos = readout.image_positions(m, b)
P = len(ipos)
side = int(round(P ** 0.5))
seq = int(b["input_ids"].shape[1])
square = side * side == P
print(f"  image tokens {P} of {seq} in the prompt; "
      f"{'a ' + str(side) + 'x' + str(side) + ' grid' if square else 'NOT SQUARE'}")
if not square:
    print("  !! the region code assumes a square grid; partitions would be wrong")
print(f"  image tokens are contiguous: {bool((ipos[1:] - ipos[:-1] == 1).all())}")
print(f"  first image position {int(ipos[0])}, last {int(ipos[-1])}  "
      f"(text follows: {int(ipos[-1]) < seq - 1})")

read = m.n_layers // 2
print(f"\n  {sizing.describe(m, seq_len=seq, read_layer=read)}")
p = sizing.plan(m, seq_len=seq, read_layer=read, steps=a.steps)
print(f"  suggested flags: {p.flags()}")
if p.chunk < 2:
    print("  !! a single path row barely fits; either read deeper, lower "
          "--max-pixels, or use a larger card")

if a.jlens:
    ck = a.jlens
    if not os.path.exists(ck):
        hits = glob.glob(ck)
        ck = hits[0] if hits else None
    if not ck:
        print(f"\n  J-lens: {a.jlens} not found")
    else:
        J = (jvlm.lens_from_checkpoint(ck) if ck.endswith(".ckpt")
             else __import__("jlens").lens.JacobianLens.load(ck))
        wid = next((int(v.shape[-1]) for v in J.jacobians.values()), None)
        ok = wid == m.d_model
        print(f"\n  J-lens {os.path.basename(ck)}: width {wid}, "
              f"layers {sorted(J.jacobians)[:8]}")
        print(f"  {'usable with this model' if ok else '!! WIDTH MISMATCH - '
              'the arm would be dropped from every comparison'}")

print("\n  nothing was written and no experiment was started.")
