# SPDX-License-Identifier: Apache-2.0
"""Per-layer image-token activations, written once and read by everything.

Every lens and every probe in the comparison reads the same tensors: the
residual stream at the image positions, at a fixed list of layers, for a fixed
list of items under a fixed prompt.  Caching them means the expensive part (one
forward pass per item) happens once per model, the probes run on the CPU, and
no method gets a different view of the model than another.

Format: one ``.npz`` per (model, dataset, prompt) holding ``L<layer>`` arrays of
shape ``[N, P, d]`` in float16, plus ``ids`` and a JSON sidecar with the
prompt, the layers, the image geometry and the item table.  ``pool="mean"``
stores ``[N, d]`` instead when only a per-image vector is wanted.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from jvlm import corpus, readout


def collect(m, items: list[dict], layers: list[int], prompt: str, out: str, *,
            pool: str | None = None, square: int = 448, every: int = 25,
            key: str = "path", final: bool = True) -> str:
    """Run every item once and write the cache.  Returns the ``.npz`` path.

    ``items`` is a list of dicts with an image path under ``key``; any other
    fields are written to the sidecar unchanged, which is where the labels
    live.  Items whose image-token count differs from the first item's are
    dropped and listed in the sidecar, so a cache never mixes geometries.
    """
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    layers = sorted(set(int(l) for l in layers))
    # the last block is what a tuned lens and a Jacobian lens transport TO, so
    # it is always cached alongside the layers being read
    LAST = int(m.n_layers) - 1
    if final and LAST not in layers:
        layers = layers + [LAST]
    acts = {l: [] for l in layers}
    kept, dropped = [], []
    P = None
    t0 = time.time()
    for k, it in enumerate(items):
        try:
            img = corpus.load_image(it[key], square=square)
            b = m.chat(prompt, [img])
            ip = readout.image_positions(m, b)
            if P is None:
                P = len(ip)
            if len(ip) != P:
                dropped.append(dict(i=k, reason=f"{len(ip)} image tokens, expected {P}"))
                continue
            A = readout.residuals(m, b, layers)
            for l in layers:
                h = A[l][ip].to(torch.float16).cpu()
                acts[l].append(h.mean(0, keepdim=True) if pool == "mean" else h[None])
            kept.append(it)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            dropped.append(dict(i=k, reason="oom"))
        if (k + 1) % every == 0:
            el = time.time() - t0
            print(f"  {k + 1}/{len(items)}  {el:.0f}s  eta {el / (k + 1) * (len(items) - k - 1) / 60:.1f}m",
                  flush=True)
    arrays = {f"L{l}": torch.cat(v).numpy() for l, v in acts.items() if v}
    if pool == "mean":
        arrays = {k: v[:, 0] if v.ndim == 3 else v for k, v in arrays.items()}
    np.savez_compressed(out, **arrays)
    side = dict(prompt=prompt, layers=layers, n=len(kept), image_tokens=P,
                pool=pool, square=square, dropped=dropped, items=kept,
                final_layer=LAST, minutes=(time.time() - t0) / 60)
    with open(out.replace(".npz", ".json"), "w") as f:
        json.dump(side, f)
    print(f"cached {len(kept)} items x {len(layers)} layers -> {out}  "
          f"({len(dropped)} dropped)", flush=True)
    return out


def load(path: str) -> tuple[dict[int, np.ndarray], dict]:
    """``({layer: array}, sidecar)`` for a cache written by :func:`collect`."""
    z = np.load(path)
    acts = {int(k[1:]): z[k] for k in z.files if k.startswith("L")}
    side = json.load(open(path.replace(".npz", ".json")))
    return acts, side


def labels(side: dict, field: str) -> np.ndarray:
    """A label vector from the sidecar's item table, in cache order."""
    vals = [it[field] for it in side["items"]]
    if all(isinstance(v, bool) for v in vals):
        return np.array(vals, dtype=int)
    uniq = sorted(set(vals))
    return np.array([uniq.index(v) for v in vals]) if len(uniq) > 2 else np.array(
        [int(v == uniq[-1]) for v in vals])
