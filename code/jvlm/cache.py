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
            key: str = "path", final: bool = True, answer_all_layers: bool = False,
            text_mean: bool = False, gen_tokens: int = 0, prompt_key: str | None = None) -> str:
    """Run every item once and write the cache.  Returns the ``.npz`` path.

    ``items`` is a list of dicts with an image path under ``key``; any other
    fields are written to the sidecar unchanged, which is where the labels
    live.  Items whose image-token count differs from the first item's are
    dropped and listed in the sidecar, so a cache never mixes geometries.

    With ``answer_all_layers`` the residual at the answer slot (the last prompt
    position) is stored at every block as ``ANS`` of shape ``[N, n_layers, d]``:
    this is what a verdict is read from, and the last row is the final state
    from which any outcome axis is recomputed exactly.  ``text_mean`` stores the
    mean over text-token positions at the read layers as ``TXT<layer>``.
    ``gen_tokens > 0`` also generates a reply per item and stores it in the
    sidecar, so behaviour labels come from what the model actually did.
    ``prompt_key`` names an item field that overrides ``prompt`` per item.
    """
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    layers = sorted(set(int(l) for l in layers))
    # the last block is what a tuned lens and a Jacobian lens transport TO, so
    # it is always cached alongside the layers being read
    LAST = int(m.n_layers) - 1
    if final and LAST not in layers:
        layers = layers + [LAST]
    acts = {l: [] for l in layers}
    ans, txt = [], {l: [] for l in layers}
    all_layers = list(range(int(m.n_layers)))
    sc = None
    if gen_tokens:
        from jvlm import tasks
        sc = tasks.Scorer(m)
    kept, dropped = [], []
    P = None
    t0 = time.time()
    for k, it in enumerate(items):
        try:
            img = corpus.load_image(it[key], square=square) if it.get(key) else None
            q = it.get(prompt_key, prompt) if prompt_key else prompt
            b = m.chat(q, [img] if img is not None else [])
            ip = readout.image_positions(m, b) if img is not None else torch.zeros(0, dtype=torch.long)
            if img is not None:
                if P is None:
                    P = len(ip)
                if len(ip) != P:
                    dropped.append(dict(i=k, reason=f"{len(ip)} image tokens, expected {P}"))
                    continue
            want = sorted(set(layers) | (set(all_layers) if answer_all_layers else set()))
            A = readout.residuals(m, b, want)
            S = b["input_ids"].shape[1]
            for l in layers:
                if img is not None and pool != "none":
                    h = A[l][ip].to(torch.float16).cpu()
                    acts[l].append(h.mean(0, keepdim=True) if pool == "mean" else h[None])
                if text_mean:
                    tp = readout.text_positions(m, b)
                    tp = tp[tp != S - 1]
                    txt[l].append(A[l][tp].float().mean(0).to(torch.float16).cpu()[None])
            if answer_all_layers:
                ans.append(torch.stack([A[l][S - 1] for l in all_layers]).to(torch.float16).cpu()[None])
            if sc is not None:
                it = dict(it, reply=sc.generate(b, max_new_tokens=gen_tokens))
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
    if ans:
        arrays["ANS"] = torch.cat(ans).numpy()
    for l, v in txt.items():
        if v:
            arrays[f"TXT{l}"] = torch.cat(v).numpy()
    # Uncompressed on purpose: zlib on a six-gigabyte cache is a single
    # thread for ten minutes while the card sits idle, and disk is not the
    # constraint on this box.
    np.savez(out, **arrays)
    side = dict(prompt=prompt, layers=layers, n=len(kept), image_tokens=P,
                pool=pool, square=square, dropped=dropped, items=kept,
                final_layer=LAST, n_layers=int(m.n_layers), answer_all_layers=answer_all_layers,
                text_mean=text_mean, gen_tokens=gen_tokens,
                minutes=(time.time() - t0) / 60)
    with open(out.replace(".npz", ".json"), "w") as f:
        json.dump(side, f)
    print(f"cached {len(kept)} items x {len(layers)} layers -> {out}  "
          f"({len(dropped)} dropped)", flush=True)
    return out


def load(path: str) -> tuple[dict[int, np.ndarray], dict]:
    """``({layer: array}, sidecar)`` for a cache written by :func:`collect`."""
    z = np.load(path)
    acts = {int(k[1:]): z[k] for k in z.files if k.startswith("L") and k[1:].isdigit()}
    side = json.load(open(path.replace(".npz", ".json")))
    side["_extra"] = {k: z[k] for k in z.files if k == "ANS" or k.startswith("TXT")}
    return acts, side


def labels(side: dict, field: str) -> np.ndarray:
    """A label vector from the sidecar's item table, in cache order."""
    vals = [it[field] for it in side["items"]]
    if all(isinstance(v, bool) for v in vals):
        return np.array(vals, dtype=int)
    uniq = sorted(set(vals))
    return np.array([uniq.index(v) for v in vals]) if len(uniq) > 2 else np.array(
        [int(v == uniq[-1]) for v in vals])


def subset(acts: dict, side: dict, keep) -> tuple[dict, dict]:
    """Restrict a loaded cache to the items where ``keep`` (a boolean array) is true."""
    import numpy as _np
    keep = _np.asarray(keep, bool)
    acts = {l: v[keep] for l, v in acts.items()}
    side = dict(side)
    side["_extra"] = {k: v[keep] for k, v in side.get("_extra", {}).items()}
    side["items"] = [it for it, k in zip(side["items"], keep) if k]
    side["n"] = int(keep.sum())
    return acts, side
