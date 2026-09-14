# SPDX-License-Identifier: Apache-2.0
"""V-Write over image patches, and the interventions that check it.

``hooked_axis`` runs the model with the image states at one layer replaced by
``X`` and returns the score; every attribution and every truth check below is
built on it.  ``path_reading`` integrates the score's gradient along the
straight path from a baseline to the item's states (a path-integrated
V-Write), so that its per-patch sum predicts what putting the baseline back
would do.  ``edit_mask`` and ``pool_mask`` turn an edited image and its source
into a per-patch mask of what changed.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageFilter

from jlens.hooks import ActivationRecorder
from vlens.lens import readout


def states_at(m, b, layers, last=None):
    """Image-position states at ``layers`` and the final answer-spot state and logits, for one batch."""
    last = m.n_layers - 1 if last is None else last
    S = b["input_ids"].shape[1]
    pos = readout.image_positions(m, b)
    with torch.no_grad(), ActivationRecorder(m.layers, at=[*layers, last]) as rec:
        m.forward(b)
        H = {l: rec.activations[l][0, pos.to(rec.activations[l].device)].float().cpu() for l in layers}
        hL = rec.activations[last][0, S - 1]
        lg = m.unembed(hL[None])
    return pos, H, hL, lg


def hooked_axis(m, b, fn, layer, pos, X, last=None):
    """The score with the states at ``pos`` in ``layer`` replaced by ``X`` [n, len(pos), d]; returns [n]."""
    last = m.n_layers - 1 if last is None else last
    n = X.shape[0]
    big = m.replicate(b, n) if n > 1 else b
    S = b["input_ids"].shape[1]

    def hook(mod, inp, o, _x=X, _p=pos):
        tup = not torch.is_tensor(o)
        hh = (o[0] if tup else o).clone()
        hh[:, _p.to(hh.device)] = _x.to(hh.device, hh.dtype)
        return (hh, *o[1:]) if tup else hh

    hd = m.layers[layer].register_forward_hook(hook)
    try:
        with ActivationRecorder(m.layers, at=[last]) as rec:
            m.forward(big)
            hL = rec.activations[last][:, S - 1]
            return fn(hL, m.unembed(hL))
    finally:
        hd.remove()


def path_reading(m, b, fn, layer, pos, h, base, steps=8, chunk=4, last=None):
    """Per-position path-integrated V-Write from ``base`` to ``h`` (both [len(pos), d]); returns numpy [len(pos)]."""
    dev = h - base
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    acc = torch.zeros(h.shape[0], dtype=torch.float64)
    for s0 in range(0, steps, chunk):
        al = alphas[s0: s0 + chunk]
        x = (base[None] + al[:, None, None] * dev[None]).to(m.device).detach().clone().requires_grad_(True)
        with torch.enable_grad():
            v = hooked_axis(m, b, fn, layer, pos, x, last).sum()
            g, = torch.autograd.grad(v, x)
        acc += (g.float().cpu().sum(0) * dev).sum(-1).double()
    return (acc / steps).float().numpy()


def one_point(m, b, fn, layer, pos, h, base, last=None):
    """The one-point map: the gradient at ``h`` dotted into ``h - base``."""
    with torch.enable_grad():
        x = h.to(m.device).detach().clone().requires_grad_(True)
        v = hooked_axis(m, b, fn, layer, pos, x[None], last).sum()
        g, = torch.autograd.grad(v, x)
    return ((g.float().cpu()) * (h - base)).sum(-1).numpy()


def edit_mask(src_path, edit_path, thr=28, side=448):
    """Where an edited image differs from its source: a [side, side] float mask in [0, 1]."""
    s = Image.open(src_path).convert("L")
    e = Image.open(edit_path).convert("L")
    if e.size != s.size:
        e = e.resize(s.size, Image.LANCZOS)
    d = np.abs(np.asarray(s, np.int16) - np.asarray(e, np.int16)).astype(np.uint8)
    d = Image.fromarray(d).filter(ImageFilter.MedianFilter(5))
    binm = (np.asarray(d) > thr).astype(np.float32)
    return np.asarray(Image.fromarray((binm * 255).astype(np.uint8)).resize((side, side), Image.BILINEAR), np.float32) / 255.0


def pool_mask(sq, G):
    """Mean of a [side, side] mask over the G x G patch grid, as a flat [G*G] array in patch order."""
    side = sq.shape[0]
    cell = side // G
    M = np.zeros(G * G, np.float32)
    for i in range(G):
        for j in range(G):
            M[i * G + j] = sq[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].mean()
    return M
