# SPDX-License-Identifier: Apache-2.0
"""V-Write over image patches, and the interventions that check it.

``hooked_axis`` runs the model with the image states at one layer replaced by
``X`` and returns the score; every attribution and every truth check below is
built on it.  ``path_reading`` integrates the score's gradient along the
straight path from a baseline to the item's states (a path-integrated
V-Write), so that its per-patch sum predicts what putting the baseline back
would do.  ``edit_mask`` and ``pool_mask`` turn an edited image and its source
into a per-patch mask of what changed.

The published attribution methods V-Write is compared with are here too, each
returning one score per image token:

* integrated gradients (Sundararajan et al. 2017) twice: ``path_reading`` from
  a zero hidden state at the block, and ``ig_pixels`` from a black image in
  pixel space through the whole model, the paper's own baseline for images;
* ``grad_x_act``, the one-point map from zero, and attribution patching (AtP,
  Nanda 2023; Kramar et al. 2024 Eq. 4, without AtP*'s GradDrop), the
  one-point map from the per-picture mean;
* ``act_patch``, which measures one patch at a time what AtP estimates;
* ``rise`` over ``rise_masks`` (Petsiuk et al. 2018): random binary s x s
  grids, upsampled bilinearly with a random shift and scored as soft masks;
* ``kernel_shap_groups`` (Lundberg and Lee 2017), with the sampler, pairing,
  full enumeration of small coalition sizes and the efficiency-constrained
  weighted least squares of the shap package's KernelExplainer;
* ``grad_cam_vision`` (Selvaraju et al. 2017) at the vision tower's last
  block, the ViT adaptation, pooled onto the language model's image tokens.

``SkipTo`` runs only the blocks after the one the states are replaced at,
which gives the same scores for about half the forward cost, and checks that
it does before it is trusted.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image, ImageFilter

from vlens.lens.hooks import ActivationRecorder
from vlens.lens import readout
from vlens.lens.room import is_oom, with_room


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


#: the largest group of integration points that fitted on the card lately
_CHUNK_HINT: dict[str, int] = {}


def path_reading(m, b, fn, layer, pos, h, base, steps=8, chunk=4, last=None, log=None):
    """Per-position path-integrated V-Write from ``base`` to ``h`` (both [len(pos), d]); returns numpy [len(pos)].

    The integration points are independent, so a chunk that will not fit is
    halved and the remaining points are taken in smaller groups. The sum over
    all of them, and therefore the reading, is unchanged.
    """
    dev = h - base
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    acc = torch.zeros(h.shape[0], dtype=torch.float64)

    def one(al):
        x = (base[None] + al[:, None, None] * dev[None]).to(m.device).detach().clone().requires_grad_(True)
        with torch.enable_grad():
            v = hooked_axis(m, b, fn, layer, pos, x, last).sum()
            g, = torch.autograd.grad(v, x)
        return (g.float().cpu().sum(0) * dev).sum(-1).double()

    # what fitted last time is where this starts. Rediscovering the limit on
    # every record would cost a failed pass per step down, and there are
    # thousands of records; but the other lane does finish, so every so often
    # this reaches for more and keeps it if it fits.
    s0 = 0
    hint = _CHUNK_HINT.get("path", int(chunk))
    _CHUNK_HINT["n"] = _CHUNK_HINT.get("n", 0) + 1
    reach = _CHUNK_HINT["n"] % 25 == 0
    c = max(min(int(chunk), hint * 2 if reach else hint), 1)
    while s0 < steps:
        al = alphas[s0: s0 + c]
        try:
            acc += one(al) if len(al) > 1 else with_room(lambda: one(al), log=log)
        except Exception as e:  # noqa: BLE001
            if not is_oom(e) or len(al) <= 1:
                raise
            c = max(c // 2, 1)
            _CHUNK_HINT["path"] = c
            torch.cuda.empty_cache()
            if log:
                log(f"    not enough room; {c} integration points at a time")
            continue
        s0 += len(al)
    _CHUNK_HINT["path"] = c
    return (acc / steps).float().numpy()


def hooked_axis_many(m, b, fn, layer, pos, X, batch: int = 32, last=None,
                     log=None) -> "np.ndarray":
    """``hooked_axis`` over a list of replacement states, in batches.

    Two lanes share one card, so the room available to a batch changes while a
    run is under way.  A batch that will not fit is halved and retried rather
    than ending the job; the arithmetic is the same either way.
    """
    out, i, bs = [], 0, max(int(batch), 1)
    while i < len(X):
        run = lambda: hooked_axis(m, b, fn, layer, pos,
                                  torch.stack(list(X[i: i + bs])).to(m.device), last)
        try:
            got = run() if bs > 1 else with_room(run, log=log)
        except Exception as e:  # noqa: BLE001
            if not is_oom(e) or bs <= 1:
                raise
            bs = max(bs // 2, 1)
            torch.cuda.empty_cache()
            if log:
                log(f"    not enough room; reading {bs} at a time")
            continue
        out.append(got.float().cpu().numpy())
        i += bs
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def grad_at(m, b, fn, layer, pos, h, last=None):
    """The score's gradient with respect to the states at ``pos`` in ``layer``, taken at ``h``; [len(pos), d] on the CPU."""
    with torch.enable_grad():
        x = h.to(m.device).detach().clone().requires_grad_(True)
        v = hooked_axis(m, b, fn, layer, pos, x[None], last).sum()
        g, = torch.autograd.grad(v, x)
    return g.float().cpu()


def one_point(m, b, fn, layer, pos, h, base, last=None, grad=None):
    """The one-point map: the gradient at ``h`` dotted into ``h - base``.

    From the per-picture mean this is attribution patching; from zero it is
    gradient times activation. Both use the same gradient, so a caller that
    wants both passes the one ``grad_at`` returned rather than paying for a
    second backward pass.
    """
    g = grad_at(m, b, fn, layer, pos, h, last) if grad is None else grad
    return (g * (h - base)).sum(-1).numpy()


def grad_x_act(m, b, fn, layer, pos, h, last=None, grad=None):
    """Gradient times activation (Shrikumar et al. 2016) over the image states at ``layer``; numpy [len(pos)].

    The gradient at the observed state dotted into the state itself, which is
    the one-point map with a zero baseline.
    """
    return one_point(m, b, fn, layer, pos, h, torch.zeros_like(h), last, grad=grad)


class _Replaced:
    """Copies of ``h`` with the patches each row of ``drop`` marks replaced by ``base``.

    ``drop`` is a boolean [n, len(pos)]. A copy is made only when a batch asks
    for it, so a few hundred replacement sets never sit in memory at once;
    indexing with a slice gives the stacked batch [k, len(pos), d].
    """

    def __init__(self, h, base, drop):
        self.h, self.base = h, base
        self.drop = torch.as_tensor(np.asarray(drop, bool))

    def __len__(self):
        return int(self.drop.shape[0])

    def __getitem__(self, i):
        return torch.where(self.drop[i][..., None], self.base, self.h)


def act_patch(m, b, fn, layer, pos, h, s0, base=None, batch=32, last=None, log=None):
    """Exact activation patching, one patch at a time; numpy float32 [len(pos)].

    Each patch's score is the fall in the answer score when the state of that
    patch alone, at ``layer``, is replaced by its ``base`` and every other
    patch is left as it was. ``s0`` is the unpatched score. Given the
    replacement attribution patching is taken from, this is the quantity that
    attribution patching (AtP; Nanda 2023, Kramar et al. 2024 Eq. 4) estimates
    with one gradient; here it is measured, with one forward per patch, taken
    in batches. Left out, ``base`` is zero, which makes this occlusion one
    patch at a time, and makes it the single-patch form of the zero
    replacement the rankings are graded by, so a caller comparing methods
    passes the replacement the one-point map uses instead.
    """
    base = torch.zeros_like(h) if base is None else base
    X = _Replaced(h, base, np.eye(h.shape[0], dtype=bool))
    with torch.no_grad():
        after = hooked_axis_many(m, b, fn, layer, pos, X, batch=batch, last=last, log=log)
    return (float(s0) - after.astype(np.float64)).astype(np.float32)


class _Blended:
    """``base + M_i * (h - base)`` for each soft mask ``M_i`` (a row of ``M``, [n, len(pos)]).

    RISE's masks take values between 0 and 1 after upsampling, so a patch is
    partly kept: its state is moved that fraction of the way from the
    replacement to its own. Copies are made a batch at a time.
    """

    def __init__(self, h, base, M):
        self.h, self.base = h, base
        self.M = torch.as_tensor(np.asarray(M, np.float32))

    def __len__(self):
        return int(self.M.shape[0])

    def __getitem__(self, i):
        w = self.M[i][..., None].to(self.h.dtype)
        return self.base + w * (self.h - self.base)


# --------------------------------------------------------------------- RISE

def rise_masks(rows, cols, n, s=7, p=0.5, rng=None, fine=8, chunk=500):
    """RISE's random masks on a ``rows x cols`` token grid; float32 [n, rows*cols] in [0, 1].

    Made as Petsiuk et al. (2018, Sec. 3.2) make them: an ``s x s`` grid whose
    cells are on with probability ``p``, upsampled bilinearly to ``(s + 1)``
    cells' worth and cropped to the picture at a uniformly random shift of up
    to one cell in each direction. The picture is taken at ``fine`` pixels a
    token and each token gets the mean of its pixels, so the shift is finer
    than a token, as in the original. (The reference resizes with
    ``mode='reflect'``; the bilinear resize here clamps at the border instead,
    which differs only in the outer half cell of the oversized mask, most of
    which the crop discards.)
    """
    rng = np.random.default_rng(0) if rng is None else rng
    H, W = rows * fine, cols * fine
    ch, cw = int(math.ceil(H / s)), int(math.ceil(W / s))
    out = np.empty((int(n), rows * cols), np.float32)
    for i0 in range(0, int(n), chunk):
        k = min(chunk, int(n) - i0)
        grid = torch.from_numpy((rng.random((k, s, s)) < p).astype(np.float32))
        up = torch.nn.functional.interpolate(grid[:, None], size=((s + 1) * ch, (s + 1) * cw),
                                             mode="bilinear", align_corners=False)[:, 0]
        dx, dy = rng.integers(0, ch, k), rng.integers(0, cw, k)
        crop = torch.stack([up[j, dx[j]:dx[j] + H, dy[j]:dy[j] + W] for j in range(k)])
        out[i0:i0 + k] = crop.view(k, rows, fine, cols, fine).mean((2, 4)).reshape(k, -1).numpy()
    return out


def rise_scores(m, b, fn, layer, pos, h, base, M, batch=32, last=None, log=None):
    """The score with each soft mask applied (kept fraction of the way from ``base``); float64 [n]."""
    with torch.no_grad():
        f = hooked_axis_many(m, b, fn, layer, pos, _Blended(h, base, M), batch=batch,
                             last=last, log=log)
    return f.astype(np.float64)


def rise(M, f, p=0.5):
    """RISE saliency (Petsiuk et al. 2018, Eq. 6): ``sum_i f(M_i) M_i / (N p)``; numpy float32 [P]."""
    M = np.asarray(M, np.float64)
    f = np.asarray(f, np.float64)
    return (M.T @ f / (len(f) * p)).astype(np.float32)


# --------------------------------------------------------------- KernelSHAP

def shap_default_samples(M):
    """shap's ``nsamples='auto'``: ``2 M + 2048``, capped at the ``2^M - 2`` coalitions there are."""
    return int(min(2 * M + 2 ** 11, 2 ** 30 if M > 30 else 2 ** M - 2))


def shap_coalitions(M, nsamples, rng=None):
    """The coalitions KernelSHAP scores and their kernel weights; ``(Z int8 [n, M], w float64 [n], sizes_enumerated)``.

    A line-for-line port of ``shap.KernelExplainer.explain``'s sampler: the
    Shapley kernel's mass per coalition size, with a size and its complement
    counted together; sizes from the smallest (and their complements) are
    enumerated in full while the budget covers them at their share of the
    kernel mass, each coalition weighted by that mass over its count; the rest
    of the budget is drawn by size from the remaining mass, each draw paired
    with its complement, a repeat adding to the weight of the first draw
    instead of costing a forward, and the drawn weights rescaled to the mass
    left. With M = 16 and 2080 samples, sizes 1 and 15 (32 coalitions) and 2
    and 14 (240) are enumerated and 1808 are drawn from sizes 3 to 13.
    """
    from itertools import combinations
    rng = np.random.default_rng(0) if rng is None else rng
    nsamples = int(min(int(nsamples), 2 ** 30 if M > 30 else 2 ** M - 2))
    n_sizes = int(math.ceil((M - 1) / 2.0))
    n_paired = int(math.floor((M - 1) / 2.0))
    weight_vector = np.array([(M - 1.0) / (i * (M - i)) for i in range(1, n_sizes + 1)])
    weight_vector[:n_paired] *= 2
    weight_vector /= weight_vector.sum()
    Z, W = [], []
    n_full, left = 0, nsamples
    rem = weight_vector.copy()
    for size in range(1, n_sizes + 1):
        nsub = math.comb(M, size) * (2 if size <= n_paired else 1)
        if left * rem[size - 1] / nsub >= 1.0 - 1e-8:
            n_full += 1
            left -= nsub
            if rem[size - 1] < 1.0:
                rem /= (1 - rem[size - 1])
            w = weight_vector[size - 1] / math.comb(M, size)
            if size <= n_paired:
                w /= 2.0
            for inds in combinations(range(M), size):
                z = np.zeros(M, np.int8)
                z[list(inds)] = 1
                Z.append(z); W.append(w)
                if size <= n_paired:
                    Z.append((1 - z).astype(np.int8)); W.append(w)
        else:
            break
    n_fixed = len(Z)
    samples_left = nsamples - n_fixed
    if n_full != n_sizes and samples_left > 0:
        rem = weight_vector.copy()
        rem[:n_paired] /= 2          # each draw is paired with its complement
        rem = rem[n_full:]
        rem /= rem.sum()
        ind_set = rng.choice(len(rem), 4 * samples_left, p=rem)
        used, k = {}, 0
        while samples_left > 0 and k < len(ind_set):
            size = int(ind_set[k]) + n_full + 1
            k += 1
            z = np.zeros(M, np.int8)
            z[rng.permutation(M)[:size]] = 1
            t = z.tobytes()
            new = t not in used
            if new:
                used[t] = len(Z)
                samples_left -= 1
                Z.append(z); W.append(1.0)
            else:
                W[used[t]] += 1.0
            if samples_left > 0 and size <= n_paired:
                if new:
                    samples_left -= 1
                    Z.append((1 - z).astype(np.int8)); W.append(1.0)
                else:
                    W[used[t] + 1] += 1.0
        W = np.asarray(W, np.float64)
        W[n_fixed:] *= weight_vector[n_full:].sum() / W[n_fixed:].sum()
    W = np.asarray(W, np.float64)
    return np.asarray(Z, np.int8).reshape(-1, M), W, n_full


def shap_solve(Z, w, y, fx, fnull):
    """KernelSHAP's weighted least squares with the efficiency constraint ``sum(phi) = fx - fnull``.

    As ``shap.KernelExplainer.solve`` with ``l1_reg`` off (the paper's
    Omega = 0): the last feature's value is eliminated through the
    constraint, the other M - 1 are fitted by weighted least squares, and the
    last is what the constraint leaves.
    """
    Z = np.asarray(Z, np.float64)
    w = np.asarray(w, np.float64)
    y = np.asarray(y, np.float64) - fnull
    total = float(fx - fnull)
    y2 = y - Z[:, -1] * total
    X = Z[:, :-1] - Z[:, -1:]
    sw = np.sqrt(w)[:, None]
    sol = np.linalg.lstsq(sw * X, sw[:, 0] * y2, rcond=None)[0]
    return np.concatenate([sol, [total - sol.sum()]])


def kernel_shap_groups(m, b, fn, layer, pos, h, base, groups, nsamples=None, rng=None,
                       batch=32, last=None, log=None, cache=None):
    """KernelSHAP (Lundberg and Lee 2017) with patch groups as the players; ``(phi [M], info)``.

    A player is a group of image tokens (``groups``, a list of index arrays);
    a coalition keeps its groups' states and replaces every other token's
    state at ``layer`` by ``base``. The coalitions and weights are shap's
    (``shap_coalitions``), ``f(x)`` and ``f(nothing)`` are scored with the
    same intervention, and ``phi`` solves the constrained least squares
    (``shap_solve``), so it sums to ``f(x) - f(base)``. ``nsamples`` left out
    is shap's default, ``2 M + 2048``. ``cache`` (a dict) keeps every
    coalition's score, so several budgets on one item pay once for the
    coalitions they share.
    """
    M = len(groups)
    nsamples = shap_default_samples(M) if nsamples is None else int(nsamples)
    Z, W, n_full = shap_coalitions(M, nsamples, rng)
    cache = {} if cache is None else cache
    P = h.shape[0]
    G = np.zeros((M, P), bool)
    for g, sel in enumerate(groups):
        G[g, np.asarray(sel)] = True
    ones, zeros = np.ones(M, np.int8), np.zeros(M, np.int8)
    todo, keys = [], set()
    for z in (ones, zeros, *Z):
        t = z.tobytes()
        if t not in cache and t not in keys:
            keys.add(t); todo.append(z)
    if todo:
        drop = (1 - np.stack(todo)).astype(bool) @ G          # [n, P], true where replaced
        with torch.no_grad():
            f = hooked_axis_many(m, b, fn, layer, pos, _Replaced(h, base, drop), batch=batch,
                                 last=last, log=log)
        for z, v in zip(todo, f):
            cache[z.tobytes()] = float(v)
    y = np.array([cache[z.tobytes()] for z in Z])
    fx, fnull = cache[ones.tobytes()], cache[zeros.tobytes()]
    phi = shap_solve(Z, W, y, fx, fnull)
    return phi, dict(nsamples=int(len(Z)), forwards=len(todo), sizes_enumerated=int(n_full),
                     fx=fx, fnull=fnull)


def spread_groups(phi, groups, P):
    """A per-group value spread evenly over its tokens, so each group's tokens sum to it; float32 [P]."""
    out = np.zeros(P, np.float32)
    for v, sel in zip(phi, groups):
        sel = np.asarray(sel)
        out[sel] = float(v) / max(len(sel), 1)
    return out


# ------------------------------------------------- the vision tower's layout

def _vision(m):
    """``(family, vision tower, block whose output is the image feature)`` for the families here.

    Qwen3-VL and Qwen3.5 keep the tower at ``model.visual`` (``blocks``, then a
    ``merger`` that joins each run of ``spatial_merge_size^2`` consecutive
    patch rows into one token); InternVL at ``model.vision_tower``
    (``encoder.layer``, the feature taken at ``vision_feature_layer``, then a
    2 x 2 pixel shuffle); Gemma 4 at ``model.vision_tower`` (``encoder.layers``,
    then a pooler that averages k x k patches by position). Any other family
    is refused, since a wrong patch-to-token map would give a map that looks
    fine and is not.
    """
    inner = getattr(m.hf, "model", m.hf)
    vis = getattr(inner, "visual", None)
    if vis is not None and hasattr(vis, "blocks") and hasattr(vis, "merger"):
        return "qwen", vis, vis.blocks[-1]
    vt = getattr(inner, "vision_tower", None)
    enc = getattr(vt, "encoder", None) if vt is not None else None
    name = type(vt).__name__ if vt is not None else ""
    if enc is not None and hasattr(enc, "layer") and name.startswith("InternVL"):
        k = getattr(m.hf.config, "vision_feature_layer", -1)
        k = k[-1] if isinstance(k, (list, tuple)) else int(k)
        # hidden_states[k] is the output of layer k - 1 (entry 0 is the embeddings)
        return "internvl", vt, enc.layer[k if k < 0 else k - 1]
    if enc is not None and hasattr(enc, "layers") and hasattr(vt, "pooler") and name.startswith("Gemma4"):
        return "gemma4", vt, enc.layers[int(vt.config.num_hidden_layers) - 1]
    raise NotImplementedError(
        f"no vision-tower layout for {type(m.hf).__name__} (tower {name or 'not found'}): "
        "Grad-CAM and pixel-space integrated gradients need to know which patches make which "
        "image token; add the family to attribution._vision and _to_tokens")


def _positions(b):
    for k in ("image_position_ids", "pixel_position_ids"):
        if torch.is_tensor(b.get(k)):
            return b[k]
    raise KeyError("the batch carries no image_position_ids for Gemma 4's pooler")


def _to_tokens(m, fam, vt, b, v, P):
    """Per-patch values ``v`` in the tower's own layout, summed onto the ``P`` image tokens; float64 [P].

    ``v`` is [rows] for Qwen (patch rows in the merge-window order the
    processor emits), [tiles, patches] for InternVL (class token dropped) and
    [images, patches] for Gemma 4 (padding patches zero).
    """
    v = v.detach().double().cpu()
    if fam == "qwen":
        k2 = int(getattr(vt, "spatial_merge_size", getattr(vt.config, "spatial_merge_size", 2))) ** 2
        if v.numel() != P * k2:
            raise ValueError(f"{v.numel()} patch rows do not make {P} tokens of {k2}")
        return v.reshape(P, k2).sum(1).numpy()
    if fam == "internvl":
        tiles, N = v.shape
        per = P // tiles
        g, gt = int(round(N ** 0.5)), int(round(per ** 0.5))
        if tiles * per != P or g * g != N or gt * gt != per or g % gt:
            raise ValueError(f"{tiles} tiles of {N} patches do not make {P} tokens")
        s = g // gt
        return v.reshape(tiles, gt, s, gt, s).sum((2, 4)).reshape(-1).numpy()
    if fam == "gemma4":
        pp = _positions(b).to(v.device)
        pv = b["pixel_values"]
        length = pv.shape[-2] // (int(vt.config.pooling_kernel_size) ** 2)
        k2 = (v.shape[1] // length)
        pooled, mask = vt.pooler._avg_pool_by_positions(v[..., None].float(), pp, length)
        out = (pooled[..., 0] * k2)[mask].double().reshape(-1)
        if out.numel() != P:
            raise ValueError(f"{out.numel()} pooled tokens, {P} image tokens")
        return out.numpy()
    raise NotImplementedError(fam)


# ----------------------------------------------------------------- Grad-CAM

def grad_cam_vision(m, b, fn, pos, last=None, log=None):
    """Grad-CAM (Selvaraju et al. 2017) at the vision tower's last block; numpy float32 [len(pos)].

    Grad-CAM takes the last convolutional layer's feature maps; for a ViT the
    standard adaptation (pytorch-grad-cam's recipe) takes the last block's
    token features as the maps, one channel per hidden unit. Each channel's
    weight is the mean over the picture's patches of the answer score's
    gradient with respect to it, a patch's map is the ReLU of its features
    dotted into the weights, and each language-model image token gets the
    sum of its patches' maps (the patches the merger, pixel shuffle or pooler
    join into it). The block's output is made a fresh leaf, so the gradient is
    the one that flows through the image feature the language model is given;
    Qwen3-VL's DeepStack features leave the tower from earlier blocks and are
    not on this path, which is what Grad-CAM at the last layer means.
    """
    fam, vt, blk = _vision(m)
    last = m.n_layers - 1 if last is None else last
    S = b["input_ids"].shape[1]
    got = {}

    def hook(mod, inp, out):
        tup = not torch.is_tensor(out)
        t = out[0] if tup else out
        leaf = t.detach().requires_grad_(True)
        got["a"] = leaf
        return (leaf, *out[1:]) if tup else leaf

    def run():
        got.clear()
        hd = blk.register_forward_hook(hook)
        try:
            with torch.enable_grad(), ActivationRecorder(m.layers, at=[last]) as rec:
                m.forward(b)
                hL = rec.activations[last][:, S - 1]
                v = fn(hL, m.unembed(hL)).sum()
                g, = torch.autograd.grad(v, got["a"])
        finally:
            hd.remove()
        return got["a"].detach(), g

    A, G = with_room(run, log=log)
    A, G = A.float(), G.float()
    P = len(pos)
    if fam == "qwen":                                   # [rows, C]
        cam = torch.relu(A @ G.mean(0))
    elif fam == "internvl":                             # [tiles, 1 + N, C]
        if getattr(m.hf.config, "vision_feature_select_strategy", "default") == "default":
            A, G = A[:, 1:], G[:, 1:]
        cam = torch.relu(A @ G.reshape(-1, G.shape[-1]).mean(0))
    else:                                               # gemma4, [images, patches, C]
        valid = ~(_positions(b) == -1).all(-1).to(A.device)
        w = (G * valid[..., None]).sum((0, 1)) / valid.sum().clamp(min=1)
        cam = torch.relu(A @ w) * valid
    return _to_tokens(m, fam, vt, b, cam, P).astype(np.float32)


# ------------------------------------------- integrated gradients on pixels

def ig_pixels(m, b, b0, fn, pos, steps=64, chunk=4, last=None, log=None):
    """Integrated gradients (Sundararajan et al. 2017) from the black image, through the whole model.

    ``b0`` is the same prompt built around an all-black picture of the same
    size, so its pixel values are the processor's normalisation of zero,
    the paper's baseline for images. The path is the straight line between
    the two pixel tensors (a straight line in raw pixels too, since the
    normalisation is affine), taken at ``steps`` midpoints; each pixel's
    attribution is its mean gradient times its change, and each image token
    gets the sum over the pixels of the patches it is made from. Returns
    ``(attr float32 [len(pos)], f(b0))``, so completeness can be checked
    against ``f(b) - f(b0)``.
    """
    fam, vt, _ = _vision(m)
    last = m.n_layers - 1 if last is None else last
    if not torch.equal(b["input_ids"], b0["input_ids"]):
        raise ValueError("the black-image prompt does not have the same tokens")
    S = b["input_ids"].shape[1]
    x1 = b["pixel_values"].float().cpu()
    x0 = b0["pixel_values"].float().cpu()
    if x1.shape != x0.shape:
        raise ValueError("the black image has a different patch layout")
    dty = b["pixel_values"].dtype
    dev = x1 - x0
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    acc = torch.zeros_like(x1, dtype=torch.float64)

    def one(al):
        n = len(al)
        shape = (n,) + (1,) * x1.ndim
        leaf = (x0[None] + al.view(shape) * dev[None]).to(m.device).requires_grad_(True)
        big = dict(m.replicate(b, n))
        big["pixel_values"] = leaf.reshape(n * x1.shape[0], *x1.shape[1:]).to(dty)
        with torch.enable_grad(), ActivationRecorder(m.layers, at=[last]) as rec:
            m.forward(big)
            hL = rec.activations[last][:, S - 1]
            v = fn(hL, m.unembed(hL)).sum()
            g, = torch.autograd.grad(v, leaf)
        return g.double().sum(0).cpu()

    s0, c = 0, max(min(int(chunk), _CHUNK_HINT.get("pixel", int(chunk))), 1)
    while s0 < steps:
        al = alphas[s0: s0 + c]
        try:
            acc += one(al) if len(al) > 1 else with_room(lambda: one(al), log=log)
        except Exception as e:  # noqa: BLE001
            if not is_oom(e) or len(al) <= 1:
                raise
            c = max(c // 2, 1)
            _CHUNK_HINT["pixel"] = c
            torch.cuda.empty_cache()
            if log:
                log(f"    not enough room; {c} pixel-path points at a time")
            continue
        s0 += len(al)
    el = acc / steps * dev.double()                     # per pixel value
    if fam == "qwen":                                   # [rows, C*T*p*p]
        per = el.sum(-1)
    elif fam == "internvl":                             # [tiles, 3, H, W]
        tiles, _, Hh, Ww = el.shape
        per_tile = len(pos) // tiles
        gt = int(round(per_tile ** 0.5))
        blk = Hh // gt
        # already at token resolution: a 2 x 2 shuffle of 14-pixel patches
        tok = el.sum(1).reshape(tiles, gt, blk, gt, blk).sum((2, 4)).reshape(-1)
        attr = tok.numpy().astype(np.float32)
        per = None
    else:                                               # gemma4, [images, patches, D]
        per = el.sum(-1)
    if per is not None:
        attr = _to_tokens(m, fam, vt, b, per, len(pos)).astype(np.float32)
    with torch.no_grad(), ActivationRecorder(m.layers, at=[last]) as rec:
        m.forward(b0)
        hL = rec.activations[last][:, S - 1]
        f0 = float(fn(hL, m.unembed(hL)).float().cpu()[0])
    return attr, f0


# ---------------------------------------------------------------- fast path

#: per (model, block): how many items the fast path has been checked on, or
#: False once it has failed a check
_SKIP_OK: dict = {}


class _Stand(torch.nn.Module):
    """Stands in for a block ``SkipTo`` does not run: hands its input on, or,
    for the block the states are replaced at, hands on the item's own output."""

    def __init__(self, state=None):
        super().__init__()
        self.state = state

    def forward(self, hidden_states, *args, **kwargs):
        if self.state is None:
            return hidden_states
        n = hidden_states.shape[0]
        return self.state.to(hidden_states.device, hidden_states.dtype).expand(n, -1, -1)


class SkipTo:
    """Run only the blocks after ``layer`` for one item: ``with SkipTo(m, b, L, ...) as bf``.

    Every score ``hooked_axis`` reads with the image states at ``layer``
    replaced depends on everything before ``layer`` only through that block's
    output, which is the same for every replacement. So the item's own output
    there is recorded once, the blocks up to it are stood in for by modules
    that hand it on, and the picture is left out of ``bf`` so the vision
    tower is not run either; the replacement hook on block ``layer`` then
    edits the recorded state as it edited the computed one.

    Whether a family's decoder allows this (one whose blocks share keys and
    values across depth does not) is checked, not assumed: on the first
    ``checks`` items of a (model, block) three replacements are scored both
    ways, and a difference beyond bf16 noise, or any error, puts the full
    model back for the rest of the run. ``bf`` is ``b`` itself whenever the
    fast path is off. Nothing that needs the vision tower or the early blocks
    (Grad-CAM, IG on pixels, reading states) may run inside the block.
    """

    def __init__(self, m, b, layer, fn=None, pos=None, h=None, base=None, checks=2,
                 enabled=True, log=None):
        self.m, self.b, self.layer, self.fn, self.pos = m, b, int(layer), fn, pos
        self.h, self.base, self.checks, self.enabled, self.log = h, base, checks, enabled, log
        self._saved = None
        self.active = False
        self.max_diff = None

    def _install(self, state):
        m = self.m
        self._saved = [m.layers[i] for i in range(self.layer + 1)]
        for i in range(self.layer):
            m.layers[i] = _Stand()
        m.layers[self.layer] = _Stand(state)

    def _remove(self):
        if self._saved is not None:
            for i, mod in enumerate(self._saved):
                self.m.layers[i] = mod
            self._saved = None

    def __enter__(self):
        m, b = self.m, self.b
        key = (m.name, self.layer)
        seen = _SKIP_OK.get(key, 0)
        if not self.enabled or seen is False:
            return b
        bf = {k: v for k, v in b.items() if k not in ("pixel_values", "pixel_values_videos")}
        try:
            with torch.no_grad(), ActivationRecorder(m.layers, at=[self.layer]) as rec:
                m.forward(b)
                state = rec.activations[self.layer][:1].detach().clone()
            need = seen < self.checks and self.fn is not None
            if need:
                X = torch.stack([self.h, self.base, 0.5 * (self.h + self.base)]).to(m.device)
                with torch.no_grad():
                    full = hooked_axis(m, b, self.fn, self.layer, self.pos, X).float().cpu()
            self._install(state)
            if need:
                with torch.no_grad():
                    fast = hooked_axis(m, bf, self.fn, self.layer, self.pos, X).float().cpu()
                d = float((fast - full).abs().max())
                # bf16 kernels differ a little with the batch size; a wrong
                # fast path (a block's input missed) differs by whole units
                tol = 0.1 + 0.05 * float(full.abs().max())
                self.max_diff = d
                if not d <= tol:                        # also catches nan
                    raise ValueError(f"the fast path differs from the full model by {d:.3g} "
                                     f"(tolerance {tol:.3g})")
                _SKIP_OK[key] = seen + 1
                if self.log:
                    self.log(f"    fast path checked against the full model: largest difference {d:.2g}")
        except Exception as e:  # noqa: BLE001
            self._remove()
            if is_oom(e):
                torch.cuda.empty_cache()
                return b                                # no room to check: full model this time
            _SKIP_OK[key] = False
            if self.log:
                self.log(f"    the fast path is off, every score runs the full model ({e})")
            return b
        self.active = True
        return bf

    def __exit__(self, *exc):
        self._remove()
        self.active = False


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
