# SPDX-License-Identifier: Apache-2.0
"""A transport whose target is the state that produced a caption word.

The model captions a picture in its own words; V-Write's gradient says which
patches carried each content word; those patches are paired with the state that
produced the word and the pair is accumulated into a ridge fit. No labels: the
caption, the alignment and the target are all the model's own.
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np
import torch

from vlens.lens.hooks import ActivationRecorder

STOP = set("""a an the this that these those is are was were be been being of in on at to for with
by from as it its their his her they he she we you i and or but not no nor so if then there here
what which who whom whose when where why how all any both each few more most other some such only
own same too very can will just should now image picture photo shows showing showed appears seems
looks like about into over under above below between during before after again further once there
""".split())


def content_words(text: str) -> list[str]:
    ws = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    return [w for w in ws if len(w) > 2 and w not in STOP]


def collect_naming_pairs(m, items, layers, **kw):
    layers = sorted(layers)
    """The (patch, name-state) pairs both naming fits are trained on."""
    store = {"X": {l: [] for l in layers}, "Y": []}
    fit_naming_map(m, items, layers, _collect=store, **kw)
    if not store["Y"]:
        return None
    X = {l: torch.cat(store["X"][l]) for l in layers}
    return X, torch.cat(store["Y"])


def fit_naming_map(m, items: Sequence[dict], layers: Sequence[int], *,
                   prompt: str = "Describe this image in one sentence.",
                   gen: int = 40, top_frac: float = 0.15, max_words: int = 8,
                   lam_grid=(1e-2, 1e-1, 1.0, 10.0), _collect=None,
                   log: Callable = print):
    """``{layer: (A, b)}`` carrying an image token to where its own name is decided.

    For each picture: the model captions it, the content words of that caption
    are found in the generated ids, and for each one the gradient of its log
    probability with respect to the image states says which patches carried it.
    Those patches are paired with the state that produced the word, and the
    pairs are accumulated into a ridge fit.
    """
    from vlens.lens.baselines import RidgeAccumulator
    tok = m.tokenizer
    LAST = m.n_layers - 1
    acc = RidgeAccumulator(list(layers), m.d_model, device=m.device)
    held = {l: [] for l in layers}
    held_y = []
    n_pairs, n_img = 0, 0
    for i, it in enumerate(items):
        img = it.get("image")
        if img is None:
            continue
        ptxt = m.chat_prompt(prompt, [img])
        b = m.build(ptxt, [img])
        n_in = b["input_ids"].shape[1]
        with torch.no_grad():
            g = m.hf.generate(**{k: v for k, v in b.items() if not k.startswith("_")},
                              max_new_tokens=gen, do_sample=False,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
        cap = tok.decode(g[0, n_in:], skip_special_tokens=True).strip()
        want = set(content_words(cap))
        if len(want) < 2:
            continue
        # rebuilt from text so every tensor the processor returns agrees
        full = m.build(ptxt + cap, [img])
        ids = full["input_ids"][0]
        if ids.numel() <= n_in:
            continue
        # Appending the caption can merge with the prompt's last token, so the
        # prefix is not required to be identical; everything below indexes the
        # rebuilt sequence, which is self-consistent either way. The search just
        # starts a little before where the caption is expected.
        picks = []
        for j in range(max(n_in - 4, 1), ids.numel()):
            w = re.sub(r"[^a-z\-']", "", tok.decode(ids[j: j + 1]).strip().lower())
            if w in want and len(w) > 2:
                picks.append(j)
            if len(picks) >= max_words:
                break
        if not picks:
            continue
        pos_img = m.image_mask(full["input_ids"])[0].nonzero(as_tuple=True)[0]
        if pos_img.numel() < 8:
            continue
        # every seventh picture is held back whole: its pairs are collected for
        # choosing the penalty and never accumulated into the fit
        is_held = (i % 7 == 0)
        with ActivationRecorder(m.layers, at=[*layers, LAST],
                                start_graph_at=min(layers)) as rec, torch.enable_grad():
            m.forward(full)
            # the recorded tensors themselves; gradients are taken with
            # respect to these, not to a view made afterwards
            H = {l: rec.activations[l] for l in [*layers, LAST]}
            for j in picks:
                q = j                              # position of the word
                wid = int(ids[j])
                lg = m.unembed(H[LAST][0, q - 1][None])
                score = torch.log_softmax(lg.float(), -1)[0, wid]
                grads = torch.autograd.grad(score, [H[l] for l in layers],
                                            retain_graph=True, allow_unused=True)
                y = H[LAST][0, q - 1].detach().float()
                for l, gr in zip(layers, grads):
                    if gr is None:
                        continue
                    hl = H[l][0, pos_img].detach().float()
                    a = (gr[0, pos_img].detach().float()
                         * (hl - hl.mean(0, keepdim=True))).sum(-1)
                    k = max(int(round(top_frac * a.numel())), 1)
                    sel = torch.topk(a, k).indices
                    Xl = hl[sel]
                    if is_held:
                        if len(held[l]) < 400:
                            held[l].append(Xl[0].detach().cpu())
                        if l == layers[0] and len(held_y) < 400:
                            held_y.append(y.cpu())
                        continue
                    Y = y[None].expand(sel.numel(), -1)
                    if _collect is not None:
                        _collect["X"][l].append(Xl.detach().cpu())
                        if l == min(layers):
                            _collect["Y"].append(Y.detach().cpu())
                        continue
                    acc.G[l] += (Xl.double().T @ Xl.double())
                    acc.C[l] += (Xl.double().T @ Y.double())
                    acc.sx[l] += Xl.double().sum(0)
                    if l == layers[0]:
                        acc.sy += Y.double().sum(0)
                        acc.n += sel.numel()
                        n_pairs += sel.numel()
        n_img += 1
        if n_img % 25 == 0:
            log(f"    naming map: {n_img} pictures, {n_pairs} patch-word pairs")
        del H
    if acc.n < 200:
        log("    naming map: too few pairs, skipped")
        return None, {}
    # the penalty is chosen on pairs the fit did not see
    best, best_err, out = None, None, None
    HY = torch.stack(held_y) if held_y else None
    for lam in lam_grid:
        M = acc.solve(lam=lam)
        if HY is None or not held[layers[0]]:
            out, best = M, lam
            break
        err = 0.0
        for l in layers:
            if not held[l]:
                continue
            HX = torch.stack(held[l][: HY.shape[0]])
            A, bb = M[l]
            err += float(((HX @ A.T + bb) - HY[: HX.shape[0]]).pow(2).mean())
        err /= max(len(layers), 1)
        log(f"    naming map lambda {lam:<7} -> held-out squared error {err:.4f}")
        if best_err is None or err < best_err:
            best, best_err, out = lam, err, M
    meta = dict(kind="image token -> the state that produced its own name",
                lam=best, pairs=int(n_pairs), pictures=int(n_img),
                top_frac=top_frac, max_words=max_words,
                note="alignment from V-Write, target from the model's own caption")
    log(f"    naming map fitted on {n_pairs} pairs from {n_img} pictures, lambda {best}")
    return out, meta


def fit_naming_tuned(m, items, layers, *, lr: float = 0.1, steps: int = 120,
                     batch: int = 128, log=print):
    """The tuned lens's objective on the naming pairs.

    Same translator and optimiser as the published lens, with the target
    distribution taken at the state that produced the word, so the naming
    target is available to the baseline as well.
    """
    from vlens.lens.baselines import PAPER, Translator
    pairs = collect_naming_pairs(m, items, layers, log=log)
    if pairs is None:
        return None
    X, Y = pairs
    dev = m.device
    tr = {l: Translator(m.d_model, None).to(dev).float() for l in layers}
    params = [p for t in tr.values() for p in t.parameters()]
    opt = torch.optim.SGD(params, lr=lr, momentum=PAPER["momentum"], nesterov=True,
                          weight_decay=PAPER["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / steps))
    g = torch.Generator().manual_seed(0)
    n = Y.shape[0]
    for step in range(steps):
        j = torch.randperm(n, generator=g)[:batch].to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            ref = torch.log_softmax(m.unembed(Y[j].to(dev)).float(), -1)
            p_ = ref.exp()
        for l in layers:
            lg = torch.log_softmax(m.unembed(tr[l](X[l][j].to(dev).float())).float(), -1)
            ((p_ * (ref - lg)).sum(-1).mean() / len(layers)).backward()
        torch.nn.utils.clip_grad_norm_(params, PAPER["clip"])
        opt.step(); sched.step()
        if step % 40 == 0:
            log(f"    naming tuned lens step {step + 1}/{steps}")
    out = {}
    for l, t in tr.items():
        A, b = t.affine()
        out[l] = (A.float().cpu(), b.float().cpu())
    return out
