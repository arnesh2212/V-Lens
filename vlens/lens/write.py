from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch

from vlens.lens import attribution, readout
from vlens.lens.hooks import ActivationRecorder
from vlens.lens.room import is_oom, with_room


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-8)


class AddAtPositions:

    def __init__(self, module, u: torch.Tensor, alpha: float, positions: torch.Tensor):
        self.module, self.u, self.alpha, self.pos, self._h = module, u, alpha, positions, None

    def __enter__(self) -> "AddAtPositions":
        def hook(mod, inp, out, _u=self.u, _a=self.alpha, _p=self.pos):
            tup = not torch.is_tensor(out)
            hh = out[0] if tup else out
            if hh.shape[1] < 2:
                return out
            hh = hh.clone()
            p = _p.to(hh.device)
            h = hh[:, p]
            hh[:, p] = h + _a * h.norm(dim=-1, keepdim=True) * _u.to(hh.device, hh.dtype)
            return (hh, *out[1:]) if tup else hh
        self._h = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc) -> None:
        self._h.remove()
        self._h = None


def write_direction(m, items: Sequence[dict], layer: int, score: Callable,
                    batch: int = 4, log: Callable = print) -> torch.Tensor:
    acc, n = None, 0
    for i, it in enumerate(items):
        def one(it=it):
            b = m.chat(it["question"], [it["image"]], system=it.get("system"), cue=it.get("cue"))
            pos = readout.image_positions(m, b)
            if len(pos) == 0:
                return None
            with torch.no_grad(), ActivationRecorder(m.layers, at=[layer]) as rec:
                m.forward(b)
                h = rec.activations[layer][0, pos.to(rec.activations[layer].device)]
            x = h.float().to(m.device).detach().clone().requires_grad_(True)
            with torch.enable_grad():
                v = attribution.hooked_axis(m, b, score, layer, pos, x[None]).sum()
                g, = torch.autograd.grad(v, x)
            return g.float().mean(0).cpu()
        try:
            got = with_room(one, log=log)
        except Exception as e:
            if not is_oom(e):
                raise
            continue
        if got is None:
            continue
        acc = got if acc is None else acc + got
        n += 1
        if log and (i + 1) % 25 == 0:
            log(f"    write direction {i + 1}/{len(items)} prompts")
    if n == 0:
        raise RuntimeError("no prompt yielded a direction")
    return unit(acc / n)


def difference_of_means(m, pos_items: Sequence[dict], neg_items: Sequence[dict], layer: int,
                        log: Callable = print) -> torch.Tensor:
    def mean_of(items):
        acc, n = None, 0
        for it in items:
            b = m.chat(it["question"], [it["image"]], system=it.get("system"), cue=it.get("cue"))
            pos = readout.image_positions(m, b)
            if len(pos) == 0:
                continue
            with torch.no_grad(), ActivationRecorder(m.layers, at=[layer]) as rec:
                m.forward(b)
                h = rec.activations[layer][0, pos.to(rec.activations[layer].device)]
            acc = h.float().mean(0).cpu() if acc is None else acc + h.float().mean(0).cpu()
            n += 1
        return None if n == 0 else acc / n
    a, b_ = mean_of(pos_items), mean_of(neg_items)
    if a is None or b_ is None:
        raise RuntimeError("not enough labelled items for a difference of means")
    return unit(a - b_)


def unembed_direction(m, token_id: int) -> torch.Tensor:
    w = m._lm_head.weight
    return unit(w[int(token_id)].detach().float().cpu())


def random_direction(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed) % (2 ** 31))
    return unit(torch.randn(d, generator=g))


def span_random_direction(H: torch.Tensor, seed: int) -> torch.Tensor:
    H = H.float()
    C = H - H.mean(0, keepdim=True)
    g = torch.Generator().manual_seed(int(seed) % (2 ** 31))
    w = torch.randn(C.shape[0], generator=g)
    return unit(w @ C)


def scaled_random_direction(H: torch.Tensor, seed: int) -> torch.Tensor:
    H = H.float()
    g = torch.Generator().manual_seed(int(seed) % (2 ** 31))
    return unit(torch.randn(H.shape[1], generator=g) * H.std(0).clamp_min(1e-8))


def shuffled_direction(u: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed) % (2 ** 31))
    return unit(u[torch.randperm(u.shape[0], generator=g)])


def image_token_states(m, items, layer: int, per_item: int = 64, seed: int = 0) -> torch.Tensor:
    from vlens.lens.hooks import ActivationRecorder
    from vlens.lens import readout
    rng = torch.Generator().manual_seed(int(seed))
    out = []
    for it in items:
        b = m.chat(it["question"], [it["image"]], system=it.get("system"), cue=it.get("cue"))
        pos = readout.image_positions(m, b)
        if len(pos) == 0:
            continue
        with torch.no_grad(), ActivationRecorder(m.layers, at=[layer]) as rec:
            m.forward(b)
            h = rec.activations[layer][0, pos.to(rec.activations[layer].device)].float().cpu()
        k = torch.randperm(h.shape[0], generator=rng)[:per_item]
        out.append(h[k])
    return torch.cat(out) if out else torch.zeros(0)
