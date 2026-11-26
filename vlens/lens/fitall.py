from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from typing import Callable, Sequence

import numpy as np
import torch

from vlens.lens import baselines as B

CACHE = os.environ.get("VLENS_MAP_CACHE", "cache/vlens_maps")
PAPER_WD = 1e-3

def _budget(v: str | None) -> int:
    if not v:
        return 6 * 2 ** 30
    x = float(v)
    return int(x * 2 ** 30) if x < 2 ** 20 else int(x)


CACHE_BUDGET = _budget(os.environ.get("VLENS_MAP_BUDGET"))


def _free_bytes(path: str) -> int:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    except OSError:
        return 0


def _make_room(path: str, need: int, log: Callable = print) -> bool:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    for stale in glob.glob(os.path.join(d, "*.tmp")):
        try:
            if time.time() - os.path.getmtime(stale) > 1800:
                os.remove(stale)
        except OSError:
            pass
    for orphan in glob.glob(os.path.join(d, "*.pt.spec")):
        if not os.path.exists(orphan[: -len(".spec")]):
            try:
                os.remove(orphan)
            except OSError:
                pass
    def entries():
        got = []
        for f in glob.glob(os.path.join(d, "*.pt")):
            try:
                got.append((os.path.getmtime(f), os.path.getsize(f), f))
            except OSError:
                pass
        return sorted(got)
    while True:
        got = entries()
        used = sum(sz for _, sz, _ in got)
        if (used + need <= CACHE_BUDGET) and (_free_bytes(d) > need * 1.2):
            return True
        if not got:
            return _free_bytes(d) > need * 1.2
        _, sz, oldest = got[0]
        try:
            os.remove(oldest)
            if os.path.exists(oldest + ".spec"):
                os.remove(oldest + ".spec")
            log(f"  cache full; dropped the oldest fit ({sz / 2 ** 30:.1f} GiB)")
        except OSError:
            return False


SHARED_WANT = ("v_read", "tuned", "jacobian")
SHARED = dict(layers=16, corpus=900, heldout=120, square=448, steps=250,
              tokens_per_step=2 ** 18, rank=64, jac_prompts=100,
              latent_captions=6000, fit_batch=8, cap=120)


def fitting_pool(records: Sequence[dict], cap: int = 120) -> list[dict]:
    seen: dict = {}
    out = []
    for r in records:
        k = (r["family"], r["subcategory"], r["framing"])
        seen[k] = seen.get(k, 0) + 1
        if seen[k] > cap:
            out.append(r)
    return out


def balanced_sample(records: Sequence[dict], n: int, key: str = "family",
                    seed: int = 0) -> list[dict]:
    if not n or n >= len(records):
        return list(records)
    rng = np.random.default_rng(seed)
    groups: dict = {}
    for r in records:
        groups.setdefault(str(r.get(key)), []).append(r)
    for g in groups.values():
        rng.shuffle(g)
    out, names = [], sorted(groups)
    while len(out) < n and any(groups[g] for g in names):
        for g in names:
            if groups[g] and len(out) < n:
                out.append(groups[g].pop())
    rng.shuffle(out)
    return out


def layer_grid(m, k: int) -> list[int]:
    lo, hi = max(getattr(m, "n_deepstack", 0), 0), m.n_layers - 2
    if k >= hi - lo + 1:
        return list(range(lo, hi + 1))
    return sorted({int(round(x)) for x in np.linspace(lo, hi, k)})


def _spec(model_name, spare, layers, corpus, heldout, square, steps, tokens_per_step,
          rank, jac_prompts, latent_captions, fit_batch, lr_grid, lam_grid, gain_grid,
          lr_probe_layers, temp_grid):
    ids = ",".join(sorted(str(r.get("id")) for r in spare))
    return json.dumps([model_name, hashlib.sha1(ids.encode()).hexdigest(), list(layers),
                       corpus, heldout, square, steps, tokens_per_step, rank,
                       jac_prompts, latent_captions, fit_batch,
                       list(lr_grid), list(lam_grid), list(gain_grid),
                       lr_probe_layers, list(temp_grid)], sort_keys=True)


def _key(*args) -> str:
    return hashlib.sha1(_spec(*args).encode()).hexdigest()[:16]


def _adopt(spec: str, path: str, log: Callable = print) -> bool:
    d = os.path.dirname(path) or "."
    for f in sorted(os.listdir(d)):
        if not f.endswith(".pt.spec"):
            continue
        q = os.path.join(d, f)
        try:
            if open(q).read() != spec:
                continue
        except OSError:
            continue
        src = q[: -len(".spec")]
        if not os.path.exists(src) or os.path.abspath(src) == os.path.abspath(path):
            continue
        os.rename(src, path)
        os.replace(q, path + ".spec")
        log(f"  the fit cached at {src} describes this one; taken over as {path}")
        return True
    return False


def _subset(maps, latent, fit_ids, meta, want):
    return ({k: v for k, v in maps.items() if k in want},
            latent if "latent" in want else None, fit_ids,
            {k: v for k, v in meta.items() if k in want})


def _load(path):
    if not os.path.exists(path):
        return None
    d = torch.load(path, map_location="cpu")
    try:
        os.utime(path, None)
        if os.path.exists(path + ".spec"):
            os.utime(path + ".spec", None)
    except OSError:
        pass
    maps = {k: {int(l): (A.float(), b.float()) for l, (A, b) in v.items()}
            for k, v in d["maps"].items()}
    for k, mm in (d.get("meta_per_map") or {}).items():
        if mm is not None and k in maps:
            maps[k]["meta"] = mm
    latent = None
    if d.get("latent") is not None:
        from vlens.lens.baselines import LatentLens
        L = d["latent"]
        latent = LatentLens(L["keys"], L["toks"], L["vocab"], L.get("meta"))
    return maps, latent, set(d["fit_ids"]), d["meta"]


def _is_layer(x) -> bool:
    return isinstance(x, int) or (isinstance(x, str) and x.lstrip("-").isdigit())


def _save(path, maps, latent, fit_ids, meta, spec: str | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    half = lambda t: t.half() if t.dtype == torch.float32 else t
    out = dict(maps={k: {int(l): (half(v[l][0]), half(v[l][1])) for l in v if _is_layer(l)}
                     for k, v in maps.items()},
               meta_per_map={k: v.get("meta") for k, v in maps.items()},
               latent=(dict(keys=latent.keys, toks=latent.toks, vocab=latent.vocab,
                            meta=latent.meta) if latent is not None else None),
               fit_ids=sorted(fit_ids), meta=meta)
    need = sum(t.numel() * t.element_size()
               for v in out["maps"].values() for pair in v.values() for t in pair) + 2 ** 28
    if out["latent"] is not None:
        for part in ("keys", "toks"):
            for t in (out["latent"][part] or {}).values():
                if torch.is_tensor(t):
                    need += t.numel() * t.element_size()
    if not _make_room(path, need):
        raise OSError("no room for a %.1f GiB fit in %s" % (need / 2 ** 30, os.path.dirname(path)))
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        torch.save(out, tmp)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, path)
    if spec is not None:
        with open(path + ".spec", "w") as fh:
            fh.write(spec)


def _gain(m, maps: dict, meta: dict, kind: str, held, layers, gain_grid, log: Callable) -> None:
    if not gain_grid:
        return
    M = {l: maps[kind][l] for l in layers}
    mu = B.mean_last_state(m, held).to(M[layers[0]][0].device)
    got, gmeta = B.pick_gain(m, M, held, layers, mu, candidates=tuple(gain_grid), log=log)
    keep = maps[kind].get("meta", {})
    maps[kind] = dict(got)
    maps[kind]["meta"] = dict(keep, **gmeta)
    meta[kind] = dict(meta.get(kind, {}), **gmeta)


def fit_all(m, spare: Sequence[dict], loader: Callable, *, layers: Sequence[int],
            corpus: int = 600, heldout: int = 80, square: int = 448,
            steps: int = 250, tokens_per_step: int = 2 ** 16,
            lr_grid=(1.0, 0.1, 0.01), lr_probe_steps: int = 30, lr_probe_layers: int = 4,
            lam_grid=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
            gain_grid=(1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0),
            temp_grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0), rank: int = 64,
            jac_prompts: int = 16, latent_captions: int = 6000,
            fit_batch: int = 8, want=("v_read", "tuned", "jacobian", "latent"),
            tuned_lr: float | None = None, tuned_wd: float | None = None,
            tuned_search: bool = False,
            cache: bool = True, results: str | None = None,
            log: Callable = print):
    spec = _spec(getattr(m, "name", str(m)), spare, layers, corpus, heldout, square, steps,
                 tokens_per_step, rank, jac_prompts, latent_captions, fit_batch,
                 lr_grid, lam_grid, gain_grid, lr_probe_layers, temp_grid)
    key = hashlib.sha1(spec.encode()).hexdigest()[:16]
    path = os.path.join(CACHE, f"{key}.pt")
    maps, meta, latent, fit_ids, need = {}, {}, None, set(), list(want)
    if cache:
        if not os.path.exists(path):
            try:
                _adopt(spec, path, log=log)
            except Exception as e:
                log(f"  could not take over an older entry ({type(e).__name__})")
        got = _load(path)
        if got is not None:
            maps, latent, fit_ids, meta = got
            have = set(maps) | ({"latent"} if latent is not None else set())
            need = [w for w in want if w not in have]
            log(f"  reusing the fit cached at {path}"
                + (f", and fitting {', '.join(need)} on top of it" if need else ""))
            if not need:
                return _subset(maps, latent, fit_ids, meta, want)
    if need and os.environ.get("VLENS_NO_FIT"):
        log(f"  VLENS_NO_FIT is set and {', '.join(need)} is not cached at {path}; stopping")
        raise SystemExit(3)
    items = B.corpus_items(spare, loader, n=corpus + 2 * heldout, square=square, seed=0)
    fit_items = items[: corpus]
    dev_items = items[corpus: corpus + heldout]
    held_items = items[corpus + heldout:]
    fit_ids |= B.fit_ids(fit_items, dev_items, held_items)
    stream = B.CachedStates(B.StateStream(m, fit_items, batch=fit_batch, seed=1),
                            layers, log=log)
    held = B.CachedStates(B.StateStream(m, dev_items, batch=fit_batch, seed=3),
                          layers, log=log)
    _ = held_items
    LAST = m.n_layers - 1

    if "v_read" in need:
        t = time.time()
        acc = B.RidgeAccumulator(layers, m.d_model, device=m.device)
        ntok = 0
        for H, n in stream:
            acc.add({l: H[l].float() for l in [*layers, LAST]}, LAST); ntok += n
        lam, maps["v_read"] = B.pick_lam(m, acc, held, layers, candidates=tuple(lam_grid),
                                         gain_grid=tuple(gain_grid), log=log)
        meta["v_read"] = dict(maps["v_read"].get("meta", {}), lam=lam, tokens=ntok,
                              seconds=round(time.time() - t))

    if "tuned" in need:
        if tuned_lr is not None:
            lr = float(tuned_lr)
            wd = float(tuned_wd) if tuned_wd is not None else PAPER_WD
            log(f"  tuned lens at lr_scale {lr}, weight decay {wd}")
        elif tuned_search:
            lr, wd = B.pick_lr(m, stream, layers, held, candidates=tuple(lr_grid),
                               probe_steps=lr_probe_steps, probe_layers=lr_probe_layers,
                               tokens_per_step=tokens_per_step, log=log)
        else:
            lr, wd = B.PAPER["lr"], B.PAPER["weight_decay"]
            log(f"  tuned lens at the published lr_scale {lr}, weight decay {wd}")
        t = time.time()
        maps["tuned"] = B.fit_tuned_lens(m, stream, layers=layers, steps=steps,
                                         tokens_per_step=tokens_per_step, lr=lr, weight_decay=wd,
                                         log_every=25, log=log)
        meta["tuned"] = dict(maps["tuned"]["meta"], lr_chosen=lr, wd_chosen=wd,
                             lr_searched=bool(tuned_search and tuned_lr is None),
                             seconds=round(time.time() - t))
        _gain(m, maps, meta, "tuned", held, layers, gain_grid, log)
    if "jacobian" in need:
        t = time.time()
        jit = B.corpus_items(spare, loader, n=jac_prompts, square=square, seed=0)
        fit_ids |= B.fit_ids(jit)
        maps["jacobian"] = B.fit_jacobian_lens(m, jit, source_layers=layers, dim_batch=8, log=log)
        meta["jacobian"] = dict(maps["jacobian"]["meta"], seconds=round(time.time() - t))
        _gain(m, maps, meta, "jacobian", held, layers, gain_grid, log)
    if "latent" in need:
        t = time.time()
        caps, cap_items = B.latent_corpus(m, spare, loader, results=results,
                                          model_name=getattr(m, "name", "").split("/")[-1],
                                          n=latent_captions, square=square, log=log)
        fit_ids |= B.fit_ids(cap_items)
        latent = B.build_latent_lens(m, caps, layers=layers, log=log)
        latent.meta["temp"], latent.meta["floor"] = B.pick_temp(
            m, latent, held, layers, candidates=tuple(temp_grid), log=log)
        latent.meta["coverage"] = {int(l): int(t.unique().numel())
                                   for l, t in latent.toks.items()}
        meta["latent"] = dict(latent.meta, seconds=round(time.time() - t))
    meta = dict(meta)
    if cache:
        try:
            _save(path, maps, latent, fit_ids, meta, spec)
            log(f"  fit cached at {path}")
        except Exception as e:
            log(f"  could not cache the fit ({type(e).__name__}: {e})")
    return _subset(maps, latent, fit_ids, meta, want)


def transport(maps: dict, latent, h, layer: int, temp_default: float = 0.05):
    out = {"logit": h}
    for k, M in maps.items():
        A, b = M[layer]
        out[k] = h @ A.T.to(h.device, h.dtype) + b.to(h.device, h.dtype)
    return out


def score_all(m, maps: dict, latent, h, layer: int, fn) -> dict:
    import torch
    out = {}
    with torch.no_grad():
        for k, v in transport(maps, latent, h, layer).items():
            out[k] = fn(None, m.unembed(v.to(torch.bfloat16))).float().cpu().numpy()
        if latent is not None:
            q = latent.similarity_logits(h.float(), layer) / max(latent.meta.get("temp", 0.05), 1e-6)
            out["latent"] = fn(None, q).float().cpu().numpy()
    return out
