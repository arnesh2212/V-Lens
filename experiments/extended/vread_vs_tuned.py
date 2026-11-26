from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/extended/a5")
    ap.add_argument("--tokens", type=int, default=4096, help="tokens per SGD step (250 steps)")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--variants", nargs="*", default=["v_read", "v_read_nogain", "v_read_fixlam", "ridge_r64",
                                                     "ridge_r256", "mse_sgd", "kl_sgd", "kl_from_vread", "kl_r64"])
    ap.add_argument("--corpus", type=int, default=900)
    ap.add_argument("--heldout", type=int, default=120)
    ap.add_argument("--fit-batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed of the fitting-pool draw (A7); 0 is fit_all's own corpus")
    a = ap.parse_args()
    log = lambda *x: print(*x, flush=True)

    import torch
    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays as SS
    from vlens.lens import baselines as B, fitall as FA

    name = a.repo.split("/")[-1]
    m = vlens.load_vlm(a.repo)
    S = FA.SHARED
    ref = json.load(open(f"results/fits/{name}__fitmeta.json"))[0]["spec"]
    layers = list(ref[2])
    lam_grid = tuple(ref[13]) if len(ref) > 13 else (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    gain_grid = tuple(ref[14]) if len(ref) > 14 else (1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0)
    assert max(gain_grid) <= 5 and min(lam_grid) < 1e-3, (lam_grid, gain_grid)
    log(f"lambda grid {lam_grid}; gain grid {gain_grid}")
    spare = FA.fitting_pool(SS.load(require_image=True), S["cap"])
    items = B.corpus_items(spare, C.load_image, n=a.corpus + 2 * a.heldout, square=448, seed=a.seed)
    fit_items, dev_items = items[: a.corpus], items[a.corpus: a.corpus + a.heldout]
    ids = B.fit_ids(items)
    stream = B.CachedStates(B.StateStream(m, fit_items, batch=a.fit_batch, seed=1), layers, log=log)
    held = B.CachedStates(B.StateStream(m, dev_items, batch=a.fit_batch, seed=3), layers, log=log)
    LAST = m.n_layers - 1
    os.makedirs(a.out, exist_ok=True)
    out_json = os.path.join(a.out, "fits.json")
    rec = json.load(open(out_json)) if os.path.exists(out_json) else {}
    maps = {}

    def save(variant, M):
        d = os.path.join(a.out, "maps", variant)
        os.makedirs(d, exist_ok=True)
        mm = {l: M[l] for l in layers}
        spec = json.dumps([a.repo, f"a5:{variant}", layers])
        FA._save(os.path.join(d, f"a5_{variant}.pt"), {"v_read": mm}, None, ids, {"v_read": {"variant": variant}}, spec)

    def kl(M):
        try:
            res, _ = B.heldout_kl(m, held, layers, {"x": {l: M[l] for l in layers}})
            return {"lens": {str(k): float(v) for k, v in res["x"].items()},
                    "logit": {str(k): float(v) for k, v in res["logit"].items()}}
        except Exception as e:
            return f"n/a ({type(e).__name__}: {e})"

    def gain(variant):
        meta = {}
        FA._gain(m, maps, meta, variant, held, layers, gain_grid, log)
        return meta.get(variant, {})

    if {"v_read", "v_read_nogain", "v_read_fixlam", "ridge_r64", "ridge_r256", "kl_from_vread"} & set(a.variants):
        t = time.time()
        acc = B.RidgeAccumulator(layers, m.d_model, device=m.device)
        ntok = 0
        for H, n in stream:
            acc.add({l: H[l].float() for l in [*layers, LAST]}, LAST); ntok += n
        t_mom = time.time() - t
        log(f"ridge moments over {ntok} tokens in {t_mom:.0f} s")
        t = time.time()
        lam, maps["v_read"] = B.pick_lam(m, acc, held, layers, candidates=lam_grid, gain_grid=gain_grid, log=log)
        rec["v_read"] = dict(lam=lam, tokens=ntok, seconds=round(t_mom + time.time() - t), dev_kl=kl(maps["v_read"]))
        save("v_read", maps["v_read"])
        if "v_read_nogain" in a.variants:
            t = time.time()
            _, maps["v_read_nogain"] = B.pick_lam(m, acc, held, layers, candidates=(lam,), gain_grid=(1.0,), log=log)
            rec["v_read_nogain"] = dict(lam=lam, seconds=round(time.time() - t), dev_kl=kl(maps["v_read_nogain"]))
            save("v_read_nogain", maps["v_read_nogain"])
        if "v_read_fixlam" in a.variants:
            t = time.time()
            _, maps["v_read_fixlam"] = B.pick_lam(m, acc, held, layers, candidates=(1e-3,), gain_grid=gain_grid, log=log)
            rec["v_read_fixlam"] = dict(lam=1e-3, seconds=round(time.time() - t), dev_kl=kl(maps["v_read_fixlam"]))
            save("v_read_fixlam", maps["v_read_fixlam"])
        for r in (64, 256):
            v = f"ridge_r{r}"
            if v not in a.variants:
                continue
            M = {}
            for l in layers:
                A, b = maps["v_read"][l]
                W = A.float().to(m.device) - torch.eye(A.shape[0], device=m.device)
                U, Sv, Vh = torch.linalg.svd(W, full_matrices=False)
                M[l] = ((torch.eye(A.shape[0], device=m.device) + (U[:, :r] * Sv[:r]) @ Vh[:r]).cpu(), b)
                del W, U, Sv, Vh
            maps[v] = M
            rec[v] = dict(rank=r, dev_kl=kl(M))
            save(v, M)
        del acc
        json.dump(rec, open(out_json, "w"), indent=1, default=str)

    def sgd(variant, loss_kind, rank=None, init=None):
        t = time.time()
        if loss_kind == "kl":
            M = B.fit_tuned_lens(m, stream, layers=layers, steps=a.steps, tokens_per_step=a.tokens,
                                 rank=rank, init=init, log_every=50, log=log)
        else:
            M = mse_tuned(m, stream, layers, steps=a.steps, tokens_per_step=a.tokens, log=log)
        maps[variant] = M
        g = gain(variant)
        rec[variant] = dict(loss=loss_kind, rank=rank, init=bool(init), steps=a.steps, tokens_per_step=a.tokens,
                            seconds=round(time.time() - t), gain=g, dev_kl=kl(maps[variant]),
                            meta={k: v for k, v in M.get("meta", {}).items() if k in ("tokens_spent", "kl_first", "kl_last")})
        save(variant, maps[variant])
        json.dump(rec, open(out_json, "w"), indent=1, default=str)

    def mse_tuned(m, stream, layers, steps, tokens_per_step, log):
        tr = {l: B.Translator(m.d_model, None).to(m.device).float() for l in layers}
        params = [p for t_ in tr.values() for p in t_.parameters()]
        opt = torch.optim.SGD(params, lr=B.sgd_step(B.PAPER["lr"]), momentum=B.PAPER["momentum"],
                              nesterov=B.PAPER["nesterov"], weight_decay=B.PAPER["weight_decay"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / steps))
        it = iter(stream); spent = 0; hist = []
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            got, tot = 0, 0.0
            while got < tokens_per_step:
                try:
                    H, n = next(it)
                except StopIteration:
                    it = iter(stream); H, n = next(it)
                y = H[LAST].float()
                scale = (y.pow(2).sum(-1).mean()).clamp_min(1e-6)
                w = n / tokens_per_step
                for l in layers:
                    loss = ((tr[l](H[l].float()) - y).pow(2).sum(-1).mean() / scale) * w
                    loss.backward(); tot += float(loss.detach())
                got += n; spent += n
            torch.nn.utils.clip_grad_norm_(params, B.PAPER["clip"])
            opt.step(); sched.step(); hist.append(tot)
            if step % 50 == 0 or step == steps - 1:
                log(f"    mse lens step {step + 1}/{steps}  rel. MSE {tot / len(layers):.4f}  tokens {spent / 1e6:.2f}M")
        M = {l: tuple(x.float().cpu() for x in t_.affine()) for l, t_ in tr.items()}
        M["meta"] = dict(tokens_spent=spent, kl_first=hist[0], kl_last=hist[-1])
        return M

    if "mse_sgd" in a.variants:
        sgd("mse_sgd", "mse")
    if "kl_sgd" in a.variants:
        sgd("kl_sgd", "kl")
    if "kl_from_vread" in a.variants and "v_read" in maps:
        init = {l: maps["v_read_nogain"][l] if "v_read_nogain" in maps else maps["v_read"][l] for l in layers}
        sgd("kl_from_vread", "kl", init=init)
    if "kl_r64" in a.variants:
        sgd("kl_r64", "kl", rank=64)
    log(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("seconds", "lam", "rank")} for k, v in rec.items()}))
    log("DONE vread_vs_tuned")


if __name__ == "__main__":
    main()
