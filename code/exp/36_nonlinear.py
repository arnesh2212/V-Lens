"""Is the ceiling linearity's fault, or the site's?

The best linear map explains a few per cent of what patching an image site does.
That number means two very different things depending on what a *nonlinear* map
can do with the same regressor.  If a small network reaches thirty per cent, the
linear-lens class is the problem and the ceiling is a statement about lenses.  If
it reaches four, the site's effect is mostly not a function of its own deviation
at all -- self-repair, position dependence, saturation -- and the ceiling is a
statement about single-site attribution.

Either answer is worth having, and a reviewer will ask for it, so it is measured
rather than argued.  Everything runs on the collection the fit already saved: no
forward passes, no model except to recover the readout.

Three predictors on identical held-out photographs:

    tangent      the corpus-averaged gradient, as a fixed linear map
    ridge        the best linear map (the ceiling of the linear class)
    MLP          one hidden layer on the same regressor
"""
from __future__ import annotations

import argparse
import glob
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
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--collections", default="/workspace/vlm/results/secant")
ap.add_argument("--tag", default="qwen2b")
ap.add_argument("--tangent", default=None)
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct",
                help="only used to recover the readout matrix if the "
                     "collection predates it being saved alongside")
ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--pca", type=int, default=256,
                help="project the regressor before the MLP; the deviations live "
                     "in far fewer dimensions than d_model and an unregularised "
                     "MLP on 2048 inputs would just memorise")
ap.add_argument("--out", default="/workspace/vlm/results/nonlinear")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

TAN, TOKENS = None, None
if a.tangent and os.path.exists(a.tangent):
    _t = torch.load(a.tangent, map_location="cpu", weights_only=False)
    TAN, TOKENS = _t["tangent"], _t.get("tokens")
_R = None


def readout_matrix(c):
    """The collection carries it; older ones do not, so rebuild it once."""
    global _R
    if "readout" in c:
        return c["readout"].double().numpy()
    if _R is None:
        if TOKENS is None:
            raise SystemExit("no readout in the collection and no token list "
                             "in the tangent file; rerun the fit")
        import jvlm
        from jvlm.patchfit import gauge_fix
        mm = jvlm.load_vlm(a.repo)
        W = gauge_fix(mm._lm_head.weight.detach().float().cpu())[TOKENS]
        _R = (W - W.mean(0, keepdim=True)).double().numpy()
        del mm
        torch.cuda.empty_cache()
    return _R

report = {}
for path in sorted(glob.glob(f"{a.collections}/{a.tag}_L*_collection.pt")):
    c = torch.load(path, map_location="cpu", weights_only=False)
    layer = int(c["layer"])
    X = c["X"].double().numpy()
    R = readout_matrix(c)
    Y = c["H"].double().numpy() @ R.T
    g = np.asarray(c["groups"])

    rng = np.random.default_rng(0)
    ug = np.unique(g)
    rng.shuffle(ug)
    held = set(ug[: max(1, len(ug) // 4)].tolist())
    te = np.array([x in held for x in g])
    tr = ~te
    xbar, ybar = X[tr].mean(0), Y[tr].mean(0)
    Yte_c = Y[te] - Y[te].mean(0)
    total = (Yte_c ** 2).sum()

    def fve(pred):
        return float(1.0 - ((Y[te] - pred) ** 2).sum() / max(total, 1e-12))

    row = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
           "n_images": int(len(ug)), "d": int(X.shape[1]),
           "k": int(Y.shape[1])}

    if TAN is not None and layer in TAN:
        A = TAN[layer].double().numpy()
        A = A - A.mean(0, keepdims=True)
        row["tangent"] = round(fve(X[te] @ A.T + (ybar - A @ xbar)), 4)

    Xc, Yc = X[tr] - xbar, Y[tr] - ybar
    S = Xc.T @ Xc / len(Xc)
    C = Yc.T @ Xc / len(Xc)
    scale = float(np.trace(S) / X.shape[1])
    best = None
    for lam in (1e-3, 1e-2, 1e-1, 1, 10, 100):
        L = np.linalg.solve((S + lam * scale * np.eye(X.shape[1])).T, C.T).T
        v = fve(X[te] @ L.T + (ybar - L @ xbar))
        if best is None or v > best[0]:
            best = (v, lam)
    row["ridge"] = round(best[0], 4)
    row["ridge_lambda"] = best[1]

    # --- the same regressor, through one hidden layer
    U, Sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    r = min(a.pca, int((Sv > Sv[0] * 1e-4).sum()))
    P = Vt[:r].T / Sv[:r] * np.sqrt(len(Xc))          # whitened projection
    Zt = torch.tensor((X - xbar) @ P, dtype=torch.float32, device=dev)
    Yt = torch.tensor(Y - ybar, dtype=torch.float32, device=dev)
    ys = float(Yt[torch.tensor(tr, device=dev)].std())
    Yt = Yt / max(ys, 1e-8)
    # a validation split carved out of the *training* photographs, so the epoch
    # is chosen without ever looking at the held-out set.  Selecting on the test
    # set would inflate exactly the number that could overturn the claim.
    tr_imgs = [x for x in ug if x not in held]
    rng.shuffle(tr_imgs)
    val = set(tr_imgs[: max(1, len(tr_imgs) // 5)])
    isval = np.array([x in val for x in g]) & tr
    fit = tr & ~isval
    tri = torch.tensor(np.nonzero(fit)[0], device=dev)
    vai = torch.tensor(np.nonzero(isval)[0], device=dev)
    tei = torch.tensor(np.nonzero(te)[0], device=dev)
    Yva = Y[isval]
    Yva_c = Yva - Yva.mean(0)
    va_total = (Yva_c ** 2).sum()

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(r, a.hidden), nn.GELU(),
                        nn.Linear(a.hidden, a.hidden), nn.GELU(),
                        nn.Linear(a.hidden, Y.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    bs = 256
    best_va, best_mlp = -9e9, float("nan")
    for ep in range(a.epochs):
        net.train()
        perm = tri[torch.randperm(len(tri), device=dev)]
        for i in range(0, len(perm), bs):
            j = perm[i: i + bs]
            loss = ((net(Zt[j]) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 5 == 4 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                pv = net(Zt[vai]).cpu().numpy() * ys + ybar
                va = float(1.0 - ((Yva - pv) ** 2).sum() / max(va_total, 1e-12))
                if va > best_va:
                    best_va = va
                    pred = net(Zt[tei]).cpu().numpy() * ys + ybar
                    best_mlp = fve(pred)
    row["mlp"] = round(best_mlp, 4)
    row["mlp_val"] = round(best_va, 4)
    row["mlp_rank"] = int(r)

    report[str(layer)] = row
    print(f"L{layer}: tangent {row.get('tangent', float('nan')):+.4f}   "
          f"ridge {row['ridge']:+.4f}   MLP {row['mlp']:+.4f}   "
          f"({row['n_train']} train / {row['n_test']} test, rank {r})", flush=True)

json.dump(report, open(f"{a.out}/{a.tag}_stats.json", "w"), indent=2)
print("\nIf the MLP is far above ridge, the ceiling is linearity's.")
print("If it is not, the ceiling is what a single site's own deviation can "
      "explain at all.", flush=True)
