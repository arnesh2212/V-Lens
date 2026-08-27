"""Every lens number in the paper, recomputed offline with an honest split.

Two things are fixed here that the online fit got wrong.

The split.  The first version chose the shrinkage parameter on the same
photographs it reported on, which hands the fitted map a maximum over the
shrinkage path that the fixed baselines never get.  Photographs are now split
three ways: fit on one set, choose lambda on the second, report on the third.

The pooling.  Samples are drawn at several counterfactual sizes, and the
response saturates, so one linear map fit across all of them is worse than one
fit at a single size -- for every lens, but it makes the pooled number a
compromise rather than a measurement.  The headline is therefore the t = 1
slice: real resample patches, one photograph's site replaced by another's.

The forward passes are the expensive part and they are already done: each fit
saved its collection of (deviation, effect) pairs.  So the headline comparison
can be redone without a GPU -- and it has to be, because the first version chose
the shrinkage parameter on the same photographs it reported on, which hands the
fitted map a maximum over the shrinkage path that the fixed baselines never get.

Here the photographs are split three ways: fit on one set, choose lambda on the
second, report on the third.  Nothing that appears below was selected on the
numbers below.

Every lens is given its best single multiplier before scoring.  Without that the
logit lens and the Jacobian lens are punished for a scale neither ever claimed
-- they predict a direction of change, not its size in logits -- and the
comparison becomes meaningless.  With it, each lens is scored on the only thing
it asserts: which way, and in what proportion, the answer moves.

Reported per layer: every candidate lens on the same reporting photographs, the
same thing as a function of counterfactual size, single-site against multi-site
transfer, and the shrinkage path with both the selection and the reported score.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, "/workspace/vlm/code")

import numpy as np
import torch

from jvlm.patchfit import PatchCollector

ap = argparse.ArgumentParser()
ap.add_argument("--collections", default="/workspace/vlm/results/secant")
ap.add_argument("--tag", default="qwen2b")
ap.add_argument("--tangent", default=None)
ap.add_argument("--jlens", default=None)
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct",
                help="only used to rebuild the readout if the collection "
                     "predates it being saved alongside")
ap.add_argument("--out", default="/workspace/vlm/results/refit")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

TAN, TOKENS = None, None
if a.tangent and os.path.exists(a.tangent):
    _t = torch.load(a.tangent, map_location="cpu", weights_only=False)
    TAN, TOKENS = _t["tangent"], _t.get("tokens")
_R = None


def readout_matrix(c):
    """The collection carries it; older ones do not, so rebuild it once."""
    global _R
    if c.R is not None:
        return c.R
    if _R is None:
        if TOKENS is None:
            raise SystemExit("no readout in the collection and no token list "
                             "in the tangent file")
        import jvlm
        from jvlm.patchfit import gauge_fix
        mm = jvlm.load_vlm(a.repo)
        W = gauge_fix(mm._lm_head.weight.detach().float().cpu())[TOKENS]
        _R = W - W.mean(0, keepdim=True)
        del mm
        torch.cuda.empty_cache()
    return _R
J = None
if a.jlens and os.path.exists(a.jlens):
    import jvlm
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))

NAMES = ["logit lens", "Jacobian lens", "our tangent", "the secant"]
report = {}
paths = sorted(glob.glob(f"{a.collections}/{a.tag}_L*_collection.pt"))
if not paths:
    raise SystemExit(f"no collections under {a.collections}")
print(f"{len(paths)} layers\n")
print(" " * 12 + "".join(f"{n:>16}" for n in NAMES))

for p in paths:
    c = PatchCollector.load(p)
    l = c.layer
    R = readout_matrix(c)
    c.R, c.k = R, R.shape[0]
    prior = TAN[l].float() if (TAN and l in TAN) else None
    if prior is not None:
        prior = prior - prior.mean(0, keepdim=True)
    L, b_, rep = c.fit(prior=prior)

    cand = {"logit lens": R, "the secant": L}
    if prior is not None:
        cand["our tangent"] = prior
    if J is not None and l in J.jacobians:
        cand["Jacobian lens"] = R @ J.jacobians[l].float().cpu()
    lad = c.evaluate(cand)

    st = np.asarray(c.steps)
    sz = np.asarray(c.sizes)
    rad = {}
    for t in sorted(set(st.tolist())):
        mk = st == t
        if mk.sum() >= 60:
            try:
                rad[str(t)] = c.evaluate(cand, mask=mk)
            except (ValueError, np.linalg.LinAlgError):
                pass
    single = (c.evaluate(cand, mask=(sz == 1)) if (sz == 1).sum() >= 60 else {})
    multi = (c.evaluate(cand, mask=(sz > 1)) if (sz > 1).sum() >= 60 else {})

    report[str(l)] = {"fit": rep, "ladder": lad, "radius": rad,
                      "single_site": single, "multi_site": multi,
                      "full_patch": rad.get("1.0", {})}
    cells = [f"{lad[n]['fve_scaled']:+.3f}" if n in lad else "--"
             for n in NAMES]
    print(f"  layer {l:<5}" + "".join(f"{x:>16}" for x in cells)
          + f"   (effect {lad['_n']['effect_rms']:.2f}, "
          f"{rep['n_fit']}/{rep['n_choose']}/{rep['n_report']} samples)",
          flush=True)

print("\nreal resample patches only (t = 1), all layers:")
print(" " * 12 + "".join(f"{n:>16}" for n in NAMES))
for l in sorted(report, key=int):
    v = report[l]["radius"].get("1.0")
    if not v:
        continue
    cells = [f"{v[n]['fve_scaled']:+.3f}" if n in v else "--" for n in NAMES]
    print(f"  layer {l:<5}" + "".join(f"{x:>16}" for x in cells)
          + f"   (effect {v['_n']['effect_rms']:.2f})")

print("\nby counterfactual size, middle layer:")
lm = sorted(report, key=int)[len(report) // 2]
print(f"{'t':>10}" + "".join(f"{n:>16}" for n in NAMES))
for t, v in sorted(report[lm]["radius"].items(), key=lambda kv: float(kv[0])):
    cells = [f"{v[n]['fve_scaled']:+.3f}" if n in v else "--" for n in NAMES]
    print(f"{float(t):>10.3f}" + "".join(f"{x:>16}" for x in cells))

print("\nsingle-site only, all layers:")
for l in sorted(report, key=int):
    s_ = report[l]["single_site"]
    if s_:
        cells = [f"{s_[n]['fve_scaled']:+.3f}" if n in s_ else "--"
                 for n in NAMES]
        print(f"  layer {l:<5}" + "".join(f"{x:>16}" for x in cells))

json.dump(report, open(f"{a.out}/{a.tag}_stats.json", "w"), indent=2,
          default=float)
print(f"\nwrote {a.out}/{a.tag}_stats.json", flush=True)
