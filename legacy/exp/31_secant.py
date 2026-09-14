"""Fit the secant, and score it against every tangent on the same photographs.

Four linear surrogates for the same causal question -- if this image site were
different, what would the model say?

  logit-lens  the site read straight through the unembedding: no transport at
              all, and the baseline any vision lens has to beat.
  reference   the corpus-averaged Jacobian read through the unembedding, i.e.
              the transport as published.
  tangent     the same derivative estimated directly, as the gradient of each
              token's answer logit with respect to an image site, so the final
              norm and the answer targeting are exact.  This is the strongest
              member of the tangent family and the secant's shrinkage prior.
  secant      the least-squares fit to real resample-patching outcomes.

Three things are measured, in order of how much they matter.

FIT.  Held-out variance explained in the true interventional effect, with whole
photographs held out.  The tangent is the infinite-shrinkage end of the secant's
own path, so the two are scored on identical data and the gap is read off one
curve.  Transfer is measured too: fit on multi-site patches, scored on
single-site ones, because a map that only works on the family it was fit to is
not a lens.

CERTIFICATE.  Under Gaussian -- indeed elliptical -- deviations with no context
dependence, the secant equals the tangent exactly.  So a measured gap is not a
free parameter: it certifies that image-site deviations are non-elliptical, or
that a site's downstream effect depends on the context it sits in.  Both are
reported, along with the interaction fraction that scopes the summed design.

READOUT.  Where the photographed object's own name sits, read at the image
tokens.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, "/workspace/vlm/code")
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch

import jvlm
from jvlm import cocodata, corpus, direct, readout
from jvlm.patchfit import PatchCollector, SecantLens, gauge_fix

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens", default=None)
ap.add_argument("--fit-images", type=int, default=600)
ap.add_argument("--samples", type=int, default=8)
ap.add_argument("--donors", type=int, default=32)
ap.add_argument("--tangent-images", type=int, default=24)
ap.add_argument("--interaction-n", type=int, default=12)
ap.add_argument("--eval-n", type=int, default=120)
ap.add_argument("--n-layers", type=int, default=4)
ap.add_argument("--wide", type=int, default=0,
                help="also fit a second lens over this many of the model's own "
                     "top answer tokens, reusing the same collection")
ap.add_argument("--vocab", type=int, default=0,
                help="fit on this many of the model's own top answer tokens "
                     "as well as the concept words; 0 keeps the concept words "
                     "only.  The map is k x d and the solve is over d, so a "
                     "large vocabulary costs almost nothing")
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/secant")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
J = None
if a.jlens and os.path.exists(a.jlens):
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))
lo, hi = max(m.n_deepstack, 1), m.n_layers - 2
LAYERS = sorted({int(round(lo + (hi - lo) * f))
                 for f in np.linspace(0.2, 0.85, a.n_layers)})

CATS = cocodata.SINGLE_WORD
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
CAT_IDS = {c: v for c, v in CAT_IDS.items() if v}
FLAT = sorted({t for v in CAT_IDS.values() for t in v})
POSI = {t: i for i, t in enumerate(FLAT)}
print(m, f"\nlayers {LAYERS}   {len(CAT_IDS)} concepts / {len(FLAT)} tokens   "
      f"square {a.square}px", flush=True)

PROMPT = "Describe this image."
fit_items = cocodata.concept_items(a.fit_images, min_area=0.04, seed=909)


def prep(path):
    return m.chat(PROMPT, [corpus.load_image(path, square=a.square)])


if a.vocab:
    # the vocabulary the model is actually choosing among on these photographs,
    # so the lens is fit where its readout will be used
    with torch.no_grad():
        tot = torch.zeros(m._lm_head.weight.shape[0])
        for it in fit_items[:64]:
            b = prep(it["path"])
            A_ = readout.residuals(m, b, [m.n_layers - 1])
            lg = m.unembed(A_[m.n_layers - 1][-1:].to(m.device))[0]
            tot += lg.float().softmax(-1).cpu()
    top = torch.topk(tot, a.vocab).indices.tolist()
    FLAT = sorted(set(FLAT) | set(top))
    POSI = {t: i for i, t in enumerate(FLAT)}
    print(f"vocabulary widened to {len(FLAT)} tokens", flush=True)


# two gauges, both free: the softmax ignores a uniform shift of all logits, and
# the readout ranks concepts against each other, so it ignores a uniform shift
# of the concept vocabulary too.  Removing both leaves only the part of the
# effect a lens is ever read for -- and makes the regression target that much
# less dominated by "patching anything moves everything".
R0 = gauge_fix(m._lm_head.weight.detach().float().cpu())[FLAT]
R = R0 - R0.mean(0, keepdim=True)

cols = {l: PatchCollector(m, l, R) for l in LAYERS}

# ---- pass one: per-position means, and a bank of donor photographs ----------
b0 = prep(fit_items[0]["path"])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
DON = {l: [] for l in LAYERS}
nm = 0
for it in fit_items:
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    A = readout.residuals(m, b, LAYERS)
    for l in LAYERS:
        v = A[l][ip].float().cpu()
        acc[l] += v.double()
        if len(DON[l]) < a.donors:
            DON[l].append(v.half())
    nm += 1
    if nm >= max(a.donors, 200):
        break
MEANS = {l: (acc[l] / max(nm, 1)).float() for l in LAYERS}
DON = {l: [d.float() for d in v] for l, v in DON.items()}
print(f"{P} image tokens; means from {nm} photographs, "
      f"{len(DON[LAYERS[0]])} donors  ({time.time()-t0:.0f}s)", flush=True)

# ---- the tangent, estimated directly ---------------------------------------
tan = direct.DirectDirections(LAYERS)
for it in fit_items[: a.tangent_images]:
    try:
        tan.add(m, prep(it["path"]), FLAT)
    except (ValueError, torch.cuda.OutOfMemoryError):
        torch.cuda.empty_cache()
        continue
TAN = {l: (v.float() - v.float().mean(0, keepdim=True))
       for l, v in tan.mean("img", normalize=False).items()}   # same gauge
TAN_STAB = tan.stability("img")
print(f"tangent from {tan.n} photographs x {len(FLAT)} tokens; stability "
      f"{np.mean(list(TAN_STAB.values())):.3f}  ({time.time()-t0:.0f}s)",
      flush=True)

# ---- the secant: resample-patch counterfactuals ------------------------------
gen = torch.Generator().manual_seed(0)
# a mixed design: single-site samples identify the per-site map, multi-site
# samples buy efficiency for the aggregated one the lens actually uses
SIZES = np.array([1, 1, 2, 4, 4, 8, 8, 16])
# how far the counterfactual goes: t = 1 is a real resample patch, smaller t a
# smaller one.  Sweeping it measures the radius over which a linear surrogate
# stays true, which is the quantity no lens has ever reported.
STEPS = np.array([1.0, 0.5, 1.0, 0.25, 1.0, 0.5, 1.0, 0.125])
inter, rel = [], []
n_ok = 0
for gi, it in enumerate(fit_items):
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    pm = torch.randperm(len(SIZES), generator=gen).numpy()
    sizes = SIZES[pm][: a.samples]
    steps = STEPS[pm][: a.samples]
    for l in LAYERS:
        cols[l].add(b, ip.cpu(), DON[l], n_samples=a.samples, group=gi,
                    generator=gen, sizes=sizes, steps=steps)
    if n_ok < a.interaction_n:
        rel.append(cols[LAYERS[0]].reliability(b, ip.cpu(), DON[LAYERS[0]],
                                               generator=gen))
        inter.append(cols[LAYERS[len(inter) % len(LAYERS)]]
                     .interaction_fraction(b, ip.cpu(), DON[LAYERS[len(inter) % len(LAYERS)]],
                                           subset=8, generator=gen))
    n_ok += 1
    if gi % 50 == 0:
        print(f"  patch {gi}/{len(fit_items)}  "
              f"{len(cols[LAYERS[0]].X)} samples  {time.time()-t0:.0f}s",
              flush=True)
REL = (round(float(np.nanmean([x["reliability"] for x in rel if x])), 4)
       if any(rel) else float("nan"))
print(f"reliability of the target (same patch, run twice): {REL}", flush=True)
INTER = ({k: round(float(np.nanmean([x[k] for x in inter])), 4)
          for k in ("raw", "scaled", "alpha")} if inter else {})
print(f"non-additivity at k=8: raw {INTER.get('raw')}  after rescaling "
      f"{INTER.get('scaled')}  (single effects sum to "
      f"{1/max(INTER.get('alpha',1),1e-9):.1f}x the joint one)  "
      f"({time.time()-t0:.0f}s)", flush=True)

maps, inters, reports = {}, {}, {}
LADDER, RADIUS, SINGLE, GAUGE, PATHMAP = {}, {}, {}, {}, {}
for l in LAYERS:
    c = cols[l]
    c.save(f"{a.out}/{tag}_L{l}_collection.pt")
    sz = np.asarray(c.sizes)
    L, b_, rep = c.fit(prior=TAN[l], keep_path=True)
    # one scalar slides the same lens from the derivative to the fitted map.
    # Scoring both criteria along it turns a comparison between methods into a
    # trade-off curve inside one method, which no difference between methods can
    # explain away.
    PATHMAP[l] = dict(c.path_maps)
    # transfer: fit only on multi-site patches, score on single-site ones
    try:
        _, _, tr = c.fit(prior=TAN[l], mask=(sz > 1))
        _, _, sg = c.fit(prior=TAN[l], mask=(sz == 1))
        rep["multi_only_fve"] = tr["held_out_fve"]
        rep["single_only_fve"] = sg["held_out_fve"]
        rep["single_only_tangent_fve"] = sg["tangent_fve"]
    except ValueError:
        pass
    rep["non_additivity"] = INTER
    # every lens is a linear map from a site's deviation to a change in the
    # vocabulary's logits, so every lens can be asked to predict the same real
    # patching outcomes on photographs none of them saw
    cand = {"logit lens": R, "our tangent": TAN[l], "the secant": L}
    if J is not None and l in J.jacobians:
        cand["Jacobian lens"] = R @ J.jacobians[l].float().cpu()
    LADDER[l] = c.evaluate(cand)
    # the same maps, scored separately at each counterfactual size
    st_ = np.asarray(c.steps)
    RADIUS[l] = {}
    for t in sorted(set(st_.tolist())):
        mk = st_ == t
        if mk.sum() < 60:
            continue
        try:
            RADIUS[l][t] = c.evaluate(cand, mask=mk)
        except (ValueError, np.linalg.LinAlgError):
            continue
    sk = np.asarray(c.sizes)
    SINGLE[l] = (c.evaluate(cand, mask=(sk == 1))
                 if (sk == 1).sum() >= 60 else {})
    # the best linear map in the ungauged readout, so the two ceilings can be
    # compared: how much of what is linearly predictable is the uniform shift a
    # softmax cannot see, and how much is the part a lens is read for
    try:
        L0, _, r0 = c.fit(readout=R0)
        GAUGE[l] = {"best_linear_ungauged": r0["held_out_fve"],
                    "best_linear_gauged": rep["held_out_fve"],
                    "logit_lens_ungauged":
                        c.evaluate({"x": R0}, readout=R0)["x"]["fve"]}
    except (ValueError, np.linalg.LinAlgError):
        GAUGE[l] = {}
    maps[l], inters[l], reports[l] = L, b_, rep
    print(f"  L{l}: lam={rep['chosen_lambda']:g}  FVE secant "
          f"{rep['held_out_fve']:+.3f} / tangent {rep['tangent_fve']:+.3f}   "
          f"best-scaled {rep['held_out_fve_gain']:+.3f} / "
          f"{rep['tangent_fve_gain']:+.3f}   corr "
          f"{rep['held_out_corr']:.3f} / {rep['tangent_corr']:.3f}   gap "
          f"{rep['relative_gap']:.3f}  rank90={rep['sigma_rank90']}", flush=True)

NAMES = ["logit lens", "Jacobian lens", "our tangent", "the secant"]
print("\nhow much of a real patch each lens predicts, on held-out photographs:")
print(" " * 12 + "".join(f"{n:>16}" for n in NAMES))
for l in LAYERS:
    cells = []
    for n in NAMES:
        v = LADDER[l].get(n)
        cells.append(f"{v['fve']:+.3f}" if v else "--")
    mag = LADDER[l]["_n"]["effect_rms"]
    print(f"  layer {l:<5}" + "".join(f"{c:>16}" for c in cells)
          + f"    (effect size {mag:.2f} logits)", flush=True)

print("\nand as a function of how big the counterfactual is (layer "
      f"{LAYERS[len(LAYERS)//2]}):")
lm = LAYERS[len(LAYERS) // 2]
print(f"{'step t':>10}" + "".join(f"{n:>16}" for n in NAMES))
for t, v in sorted(RADIUS.get(lm, {}).items()):
    cells = [f"{v[n]['fve']:+.3f}" if n in v else "--" for n in NAMES]
    print(f"{t:>10.3f}" + "".join(f"{c:>16}" for c in cells), flush=True)

lens = SecantLens(maps, inters, MEANS, FLAT, reports)
lens.save(f"{a.out}/{tag}_secant.pt")

if a.wide:
    # the effects were recorded in residual space, so a second lens over the
    # model's whole working vocabulary costs one more solve and no forward
    # passes at all.  There is no tangent to shrink toward at this width -- one
    # backward pass per token is what makes that infeasible, and is the reason
    # the comparison above is run on the concept words.
    with torch.no_grad():
        tot = torch.zeros(m._lm_head.weight.shape[0])
        for it in fit_items[:64]:
            A_ = readout.residuals(m, prep(it["path"]), [m.n_layers - 1])
            tot += m.unembed(A_[m.n_layers - 1][-1:].to(m.device))[0].float(
                ).softmax(-1).cpu()
    WIDE = sorted(set(FLAT) | set(torch.topk(tot, a.wide).indices.tolist()))
    RW = gauge_fix(m._lm_head.weight.detach().float().cpu())[WIDE]
    RW = RW - RW.mean(0, keepdim=True)
    wmaps, winters, wrep = {}, {}, {}
    for l in LAYERS:
        Lw, bw, rw = cols[l].fit(readout=RW)
        wmaps[l], winters[l], wrep[l] = Lw, bw, rw
        print(f"  wide L{l}: lambda={rw['chosen_lambda']:g}  FVE "
              f"{rw['held_out_fve']:+.3f}  corr {rw['held_out_corr']:.3f}",
              flush=True)
    SecantLens(wmaps, winters, MEANS, WIDE, wrep).save(
        f"{a.out}/{tag}_secant_wide.pt")
    print(f"wide lens over {len(WIDE)} tokens saved", flush=True)
torch.save({"tangent": TAN, "stability": TAN_STAB, "tokens": FLAT},
           f"{a.out}/{tag}_tangent.pt")

# ---- is the deviation elliptical?  (if it were, the gap would be zero) -------
def ellipticity(X, r=64, n_dir=256, seed=0):
    """Spread of directional kurtosis after whitening, against a Gaussian null.

    Every elliptical law has direction-independent whitened kurtosis, so a
    spread above the null is a certificate of non-ellipticity -- the condition
    under which the secant must differ from the tangent.
    """
    rng = np.random.default_rng(seed)
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    r = min(r, (S > S[0] * 1e-3).sum())
    Z = U[:, :r] * np.sqrt(len(Xc))                      # whitened scores
    def spread(Z):
        D = rng.normal(size=(Z.shape[1], n_dir))
        D /= np.linalg.norm(D, axis=0, keepdims=True)
        p = Z @ D
        k = (p ** 4).mean(0) / (p ** 2).mean(0) ** 2
        return float(k.std() / k.mean()), float(k.mean())
    obs, kobs = spread(Z)
    null, knull = spread(rng.normal(size=Z.shape))
    return {"kurtosis_spread": round(obs, 4), "gaussian_null": round(null, 4),
            "ratio": round(obs / max(null, 1e-9), 2),
            "mean_kurtosis": round(kobs, 3), "null_kurtosis": round(knull, 3),
            "rank": int(r)}


ELL = {str(l): ellipticity(np.stack(cols[l].X)) for l in LAYERS}
print("\nnon-ellipticity of image-site deviations:")
for l, e in ELL.items():
    print(f"  L{l}: spread {e['kurtosis_spread']:.3f} vs Gaussian null "
          f"{e['gaussian_null']:.3f}  ({e['ratio']}x), kurtosis "
          f"{e['mean_kurtosis']}", flush=True)

# ---- readout ----------------------------------------------------------------
ev = cocodata.concept_items(a.eval_n, min_area=0.08, seed=7,
                            tokenizer=m.tokenizer)
rows = []
for k, it in enumerate(ev):
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    A = readout.residuals(m, b, LAYERS)
    here = set(it["present"])
    for l in LAYERS:
        h = A[l][ip].float().cpu()
        d = h - MEANS[l]
        # the transport and the centring are separate choices, so both are
        # varied: otherwise "centred" and "transported" are confounded and the
        # comparison says nothing about either
        views = {"secant": d @ maps[l].T + inters[l],
                 "tangent": d @ TAN[l].T,
                 "tangent, uncentred": h @ TAN[l].T,
                 "logit-lens": m.unembed(h.to(m.device)).float().cpu()[:, FLAT],
                 "logit-lens, centred": d @ R.T}
        for lam, (Lm, bm) in PATHMAP.get(l, {}).items():
            views[f"shrink {lam:g}"] = d @ Lm.T + bm
        if J is not None and l in J.jacobians:
            lg, _ = readout.lens_logits(m, J, b, layers=[l],
                                        positions=ip.tolist())
            views["reference"] = lg[l].float().cpu()[:, FLAT]
        for name, lgv in views.items():
            best = lgv.max(0).values
            order = best.argsort(descending=True)
            rank = torch.empty_like(order)
            rank[order] = torch.arange(len(order))
            for c in CAT_IDS:
                rows.append(dict(i=k, rung=name, layer=l,
                                 depth=round(l / (m.n_layers - 1), 3), cat=c,
                                 present=int(c in here),
                                 target=int(c == it["concept"]),
                                 rank=int(min(rank[POSI[t]] for t in CAT_IDS[c]))))
    if k % 25 == 0:
        print(f"  eval {k}/{len(ev)}  {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg]); o = allv.argsort()
    rk = np.empty(len(allv), float); rk[o] = np.arange(len(allv))
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2)
                 / (len(pos) * len(neg)))


summ = []
for name in df.rung.unique():
    for l in LAYERS:
        d = df[(df.rung == name) & (df.layer == l)]
        tgt = d[d.target == 1]["rank"]
        summ.append(dict(rung=name, layer=l, depth=round(l / (m.n_layers - 1), 3),
                         median_rank=float(tgt.median()),
                         top1=float((tgt == 0).mean()),
                         top5=float((tgt < 5).mean()),
                         auc=auc(-d[d.present == 1]["rank"].values.astype(float),
                                 -d[d.present == 0]["rank"].values.astype(float))))
S = pd.DataFrame(summ)
S.to_csv(f"{a.out}/{tag}_summary.csv", index=False)
print("\n", S.round(4).to_string(index=False))
tab = S.groupby("rung").agg(best_auc=("auc", "max"), best_top5=("top5", "max"),
                            best_top1=("top1", "max"),
                            best_rank=("median_rank", "min"))
print("\n", tab.round(4).to_string())
json.dump({"repo": a.repo, "layers": LAYERS, "n_eval": int(df.i.nunique()),
           "n_concepts": len(CAT_IDS), "n_tokens": len(FLAT),
           "n_image_tokens": P, "readout": tab.round(4).to_dict(),
           "fit": {str(l): reports[l] for l in LAYERS},
           "shrinkage_path": {str(l): reports[l]["path"] for l in LAYERS},
           "ladder": {str(l): LADDER[l] for l in LAYERS},
           "radius": {str(l): {str(t): v for t, v in RADIUS[l].items()}
                      for l in LAYERS},
           "single_site": {str(l): SINGLE[l] for l in LAYERS},
           "ungauged": {str(l): GAUGE[l] for l in LAYERS},
           "ellipticity": ELL, "non_additivity": INTER, "reliability": REL,
           "tangent_stability": {str(k): round(v, 4) for k, v in TAN_STAB.items()},
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
print("done", flush=True)
