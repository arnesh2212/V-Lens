"""What the VJ-Lens reading is actually for, measured against both baselines.

The three-lens comparison as first run made two choices that quietly cost the
gradient lens everything it has.

The first is the baseline.  ``do_map`` reads ``grad . (h - b)`` for a reference
activation ``b``.  The comparison computed a 100-image corpus mean and then
passed a *single random donor image* instead, so every reading carried one
image's worth of noise in the term it subtracts.

The second is the normalisation.  Z-scoring down the vocabulary at each position
is the right move for a map -- it stops a picture of activation norm being
mistaken for a picture of meaning -- but the metric that asks whether a lens
*predicts what patching does* is asking about a magnitude, and z-scoring is
exactly the operation that throws magnitude away.

And the raw magnitude is not an arbitrary score.  Replacing the activations in a
region R with a reference b changes the answer's margin by, to first order,

    delta ~= sum_{p in R} grad_p . (h[p] - b[p]),

which is what ``do_map`` returns summed over R.  So VJ-Lens does not merely
correlate with the effect of an intervention; it is the intervention's
first-order estimate, with the sign and the units.  Neither the logit lens nor
the Jacobian lens has any such reading -- their outputs are similarities, not
predicted changes.

This measures, on identical photographs, identical layers and identical patches:

    points/energy    with the corpus mean as baseline, and pooled over regions
                     of r x r patches for r in 1, 2, 3 -- the region coordinate
                     being the method's one parameter, and 1 its noisiest corner
    predicts         Spearman against the true drop, for every lens, and for
                     VJ-Lens both z-scored and raw
    taylor           the raw prediction against the true drop directly:
                     correlation, and the slope of a line through the origin,
                     which is 1 if the first-order estimate is calibrated
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
from scipy import stats as sps

import jvlm
from jvlm import cocodata, corpus, readout, stats as st, vmap

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens", default=None)
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--mean-images", type=int, default=100)
ap.add_argument("--n-layers", type=int, default=4)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-frac", type=float, default=0.45)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--pools", type=int, nargs="*", default=[1, 2, 3, 6])
ap.add_argument("--ig-steps", type=int, default=4,
                help="path length for the integrated arm; 0 disables it")
ap.add_argument("--out", default="/workspace/vlm/results/decide")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
LAYERS = [int(round(f * (m.n_layers - 1)))
          for f in np.linspace(0.18, 0.88, a.n_layers)]
J = None
if a.jlens:
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))
    print(f"J-lens loaded: layers {min(J.jacobians)}..{max(J.jacobians)}",
          flush=True)

CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
FLAT = sorted({t for ids in CAT_IDS.values() for t in ids})
POSI = {t: i for i, t in enumerate(FLAT)}
print(m, f"\nlayers {LAYERS}   {len(CATS)} categories", flush=True)

PROMPT = "Describe this image."


def prep(path):
    return m.chat(PROMPT, [corpus.load_image(path, square=a.square)])


def margin(logits, ids):
    v = logits[FLAT].float()
    return float(max(v[POSI[t]] for t in ids) - v.mean())


def pool_heat(heat: torch.Tensor, G: int, r: int) -> torch.Tensor:
    """Mean over r x r blocks, broadcast back so every patch carries its region.

    This is the region coordinate applied to a map: a patch is scored by the
    region it belongs to rather than by itself alone.  r = 1 leaves the map at
    single-patch resolution, which is the corner of the family with the most
    variance in it.
    """
    if r <= 1:
        return heat
    g = heat.reshape(G, G)
    pad = (-G) % r
    if pad:
        g = torch.nn.functional.pad(g[None, None], (0, pad, 0, pad),
                                    mode="replicate")[0, 0]
    H = g.shape[0]
    blocks = g.reshape(H // r, r, H // r, r).mean((1, 3))
    up = blocks.repeat_interleave(r, 0).repeat_interleave(r, 1)[:G, :G]
    return up.reshape(-1)


pool = cocodata.concept_items(6000, categories=CATS, min_area=0.04, seed=7)
b0 = prep(pool[0]["path"])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
nrm = {l: torch.zeros(P, dtype=torch.float64) for l in LAYERS}
bank = {l: [] for l in LAYERS}
nm = 0
for it in pool[: a.mean_images]:
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    A = readout.residuals(m, b, LAYERS)
    for l in LAYERS:
        v = A[l][ip].float().cpu()
        acc[l] += v.double()
        nrm[l] += v.norm(dim=-1).double()
        if len(bank[l]) < 16:
            bank[l].append(v)
    nm += 1
MEANS = {l: (acc[l] / max(nm, 1)).float() for l in LAYERS}
NORM = {l: (nrm[l] / max(nm, 1)).float() for l in LAYERS}
REG = {l: (NORM[l] > NORM[l].mean() + 3 * NORM[l].std()) for l in LAYERS}
print(f"means from {nm}; {P} tokens on a {G}x{G} grid "
      f"({time.time() - t0:.0f}s); {int(sum(REG[LAYERS[0]]))} register tokens",
      flush=True)

items = cocodata.concept_items(a.n, categories=CATS, min_area=0.07, seed=808,
                               tokenizer=m.tokenizer)
rows, seen = [], 0
for k, it in enumerate(items):
    c = it["concept"]
    if c not in CAT_IDS:
        continue
    boxes = cocodata.object_boxes(it["image_id"], c)
    if not boxes:
        continue
    inside = sorted({t for bx in boxes for t in cocodata.box_tokens(bx, G)})
    nb = len(inside)
    if not (3 <= nb <= int(a.max_frac * P)):
        continue
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    ids = CAT_IDS[c]
    col = [POSI[t] for t in ids]
    A = readout.residuals(m, b, LAYERS + [m.n_layers - 1])
    base = margin(m.unembed(A[m.n_layers - 1][-1:].to(m.device))[0], ids)
    donor = {l: bank[l][k % len(bank[l])] for l in LAYERS}
    ins = torch.tensor(inside)

    for l in LAYERS:
        h = A[l][ip].float().cpu()
        lg = m.unembed(h.to(m.device)).float().cpu()[:, FLAT].T       # [k, P]
        raw = {"logit lens": lg}
        if J is not None and l in J.jacobians:
            jl, _ = readout.lens_logits(m, J, b, layers=[l],
                                        positions=ip.tolist())
            raw["J-lens"] = jl[l].float().cpu()[:, FLAT].T
        # the two baselines, one backward pass each
        raw["VJ-Lens"] = vmap.do_map(m, b, FLAT, [l], ip.tolist(),
                                     means={l: MEANS[l]})[l]
        raw["VJ-Lens (donor)"] = vmap.do_map(m, b, FLAT, [l], ip.tolist(),
                                             means={l: donor[l]})[l]
        if a.ig_steps > 1:
            raw["VJ-Lens (path)"] = vmap.ig_map(
                m, b, FLAT, [l], ip.tolist(), means={l: MEANS[l]},
                steps=a.ig_steps)[l]

        # the true effect of replacing the object's patches with the donor
        pos = ip[ins].to(m.device)
        vals = donor[l][ins].to(m.device)

        def hook(mod, inp, out, _p=pos, _v=vals):
            tup = not torch.is_tensor(out)
            h_ = (out[0] if tup else out).clone()
            h_[0, _p] = _v.to(h_.dtype)
            return (h_, *out[1:]) if tup else h_

        hd = m.layers[l].register_forward_hook(hook)
        try:
            with torch.no_grad():
                AA = readout.residuals(m, b, [m.n_layers - 1])
                v_after = margin(m.unembed(
                    AA[m.n_layers - 1][-1:].to(m.device))[0], ids)
        finally:
            hd.remove()
        drop = base - v_after

        for name, mat in raw.items():
            z = vmap.normalise(mat)
            for r in a.pools:
                heat = z[col].max(0).values.clone()
                heat[REG[l]] = heat[~REG[l]].min() if (~REG[l]).any() else 0.0
                ph = pool_heat(heat, G, r)
                pt = vmap.pointing(ph, set(inside), G)
                rows.append(dict(i=k, concept=c, lens=name, layer=l, pool=r,
                                 depth=round(l / (m.n_layers - 1), 3),
                                 n_tok=nb, drop=drop, base=base,
                                 pred_z=float(heat[ins].sum()), **pt))
            # the raw, un-z-scored reading over the region: for VJ-Lens with the
            # donor baseline this is the first-order estimate of `drop` itself,
            # in the same units, so it can be compared to it directly and not
            # only ranked against it
            sub = mat[:, ins]
            taylor = float(sub[col].max(0).values.sum() - sub.mean(0).sum())
            for r_ in rows[-len(a.pools):]:
                r_["pred_raw"] = taylor
    seen += 1
    if k % 10 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  kept {seen}  {el:.0f}s  "
              f"eta {el / (k + 1) * (len(items) - k - 1) / 60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")

pt = df.groupby(["lens", "pool", "layer"]).agg(
    n=("i", "count"), point=("hit", "mean"), point5=("hit5", "mean"),
    energy=("mass_in", "mean"), chance=("chance", "mean")).reset_index()
pt.to_csv(f"{a.out}/{tag}_points.csv", index=False)
print("\nPOINTING AND ENERGY, corpus mean baseline, by region size\n",
      pt.round(3).to_string(index=False), flush=True)

best = pt.groupby(["lens", "pool"])[["point", "point5", "energy"]].max()
print("\nBEST OVER DEPTH\n", best.round(3).to_string(), flush=True)

cor = []
for lens in df.lens.unique():
    for l in LAYERS:
        d = df[(df.lens == lens) & (df.layer == l) & (df.pool == a.pools[0])]
        d = d.dropna(subset=["drop", "pred_z", "pred_raw"])
        if len(d) < 8:
            continue
        row = dict(lens=lens, layer=l, n=len(d),
                   z=sps.spearmanr(d.pred_z, d.drop).statistic,
                   raw=sps.spearmanr(d.pred_raw, d.drop).statistic,
                   pearson_raw=sps.pearsonr(d.pred_raw, d.drop).statistic)
        # slope of a line through the origin: 1 means the first-order estimate
        # is calibrated, not merely monotone in the truth
        x = d.pred_raw.values
        row["slope"] = float((x * d.drop.values).sum() / (x * x).sum()) \
            if (x * x).sum() > 0 else np.nan
        cor.append(row)
cdf = pd.DataFrame(cor)
cdf.to_csv(f"{a.out}/{tag}_predicts.csv", index=False)
print("\nPREDICTS THE DROP  (spearman; raw = un-z-scored region sum)\n",
      cdf.round(3).to_string(index=False), flush=True)

out = {"repo": a.repo, "n": int(df.i.nunique()), "grid": G, "P": P,
       "layers": LAYERS, "n_layers": m.n_layers, "pools": a.pools,
       "points": pt.round(4).to_dict("records"),
       "best": best.round(4).reset_index().to_dict("records"),
       "predicts": cdf.round(4).to_dict("records"),
       "minutes": round((time.time() - t0) / 60, 1)}
json.dump(st.jsonable(out), open(f"{a.out}/{tag}_stats.json", "w"), indent=2,
          default=float)
print("done", flush=True)
