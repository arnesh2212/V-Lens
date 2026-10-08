"""Three lenses, every metric, on the same photographs.

The comparison the paper turns on, done once and properly.  Three readings of
the same image token, differing only in what sits between the activation and
the vocabulary:

    logit lens   W_U h            -- no transport at all
    J-lens       W_U norm(J h)    -- transport by the corpus-averaged Jacobian,
                                     one matrix per layer, positions averaged
    ATLAS        v_t(R) . delta   -- transport by the Jacobian at *this* region,
                                     never averaged over positions

Every score is standardised down the vocabulary at each position, so a map is a
map of which word a place is pushing hardest and not a map of how much residual
happens to sit there.  Without that, ATLAS carries a norm confound the logit
lens does not, and the comparison is unfair in our favour on some metrics and
against us on others.

Four measurements, all on identical photographs and identical budgets:

    points     does the map's peak land inside the object's box
    energy     what share of the map's mass does
    predicts   does the map predict what covering the object really does
    deletes    does covering in the map's order destroy the object fastest
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, "code")
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
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--mean-images", type=int, default=120)
ap.add_argument("--n-layers", type=int, default=6)
ap.add_argument("--steps", type=int, nargs="*", default=[0, 6, 14, 28, 48, 76])
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-frac", type=float, default=0.45)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/compare")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lo, hi = max(m.n_deepstack, 1), m.n_layers - 2
LAYERS = sorted({int(round(lo + (hi - lo) * f))
                 for f in np.linspace(0.1, 0.9, a.n_layers)})
J = None
if a.jlens and os.path.exists(a.jlens):
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))
    print(f"J-lens loaded: layers {min(J.jacobians)}..{max(J.jacobians)}",
          flush=True)
CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
FLAT = sorted({t for v in CAT_IDS.values() for t in v})
POSI = {t: i for i, t in enumerate(FLAT)}
print(m, f"\nlayers {LAYERS}   {len(CATS)} categories", flush=True)

PROMPT = "Describe this image."


def prep(path):
    return m.chat(PROMPT, [corpus.load_image(path, square=a.square)])


def margin(logits, ids):
    v = logits[FLAT].float()
    return float(max(v[POSI[t]] for t in ids) - v.mean())


pool = cocodata.concept_items(6000, min_area=0.04, seed=7)
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
        acc[l] += v.double(); nrm[l] += v.norm(dim=-1).double()
        if len(bank[l]) < 16:
            bank[l].append(v)
    nm += 1
MEANS = {l: (acc[l] / max(nm, 1)).float() for l in LAYERS}
NORM = {l: (nrm[l] / max(nm, 1)).float() for l in LAYERS}
REG = {l: (NORM[l] > NORM[l].mean() + 3 * NORM[l].std()) for l in LAYERS}
STEPS = [s for s in a.steps if s <= P]
print(f"means from {nm}; {P} tokens on a {G}x{G} grid ({time.time()-t0:.0f}s)",
      flush=True)

items = cocodata.concept_items(a.n, min_area=0.07, seed=808,
                               tokenizer=m.tokenizer)
rows, dele = [], []
seen = 0
for k, it in enumerate(items):
    c = it["concept"]
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
    rng = np.random.default_rng(700 + k)
    out_pool = [t for t in range(P) if t not in set(inside)]
    ctrl = sorted(rng.choice(out_pool, size=min(nb, len(out_pool)),
                             replace=False).tolist())

    for l in LAYERS:
        h = A[l][ip].float().cpu()
        donor = bank[l][k % len(bank[l])]
        lg = m.unembed(h.to(m.device)).float().cpu()[:, FLAT].T      # [k, P]
        maps = {"logit lens": vmap.normalise(lg)}
        if J is not None and l in J.jacobians:
            jl, _ = readout.lens_logits(m, J, b, layers=[l],
                                        positions=ip.tolist())
            maps["J-lens"] = vmap.normalise(jl[l].float().cpu()[:, FLAT].T)
        maps["ATLAS"] = vmap.normalise(
            vmap.do_map(m, b, FLAT, [l], ip.tolist(), means={l: donor})[l])
        for name, mm in maps.items():
            heat = mm[col].max(0).values.clone()
            heat[REG[l]] = heat[~REG[l]].min() if (~REG[l]).any() else 0.0
            r = vmap.pointing(heat, set(inside), G)
            rows.append(dict(i=k, concept=c, lens=name, layer=l,
                             depth=round(l / (m.n_layers - 1), 3),
                             n_tok=nb, pred=float(heat[inside].sum()), **r))
            # deletion, in this map's order
            order = heat.argsort(descending=True).tolist()
            for s in STEPS:
                if s == 0:
                    dele.append(dict(i=k, lens=name, layer=l, frac=0.0,
                                     rel=1.0))
                    continue
                tk = order[:s]
                pos = ip[torch.tensor(tk)].to(m.device)
                vals = donor[torch.tensor(tk)].to(m.device)

                def hook(mod, inp, out, _p=pos, _v=vals):
                    tup = not torch.is_tensor(out)
                    h_ = (out[0] if tup else out).clone()
                    h_[0, _p] = _v.to(h_.dtype)
                    return (h_, *out[1:]) if tup else h_

                hd = m.layers[l].register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        AA = readout.residuals(m, b, [m.n_layers - 1])
                        v = margin(m.unembed(
                            AA[m.n_layers - 1][-1:].to(m.device))[0], ids)
                finally:
                    hd.remove()
                dele.append(dict(i=k, lens=name, layer=l, frac=s / P,
                                 rel=v / base if base > 0 else np.nan))

        # what covering the object really does, once per layer
        for arm, tk in (("object", inside), ("control", ctrl)):
            pos = ip[torch.tensor(tk)].to(m.device)
            vals = donor[torch.tensor(tk)].to(m.device)

            def hook2(mod, inp, out, _p=pos, _v=vals):
                tup = not torch.is_tensor(out)
                h_ = (out[0] if tup else out).clone()
                h_[0, _p] = _v.to(h_.dtype)
                return (h_, *out[1:]) if tup else h_

            hd = m.layers[l].register_forward_hook(hook2)
            try:
                with torch.no_grad():
                    AA = readout.residuals(m, b, [m.n_layers - 1])
                    v = margin(m.unembed(
                        AA[m.n_layers - 1][-1:].to(m.device))[0], ids)
            finally:
                hd.remove()
            if arm == "object":
                for r_ in rows[-len(maps):]:
                    if r_["i"] == k and r_["layer"] == l:
                        r_["drop"] = base - v
    seen += 1
    if k % 10 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  kept {seen}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_point.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_point.parquet")
dd = pd.DataFrame(dele)
dd.to_parquet(f"{a.out}/{tag}_del.parquet")

pt = df.groupby(["lens", "layer"]).agg(
    n=("i", "count"), point=("hit", "mean"), point5=("hit5", "mean"),
    energy=("mass_in", "mean"), chance=("chance", "mean")).reset_index()
pt.to_csv(f"{a.out}/{tag}_summary.csv", index=False)
print("\nPOINTING AND ENERGY\n", pt.round(3).to_string(index=False))

cor = []
for lens in df.lens.unique():
    for l in LAYERS:
        d = df[(df.lens == lens) & (df.layer == l)].dropna(subset=["drop"])
        if len(d) > 10:
            cor.append(dict(lens=lens, layer=l, n=len(d),
                            spearman=float(sps.spearmanr(d.pred, d["drop"]).statistic)))
co = pd.DataFrame(cor)
print("\nPREDICTS THE DROP\n",
      co.pivot_table(index="layer", columns="lens", values="spearman")
      .round(3).to_string())

auc_ = []
for lens in dd.lens.unique():
    for l in LAYERS:
        s_ = dd[(dd.lens == lens) & (dd.layer == l)].groupby("frac")["rel"].mean()
        if len(s_) > 2:
            auc_.append(dict(lens=lens, layer=l,
                             auc=float(np.trapz(s_.values, s_.index))))
au = pd.DataFrame(auc_)
print("\nDELETION AREA (lower is better)\n",
      au.pivot_table(index="layer", columns="lens", values="auc")
      .round(3).to_string())

best = pd.DataFrame({
    "points": pt.groupby("lens").point.max(),
    "top5": pt.groupby("lens").point5.max(),
    "energy": pt.groupby("lens").energy.max(),
    "predicts": co.groupby("lens").spearman.max(),
    "deletion": au.groupby("lens").auc.min(),
})
print("\nBEST OVER DEPTH, PER LENS\n", best.round(3).to_string())
json.dump({"repo": a.repo, "layers": LAYERS, "n": int(df.i.nunique()),
           "chance": float(pt.chance.mean()),
           "best": st.jsonable(best.round(4).to_dict("index")),
           "pointing": st.jsonable(pt.round(4).to_dict("index")),
           "predicts": st.jsonable(co.round(4).to_dict("index")),
           "deletion": st.jsonable(au.round(4).to_dict("index")),
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
print("done", flush=True)
