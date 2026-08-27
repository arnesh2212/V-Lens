"""Does a lens point at the object, and does it predict what covering it does?

This replaces a fitted map that failed, and it is built around four corrections
that the failure and a hostile review made obvious.

*The unit is the object, not the token.*  A single image token barely matters --
image tokens are famously prunable after early layers -- so leave-one-token-out
measures mostly nothing.  Covering the tokens an object occupies breaks the
redundancy and produces an effect large enough to be predicted.

*The prediction must match the intervention.*  If the counterfactual replaces a
token with a donor photograph's, then the first-order prediction of *that*
intervention is ``g . (h - h_donor)``, not ``g . (h - mean)``.  The earlier
version mismatched these, which makes any failure uninterpretable.

*The metric is a logit difference.*  A raw logit lives partly in a direction the
softmax cannot see, and the category token closest to the mean unembedding
direction wins everywhere -- which is exactly how the previous fit came to read
"Knife" off every photograph.  Scores are therefore differences from the mean
over the category set.

*Beating chance is not the bar.*  Raw attention, attention rollout, the logit
lens itself, and a size-matched random box all run alongside, because the claim
is comparative or it is nothing.
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
ap.add_argument("--n", type=int, default=180)
ap.add_argument("--mean-images", type=int, default=120)
ap.add_argument("--n-layers", type=int, default=6)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-frac", type=float, default=0.45)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/vmap")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lo, hi = max(m.n_deepstack, 1), m.n_layers - 2
LAYERS = sorted({int(round(lo + (hi - lo) * f))
                 for f in np.linspace(0.05, 0.95, a.n_layers)})
CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
FLAT = sorted({t for v in CAT_IDS.values() for t in v})
POSI = {t: i for i, t in enumerate(FLAT)}
print(m, f"\nlayers {LAYERS}   {len(CATS)} categories / {len(FLAT)} tokens",
      flush=True)

PROMPT = "Describe this image."


def prep(path):
    return m.chat(PROMPT, [corpus.load_image(path, square=a.square)])


def cat_margin(logits: torch.Tensor, ids) -> float:
    """A logit difference: this word against the average category word.

    A raw logit is not comparable across tokens -- part of it is a shift the
    softmax cannot see, and the token nearest the mean unembedding direction
    scores highest everywhere regardless of the picture.
    """
    v = logits[FLAT].float()
    return float(max(v[POSI[t]] for t in ids) - v.mean())


pool = cocodata.concept_items(6000, min_area=0.04, seed=7)
b0 = prep(pool[0]["path"])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
assert G * G == P, f"{P} image tokens is not a square grid"

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
        if len(bank[l]) < 24:
            bank[l].append(v)
    nm += 1
MEANS = {l: (acc[l] / max(nm, 1)).float() for l in LAYERS}
NORM = {l: (nrm[l] / max(nm, 1)).float() for l in LAYERS}
# high-norm "register" tokens carry global information, not local content, and
# dominate any map that does not know about them
REG = {l: set((NORM[l] > NORM[l].mean() + 3 * NORM[l].std()).nonzero()
              .flatten().tolist()) for l in LAYERS}
print(f"means from {nm} photographs; register-like tokens per layer: "
      f"{[len(REG[l]) for l in LAYERS]}  ({time.time()-t0:.0f}s)", flush=True)

items = cocodata.concept_items(a.n, min_area=0.06, seed=515,
                               tokenizer=m.tokenizer)
rows, pred = [], []
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
    A = readout.residuals(m, b, LAYERS + [m.n_layers - 1])
    base = cat_margin(m.unembed(A[m.n_layers - 1][-1:].to(m.device))[0], ids)

    # a size- and shape-matched box somewhere else in the same picture
    rng = np.random.default_rng(1000 + k)
    out_pool = [t for t in range(P) if t not in set(inside)]
    ctrl = sorted(rng.choice(out_pool, size=min(nb, len(out_pool)),
                             replace=False).tolist())
    donor = {l: bank[l][k % len(bank[l])] for l in LAYERS}

    for l in LAYERS:
        h = A[l][ip].float().cpu()
        dev_mu = h - MEANS[l]
        dev_dn = h - donor[l]
        # the two maps, plus the baseline every vision lens is measured against
        lg = m.unembed(h.to(m.device)).float().cpu()
        say = (lg[:, FLAT] - lg[:, FLAT].mean(1, keepdim=True))
        say = say[:, [POSI[t] for t in ids]].max(1).values
        do_mu = vmap.do_map(m, b, ids, [l], ip.tolist(),
                            means={l: MEANS[l]})[l].sum(0)
        do_dn = vmap.do_map(m, b, ids, [l], ip.tolist(),
                            means={l: donor[l]})[l].sum(0)
        maps = {"logit lens": say, "V-Lens (mean baseline)": do_mu,
                "V-Lens (donor baseline)": do_dn}
        for name, heat in maps.items():
            for masked in (False, True):
                hm = heat.clone()
                if masked:
                    for t in REG[l]:
                        hm[t] = hm.min() - 1
                r = vmap.pointing(hm, set(inside), G)
                rows.append(dict(i=k, concept=c, lens=name, layer=l,
                                 depth=round(l / (m.n_layers - 1), 3),
                                 masked=int(masked), n_tok=nb, **r))

        # what covering those tokens really does
        for arm, tk in (("object", inside), ("control", ctrl)):
            pos = ip[torch.tensor(tk)].to(m.device)
            vals = donor[l][torch.tensor(tk)].to(m.device)

            def hook(mod, inp, out, _p=pos, _v=vals):
                tup = not torch.is_tensor(out)
                h_ = (out[0] if tup else out).clone()
                h_[0, _p] = _v.to(h_.dtype)
                return (h_, *out[1:]) if tup else h_

            hd = m.layers[l].register_forward_hook(hook)
            try:
                with torch.no_grad():
                    AA = readout.residuals(m, b, [m.n_layers - 1])
                    after = cat_margin(
                        m.unembed(AA[m.n_layers - 1][-1:].to(m.device))[0], ids)
            finally:
                hd.remove()
            pred.append(dict(i=k, concept=c, layer=l, arm=arm, n_tok=len(tk),
                             drop=base - after,
                             p_do=float(do_dn[tk].sum()),
                             p_say=float(say[tk].sum())))
    seen += 1
    if k % 15 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  kept {seen}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_point.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_point.parquet")
pt = pd.DataFrame(pred)
pt.to_parquet(f"{a.out}/{tag}_patch.parquet")

agg = df[df.masked == 1].groupby(["lens", "layer"]).agg(
    n=("i", "count"), point=("hit", "mean"), point5=("hit5", "mean"),
    energy=("mass_in", "mean"), chance=("chance", "mean")).reset_index()
agg["lift"] = agg["point"] / agg["chance"]
agg.to_csv(f"{a.out}/{tag}_summary.csv", index=False)
print("\nPOINTING GAME (register tokens masked)\n",
      agg.round(3).to_string(index=False))

out = {"repo": a.repo, "layers": LAYERS, "grid": G, "n": int(df.i.nunique()),
       "n_layers": m.n_layers,
       "pointing": st.jsonable(agg.round(4).to_dict("index")),
       "minutes": round((time.time() - t0) / 60, 1)}

o = pt[pt.arm == "object"]
cc = pt.groupby(["layer", "arm"])["drop"].mean().unstack().round(3)
print("\nWHAT COVERING THE TOKENS REALLY DOES (logit-difference drop)\n",
      cc.to_string())
cor = {}
for l in sorted(o.layer.unique()):
    d = o[o.layer == l]
    if len(d) > 10:
        cor[str(l)] = {
            "do_spearman": round(float(sps.spearmanr(d.p_do, d["drop"]).statistic), 4),
            "say_spearman": round(float(sps.spearmanr(d.p_say, d["drop"]).statistic), 4),
            "do_partial_size": round(float(sps.spearmanr(
                d.p_do / d.n_tok, d["drop"] / d.n_tok).statistic), 4),
            "n": int(len(d))}
out["patch_drop"] = st.jsonable(cc.to_dict())
out["predicts_drop"] = cor
print("\nDOES THE MAP PREDICT THE DROP? (Spearman over photographs)")
for l, v in cor.items():
    print(f"  layer {l:<4} V-Lens {v['do_spearman']:+.3f}   "
          f"logit lens {v['say_spearman']:+.3f}   "
          f"per-token {v['do_partial_size']:+.3f}   (n={v['n']})")
json.dump(out, open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
print("done", flush=True)
