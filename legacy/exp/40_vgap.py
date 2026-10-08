"""V-Lens reads two things at once, and the gap between them is the point.

A lens that reads well and a lens that predicts well are different maps.  So
V-Lens does not choose.  At every image site it reports a pair:

    say   what the model is disposed to put into words about this patch
    do    what this patch is actually doing to the answer

Both are linear readouts of the same activation, both come out of one fit, and
the shrinkage parameter is what moves between them.  Reported together they give
something no single-headed lens can:

    the V-gap   say  minus  do,  standardised across the vocabulary

A concept the model can see and is using has a small gap.  A concept it will
happily say but is not using has a large one -- and that is the signature of
every failure a monitor cares about.  This measures it in three conditions where
the ground truth is known by construction:

    present    the object is in the photograph
    absent     it is not, and nothing suggests it
    told       it is not, and the text says it is        (text-sourced)
    planted    it is not, and its legible direction has been written into the
               image sites                                (the forgery, analytically)

If the gap separates the first from the last two, V-Lens can tell what the model
saw from what it was told and from what was planted -- which is exactly the
question the model itself answers at chance.
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

import jvlm
from jvlm import cocodata, corpus, readout, tasks
from jvlm.patchfit import SecantLens, gauge_fix

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--vlens", required=True)
ap.add_argument("--layer", type=int, default=None)
ap.add_argument("--n", type=int, default=160)
ap.add_argument("--plant-alpha", type=float, default=0.3)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/vgap")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
V = SecantLens.load(a.vlens)
LAYERS = sorted(V.maps)
LAYER = a.layer if a.layer is not None else LAYERS[0]
sc = tasks.Scorer(m)
print(m, f"\nV-Lens at layer {LAYER} of {LAYERS}", flush=True)

CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: [t for t in cocodata.concept_token_ids(m.tokenizer, c)
               if t in V._pos] for c in CATS}
CATS = [c for c in CATS if CAT_IDS[c]]
FLAT = V.token_ids
POSI = {t: i for i, t in enumerate(FLAT)}
W = gauge_fix(m._lm_head.weight.detach().float().cpu())
print(f"{len(CATS)} concepts scorable by both heads", flush=True)

PLAIN = "Describe this image."


def heads(batch, ip):
    """Three readouts, each standardised across the vocabulary of concepts.

    ``say`` and ``do`` are read at the image tokens.  ``answer`` is the same
    plain readout at the last prompt position, where the picture and the text
    have both been taken in -- and it is the third reading that makes provenance
    recoverable.  A concept that arrived through the text cannot appear at an
    image site at all, because the image tokens precede it and attention only
    looks left.  So a concept legible at the answer and absent from the image
    sites is one the model was told about.
    """
    A = readout.residuals(m, batch, [LAYER, m.n_layers - 1])
    h = A[LAYER][ip].float().cpu()
    fin = A[m.n_layers - 1][-1:].float()
    scores = {"say": m.unembed(h.to(m.device)).float().cpu()[:, FLAT],
              "do": V.read(h, LAYER),
              "answer": m.unembed(fin.to(m.device)).float().cpu()[:, FLAT]}
    out = {}
    for name, sc_ in scores.items():
        best = sc_.max(0).values
        v = torch.tensor([max(float(best[POSI[t]]) for t in CAT_IDS[c])
                          for c in CATS])
        out[name] = ((v - v.mean()) / v.std().clamp_min(1e-6)).numpy()
    return out


items = cocodata.concept_items(a.n, min_area=0.08, seed=31337,
                               tokenizer=m.tokenizer)
CIDX = {c: i for i, c in enumerate(CATS)}
rows = []
t0 = time.time()
for k, it in enumerate(items):
    c, d = it["concept"], it["distractor"]
    if c not in CIDX or d not in CIDX:
        continue
    im = corpus.load_image(it["path"], square=a.square)
    here = set(it["present"])

    # --- present, and absent, from one clean forward
    b = m.chat(PLAIN, [im])
    ip = readout.image_positions(m, b)
    H = heads(b, ip)
    for cond, cc in (("present", c), ("absent", d)):
        i_ = CIDX[cc]
        rows.append(dict(i=k, cond=cond, concept=cc,
                         say=float(H["say"][i_]), do=float(H["do"][i_]),
                         answer=float(H["answer"][i_]),
                         gap=float(H["say"][i_] - H["do"][i_]),
                         seen=float(H["say"][i_] - H["answer"][i_])))

    # --- told: the same photograph, with a sentence asserting the absent object
    bt = m.chat(f"The scene also contains a {d}. {PLAIN}", [im])
    ipt = readout.image_positions(m, bt)
    Ht = heads(bt, ipt)
    i_ = CIDX[d]
    rows.append(dict(i=k, cond="told", concept=d,
                     say=float(Ht["say"][i_]), do=float(Ht["do"][i_]),
                     answer=float(Ht["answer"][i_]),
                     gap=float(Ht["say"][i_] - Ht["do"][i_]),
                     seen=float(Ht["say"][i_] - Ht["answer"][i_])))

    # --- planted: the absent object's legible direction written into the sites
    u = W[CAT_IDS[d]].mean(0, keepdim=True)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    iv = readout.Intervention(m, {LAYER: u.to(m.device).to(m.dtype)},
                              channel="img", mode="add", alpha=a.plant_alpha)
    with iv:
        bp = iv.set_batch(m.chat(PLAIN, [im]))
        ipp = readout.image_positions(m, bp)
        Hp = heads(bp, ipp)
    rows.append(dict(i=k, cond="planted", concept=d,
                     say=float(Hp["say"][i_]), do=float(Hp["do"][i_]),
                     answer=float(Hp["answer"][i_]),
                     gap=float(Hp["say"][i_] - Hp["do"][i_]),
                     seen=float(Hp["say"][i_] - Hp["answer"][i_])))

    if k % 20 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg]); o = allv.argsort()
    r = np.empty(len(allv), float); r[o] = np.arange(len(allv))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2)
                 / (len(pos) * len(neg)))


FEATS = ["say", "do", "answer", "gap", "seen"]
agg = df.groupby("cond")[FEATS].mean()
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", agg.round(3).to_string())

P = df[df.cond == "present"]
out = {"repo": a.repo, "layer": LAYER, "n": int(df.i.nunique()),
       "plant_alpha": a.plant_alpha,
       "summary": {str(k): {kk: float(vv) for kk, vv in v.items()}
                   for k, v in agg.round(4).to_dict("index").items()}}
print("\nseparating a genuinely seen object from each of the others:")
for other in ("told", "planted", "absent"):
    O = df[df.cond == other]
    d_ = {}
    for feat in FEATS:
        # the seen object should score lower on the gap than a planted one
        d_[feat] = round(auc(-P[feat].values, -O[feat].values), 4)
    out[f"present_vs_{other}"] = d_
    print(f"  vs {other:<8} " + "  ".join(f"{f} {d_[f]:.3f}" for f in FEATS))
json.dump(out, open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
print("\nA readout alone cannot tell these apart. The pair can, or it cannot, "
      "and the middle column is the control that says which.", flush=True)
