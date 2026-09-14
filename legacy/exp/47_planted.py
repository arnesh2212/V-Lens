"""Can the pair tell a planted concept from a real one?

A monitor that reads a logit lens off image tokens can be lied to: write a
concept's direction into those tokens with a rank-one edit and the lens reports
it.  That is the attack that matters, because it makes a monitor report
something the model is not using.

The question here is whether the second head is fooled too.  Four conditions,
ground truth known by construction:

    present   the object is in the photograph
    absent    it is not, and nothing suggests it
    told      it is not, but the text says it is
    planted   it is not, and its output-embedding direction has been written
              into the image tokens

Both heads are scored for the queried object and standardised across the 65
categories so the two are comparable.  A detector is only interesting if it
beats each head alone, so `say`, `do` and the gap are all reported, and so is
the model's own answer as the baseline any of them has to beat.
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
from jvlm import cocodata, corpus, readout, stats as st, vmap

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--n", type=int, default=170)
ap.add_argument("--mean-images", type=int, default=120)
ap.add_argument("--layer", type=int, default=None)
ap.add_argument("--plant-alpha", type=float, default=0.3)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/planted")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
LAYER = a.layer if a.layer is not None else int(round((m.n_layers - 1) * 0.5))
CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
FLAT = sorted({t for v in CAT_IDS.values() for t in v})
POSI = {t: i for i, t in enumerate(FLAT)}
CIDX = {c: i for i, c in enumerate(CATS)}
W = m._lm_head.weight.detach().float()
print(m, f"\nlayer {LAYER} of {m.n_layers}   {len(CATS)} categories", flush=True)

PLAIN = "Describe this image."


def prep(path, q=PLAIN):
    return m.chat(q, [corpus.load_image(path, square=a.square)])


pool = cocodata.concept_items(6000, min_area=0.04, seed=7)
b0 = prep(pool[0]["path"])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
acc = torch.zeros(P, m.d_model, dtype=torch.float64)
bank = []
nm = 0
for it in pool[: a.mean_images]:
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    v = readout.residuals(m, b, [LAYER])[LAYER][ip].float().cpu()
    acc += v.double()
    if len(bank) < 16:
        bank.append(v)
    nm += 1
MEAN = (acc / max(nm, 1)).float()
print(f"means from {nm} photographs ({time.time()-t0:.0f}s)", flush=True)


def heads(batch, ip, k):
    """say, do and the answer-position reading, standardised over categories."""
    A = readout.residuals(m, batch, [LAYER, m.n_layers - 1])
    h = A[LAYER][ip].float().cpu()
    lg = m.unembed(h.to(m.device)).float().cpu()[:, FLAT]
    lg = lg - lg.mean(1, keepdim=True)
    fin = m.unembed(A[m.n_layers - 1][-1:].to(m.device)).float().cpu()[:, FLAT]
    fin = fin - fin.mean(1, keepdim=True)
    DO = vmap.do_map(m, batch, FLAT, [LAYER], ip.tolist(),
                     means={LAYER: bank[k % len(bank)]})[LAYER]
    out = {}
    for name, mat in (("say", lg.max(0).values), ("do", DO.max(1).values),
                      ("answer", fin[0])):
        v = torch.tensor([max(float(mat[POSI[t]]) for t in CAT_IDS[c])
                          for c in CATS])
        out[name] = ((v - v.mean()) / v.std().clamp_min(1e-6)).numpy()
    return out


items = cocodata.concept_items(a.n, min_area=0.08, seed=31337,
                               tokenizer=m.tokenizer)
rows = []
for k, it in enumerate(items):
    c, d = it["concept"], it["distractor"]
    if c not in CIDX or d not in CIDX:
        continue
    im = corpus.load_image(it["path"], square=a.square)
    b = m.chat(PLAIN, [im])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue

    def rec(cond, obj, H):
        i_ = CIDX[obj]
        return dict(i=k, cond=cond, concept=obj,
                    say=float(H["say"][i_]), do=float(H["do"][i_]),
                    answer=float(H["answer"][i_]),
                    gap=float(H["say"][i_] - H["do"][i_]),
                    seen=float(H["say"][i_] - H["answer"][i_]))

    H = heads(b, ip, k)
    rows.append(rec("present", c, H))
    rows.append(rec("absent", d, H))

    bt = m.chat(f"The scene also contains a {d}. {PLAIN}", [im])
    ipt = readout.image_positions(m, bt)
    if len(ipt) == P:
        rows.append(rec("told", d, heads(bt, ipt, k)))

    u = W[CAT_IDS[d]].mean(0, keepdim=True)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    iv = readout.Intervention(m, {LAYER: u.to(m.device).to(m.dtype)},
                              channel="img", mode="add", alpha=a.plant_alpha)
    with iv:
        bp = iv.set_batch(m.chat(PLAIN, [im]))
        ipp = readout.image_positions(m, bp)
        if len(ipp) == P:
            rows.append(rec("planted", d, heads(bp, ipp, k)))
    if k % 20 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")


def auc(pos, neg):
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    v = np.concatenate([pos, neg]); o = v.argsort()
    r = np.empty(len(v), float); r[o] = np.arange(len(v))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2)
                 / (len(pos) * len(neg)))


FEATS = ["say", "do", "answer", "gap", "seen"]
agg = df.groupby("cond")[FEATS].mean()
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", agg.round(3).to_string())
P_ = df[df.cond == "present"]
out = {"repo": a.repo, "layer": LAYER, "n": int(df.i.nunique()),
       "plant_alpha": a.plant_alpha,
       "means": st.jsonable(agg.round(4).to_dict("index"))}
print(f"\nseparating a real object from each of the others (AUROC)\n"
      f"{'':<10}" + "".join(f"{f:>9}" for f in FEATS))
for other in ("planted", "told", "absent"):
    O = df[df.cond == other]
    if not len(O):
        continue
    d_ = {f: round(auc(O[f].values, P_[f].values), 4) for f in FEATS}
    out[f"{other}_vs_present"] = d_
    print(f"{other:<10}" + "".join(f"{d_[f]:>9.3f}" for f in FEATS))
json.dump(out, open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
print("\ndone", flush=True)
