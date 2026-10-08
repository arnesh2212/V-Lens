"""The deletion curve: does the order a map puts patches in actually matter?

The pointing game asks where a map's peak lands.  The deletion curve asks
something stricter and more useful: take the patches in the order the map ranks
them, cover them a few at a time, and watch how fast the model's score for the
object falls.  A map that knows which patches carry the object destroys it
quickly; a map that does not may as well be deleting at random.

This is the standard faithfulness measure in the saliency literature, and it is
the right one here because it does not care where the object *is* -- only whether
the map's ordering is the ordering that matters.  Reported as the area under the
curve, normalised so 1.0 is "no better than deleting in a random order".

Three maps are compared on identical photographs and identical deletion budgets:
the logit lens, the causal head, and a random order.
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
from jvlm import cocodata, corpus, readout, stats as st, vmap

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--mean-images", type=int, default=100)
ap.add_argument("--layers", type=float, nargs="*", default=[0.3, 0.5, 0.7])
ap.add_argument("--steps", type=int, nargs="*",
                default=[0, 4, 8, 16, 24, 36, 52, 72, 98])
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/deletion")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lo, hi = max(m.n_deepstack, 1), m.n_layers - 2
LAYERS = sorted({int(round(lo + (hi - lo) * f)) for f in a.layers})
CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
FLAT = sorted({t for v in CAT_IDS.values() for t in v})
POSI = {t: i for i, t in enumerate(FLAT)}
print(m, f"\nlayers {LAYERS}", flush=True)

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
bank = {l: [] for l in LAYERS}
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
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
        if len(bank[l]) < 16:
            bank[l].append(v)
    nm += 1
MEANS = {l: (acc[l] / max(nm, 1)).float() for l in LAYERS}
STEPS = [s for s in a.steps if s <= P]
print(f"means from {nm}; deleting up to {STEPS[-1]} of {P} tokens "
      f"({time.time()-t0:.0f}s)", flush=True)

items = cocodata.concept_items(a.n, min_area=0.08, seed=606,
                               tokenizer=m.tokenizer)
rows = []
for k, it in enumerate(items):
    c = it["concept"]
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    ids = CAT_IDS[c]
    A = readout.residuals(m, b, LAYERS + [m.n_layers - 1])
    base = margin(m.unembed(A[m.n_layers - 1][-1:].to(m.device))[0], ids)
    if base <= 0:
        continue                     # nothing to destroy
    rng = np.random.default_rng(500 + k)
    for l in LAYERS:
        h = A[l][ip].float().cpu()
        donor = bank[l][k % len(bank[l])]
        lg = m.unembed(h.to(m.device)).float().cpu()[:, FLAT]
        lg = lg - lg.mean(1, keepdim=True)
        say = lg[:, [POSI[t] for t in ids]].max(1).values
        do = vmap.do_map(m, b, ids, [l], ip.tolist(),
                         means={l: donor})[l].sum(0)
        orders = {"logit lens": say.argsort(descending=True).tolist(),
                  "V-Lens": do.argsort(descending=True).tolist(),
                  "random": rng.permutation(P).tolist()}
        for name, order in orders.items():
            for s in STEPS:
                if s == 0:
                    rows.append(dict(i=k, concept=c, layer=l, order=name,
                                     step=0, frac=0.0, margin=base,
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
                rows.append(dict(i=k, concept=c, layer=l, order=name, step=s,
                                 frac=s / P, margin=v, rel=v / base))
    if k % 10 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
curve = df.groupby(["layer", "order", "frac"])["rel"].mean().reset_index()
curve.to_csv(f"{a.out}/{tag}_curve.csv", index=False)
print("\nDELETION CURVE -- share of the object's score left after covering "
      "the top-ranked patches")
piv = curve.pivot_table(index=["layer", "frac"], columns="order", values="rel")
print(piv.round(3).to_string())

auc_ = {}
for l in LAYERS:
    d = curve[curve.layer == l]
    row = {}
    for o in d["order"].unique():
        s = d[d.order == o].sort_values("frac")
        row[o] = float(np.trapz(s["rel"], s["frac"]) / s["frac"].max())
    if "random" in row and row["random"]:
        row = {k_: round(v / row["random"], 4) for k_, v in row.items()}
    auc_[str(l)] = row
print("\nAREA UNDER THE DELETION CURVE, relative to deleting at random "
      "(lower is a better ordering)")
for l, r in auc_.items():
    print(f"  layer {l:<4}" + "  ".join(f"{k_} {v:.3f}" for k_, v in r.items()))
json.dump({"repo": a.repo, "layers": LAYERS, "n": int(df.i.nunique()),
           "steps": STEPS, "n_tokens": P, "auc_vs_random": auc_,
           "curve": st.jsonable(piv.round(4).to_dict("index")),
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
print("done", flush=True)
