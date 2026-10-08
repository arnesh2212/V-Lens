"""Does the gap between the two maps predict a hallucination?

POPE asks a vision-language model, one object at a time, "is there a {object} in
the image?"  Half the questions name something that is there and half something
that is not, and the interesting half is the second: a model that says yes to an
absent object has hallucinated it.  The negatives come in three flavours, which
is what makes the benchmark hard -- drawn at random, drawn from the objects that
are commonest overall, and drawn from the objects that most often keep company
with what *is* in the picture.  We rebuild it from the COCO annotations rather
than downloading it, so the construction is visible and the split is ours.

The question here is not whether the model hallucinates.  It is whether V-Lens
sees it coming.  For each queried object we compute both maps over the image and
ask four predictors to separate the hallucinations from the correct refusals:

    say        the logit lens's score for that object over the image tokens
    do         how much those tokens are pushing the answer toward it
    gap        say minus do, standardised
We deliberately do not report the model's own yes/no margin as a baseline: a
hallucination is *defined* here as the model answering yes, so that margin
separates the classes by construction and would score a meaningless 1.000. The
comparison that means something is between things read off the image tokens
before the answer is formed.

If the gap adds nothing over say alone and do alone, it is a story and not a
tool, so the comparison is the point.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter

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
ap.add_argument("--images", type=int, default=220)
ap.add_argument("--mean-images", type=int, default=120)
ap.add_argument("--per-image", type=int, default=3)
ap.add_argument("--n-layers", type=int, default=4)
ap.add_argument("--ig-steps", type=int, default=4)
ap.add_argument("--jlens", default=None)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/pope")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lo, hi = max(m.n_deepstack, 1), m.n_layers - 2
LAYERS = sorted({int(round(lo + (hi - lo) * f))
                 for f in np.linspace(0.2, 0.85, a.n_layers)})
CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
FLAT = sorted({t for v in CAT_IDS.values() for t in v})
POSI = {t: i for i, t in enumerate(FLAT)}


def ids_of(w):
    out = []
    for f in (w, " " + w, w.capitalize(), " " + w.capitalize()):
        t = m.tokenizer.encode(f, add_special_tokens=False)
        if len(t) == 1:
            out.append(t[0])
    return sorted(set(out))


YES, NO = ids_of("yes"), ids_of("no")
print(m, f"\nlayers {LAYERS}   yes={len(YES)} no={len(NO)}", flush=True)


def prep(path, q):
    return m.chat(q, [corpus.load_image(path, square=a.square)])


pool = cocodata.concept_items(6000, min_area=0.04, seed=7)
b0 = prep(pool[0]["path"], "Describe this image.")
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
bank = {l: [] for l in LAYERS}
nm = 0
for it in pool[: a.mean_images]:
    b = prep(it["path"], "Describe this image.")
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
J = None
if a.jlens:
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))
    print(f"J-lens loaded: layers {min(J.jacobians)}..{max(J.jacobians)}",
          flush=True)
print(f"means from {nm} photographs ({time.time()-t0:.0f}s)", flush=True)

# ---- build the three POPE splits from the annotations ------------------------
freq = Counter()
for it in pool:
    freq.update(c for c in it["present"] if c in CATS)
POPULAR = [c for c, _ in freq.most_common()]
rng = np.random.default_rng(4242)
imgs = cocodata.concept_items(a.images, min_area=0.05, seed=31,
                              tokenizer=m.tokenizer)
qs = []
for it in imgs:
    here = sorted(c for c in it["present"] if c in CATS)
    absent = [c for c in CATS if c not in set(it["present"])]
    if not here or len(absent) < 6:
        continue
    coocc = Counter()
    for h in here:
        coocc.update(cocodata.cooccurring(h) if hasattr(cocodata, "cooccurring")
                     else {})
    adv = [c for c, _ in coocc.most_common() if c in set(absent)]
    for _ in range(a.per_image):
        qs.append((it, rng.choice(here), "present", "yes"))
    qs.append((it, rng.choice(absent), "random", "no"))
    pop = next((c for c in POPULAR if c in set(absent)), None)
    if pop:
        qs.append((it, pop, "popular", "no"))
    if adv:
        qs.append((it, adv[0], "adversarial", "no"))
print(f"{len(qs)} questions over {len(imgs)} photographs", flush=True)

rows = []
for k, (it, obj, split, truth) in enumerate(qs):
    q = f"Is there a {obj} in this image?  Answer yes or no."
    b = prep(it["path"], q)
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    ids = CAT_IDS[obj]
    A = readout.residuals(m, b, LAYERS + [m.n_layers - 1])
    lg = m.unembed(A[m.n_layers - 1][-1:].to(m.device))[0].float()
    conf = float(lg[YES].max() - lg[NO].max())
    said = "yes" if conf > 0 else "no"
    donor = {l: bank[l][k % len(bank[l])] for l in LAYERS}
    rec = dict(i=k, image_id=int(it["image_id"]), obj=obj, split=split,
               truth=truth, said=said, correct=int(said == truth),
               hallucinated=int(truth == "no" and said == "yes"), conf=conf)
    for l in LAYERS:
        h = A[l][ip].float().cpu()
        v = m.unembed(h.to(m.device)).float().cpu()[:, FLAT]
        v = v - v.mean(1, keepdim=True)
        say = float(v[:, [POSI[t] for t in ids]].max(1).values.max())
        rec[f"say_{l}"] = say
        # the corpus mean is the baseline the method is defined against; the
        # single-donor version is kept beside it because an earlier run used it
        rec[f"do_{l}"] = float(vmap.do_map(m, b, ids, [l], ip.tolist(),
                                           means={l: MEANS[l]})[l].sum(0).max())
        rec[f"dodonor_{l}"] = float(vmap.do_map(m, b, ids, [l], ip.tolist(),
                                                means={l: donor[l]})[l]
                                    .sum(0).max())
        if a.ig_steps > 1:
            rec[f"ig_{l}"] = float(vmap.ig_map(m, b, ids, [l], ip.tolist(),
                                               means={l: MEANS[l]},
                                               steps=a.ig_steps)[l]
                                   .sum(0).max())
        if J is not None and l in J.jacobians:
            jl, _ = readout.lens_logits(m, J, b, layers=[l],
                                        positions=ip.tolist())
            jv = jl[l].float().cpu()[:, FLAT]
            jv = jv - jv.mean(1, keepdim=True)
            rec[f"jlens_{l}"] = float(
                jv[:, [POSI[t] for t in ids]].max(1).values.max())
    rows.append(rec)
    if k % 60 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(qs)}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(qs)-k-1)/60:.1f}m", flush=True)
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


acc_tab = df.groupby("split").agg(n=("i", "count"), accuracy=("correct", "mean"),
                                  yes_rate=("said", lambda s: (s == "yes").mean()))
print("\nPOPE accuracy\n", acc_tab.round(3).to_string())

# standardise each feature within split so scales are comparable
neg = df[df.truth == "no"].copy()
ARMS = ["say", "do", "dodonor", "ig", "jlens"]
best = {}
for l in LAYERS:
    have = [f for f in ARMS if f"{f}_{l}" in neg.columns]
    for f in have:
        col = f"{f}_{l}"
        neg[col + "_z"] = (neg[col] - neg[col].mean()) / neg[col].std().clip(1e-9)
    # the gap between what a place says and what it does: a hallucination is a
    # word the content reading supports and the causal reading does not
    causal = "ig" if f"ig_{l}" in neg.columns else "do"
    neg[f"gap_{l}"] = neg[f"say_{l}_z"] - neg[f"{causal}_{l}_z"]
    h = neg[neg.hallucinated == 1]
    c = neg[neg.hallucinated == 0]
    row = {f: round(auc(h[f"{f}_{l}"].values, c[f"{f}_{l}"].values), 4)
           for f in have}
    row["gap"] = round(auc(h[f"gap_{l}"].values, c[f"gap_{l}"].values), 4)
    # deliberately not reported as a baseline: a hallucination is *defined*
    # as the model answering yes, so its own yes/no margin separates the two
    # classes by construction and scores 1.000 for no good reason
    row["confidence_degenerate"] = round(
        auc(h["conf"].values, c["conf"].values), 4)
    row["n_hallucinated"] = int(len(h))
    row["n_correct"] = int(len(c))
    best[str(l)] = row

print("\nPREDICTING A HALLUCINATION before it is spoken (AUROC over the "
      "absent-object questions)")
cols = [c for c in ("say", "jlens", "do", "dodonor", "ig", "gap")
        if any(c in v for v in best.values())]
print("  layer" + "".join(f"{c:>10}" for c in cols))
for l, v in best.items():
    print(f"{l:>7}" + "".join(
        (f"{v[c]:>10.3f}" if c in v and v[c] == v[c] else f"{'-':>10}")
        for c in cols))
json.dump({"repo": a.repo, "layers": LAYERS, "n": int(len(df)),
           "accuracy": st.jsonable(acc_tab.round(4).to_dict("index")),
           "detect": best,
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
print("done", flush=True)
