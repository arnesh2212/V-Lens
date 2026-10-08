"""The other diagonal: a lesion that takes the choice instead of the name.

Erasing a concept's J-lens direction costs verbal report and spares the forced
choice.  On its own that is a single dissociation, and a single dissociation is
compatible with the boring story that the letter channel is simply more robust
to any perturbation.

So we build the opposite lesion.  For each concept we take the difference of
means between the image-token residuals of photographs that contain it and those
that do not -- a direction estimated purely from what the pictures do to the
model, with no reference to any word -- and erase that instead, at matched rank
and at the same positions.  If the letter channel now falls while the word
channel is comparatively spared, the two capacities are separately targetable and
the dissociation is double.

We also report the component of the perceptual direction that lies inside the
concept's J-space, which is the quantity that says how much of what the picture
writes is verbalizable in the first place.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
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

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--concepts", type=int, default=12)
ap.add_argument("--per-concept", type=int, default=40, help="images per side")
ap.add_argument("--test-per-concept", type=int, default=8)
ap.add_argument("--rank", type=int, default=4, help="perceptual subspace rank")
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/perceptual")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lens = (jvlm.lens_from_checkpoint(a.lens) if a.lens.endswith(".ckpt")
        else __import__("jlens").lens.JacobianLens.load(a.lens))
sc = tasks.Scorer(m)
L = list(lens.source_layers)
print(m, lens, flush=True)

rng = random.Random(13)
pool = cocodata.concept_items(8000, min_area=0.05, seed=3)
by_c: dict[str, list] = {}
for it in pool:
    by_c.setdefault(it["concept"], []).append(it)
concepts = sorted(by_c, key=lambda c: -len(by_c[c]))[: a.concepts]
print("concepts:", concepts, flush=True)

all_ids = sorted({it["image_id"] for it in pool})


def image_mean(image_ids, layers):
    """Mean residual over image-token positions, per layer, over a set of images."""
    acc = {l: torch.zeros(m.d_model, device=m.device) for l in layers}
    n = 0
    for iid in image_ids:
        im = corpus.load_image(str(cocodata.image_path(iid)))
        b = m.chat("Describe this image.", [im])
        ipos = readout.image_positions(m, b)
        if len(ipos) == 0:
            continue
        acts = readout.residuals(m, b, layers)
        for l in layers:
            acc[l] += acts[l][ipos].mean(0)
        n += 1
    return {l: acc[l] / max(n, 1) for l in layers}, n


# Orthonormal bases: one per concept (its own surface forms) and one for the
# whole single-word category vocabulary, which is the workspace a photograph of
# a COCO scene could plausibly occupy.
VOCAB_IDS = [cocodata.primary_token_id(m.tokenizer, x)
             for x in cocodata.SINGLE_WORD]
QVOCAB, QCONCEPT = {}, {}
for l in L:
    Vv = readout.lens_vectors(m, lens, VOCAB_IDS, l)
    QVOCAB[l], _ = torch.linalg.qr(Vv.T.float())
for c in concepts:
    ids = cocodata.concept_token_ids(m.tokenizer, c)
    QCONCEPT[c] = {}
    for l in L:
        Vc = readout.lens_vectors(m, lens, ids, l)
        QCONCEPT[c][l], _ = torch.linalg.qr(Vc.T.float())

# ------------------------------------------------- perceptual directions
t0 = time.time()
percept: dict[str, dict[int, torch.Tensor]] = {}
geom = []
for ci, c in enumerate(concepts):
    pos_ids = [it["image_id"] for it in by_c[c]][: a.per_concept]
    have = set(pos_ids)
    neg_ids = [i for i in all_ids
               if c not in cocodata.present(i) and i not in have][: a.per_concept]
    mu_p, np_ = image_mean(pos_ids, L)
    mu_n, nn_ = image_mean(neg_ids, L)

    spec = {}
    for l in L:
        d = (mu_p[l] - mu_n[l]).float()
        d = d / d.norm().clamp_min(1e-8)
        # widen to a rank-k subspace with orthogonal random directions so the
        # erasure is rank-matched to the concept's surface-form span
        extra = torch.randn(a.rank - 1, m.d_model,
                            generator=torch.Generator().manual_seed(ci * 97 + l))
        extra = extra.to(d.device)
        extra = extra - (extra @ d).unsqueeze(-1) * d
        extra = extra / extra.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        spec[l] = torch.cat([d[None], extra], 0)

        # --- the verbalizable fraction -------------------------------------
        # How much of what the picture writes about c lies inside the workspace.
        # Reported against two nulls: a random direction of the same dimension
        # (which lands k/d of its energy in any k-subspace by symmetry), and the
        # same projector applied to a random direction, measured rather than
        # assumed.
        Qc = QCONCEPT[c][l]                      # this concept's own J-span
        Qv = QVOCAB[l]                           # the whole concept vocabulary
        rnd = torch.randn(m.d_model, generator=torch.Generator().manual_seed(l))
        rnd = (rnd / rnd.norm()).to(d.device)
        geom.append(dict(
            concept=c, layer=l, depth=l / (m.n_layers - 1),
            in_jspace=float(((d @ Qc) ** 2).sum()),
            in_vocab_jspace=float(((d @ Qv) ** 2).sum()),
            null_concept=float(((rnd @ Qc) ** 2).sum()),
            null_vocab=float(((rnd @ Qv) ** 2).sum()),
            dim_concept=int(Qc.shape[1]), dim_vocab=int(Qv.shape[1]),
            d_model=int(m.d_model)))
    percept[c] = spec
    print(f"  {c}: {np_}+{nn_} images  {time.time()-t0:.0f}s", flush=True)

g = pd.DataFrame(geom)
g.to_parquet(f"{a.out}/{tag}_geometry.parquet")
print("\nverbalizable fraction of the perceptual direction, by depth "
      "(value vs its random-direction null):")
print(g.assign(b=(g.depth * 4).astype(int) / 4).groupby("b")[
    ["in_jspace", "null_concept", "in_vocab_jspace", "null_vocab"]]
    .mean().round(4).to_string(), flush=True)


def jspec(concept):
    ids = cocodata.concept_token_ids(m.tokenizer, concept)
    return {l: readout.lens_vectors(m, lens, ids, l) for l in L}


def rspec(seed):
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for l in L:
        V = torch.randn(a.rank, m.d_model, generator=gen)
        out[l] = (V / V.norm(dim=-1, keepdim=True)).to(m.device)
    return out


# ------------------------------------------------------------ the test
ARMS = [("clean", None), ("workspace", "j"), ("perceptual", "p"), ("random", "r")]
rows = []
for ci, c in enumerate(concepts):
    tests = by_c[c][a.per_concept : a.per_concept + a.test_per_concept]
    jd = jspec(c)
    for k, it in enumerate(tests):
        d = cocodata.distractor(it["image_id"], c, rng, hard=True)
        im = corpus.load_image(it["path"])
        for arm, which in ARMS:
            iv = (readout.NoIntervention() if which is None else
                  readout.Intervention(
                      m, {"j": jd, "p": percept[c], "r": rspec(ci * 31 + k)}[which],
                      channel="all", mode="ablate", alpha=1.0))

            def multi(prompts, image=im, _iv=iv):
                return sc.multi_last_logits(
                    _iv.set_batch(m.chat_batch(prompts, [image])))

            with iv:
                r = tasks.run_two_channels(sc, c, d, multi)
            rows.append(dict(concept=c, image_id=it["image_id"], distractor=d,
                             i=f"{c}_{k}", arm=arm, cond=arm,
                             energy=float(np.mean(iv.energy))
                             if getattr(iv, "energy", None) else 0.0, **r))
    print(f"  tested {c}  {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
agg = df.groupby("arm").agg(
    n=("i", "count"),
    word_correct=("word_correct", "mean"), word_margin=("word_margin", "mean"),
    letter_correct=("letter_correct", "mean"), letter_margin=("letter_margin", "mean"),
    energy=("energy", "mean")).reindex([x[0] for x in ARMS])
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", agg.round(3).to_string())

from scipy.stats import spearmanr  # noqa: E402

from jvlm import stats as st  # noqa: E402

out = {"repo": a.repo, "concepts": concepts,
       "V_concept_mean": float(g.in_jspace.mean()),
       "V_concept_null": float(g.null_concept.mean()),
       "V_vocab_mean": float(g.in_vocab_jspace.mean()),
       "V_vocab_null": float(g.null_vocab.mean()),
       "minutes": round((time.time() - t0) / 60, 1)}
for arm in ("workspace", "perceptual", "random"):
    out[arm] = st.interaction(df, "clean", arm, "word_margin", "letter_margin")
    print(f"{arm}: word-minus-letter interaction dd = {out[arm].get('dd'):.2f}, "
          f"p = {out[arm].get('p'):.2g}")

# ---- does geometry predict behaviour?
# The theory says erasing the workspace removes only the verbalizable part of
# the perceptual evidence, so the decision should suffer in proportion to how
# much of that evidence was verbalizable -- while the naming channel should
# collapse regardless.  V is computed from weights and image statistics with no
# behavioural data at all, so a correlation across concepts is a real
# prediction rather than a fit.
band = g[(g.depth >= 0.3) & (g.depth <= 0.9)]
Vc = band.groupby("concept")[["in_jspace", "in_vocab_jspace"]].mean()
delta = (df[df.arm == "clean"].groupby("concept")[["word_margin", "letter_margin"]].mean()
         - df[df.arm == "workspace"].groupby("concept")[["word_margin", "letter_margin"]].mean())
j = Vc.join(delta, how="inner").dropna()
j.to_csv(f"{a.out}/{tag}_prediction.csv")
if len(j) >= 5:
    for key in ("in_jspace", "in_vocab_jspace"):
        for tgt in ("letter_margin", "word_margin"):
            rho, pv = spearmanr(j[key], j[tgt])
            out[f"rho_{key}_{tgt}"] = float(rho)
            out[f"p_{key}_{tgt}"] = float(pv)
            print(f"  V({key}) vs drop in {tgt}: rho = {rho:+.3f}  p = {pv:.3g}"
                  f"  (n = {len(j)} concepts)")
json.dump(out, open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
