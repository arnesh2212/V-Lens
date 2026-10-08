"""The inversion as an intervention, not a correlation.

If legibility and causal fidelity are different subspaces, then at one image site
there should be two directions that do opposite things:

    the legible one   the object name's own unembedding row -- what every lens
                      that reads well is reading
    the causal one    the secant's row for that token -- the direction the best
                      linear predictor of patching says matters

Push along each with matched norm, at the image tokens, and measure both sides:
what a monitor reads off those sites, and what the model actually answers.  The
claim predicts a crossing.  Pushing the legible direction should move the readout
and leave the answer; pushing the causal direction should move the answer and
leave the readout.

This is the experiment that decides whether the anti-correlation is a fact about
the model or an artefact of comparing two metrics.  A rank-matched random
direction runs in every condition, at the same norm, as the floor.
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
from jvlm import cocodata, corpus, readout, stats as st, tasks
from jvlm.patchfit import SecantLens, gauge_fix

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--secant", required=True)
ap.add_argument("--layer", type=int, default=None)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--alphas", type=float, nargs="*", default=[0.1, 0.25, 0.5])
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/double")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
S = SecantLens.load(a.secant)
LAYER = a.layer if a.layer is not None else sorted(S.maps)[0]
sc = tasks.Scorer(m)
W = gauge_fix(m._lm_head.weight.detach().float().cpu())
print(m, f"\nperturbing image sites at layer {LAYER}", flush=True)

CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
FLAT = sorted({t for v in CAT_IDS.values() for t in v})
POSI = {t: i for i, t in enumerate(FLAT)}

# every arm adds alpha times the *local residual norm* along a unit direction,
# so the arms are norm-matched by construction and alpha means one thing
SIG = float((S.means[LAYER] - S.means[LAYER].mean(0)).norm(dim=-1).mean())
print(f"alpha is a fraction of the local residual norm; a typical real "
      f"deviation at this layer has norm {SIG:.2f}", flush=True)

items = cocodata.concept_items(a.n, min_area=0.08, seed=2024, hard=True,
                               tokenizer=m.tokenizer)
rows = []
t0 = time.time()
for k, it in enumerate(items):
    c, d = it["concept"], it["distractor"]
    ids = [t for t in CAT_IDS[c] if t in S._pos]
    if not ids:
        continue
    im = corpus.load_image(it["path"], square=a.square)

    legible = W[ids].clone()                       # what a lens reads
    legible = legible / legible.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    causal = S.vectors(ids, LAYER)                 # what the secant predicts
    g = torch.Generator().manual_seed(9000 + k)
    rand = torch.randn(len(ids), m.d_model, generator=g)
    rand = rand / rand.norm(dim=-1, keepdim=True)
    DIRS = {"legible": legible, "causal": causal, "random": rand}

    arms = [("clean", None, 0.0)]
    for nm in DIRS:
        for al in a.alphas:
            for sgn in (1.0, -1.0):
                arms.append((nm, nm, sgn * al))

    for arm, nm, al in arms:
        if nm is None:
            iv = readout.NoIntervention()
        else:
            v = DIRS[nm].mean(0, keepdim=True)
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            iv = readout.Intervention(m, {LAYER: v.to(m.device).to(m.dtype)},
                                      channel="img", mode="add", alpha=al)
        with iv:
            b = iv.set_batch(m.chat("Describe this image.", [im]))
            ip = readout.image_positions(m, b)
            A = readout.residuals(m, b, [LAYER])
            # what a monitor reads off those sites
            lg = m.unembed(A[LAYER][ip].to(m.device)).float().cpu()[:, FLAT]
            best = lg.max(0).values
            order = best.argsort(descending=True)
            rank = torch.empty_like(order)
            rank[order] = torch.arange(len(order))
            read_rank = int(min(rank[POSI[t]] for t in CAT_IDS[c]
                                if t in POSI))
            read_score = float(max(best[POSI[t]] for t in CAT_IDS[c]
                                   if t in POSI))

            def multi(prompts, _iv=iv):
                return sc.multi_last_logits(
                    _iv.set_batch(m.chat_batch(prompts, [im])))

            r = tasks.run_two_channels(sc, c, d, multi)
        rows.append(dict(i=k, concept=c, distractor=d, arm=arm,
                         direction=nm or "clean", alpha=al,
                         cond=f"{nm}@{al:+g}" if nm else "clean",
                         read_rank=read_rank, read_score=read_score, **r))
    if k % 15 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
base = df[df.arm == "clean"].set_index("i")

# everything as a change from that photograph's own clean run
for col in ("read_rank", "read_score", "word_margin", "letter_margin"):
    df[f"d_{col}"] = df[col].values - base[col].reindex(df.i).values

agg = df[df.arm != "clean"].groupby(["direction", "alpha"]).agg(
    n=("i", "count"),
    read_rank=("d_read_rank", "mean"), read_score=("d_read_score", "mean"),
    word=("d_word_margin", "mean"), letter=("d_letter_margin", "mean"))
agg["read_per_answer"] = (agg.read_score.abs()
                          / agg.word.abs().replace(0, np.nan))
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\nchange from each photograph's own clean run:\n",
      agg.round(3).to_string())

out = {"repo": a.repo, "layer": LAYER, "sigma": SIG,
       "n": int(df.i.nunique()), "summary": st.jsonable(agg.round(4).to_dict()),
       "minutes": round((time.time() - t0) / 60, 1)}
for al in a.alphas:
    for x, y in (("legible", "causal"), ("causal", "random"),
                 ("legible", "random")):
        ca, cb = f"{x}@{al:+g}", f"{y}@{al:+g}"
        if {ca, cb} <= set(df.cond):
            out[f"read_{x}_vs_{y}@{al}"] = st.wilcoxon(df, cb, ca, "d_read_score")
            out[f"answer_{x}_vs_{y}@{al}"] = st.wilcoxon(df, cb, ca, "d_word_margin")
print("\nreadout moved more by the legible direction, answer moved more by the "
      "causal one?  the two Wilcoxons above the alpha=1 line say so or not.")
json.dump(out, open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
