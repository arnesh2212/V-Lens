"""Where the tangent slips, does the secant grip?

Erasing a concept's tangent direction at the image tokens does essentially
nothing (experiment 29: 0.00 of the word channel lost, against 0.75 when the
same direction is erased everywhere).  That is a real finding about the tangent:
the corpus-averaged derivative of a token's logit points, at a late layer, close
to that token's own unembedding row, so it is a handle on *emitting the word*,
not on *having seen the thing*.

The secant is fit to a different question -- what actually happens to the answer
when this image site is overwritten -- so its directions have no reason to point
at the word.  This asks whether they point at the evidence instead.

The two response channels do the separating.  Both arms ask the same question
about the same photograph; one is answered with the object's name, the other
with an arbitrary letter key.  A handle on production kills the word channel and
spares the letter channel.  A handle on perception kills both, because a model
that no longer has the evidence cannot pick the right letter either.
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
from jvlm import cocodata, corpus, direct, readout, tasks
from jvlm.patchfit import SecantLens

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--secant", required=True)
ap.add_argument("--jlens", default=None)
ap.add_argument("--n", type=int, default=90)
ap.add_argument("--fit-prompts", type=int, default=16)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/secantcausal")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
S = SecantLens.load(a.secant)
LAYERS = sorted(S.maps)
J = None
if a.jlens and os.path.exists(a.jlens):
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))
print(m, f"\nsecant at layers {LAYERS}   alpha {a.alpha}", flush=True)

pool = cocodata.concept_items(6000, min_area=0.05, seed=3)
by_c: dict[str, list] = {}
for it in pool:
    by_c.setdefault(it["concept"], []).append(it)

sc = tasks.Scorer(m)
items = cocodata.concept_items(a.n, min_area=0.08, seed=11, hard=True,
                               tokenizer=m.tokenizer)
_tan: dict[str, dict] = {}


def tangent_dirs(c):
    """Corpus-averaged logit gradients, on photographs disjoint from the test."""
    if c in _tan:
        return _tan[c]
    ids = cocodata.concept_token_ids(m.tokenizer, c)
    acc = direct.DirectDirections(LAYERS)
    used = {it["image_id"] for it in items if it["concept"] == c}
    for x in [y for y in by_c.get(c, []) if y["image_id"] not in used][: a.fit_prompts]:
        b = m.chat("Describe this image.",
                   [corpus.load_image(x["path"], square=a.square)])
        try:
            acc.add(m, b, ids)
        except (ValueError, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            continue
    _tan[c] = ({l: v.to(m.device) for l, v in acc.mean("img").items()}
               if acc.n else None)
    return _tan[c]


rows = []
t0 = time.time()
for k, it in enumerate(items):
    c, d = it["concept"], it["distractor"]
    ids = cocodata.concept_token_ids(m.tokenizer, c)
    im = corpus.load_image(it["path"], square=a.square)

    ids = [t for t in ids if t in S._pos]
    if not ids:
        continue
    specs = {"secant": {l: S.vectors(ids, l).to(m.device).to(m.dtype)
                        for l in LAYERS}}
    td = tangent_dirs(c)
    if td:
        specs["tangent"] = td
    if J is not None:
        specs["reference"] = {l: readout.lens_vectors(m, J, ids, l)
                              for l in LAYERS if l in J.jacobians}
    # the most legible lens there is: the token's own output embedding, no
    # transport at all.  If legibility and causal fidelity really run in
    # opposite directions, this should be the best handle on the word and the
    # worst handle on anything else.
    W = m._lm_head.weight[ids].detach().float()
    W = W / W.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    specs["unembed"] = {l: W.to(m.device).to(m.dtype) for l in LAYERS}

    arms = [("clean", None, "all")]
    for nm in specs:
        arms += [(f"{nm}_img", nm, "img"), (f"{nm}_all", nm, "all")]
    for arm, nm, chan in arms:
        iv = (readout.NoIntervention() if nm is None else
              readout.Intervention(m, specs[nm], channel=chan, mode="ablate",
                                   alpha=a.alpha))

        def multi(prompts, _iv=iv):
            return sc.multi_last_logits(_iv.set_batch(m.chat_batch(prompts, [im])))

        with iv:
            r = tasks.run_two_channels(sc, c, d, multi)
        rows.append(dict(i=k, concept=c, distractor=d, arm=arm, cond=arm,
                         channel=chan, rank=len(ids),
                         energy=float(np.mean(iv.energy))
                         if getattr(iv, "energy", None) else 0.0, **r))
    if k % 15 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
order = [x for x in ("clean", "secant_img", "tangent_img", "reference_img",
                     "unembed_img", "secant_all", "tangent_all",
                     "reference_all", "unembed_all")
         if x in set(df.arm)]
agg = df.groupby("arm").agg(
    n=("i", "count"), energy=("energy", "mean"),
    word_correct=("word_correct", "mean"), word_margin=("word_margin", "mean"),
    letter_correct=("letter_correct", "mean"),
    letter_margin=("letter_margin", "mean")).reindex(order)
cw = float(agg.loc["clean", "word_correct"])
cl = float(agg.loc["clean", "letter_correct"])
agg["word_lost"] = cw - agg["word_correct"]
agg["letter_lost"] = cl - agg["letter_correct"]
# a handle on production takes the word and spares the letter; a handle on
# perception takes both, so the ratio is the diagnostic
agg["letter_share"] = agg["letter_lost"] / agg["word_lost"].replace(0, np.nan)
agg["word_per_energy"] = agg["word_lost"] / agg["energy"].replace(0, np.nan)
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", agg.round(4).to_string())

from jvlm import stats as st  # noqa: E402
out = {"repo": a.repo, "n": int(df.i.nunique()), "alpha": a.alpha,
       "layers": LAYERS, "summary": st.jsonable(agg.round(4).to_dict()),
       "minutes": round((time.time() - t0) / 60, 1)}
for arm in order[1:]:
    out[f"vs_clean_{arm}"] = st.interaction(df, "clean", arm,
                                            "word_margin", "letter_margin")
for x in ("tangent_img", "reference_img", "unembed_img", "unembed_all"):
    if x in order:
        out[f"secant_vs_{x}_word"] = st.wilcoxon(df, x, "secant_img",
                                                 "word_margin")
        out[f"secant_vs_{x}_letter"] = st.wilcoxon(df, x, "secant_img",
                                                   "letter_margin")
        print(f"\nsecant_img against {x}:  word "
              f"{out[f'secant_vs_{x}_word']}\n   letter "
              f"{out[f'secant_vs_{x}_letter']}")
json.dump(out, open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
