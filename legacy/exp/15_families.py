"""Is the dissociation a fact about nouns, or about the workspace?

The same erasure and the same two response channels, run over four families of
concept that COCO can supply ground truth for: object identity, colour, count,
and spatial relation.  Only the first is a concrete noun; the other three are the
cases a vocabulary readout is supposed to be bad at, which is exactly why they
are the interesting test.

If the workspace is a report interface rather than a noun dictionary, erasing a
concept's direction should take the word channel and spare the letter channel in
every family.
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
from jvlm import cocodata, concepts, corpus, readout, tasks

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--per-family", type=int, default=70)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/families")
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

items = []
for fam, fn in concepts.FAMILIES.items():
    got = fn(a.per_family, seed=5)
    print(f"  {fam}: {len(got)} items", flush=True)
    items.extend(got)
print(f"{len(items)} items total", flush=True)

_cache: dict = {}


def spec(word: str):
    if word not in _cache:
        ids = cocodata.concept_token_ids(m.tokenizer, word)
        _cache[word] = {l: readout.lens_vectors(m, lens, ids, l) for l in L}
    return _cache[word]


def rand_spec(seed: int, k: int):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for l in L:
        V = torch.randn(k, m.d_model, generator=g)
        out[l] = (V / V.norm(dim=-1, keepdim=True)).to(m.device)
    return out


ARMS = [("clean", None), ("erase", "c"), ("random", "r")]
rows = []
t0 = time.time()
for k, it in enumerate(items):
    c, d = it["concept"], it["distractor"]
    # a margin between two options sharing a token is clipped toward zero and
    # carries no signal, so those items are dropped rather than averaged in
    if c == d or not cocodata.disjoint(m.tokenizer, c, d):
        continue
    im = corpus.load_image(it["path"])
    q = it["question"]
    nvar = len(cocodata.concept_token_ids(m.tokenizer, c))

    for arm, which in ARMS:
        iv = (readout.NoIntervention() if which is None else
              readout.Intervention(
                  m, spec(c) if which == "c" else rand_spec(k, nvar),
                  channel="all", mode="ablate", alpha=1.0))

        def multi(prompts, image=im, _iv=iv):
            return sc.multi_last_logits(_iv.set_batch(m.chat_batch(prompts, [image])))

        # the same question twice, once for the word and once for a letter,
        # both option orders, all four in one forward
        prompts, plan = [], []
        for sign, (x, y) in ((1.0, (c, d)), (-1.0, (d, c))):
            prompts.append(f"{q}, the {x} or the {y}?\nAnswer with one word.")
            plan.append(("word", sign, x, y))
        for sign, (x, y) in ((1.0, (c, d)), (-1.0, (d, c))):
            prompts.append(f"{q}?\n(A) {x}\n(B) {y}\nAnswer with a single letter.")
            plan.append(("letter", sign, x, y))
        with iv:
            lg = multi(prompts)
        acc: dict[str, list] = {}
        for j, (ch, sign, x, y) in enumerate(plan):
            v = (sign * sc.letter_margin(lg[j]) if ch == "letter"
                 else sign * sc.word_margin(lg[j], x, y))
            acc.setdefault(ch, []).append(v)
        rows.append(dict(
            i=k, family=it["kind"], image_id=it["image_id"], concept=c,
            distractor=d, subject=it["subject"], arm=arm, cond=arm,
            word_margin=float(np.mean(acc["word"])),
            word_correct=float(np.mean([v > 0 for v in acc["word"]])),
            letter_margin=float(np.mean(acc["letter"])),
            letter_correct=float(np.mean([v > 0 for v in acc["letter"]]))))

    if k % 25 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
agg = df.groupby(["family", "arm"]).agg(
    n=("i", "count"),
    word_correct=("word_correct", "mean"),
    letter_correct=("letter_correct", "mean"),
    word_margin=("word_margin", "mean"),
    letter_margin=("letter_margin", "mean"))
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", agg.round(3).to_string())

from jvlm import stats as st  # noqa: E402
out = {"repo": a.repo, "n": len(items),
       "minutes": round((time.time() - t0) / 60, 1)}
for fam in df.family.unique():
    sub = df[df.family == fam].copy()
    sub["concept"] = sub["concept"].astype(str)
    out[fam] = st.interaction(sub, "clean", "erase", "word_margin", "letter_margin")
    print(f"{fam}: dd = {out[fam].get('dd'):.2f} "
          f"[{out[fam].get('ci_lo', float('nan')):.2f},"
          f"{out[fam].get('ci_hi', float('nan')):.2f}]  p = {out[fam].get('p'):.2g}")
json.dump(out, open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
