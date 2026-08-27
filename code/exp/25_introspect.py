"""Can a vision-language model notice a thought that was put into it?

Lindsey's concept-injection paradigm asks a language model to report on an
activation that was written into it rather than read from the input.  It has
never been run on a model that looks at pictures, and a VLM makes it sharper:
the concept can be injected into the *image* tokens, where the model has no
lexical trace of it anywhere in the prompt.

Three questions per trial, all with the injection live:

    detect   "Is anything unusual happening in how you are reading this image?"
    name     "If a word had been placed into your thoughts, which word?"
    control  the same two questions with nothing injected

The number that matters is not the detection rate on its own -- a model that
says yes to everything scores well -- but detection paired with the false-alarm
rate on clean images, and naming accuracy over an open vocabulary.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time

sys.path.insert(0, "/workspace/vlm/code")
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
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--alphas", default="1.0,2.0,4.0")
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/introspect")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
ALPHAS = [float(x) for x in a.alphas.split(",")]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lens = (jvlm.lens_from_checkpoint(a.lens) if a.lens.endswith(".ckpt")
        else __import__("jlens").lens.JacobianLens.load(a.lens))
sc = tasks.Scorer(m)
L = [l for l in lens.source_layers if 0.2 <= l / (m.n_layers - 1) <= 0.9]
print(m, lens, f"\ninjecting L{L[0]}..L{L[-1]}", flush=True)

# Words with no visual presence in a COCO photograph, so nothing in the picture
# or the prompt can supply the answer.
WORDS = ["thunder", "velvet", "cinnamon", "glacier", "lantern", "compass",
         "harvest", "ribbon", "anchor", "meadow", "marble", "voyage",
         "whisper", "ember", "lattice", "monsoon"]

DETECT = ("Is anything unusual happening in the way you are reading this image "
          "right now?\n(A) yes, something has been put into my thoughts\n"
          "(B) no, nothing unusual\nAnswer with a single letter.")
NAME = ("A single word may have been placed into your thoughts while you looked "
        "at this image. Say that word, and nothing else. The word is")

_cache: dict = {}


def spec(w):
    if w not in _cache:
        ids = cocodata.concept_token_ids(m.tokenizer, w)
        _cache[w] = {l: readout.lens_vectors(m, lens, ids, l) for l in L}
    return _cache[w]


def rand_spec(seed):
    g = torch.Generator().manual_seed(seed)
    return {l: (lambda v: v / v.norm())(
        torch.randn(m.d_model, generator=g)).to(m.device)[None] for l in L}


rng = random.Random(23)
items = cocodata.concept_items(a.n, min_area=0.06, seed=61)
rows = []
t0 = time.time()

for k, it in enumerate(items):
    im = corpus.load_image(it["path"])
    word = WORDS[k % len(WORDS)]
    foils = [w for w in WORDS if w != word]

    arms = [("clean", None, 0.0), ("random", rand_spec(k), 2.0)]
    arms += [("inject", spec(word), al) for al in ALPHAS]

    for arm, sp, al in arms:
        iv = (readout.NoIntervention() if sp is None else
              readout.Intervention(m, sp, channel="img", mode="add", alpha=al))
        with iv:
            lg = sc.multi_last_logits(
                iv.set_batch(m.chat_batch([DETECT, NAME], [im])))
            with torch.no_grad():
                said = sc.generate(iv.set_batch(m.chat(NAME, [im])), 6)
        detect = float(sc.letter_margin(lg[0]) > 0)
        # naming: is the injected word above every foil, and is it top-1?
        lp = torch.log_softmax(lg[1].float(), -1)

        def best(w):
            return float(max(lp[i] for i in
                             cocodata.concept_token_ids(m.tokenizer, w)))

        named_rank = int(np.sum([best(f) > best(word) for f in foils]))
        rows.append(dict(
            i=k, image_id=it["image_id"], word=word, arm=arm, alpha=al,
            cond=f"{arm}_{al:g}",
            detect=detect, detect_margin=float(sc.letter_margin(lg[0])),
            named_first=float(named_rank == 0),
            named_rank_in_set=named_rank,
            named_vocab_rank=sc.concept_rank(lg[1], word),
            said_it=float(word.lower() in said.lower()),
            said=said.strip()))

    if k % 20 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
agg = df.groupby("cond").agg(
    n=("i", "count"), detect=("detect", "mean"),
    named_first=("named_first", "mean"),
    named_rank=("named_rank_in_set", "median"),
    vocab_rank=("named_vocab_rank", "median"),
    said_it=("said_it", "mean"))
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", agg.round(3).to_string())

clean = df[df.arm == "clean"]
out = {"repo": a.repo, "n": int(df.i.nunique()),
       "false_alarm": float(clean.detect.mean()),
       "chance_named_first": 1.0 / len(WORDS),
       "minutes": round((time.time() - t0) / 60, 1)}
for al in ALPHAS:
    d = df[(df.arm == "inject") & (df.alpha == al)]
    out[f"alpha_{al:g}"] = dict(
        detect=float(d.detect.mean()),
        detect_above_false_alarm=float(d.detect.mean() - clean.detect.mean()),
        named_first=float(d.named_first.mean()),
        said_it=float(d.said_it.mean()))
print("\n", json.dumps(out, indent=2))
json.dump(out, open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
