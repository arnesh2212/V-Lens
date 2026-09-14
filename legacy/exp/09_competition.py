"""Do the two input channels compete for one workspace?

Anthropic report that the workspace holds on the order of ten to twenty-five
concepts at once.  A language model has one input channel, so a capacity limit
can only be probed by making the text busier.  A VLM has two, which makes a
sharper test available: hold the text fixed, make the *picture* busier, and watch
what happens to a concept that arrived through the text.

Each trial gives the model a word to hold ("Remember this word: X") alongside a
photograph carrying a controlled number of distinct COCO objects.  We then read

    rank of X in the workspace at the final prompt position
    how many of the picture's own objects occupy the workspace top-k
    whether the model can still report X when asked

If the workspace is a shared, capacity-limited resource, the third falls as the
second rises -- and the failure is a mechanism, not an accident of prompting.
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
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--topk", type=int, default=25)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/competition")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lens = (jvlm.lens_from_checkpoint(a.lens) if a.lens.endswith(".ckpt")
        else __import__("jlens").lens.JacobianLens.load(a.lens))
sc = tasks.Scorer(m)
LAYERS = [l for l in lens.source_layers if 0.35 <= l / (m.n_layers - 1) <= 0.90]
print(m, lens, f"\nreading L{LAYERS[0]}..{LAYERS[-1]}", flush=True)

# Words to hold in mind.  Deliberately not COCO objects, so the picture can
# never supply the answer.
HOLD = ["velvet", "thunder", "marble", "compass", "harvest", "lantern",
        "voyage", "cinnamon", "glacier", "ribbon", "anchor", "meadow"]

d = cocodata._load()
rng = random.Random(17)
by_count: dict[int, list[int]] = {}
for iid, objs in d["per_image"].items():
    k = len(objs & set(cocodata.SINGLE_WORD))
    by_count.setdefault(min(k, 9), []).append(iid)

# A balanced ladder of scene complexity.
LADDER = [1, 2, 3, 4, 5, 6, 7, 8]
per = max(4, a.n // len(LADDER))
trials = []
for k in LADDER:
    pool = by_count.get(k, [])
    rng.shuffle(pool)
    for iid in pool[:per]:
        trials.append(dict(image_id=iid, n_objects=k,
                           hold=rng.choice(HOLD),
                           path=str(cocodata.image_path(iid))))
rng.shuffle(trials)
print(f"{len(trials)} trials over {LADDER}", flush=True)

CATS = cocodata.SINGLE_WORD
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
PROMPT = ("Remember this word: {}.\nNow look at the photograph.")
RECALL = "\nWhat was the word you were asked to remember? Answer with one word."

rows = []
t0 = time.time()
for i, tr in enumerate(trials):
    im = corpus.load_image(tr["path"])
    pre = PROMPT.format(tr["hold"])
    b = m.chat(pre, [im])
    acts = readout.residuals(m, b, LAYERS)
    ll, _ = readout.lens_logits(m, lens, b, layers=LAYERS, positions=[-1],
                                cached=acts)

    hold_ids = m.tokenizer.encode(" " + tr["hold"], add_special_tokens=False)[:1]
    here = cocodata.present(tr["image_id"]) & set(cocodata.SINGLE_WORD)

    best_rank, occupancy, pr = 10**9, 0, 0.0
    for l in LAYERS:
        lg = ll[l][0]
        r = int((lg > lg[hold_ids[0]]).sum())
        best_rank = min(best_rank, r)
        top = set(lg.topk(a.topk).indices.tolist())
        occupancy = max(occupancy,
                        sum(any(t in top for t in CAT_IDS[c]) for c in here))
        # Participation ratio over the object vocabulary: the effective number
        # of object words the workspace is holding at this position.  The
        # J-lens vectors are an overcomplete, correlated dictionary, so a
        # coefficient decomposition is ill-posed; a normalised distribution
        # over a fixed vocabulary is not.
        cat_scores = torch.stack([
            torch.stack([lg[t] for t in CAT_IDS[c]]).max() for c in CATS])
        p = torch.softmax(cat_scores.float(), 0)
        pr = max(pr, float(1.0 / (p ** 2).sum()))

    rec = sc.last_logits(m.chat(pre + RECALL, [im]))
    rec_rank = int((rec > rec[hold_ids[0]]).sum())
    rows.append(dict(image_id=tr["image_id"], n_objects=tr["n_objects"],
                     hold=tr["hold"], hold_rank=best_rank,
                     occupancy=occupancy, participation=pr,
                     recall_rank=rec_rank, recalled=float(rec_rank == 0)))
    if i % 50 == 0:
        print(f"  {i}/{len(trials)}  {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
g = df.groupby("n_objects").agg(
    n=("hold", "count"), hold_rank=("hold_rank", "median"),
    occupancy=("occupancy", "mean"), participation=("participation", "mean"),
    recalled=("recalled", "mean"), recall_rank=("recall_rank", "median"))
g.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", g.round(3).to_string())

from scipy.stats import spearmanr  # noqa: E402
summ = dict(n=len(df), minutes=round((time.time() - t0) / 60, 1),
            participation_mean=float(df.participation.mean()),
            participation_by_objects=g.participation.round(3).to_dict())
for xa, xb in (("n_objects", "occupancy"), ("n_objects", "participation"),
               ("occupancy", "hold_rank"), ("participation", "hold_rank"),
               ("occupancy", "recalled"), ("participation", "recalled"),
               ("n_objects", "recalled")):
    rho, pv = spearmanr(df[xa], df[xb])
    summ[f"rho_{xa}_{xb}"] = float(rho)
    summ[f"p_{xa}_{xb}"] = float(pv)
json.dump(summ, open(f"{a.out}/{tag}_stats.json", "w"), indent=2)
print("\n", json.dumps(summ, indent=2))
