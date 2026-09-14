"""Knowing it is not there, and not being able to say where it came from.

Two questions about the same photograph and the same sentence, asked back to
back, both answered with an arbitrary letter:

  presence   "Is a {b} visible in the photograph?"          -- can it check?
  source     "One of these you can see in the photograph and the other you were
              only told about. Which one can you see?"      -- can it attribute?

The first is answerable by looking.  The second is answerable by looking and then
saying which way round it came out; it adds no perceptual demand at all.  A model
that answers the first and not the second is not failing to see.  It is failing
to report a fact about its own input.

Every trial runs both, on the same forward budget, with both option orders, so
the comparison is within-item and the letter bias cancels.
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

import jvlm
from jvlm import cocodata, corpus, tasks

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/source")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
sc = tasks.Scorer(m)
print(m, flush=True)

ASSERT = "The scene also contains a {b}."
PRESENCE = ("Is a {b} visible in the photograph?\n(A) yes\n(B) no\n"
            "Answer with a single letter.")
PRESENCE_A = ("Is a {a} visible in the photograph?\n(A) yes\n(B) no\n"
              "Answer with a single letter.")
SOURCE = ("One of these you can see in the photograph; the other you were only "
          "told about in the sentence above.\nWhich one can you see?\n"
          "(A) {x}\n(B) {y}\nAnswer with a single letter.")
TOLD = ("One of these you can see in the photograph; the other you were only "
        "told about in the sentence above.\nWhich one were you only told "
        "about?\n(A) {x}\n(B) {y}\nAnswer with a single letter.")

rng = random.Random(7)
items = cocodata.concept_items(a.n, min_area=0.08, seed=11,
                               hard=True, tokenizer=m.tokenizer)
rows = []
t0 = time.time()
for k, it in enumerate(items):
    A, B = it["concept"], it["distractor"]      # A is in the picture, B is not
    im = corpus.load_image(it["path"])
    pre = ASSERT.format(b=B)

    prompts = [
        f"{pre}\n{PRESENCE.format(b=B)}",        # 0 correct answer: B (no)
        f"{pre}\n{PRESENCE_A.format(a=A)}",      # 1 correct answer: A (yes)
        f"{pre}\n{SOURCE.format(x=A, y=B)}",     # 2 correct: A
        f"{pre}\n{SOURCE.format(x=B, y=A)}",     # 3 correct: B
        f"{pre}\n{TOLD.format(x=A, y=B)}",       # 4 correct: B
        f"{pre}\n{TOLD.format(x=B, y=A)}",       # 5 correct: A
    ]
    lg = sc.multi_last_logits(m.chat_batch(prompts, [im]))
    mg = [sc.letter_margin(x) for x in lg]

    presence = float(((mg[0] < 0) + (mg[1] > 0)) / 2)     # absent: no; present: yes
    source = float(((mg[2] > 0) + (mg[3] < 0)) / 2)       # pick the seen one
    told = float(((mg[4] < 0) + (mg[5] > 0)) / 2)         # pick the told one

    # The two questions are complementary: whatever it names as the thing it can
    # see, it must not also name as the thing it was only told about.  A model
    # with any usable source signal passes this whatever its accuracy; a model
    # answering both from the same lexical prior fails it every time.  This is
    # the statistic that does not depend on which phrasing we happen to prefer.
    picks_seen = [mg[2] > 0, mg[3] < 0]                   # True -> picked A
    picks_told = [mg[4] > 0, mg[5] < 0]                   # True -> picked A
    consistent = float(np.mean([p != q for p, q in zip(picks_seen, picks_told)]))

    rows.append(dict(i=k, concept=A, distractor=B, image_id=it["image_id"],
                     presence_correct=presence, source_correct=source,
                     told_correct=told, consistent=consistent,
                     presence_margin=float((-mg[0] + mg[1]) / 2),
                     source_margin=float((mg[2] - mg[3]) / 2),
                     told_margin=float((-mg[4] + mg[5]) / 2)))
    if k % 40 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")

from scipy import stats as sst  # noqa: E402

from jvlm import stats as st  # noqa: E402

summ = {"repo": a.repo, "n": len(df), "minutes": round((time.time() - t0) / 60, 1)}
for col in ("presence_correct", "source_correct", "told_correct",
            "consistent"):
    mean, lo, hi = st.cluster_bootstrap_ci(df, col)
    summ[col] = dict(mean=mean, lo=lo, hi=hi)
    print(f"{col:20s} {100*mean:5.1f}%  [{100*lo:.1f}, {100*hi:.1f}]")
w = sst.wilcoxon(df.presence_correct, df.source_correct)
summ["presence_vs_source_p"] = float(w.pvalue)
print(f"\npresence vs source, Wilcoxon p = {w.pvalue:.3g}  (n = {len(df)})")
pd.DataFrame([{k: (v["mean"] if isinstance(v, dict) else v)
               for k, v in summ.items() if k.endswith("_correct")}]).to_csv(
    f"{a.out}/{tag}_summary.csv", index=False)
json.dump(summ, open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
