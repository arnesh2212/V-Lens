"""Is there one production channel, or one per language?

Erasing a concept's English directions leaves the model able to name it in
Chinese.  That already rules out a single amodal store of "the concept, ready to
be said".  But it leaves two possibilities open: either the Chinese name is
re-derived from the use copy on demand, or the model keeps a separate output
buffer per language and we only emptied one of them.

Erasing the languages one at a time and then together separates these.  If the
buffers are separate, each erasure costs its own language and the union costs
both.  If the name is re-derived, erasing every language we can enumerate should
still leave some naming intact, because the derivation runs from somewhere we
did not touch.

Either way, the letter key should survive all of it -- and that is the point.
No amount of emptying output buffers should touch a decision that never went
through one.
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
from jvlm import cocodata, corpus, lexicon, readout, tasks

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/language")
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

ROUTES = ["en", "zh", "fr"]
ARMS = [("clean", []), ("en", ["en"]), ("zh", ["zh"]), ("fr", ["fr"]),
        ("en+zh", ["en", "zh"]), ("en+zh+fr", ["en", "zh", "fr"])]


def ids_for(c, langs):
    out = []
    for r in langs:
        out += lexicon.route_token_ids(m.tokenizer, c, r)
    return sorted(set(out))


items = cocodata.concept_items(a.n, min_area=0.08, seed=11,
                               hard=True, tokenizer=m.tokenizer)
rows = []
t0 = time.time()
for k, it in enumerate(items):
    c, d = it["concept"], it["distractor"]
    usable = [r for r in ROUTES if lexicon.usable(c, d, r, m.tokenizer)]
    if len(usable) < 3:
        continue
    im = corpus.load_image(it["path"])

    plan, prompts = [], []
    for r in usable:
        x, y = lexicon.route_word(c, r), lexicon.route_word(d, r)
        ix = lexicon.route_token_ids(m.tokenizer, c, r)
        iy = lexicon.route_token_ids(m.tokenizer, d, r)
        cue = lexicon.ROUTE_CUE[r]
        for sign, (p_, q_, ip, iq) in ((1.0, (x, y, ix, iy)),
                                       (-1.0, (y, x, iy, ix))):
            plan.append((r, sign, ip, iq))
            prompts.append(tasks.word_prompt(p_, q_, cue))
    for sign, (x, y) in ((1.0, (c, d)), (-1.0, (d, c))):
        plan.append(("letter", sign, None, None))
        prompts.append(tasks.letter_prompt(x, y))

    for arm, langs in ARMS:
        if langs:
            tids = ids_for(c, langs)
            spec = {l: readout.lens_vectors(m, lens, tids, l) for l in L}
            iv = readout.Intervention(m, spec, channel="all", mode="ablate",
                                      alpha=1.0)
        else:
            iv = readout.NoIntervention()
        with iv:
            lg = sc.multi_last_logits(iv.set_batch(m.chat_batch(prompts, [im])))
        acc: dict[str, list] = {}
        for j, (r, sign, ip, iq) in enumerate(plan):
            v = (sign * sc.letter_margin(lg[j]) if r == "letter"
                 else sign * sc.id_margin(lg[j], ip, iq))
            acc.setdefault(r, []).append(v)
        rec = dict(i=k, concept=c, distractor=d, arm=arm, cond=arm,
                   n_erased=len(langs),
                   energy=float(np.mean(iv.energy))
                   if getattr(iv, "energy", None) else 0.0)
        for r, vals in acc.items():
            rec[f"{r}_margin"] = float(np.mean(vals))
            rec[f"{r}_correct"] = float(np.mean([v > 0 for v in vals]))
        rows.append(rec)

    if k % 20 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
cols = [c for c in df.columns if c.endswith("_correct")]
agg = df.groupby("arm")[cols + ["energy"]].mean().reindex([x[0] for x in ARMS])
agg["n"] = df.groupby("arm").size().reindex([x[0] for x in ARMS])
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", agg.round(3).to_string())
json.dump({"repo": a.repo, "n": int(df.i.nunique()),
           "summary": agg.round(4).to_dict(),
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
