"""The whole method, on any model, without fitting a lens.

Because the directions are gradients of a token's answer logit, they need one
backward pass per token per prompt -- not the ``d_model`` passes a transport
matrix costs.  A concept's directions over a small corpus take seconds, and the
four-channel erasure test that carries the paper's main claim then runs in
minutes on a model whose lens would have taken hours to fit.

That is what this script is: the routes experiment, with the fit removed.  It
takes a model name and nothing else, which is what makes the scaling section
possible at all on one card.
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
from jvlm import cocodata, corpus, direct, lexicon, readout, tasks

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--fit-prompts", type=int, default=20)
ap.add_argument("--layer-stride", type=int, default=1)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/directroutes")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
sc = tasks.Scorer(m)
L = list(range(max(m.n_deepstack, 1), m.n_layers - 1))[:: a.layer_stride]
print(m, f"\ndirections at {len(L)} layers, no lens fit", flush=True)

ROUTES = ["en", "zh", "fr", "syn"]
pool = cocodata.concept_items(6000, min_area=0.05, seed=3)
by_c: dict[str, list] = {}
for it in pool:
    by_c.setdefault(it["concept"], []).append(it)

items = cocodata.concept_items(a.n, min_area=0.08, seed=11, hard=True,
                               tokenizer=m.tokenizer)
_cache: dict = {}


def dirs_for(c, langs):
    """Direct directions for a concept's names in the given languages."""
    key = (c, tuple(langs))
    if key in _cache:
        return _cache[key]
    ids = sorted({t for r in langs
                  for t in lexicon.route_token_ids(m.tokenizer, c, r)})
    if not ids:
        _cache[key] = (None, 0.0, 0)
        return _cache[key]
    acc = direct.DirectDirections(L)
    used = {it["image_id"] for it in items if it["concept"] == c}
    for x in [y for y in by_c.get(c, []) if y["image_id"] not in used][: a.fit_prompts]:
        b = m.chat("Describe this image.", [corpus.load_image(x["path"])])
        try:
            acc.add(m, b, ids)
        except (ValueError, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            continue
    if acc.n == 0:
        _cache[key] = (None, 0.0, 0)
    else:
        st = acc.stability()
        _cache[key] = ({l: v.to(m.device) for l, v in acc.mean().items()},
                       float(np.mean(list(st.values()))), acc.n)
    return _cache[key]


ARMS = [("clean", []), ("en", ["en"]), ("all_langs", ["en", "zh", "fr"])]
rows = []
t0 = time.time()
for k, it in enumerate(items):
    c, d = it["concept"], it["distractor"]
    usable = [r for r in ROUTES if lexicon.usable(c, d, r, m.tokenizer)]
    if "en" not in usable:
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
        stab, nfit = 0.0, 0
        if langs:
            spec, stab, nfit = dirs_for(c, langs)
            if spec is None:
                continue
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
                   stability=stab, n_fit=nfit,
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
agg = df.groupby("arm")[cols + ["energy", "stability"]].mean().reindex(
    [x[0] for x in ARMS])
agg["n"] = df.groupby("arm").size().reindex([x[0] for x in ARMS])
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", agg.round(3).to_string())
json.dump({"repo": a.repo, "n": int(df.i.nunique()), "layers": len(L),
           "fit_prompts": a.fit_prompts,
           "backward_passes": a.fit_prompts * 5 * df.concept.nunique(),
           "summary": agg.round(4).to_dict(),
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
