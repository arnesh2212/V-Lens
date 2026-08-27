"""Saying is not choosing: inject the verbal copy and the use copy separately.

The photograph does not contain concept c.  We take the direction a photograph
of c *would* have written, split it into the part inside the production
channel's row space and the part outside, and add one or the other at the image
tokens.

    verbal copy  ->  the model should say c, and still choose correctly
    use copy     ->  the model should choose c, and still not say it

Ablation gives necessity and invites the reply that erasing anything breaks
something.  This gives sufficiency in both directions from a single decomposition
of one vector, which no "the intervention is just damage" account survives: the
two halves sum to the same vector and do opposite things.
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
from jvlm import cocodata, corpus, percept, readout, tasks

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--concepts", type=int, default=12)
ap.add_argument("--per-concept", type=int, default=40, help="images per side")
ap.add_argument("--test-per-concept", type=int, default=10)
ap.add_argument("--rank", type=int, default=64, help="channel row-space rank")
ap.add_argument("--alphas", default="0.5,1.0,2.0")
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/patch")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
ALPHAS = [float(x) for x in a.alphas.split(",")]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lens = (jvlm.lens_from_checkpoint(a.lens) if a.lens.endswith(".ckpt")
        else __import__("jlens").lens.JacobianLens.load(a.lens))
sc = tasks.Scorer(m)
L = [l for l in lens.source_layers if 0.15 <= l / (m.n_layers - 1) <= 0.95]
print(m, lens, f"\ninjecting at L{L[0]}..L{L[-1]}", flush=True)

VOCAB = [cocodata.primary_token_id(m.tokenizer, c) for c in cocodata.SINGLE_WORD]
QCHAN = {l: percept.channel_basis(m, lens, VOCAB, l, rank=a.rank) for l in L}
print(f"channel row space truncated to rank {QCHAN[L[0]].shape[1]} of {m.d_model}",
      flush=True)

rng = random.Random(31)
pool = cocodata.concept_items(8000, min_area=0.05, seed=3)
by_c: dict[str, list] = {}
for it in pool:
    by_c.setdefault(it["concept"], []).append(it)
concepts = sorted(by_c, key=lambda c: -len(by_c[c]))[: a.concepts]
all_ids = sorted({it["image_id"] for it in pool})
print("concepts:", concepts, flush=True)

t0 = time.time()
rows, geo = [], []
for ci, c in enumerate(concepts):
    pos_ids = [it["image_id"] for it in by_c[c]][: a.per_concept]
    have = set(pos_ids)
    neg_ids = [i for i in all_ids
               if c not in cocodata.present(i) and i not in have][: a.per_concept]
    E = percept.perceptual_direction(m, pos_ids, neg_ids, L)

    verbal, use, rand = {}, {}, {}
    for l in L:
        ea, eb = E[l]
        e = (ea + eb) / 2
        v, u = percept.split_direction(e, QCHAN[l])
        frac, null = percept.verbalizable_fraction(ea, eb, QCHAN[l])
        geo.append(dict(concept=c, layer=l, depth=l / (m.n_layers - 1),
                        V=frac, null=null,
                        excess=(frac - null) / (1 - null) if frac == frac else float("nan"),
                        norm_verbal=float(v.norm()), norm_use=float(u.norm())))
        # each injected direction is a unit vector, so the intervention strength
        # is the same for both halves and the comparison is not a norm effect
        verbal[l] = (v / v.norm().clamp_min(1e-8))[None]
        use[l] = (u / u.norm().clamp_min(1e-8))[None]
        g = torch.randn(m.d_model, generator=torch.Generator().manual_seed(ci * 71 + l))
        rand[l] = (g / g.norm()).to(v.device)[None]

    # test images that do NOT contain c, and were not used to estimate the
    # direction -- otherwise part of what we inject was measured on the very
    # image we then evaluate on
    used = set(pos_ids) | set(neg_ids)
    tests = [i for i in all_ids
             if c not in cocodata.present(i) and i not in used]
    rng.shuffle(tests)
    tests = tests[: a.test_per_concept]
    for k, iid in enumerate(tests):
        present = sorted(cocodata.present(iid) & set(cocodata.SINGLE_WORD))
        if not present:
            continue
        real = present[0]                    # something that IS in the picture
        im = corpus.load_image(str(cocodata.image_path(iid)))

        for arm, spec in (("clean", None), ("verbal", verbal),
                          ("use", use), ("random", rand)):
            for al in ([0.0] if arm == "clean" else ALPHAS):
                iv = (readout.NoIntervention() if spec is None else
                      readout.Intervention(m, spec, channel="img",
                                           mode="add", alpha=al))

                def multi(prompts, image=im, _iv=iv):
                    return sc.multi_last_logits(
                        _iv.set_batch(m.chat_batch(prompts, [image])))

                with iv:
                    # does it SAY c, unprompted?
                    lg = multi([tasks.REPORT_CLOZE,
                                tasks.word_prompt(c, real),
                                tasks.word_prompt(real, c),
                                tasks.letter_prompt(c, real),
                                tasks.letter_prompt(real, c)])
                say_rank = sc.concept_rank(lg[0], c)
                say_lp = sc.concept_logprob(lg[0], c)
                word = [sc.word_margin(lg[1], c, real),
                        -sc.word_margin(lg[2], real, c)]
                letter = [sc.letter_margin(lg[3]), -sc.letter_margin(lg[4])]
                rows.append(dict(
                    concept=c, image_id=iid, real=real, arm=arm, alpha=al,
                    i=f"{c}_{k}", cond=f"{arm}_{al:g}",
                    say_rank=say_rank, say_logprob=say_lp,
                    # positive margin = the model prefers the ABSENT injected
                    # concept over the object that is really there
                    word_margin=float(np.mean(word)),
                    word_picks_injected=float(np.mean([v > 0 for v in word])),
                    letter_margin=float(np.mean(letter)),
                    letter_picks_injected=float(np.mean([v > 0 for v in letter]))))
    print(f"  {c}  {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")
    pd.DataFrame(geo).to_parquet(f"{a.out}/{tag}_geometry.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
g = pd.DataFrame(geo)
g.to_parquet(f"{a.out}/{tag}_geometry.parquet")

agg = df.groupby(["arm", "alpha"]).agg(
    n=("i", "count"), say_rank=("say_rank", "median"),
    say_logprob=("say_logprob", "mean"),
    word_picks=("word_picks_injected", "mean"),
    letter_picks=("letter_picks_injected", "mean"),
    word_margin=("word_margin", "mean"), letter_margin=("letter_margin", "mean"))
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", agg.round(3).to_string())
print("\nverbalizable fraction by depth (value, null, excess):")
print(g.assign(b=(g.depth * 4).astype(int) / 4).groupby("b")[
    ["V", "null", "excess"]].mean().round(4).to_string())

from jvlm import stats as st  # noqa: E402
out = {"repo": a.repo, "rank": int(QCHAN[L[0]].shape[1]),
       "V_mean": float(g.V.mean()), "V_null": float(g.null.mean()),
       "V_excess": float(g.excess.mean()),
       "minutes": round((time.time() - t0) / 60, 1)}
mid = f"{ALPHAS[len(ALPHAS)//2]:g}"
for arm in ("verbal", "use", "random"):
    sub = df[df.cond.isin(["clean_0", f"{arm}_{mid}"])]
    out[arm] = st.interaction(sub, "clean_0", f"{arm}_{mid}",
                              "word_margin", "letter_margin")
    print(f"{arm}: word-minus-letter interaction dd = {out[arm].get('dd'):.2f}  "
          f"p = {out[arm].get('p'):.2g}")
json.dump(out, open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
