"""A better lens should give a better handle. Does it?

A readout that reads more clearly is a nice thing to have; a readout whose
directions *intervene* more effectively is the thing that matters, because every
causal claim in this work is made by erasing one.

So: erase a concept with each lens's directions, at every position and then at
the image positions only, and compare.

The second half is the one that matters.  The reference direction at a late
layer is close to the token's own unembedding row, so removing it everywhere
works by suppressing the word where the word is emitted -- and removing it at the
image tokens alone does almost nothing.  The V-Lens direction is the gradient of
the answer logit with respect to an image site, which is a different geometry:
it is the handle on the *visual buffer*.  Whether it grips where the reference
one slips is a clean, falsifiable question about whether the re-derivation bought
anything.

Matched rank is not enough on its own, because a direction that removes more of
the residual will do more damage whatever it points at.  The removed energy
||dh||/||h|| is recorded for every arm, so damage can be read per unit of
disturbance, with the letter channel as the yardstick for total disturbance.
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
from jvlm.vlens import VLens

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens", required=True)
ap.add_argument("--vlens", default=None)
ap.add_argument("--n", type=int, default=90)
ap.add_argument("--fit-prompts", type=int, default=24,
                help="prompts per concept for the direct estimator")
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/lenscausal")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
     else __import__("jlens").lens.JacobianLens.load(a.jlens))
V = VLens.load(a.vlens) if (a.vlens and os.path.exists(a.vlens)) else None
L = [l for l in J.source_layers]
print(m, f"\nlayers {L[0]}..{L[-1]}   V-Lens fit: {'yes' if V else 'no'}",
      flush=True)

pool = cocodata.concept_items(6000, min_area=0.05, seed=3)
by_c: dict[str, list] = {}
for it in pool:
    by_c.setdefault(it["concept"], []).append(it)

sc = tasks.Scorer(m)
items = cocodata.concept_items(a.n, min_area=0.08, seed=11, hard=True,
                               tokenizer=m.tokenizer)
_direct: dict[str, dict] = {}


def direct_dirs(c):
    """Corpus-averaged gradient directions, on prompts disjoint from the test."""
    if c in _direct:
        return _direct[c]
    ids = cocodata.concept_token_ids(m.tokenizer, c)
    acc = direct.DirectDirections(L)
    used = {it["image_id"] for it in items if it["concept"] == c}
    fit_on = [x for x in by_c.get(c, []) if x["image_id"] not in used]
    for x in fit_on[: a.fit_prompts]:
        b = m.chat("Describe this image.", [corpus.load_image(x["path"])])
        try:
            acc.add(m, b, ids)
        except ValueError:
            continue
    _direct[c] = (acc.mean("img"), acc.stability("img"), acc.n,
                  acc.by_channel())
    return _direct[c]


rows = []
t0 = time.time()
for k, it in enumerate(items):
    c, d = it["concept"], it["distractor"]
    ids = cocodata.concept_token_ids(m.tokenizer, c)
    im = corpus.load_image(it["path"])
    b0 = m.chat("Describe this image.", [im])
    acts = readout.residuals(m, b0, [m.n_layers - 1])
    h_final = acts[m.n_layers - 1][-1]

    specs = {"reference": {l: readout.lens_vectors(m, J, ids, l) for l in L}}
    if V is not None:
        specs["vlens"] = {l: V.vectors(m, ids, l, h_final)
                          for l in L if l in V.jacobians}
    dd, stab, ndir, by_ch = direct_dirs(c)
    if dd:
        specs["direct"] = {l: dd[l].to(m.device) for l in dd}

    arms = [("clean", None, "all")]
    for nm in specs:
        arms += [(f"{nm}_all", nm, "all"), (f"{nm}_img", nm, "img")]
    arms.append(("channel", "__channel__", "matched"))
    for arm, nm, chan in arms:
        if nm == "__channel__":
            if not by_ch:
                continue
            iv = readout.ChannelIntervention(
                m, {ch: {l: v.to(m.device) for l, v in d.items()}
                    for ch, d in by_ch.items() if ch in ("img", "txt")},
                mode="ablate", alpha=1.0)
        elif nm is None:
            iv = readout.NoIntervention()
        else:
            iv = readout.Intervention(m, specs[nm], channel=chan,
                                      mode="ablate", alpha=1.0)

        def multi(prompts, _iv=iv):
            return sc.multi_last_logits(_iv.set_batch(m.chat_batch(prompts, [im])))

        with iv:
            r = tasks.run_two_channels(sc, c, d, multi)
        rows.append(dict(i=k, concept=c, distractor=d, arm=arm, cond=arm,
                         channel=chan, rank=len(ids), n_fit=ndir,
                         stability=float(np.mean(list(stab.values()))) if stab else 0.0,
                         energy=float(np.mean(iv.energy))
                         if getattr(iv, "energy", None) else 0.0, **r))
    if k % 15 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
order = [x for x in ("clean", "reference_all", "reference_img",
                     "vlens_all", "vlens_img", "direct_all", "direct_img",
                     "channel")
         if x in set(df.arm)]
agg = df.groupby("arm").agg(
    n=("i", "count"), energy=("energy", "mean"),
    word_correct=("word_correct", "mean"), word_margin=("word_margin", "mean"),
    letter_correct=("letter_correct", "mean"),
    letter_margin=("letter_margin", "mean"),
    stability=("stability", "mean")).reindex(order)
clean_w = float(agg.loc["clean", "word_correct"])
agg["word_lost"] = clean_w - agg["word_correct"]
agg["lost_per_energy"] = agg["word_lost"] / agg["energy"].replace(0, np.nan)
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", agg.round(4).to_string())

from jvlm import stats as st  # noqa: E402
out = {"repo": a.repo, "n": int(df.i.nunique()),
       "summary": agg.round(4).to_dict(),
       "minutes": round((time.time() - t0) / 60, 1)}
for arm in order[1:]:
    out[f"vs_clean_{arm}"] = st.interaction(df, "clean", arm,
                                            "word_margin", "letter_margin")
for x, y in (("reference_all", "direct_all"), ("reference_img", "direct_img")):
    if x in order and y in order:
        out[f"{y}_vs_{x}_word"] = st.wilcoxon(df, x, y, "word_margin")
        print(f"\n{y} against {x} on the word channel:",
              out[f"{y}_vs_{x}_word"])
json.dump(out, open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
