"""A random control matched in energy, not only in rank.

A rank-matched random subspace is the obvious control and it is not sufficient:
the residual stream is anisotropic, so a concept's own directions can carry far
more of the residual's mass than a random subspace of the same dimension does.
If the random control deletes less, it is an easier control, and the comparison
flatters the result.

So this measures how much of the residual each ablation actually removes, and
then reruns the random one with its strength scaled so the two remove the same
amount.  If the concept ablation still takes the word channel and the
energy-matched random one still does not, the effect is about which directions
were removed rather than how much was removed.
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
from jvlm import cocodata, corpus, readout, tasks

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/energy")
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

items = cocodata.concept_items(a.n, min_area=0.08, seed=11,
                               hard=True, tokenizer=m.tokenizer)
rows = []
t0 = time.time()
_c: dict = {}


def cspec(c):
    if c not in _c:
        ids = cocodata.concept_token_ids(m.tokenizer, c)
        _c[c] = ({l: readout.lens_vectors(m, lens, ids, l) for l in L}, len(ids))
    return _c[c]


def rspec(seed, k):
    g = torch.Generator().manual_seed(seed)
    return {l: (lambda V: V / V.norm(dim=-1, keepdim=True))(
        torch.randn(k, m.d_model, generator=g)).to(m.device) for l in L}


for k, it in enumerate(items):
    c, d = it["concept"], it["distractor"]
    im = corpus.load_image(it["path"])
    spec, nvar = cspec(c)
    rnd = rspec(k, nvar)

    def measure(iv):
        def multi(prompts, _iv=iv):
            return sc.multi_last_logits(_iv.set_batch(m.chat_batch(prompts, [im])))
        with iv:
            r = tasks.run_two_channels(sc, c, d, multi)
        r["energy"] = float(np.mean(iv.energy)) if getattr(iv, "energy", None) else 0.0
        return r

    base = measure(readout.NoIntervention())
    conc = measure(readout.Intervention(m, spec, channel="all",
                                        mode="ablate", alpha=1.0))
    rand1 = measure(readout.Intervention(m, rnd, channel="all",
                                         mode="ablate", alpha=1.0))
    # scale the random ablation so it removes the same share of the residual
    scale = (conc["energy"] / rand1["energy"]) if rand1["energy"] > 1e-9 else 1.0
    scale = float(np.clip(scale, 0.1, 40.0))
    randm = measure(readout.Intervention(m, rnd, channel="all",
                                         mode="ablate", alpha=scale))

    for name, r in (("clean", base), ("concept", conc),
                    ("random_rank", rand1), ("random_energy", randm)):
        rows.append(dict(i=k, concept=c, distractor=d, cond=name, arm=name,
                         alpha=(scale if name == "random_energy" else
                                (0.0 if name == "clean" else 1.0)), **r))
    if k % 20 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  {el:.0f}s  scale={scale:.1f}  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
agg = df.groupby("arm").agg(
    n=("i", "count"), energy=("energy", "mean"), alpha=("alpha", "mean"),
    word_correct=("word_correct", "mean"), word_margin=("word_margin", "mean"),
    letter_correct=("letter_correct", "mean"),
    letter_margin=("letter_margin", "mean")).reindex(
    ["clean", "concept", "random_rank", "random_energy"])
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", agg.round(4).to_string())
json.dump({"repo": a.repo, "n": int(df.i.nunique()),
           "summary": agg.round(4).to_dict(),
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
