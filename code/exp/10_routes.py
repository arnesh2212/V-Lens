"""Which part of the direction does the work, and which report routes does it cut?

Two crossed factors.

COMPONENT -- decompose the J-lens direction of concept c at layer l into the part
that lies in the span of the concept's own unembedding rows and the part
orthogonal to it:

    V = J_l^T W_U[T_c],   V_par = proj onto span(W_U[T_c]),   V_perp = V - V_par

The deflationary reading of an erasure result is that everything lives in
``V_par``, because at late layers J is close to the identity.  ``V_perp`` is
whatever the averaged Jacobian adds beyond the unembedding geometry.

ROUTE -- ask for the same visual discrimination through four output channels:
English word, Chinese word, English near-synonym, and an arbitrary letter key.
Only the first shares unembedding rows with the erased directions.

The predictions are opposite.  If the effect is lexical suppression, ``V_par``
carries it all and only the English route falls.  If the erased direction carries
the concept, ``V_perp`` is causally live and the other verbal routes fall too.
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
from jvlm import cocodata, corpus, lexicon, readout, tasks

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/routes")
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

_cache: dict = {}


def components(concept: str) -> dict[str, dict[int, torch.Tensor]]:
    """``{"par"|"perp"|"full"|"unemb": {layer: [k, d]}}`` for one concept."""
    if concept in _cache:
        return _cache[concept]
    ids = cocodata.concept_token_ids(m.tokenizer, concept)
    W = m._lm_head.weight[ids].detach().float()                # [k, d]
    Qw, _ = torch.linalg.qr(W.T)                               # [d, k] basis of W
    out = {"par": {}, "perp": {}, "full": {}, "unemb": {}}
    stats = []
    for l in L:
        J = lens.jacobians[l].to(W.device).float()
        V = W @ J                                              # [k, d]
        Vpar = (V @ Qw) @ Qw.T
        Vperp = V - Vpar
        for key, M in (("par", Vpar), ("perp", Vperp), ("full", V), ("unemb", W)):
            M = M / M.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            out[key][l] = M
        cos = torch.nn.functional.cosine_similarity(
            V / V.norm(dim=-1, keepdim=True),
            W / W.norm(dim=-1, keepdim=True), dim=-1).mean().item()
        stats.append(dict(concept=concept, layer=l, depth=l / (m.n_layers - 1),
                          cos_V_WU=cos,
                          frac_par=float((Vpar.norm() / V.norm()) ** 2)))
    out["_stats"] = stats
    _cache[concept] = out
    return out


ROUTES = ["en", "zh", "syn"]
ARMS = [("clean", None), ("full", "full"), ("par", "par"),
        ("perp", "perp"), ("unemb", "unemb")]

items = cocodata.concept_items(a.n, min_area=0.08, seed=11,
                               hard=True, tokenizer=m.tokenizer)
rows, geom = [], []
t0 = time.time()

for k, it in enumerate(items):
    c, d = it["concept"], it["distractor"]
    routes = [r for r in ROUTES if lexicon.usable(c, d, r, m.tokenizer)]
    if "en" not in routes:
        continue
    im = corpus.load_image(it["path"])
    comp = components(c)
    if k == 0 or c not in {g["concept"] for g in geom}:
        geom.extend(comp["_stats"])

    for arm, which in ARMS:
        iv = (readout.NoIntervention() if which is None
              else readout.Intervention(m, comp[which], channel="all",
                                        mode="ablate", alpha=1.0))

        def multi(prompts, image=im, _iv=iv):
            return sc.multi_last_logits(_iv.set_batch(m.chat_batch(prompts, [image])))

        # every prompt for this item goes through in one forward
        plan, prompts = [], []
        for route in routes:
            x, y = lexicon.route_word(c, route), lexicon.route_word(d, route)
            ix = lexicon.route_token_ids(m.tokenizer, c, route)
            iy = lexicon.route_token_ids(m.tokenizer, d, route)
            cue = lexicon.ROUTE_CUE[route]
            for sign, (p_, q_, ip, iq) in ((1.0, (x, y, ix, iy)),
                                           (-1.0, (y, x, iy, ix))):
                plan.append((route, sign, ip, iq))
                prompts.append(tasks.word_prompt(p_, q_, cue))
        for sign, (p_, q_) in ((1.0, (c, d)), (-1.0, (d, c))):
            plan.append(("letter", sign, None, None))
            prompts.append(tasks.letter_prompt(p_, q_))

        with iv:
            lg = multi(prompts)
        acc: dict[str, list] = {}
        for j, (route, sign, ip, iq) in enumerate(plan):
            v = (sign * sc.letter_margin(lg[j]) if route == "letter"
                 else sign * sc.id_margin(lg[j], ip, iq))
            acc.setdefault(route, []).append(v)
        r = {}
        for route, vals in acc.items():
            r[f"{route}_margin"] = float(np.mean(vals))
            r[f"{route}_correct"] = float(np.mean([v > 0 for v in vals]))
        rows.append(dict(i=k, image_id=it["image_id"], concept=c, distractor=d,
                         arm=arm, n_routes=len(routes), **r))

    if k % 10 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
pd.DataFrame(geom).to_parquet(f"{a.out}/{tag}_geometry.parquet")

cols = [c for c in df.columns if c.endswith("_correct") or c.endswith("_margin")]
agg = df.groupby("arm")[cols].mean().reindex([x[0] for x in ARMS])
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", agg.round(3).to_string())
g = pd.DataFrame(geom)
print("\nGEOMETRY  mean cos(V, W_U) by depth band:")
print(g.assign(band=(g.depth * 4).astype(int) / 4)
       .groupby("band")[["cos_V_WU", "frac_par"]].mean().round(3).to_string())
json.dump({"repo": a.repo, "lens": a.lens, "n": len(items),
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_meta.json", "w"), indent=2)
