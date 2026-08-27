"""A behavioural ordering predicted from the weights, with nothing fitted.

Erasing the span of a concept's English pullbacks costs other output channels in
proportion to how much of *their* pullbacks lie inside that span.  The lemma
gives the bound; the Gram overlap

    g_s(c) = || P_{S_c} u_s || / || u_s ||

is computable from the unembedding and the averaged Jacobian alone, before any
image is shown.  So the prediction is a ranking: across six ways of answering the
same visual question -- English, Chinese, French, an English synonym, a letter
key, a digit key -- the fractional damage should follow g.

Nothing here is fitted to behaviour.  A rank correlation between a number read
off the weights and a number measured from the model's answers is the cleanest
evidence that the channel picture is doing real work.
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
ap.add_argument("--n", type=int, default=110)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/collateral")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lens = (jvlm.lens_from_checkpoint(a.lens) if a.lens.endswith(".ckpt")
        else __import__("jlens").lens.JacobianLens.load(a.lens))
sc = tasks.Scorer(m)
L = list(lens.source_layers)
MID = [l for l in L if 0.4 <= l / (m.n_layers - 1) <= 0.9]
print(m, lens, flush=True)

WORD_ROUTES = ["en", "zh", "fr", "syn"]
KEY_ROUTES = ["letter", "digit"]
def _single_token_ids(text_forms) -> list[int]:
    """Ids of the forms the tokenizer emits as one token, and only those."""
    out = set()
    for f in text_forms:
        enc = m.tokenizer.encode(f, add_special_tokens=False)
        if len(enc) == 1:
            out.add(enc[0])
    return sorted(out)


KEY_IDS: dict[str, dict[str, list[int]]] = {}
for _k, (_o1, _o2) in lexicon.ARBITRARY.items():
    # " 1" tokenizes as [space, 1] here, so first-token sets would put the
    # shared space token in BOTH options and clip the margin to zero.
    KEY_IDS[_k] = {o: _single_token_ids((o, f" {o}")) for o in (_o1, _o2)}
    _a, _b = set(KEY_IDS[_k][_o1]), set(KEY_IDS[_k][_o2])
    if not (_a and _b and not (_a & _b)):
        raise ValueError(f"key ids for {_k} are empty or overlapping: {KEY_IDS[_k]}")


def pullbacks(token_ids, layer):
    """Rows of the site-to-vocabulary channel for a set of tokens."""
    return readout.lens_vectors(m, lens, token_ids, layer, normalize=False).float()


def overlap(u: torch.Tensor, Q: torch.Tensor) -> float:
    """||P u|| / ||u||, averaged over the rows of u."""
    Q = Q.to(u.device, u.dtype)
    num = (u @ Q).norm(dim=-1)
    den = u.norm(dim=-1).clamp_min(1e-8)
    return float((num / den).mean())


items = cocodata.concept_items(a.n, min_area=0.08, seed=11, hard=True,
                               tokenizer=m.tokenizer)
rows, geo = [], []
t0 = time.time()
_basis: dict = {}

for k, it in enumerate(items):
    c, d = it["concept"], it["distractor"]
    routes = [r for r in WORD_ROUTES if lexicon.usable(c, d, r, m.tokenizer)]
    if "en" not in routes:
        continue
    im = corpus.load_image(it["path"])

    # the erased span: the concept's ENGLISH surface forms only
    en_ids = cocodata.concept_token_ids(m.tokenizer, c)
    if c not in _basis:
        spec, Qs = {}, {}
        for l in L:
            V = readout.lens_vectors(m, lens, en_ids, l)
            spec[l] = V
            Qs[l], _ = torch.linalg.qr(V.T.float())
        _basis[c] = (spec, Qs)
    spec, Qs = _basis[c]

    # ---- geometry, computed before any image is shown
    g_of = {}
    for r in routes:
        ids = (lexicon.route_token_ids(m.tokenizer, c, r)
               + lexicon.route_token_ids(m.tokenizer, d, r))
        g_of[r] = float(np.mean([overlap(pullbacks(ids, l), Qs[l]) for l in MID]))
    for r in KEY_ROUTES:
        ids = sorted({t for v in KEY_IDS[r].values() for t in v})
        g_of[r] = float(np.mean([overlap(pullbacks(ids, l), Qs[l]) for l in MID]))
    for r, gv in g_of.items():
        geo.append(dict(concept=c, route=r, g=gv))

    # ---- behaviour, clean and erased
    def build():
        plan, prompts = [], []
        for r in routes:
            x, y = lexicon.route_word(c, r), lexicon.route_word(d, r)
            ix = lexicon.route_token_ids(m.tokenizer, c, r)
            iy = lexicon.route_token_ids(m.tokenizer, d, r)
            cue = lexicon.ROUTE_CUE[r]
            for sign, (p_, q_, ip, iq) in ((1.0, (x, y, ix, iy)),
                                           (-1.0, (y, x, iy, ix))):
                plan.append((r, sign, ip, iq))
                prompts.append(tasks.word_prompt(p_, q_, cue))
        for r in KEY_ROUTES:
            o1, o2 = lexicon.ARBITRARY[r]
            for sign, (x, y) in ((1.0, (c, d)), (-1.0, (d, c))):
                plan.append((r, sign, KEY_IDS[r][o1], KEY_IDS[r][o2]))
                prompts.append(
                    f"Which of these appears in the image?\n({o1}) {x}\n({o2}) {y}\n"
                    f"Answer with a single character.")
        return plan, prompts

    plan, prompts = build()
    out = {}
    for arm, which in (("clean", None), ("erase", "c")):
        iv = (readout.NoIntervention() if which is None else
              readout.Intervention(m, spec, channel="all", mode="ablate", alpha=1.0))
        with iv:
            lg = sc.multi_last_logits(
                iv.set_batch(m.chat_batch(prompts, [im])))
        acc: dict[str, list] = {}
        for j, (r, sign, ip, iq) in enumerate(plan):
            acc.setdefault(r, []).append(sign * sc.id_margin(lg[j], ip, iq))
        out[arm] = {r: (float(np.mean(v)),
                        float(np.mean([x > 0 for x in v]))) for r, v in acc.items()}

    for r in out["clean"]:
        cm, cc = out["clean"][r]
        am, ac = out["erase"][r]
        rows.append(dict(i=k, concept=c, distractor=d, route=r, g=g_of[r],
                         clean_margin=cm, erased_margin=am,
                         clean_correct=cc, erased_correct=ac,
                         frac_loss=float((cm - am) / cm) if abs(cm) > 1e-6 else np.nan,
                         acc_loss=float(cc - ac)))

    if k % 10 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
pd.DataFrame(geo).to_parquet(f"{a.out}/{tag}_geometry.parquet")

agg = df.groupby("route").agg(
    n=("i", "count"), g=("g", "mean"),
    clean_correct=("clean_correct", "mean"),
    erased_correct=("erased_correct", "mean"),
    frac_loss=("frac_loss", "median"), acc_loss=("acc_loss", "mean")
).sort_values("g", ascending=False)
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", agg.round(3).to_string())

from scipy.stats import spearmanr  # noqa: E402

per_route = agg.reset_index()
rho_r, p_r = spearmanr(per_route.g, per_route.acc_loss)
# a "fraction of the margin lost" is only interpretable when the clean margin
# pointed the right way to begin with
ok = df.dropna(subset=["frac_loss"])
ok = ok[ok.clean_margin > 0]
rho_i, p_i = spearmanr(ok.g, ok.frac_loss)
summ = dict(repo=a.repo, n_items=int(df.i.nunique()),
            routes=list(per_route.route),
            rho_route_level=float(rho_r), p_route_level=float(p_r),
            rho_item_level=float(rho_i), p_item_level=float(p_i),
            n_item_pairs=int(len(ok)),
            minutes=round((time.time() - t0) / 60, 1))
print(f"\ngeometry vs damage: route level rho = {rho_r:+.3f} (p = {p_r:.3g}), "
      f"item level rho = {rho_i:+.3f} (p = {p_i:.3g})")
json.dump(summ, open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
