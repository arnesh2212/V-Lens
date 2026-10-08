"""Where was the hallucination funded?

A reading of this lens is in logits, and summed over every image position the
readings equal the change in a token's logit caused by replacing the picture
with the per-position corpus mean.  That is the completeness property, and it
buys a sentence no similarity map can write: for a wrong answer, *the image's
total contribution to the answer the model gave* is a signed number, so a wrong
answer can be classified by who paid for it.

For each absent-object question on a POPE-style benchmark rebuilt from COCO we
take the answer token the model actually produced -- it says "yes" to an object
that is not in the photograph -- and price the picture's contribution to the
yes-over-no margin at the answer position:

    total  =  sum over image positions of  (h - h_bar) . mean gradient of the
              margin along the straight path from h_bar to h

Two regimes follow, and they are exhaustive because the number has a sign:

    total > 0    the picture raised the margin toward "yes".  The image funded
                 the error.  A MISPERCEPTION.
    total <= 0   the picture pushed toward "no" and the model said yes anyway.
                 The language prior outvoted the eyes.  An OUTVOTED PERCEPTION.

CHOOSING THE LAYER.  The reading is defined at a layer, and what it prices is
what the image *positions* still contribute from that layer on.  A pilot on
fourteen photographs settled this before any census was computed, and settled
it against the first plan: at the middle of the stack, replacing the whole
picture with the baseline moves the yes-over-no margin by only 0.3 logits on
average, because by then the picture's case has already been copied into the
text positions and taking the image away changes almost nothing.  A number that
small is under the closure residual, and the run refuses to report a census
built on it.  So the run begins with a depth scan -- for a sample of questions,
how much the margin actually moves when the image is replaced, at every depth --
and prices the census at the earliest layer the reading is defined at, one past
the DeepStack injection, where the image positions still hold the whole of the
image's contribution.  The scan is printed, so the choice can be checked; it
looks only at effect size and closure, never at which answers were wrong.

VERIFYING COMPLETENESS ON THESE ITEMS.  The claim rests entirely on closure, so
the r = 0.96 measured on caption prompts is not imported.  Per item, the sum of
the readings is checked against the true change in the margin from an ablation
forward pass -- correlation, relative error, and the only thing the census
actually needs, whether the two agree on the SIGN.  On a subsample the same
check is repeated across a vocabulary of object words, which is the form
exp/59_complete.py reports.  If sign agreement is poor the run says the census
is not measurable and does not report it.

TWO BASELINES.  The corpus mean is an averaged, off-manifold image, and a model
shown one might refuse everything, which would make every error look funded.
Every item is therefore also priced against three *real* photographs standing in
for this one, and both answers are reported side by side.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter

sys.path.insert(0, "code")
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch

import jvlm
from jvlm import budget, cocodata, corpus, readout, stats as st, vmap
from jlens.hooks import ActivationRecorder

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--images", type=int, default=500)
ap.add_argument("--mean-images", type=int, default=120)
ap.add_argument("--layer", type=int, default=-1,
                help="-1 = the earliest layer the reading is defined at")
ap.add_argument("--extra-layers", type=int, nargs="*", default=None)
ap.add_argument("--extra-every", type=int, default=6)
ap.add_argument("--scan-n", type=int, default=24)
ap.add_argument("--scan-layers", type=int, default=8)
ap.add_argument("--ig-steps", type=int, default=8)
ap.add_argument("--ig-chunk", type=int, default=2)
ap.add_argument("--comp-n", type=int, default=60)
ap.add_argument("--comp-vocab", type=int, default=24)
ap.add_argument("--pool", type=int, default=2)
ap.add_argument("--max-cards", type=int, default=400)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/census")
ap.add_argument("--tag", default=None)
ap.add_argument("--resume", action="store_true",
                help="pick up from whatever is already on disk")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
LAST = m.n_layers - 1
LO = max(m.n_deepstack, 1)          # the earliest layer a reading is defined at
SCAN = sorted({*range(LO, LAST, max(1, (LAST - LO) // a.scan_layers)), LAST - 1})
CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}


def ids_of(w):
    out = []
    for f in (w, " " + w, w.capitalize(), " " + w.capitalize()):
        t = m.tokenizer.encode(f, add_special_tokens=False)
        if len(t) == 1:
            out.append(t[0])
    return sorted(set(out))


YES, NO = ids_of("yes"), ids_of("no")


def canonical(obj):
    """One id for the object word: " obj" where the tokenizer emits it whole.

    The census only needs the yes-over-no margin, and every extra vocabulary
    item is another backward pass through most of the stack.  The object's own
    reading rides along as a secondary column, so it is taken on the canonical
    surface form rather than on all four of them.
    """
    want = m.tokenizer.encode(f" {obj}", add_special_tokens=False)
    if len(want) == 1 and want[0] in CAT_IDS[obj]:
        return [want[0]]
    return CAT_IDS[obj][:1]
print(m, f"\nscan layers {SCAN}; yes={len(YES)} no={len(NO)}", flush=True)


def prep(path, q):
    return m.chat(q, [corpus.load_image(path, square=a.square)])


@torch.no_grad()
def margin_under(b, repl, ip, yi, ni, layer):
    """The yes-over-no margin with the image positions replaced by ``repl``.

    ``vmap.completeness`` answers the same question but re-runs the clean pass
    every time it is called; the scan and the donor contrast below need many
    baselines against one clean pass, so they get their own forward.
    """
    hd = None
    if repl is not None:
        pos = torch.as_tensor(ip, dtype=torch.long)

        def hook(mod, inp, o, _v=repl, _p=pos):
            tup = not torch.is_tensor(o)
            hh = (o[0] if tup else o).clone()
            hh[0, _p.to(hh.device)] = _v.to(hh.device).to(hh.dtype)
            return (hh, *o[1:]) if tup else hh

        hd = m.layers[layer].register_forward_hook(hook)
    try:
        with ActivationRecorder(m.layers, at=[LAST]) as rec:
            m.forward(b)
            lg = m.unembed(rec.activations[LAST][0, -1][None])[0].float()
            return float(lg[yi] - lg[ni])
    finally:
        if hd is not None:
            hd.remove()


# ---- the baseline: the per-position corpus mean ------------------------------
# Image tokens precede the question under a causal mask, so their residuals do
# not depend on the prompt; a mean collected under "Describe this image." is the
# same mean a POPE prompt would give, and the run checks that on the first item.
pool = cocodata.concept_items(6000, min_area=0.04, seed=7)
b0 = prep(pool[0]["path"], "Describe this image.")
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
C = -(-G // a.pool)
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in SCAN}
bank = {l: [] for l in SCAN}
nm = 0
for it in pool[: a.mean_images]:
    b = prep(it["path"], "Describe this image.")
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    A = readout.residuals(m, b, SCAN)
    for l in SCAN:
        v = A[l][ip].float().cpu()
        acc[l] += v.double()
        if len(bank[l]) < 24:
            bank[l].append(v)
    nm += 1
MEANS = {l: (acc[l] / max(nm, 1)).float() for l in SCAN}
print(f"means from {nm}; {P} tokens on {G}x{G}, pooled to {C}x{C} "
      f"({time.time()-t0:.0f}s)", flush=True)

bq = prep(pool[0]["path"], "Is there a zebra in this image?  Answer yes or no.")
if len(readout.image_positions(m, bq)) == P:
    d0 = readout.residuals(m, b0, [SCAN[0]])[SCAN[0]][IPOS].float()
    d1 = readout.residuals(m, bq, [SCAN[0]])[SCAN[0]][
        readout.image_positions(m, bq)].float()
    print("image residuals identical across the two prompts: "
          f"max|diff| = {float((d0-d1).abs().max()):.2e}", flush=True)


def pool_map(v: np.ndarray) -> np.ndarray:
    """Sum the per-position readings into C x C regions.  Sum, not mean: the
    readings are a budget, so a region's price is its patches' prices added."""
    a2 = np.zeros(C * a.pool * C * a.pool, dtype=float)
    a2[:len(v)] = v
    return a2.reshape(C, a.pool, C, a.pool).sum((1, 3)).reshape(-1)


# ---- the questions: absent objects only, three flavours ----------------------
freq = Counter()
for it in pool:
    freq.update(c for c in it["present"] if c in CATS)
POPULAR = [c for c, _ in freq.most_common()]
rng = np.random.default_rng(4242)
imgs = cocodata.concept_items(a.images, min_area=0.05, seed=31,
                              tokenizer=m.tokenizer)
qs = []
# concept_items samples (photograph, category) pairs, so one photograph can come
# back several times; a census of proportions must not count a question twice
asked = set()
for it in imgs:
    here = sorted(c for c in it["present"] if c in CATS)
    absent = [c for c in CATS if c not in set(it["present"])]
    if not here or len(absent) < 6:
        continue
    coocc = Counter()
    for h in here:
        coocc.update(cocodata.cooccurring(h))
    adv = [c for c, _ in coocc.most_common() if c in set(absent)]
    pop = next((c for c in POPULAR if c in set(absent)), None)
    for obj, split in ((rng.choice(absent), "random"), (pop, "popular"),
                       (adv[0] if adv else None, "adversarial")):
        key = (int(it["image_id"]), str(obj))
        if obj and key not in asked:
            asked.add(key)
            qs.append((it, str(obj), split))
print(f"{len(qs)} absent-object questions over "
      f"{len({int(i['image_id']) for i, _, _ in qs})} photographs", flush=True)


def ask(it, obj):
    """Everything one question needs before any reading is taken."""
    q = f"Is there a {obj} in this image?  Answer yes or no."
    b = prep(it["path"], q)
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        return None
    A = readout.residuals(m, b, [LAST])
    lg = m.unembed(A[LAST][-1:].to(m.device))[0].float()
    yi = YES[int(lg[YES].argmax())]
    ni = NO[int(lg[NO].argmax())]
    return dict(q=q, b=b, ip=ip, yi=yi, ni=ni, conf=float(lg[yi] - lg[ni]),
                top1=int(lg.argmax()))


# the chunked path integral is the same arithmetic as the batched one; a wiring
# bug would show here as a large number, so it is measured rather than assumed
dchk = ask(qs[0][0], qs[0][1])
if dchk is not None:
    print("chunked path == batched path: max|diff| = "
          f"{budget.agrees(m, dchk['b'], [dchk['yi'], dchk['ni']], LO, dchk['ip'].tolist(), MEANS[LO]):.2e}",
          flush=True)

# ---- the depth scan: where does the image's contribution still live? ---------
scan = []
SCANF = f"{a.out}/{tag}_scan.parquet"
if a.resume and os.path.exists(SCANF):
    scan = pd.read_parquet(SCANF).to_dict("records")
    print(f"scan reused from disk ({len(scan)} rows)", flush=True)
for k, (it, obj, split) in enumerate([] if scan else qs[: a.scan_n]):
    d = ask(it, obj)
    if d is None:
        continue
    for l in SCAN:
        scan.append(dict(i=k, layer=l, depth=round(l / LAST, 3),
                         delta=d["conf"] - margin_under(d["b"], MEANS[l],
                                                        d["ip"].tolist(),
                                                        d["yi"], d["ni"], l)))
sdf = pd.DataFrame(scan)
sdf.to_parquet(SCANF)      # saved now, so a killed run does not redo it
prof = (sdf.assign(a=lambda x: x.delta.abs())
        .groupby(["layer", "depth"]).a.mean().reset_index()
        .rename(columns={"a": "mean_abs_delta"}))
prof["retained"] = (prof.mean_abs_delta / prof.mean_abs_delta.max()).round(3)
print("\nHOW MUCH OF THE ANSWER THE IMAGE POSITIONS STILL CARRY, BY DEPTH\n"
      "(mean |change in the yes-over-no margin| when the picture is replaced "
      f"by the baseline at that layer; n = {sdf.i.nunique()} questions)")
print(prof.round(3).to_string(index=False), flush=True)

L = a.layer if a.layer >= 0 else LO
EXTRA = (a.extra_layers if a.extra_layers is not None
         else [l for l in (SCAN[len(SCAN) // 2], SCAN[-1]) if l != L])
print(f"\npricing the census at layer {L} (depth {L/LAST:.2f}); "
      f"robustness at {EXTRA}", flush=True)
for l in set([L, *EXTRA]) - set(SCAN):
    raise SystemExit(f"layer {l} has no baseline; add it to the scan")

# ---- the census --------------------------------------------------------------
COMP_EVERY = max(1, len(qs) // max(a.comp_n, 1))
rows, deep, cards = [], [], []
done = set()
if a.resume:
    # this card is shared and a run can be killed out from under it, so the
    # loop picks up from the part-written files rather than starting again
    for f, sink in ((f"{a.out}/{tag}_raw.parquet", "rows"),
                    (f"{a.out}/{tag}_depth.parquet", "deep")):
        if os.path.exists(f):
            recs = pd.read_parquet(f).to_dict("records")
            (rows if sink == "rows" else deep).extend(recs)
    if os.path.exists(f"{a.out}/{tag}_cards.json"):
        cards = json.load(open(f"{a.out}/{tag}_cards.json"))
    done = {int(r["i"]) for r in rows}
    print(f"resuming: {len(rows)} questions, {len(cards)} cards already on disk",
          flush=True)


def save():
    pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")
    if deep:
        pd.DataFrame(deep).to_parquet(f"{a.out}/{tag}_depth.parquet")
    json.dump(cards, open(f"{a.out}/{tag}_cards.json", "w"))


for k, (it, obj, split) in enumerate(qs):
    if k in done:
        continue
    d = ask(it, obj)
    if d is None:
        continue
    b, ip, yi, ni, conf = d["b"], d["ip"], d["yi"], d["ni"], d["conf"]
    said = "yes" if conf > 0 else "no"
    oid = canonical(obj)
    ids = [yi, ni] + oid
    sc = budget.ig_map_chunked(m, b, ids, L, ip.tolist(), MEANS[L],
                               steps=a.ig_steps, chunk=a.ig_chunk)
    margin = (sc[0] - sc[1]).numpy()
    comp = vmap.completeness(m, b, ids, L, ip.tolist(), MEANS[L], sc)
    tr, cl = comp["truth"], comp["claim"]
    dm = [margin_under(b, bank[L][(k + j) % len(bank[L])], ip.tolist(), yi, ni, L)
          for j in range(3)]
    tiles = pool_map(margin)
    rec = dict(i=k, image_id=int(it["image_id"]), obj=obj, split=split,
               said=said, hallucinated=int(said == "yes"), conf=conf,
               layer=L, depth=round(L / LAST, 3),
               yesno=int(d["top1"] in YES or d["top1"] in NO),
               total=float(margin.sum()),
               truth=float(tr[0] - tr[1]),
               claim=float(cl[0] - cl[1]),
               truth_donor=float(conf - float(np.mean(dm))),
               base_donor=float(np.mean(dm)),
               base_donor_sd=float(np.std(dm)),
               yes_total=float(sc[0].sum()), no_total=float(sc[1].sum()),
               yes_truth=float(tr[0]), no_truth=float(tr[1]),
               obj_total=float(max(sc[2 + j].sum() for j in range(len(oid)))),
               obj_truth=float(max(tr[2 + j] for j in range(len(oid)))),
               top_region=float(tiles.max()), bot_region=float(tiles.min()))
    rec["base_margin"] = rec["conf"] - rec["truth"]
    rows.append(rec)

    if k % COMP_EVERY == 0 and sum("comp_corr" in r for r in rows) < a.comp_n:
        r2 = np.random.default_rng(900 + k)
        words = [obj] + list(r2.choice([w for w in CATS if w != obj],
                                       size=min(a.comp_vocab - 1, len(CATS) - 1),
                                       replace=False))
        vids = sorted({t for w in words for t in CAT_IDS[w]})[: 4 * a.comp_vocab]
        vsc = budget.ig_map_chunked(m, b, vids, L, ip.tolist(), MEANS[L],
                                    steps=a.ig_steps, chunk=a.ig_chunk)
        vc = vmap.completeness(m, b, vids, L, ip.tolist(), MEANS[L], vsc)
        rec["comp_corr"] = vc["corr"]
        rec["comp_rel"] = vc["rel_error"]
        rec["comp_k"] = len(vids)

    if k % a.extra_every == 0 and EXTRA:
        for l2 in EXTRA:
            s2 = budget.ig_map_chunked(m, b, [yi, ni], l2, ip.tolist(),
                                       MEANS[l2], steps=a.ig_steps,
                                       chunk=a.ig_chunk)
            c2 = vmap.completeness(m, b, [yi, ni], l2, ip.tolist(), MEANS[l2],
                                   s2)
            deep.append(dict(i=k, layer=l2, depth=round(l2 / LAST, 3),
                             hallucinated=rec["hallucinated"], conf=conf,
                             total=float((s2[0] - s2[1]).sum()),
                             truth=float(c2["truth"][0] - c2["truth"][1])))

    # every hallucination gets a card -- the worked rows are chosen from them by
    # a stated rule -- plus a handful of correct refusals for contrast
    n_ref = sum(1 for cc in cards if cc["said"] == "no")
    if len(cards) < a.max_cards and (said == "yes"
                                     or (n_ref < 30 and k % 17 == 0)):
        cards.append(dict(image_id=int(it["image_id"]), path=it["path"],
                          obj=obj, split=split, question=d["q"], said=said,
                          conf=conf, total=rec["total"], truth=rec["truth"],
                          truth_donor=rec["truth_donor"],
                          base_margin=rec["base_margin"], coarse=C, pool=a.pool,
                          grid=G, tiles=[round(float(x), 4) for x in tiles]))
    if k % 25 == 0:
        el = time.time() - t0
        nleft = sum(1 for j in range(k + 1, len(qs)) if j not in done)
        print(f"  {k}/{len(qs)}  {len(rows)} kept  {el:.0f}s  "
              f"eta {el/max(len(rows)-len(done), 1)*nleft/60:.1f}m", flush=True)
        save()

save()
df = pd.DataFrame(rows)

# ---- 1. does the arithmetic close on THESE items? ----------------------------
res = (df.claim - df.truth).abs()
denom = df.truth.abs().mean()
closure = {
    "n": int(len(df)),
    "corr_margin": float(np.corrcoef(df.claim, df.truth)[0, 1]),
    "rel_error_margin": float(res.mean() / max(denom, 1e-9)),
    "median_abs_residual": float(res.median()),
    "sign_agree": float((np.sign(df.claim) == np.sign(df.truth)).mean()),
    "mean_abs_truth": float(denom),
    "greedy_is_yes_or_no": float(df.yesno.mean()),
    "base_margin_mean": float(df.base_margin.mean()),
    "base_margin_donor": float(df.base_donor.mean()),
    "clean_margin_mean": float(df.conf.mean()),
}
# The two baselines can disagree about who funded an error, so the run measures
# which of them carries information about the item.  Substituting an unrelated
# photograph does not remove this scene's evidence, it replaces the scene; if
# the margin it produces barely varies from question to question, then "the
# image funded it" against that baseline means little more than "the model's
# margin here is above average", and the number says so.
baselines = {
    "sd_across_items_mean_baseline": float(df.base_margin.std()),
    "sd_across_items_donor_baseline": float(df.base_donor.std()),
    "sd_within_item_across_donors": float(df.base_donor_sd.mean()),
    "corr_donor_vs_mean_baseline": float(np.corrcoef(df.base_donor,
                                                     df.base_margin)[0, 1]),
    "corr_donor_baseline_vs_conf": float(np.corrcoef(df.base_donor,
                                                     df.conf)[0, 1]),
    "corr_mean_baseline_vs_conf": float(np.corrcoef(df.base_margin,
                                                    df.conf)[0, 1]),
    "sd_clean_margin": float(df.conf.std()),
}
closure["baselines"] = baselines
cm = df.dropna(subset=["comp_corr"]) if "comp_corr" in df.columns else df.iloc[:0]
if len(cm):
    closure["vocab_n_items"] = int(len(cm))
    closure["vocab_k_words"] = int(cm.comp_k.median())
    closure["vocab_corr_mean"] = float(cm.comp_corr.mean())
    closure["vocab_corr_median"] = float(cm.comp_corr.median())
    closure["vocab_rel_error"] = float(cm.comp_rel.mean())
TAU = float(res.median())          # the measurement's own noise floor
big = df[df.truth.abs() > TAU]
closure["sign_agree_above_tau"] = float(
    (np.sign(big.claim) == np.sign(big.truth)).mean()) if len(big) else float("nan")
closure["n_above_tau"] = int(len(big))
# a sign the reading gets wrong on an item whose true effect is under the
# residual is not a failure of the method, it is the method saying "too close
# to call"; the gate is therefore about the items where there is an effect
OK = (closure["sign_agree_above_tau"] >= 0.90
      and closure["corr_margin"] >= 0.90)

print("\nDOES THE ARITHMETIC CLOSE ON THE POPE ITEMS?")
for k2, v in closure.items():
    if k2 != "baselines":
        print(f"  {k2:24s} {v}")
print("\nWHAT EACH BASELINE IS")
for k2, v in baselines.items():
    print(f"  {k2:32s} {v:+.3f}")

h = df[df.hallucinated == 1]
c = df[df.hallucinated == 0]


def split_of(v, tau):
    v = np.asarray(v, float)
    if not len(v):
        return {k: float("nan") for k in
                ("funded", "outvoted", "funded_clear", "near_zero",
                 "against_clear", "median", "mean")} | {"n": 0}
    return dict(n=int(len(v)),
                funded=float((v > 0).mean()),
                outvoted=float((v <= 0).mean()),
                funded_clear=float((v > tau).mean()),
                near_zero=float((np.abs(v) <= tau).mean()),
                against_clear=float((v < -tau).mean()),
                median=float(np.median(v)),
                mean=float(np.mean(v)))


def auc(pos, neg):
    """Rank-based separation, the same estimator exp/45_pope.py reports."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    v = np.concatenate([pos, neg])
    o = v.argsort()
    r = np.empty(len(v), float)
    r[o] = np.arange(len(v))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2)
                 / (len(pos) * len(neg)))


# If the census says the prior outvoted the eyes, then the prior's own answer --
# what the model says to this question with a contentless image in front of it --
# should already predict which questions it will get wrong, before the
# photograph is consulted at all.  That is a number, so it is measured.
predicts = {
    "prior_margin_hallucinated": float(h.base_margin.mean()) if len(h) else float("nan"),
    "prior_margin_refused": float(c.base_margin.mean()) if len(c) else float("nan"),
    "auroc_prior_margin": auc(h.base_margin, c.base_margin),
    "auroc_image_contribution": auc(h.total, c.total),
    "auroc_own_margin_degenerate": auc(h.conf, c.conf),
}
print("\nWHAT THE PRIOR ALONE ALREADY KNOWS")
for k2, v in predicts.items():
    print(f"  {k2:32s} {v:+.3f}")

census = {"tau": TAU,
          "hallucinations": {"reading": split_of(h.total, TAU),
                             "ablation": split_of(h.truth, TAU),
                             "ablation_donor": split_of(h.truth_donor, TAU)},
          "correct_refusals": {"reading": split_of(c.total, TAU),
                               "ablation": split_of(c.truth, TAU)},
          "by_split": {s: split_of(h[h["split"] == s].total, TAU)
                       for s in sorted(h["split"].unique())},
          "yes_rate": {s: float(df[df["split"] == s].hallucinated.mean())
                       for s in sorted(df["split"].unique())}}
print("\nTHE FUNDING CENSUS  (n hallucinations = "
      f"{len(h)} of {len(df)} absent-object questions)")
for nm2, dd in (("reading", census["hallucinations"]["reading"]),
                ("ablation", census["hallucinations"]["ablation"]),
                ("ablation (donor)", census["hallucinations"]["ablation_donor"])):
    print(f"  {nm2:18s} image-funded {dd['funded']:.3f}   "
          f"outvoted {dd['outvoted']:.3f}   median {dd['median']:+.3f}")
print("  correct refusals   image-funded "
      f"{census['correct_refusals']['reading']['funded']:.3f}  "
      "(a picture that agrees with a correct no reads as negative here)")
if not OK:
    print("\n*** CLOSURE FAILED ON THESE ITEMS -- the census above is not "
          "measurable and must not be reported. ***")

dep = pd.DataFrame(deep) if deep else pd.DataFrame()
depth_tab = []
if len(dep):
    for l2, g in dep.groupby("layer"):
        gh = g[g.hallucinated == 1]
        depth_tab.append(dict(
            layer=int(l2), depth=float(g.depth.iloc[0]), n=int(len(gh)),
            funded=float((gh.total > 0).mean()) if len(gh) else float("nan"),
            mean_abs_truth=float(g.truth.abs().mean()),
            sign_agree=float((np.sign(g.total) == np.sign(g.truth)).mean()),
            corr=float(np.corrcoef(g.total, g.truth)[0, 1])))
    print("\nROBUSTNESS ACROSS DEPTH (subsample)")
    print(pd.DataFrame(depth_tab).round(3).to_string(index=False))

out = {"repo": a.repo, "layer": L, "depth": round(L / LAST, 3),
       "n_layers": m.n_layers, "n_deepstack": m.n_deepstack, "P": P, "grid": G,
       "coarse": C, "ig_steps": a.ig_steps, "scan_layers": SCAN,
       "depth_profile": prof.round(4).to_dict("records"),
       "n_images": int(df.image_id.nunique()),
       "n_questions": int(len(df)), "n_hallucinated": int(len(h)),
       "closure": closure, "closure_ok": bool(OK), "census": census,
       "predicts": predicts,
       "by_depth": depth_tab, "minutes": round((time.time() - t0) / 60, 1)}
json.dump(st.jsonable(out), open(f"{a.out}/{tag}_stats.json", "w"), indent=2,
          default=float)
print("\ndone", flush=True)
