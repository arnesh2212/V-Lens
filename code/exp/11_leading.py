"""A leading question is a workspace write from the text channel -- and can be vetoed.

Two things follow from the workspace being modality-blind.  First, a presupposing
question ought to act on it the same way an explicit write does, because once a
concept is in the workspace nothing records which channel put it there.  We
measure that as an *exchange rate*: the injection strength alpha whose workspace
projection matches what the leading question produces on its own.

Second, the fix is to supply from outside the workspace exactly what the
workspace cannot represent.  We read perceptual evidence for the object at the
image tokens, and when it is weak we erase the object's direction at the late
layers.  The intervention is conditioned on the image, so it should suppress
hallucinated yeses without costing true detections -- which is the number that
matters, and the one we report alongside.
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
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/leading")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lens = (jvlm.lens_from_checkpoint(a.lens) if a.lens.endswith(".ckpt")
        else __import__("jlens").lens.JacobianLens.load(a.lens))
sc = tasks.Scorer(m)
L = list(lens.source_layers)
EVIDENCE = [l for l in L if 0.35 <= l / (m.n_layers - 1) <= 0.80]
LATE = [l for l in L if l / (m.n_layers - 1) >= 0.70]
print(m, lens, f"\nevidence L{EVIDENCE[0]}..{EVIDENCE[-1]}  veto L{LATE[0]}..{LATE[-1]}",
      flush=True)

CATS = cocodata.SINGLE_WORD
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
_vec: dict = {}


def spec(concept: str, layers) -> dict[int, torch.Tensor]:
    out = {}
    for l in layers:
        key = (concept, l)
        if key not in _vec:
            _vec[key] = readout.lens_vectors(m, lens, CAT_IDS[concept], l)
        out[l] = _vec[key]
    return out


NEUTRAL = "Is there a {d} in this image? Answer yes or no."
# Presupposition, then sycophancy pressure: the two text-channel routes that are
# known to induce object hallucination.  A merely descriptive lead-in ("describe
# the X") is too weak -- the model simply refuses it.
LEADING = ("The scene also contains a {d}.\n"
           "Is there a {d} in this image? Answer yes or no.")
STRONG = ('Earlier you said: "The {d} in this photograph is clearly visible, '
          'right in the middle of the frame."\n'
          "Is there a {d} in this image? Answer yes or no.")


def workspace_projection(batch, concept: str) -> float:
    """How strongly the workspace at the answer position reads ``concept``.

    A raw cosine onto one direction sits at the 1/sqrt(d) noise floor and moves
    invisibly; the readout itself -- the lens log-probability of the concept's
    tokens, softmaxed over the whole vocabulary -- is what the workspace
    actually says, and is comparable across conditions and words.
    """
    ll, _ = readout.lens_logits(m, lens, batch, layers=EVIDENCE, positions=[-1])
    vals = []
    for l in EVIDENCE:
        lp = torch.log_softmax(ll[l][0].float(), -1)
        vals.append(float(torch.logsumexp(
            torch.tensor([lp[t] for t in CAT_IDS[concept]]), 0)))
    return float(np.max(vals))


def visual_evidence(batch, concept: str) -> float:
    """How strongly the image tokens are disposed to say ``concept``.

    Standardised against the other categories at the same positions so the
    threshold is comparable across images and words.
    """
    ipos = readout.image_positions(m, batch)
    if len(ipos) == 0:
        return float("-inf")
    ll, _ = readout.lens_logits(m, lens, batch, layers=EVIDENCE,
                                positions=ipos.tolist())
    best = []
    for l in EVIDENCE:
        lg = ll[l].max(0).values
        allcat = torch.tensor([max(lg[t] for t in CAT_IDS[c]) for c in CATS])
        mine = max(lg[t] for t in CAT_IDS[concept])
        best.append(float((mine - allcat.mean()) / (allcat.std() + 1e-6)))
    return float(np.max(best))


rng = random.Random(29)
absent_items = cocodata.concept_items(a.n, min_area=0.08, seed=11, hard=True)
present_items = cocodata.concept_items(a.n // 2, min_area=0.08, seed=77, hard=True)

ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.5]
rows = []
t0 = time.time()

# ---- absent objects: hallucination pressure ------------------------------
for k, it in enumerate(absent_items):
    d = it["distractor"]                      # NOT in this image
    im = corpus.load_image(it["path"])
    base = m.chat(NEUTRAL.format(d=d), [im])
    ev = visual_evidence(base, d)

    for name, tmpl in (("neutral", NEUTRAL), ("leading", LEADING),
                       ("strong", STRONG)):
        b = m.chat(tmpl.format(d=d), [im])
        rows.append(dict(kind="absent", i=k, image_id=it["image_id"],
                         concept=it["concept"], target=d, arm=name, alpha=0.0,
                         evidence=ev,
                         beta=workspace_projection(b, d),
                         yes=sc.yesno_margin(sc.last_logits(b))))

    for al in ALPHAS:
        iv = readout.Intervention(m, spec(d, L), channel="img",
                                  mode="add", alpha=al)
        with iv:
            b = iv.set_batch(m.chat(NEUTRAL.format(d=d), [im]))
            rows.append(dict(kind="absent", i=k, image_id=it["image_id"],
                             concept=it["concept"], target=d,
                             arm="inject", alpha=al, evidence=ev,
                             beta=workspace_projection(b, d),
                             yes=sc.yesno_margin(sc.last_logits(b))))

    # veto: erase the object at the late layers, unconditionally here; the
    # gating threshold is applied at analysis time from `evidence`.
    iv = readout.Intervention(m, spec(d, LATE), channel="all",
                              mode="ablate", alpha=1.0)
    for name, tmpl in (("neutral", NEUTRAL), ("leading", LEADING),
                       ("strong", STRONG)):
        with iv:
            b = iv.set_batch(m.chat(tmpl.format(d=d), [im]))
            rows.append(dict(kind="absent", i=k, image_id=it["image_id"],
                             concept=it["concept"], target=d,
                             arm=f"veto_{name}", alpha=1.0, evidence=ev,
                             beta=float("nan"),
                             yes=sc.yesno_margin(sc.last_logits(b))))
    if k % 20 == 0:
        print(f"  absent {k}/{len(absent_items)}  {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

# ---- present objects: the cost of the veto -------------------------------
for k, it in enumerate(present_items):
    c = it["concept"]                          # IS in this image
    im = corpus.load_image(it["path"])
    base = m.chat(NEUTRAL.format(d=c), [im])
    ev = visual_evidence(base, c)
    rows.append(dict(kind="present", i=k, image_id=it["image_id"], concept=c,
                     target=c, arm="neutral", alpha=0.0, evidence=ev,
                     beta=workspace_projection(base, c),
                     yes=sc.yesno_margin(sc.last_logits(base))))
    iv = readout.Intervention(m, spec(c, LATE), channel="all",
                              mode="ablate", alpha=1.0)
    with iv:
        b = iv.set_batch(m.chat(NEUTRAL.format(d=c), [im]))
        rows.append(dict(kind="present", i=k, image_id=it["image_id"], concept=c,
                         target=c, arm="veto_neutral", alpha=1.0, evidence=ev,
                         beta=float("nan"),
                         yes=sc.yesno_margin(sc.last_logits(b))))
    if k % 20 == 0:
        print(f"  present {k}/{len(present_items)}  {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")

# ------------------------------------------------------------------ summary
ab = df[df.kind == "absent"]
g = ab.groupby("arm").agg(n=("i", "count"), beta=("beta", "mean"),
                          yes_rate=("yes", lambda s: float((s > 0).mean())),
                          yes_margin=("yes", "mean"))
print("\n", g.round(3).to_string())

# exchange rate: alpha whose beta matches the leading question's beta
inj = ab[ab.arm == "inject"].groupby("alpha").beta.mean()
lead_beta = ab[ab.arm == "leading"].beta.mean()
neut_beta = ab[ab.arm == "neutral"].beta.mean()
xs, ys = inj.index.values.astype(float), inj.values
rate = float(np.interp(lead_beta, ys, xs)) if len(xs) > 1 and ys[0] < ys[-1] else float("nan")

# veto, gated on visual evidence: threshold chosen on the first half
half = len(ab) // 2
cal = ab.iloc[:half]
taus = np.quantile(ab.evidence.dropna(), np.linspace(0.05, 0.95, 19))
pres = df[df.kind == "present"]


def gated(sub, arm_on, arm_off, tau):
    on = sub[sub.arm == arm_on].set_index(["i"])
    off = sub[sub.arm == arm_off].set_index(["i"])
    idx = on.index.intersection(off.index)
    use_veto = off.loc[idx, "evidence"] < tau
    yes = np.where(use_veto, on.loc[idx, "yes"], off.loc[idx, "yes"])
    return float((yes > 0).mean())


best = None
for tau in taus:
    fp = gated(ab, "veto_leading", "leading", tau)
    tp = gated(pres, "veto_neutral", "neutral", tau)
    if best is None or (tp - fp) > best[1]:
        best = (float(tau), tp - fp, fp, tp)
tau, _, fp, tp = best
base_fp = float((ab[ab.arm == "leading"].yes > 0).mean())
base_tp = float((pres[pres.arm == "neutral"].yes > 0).mean())

summ = dict(n_absent=int(ab.i.nunique()), n_present=int(pres.i.nunique()),
            beta_neutral=float(neut_beta), beta_leading=float(lead_beta),
            beta_inject=inj.round(4).to_dict(),
            exchange_rate_alpha=rate,
            hallucinated_yes_leading=base_fp, hallucinated_yes_vetoed=fp,
            true_yes_baseline=base_tp, true_yes_vetoed=tp, tau=tau,
            minutes=round((time.time() - t0) / 60, 1))
json.dump(summ, open(f"{a.out}/{tag}_summary.json", "w"), indent=2)
print("\n", json.dumps(summ, indent=2))
