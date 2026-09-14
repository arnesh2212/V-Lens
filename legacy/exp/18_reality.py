"""Reality monitoring: can the model tell what it saw from what it was told?

The obvious design -- image carries A, text carries B, ask which was which --
has a confound that makes it worthless as a representational measure: for the
concept A, the text never contains its name, and for B it always does.  A probe
then reaches ceiling by detecting a token in the prompt rather than anything
about provenance.  (We ran it; it reads 0.99, and a random subspace of matched
dimension reads 0.96, which is the signature of a lexical shortcut.)

So the text is held *identical* across conditions.  Every trial asserts
"The scene also contains a {c}."  In half the trials the photograph really does
contain c, and in half it does not.  Nothing lexical distinguishes them; the only
difference is whether the picture backs the assertion up.  The question is then
exactly reality monitoring:

    probe      is "c is really visible" decodable -- from the whole residual,
               from its production-channel component, from the complement, and
               from a matched random subspace?
    behaviour  asked directly, can the model answer it?
    control    the same question about an object the text never mentioned,
               which it can certainly answer.
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
ap.add_argument("--concepts", type=int, default=16)
ap.add_argument("--per-concept", type=int, default=50, help="per side")
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/reality")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lens = (jvlm.lens_from_checkpoint(a.lens) if a.lens.endswith(".ckpt")
        else __import__("jlens").lens.JacobianLens.load(a.lens))
sc = tasks.Scorer(m)
LAYERS = [l for l in lens.source_layers
          if 0.2 <= l / (m.n_layers - 1) <= 0.95][::2]
print(m, lens, f"\nreading L{LAYERS[0]}..L{LAYERS[-1]}", flush=True)

rng = random.Random(101)
pool = cocodata.concept_items(8000, min_area=0.05, seed=3)
by_c: dict[str, list] = {}
for it in pool:
    by_c.setdefault(it["concept"], []).append(it)
concepts = sorted(by_c, key=lambda c: -len(by_c[c]))[: a.concepts]
all_ids = sorted({it["image_id"] for it in pool})
print("concepts:", concepts, flush=True)

ASSERT = "The scene also contains a {c}."
QUESTION = "Answer in one word: what is the overall mood of this scene?"
# The control has to differ from CHECK in exactly one thing -- whether the
# object is one the text asserted -- so it carries the same assertion prefix and
# the same wording.  And both are asked in both option orders, because an
# uncounterbalanced A/B question measures the model's letter bias as much as its
# answer.
CONTROL = ("The text above says the scene contains a {c}. Looking only at the "
           "photograph, is a {c} really visible?\n(A) {o1}\n(B) {o2}\n"
           "Answer with a single letter.")
YESNO = [("yes, it is really there", "no, only the text says so"),
         ("no, only the text says so", "yes, it is really there")]
CHECK = CONTROL

trials = []
for c in concepts:
    yes = [it["image_id"] for it in by_c[c]][: a.per_concept]
    no = [i for i in all_ids if c not in cocodata.present(i)]
    rng.shuffle(no)
    for iid in yes:
        trials.append(dict(concept=c, image_id=iid, really=1))
    for iid in no[: a.per_concept]:
        trials.append(dict(concept=c, image_id=iid, really=0))
rng.shuffle(trials)
print(f"{len(trials)} trials", flush=True)

H = {l: [] for l in LAYERS}
rows = []
t0 = time.time()
for k, tr in enumerate(trials):
    c = tr["concept"]
    im = corpus.load_image(str(cocodata.image_path(tr["image_id"])))
    pre = ASSERT.format(c=c)
    b = m.chat(f"{pre} {QUESTION}", [im])
    acts = readout.residuals(m, b, LAYERS)
    for l in LAYERS:
        H[l].append(acts[l][-1].cpu().numpy())

    # behaviour: the reality check, and a control question of identical format
    # about an object the text never mentioned
    here = sorted(cocodata.present(tr["image_id"]) & set(cocodata.SINGLE_WORD))
    ctl_obj = here[0] if here else None
    prompts, plan = [], []
    for oi, (o1, o2) in enumerate(YESNO):
        prompts.append(f"{pre}\n{CHECK.format(c=c, o1=o1, o2=o2)}")
        plan.append(("check", 1.0 if oi == 0 else -1.0))
    absent = rng.choice([x for x in cocodata.SINGLE_WORD if x not in here])
    for name, obj in (("ctl_present", ctl_obj), ("ctl_absent", absent)):
        if obj is None:
            continue
        for oi, (o1, o2) in enumerate(YESNO):
            prompts.append(f"{pre}\n{CONTROL.format(c=obj, o1=o1, o2=o2)}")
            plan.append((name, 1.0 if oi == 0 else -1.0))
    lg = sc.multi_last_logits(m.chat_batch(prompts, [im]))
    acc: dict[str, list] = {}
    for j, (name, sign) in enumerate(plan):
        # positive means "yes, it is really there", with the option order undone
        acc.setdefault(name, []).append(sign * sc.letter_margin(lg[j]))
    check = float(np.mean(acc["check"]))
    row = dict(concept=c, image_id=tr["image_id"], really=tr["really"],
               check_margin=check,
               check_correct=float((check > 0) == bool(tr["really"])))
    if "ctl_present" in acc:
        row["ctl_present"] = float(np.mean(acc["ctl_present"]) > 0)
        row["ctl_absent"] = float(np.mean(acc["ctl_absent"]) < 0)
    rows.append(row)
    if k % 100 == 0:
        print(f"  {k}/{len(trials)}  {time.time()-t0:.0f}s", flush=True)

beh = pd.DataFrame(rows)
beh.to_parquet(f"{a.out}/{tag}_behaviour.parquet")
ctl = float(np.nanmean([beh.get("ctl_present", pd.Series(dtype=float)).mean(),
                        beh.get("ctl_absent", pd.Series(dtype=float)).mean()]))
print(f"\nBEHAVIOUR  reality check {beh.check_correct.mean():.3f}   "
      f"plain presence question {ctl:.3f}   (n={len(beh)})", flush=True)

Hm = {l: np.stack(v) for l, v in H.items()}
np.savez_compressed(f"{a.out}/{tag}_acts.npz", **{f"L{l}": Hm[l] for l in LAYERS})
pd.DataFrame(trials).to_csv(f"{a.out}/{tag}_trials.csv", index=False)

# ------------------------------------------------------------------ probes
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

VOCAB = [cocodata.primary_token_id(m.tokenizer, x) for x in cocodata.SINGLE_WORD]


def cv_auc(X, y, seed=0):
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 6:
        return float("nan")
    s = np.zeros(len(y))
    for tr_i, te_i in StratifiedKFold(5, shuffle=True,
                                      random_state=seed).split(X, y):
        mu, sd = X[tr_i].mean(0), X[tr_i].std(0) + 1e-6
        clf = LogisticRegression(max_iter=400, C=0.1)
        clf.fit((X[tr_i] - mu) / sd, y[tr_i])
        s[te_i] = clf.decision_function((X[te_i] - mu) / sd)
    return float(roc_auc_score(y, s))


meta = pd.DataFrame(trials)
probe_rows = []
for l in LAYERS:
    Q = percept.channel_basis(m, lens, VOCAB, l, rank=a.rank).cpu().numpy()
    g = np.random.default_rng(l)
    Qr, _ = np.linalg.qr(g.standard_normal((Q.shape[0], Q.shape[1])))
    X = Hm[l]
    views = {"full": X, "jspace": X @ Q, "orthogonal": X - (X @ Q) @ Q.T,
             "random": X @ Qr[:, :Q.shape[1]]}
    for c in concepts:
        sel = (meta.concept == c).values
        y = meta.really[sel].astype(int).values
        for view, V in views.items():
            probe_rows.append(dict(layer=l, depth=l / (m.n_layers - 1),
                                   concept=c, view=view, dim=V.shape[1],
                                   n=int(sel.sum()), auc=cv_auc(V[sel], y)))
    print(f"  probed L{l}", flush=True)

pdf = pd.DataFrame(probe_rows)
pdf.to_parquet(f"{a.out}/{tag}_probes.parquet")
piv = pdf.groupby(["view", "layer"]).auc.mean().unstack("view")
piv.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", piv.round(3).to_string())
print("\nPOOLED:", pdf.groupby("view").auc.mean().round(4).to_dict())
json.dump({"repo": a.repo, "n_trials": len(trials), "rank": int(a.rank),
           "reality_check": float(beh.check_correct.mean()),
           "plain_presence": ctl,
           "pooled_auc": pdf.groupby("view").auc.mean().round(4).to_dict(),
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
