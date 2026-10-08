"""Is the workspace modality-blind?

Matched-cue design.  Every trial shows an image containing concept A and a
sentence naming concept B, which is absent from the picture.  So both concepts
are active in the same forward pass, one having arrived through the image and
one through the text, and the lexical content of the context is balanced across
the two labels once concepts are pooled.

For each concept we train a linear probe at the final prompt position to answer
"did this concept reach me through the image or through the text?", and we do it
four times over:

    full        the whole residual stream
    J-space     its projection onto the span of the concept vocabulary's
                J-lens vectors  -- the verbalizable workspace
    orthogonal  the remainder
    random      a matched-dimension random subspace (the capacity control,
                without which a low AUC inside J-space proves nothing)

The same four readouts are also scored on a CONTENT probe -- is this concept
active at all -- which is the positive control: J-space should be content-rich.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time

sys.path.insert(0, "code")
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import jvlm
from jvlm import cocodata, corpus, readout

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--concepts", type=int, default=16)
ap.add_argument("--per-concept", type=int, default=60)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/provenance")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lens = (jvlm.lens_from_checkpoint(a.lens) if a.lens.endswith(".ckpt")
        else __import__("jlens").lens.JacobianLens.load(a.lens))
print(m, lens, flush=True)

LAYERS = [l for l in lens.source_layers
          if 0.2 <= l / (m.n_layers - 1) <= 0.95][::2]
rng = random.Random(5)

# ------------------------------------------------------------------ trials
# Concepts with enough large-area COCO images to fill both roles.
pool = cocodata.concept_items(6000, min_area=0.03, seed=2)
by_concept: dict[str, list] = {}
for it in pool:
    by_concept.setdefault(it["concept"], []).append(it)
concepts = sorted(by_concept, key=lambda c: -len(by_concept[c]))[: a.concepts]
print("concepts:", concepts, flush=True)

CUE = "The scene also contains a {}."
QUESTION = "Answer in one word: what is the overall mood of this scene?"

SOURCE_Q = (
    "One of these you can actually see in the photograph; the other you were "
    "only told about in the text.\nWhich one can you actually see?\n"
    "(A) {}\n(B) {}\nAnswer with a single letter."
)
CONTENT_Q = (
    "Which of these is part of the scene?\n(A) {}\n(B) {}\n"
    "Answer with a single letter."
)

trials = []
for A in concepts:
    imgs = by_concept[A][: a.per_concept]
    for it in imgs:
        B = rng.choice([c for c in concepts
                        if c != A and c not in it["present"]])
        trials.append(dict(image_id=it["image_id"], path=it["path"],
                           seen=A, told=B))
rng.shuffle(trials)
print(f"{len(trials)} trials", flush=True)

# ------------------------------------------------------------- collect acts
H = {l: [] for l in LAYERS}
meta = []
t0 = time.time()
for k, tr in enumerate(trials):
    im = corpus.load_image(tr["path"])
    prompt = CUE.format(tr["told"]) + " " + QUESTION
    b = m.chat(prompt, [im])
    acts = readout.residuals(m, b, LAYERS)
    for l in LAYERS:
        H[l].append(acts[l][-1].cpu().numpy())
    meta.append(tr)
    if k % 100 == 0:
        print(f"  {k}/{len(trials)}  {time.time()-t0:.0f}s", flush=True)

# ----------------------------------------------------- behavioural readout
# The probe result is only half an argument.  If provenance really is missing
# from the workspace, the model should be unable to USE it -- so we ask it
# directly, and pair that against a content question of identical format whose
# answer is in the workspace.  A gap between the two is the behavioural
# signature of information the model holds but cannot consult.
from jvlm import tasks  # noqa: E402

sc = tasks.Scorer(m)
brows = []
for k, tr in enumerate(trials[: min(len(trials), 400)]):
    im = corpus.load_image(tr["path"])
    A, B = tr["seen"], tr["told"]
    C = rng.choice([c for c in concepts if c not in (A, B)])
    pre = CUE.format(B)

    def ask(q, image=im, pre=pre):
        return sc.last_logits(m.chat(pre + "\n" + q, [image]))

    src, cont = [], []
    for order, (x, y) in enumerate([(A, B), (B, A)]):
        sign = 1.0 if order == 0 else -1.0
        src.append(sign * sc.letter_margin(ask(SOURCE_Q.format(x, y))))
    for order, (x, y) in enumerate([(A, C), (C, A)]):
        sign = 1.0 if order == 0 else -1.0
        cont.append(sign * sc.letter_margin(ask(CONTENT_Q.format(x, y))))
    brows.append(dict(image_id=tr["image_id"], seen=A, told=B, foil=C,
                      source_margin=float(np.mean(src)),
                      source_correct=float(np.mean([v > 0 for v in src])),
                      content_margin=float(np.mean(cont)),
                      content_correct=float(np.mean([v > 0 for v in cont]))))
    if k % 100 == 0:
        print(f"  behaviour {k}  {time.time()-t0:.0f}s", flush=True)

bdf = pd.DataFrame(brows)
bdf.to_parquet(f"{a.out}/{tag}_behaviour.parquet")
print("\nBEHAVIOUR  source_correct=%.3f  content_correct=%.3f  (n=%d)" % (
    bdf.source_correct.mean(), bdf.content_correct.mean(), len(bdf)), flush=True)

H = {l: np.stack(v) for l, v in H.items()}
meta = pd.DataFrame(meta)
np.savez_compressed(f"{a.out}/{tag}_acts.npz",
                    **{f"L{l}": H[l] for l in LAYERS})
meta.to_csv(f"{a.out}/{tag}_trials.csv", index=False)

# --------------------------------------------------------------- subspaces
cat_ids = [cocodata.primary_token_id(m.tokenizer, c) for c in cocodata.SINGLE_WORD]


def basis_for(layer: int) -> np.ndarray:
    """Orthonormal basis of the concept vocabulary's J-space at ``layer``."""
    V = readout.lens_vectors(m, lens, cat_ids, layer)          # [K, d]
    Q, _ = torch.linalg.qr(V.T.float().cpu())                  # [d, k]
    return Q.numpy()


def random_basis(d: int, k: int, seed: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(g.standard_normal((d, k)))
    return Q[:, :k]


def cv_auc(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 6:
        return float("nan")
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    s = np.zeros(len(y))
    for tr_i, te_i in skf.split(X, y):
        mu, sd = X[tr_i].mean(0), X[tr_i].std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=0.1)
        clf.fit((X[tr_i] - mu) / sd, y[tr_i])
        s[te_i] = clf.decision_function((X[te_i] - mu) / sd)
    return float(roc_auc_score(y, s))


rows = []
for l in LAYERS:
    Xf = H[l]
    Qj = basis_for(l)
    k = Qj.shape[1]
    Qr = random_basis(Xf.shape[1], k, seed=l)
    views = {
        "full": Xf,
        "jspace": Xf @ Qj,
        "orthogonal": Xf - (Xf @ Qj) @ Qj.T,
        "random": Xf @ Qr,
    }
    for c in concepts:
        # PROVENANCE: c arrived via image (seen) vs via text (told)
        mask = (meta.seen == c) | (meta.told == c)
        y_prov = (meta.seen[mask] == c).astype(int).values
        # CONTENT: c active at all vs absent from both channels
        y_cont = mask.astype(int).values
        for view, X in views.items():
            rows.append(dict(layer=l, depth=l / (m.n_layers - 1), concept=c,
                             view=view, dim=X.shape[1], probe="provenance",
                             n=int(mask.sum()),
                             auc=cv_auc(X[mask.values], y_prov)))
            rows.append(dict(layer=l, depth=l / (m.n_layers - 1), concept=c,
                             view=view, dim=X.shape[1], probe="content",
                             n=len(y_cont), auc=cv_auc(X, y_cont)))
    print(f"  probed L{l}", flush=True)

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_probes.parquet")
piv = df.groupby(["probe", "view", "layer"]).auc.mean().unstack("view")
piv.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", piv.to_string())
json.dump({"repo": a.repo, "lens": a.lens, "concepts": concepts,
           "n_trials": len(trials), "layers": LAYERS,
           "source_correct": float(bdf.source_correct.mean()),
           "content_correct": float(bdf.content_correct.mean()),
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_meta.json", "w"), indent=2)
