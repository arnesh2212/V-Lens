"""Pool the reality-monitoring probe across concepts.

Ninety trials per concept in two thousand dimensions is not much to fit a probe
on, and a per-concept AUC computed there is noisy however careful the
cross-validation.  Pooling all trials into one probe -- "is the asserted object
really in this photograph" regardless of which object it is -- is both
better-powered and the more interesting question, since a single direction that
works across concepts is a reality signal rather than a per-object detector.

Concept-wise standardisation before pooling removes the per-concept mean, so the
probe cannot succeed by learning which concept a trial is about.  Folds are
grouped by image so the same photograph never appears on both sides of a split.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "/workspace/vlm/code")
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from jvlm import cocodata

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--dir", default="/workspace/vlm/results/reality")
ap.add_argument("--tag", default="qwen2b")
ap.add_argument("--rank", type=int, default=64)
a = ap.parse_args()

meta = pd.read_csv(f"{a.dir}/{a.tag}_trials.csv")
z = np.load(f"{a.dir}/{a.tag}_acts.npz")
layers = sorted(int(k[1:]) for k in z.files)
print(f"{len(meta)} trials, {meta.concept.nunique()} concepts, "
      f"layers {layers[0]}..{layers[-1]}", flush=True)

cfg = transformers.AutoConfig.from_pretrained(a.repo)
hf = getattr(transformers, cfg.architectures[0]).from_pretrained(
    a.repo, dtype=torch.float32, device_map="cpu")
tok = transformers.AutoProcessor.from_pretrained(a.repo).tokenizer
lens = __import__("jlens").lens.JacobianLens.load(a.lens)
W = hf.lm_head.weight.detach()[
    [cocodata.primary_token_id(tok, c) for c in cocodata.SINGLE_WORD]].float()
n_layers = int(hf.config.get_text_config().num_hidden_layers)
del hf


def basis(layer):
    V = W @ lens.jacobians[layer].float()
    V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    U, _, _ = torch.linalg.svd(V.T, full_matrices=False)
    Q = U[:, : a.rank].numpy()
    g = np.random.default_rng(layer)
    Qr, _ = np.linalg.qr(g.standard_normal((Q.shape[0], Q.shape[1])))
    return Q, Qr[:, : Q.shape[1]]


def grouped_auc(X, y, groups, seed=0):
    s = np.zeros(len(y))
    skf = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(max_iter=600, C=0.05)
        clf.fit((X[tr] - mu) / sd, y[tr])
        s[te] = clf.decision_function((X[te] - mu) / sd)
    return float(roc_auc_score(y, s))


y = meta.really.astype(int).values
groups = meta.image_id.values
rows = []
for l in layers:
    X = z[f"L{l}"].astype(np.float32)
    # remove the per-concept mean so the probe cannot read concept identity
    Xc = X.copy()
    for c in meta.concept.unique():
        sel = (meta.concept == c).values
        Xc[sel] -= Xc[sel].mean(0)
    Q, Qr = basis(l)
    views = {"full": Xc, "jspace": Xc @ Q,
             "orthogonal": Xc - (Xc @ Q) @ Q.T, "random": Xc @ Qr}
    for view, V in views.items():
        rows.append(dict(layer=l, depth=l / (n_layers - 1), view=view,
                         dim=V.shape[1], n=len(y), auc=grouped_auc(V, y, groups)))
    print(f"  L{l}: " + "  ".join(
        f"{r['view']} {r['auc']:.3f}" for r in rows[-4:]), flush=True)

df = pd.DataFrame(rows)
df.to_parquet(f"{a.dir}/{a.tag}_pooled_probes.parquet")
piv = df.pivot(index="layer", columns="view", values="auc")
piv.to_csv(f"{a.dir}/{a.tag}_pooled_summary.csv")
print("\n", piv.round(3).to_string())
print("\nPOOLED over layers:", df.groupby("view").auc.mean().round(4).to_dict())
json.dump({"repo": a.repo, "rank": a.rank, "n": len(y),
           "by_view": df.groupby("view").auc.mean().round(4).to_dict(),
           "best_layer": {v: float(df[df.view == v].auc.max())
                          for v in df.view.unique()}},
          open(f"{a.dir}/{a.tag}_pooled_stats.json", "w"), indent=2)
