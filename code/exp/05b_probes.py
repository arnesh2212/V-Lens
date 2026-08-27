"""Probe the saved provenance activations, on CPU, in parallel.

The activations and the trial table are written before any probing happens, so
this runs entirely off the GPU.  Splitting it out matters for a second reason:
fitting a few thousand logistic regressions single-threaded takes over an hour
while holding a model resident on the accelerator, which is exactly the memory
the next lens fit needs.
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
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--dir", default="/workspace/vlm/results/provenance")
ap.add_argument("--tag", default="qwen2b")
ap.add_argument("--jobs", type=int, default=14)
a = ap.parse_args()

meta = pd.read_csv(f"{a.dir}/{a.tag}_trials.csv")
z = np.load(f"{a.dir}/{a.tag}_acts.npz")
layers = sorted(int(k[1:]) for k in z.files)
concepts = sorted(set(meta.seen) | set(meta.told))
print(f"{len(meta)} trials, layers {layers[0]}..{layers[-1]}, "
      f"{len(concepts)} concepts", flush=True)

# The workspace basis has to be rebuilt, but only the weights are needed for
# that -- no forward pass, so the model stays on the CPU.
import transformers  # noqa: E402

from jvlm import cocodata  # noqa: E402

cfg = transformers.AutoConfig.from_pretrained(a.repo)
cls = getattr(transformers, cfg.architectures[0])
hf = cls.from_pretrained(a.repo, dtype=torch.float32, device_map="cpu")
W_U = hf.lm_head.weight.detach()
N_LAYERS = int(hf.config.get_text_config().num_hidden_layers)
tok = transformers.AutoProcessor.from_pretrained(a.repo).tokenizer
lens = __import__("jlens").lens.JacobianLens.load(a.lens)
cat_ids = [cocodata.primary_token_id(tok, c) for c in cocodata.SINGLE_WORD]
Wc = W_U[cat_ids].float()


def bases(layer: int):
    J = lens.jacobians[layer].float()
    V = Wc @ J
    V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    Qj, _ = torch.linalg.qr(V.T)
    k = Qj.shape[1]
    g = np.random.default_rng(layer)
    Qr, _ = np.linalg.qr(g.standard_normal((Qj.shape[0], k)))
    return Qj.numpy(), Qr[:, :k]


def cv_auc(X, y, seed=0):
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 6:
        return float("nan")
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    s = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(max_iter=400, C=0.1)
        clf.fit((X[tr] - mu) / sd, y[tr])
        s[te] = clf.decision_function((X[te] - mu) / sd)
    return float(roc_auc_score(y, s))


def one_layer(layer, Xf, Qj, Qr, seen, told, concepts):
    """Everything a worker needs is passed in: closures over a loaded model, an
    open npz handle or a torch tensor are not picklable, and loky pickles."""
    views = {"full": Xf, "jspace": Xf @ Qj,
             "orthogonal": Xf - (Xf @ Qj) @ Qj.T, "random": Xf @ Qr}
    rows = []
    for c in concepts:
        mask = (seen == c) | (told == c)
        y_prov = (seen[mask] == c).astype(int)
        y_cont = mask.astype(int)
        for view, X in views.items():
            rows.append(dict(layer=layer, concept=c, view=view, dim=X.shape[1],
                             probe="provenance", n=int(mask.sum()),
                             auc=cv_auc(X[mask], y_prov)))
            rows.append(dict(layer=layer, concept=c, view=view, dim=X.shape[1],
                             probe="content", n=len(y_cont),
                             auc=cv_auc(X, y_cont)))
    print(f"  layer {layer} done", flush=True)
    return rows


seen = meta.seen.to_numpy()
told = meta.told.to_numpy()
tasks = []
for l in layers:
    Qj, Qr = bases(l)
    tasks.append((l, z[f"L{l}"].astype(np.float32), Qj.astype(np.float32),
                  Qr.astype(np.float32)))
del hf, W_U, Wc
out = Parallel(n_jobs=a.jobs, backend="loky")(
    delayed(one_layer)(l, X, Qj, Qr, seen, told, concepts)
    for l, X, Qj, Qr in tasks)
df = pd.DataFrame([r for rs in out for r in rs])
df["depth"] = df.layer / (N_LAYERS - 1)
df.to_parquet(f"{a.dir}/{a.tag}_probes.parquet")
piv = df.groupby(["probe", "view", "layer"]).auc.mean().unstack("view")
piv.to_csv(f"{a.dir}/{a.tag}_summary.csv")
print("\n", piv.round(3).to_string())
print("\nPOOLED over layers:")
print(df.groupby(["probe", "view"]).auc.mean().unstack("view").round(4).to_string())
json.dump({"repo": a.repo, "n_trials": len(meta), "layers": layers,
           "pooled": df.groupby(["probe", "view"]).auc.mean().round(4).to_dict()},
          open(f"{a.dir}/{a.tag}_probe_meta.json", "w"), indent=2, default=float)
