# SPDX-License-Identifier: Apache-2.0
"""Is the read just a tuned lens?

The read is one affine map per layer from the answer spot to its final state,
fitted by ridge on unlabelled activations.  A tuned lens is one affine map per
layer fitted to match the model's output distribution, by KL descent from the
same ridge solution.  They are close cousins, and the honest way to say what is
new here is to run both and report the gap.

Three maps, all fitted on the same unlabelled corpus and none of them touching a
test label:
  read          ridge to the final state (ours)
  tuned lens    the same, then two epochs of KL descent against the model's own
                output distribution (Belrose et al.'s objective)
  identity      no map at all, which is what HiddenDetect reads

Each is scored two ways on the same held-out items: the vocabulary way, by how
well its predicted distribution matches the model's next-token distribution, and
the way this page cares about, by whether the refusal score read through it says
what the model is about to do.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--variants", nargs="*", default=["SD_TYPO"])
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--gate-dir", default="/workspace/vlm/results/gate")
ap.add_argument("--read", default=None, help="results/reader/<model>__reader.pt from the disjoint corpus")
ap.add_argument("--out", default="/workspace/vlm/results/tunedcheck")
a = ap.parse_args()

import vlens  # noqa: E402
from vlens.lens import cache, depth, probes, reader, tuned, verdict  # noqa: E402
from vlens import judge  # noqa: E402

name = a.repo.split("/")[-1]
os.makedirs(a.out, exist_ok=True)
say = lambda *x: print(*x, flush=True)
T0 = time.time()
m = vlens.load_vlm(a.repo)
ALLL = list(range(max(m.n_deepstack, 1), m.n_layers - 1))
frac = {l: depth.to_frac(m, l) for l in range(m.n_layers)}

# the fit corpus: the answer-spot states from the disjoint COCO run, no labels, no safety images
fit_path = f"/workspace/vlm/results/reader/{name}__fit_ans.npz"
if not os.path.exists(fit_path):
    sys.exit(f"no fit corpus at {fit_path}; run v09_reader_eval.py first")
_, fside = cache.load(fit_path)
FIT = fside["_extra"]["ANS"].astype(np.float32)
say(f"fit corpus {FIT.shape[0]} answer spots x {FIT.shape[1]} blocks")

acts_fit = {l: FIT[:, l] for l in [*ALLL, FIT.shape[1] - 1]}
R = reader.Reader.load(a.read) if a.read else reader.fit(FIT, ALLL)
TL_ridge = tuned.fit_from_cache(acts_fit, FIT.shape[1] - 1, lam=1e-2)
TL = tuned.refine_kl(TL_ridge, m, acts_fit, FIT.shape[1] - 1, epochs=a.epochs)
say("maps fitted: read (ridge), tuned lens (ridge then KL)")

# the test set
acts, side = cache.load(f"{a.gate_dir}/{name}__eval.npz")
keep = np.array([it.get("variant") in a.variants for it in side["items"]])
its = [it for it, k in zip(side["items"], keep) if k]
ANS = side["_extra"]["ANS"].astype(np.float32)[keep]
y_r = np.array([judge.item_refused(it) for it in its], int)
r_ref = verdict.dim_direction(ANS[:, -1], y_r).numpy()
say(f"test {len(its)} items, refused {y_r.mean():.2f}")

# how well each map predicts the model's own next-token distribution, on the test states
def kl_of(mapper, l):
    X = torch.as_tensor(ANS[:, l]).to(m.device)
    with torch.no_grad():
        ref = torch.log_softmax(m.unembed(torch.as_tensor(ANS[:, -1]).to(m.device)).float(), -1)
        got = torch.log_softmax(m.unembed(torch.as_tensor(mapper(ANS[:, l], l)).to(m.device)).float(), -1)
        return float((ref.exp() * (ref - got)).sum(-1).mean())

MAPPERS = {"V-Read": lambda X, l: R.to_final(X, l),
           "tuned lens": lambda X, l: (torch.as_tensor(X).float() @ TL.maps[l][0].T + TL.maps[l][1]).numpy(),
           "identity": lambda X, l: X}
rows = []
say(f"\n{'depth':>6} | {'KL to the model’s own distribution':^34} | {'AUROC, about to refuse':^30}")
say(f"{'':>6} | " + " ".join(f"{k:>10}" for k in MAPPERS) + " | " + " ".join(f"{k:>9}" for k in MAPPERS))
for l in ALLL:
    kl = {k: kl_of(f, l) for k, f in MAPPERS.items()}
    au = {}
    for k, f in MAPPERS.items():
        s_ = np.asarray(f(ANS[:, l], l)) @ r_ref
        au[k] = probes.readout_score(s_, y_r, shuffles=5)["auroc"]
    rows.append(dict(layer=l, frac=frac[l], kl=kl, auroc=au))
    say(f"{frac[l]:>6.2f} | " + " ".join(f"{kl[k]:>10.3f}" for k in MAPPERS) + " | " + " ".join(f"{au[k]:>9.3f}" for k in MAPPERS))

json.dump(dict(repo=a.repo, n_fit=int(FIT.shape[0]), n_test=len(its), epochs=a.epochs, rows=rows),
          open(f"{a.out}/{name}_tunedcheck.json", "w"), indent=1, default=float)
say(f"\nwrote {a.out}/{name}_tunedcheck.json  ({(time.time() - T0) / 60:.1f} min)")
