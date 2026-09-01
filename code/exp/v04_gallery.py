# SPDX-License-Identifier: Apache-2.0
"""Visual examples: what each image patch pushes the answer toward, layer by layer.

For a handful of items from a gate cache, draw the photograph beside a row of
per-patch maps at each cached depth, for two readings of the same activations:

  Verdict Lens   V_l(p) = u_l^img . (h_l[p] - mu_l^img)   for the chosen axis
  identity       r . h_l[p]                               (the prompt-blind logit-lens-style read)

Both are label-free.  The identity map cannot depend on the request at image
positions (causal mask), and neither can ours at those positions; the
difference is the transport: ours carries the patch to the answer slot through
the model's own average Jacobian, the identity does not.  The maps are shown
with the model's own reply and its refusal label, so a reader can check whether
the patches the lens lights up are the ones that plausibly drove the decision.

Runs on the CPU from the cache; no model needed.  Writes one PNG per item and
one contact sheet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--axis", default="refusal", choices=["refusal", "starters", "yesno", "harm"])
ap.add_argument("--n", type=int, default=6, help="items per class")
ap.add_argument("--gate-dir", default="/workspace/vlm/results/gate")
ap.add_argument("--out", default="/workspace/vlm/figures/gallery")
a = ap.parse_args()

from jvlm import cache, judge, plot, verdict  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

name = a.repo.split("/")[-1]
plot.use()
os.makedirs(a.out, exist_ok=True)
acts, side = cache.load(f"{a.gate_dir}/{name}__eval.npz")
VL = verdict.VerdictLens.load(f"{a.gate_dir}/{name}__verdict_carrier.pt")
ANS = side["_extra"]["ANS"].astype(np.float32)
its = side["items"]; L = VL.layers; grid = int(round(side["image_tokens"] ** 0.5))
y_ref = np.array([judge.kind(it["reply"]) == "refusal" for it in its], int)
y_harm = np.array([it["harmful"] for it in its], int)
r = verdict.dim_direction(ANS[:, -1], y_ref).numpy()      # the final-layer refusal direction
G = json.load(open(f"{a.gate_dir}/{name}_gate.json"))
frac = {int(l): v["frac"] for l, v in G["calibration"].items()}

# pick items: refused harmful, complied harmful, complied benign
rng = np.random.default_rng(1)
def pick(mask, k):
    idx = np.nonzero(mask)[0]
    return list(rng.permutation(idx)[:k])
picks = [("refused, harmful", pick((y_ref == 1) & (y_harm == 1), a.n)),
         ("complied, harmful", pick((y_ref == 0) & (y_harm == 1), a.n)),
         ("complied, benign", pick((y_ref == 0) & (y_harm == 0), a.n))]


def maps_for(i):
    """Per-depth patch maps for item i, both readings, on a shared symmetric scale each."""
    V, I = [], []
    for l in L:
        h = acts[l][i].astype(np.float32)                       # [P, d]
        V.append(VL.read(h, a.axis, "img", l).reshape(grid, grid))
        I.append((h @ r).reshape(grid, grid))
    return np.stack(V), np.stack(I)


def draw(i, tag):
    V, I = maps_for(i)
    it = its[i]
    fig = plt.figure(figsize=(1.9 * (len(L) + 1), 4.3))
    gs = fig.add_gridspec(2, len(L) + 1, width_ratios=[1.4] + [1] * len(L), wspace=0.08, hspace=0.25)
    ax = fig.add_subplot(gs[:, 0]); ax.imshow(Image.open(it["path"]).convert("RGB")); ax.set_axis_off()
    reply = it["reply"].replace("\n", " ")[:90]
    ax.set_title(f"{tag}\n{judge.kind(it['reply'])}: “{reply}…”", fontsize=8.5, loc="left", color=plot.INK2)
    for row, (M, lab) in enumerate(((V, "Verdict Lens"), (I, "identity"))):
        s = np.abs(M).max() or 1.0
        for j, l in enumerate(L):
            ax = fig.add_subplot(gs[row, j + 1])
            ax.imshow(M[j], cmap="RdBu_r", vmin=-s, vmax=s, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if row == 0: ax.set_title(f"depth {frac[l]:.2f}", fontsize=9, color=plot.INK2)
            if j == 0: ax.set_ylabel(lab, fontsize=9.5, color=plot.INK2)
    p = f"{a.out}/{name}_{a.axis}_{tag.replace(', ', '_').replace(' ', '')}_{i}.png"
    plot.save(fig, p, dpi=170)
    return p, V, I

sheet = []
for tag, idx in picks:
    for i in idx:
        p, V, I = draw(i, tag)
        sheet.append((tag, i, p, float(np.abs(V).max()), float(np.abs(I).max())))
        print("wrote", p, flush=True)

# summary: how concentrated is each map?  (share of |V| mass in the top 10% of patches)
def conc(M):
    v = np.abs(M).ravel(); v = np.sort(v)[::-1]; k = max(1, len(v) // 10)
    return float(v[:k].sum() / (v.sum() + 1e-9))
stats = {}
for l_i, l in enumerate(L):
    cv = [conc(maps_for(i)[0][l_i]) for _, idx in picks for i in idx]
    ci = [conc(maps_for(i)[1][l_i]) for _, idx in picks for i in idx]
    stats[l] = dict(frac=frac[l], verdict_top10_share=float(np.mean(cv)), identity_top10_share=float(np.mean(ci)))
    print(f"  depth {frac[l]:.2f}: share of |map| in the top 10% of patches  verdict {np.mean(cv):.2f}  identity {np.mean(ci):.2f}")
json.dump(dict(items=[dict(tag=t, i=int(i), png=p) for t, i, p, _, _ in sheet], concentration=stats),
          open(f"{a.out}/{name}_{a.axis}_gallery.json", "w"), indent=1)
print("done")
