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
ap.add_argument("--per-item", action="store_true",
                help="add the per-item row: this item's own gradient of the refusal margin at each image "
                     "position, times its centred activation (attribution patching with the corpus mean as "
                     "baseline).  Needs the model on the GPU.")
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
# per-position corpus mean at each depth, so both maps read a deviation from what
# that patch position usually holds rather than the position's constant offset
MU = {l: acts[l].astype(np.float32).mean(0) for l in L}    # [P, d]
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


M_MODEL = None
if a.per_item:
    import jvlm
    from jvlm import corpus, readout
    from jlens.hooks import ActivationRecorder
    M_MODEL = jvlm.load_vlm(a.repo)
    R_DEV = torch.as_tensor(r).to(M_MODEL.device)


def per_item_maps(i):
    """This item's own gradient of r . h_L[a] at every image position and cached depth,
    times its centred activation: the per-item member of the same family."""
    it = its[i]
    b = M_MODEL.chat(side["prompt"], [corpus.load_image(it["path"], square=448)])
    ip = readout.image_positions(M_MODEL, b); S = b["input_ids"].shape[1]
    with ActivationRecorder(M_MODEL.layers, at=[*L, M_MODEL.n_layers - 1], start_graph_at=min(L)) as rec, torch.enable_grad():
        M_MODEL.forward(b)
        A = [rec.activations[l] for l in L]
        hL = rec.activations[M_MODEL.n_layers - 1][0, S - 1]
        g = torch.autograd.grad((hL.float() @ R_DEV), A)
    out = []
    for l, gl, al in zip(L, g, A):
        gp = gl[0, ip.to(gl.device)].float().cpu().numpy()
        hp = al[0, ip.to(al.device)].detach().float().cpu().numpy() - MU[l]
        out.append((gp * hp).sum(-1).reshape(grid, grid))
    return np.stack(out)


def maps_for(i):
    """Per-depth patch maps for item i: the fixed-direction lens, the identity, and (optionally) per-item."""
    V, I = [], []
    for l in L:
        h = acts[l][i].astype(np.float32) - MU[l]               # [P, d], centred per position
        V.append((h @ VL.dir(a.axis, "img", l).numpy()).reshape(grid, grid))
        I.append((h @ r).reshape(grid, grid))
    return np.stack(V), np.stack(I)


def draw(i, tag):
    V, I = maps_for(i)
    rows = [(V, "lens, fixed direction"), (I, "identity")]
    if M_MODEL is not None:
        rows.append((per_item_maps(i), "lens, this item's gradient"))
    it = its[i]
    fig = plt.figure(figsize=(1.9 * (len(L) + 1), 2.15 * len(rows)))
    gs = fig.add_gridspec(len(rows), len(L) + 1, width_ratios=[1.4] + [1] * len(L), wspace=0.08, hspace=0.25)
    ax = fig.add_subplot(gs[:, 0]); ax.imshow(Image.open(it["path"]).convert("RGB")); ax.set_axis_off()
    reply = it["reply"].replace("\n", " ")[:70]
    ax.set_title(tag, fontsize=9, loc="left", color=plot.INK2)
    ax.text(0, -0.03, f"{judge.kind(it['reply'])}: “{reply}…”", transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color=plot.INK2, wrap=True)
    for row, (M, lab) in enumerate(rows):
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
    return p, V, I, (rows[2][0] if len(rows) > 2 else None)

sheet = []; PI = {}
for tag, idx in picks:
    for i in idx:
        p, V, I, Pm = draw(i, tag)
        if Pm is not None:
            PI[i] = Pm
        sheet.append((tag, i, p, float(np.abs(V).max()), float(np.abs(I).max())))
        print("wrote", p, flush=True)

# localisation: on SD_TYPO items the harmful phrase is printed in the bottom band of the
# image.  Share of |map| mass in the bottom quarter of rows; chance is 0.25.
def band_share(M):
    rows = grid // 4
    return float(np.abs(M[-rows:]).sum() / (np.abs(M).sum() + 1e-9))
typo = [i for i in range(len(its)) if its[i].get("variant") == "SD_TYPO"]
if typo:
    print(f"\ntypography band (bottom quarter) share of |map|, chance 0.25, over {len(typo)} SD_TYPO items:")
    per = {}
    if M_MODEL is not None:
        for i in typo[:60]:
            per[i] = per_item_maps(i)
    for l_i, l in enumerate(L):
        bv = np.mean([band_share(maps_for(i)[0][l_i]) for i in typo[:300]])
        bi = np.mean([band_share(maps_for(i)[1][l_i]) for i in typo[:300]])
        line = f"  depth {frac[l]:.2f}:  fixed-direction {bv:.3f}   identity {bi:.3f}"
        if per:
            line += f"   per-item gradient x activation {np.mean([band_share(per[i][l_i]) for i in per]):.3f}  (n={len(per)})"
        print(line)

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
