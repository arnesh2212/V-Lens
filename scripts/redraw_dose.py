# SPDX-License-Identifier: Apache-2.0
"""Redraw the dose figure from finished steering runs, one panel per model."""
import argparse, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import numpy as np, matplotlib.pyplot as plt
from jvlm import plot

ap = argparse.ArgumentParser(); ap.add_argument("jsons", nargs="+"); ap.add_argument("--frac", type=float, default=0.48)
ap.add_argument("--out", default=None); a = ap.parse_args()
plot.use()
NAMES = {"writer": "write", "push": "write", "lens": "write", "DIM (labelled)": "difference of means, labelled",
         "difference of means (labelled)": "difference of means, labelled", "logit-lens": "output-layer row",
         "logit lens": "output-layer row", "output-layer row": "output-layer row", "random": "random, same strength"}
ORDER = ["write", "difference of means, labelled", "output-layer row", "random, same strength"]
COL = {"write": plot.SERIES[0], "difference of means, labelled": "#7a7a7a", "output-layer row": plot.SERIES[1], "random, same strength": "#c9c8c1"}
NICE = {"Qwen3-VL-2B-Instruct": "Qwen3-VL-2B", "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B"}
fig, axs = plt.subplots(1, len(a.jsons), figsize=(4.1 * len(a.jsons), 3.1), sharey=True)
axs = list(axs) if len(a.jsons) > 1 else [axs]
for ax, path in zip(axs, a.jsons):
    d = json.load(open(path)); name = os.path.basename(path).replace("_steer.json", "")
    agg = defaultdict(list)
    for it in d["items"]:
        for r in it["runs"]:
            agg[(r["mode"], NAMES.get(r["direction"], r["direction"]), round(r["frac"], 2), r["alpha"])].append(r["said_B"])
    fr = min({k[2] for k in agg}, key=lambda z: abs(z - a.frac))
    for dn in ORDER:
        y = [np.mean(agg[("add", dn, fr, x)]) if agg.get(("add", dn, fr, x)) else np.nan for x in d["alphas"]]
        ax.plot(d["alphas"], y, color=COL[dn], lw=1.6, marker="o", ms=4)
    ax.set_title(f"{NICE.get(name, name)}, depth {fr:g}", loc="left", fontsize=10, color=plot.INK2)
    ax.set_xlabel("dose, as a fraction of the state’s own size"); plot.despine(ax); ax.set_ylim(-0.02, 1.04)
axs[0].set_ylabel("answer becomes the written word")
h = [plt.Line2D([], [], color=COL[k], marker="o", label=k) for k in ORDER]
fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.17))
fig.tight_layout(); plot.save(fig, a.out or "figures/steer_dose.png")
