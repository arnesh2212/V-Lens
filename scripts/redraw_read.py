# SPDX-License-Identifier: Apache-2.0
"""Redraw the decision figure from results/gate/reader_regression.json with the current names."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import matplotlib.pyplot as plt
from jvlm import plot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "results/gate/reader_regression.json")))
plot.use()
NICE = {"Qwen3-VL-2B-Instruct": "Qwen3-VL-2B", "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B", "InternVL3-8B-hf": "InternVL3-8B"}
SER = [("ridge", "V-Read (no labels)", plot.SERIES[0], "-", "o"),
       ("probe", "linear probe (labelled)", plot.SERIES[2], ":", "s"),
       ("identity_perfold", "raw reading (HiddenDetect)", plot.SERIES[1], "-", None)]
models = [k for k in d if k in NICE]
fig, axs = plt.subplots(1, len(models), figsize=(3.5 * len(models), 3.0), sharey=True)
axs = [axs] if len(models) == 1 else list(axs)
for ax, mk in zip(axs, models):
    ls = sorted(d[mk], key=lambda z: d[mk][z]["frac"])
    x = [d[mk][l]["frac"] for l in ls]
    for key, lab, c, style, mk_ in SER:
        y = [d[mk][l][key] for l in ls]
        ax.plot(x, y, color=c, lw=1.6, ls=style, marker=mk_, ms=4.5)
    ax.set_title(NICE[mk], loc="left", fontsize=10, color=plot.INK2)
    ax.set_xlabel("depth"); plot.despine(ax)
axs[0].set_ylabel("AUROC, refused vs complied"); axs[0].set_ylim(0.4, 1.02)
h = [plt.Line2D([], [], color=c, ls=style, marker=mk_, label=lab) for _, lab, c, style, mk_ in SER]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
plot.save(fig, os.path.join(ROOT, "figures/reader_decision.png"))
