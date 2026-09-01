# SPDX-License-Identifier: Apache-2.0
"""Redraw the where-to-read figure from results/gate/*_total_verdict.json, current names."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import matplotlib.pyplot as plt
from jvlm import plot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = [("Qwen3-VL-2B-Instruct", "Qwen3-VL-2B"), ("Qwen3-VL-8B-Instruct", "Qwen3-VL-8B")]
SER = [("total", "write, everywhere", plot.SERIES[0], "-", "o"),
       ("txt", "write, the words", "#4a3aa7", "-", "^"),
       ("ans", "write, the answer spot", plot.SERIES[0], "--", None),
       ("img", "write, the picture", "#8a8880", "-", None),
       ("identity", "raw reading", plot.SERIES[1], "-", None),
       ("probe", "linear probe (labelled)", plot.SERIES[2], ":", "s")]
plot.use()
fig, axs = plt.subplots(1, len(MODELS), figsize=(3.9 * len(MODELS), 3.0), sharey=True)
axs = list(axs) if len(MODELS) > 1 else [axs]
for ax, (mk, nice) in zip(axs, MODELS):
    p = os.path.join(ROOT, f"results/gate/{mk}_total_verdict.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p)); ls = sorted(d, key=lambda z: d[z]["frac"]); x = [d[l]["frac"] for l in ls]
    for key, lab, c, style, mkr in SER:
        if key not in d[ls[0]]:
            continue
        ax.plot(x, [d[l][key] for l in ls], color=c, lw=1.5 if key in ("total", "identity") else 1.1, ls=style, marker=mkr, ms=4)
    plot.chance_line(ax, 0.5)
    ax.set_title(nice, loc="left", fontsize=10, color=plot.INK2); ax.set_xlabel("depth"); plot.despine(ax)
axs[0].set_ylabel("AUROC, refused vs complied"); axs[0].set_ylim(0.4, 1.02)
h = [plt.Line2D([], [], color=c, ls=style, marker=mkr, label=lab) for _, lab, c, style, mkr in SER]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.13))
fig.tight_layout(); plot.save(fig, os.path.join(ROOT, "figures/gate_channels.png"))
