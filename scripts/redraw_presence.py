# SPDX-License-Identifier: Apache-2.0
"""Two-model presence figure from results/regions*/<model>_regions.json."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import numpy as np, matplotlib.pyplot as plt
from jvlm import plot

ap = argparse.ArgumentParser(); ap.add_argument("jsons", nargs="+"); ap.add_argument("--out", default="figures/presence.png")
a = ap.parse_args(); plot.use()
NICE = {"Qwen3-VL-2B-Instruct": "Qwen3-VL-2B", "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B"}
SER = [("reader", "V-Read (no labels)", plot.SERIES[0], "-"), ("probe", "linear probe (labelled)", plot.SERIES[2], ":"),
       ("logit lens", "logit lens", plot.SERIES[1], "-"), ("write", "V-Write", "#8a8880", "-")]
fig, axs = plt.subplots(2, len(a.jsons), figsize=(3.9 * len(a.jsons), 5.2), sharex=True, sharey=True)
axs = np.atleast_2d(axs); axs = axs.reshape(2, -1)
for j, path in enumerate(a.jsons):
    d = json.load(open(path)); E = d["emergence"]; name = os.path.basename(path).replace("_regions.json", "")
    dep = E["depth_of"]; ls = sorted(dep, key=lambda z: dep[z]); xs = [dep[l] for l in ls]
    for key, lab, c, style in SER:
        axs[0, j].plot(xs, [E["auroc"][key][l] for l in ls], color=c, lw=1.5 if key == "reader" else 1.1, ls=style)
        y = [E["hallucination_auroc"][key][l] for l in ls]
        if not all(np.isnan(np.array(y, float))):
            axs[1, j].plot(xs, y, color=c, lw=1.5 if key == "reader" else 1.1, ls=style)
    for ax in (axs[0, j], axs[1, j]):
        plot.chance_line(ax, 0.5); plot.despine(ax)
    axs[0, j].set_title(NICE.get(name, name), loc="left", fontsize=10, color=plot.INK2)
    axs[1, j].set_xlabel("depth")
    axs[1, j].text(0.02, 0.06, f"{E['n_hallucinated']} hallucinated yes answers", transform=axs[1, j].transAxes, fontsize=8, color=plot.INK2)
axs[0, 0].set_ylabel("is the object there?\nAUROC, present vs absent")
axs[1, 0].set_ylabel("among the model’s yes answers,\ntrue vs hallucinated")
axs[0, 0].set_ylim(0.15, 1.02)
h = [plt.Line2D([], [], color=c, ls=style, label=lab) for _, lab, c, style in SER]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.05))
fig.tight_layout(); plot.save(fig, a.out)
print("wrote", a.out)
