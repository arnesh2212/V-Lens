# SPDX-License-Identifier: Apache-2.0
"""Redraw the regions figure from results/regions_hard/<model>_regions.json.

One row per model, three panels: region-naming accuracy, the mass inside the
object's own boxes over chance (concentration_median, already expressed as a
ratio where 1 = chance), and the pointing game, each against depth for the
four instruments, with a chance line per panel.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import matplotlib.pyplot as plt

from jvlm import plot

ap = argparse.ArgumentParser()
ap.add_argument("jsons", nargs="+")
ap.add_argument("--out", default="figures/regions.png")
a = ap.parse_args()
plot.use()

NICE = {"Qwen3-VL-2B-Instruct": "Qwen3-VL-2B", "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B"}
INSTR = [("V-Write, this item", plot.SERIES[0]),
          ("V-Write", plot.SERIES[4]),
          ("logit lens", plot.SERIES[1]),
          ("logit lens, centred", plot.SERIES[2])]
PANELS = [("accuracy", "naming the object", None),
          ("concentration_median", "mass in the object's boxes, over chance", 1.0),
          ("pointing", "pointing game", None)]

paths = []
for p in a.jsons:
    if not os.path.exists(p):
        print(f"missing, skipped: {p}")
        continue
    paths.append(p)
if not paths:
    print("no regions JSONs found; nothing to draw")
    sys.exit(0)

nrows = len(paths)
fig, axs = plt.subplots(nrows, 3, figsize=(3.5 * 3, 2.9 * nrows), squeeze=False)

used_instr = set()
for i, path in enumerate(paths):
    d = json.load(open(path))
    name = os.path.basename(path).replace("_regions.json", "")
    nice = NICE.get(name, name)
    depths = d["depths"]
    layers = sorted(depths, key=lambda l: depths[l])
    xs = [depths[l] for l in layers]
    bank_chance = 1.0 / len(d["bank"]) if d.get("bank") else None
    box_chance = d.get("box_share")
    regions = d["regions"]
    for j, (metric_key, title, fixed_chance) in enumerate(PANELS):
        ax = axs[i, j]
        if metric_key not in regions:
            ax.set_visible(False)
            continue
        met = regions[metric_key]
        for key, c in INSTR:
            if key not in met:
                continue
            ys = [met[key].get(l) for l in layers]
            if all(y is None for y in ys):
                continue
            ax.plot(xs, ys, color=c, lw=1.5)
            used_instr.add(key)
        chance = fixed_chance if fixed_chance is not None else (bank_chance if metric_key == "accuracy" else box_chance)
        if chance is not None:
            plot.chance_line(ax, chance)
        ax.set_title(f"{nice}, {title}", loc="left", fontsize=9.5, color=plot.INK2)
        ax.set_xlabel("depth")
        plot.despine(ax)
    axs[i, 0].set_ylabel("top-1 accuracy")
    axs[i, 1].set_ylabel("mass share / box share")
    axs[i, 2].set_ylabel("hit rate")

fig.tight_layout()
handles = [plt.Line2D([], [], color=c, lw=1.5, label=key) for key, c in INSTR if key in used_instr]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
           bbox_to_anchor=(0.5, -0.14 / nrows))
plot.save(fig, a.out)
print("wrote", a.out)
