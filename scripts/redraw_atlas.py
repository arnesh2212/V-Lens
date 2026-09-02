# SPDX-License-Identifier: Apache-2.0
"""Redraw the atlas summary figure from results/atlas/<model>_atlas.json.

A 2 by 2 grid: rank agreement with patching on top, top-1 region hit rate on
the bottom; refusal on the left, yes/no on the right.  Six instruments per
panel, so the series are named in a legend below rather than end labels.
GLIMPSE has no depth and no resample/cover correlation, only the absolute-value
one, so it is drawn dashed and named as such in the legend.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import numpy as np
import matplotlib.pyplot as plt

from jvlm import plot

ap = argparse.ArgumentParser()
ap.add_argument("json")
ap.add_argument("--out", default="figures/atlas_summary.png")
a = ap.parse_args()

if not os.path.exists(a.json):
    print(f"missing, nothing to draw: {a.json}")
    sys.exit(0)

plot.use()
NICE = {"Qwen3-VL-2B-Instruct": "Qwen3-VL-2B", "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B"}
INSTR = [("V-Write, path", plot.SERIES[0]),
          ("V-Write, one point", plot.SERIES[4]),
          ("V-Write, averaged", plot.SERIES[2]),
          ("identity", plot.SERIES[1]),
          ("grad x act", plot.SERIES[3]),
          ("GLIMPSE", "#7a7a7a")]
KINDS = [("refusal", "refusal margin, typography items"), ("yesno", "yes/no margin, COCO photographs")]

d = json.load(open(a.json))


def rename(o):
    """Runs from before the rename store the write maps under "write, ...";
    read either spelling rather than silently dropping the series."""
    if isinstance(o, dict):
        return {(k.replace("write,", "V-Write,") if isinstance(k, str) and k.startswith("write,") else k): rename(v)
                for k, v in o.items()}
    if isinstance(o, list):
        return [rename(x) for x in o]
    return o


d = rename(d)
name = os.path.basename(a.json).replace("_atlas.json", "")
nice = NICE.get(name, name)
depths = d["depths"]
layers = sorted(depths, key=lambda l: depths[l])
xs = [depths[l] for l in layers]
regions = d.get("regions")
summary = d["summary"]

fig, axs = plt.subplots(2, 2, figsize=(8.6, 5.4), sharex=True)
used = set()
for j, (kind, title) in enumerate(KINDS):
    if kind not in summary:
        for row in axs[:, j]:
            row.set_visible(False)
        continue
    S = summary[kind]
    for key, c in INSTR:
        if key not in S:
            continue
        metric = "rho_abs" if key == "GLIMPSE" else "rho_resample"
        ls = "--" if key == "GLIMPSE" else "-"
        vals = S[key].get(metric, {})
        ys = [vals[l][0] if l in vals else None for l in layers]
        if all(y is None for y in ys):
            continue
        axs[0, j].plot(xs, ys, color=c, lw=1.7 if key == "V-Write, path" else 1.1, ls=ls)
        used.add(key)
        t1 = S[key].get("top1", {})
        ys1 = [t1[l][1] if l in t1 else None for l in layers]
        if not all(y is None for y in ys1):
            axs[1, j].plot(xs, ys1, color=c, lw=1.7 if key == "V-Write, path" else 1.1, ls=ls)
    plot.chance_line(axs[0, j], 0.0)
    if regions:
        plot.chance_line(axs[1, j], 1.0 / regions)
    axs[0, j].set_title(f"{nice}, {title}", loc="left", fontsize=9.5, color=plot.INK2)
    axs[1, j].set_xlabel("depth")

axs[0, 0].set_ylabel("rank agreement with patching")
axs[1, 0].set_ylabel("strongest region is the right one")
for ax in axs.flat:
    plot.despine(ax)

LAB = {"V-Write, path": "V-Write, path", "V-Write, one point": "V-Write, one point", "V-Write, averaged": "V-Write, averaged",
       "identity": "identity (logit lens)", "grad x act": "grad x act", "GLIMPSE": "GLIMPSE (|.| version, dashed)"}
handles = [plt.Line2D([], [], color=c, lw=1.5, ls="--" if key == "GLIMPSE" else "-", label=LAB[key])
           for key, c in INSTR if key in used]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.14))
fig.tight_layout()
plot.save(fig, a.out)
print("wrote", a.out)
