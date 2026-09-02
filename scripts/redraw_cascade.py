# SPDX-License-Identifier: Apache-2.0
"""Redraw the cascade figure from results/cascade/<model>_cascade.json.

Two panels sharing a y axis: AUROC (harmful vs benign) against depth for each
of the four order-of-arrival scores (content, harm, refusal, opener), the
reader's transport on the left and the logit lens on the right, with a dotted
line at 0.7 (the onset threshold used in the underlying experiment) and a
chance line at 0.5.  One row per model JSON given (up to two).
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
ap.add_argument("--out", default="figures/cascade.png")
a = ap.parse_args()
plot.use()

NICE = {"Qwen3-VL-2B-Instruct": "Qwen3-VL-2B", "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B"}
SCORES = [("content", "content", plot.SERIES[0]),
          ("harm", "harm", plot.SERIES[1]),
          ("refusal", "refusal", plot.SERIES[2]),
          ("opener", "opener", plot.SERIES[3])]
TRANSPORTS = [("reader", "V-Read"), ("logit lens", "logit lens")]

paths = []
for p in a.jsons:
    if not os.path.exists(p):
        print(f"missing, skipped: {p}")
        continue
    paths.append(p)
if not paths:
    print("no cascade JSONs found; nothing to draw")
    sys.exit(0)

nrows = len(paths)
fig, axs = plt.subplots(nrows, 2, figsize=(7.8, 2.9 * nrows), sharey=True, sharex=True, squeeze=False)

used_scores = set()
all_y = []
for i, path in enumerate(paths):
    d = json.load(open(path))
    name = os.path.basename(path).replace("_cascade.json", "")
    nice = NICE.get(name, name)
    rows = d["rows"]
    for j, (transport_key, transport_label) in enumerate(TRANSPORTS):
        ax = axs[i, j]
        sub = sorted((r for r in rows if r["transport"] == transport_key), key=lambda r: r["frac"])
        xs = [r["frac"] for r in sub]
        for key, lab, c in SCORES:
            ys = [r["auroc"].get(key) for r in sub]
            if not xs or all(y is None for y in ys):
                continue
            ax.plot(xs, ys, color=c, lw=1.5)
            used_scores.add(key)
            all_y.extend(y for y in ys if y is not None)
        ax.axhline(0.7, color=plot.INK2, lw=0.6, ls=":", zorder=1)
        plot.chance_line(ax, 0.5)
        ax.set_title(f"{nice}, {transport_label}", loc="left", fontsize=9.5, color=plot.INK2)
        plot.despine(ax)
    axs[i, 0].set_ylabel("AUROC,\nharmful vs benign")

for j in range(2):
    axs[-1, j].set_xlabel("depth")
lo = min(all_y) if all_y else 0.4
axs[0, 0].set_ylim(min(0.45, lo - 0.05), 1.02)

handles = [plt.Line2D([], [], color=c, lw=1.5, label=lab) for key, lab, c in SCORES if key in used_scores]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
           bbox_to_anchor=(0.5, -0.1 / nrows))
fig.tight_layout()
plot.save(fig, a.out)
print("wrote", a.out)
