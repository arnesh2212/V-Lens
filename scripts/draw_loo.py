# SPDX-License-Identifier: Apache-2.0
"""Leave-one-benchmark-out at a fifth of the depth, as a dot plot.

One row per held-out benchmark, one dot per instrument: the map refitted on that
set's own unlabelled states, the same map fitted on COCO photographs, the score
read raw, and a probe trained with labels on the same rows.  The point of the
figure is the gap between the two maps, and that the label-free one sits to the
right of the labelled probe on nearly every row.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import matplotlib.pyplot as plt  # noqa: E402
from jvlm import plot  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results/reader/Qwen3-VL-8B-Instruct_loo.json")
OUT = os.path.join(ROOT, "figures/loo_refit.png")
NAMES = {"mmsb_typo": "MM-SafetyBench", "figstep": "FigStep", "hades": "HADES", "vlguard": "VLGuard",
         "vlsbench": "VLSBench", "siuo": "SIUO", "mossbench_contrast": "MOSSBench", "spavl_harm": "SPA-VL"}
ORDER = ["mmsb_typo", "figstep", "hades", "vlguard", "vlsbench", "siuo", "mossbench_contrast", "spavl_harm"]
SER = [("reader_family", "V-Read, map refit here", plot.SERIES[0], "o", 46),
       ("probe", "linear probe (labelled)", plot.SERIES[2], "s", 34),
       ("hidden", "raw reading", plot.SERIES[1], "^", 34),
       ("reader", "V-Read, map from COCO", plot.MUTED, "v", 34)]

plot.use()
d = json.load(open(SRC))
fracs = sorted({round(r["frac"], 2) for r in d["results"]})
frac = min(fracs, key=lambda z: abs(z - 0.2))
by = {(r["held"], round(r["frac"], 2)): r["decision"] for r in d["results"] if r.get("decision")}
rows = [h for h in ORDER if (h, frac) in by]

fig, ax = plt.subplots(figsize=(7.4, 3.6))
for i, held in enumerate(rows):
    y = len(rows) - 1 - i
    dec = by[(held, frac)]
    vals = [dec[k] for k, *_ in SER]
    ax.plot([min(vals), max(vals)], [y, y], color=plot.AXIS, lw=1.0, zorder=1)
    for k, _lab, c, mk, sz in SER:
        ax.scatter(dec[k], y, s=sz, color=c, marker=mk, zorder=3, linewidths=0)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([NAMES.get(h, h) for h in reversed(rows)], fontsize=9)
ax.set_xlim(0.3, 1.02)
ax.set_xlabel(f"AUROC, does the model refuse this item, at depth {frac:.2g}")
ax.axvline(0.5, color=plot.AXIS, lw=1.0, zorder=1)
ax.text(0.503, len(rows) - 0.35, "chance", fontsize=8.5, color=plot.INK3, ha="left", va="center")
plot.despine(ax)
ax.grid(axis="y", visible=False)
h = [plt.Line2D([], [], color=c, marker=mk, ls="none", label=lab) for _k, lab, c, mk, _s in SER]
fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.18))
fig.tight_layout()
plot.save(fig, OUT)
print("wrote", OUT)
