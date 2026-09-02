# SPDX-License-Identifier: Apache-2.0
"""Refusal steering, redrawn from results/steer/<model>_refusal_steer.json."""
import argparse, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import numpy as np, matplotlib.pyplot as plt
from jvlm import plot

ap = argparse.ArgumentParser(); ap.add_argument("json"); ap.add_argument("--out", default="figures/refusal_steer.png")
a = ap.parse_args(); plot.use()
d = json.load(open(a.json)); rows = d["results"]
base = [r for r in rows if r["direction"] == "baseline"][0]
NAME = {"V-Write": "V-Write", "DIM (labelled)": "difference of means, labelled", "raw (the final score)": "the raw direction",
        "read inverted": "V-Read, inverted", "random": "random, same strength"}
COL = {"V-Write": plot.SERIES[0], "difference of means, labelled": plot.SERIES[5], "the raw direction": plot.SERIES[1],
       "V-Read, inverted": plot.SERIES[6], "random, same strength": plot.MUTED}
agg = defaultdict(dict)
for r in rows:
    if r["direction"] == "baseline" or r["channel"] != "ans":
        continue
    agg[(round(r["frac"], 2), NAME.get(r["direction"], r["direction"]))][r["alpha"]] = r
fracs = sorted({k[0] for k in agg}); als = sorted({a_ for v in agg.values() for a_ in v})
fig, axs = plt.subplots(1, len(fracs), figsize=(3.5 * len(fracs), 3.1), sharey=True)
axs = list(np.atleast_1d(axs))
for ax, f in zip(axs, fracs):
    for nm in COL:
        v = agg.get((f, nm))
        if not v:
            continue
        ax.plot(als, [v[x]["refusal_benign_after_add"] if x in v else np.nan for x in als], color=COL[nm], lw=1.6, marker="o", ms=4)
    ax.axhline(base["refusal_benign_after_add"], color=plot.INK2, lw=0.8, ls=":")
    ax.text(als[-1], base["refusal_benign_after_add"] + 0.03, "no edit", ha="right", fontsize=8.5, color=plot.INK2)
    ax.set_title(f"depth {f:g}", loc="left", fontsize=10, color=plot.INK2)
    ax.set_xlabel("dose, as a fraction of the state’s own size"); plot.despine(ax); ax.set_ylim(-0.03, 1.05)
axs[0].set_ylabel("benign prompts now refused")
h = [plt.Line2D([], [], color=COL[k], marker="o", label=k) for k in COL]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.14))
fig.tight_layout(); plot.save(fig, a.out)
