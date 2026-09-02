# SPDX-License-Identifier: Apache-2.0
"""Redraw the one-word-question figures from results/tasks/<model>_tasks.json.

Two figures per model.  The first is accuracy among the candidate words against
depth, V-Read beside the logit lens.  The second follows one item per question:
the log-probability the reading gives each candidate word, at every depth, with
the true answer drawn heavy.  The second figure gets a row per instrument, so
the depth at which the right word separates can be compared directly.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from jvlm import plot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("json")
ap.add_argument("--figs", default="figures/tasks")
a = ap.parse_args()
plot.use()

NICE = {"colour": "colour of an object", "shape": "shape of an object", "material": "what it is made of",
        "count": "how many of a shape", "spatial": "left or right",
        "twostep": "colour of the object beside another"}
d = json.load(open(a.json))
name = os.path.basename(a.json).replace("_tasks.json", "")
res = d["tasks"]
TASKS = [k for k in ("colour", "shape", "material", "count", "spatial", "twostep") if k in res]
os.makedirs(a.figs, exist_ok=True)

# ---- accuracy against depth ------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
for ax, tname in zip(axs, ("V-Read", "logit lens")):
    for k, c in zip(TASKS, plot.SERIES):
        ax.plot(res[k]["depths"], res[k]["accuracy"][tname], color=c, lw=1.5)
    ax.set_title(tname, loc="left", fontsize=10, color=plot.INK2)
    ax.set_xlabel("depth"); plot.despine(ax); ax.set_ylim(0, 1.02)
axs[0].set_ylabel("answer is the right one")
h = [plt.Line2D([], [], color=c, label=NICE.get(k, k)) for k, c in zip(TASKS, plot.SERIES)]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.16))
fig.tight_layout(); plot.save(fig, f"{a.figs}/{name}_tasks.png"); plt.close(fig)

# ---- one item per question, both instruments -------------------------------
KEYS = [("read", "V-Read"), ("logit", "the logit lens")]
fig, axs = plt.subplots(2, len(TASKS), figsize=(2.35 * len(TASKS), 4.6), sharey="row", squeeze=False)
for j, k in enumerate(TASKS):
    r = res[k]
    ex = d["examples"][k] if isinstance(d.get("examples"), dict) else r.get("example", {})
    words = r["words"]; truth_w = ex.get("answer")
    for i, (field, label) in enumerate(KEYS):
        ax = axs[i, j]
        tr = np.asarray(ex[field], float)
        for wi, w in enumerate(words):
            ax.plot(r["depths"], tr[:, wi], color=plot.SERIES[wi % len(plot.SERIES)],
                    lw=1.9 if w == truth_w else 0.8, alpha=1.0 if w == truth_w else 0.45)
        plot.despine(ax)
        if i == 0:
            ax.set_title(f"{NICE.get(k, k)}\nanswer: {truth_w}", loc="left", fontsize=8.5, color=plot.INK2)
        else:
            ax.set_xlabel("depth")
        if j == 0:
            ax.set_ylabel(f"{label}\nlog-probability", fontsize=9)
fig.tight_layout(); plot.save(fig, f"{a.figs}/{name}_task_traces.png"); plt.close(fig)
print(f"wrote {a.figs}/{name}_tasks.png and {a.figs}/{name}_task_traces.png")
