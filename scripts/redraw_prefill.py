# SPDX-License-Identifier: Apache-2.0
"""Redraw the prefill figure from results/prefill/<model>_prefill.json.

Two panels per model: AUROC (harmful vs benign prefilled items, both
complying) against depth for the four instruments, and a grouped bar chart of
the mean reading by prefill condition for harmful and benign items, with
error bars at one standard deviation.
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
ap.add_argument("jsons", nargs="+")
ap.add_argument("--out", default="figures/prefill.png")
a = ap.parse_args()
plot.use()

NICE = {"Qwen3-VL-2B-Instruct": "Qwen3-VL-2B", "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B"}
INSTR = [("reader", "the read", plot.SERIES[0]),
          ("write", "the write", plot.SERIES[2]),
          ("identity", "the raw reading", plot.SERIES[1]),
          ("probe", "linear probe (labelled)", plot.SERIES[3])]

paths = []
for p in a.jsons:
    if not os.path.exists(p):
        print(f"missing, skipped: {p}")
        continue
    paths.append(p)
if not paths:
    print("no prefill JSONs found; nothing to draw")
    sys.exit(0)

nrows = len(paths)
fig, axs = plt.subplots(nrows, 2, figsize=(7.8, 3.1 * nrows), squeeze=False)

used_instr = set()
for i, path in enumerate(paths):
    d = json.load(open(path))
    name = os.path.basename(path).replace("_prefill.json", "")
    nice = NICE.get(name, name)
    rows = sorted(d["rows"], key=lambda r: r["frac"])
    xs = [r["frac"] for r in rows]
    ax0 = axs[i, 0]
    for key, lab, c in INSTR:
        pts = [(r["frac"], r["auroc"][key]) for r in rows if key in r["auroc"]]
        if not pts:
            continue
        px, py = zip(*pts)
        ax0.plot(px, py, color=c, lw=1.5, marker="o" if len(pts) < len(rows) else None, ms=4)
        used_instr.add(key)
    plot.chance_line(ax0, 0.5)
    ax0.set_title(f"{nice}, harmful vs benign under the prefill", loc="left", fontsize=9.5, color=plot.INK2)
    ax0.set_xlabel("depth"); ax0.set_ylabel("AUROC"); ax0.set_ylim(0.4, 1.02)
    plot.despine(ax0)

    ax1 = axs[i, 1]
    rbc = d.get("reading_by_condition", {})
    conditions = list(d.get("prefills", rbc).keys()) or list(rbc.keys())
    conditions = [c for c in conditions if c in rbc]
    if conditions:
        xsc = np.arange(len(conditions))
        for k, (kind, col) in enumerate((("harmful", plot.SERIES[1]), ("benign", plot.SERIES[0]))):
            vals = [rbc[c][kind] for c in conditions]
            errs = [rbc[c].get(f"{kind}_sd") for c in conditions]
            errs = errs if all(e is not None for e in errs) else None
            ax1.bar(xsc + (k - 0.5) * 0.36, vals, 0.34, yerr=errs, capsize=2.5, color=col, label=kind,
                    error_kw=dict(ecolor=plot.INK2, elinewidth=0.8))
        ax1.set_xticks(xsc)
        ax1.set_xticklabels(conditions, fontsize=9)
        instrument = next((rbc[c].get("instrument") for c in conditions if rbc[c].get("instrument")), None)
        NICE_INSTR = {key: lab for key, lab, _ in INSTR}
        lab = f"mean reading ({NICE_INSTR.get(instrument, instrument)})" if instrument else "mean reading"
        ax1.set_ylabel(lab)
        depth_mid = d.get("depth_mid")
        title = "reading by prefill condition" + (f", depth {depth_mid:.2f}" if depth_mid is not None else "")
        ax1.set_title(f"{nice}, {title}", loc="left", fontsize=9.5, color=plot.INK2)
        ax1.legend(frameon=False, fontsize=9, loc="best")
    plot.despine(ax1)

axs[-1, 0].set_xlabel("depth")
fig.tight_layout()
handles = [plt.Line2D([], [], color=c, lw=1.5, label=lab) for key, lab, c in INSTR if key in used_instr]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
           bbox_to_anchor=(0.27, -0.12 / nrows))
plot.save(fig, a.out)
print("wrote", a.out)
