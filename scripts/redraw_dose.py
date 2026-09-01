# SPDX-License-Identifier: Apache-2.0
"""Redraw the dose figure from a finished steering run, with the current names."""
import argparse, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import numpy as np, matplotlib.pyplot as plt
from jvlm import plot

ap = argparse.ArgumentParser(); ap.add_argument("json"); ap.add_argument("--frac", type=float, default=0.48)
ap.add_argument("--out", default=None); a = ap.parse_args()
d = json.load(open(a.json)); plot.use()
name = os.path.basename(a.json).replace("_steer.json", "")
NAMES = {"writer": "write", "push": "write", "lens": "write", "DIM (labelled)": "difference of means, labelled",
         "difference of means (labelled)": "difference of means, labelled", "logit-lens": "output-layer row",
         "logit lens": "output-layer row", "output-layer row": "output-layer row", "random": "random, same strength"}
COL = {"write": plot.SERIES[0], "difference of means, labelled": "#7a7a7a", "output-layer row": plot.SERIES[1], "random, same strength": "#c9c8c1"}
agg = defaultdict(list)
for it in d["items"]:
    for r in it["runs"]:
        agg[(r["mode"], NAMES.get(r["direction"], r["direction"]), round(r["frac"], 2), r["alpha"])].append((r["said_B"], r["retained"]))
al = d["alphas"]
fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.1), gridspec_kw=dict(wspace=0.34))
ends = []
for dn in ("write", "difference of means, labelled", "output-layer row", "random, same strength"):
    ys, rs = [], []
    for x in al:
        v = agg.get(("add", dn, a.frac, x), [])
        ys.append(np.mean([z[0] for z in v]) if v else np.nan)
        keep = [z[1] for z in v if z[1] is not None]
        rs.append(np.mean(keep) if keep else np.nan)
    axs[0].plot(al, ys, color=COL[dn], lw=1.6, marker="o", ms=4)
    axs[1].plot(al, rs, color=COL[dn], lw=1.6, marker="o", ms=4)
    ends.append((ys[-1], dn, COL[dn]))
ends.sort(key=lambda z: -z[0])
for (yv, dn, c), yy in zip(ends, plot.stagger([z[0] for z in ends], 0.085)):
    plot.end_label(axs[0], al[-1], yy, dn, c)
axs[0].set_ylabel(f"answer becomes the written word\n(depth {a.frac})"); axs[1].set_ylabel("unrelated question still right")
for ax in axs: ax.set_xlabel("dose, as a fraction of the state’s own size"); ax.set_ylim(-0.02, 1.02); plot.despine(ax)
fig.tight_layout(); plot.save(fig, a.out or f"/workspace/vlm/figures/steer/{name}_steer_dose.png")
