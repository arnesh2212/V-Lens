# SPDX-License-Identifier: Apache-2.0
"""The gate figures, drawn from the saved JSON so they can be redrawn at will.

Two panels per model: the decision (refused vs complied) read at the answer
slot against depth, for the lens, the raw reading, the labelled
difference of means and the trained probe; and the interventional check, the
held-out R^2 with which each direction predicts what patching the answer slot
does to the refusal margin.  With several models, one figure per panel with a
small multiple per model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt

from jvlm import plot

ap = argparse.ArgumentParser()
ap.add_argument("jsons", nargs="+")
ap.add_argument("--out", default="/home/arnesg/vlm/figures")
a = ap.parse_args()
plot.use()
os.makedirs(a.out, exist_ok=True)

runs = [json.load(open(p)) for p in a.jsons]
short = lambda r: r["model"].split("/")[-1].replace("-Instruct", "")
C = {"verdict": plot.BLUE, "identity": plot.ORANGE, "dim": plot.MUTED, "probe": plot.AQUA}


def curve(d, key):
    c = d["curves"][key]
    ls = sorted(c, key=int)
    return [c[l]["frac"] for l in ls], [c[l]["auroc"] for l in ls]


# ---- panel 1: the decision against depth -----------------------------------
fig, axs = plot.small_multiples(len(runs), cols=len(runs), panel=(4.2, 3.1))
for ax, d in zip(axs, runs):
    x, y = curve(d, "identity(HiddenDetect)"); ax.plot(x, y, color=C["identity"], lw=1.6)
    x, y = curve(d, "DIM(labelled,5-fold)"); ax.plot(x, y, color=C["dim"], lw=1.6)
    x, y = curve(d, "verdict(carrier,refusal)"); ax.plot(x, y, color=C["verdict"], marker="o")
    pr = d["probe_refused"]
    fr = {int(l): v["frac"] for l, v in d["calibration"].items()}   # layer -> depth, as the run defined it
    px = [fr[r["layer"]] for r in pr]
    ax.plot(px, [r["score"] for r in pr], color=C["probe"], marker="s", ls=":", ms=5)
    ax.axhline(0.5, color=plot.AXIS, lw=0.9, zorder=1)
    ax.set_ylim(0.4, 1.02); ax.set_xlim(0, 1)
    ax.set_xlabel("depth"); plot.panel_label(ax, f"{short(d)}, refused {d['refused_rate']:.0%} of items")
axs[0].set_ylabel("AUROC, refused vs complied")
h = [plt.Line2D([], [], color=C["verdict"], marker="o", label="write (no labels)"),
     plt.Line2D([], [], color=C["identity"], label="raw reading (HiddenDetect)"),
     plt.Line2D([], [], color=C["dim"], label="difference of means (labelled)"),
     plt.Line2D([], [], color=C["probe"], marker="s", ls=":", label="linear probe (labelled)")]
fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncols=2, frameon=False)
plot.save(fig, f"{a.out}/gate_decision.png")

# ---- panel 2: the interventional check -------------------------------------
fig, axs = plot.small_multiples(len(runs), cols=len(runs), panel=(4.2, 3.1))
for ax, d in zip(axs, runs):
    cal = d["calibration"]; ls = sorted(cal, key=int)
    x = [cal[l]["frac"] for l in ls]
    ax.plot(x, [cal[l]["identity"]["r2_heldout"] for l in ls], color=C["identity"], marker="o", ms=5)
    ax.plot(x, [cal[l]["verdict"]["r2_heldout"] for l in ls], color=C["verdict"], marker="o")
    ax.axhline(0, color=plot.AXIS, lw=0.9, zorder=1)
    ax.set_ylim(-0.1, 1.02); ax.set_xlim(0, 1)
    ax.set_xlabel("depth"); plot.panel_label(ax, short(d))
axs[0].set_ylabel("held-out R², predicted vs measured\nchange in the refusal margin")
h = [plt.Line2D([], [], color=C["verdict"], marker="o", label="write direction"),
     plt.Line2D([], [], color=C["identity"], marker="o", label="raw direction")]
fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncols=2, frameon=False)
plot.save(fig, f"{a.out}/gate_calibration.png")

# ---- panel 3: distinctness -------------------------------------------------
fig, axs = plot.small_multiples(len(runs), cols=len(runs), panel=(4.2, 2.6))
for ax, d in zip(axs, runs):
    cd = d["cos_dim"]; ls = sorted(cd, key=int)
    x = [cd[l].get("frac", None) for l in ls]
    if any(v is None for v in x):
        x = [d["calibration"][l]["frac"] for l in ls]
    ax.plot(x, [abs(cd[l]["carrier"]) for l in ls], color=C["verdict"], marker="o")
    ax.plot(x, [abs(cd[l]["identity"]) for l in ls], color=C["identity"], marker="o", ms=5)
    ax.axhline(0.9, color=plot.AXIS, lw=0.9, zorder=1)
    ax.set_ylim(0, 1.02); ax.set_xlim(0, 1)
    ax.set_xlabel("depth"); plot.panel_label(ax, short(d))
axs[0].set_ylabel("|cos| with the labelled\ndifference-of-means direction")
h = [plt.Line2D([], [], color=C["verdict"], marker="o", label="write direction"),
     plt.Line2D([], [], color=C["identity"], marker="o", label="raw (the final refusal score)")]
fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncols=2, frameon=False)
plot.save(fig, f"{a.out}/gate_distinct.png")
print("wrote", a.out)
