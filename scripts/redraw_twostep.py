# SPDX-License-Identifier: Apache-2.0
"""Redraw the two-step figures from results/twostep/<model>_twostep.json."""
import argparse, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import numpy as np, matplotlib.pyplot as plt
from jvlm import plot

ap = argparse.ArgumentParser(); ap.add_argument("json"); ap.add_argument("--figs", default="figures/twostep")
a = ap.parse_args(); os.makedirs(a.figs, exist_ok=True)
d = json.load(open(a.json)); name = os.path.basename(a.json).replace("_twostep.json", "")
NICE = {"Qwen3-VL-2B-Instruct": "Qwen3-VL-2B", "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B"}
nice = NICE.get(name, name)
plot.use()

# 1. when the answer arrives
AC = d.get("answer_curves") or {}
if AC:
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 2.9), sharey=True)
    SER = [("main_answer", "the two-step answer", plot.SERIES[0], "-"),
           ("control_answer", "the one-step answer", plot.SERIES[2], "--"),
           ("main_landmark", "the landmark’s colour, the trap", plot.SERIES[1], "-")]
    for ax, t in zip(axs, ("read", "logit lens")):
        xs = sorted(float(k) for k in AC[t])
        for key, lab, c, ls in SER:
            ax.plot(xs, [AC[t][f"{x}"][key] if f"{x}" in AC[t] else AC[t][str(x)][key] for x in xs], color=c, lw=1.5, ls=ls)
        ax.set_title({"read": "V-Read"}.get(t, t), loc="left", fontsize=10, color=plot.INK2); ax.set_xlabel("depth"); plot.despine(ax)
    axs[0].set_ylabel("top colour is this one"); axs[0].set_ylim(-0.02, 1.04)
    h = [plt.Line2D([], [], color=c, ls=ls, label=lab) for _, lab, c, ls in SER]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(nice, x=0.01, ha="left", fontsize=10, color=plot.INK3 if hasattr(plot, "INK3") else plot.INK2)
    fig.tight_layout(); plot.save(fig, f"{a.figs}/{name}_twostep_answer.png"); plt.close(fig)

# 2. where the write sits, and what covering costs
if d.get("write_share"):
 KEYS = ["landmark", "referent", "distractor", "floor"]
 LAB = {"landmark": "the landmark", "referent": "the object asked about", "distractor": "a third object", "floor": "bare floor"}
 COL = dict(zip(KEYS, [plot.SERIES[1], plot.SERIES[0], plot.SERIES[2], "#8a8880"]))
 xs = sorted(float(k) for k in d["write_share"])
 fig, axs = plt.subplots(1, 2, figsize=(7.8, 2.9))
 for k in KEYS:
    axs[0].plot(xs, [d["write_share"][str(x)][k] for x in xs], color=COL[k], lw=1.5)
    axs[1].plot(xs, [abs(d["covering"][str(x)][k]) for x in xs], color=COL[k], lw=1.5)
 plot.chance_line(axs[0], 1.0)
 axs[0].set_yscale("log"); axs[0].set_ylabel("share of V-Write, over chance")
 axs[1].set_ylabel("cost of covering it,\nin the answer’s log-odds")
 axs[0].set_title("where V-Write sits", loc="left", fontsize=10, color=plot.INK2)
 axs[1].set_title("what the region is worth", loc="left", fontsize=10, color=plot.INK2)
 for ax in axs: ax.set_xlabel("depth"); plot.despine(ax)
 h = [plt.Line2D([], [], color=COL[k], label=LAB[k]) for k in KEYS]
 fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.1))
 fig.tight_layout(); plot.save(fig, f"{a.figs}/{name}_twostep_regions.png"); plt.close(fig)

# 3. the write, by depth and place
if d.get("swap"):
    agg = defaultdict(list)
    for r in d["swap"]:
        agg[(r["direction"], r["channel"], r.get("frac", d.get("depth_swap")), r["alpha"])].append(r["said_tgt_add"])
    als = sorted({r["alpha"] for r in d["swap"]}); frs = sorted({k[2] for k in agg})
    if len(frs) > 1 or len(als) > 1:
        fig, axs = plt.subplots(1, 2, figsize=(7.8, 2.9), sharey=True)
        for ax, chan in zip(axs, ("img", "all")):
            for f_, c in zip(frs, plot.SERIES):
                y = [np.mean(agg[("write", chan, f_, al)]) if agg.get(("write", chan, f_, al)) else np.nan for al in als]
                ax.plot(als, y, color=c, lw=1.5, marker="o", ms=4)
            y = [np.mean(agg[("random", chan, frs[-1], al)]) if agg.get(("random", chan, frs[-1], al)) else np.nan for al in als]
            ax.plot(als, y, color="#c9c8c1", lw=1.2)
            ax.set_title("written into the picture" if chan == "img" else "written everywhere", loc="left", fontsize=10, color=plot.INK2)
            ax.set_xlabel("dose"); plot.despine(ax)
        axs[0].set_ylabel("answer becomes the written colour"); axs[0].set_ylim(-0.02, 1.04)
        h = [plt.Line2D([], [], color=c, marker="o", label=f"depth {f_:.2f}") for f_, c in zip(frs, plot.SERIES)] + \
            [plt.Line2D([], [], color="#c9c8c1", label="random, same strength")]
        fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.1))
        fig.tight_layout(); plot.save(fig, f"{a.figs}/{name}_twostep_write.png"); plt.close(fig)
print("redrew", name)
