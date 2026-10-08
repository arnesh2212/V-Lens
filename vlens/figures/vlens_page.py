# SPDX-License-Identifier: Apache-2.0
"""Every figure of the V-Lens page (paper/tool.html), redrawn from saved results.

Each function draws one figure (or one small family of figures) from the JSON
an experiment wrote, so a figure can be redrawn without a GPU.  ``build/figures.py
vlens`` calls them all with the default result paths.  The atlas gallery, the
scenario pages and the slice pages are drawn by their experiments directly.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from vlens.figures import plot
from vlens.paths import FIGURES, RESULTS

NICE = {"Qwen3-VL-2B-Instruct": "Qwen3-VL-2B", "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B", "InternVL3-8B-hf": "InternVL3-8B"}
MODELS = [("Qwen3-VL-2B-Instruct", "Qwen3-VL-2B"), ("Qwen3-VL-8B-Instruct", "Qwen3-VL-8B"), ("InternVL3-8B-hf", "InternVL3-8B")]


def _existing(paths):
    out = []
    for p in paths:
        if os.path.exists(p):
            out.append(p)
        else:
            print(f"missing, skipped: {p}")
    return out


def _legend(fig, handles, ncol, y, fontsize=9):
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=fontsize, bbox_to_anchor=(0.5, y))


# ---------------------------------------------------------------- the gate
def gate(jsons, out_dir=FIGURES):
    """The gate figures: the decision against depth, the interventional check, distinctness."""
    plot.use()
    os.makedirs(out_dir, exist_ok=True)
    runs = [json.load(open(p)) for p in jsons]
    short = lambda r: r["model"].split("/")[-1].replace("-Instruct", "").replace("-hf", "")
    C = {"verdict": plot.BLUE, "identity": plot.ORANGE, "dim": plot.MUTED, "probe": plot.AQUA}

    def curve(d, key):
        c = d["curves"][key]
        ls = sorted(c, key=int)
        return [c[l]["frac"] for l in ls], [c[l]["auroc"] for l in ls]

    fig, axs = plot.small_multiples(len(runs), cols=len(runs), panel=(4.2, 3.1))
    for ax, d in zip(axs, runs):
        x, y = curve(d, "identity(HiddenDetect)"); ax.plot(x, y, color=C["identity"], lw=1.6)
        x, y = curve(d, "DIM(labelled,5-fold)"); ax.plot(x, y, color=C["dim"], lw=1.6)
        x, y = curve(d, "verdict(carrier,refusal)"); ax.plot(x, y, color=C["verdict"], marker="o")
        pr = d["probe_refused"]
        fr = {int(l): v["frac"] for l, v in d["calibration"].items()}
        ax.plot([fr[r["layer"]] for r in pr], [r["score"] for r in pr], color=C["probe"], marker="s", ls=":", ms=5)
        ax.axhline(0.5, color=plot.AXIS, lw=0.9, zorder=1)
        ax.set_ylim(0.4, 1.02); ax.set_xlim(0, 1)
        ax.set_xlabel("depth"); plot.panel_label(ax, f"{short(d)}, refused {d['refused_rate']:.0%} of items")
    axs[0].set_ylabel("AUROC, refused vs complied")
    h = [plt.Line2D([], [], color=C["verdict"], marker="o", label="V-Write (no labels)"),
         plt.Line2D([], [], color=C["identity"], label="raw reading (HiddenDetect)"),
         plt.Line2D([], [], color=C["dim"], label="difference of means (labelled)"),
         plt.Line2D([], [], color=C["probe"], marker="s", ls=":", label="linear probe (labelled)")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncols=2, frameon=False)
    plot.save(fig, f"{out_dir}/gate_decision.png")

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
    h = [plt.Line2D([], [], color=C["verdict"], marker="o", label="V-Write direction"),
         plt.Line2D([], [], color=C["identity"], marker="o", label="raw direction")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncols=2, frameon=False)
    plot.save(fig, f"{out_dir}/gate_calibration.png")

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
    h = [plt.Line2D([], [], color=C["verdict"], marker="o", label="V-Write direction"),
         plt.Line2D([], [], color=C["identity"], marker="o", label="raw (the final refusal score)")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncols=2, frameon=False)
    plot.save(fig, f"{out_dir}/gate_distinct.png")


def gate_channels(out=f"{FIGURES}/gate_channels.png"):
    """Where to read: the decision from each group of positions, from results/gate/*_total_verdict.json."""
    SER = [("total", "V-Write, everywhere", plot.SERIES[0], "-", "o"),
           ("txt", "V-Write, the words", "#4a3aa7", "-", "^"),
           ("ans", "V-Write, the answer spot", plot.SERIES[0], "--", None),
           ("img", "V-Write, the picture", "#8a8880", "-", None),
           ("identity", "raw reading", plot.SERIES[1], "-", None),
           ("probe", "linear probe (labelled)", plot.SERIES[2], ":", "s")]
    plot.use()
    fig, axs = plt.subplots(1, len(MODELS), figsize=(3.9 * len(MODELS), 3.0), sharey=True)
    axs = list(axs) if len(MODELS) > 1 else [axs]
    for ax, (mk, nice) in zip(axs, MODELS):
        p = os.path.join(RESULTS, "gate", f"{mk}_total_verdict.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p)); ls = sorted(d, key=lambda z: d[z]["frac"]); x = [d[l]["frac"] for l in ls]
        for key, lab, c, style, mkr in SER:
            if key not in d[ls[0]]:
                continue
            ax.plot(x, [d[l][key] for l in ls], color=c, lw=1.5 if key in ("total", "identity") else 1.1, ls=style, marker=mkr, ms=4)
        plot.chance_line(ax, 0.5)
        ax.set_title(nice, loc="left", fontsize=10, color=plot.INK2); ax.set_xlabel("depth"); plot.despine(ax)
    axs[0].set_ylabel("AUROC, refused vs complied"); axs[0].set_ylim(0.4, 1.02)
    h = [plt.Line2D([], [], color=c, ls=style, marker=mkr, label=lab) for _, lab, c, style, mkr in SER]
    _legend(fig, h, 3, -0.13)
    fig.tight_layout(); plot.save(fig, out)


def reader_decision(src=f"{RESULTS}/gate/reader_regression.json", out=f"{FIGURES}/reader_decision.png"):
    """The decision read by V-Read, a labelled probe and the raw reading, against depth."""
    d = json.load(open(src))
    plot.use()
    SER = [("ridge", "V-Read (no labels)", plot.SERIES[0], "-", "o"),
           ("probe", "linear probe (labelled)", plot.SERIES[2], ":", "s"),
           ("identity_perfold", "raw reading (HiddenDetect)", plot.SERIES[1], "-", None)]
    models = [k for k in d if k in NICE]
    fig, axs = plt.subplots(1, len(models), figsize=(3.5 * len(models), 3.0), sharey=True)
    axs = [axs] if len(models) == 1 else list(axs)
    for ax, mk in zip(axs, models):
        ls = sorted(d[mk], key=lambda z: d[mk][z]["frac"])
        x = [d[mk][l]["frac"] for l in ls]
        for key, lab, c, style, mk_ in SER:
            ax.plot(x, [d[mk][l][key] for l in ls], color=c, lw=1.6, ls=style, marker=mk_, ms=4.5)
        ax.set_title(NICE[mk], loc="left", fontsize=10, color=plot.INK2)
        ax.set_xlabel("depth"); plot.despine(ax)
    axs[0].set_ylabel("AUROC, refused vs complied"); axs[0].set_ylim(0.4, 1.02)
    h = [plt.Line2D([], [], color=c, ls=style, marker=mk_, label=lab) for _, lab, c, style, mk_ in SER]
    _legend(fig, h, 3, -0.06, 9.5)
    fig.tight_layout()
    plot.save(fig, out)


def loo_refit(src=f"{RESULTS}/reader/Qwen3-VL-8B-Instruct_loo.json", out=f"{FIGURES}/loo_refit.png"):
    """Leave-one-benchmark-out at a fifth of the depth, as a dot plot: one row per held-out set."""
    NAMES = {"mmsb_typo": "MM-SafetyBench", "figstep": "FigStep", "hades": "HADES", "vlguard": "VLGuard",
             "vlsbench": "VLSBench", "siuo": "SIUO", "mossbench_contrast": "MOSSBench", "spavl_harm": "SPA-VL"}
    ORDER = ["mmsb_typo", "figstep", "hades", "vlguard", "vlsbench", "siuo", "mossbench_contrast", "spavl_harm"]
    SER = [("reader_family", "V-Read, map refit here", plot.SERIES[0], "o", 46),
           ("probe", "linear probe (labelled)", plot.SERIES[2], "s", 34),
           ("hidden", "raw reading", plot.SERIES[1], "^", 34),
           ("reader", "V-Read, map from COCO", plot.MUTED, "v", 34)]
    plot.use()
    d = json.load(open(src))
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
    _legend(fig, h, 2, -0.18)
    fig.tight_layout()
    plot.save(fig, out)


# ---------------------------------------------------------------- presence, steering
def presence(jsons, out=f"{FIGURES}/presence_hard.png"):
    """Is the object there, and among the model's yes answers, true vs hallucinated: from <model>_regions.json."""
    plot.use()
    SER = [("reader", "V-Read (no labels)", plot.SERIES[0], "-"), ("probe", "linear probe (labelled)", plot.SERIES[2], ":"),
           ("logit lens", "logit lens", plot.SERIES[1], "-"), ("write", "V-Write", "#8a8880", "-")]
    fig, axs = plt.subplots(2, len(jsons), figsize=(3.9 * len(jsons), 5.2), sharex=True, sharey=True)
    axs = np.atleast_2d(axs).reshape(2, -1)
    for j, path in enumerate(jsons):
        d = json.load(open(path)); E = d["emergence"]; name = os.path.basename(path).replace("_regions.json", "")
        dep = E["depth_of"]; ls = sorted(dep, key=lambda z: dep[z]); xs = [dep[l] for l in ls]
        for key, lab, c, style in SER:
            axs[0, j].plot(xs, [E["auroc"][key][l] for l in ls], color=c, lw=1.5 if key == "reader" else 1.1, ls=style)
            y = [E["hallucination_auroc"][key][l] for l in ls]
            if not all(np.isnan(np.array(y, float))):
                axs[1, j].plot(xs, y, color=c, lw=1.5 if key == "reader" else 1.1, ls=style)
        for ax in (axs[0, j], axs[1, j]):
            plot.chance_line(ax, 0.5); plot.despine(ax)
        axs[0, j].set_title(NICE.get(name, name), loc="left", fontsize=10, color=plot.INK2)
        axs[1, j].set_xlabel("depth")
        axs[1, j].text(0.02, 0.06, f"{E['n_hallucinated']} hallucinated yes answers", transform=axs[1, j].transAxes, fontsize=8, color=plot.INK2)
    axs[0, 0].set_ylabel("is the object there?\nAUROC, present vs absent")
    axs[1, 0].set_ylabel("among the model’s yes answers,\ntrue vs hallucinated")
    axs[0, 0].set_ylim(0.15, 1.02)
    h = [plt.Line2D([], [], color=c, ls=style, label=lab) for _, lab, c, style in SER]
    _legend(fig, h, 4, -0.05)
    fig.tight_layout(); plot.save(fig, out)


def steer_dose(jsons, frac=0.48, out=f"{FIGURES}/steer_dose.png"):
    """The dose figure: the answer becomes the written word, per direction, one panel per model."""
    plot.use()
    NAMES = {"writer": "V-Write", "push": "V-Write", "lens": "V-Write", "DIM (labelled)": "difference of means, labelled",
             "difference of means (labelled)": "difference of means, labelled", "logit-lens": "output-layer row",
             "logit lens": "output-layer row", "output-layer row": "output-layer row", "random": "random, same strength"}
    ORDER = ["V-Write", "difference of means, labelled", "output-layer row", "random, same strength"]
    COL = {"V-Write": plot.SERIES[0], "difference of means, labelled": "#7a7a7a", "output-layer row": plot.SERIES[1], "random, same strength": "#c9c8c1"}
    fig, axs = plt.subplots(1, len(jsons), figsize=(4.1 * len(jsons), 3.1), sharey=True)
    axs = list(axs) if len(jsons) > 1 else [axs]
    for ax, path in zip(axs, jsons):
        d = json.load(open(path)); name = os.path.basename(path).replace("_steer.json", "")
        agg = defaultdict(list)
        for it in d["items"]:
            for r in it["runs"]:
                agg[(r["mode"], NAMES.get(r["direction"], r["direction"]), round(r["frac"], 2), r["alpha"])].append(r["said_B"])
        fr = min({k[2] for k in agg}, key=lambda z: abs(z - frac))
        for dn in ORDER:
            y = [np.mean(agg[("add", dn, fr, x)]) if agg.get(("add", dn, fr, x)) else np.nan for x in d["alphas"]]
            ax.plot(d["alphas"], y, color=COL[dn], lw=1.6, marker="o", ms=4)
        ax.set_title(f"{NICE.get(name, name)}, depth {fr:g}", loc="left", fontsize=10, color=plot.INK2)
        ax.set_xlabel("dose, as a fraction of the state’s own size"); plot.despine(ax); ax.set_ylim(-0.02, 1.04)
    axs[0].set_ylabel("answer becomes the written word")
    h = [plt.Line2D([], [], color=COL[k], marker="o", label=k) for k in ORDER]
    _legend(fig, h, 2, -0.17)
    fig.tight_layout(); plot.save(fig, out)


def refusal_steer(src, out=f"{FIGURES}/refusal_steer.png"):
    """Refusal steering: benign prompts now refused, per direction and dose, one panel per depth."""
    plot.use()
    d = json.load(open(src)); rows = d["results"]
    base = [r for r in rows if r["direction"] == "baseline"][0]
    NAME = {"V-Write": "V-Write", "write": "V-Write",
            "DIM (labelled)": "difference of means, labelled", "raw (the final score)": "the raw direction",
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
    _legend(fig, h, 3, -0.14)
    fig.tight_layout(); plot.save(fig, out)


# ---------------------------------------------------------------- atlas, cascade, prefill, regions
def atlas_summary(src, out=f"{FIGURES}/atlas_summary.png"):
    """Rank agreement with patching and top-1 region hit rate, refusal and yes/no, six instruments."""
    if not os.path.exists(src):
        print(f"missing, nothing to draw: {src}"); return
    plot.use()
    INSTR = [("V-Write, path", plot.SERIES[0]), ("V-Write, one point", plot.SERIES[4]), ("V-Write, averaged", plot.SERIES[2]),
             ("identity", plot.SERIES[1]), ("grad x act", plot.SERIES[3]), ("GLIMPSE", "#7a7a7a")]
    KINDS = [("refusal", "refusal margin, typography items"), ("yesno", "yes/no margin, COCO photographs")]

    def rename(o):  # runs from before the rename store the write maps under "write, ..."
        if isinstance(o, dict):
            return {(k.replace("write,", "V-Write,") if isinstance(k, str) and k.startswith("write,") else k): rename(v) for k, v in o.items()}
        if isinstance(o, list):
            return [rename(x) for x in o]
        return o

    d = rename(json.load(open(src)))
    name = os.path.basename(src).replace("_atlas.json", ""); nice = NICE.get(name, name)
    depths = d["depths"]; layers = sorted(depths, key=lambda l: depths[l]); xs = [depths[l] for l in layers]
    regions = d.get("regions"); summary = d["summary"]
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
    handles = [plt.Line2D([], [], color=c, lw=1.5, ls="--" if key == "GLIMPSE" else "-", label=LAB[key]) for key, c in INSTR if key in used]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    plot.save(fig, out)


def cascade(jsons, out=f"{FIGURES}/cascade.png"):
    """Order of arrival: AUROC against depth for content, harm, refusal and opener, V-Read beside the logit lens."""
    paths = _existing(jsons)
    if not paths:
        return
    plot.use()
    SCORES = [("content", "content", plot.SERIES[0]), ("harm", "harm", plot.SERIES[1]), ("refusal", "refusal", plot.SERIES[2]), ("opener", "opener", plot.SERIES[3])]
    TRANSPORTS = [("reader", "V-Read"), ("logit lens", "logit lens")]
    nrows = len(paths)
    fig, axs = plt.subplots(nrows, 2, figsize=(7.8, 2.9 * nrows), sharey=True, sharex=True, squeeze=False)
    used_scores = set(); all_y = []
    for i, path in enumerate(paths):
        d = json.load(open(path)); name = os.path.basename(path).replace("_cascade.json", ""); nice = NICE.get(name, name)
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
                used_scores.add(key); all_y.extend(y for y in ys if y is not None)
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
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.1 / nrows))
    fig.tight_layout()
    plot.save(fig, out)


def prefill(jsons, out=f"{FIGURES}/prefill.png"):
    """The objection the model never speaks: AUROC under the prefill, and the mean reading by condition."""
    paths = _existing(jsons)
    if not paths:
        return
    plot.use()
    INSTR = [("reader_family", "V-Read, map refit here", plot.SERIES[0]), ("reader", "V-Read, map from elsewhere", plot.SERIES[4]),
             ("V-Write", "V-Write", plot.SERIES[2]), ("identity", "the raw reading", plot.SERIES[1]), ("probe", "linear probe (labelled)", plot.SERIES[3])]
    nrows = len(paths)
    fig, axs = plt.subplots(nrows, 2, figsize=(7.8, 3.1 * nrows), squeeze=False)
    used_instr = set()
    for i, path in enumerate(paths):
        d = json.load(open(path)); name = os.path.basename(path).replace("_prefill.json", ""); nice = NICE.get(name, name)
        rows = sorted(d["rows"], key=lambda r: r["frac"])
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
        conditions = [c for c in (list(d.get("prefills", rbc).keys()) or list(rbc.keys())) if c in rbc]
        if conditions:
            xsc = np.arange(len(conditions))
            for k, (kind, col) in enumerate((("harmful", plot.SERIES[1]), ("benign", plot.SERIES[0]))):
                vals = [rbc[c][kind] for c in conditions]
                errs = [rbc[c].get(f"{kind}_sd") for c in conditions]
                errs = errs if all(e is not None for e in errs) else None
                ax1.bar(xsc + (k - 0.5) * 0.36, vals, 0.34, yerr=errs, capsize=2.5, color=col, label=kind, error_kw=dict(ecolor=plot.INK2, elinewidth=0.8))
            ax1.set_xticks(xsc); ax1.set_xticklabels(conditions, fontsize=9)
            instrument = next((rbc[c].get("instrument") for c in conditions if rbc[c].get("instrument")), None)
            NICE_INSTR = {"reader_family": "V-Read", "reader": "V-Read", "V-Write": "V-Write", "identity": "raw"}
            ax1.set_ylabel(f"mean reading ({NICE_INSTR.get(instrument, instrument)})" if instrument else "mean reading")
            depth_mid = d.get("depth_mid")
            ax1.set_title(f"{nice}, reading by prefill condition" + (f", depth {depth_mid:.2f}" if depth_mid is not None else ""), loc="left", fontsize=9.5, color=plot.INK2)
            ax1.legend(frameon=False, fontsize=9, loc="best")
        plot.despine(ax1)
    axs[-1, 0].set_xlabel("depth")
    fig.tight_layout()
    handles = [plt.Line2D([], [], color=c, lw=1.5, label=lab) for key, lab, c in INSTR if key in used_instr]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.27, -0.12 / nrows))
    plot.save(fig, out)


def regions(jsons, out=f"{FIGURES}/regions.png"):
    """Naming the object, mass in its boxes over chance, and the pointing game, against depth."""
    paths = _existing(jsons)
    if not paths:
        return
    plot.use()
    ALIAS = {"V-Write, this item": "write, this item", "V-Write": "write"}
    INSTR = [("V-Write, this item", plot.SERIES[0]), ("V-Write", plot.SERIES[4]), ("logit lens", plot.SERIES[1]), ("logit lens, centred", plot.SERIES[2])]
    PANELS = [("accuracy", "naming the object", None), ("concentration_median", "mass in the object's boxes, over chance", 1.0), ("pointing", "pointing game", None)]
    nrows = len(paths)
    fig, axs = plt.subplots(nrows, 3, figsize=(3.5 * 3, 2.9 * nrows), squeeze=False)
    used_instr = set()
    for i, path in enumerate(paths):
        d = json.load(open(path)); name = os.path.basename(path).replace("_regions.json", ""); nice = NICE.get(name, name)
        depths = d["depths"]; layers = sorted(depths, key=lambda l: depths[l]); xs = [depths[l] for l in layers]
        bank_chance = 1.0 / len(d["bank"]) if d.get("bank") else None
        box_chance = d.get("box_share"); reg = d["regions"]
        for j, (metric_key, title, fixed_chance) in enumerate(PANELS):
            ax = axs[i, j]
            if metric_key not in reg:
                ax.set_visible(False); continue
            met = reg[metric_key]
            for key, c in INSTR:
                src = key if key in met else ALIAS.get(key)
                if src not in met:
                    continue
                ys = [met[src].get(l) for l in layers]
                if all(y is None for y in ys):
                    continue
                ax.plot(xs, ys, color=c, lw=1.5); used_instr.add(key)
            chance = fixed_chance if fixed_chance is not None else (bank_chance if metric_key == "accuracy" else box_chance)
            if chance is not None:
                plot.chance_line(ax, chance)
            ax.set_title(f"{nice}, {title}", loc="left", fontsize=9.5, color=plot.INK2)
            ax.set_xlabel("depth"); plot.despine(ax)
        axs[i, 0].set_ylabel("top-1 accuracy"); axs[i, 1].set_ylabel("mass share / box share"); axs[i, 2].set_ylabel("hit rate")
    fig.tight_layout()
    handles = [plt.Line2D([], [], color=c, lw=1.5, label=key) for key, c in INSTR if key in used_instr]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.14 / nrows))
    plot.save(fig, out)


# ---------------------------------------------------------------- one-word questions, two-step
def tasks(src, figs=f"{FIGURES}/tasks"):
    """When does the answer arrive, for six one-word questions: accuracy by depth, and one item's traces."""
    plot.use()
    NICE_T = {"colour": "colour of an object", "shape": "shape of an object", "material": "what it is made of",
              "count": "how many of a shape", "spatial": "left or right", "twostep": "colour of the object beside another"}
    d = json.load(open(src)); name = os.path.basename(src).replace("_tasks.json", ""); res = d["tasks"]
    TASKS = [k for k in ("colour", "shape", "material", "count", "spatial", "twostep") if k in res]
    os.makedirs(figs, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    for ax, tname in zip(axs, ("V-Read", "logit lens")):
        for k, c in zip(TASKS, plot.SERIES):
            ax.plot(res[k]["depths"], res[k]["accuracy"][tname], color=c, lw=1.5)
        ax.set_title(tname, loc="left", fontsize=10, color=plot.INK2)
        ax.set_xlabel("depth"); plot.despine(ax); ax.set_ylim(0, 1.02)
    axs[0].set_ylabel("answer is the right one")
    h = [plt.Line2D([], [], color=c, label=NICE_T.get(k, k)) for k, c in zip(TASKS, plot.SERIES)]
    _legend(fig, h, 3, -0.16)
    fig.tight_layout(); plot.save(fig, f"{figs}/{name}_tasks.png"); plt.close(fig)
    KEYS = [("read", "V-Read"), ("logit", "the logit lens")]
    fig, axs = plt.subplots(2, len(TASKS), figsize=(2.35 * len(TASKS), 4.6), sharey="row", squeeze=False)
    for j, k in enumerate(TASKS):
        r = res[k]
        ex = d["examples"][k] if isinstance(d.get("examples"), dict) else r.get("example", {})
        words = r["words"]; truth_w = ex.get("answer")
        for i, (field, label) in enumerate(KEYS):
            ax = axs[i, j]; tr = np.asarray(ex[field], float)
            for wi, w in enumerate(words):
                ax.plot(r["depths"], tr[:, wi], color=plot.SERIES[wi % len(plot.SERIES)], lw=1.9 if w == truth_w else 0.8, alpha=1.0 if w == truth_w else 0.45)
            plot.despine(ax)
            if i == 0:
                ax.set_title(f"{NICE_T.get(k, k)}\nanswer: {truth_w}", loc="left", fontsize=8.5, color=plot.INK2)
            else:
                ax.set_xlabel("depth")
            if j == 0:
                ax.set_ylabel(f"{label}\nlog-probability", fontsize=9)
    fig.tight_layout(); plot.save(fig, f"{figs}/{name}_task_traces.png"); plt.close(fig)


def twostep(src, figs=f"{FIGURES}/twostep"):
    """A two-step question watched as it is worked out: when the answer arrives, where the write sits, the write by dose."""
    os.makedirs(figs, exist_ok=True)
    d = json.load(open(src)); name = os.path.basename(src).replace("_twostep.json", ""); nice = NICE.get(name, name)
    plot.use()
    AC = d.get("answer_curves") or {}
    if AC:
        fig, axs = plt.subplots(1, 2, figsize=(7.8, 2.9), sharey=True)
        SER = [("main_answer", "the two-step answer", plot.SERIES[0], "-"), ("control_answer", "the one-step answer", plot.SERIES[2], "--"),
               ("main_landmark", "the landmark’s colour, the trap", plot.SERIES[1], "-")]
        for ax, t in zip(axs, ("read", "logit lens")):
            xs = sorted(float(k) for k in AC[t])
            for key, lab, c, ls in SER:
                ax.plot(xs, [AC[t][f"{x}"][key] if f"{x}" in AC[t] else AC[t][str(x)][key] for x in xs], color=c, lw=1.5, ls=ls)
            ax.set_title({"read": "V-Read"}.get(t, t), loc="left", fontsize=10, color=plot.INK2); ax.set_xlabel("depth"); plot.despine(ax)
        axs[0].set_ylabel("top colour is this one"); axs[0].set_ylim(-0.02, 1.04)
        h = [plt.Line2D([], [], color=c, ls=ls, label=lab) for _, lab, c, ls in SER]
        _legend(fig, h, 3, -0.1)
        fig.suptitle(nice, x=0.01, ha="left", fontsize=10, color=plot.INK3)
        fig.tight_layout(); plot.save(fig, f"{figs}/{name}_twostep_answer.png"); plt.close(fig)
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
        for ax in axs:
            ax.set_xlabel("depth"); plot.despine(ax)
        h = [plt.Line2D([], [], color=COL[k], label=LAB[k]) for k in KEYS]
        _legend(fig, h, 4, -0.1)
        fig.tight_layout(); plot.save(fig, f"{figs}/{name}_twostep_regions.png"); plt.close(fig)
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
            _legend(fig, h, 4, -0.1)
            fig.tight_layout(); plot.save(fig, f"{figs}/{name}_twostep_write.png"); plt.close(fig)


# ---------------------------------------------------------------- schematics
def method(out=f"{FIGURES}/method.png"):
    """A schematic of the two directions: the read and the write, both at the answer spot."""
    plot.use()
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 5.2); ax.set_axis_off()
    BLUE, ORANGE, GREY, INK = plot.SERIES[0], plot.SERIES[1], "#c9c8c1", plot.INK2
    for lab, x0, w, col in [("the picture", 0.4, 3.2, "#dfe7ef"), ("the words", 3.9, 2.0, "#efe7df"), ("answer spot", 6.1, 1.0, "#e6e6e0")]:
        ax.add_patch(Rectangle((x0, 0.45), w, 0.7, facecolor=col, edgecolor="none"))
        ax.text(x0 + w / 2, 0.8, lab, ha="center", va="center", fontsize=9, color=INK)
    for i, y in enumerate([1.5, 2.3, 3.1, 3.9]):
        ax.add_patch(Rectangle((6.1, y), 1.0, 0.45, facecolor="#f0efe9", edgecolor="none"))
        ax.text(6.6, y + 0.22, ["layer l", "", "", "last layer"][i], ha="center", va="center", fontsize=8.5, color=INK)
    ax.text(6.6, 4.75, "the first answer token", ha="center", fontsize=9, color=INK)
    ax.add_patch(FancyArrowPatch((7.25, 1.72), (7.25, 3.95), arrowstyle="-|>", mutation_scale=13, color=BLUE, lw=1.6, connectionstyle="arc3,rad=0.35"))
    ax.text(8.15, 3.3, "V-Read", color=BLUE, fontsize=10, ha="left")
    ax.text(8.15, 2.95, "one small map per layer,\nfitted with no labels:\nwhat the answer\nis going to be", color=INK, fontsize=8.2, ha="left", va="top")
    for x in (2.0, 4.9):
        ax.add_patch(FancyArrowPatch((6.2, 1.6), (x, 1.25), arrowstyle="-|>", mutation_scale=12, color=ORANGE, lw=1.5, connectionstyle="arc3,rad=0.28"))
    ax.text(0.4, 3.95, "V-Write", color=ORANGE, fontsize=10)
    ax.text(0.4, 3.6, "the model’s own gradient of the score,\none direction for each group of positions:\nwhat moves the answer, and the handle\nto move it with", color=INK, fontsize=8.2, va="top")
    ax.text(6.6, 4.35, "a score", ha="center", fontsize=9, color=INK)
    ax.plot([6.1, 7.1], [4.32, 4.32], color=GREY, lw=0.8)
    fig.tight_layout()
    plot.save(fig, out)


def flow(out=f"{FIGURES}/flow.png"):
    """The V-Lens flow chart: one prompt, one layer stack, one score, three moves (the read, the write, the check)."""
    plot.use()
    BLUE, ORANGE, AQUA = plot.SERIES[0], plot.SERIES[1], plot.SERIES[2]
    INK, INK2, MUTED = plot.INK, plot.INK2, plot.MUTED
    TXT = dict(zorder=6)
    PIC_X, WORD_X, ANS_X = (2.7, 4.9), (5.0, 6.3), (6.4, 7.3)
    STACK_X0, STACK_X1 = PIC_X[0], ANS_X[1]
    PROMPT_Y, LAYER0_Y, LAYERL_Y, LAYERLAST_Y = (0.15, 0.75), (1.05, 1.55), (1.8, 2.3), (2.6, 3.1)
    ELLIPSIS_Y = [1.675, 2.45]
    SCORE_Y, SCORE_X = (3.4, 4.1), (6.15, 7.55)
    FILL = {"pic": "#dfe7ef", "word": "#efe7df", "ans": "#f6e2b8"}
    cellx = lambda g: {"pic": PIC_X, "word": WORD_X, "ans": ANS_X}[g]
    midx = lambda g: sum(cellx(g)) / 2
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.set_xlim(-0.3, 10.15); ax.set_ylim(-0.05, 4.35); ax.set_axis_off()

    def cell(y0, y1, group):
        x0, x1 = cellx(group)
        ec = INK2 if group == "ans" else "none"
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=FILL[group], edgecolor=ec, lw=1.2 if ec != "none" else 0, zorder=3))

    def arrow(p0, p1, color, rad=0.0, lw=1.6, ms=12, style="-|>", z=5):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                     connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))

    for y0, y1 in (LAYER0_Y, LAYERL_Y, LAYERLAST_Y):
        for g in ("pic", "word", "ans"):
            cell(y0, y1, g)
    for g in ("pic", "word", "ans"):
        ax.plot([midx(g), midx(g)], [PROMPT_Y[1], LAYERLAST_Y[1]], color=plot.GRID, lw=0.9, zorder=1)
    for ey in ELLIPSIS_Y:
        ax.text((STACK_X0 + STACK_X1) / 2, ey, "⋮", ha="center", va="center", fontsize=13, color=MUTED, **TXT)
    labels = {"pic": "the picture", "word": "the words", "ans": "the answer\nspot"}
    for g in ("pic", "word", "ans"):
        x0, x1 = cellx(g)
        ax.add_patch(Rectangle((x0, PROMPT_Y[0]), x1 - x0, PROMPT_Y[1] - PROMPT_Y[0], facecolor=FILL[g], edgecolor=INK2 if g == "ans" else "none", lw=1.2, zorder=3))
        ax.text((x0 + x1) / 2, sum(PROMPT_Y) / 2, labels[g], ha="center", va="center", fontsize=8.6, color=INK, **TXT)
    ax.text((STACK_X0 + STACK_X1) / 2, PROMPT_Y[1] + 0.14, "one prompt, three groups of positions", ha="center", va="bottom", fontsize=7.6, color=MUTED, style="italic", **TXT)
    ax.text(STACK_X0, LAYER0_Y[1] + 0.05, "layer 0", ha="left", va="bottom", fontsize=8, color=INK2, **TXT)
    ax.text(STACK_X0, LAYERL_Y[1] + 0.05, "layer l", ha="left", va="bottom", fontsize=8, color=INK2, **TXT)
    ax.text(STACK_X0, LAYERLAST_Y[1] + 0.05, "layer L, the last layer", ha="left", va="bottom", fontsize=8, color=INK2, **TXT)
    ans_mid = midx("ans")
    arrow((ans_mid, LAYERLAST_Y[1]), (ans_mid, SCORE_Y[0]), INK, lw=1.4, ms=12)
    ax.add_patch(Rectangle((SCORE_X[0], SCORE_Y[0]), SCORE_X[1] - SCORE_X[0], SCORE_Y[1] - SCORE_Y[0], facecolor="#f4f3ee", edgecolor=INK2, lw=1.1, zorder=4))
    ax.text(sum(SCORE_X) / 2, SCORE_Y[1] - 0.10, "a score m", ha="center", va="top", fontsize=9.3, color=INK, **TXT)
    ax.text(sum(SCORE_X) / 2, SCORE_Y[0] + 0.09, "m(x) = m(h_L[a])", ha="center", va="bottom", fontsize=7.4, color=INK2, family="monospace", **TXT)
    read_x = ANS_X[1]
    arrow((read_x, LAYERL_Y[1] + 0.02), (read_x, LAYERLAST_Y[0] - 0.02), BLUE, rad=0.75, lw=1.7, ms=12)
    rx = STACK_X1 + 0.35
    ax.text(rx, 2.78, "V-Read", color=BLUE, fontsize=9, ha="left", va="bottom", **TXT)
    ax.text(rx, 2.58, "A_l h_l[a] + b_l ≈ h_L[a]", ha="left", fontsize=6.9, color=INK, family="monospace", **TXT)
    ax.text(rx, 2.38, "one affine map per layer, fit with\nno labels, reused for any score", ha="left", va="top", fontsize=6.7, color=INK2, **TXT)
    targets = {"pic": (midx("pic") + 0.05, LAYERL_Y[1]), "word": (midx("word"), LAYERL_Y[1]), "ans": (ANS_X[0] + 0.05, (LAYERL_Y[0] + LAYERL_Y[1]) / 2)}
    starts = {"pic": (SCORE_X[0] + 0.06, SCORE_Y[0]), "word": (SCORE_X[0] + 0.30, SCORE_Y[0]), "ans": (SCORE_X[0] + 0.55, SCORE_Y[0])}
    rad_for = {"pic": -0.32, "word": -0.22, "ans": -0.10}
    for g in ("pic", "word", "ans"):
        arrow(starts[g], targets[g], ORANGE, rad=rad_for[g], lw=1.5, ms=11)
    ax.text(0.0, LAYERL_Y[1] + 0.30, "V-Write", color=ORANGE, fontsize=9, ha="left", **TXT)
    ax.text(0.0, LAYERL_Y[1] + 0.06, "u_l^c = mean of d m(x)/d h_l[p]", ha="left", fontsize=6.9, color=INK, family="monospace", **TXT)
    ax.text(0.0, LAYERL_Y[0] + 0.24, "reading of p:  u_l^c . (h_l[p] - mu_l^c)", ha="left", fontsize=6.9, color=INK, family="monospace", **TXT)
    ax.text(0.0, LAYERL_Y[0] + 0.02, "one backward pass gives every layer\nand position at once", ha="left", va="top", fontsize=6.7, color=INK2, **TXT)
    swap_y = 1.9
    arrow((STACK_X1 + 0.9, swap_y), (ANS_X[1] + 0.02, swap_y), AQUA, lw=1.6, ms=12)
    ax.text(STACK_X1 + 1.0, swap_y, "swap in h_l[a]", ha="left", va="center", fontsize=6.9, color=INK, **TXT)
    arrow((SCORE_X[1], SCORE_Y[0] + 0.10), (SCORE_X[1] + 0.85, SCORE_Y[0] + 0.30), AQUA, rad=-0.35, lw=1.6, ms=11)
    ax.text(SCORE_X[1] + 0.95, SCORE_Y[0] + 0.30, "measured Δm", color=AQUA, fontsize=8, ha="left", va="center", **TXT)
    ax.text(rx, 1.55, "the check", color=AQUA, fontsize=9, ha="left", **TXT)
    ax.text(rx, 1.35, "swap another item's state in;\ncompare the true change in m to\nthe read's prediction; held-out R²", ha="left", va="top", fontsize=6.7, color=INK2, **TXT)
    plot.save(fig, out)


# ---------------------------------------------------------------- everything, from the default result paths
def all_figures():
    """Redraw every V-Lens page figure whose results are on disk."""
    R = RESULTS
    g = [p for p in (f"{R}/gate/{mk}_gate.json" for mk, _n in MODELS) if os.path.exists(p)]
    if g:
        gate(g)
    if os.path.exists(f"{R}/gate/reader_regression.json"):
        reader_decision()
    gate_channels()
    if os.path.exists(f"{R}/reader/Qwen3-VL-8B-Instruct_loo.json"):
        loo_refit()
    for fn, sub, suffix, kw in ((presence, "regions_hard", "_regions.json", {}), (steer_dose, "steer", "_steer.json", {}),
                                (cascade, "cascade", "_cascade.json", {}), (prefill, "prefill", "_prefill.json", {}), (regions, "regions_hard", "_regions.json", {"out": f"{FIGURES}/regions.png"})):
        js = [p for p in (f"{R}/{sub}/{mk}{suffix}" for mk, _n in MODELS) if os.path.exists(p)]
        if js:
            fn(js, **kw)
    for mk, _n in MODELS:
        for fn, sub, suffix in ((refusal_steer, "steer", "_refusal_steer.json"), (atlas_summary, "atlas", "_atlas.json"), (tasks, "tasks", "_tasks.json"), (twostep, "twostep", "_twostep.json")):
            p = f"{R}/{sub}/{mk}{suffix}"
            if os.path.exists(p):
                fn(p)
    method(); flow()
