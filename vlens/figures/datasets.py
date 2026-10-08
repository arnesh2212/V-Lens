# SPDX-License-Identifier: Apache-2.0
"""Every figure of the datasets page (paper/datasets.html), from the result JSONs.

Reading and attribution figures per model (``per_model``), behaviour figures
across models (``authority_kinds``, ``decoy_replies``, ``nusc_obeys``), the
Authority showcase with verbatim replies (``authority_showcase``) and the
example panels of each dataset (``dataset_panels``).  ``all_figures`` draws
them all; ``build/figures.py datasets`` is the command.
"""
from __future__ import annotations

import json
import os
import re
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from PIL import Image, ImageDraw, ImageFont

from vlens.figures import plot
from vlens.paths import ASSETS, DATA, FIGURES, RESULTS

RES = os.path.join(RESULTS, "vlens")
RESL = os.path.join(RESULTS, "vlens_long")
FIG = os.path.join(FIGURES, "vlens")
MODELS = [("Qwen3-VL-8B-Instruct", "Qwen3-VL-8B"), ("InternVL3-8B-hf", "InternVL3-8B")]
os.makedirs(FIG, exist_ok=True)
plot.use()


def _report():
    return json.load(open(os.path.join(RES, "report.json")))


def J(p):
    return json.load(open(p)) if os.path.exists(p) else None


def local_image(rel):
    """Image path on this machine for a path relative to the dataset root."""
    return os.path.join(DATA, rel.replace("data/", ""))


# ---------------------------------------------------------------- attribution pages
def attribution_pages(model, nice):
    d = J(os.path.join(RES, f"{model}__attribution.json"))
    if not d:
        return
    G = d["grid"]
    beh = {r["id"]: r for r in (J(os.path.join(RES, f"{model}__nusc_edit.json")) or {"rows": []})["rows"]}
    beh.update({r["id"]: r for r in (J(os.path.join(RES, f"{model}__gemini_swap.json")) or {"rows": []})["rows"]})
    picks = [c for c in d["cases"] if c["depths"] and c["fam"] == "nusc_operative"][:6] + \
            [c for c in d["cases"] if c["depths"] and c["fam"] == "nusc_eval"][:3] + \
            [c for c in d["cases"] if c["depths"] and c["fam"] == "gemini_swap"][:3]
    for c in picks:
        r = beh.get(c["id"], {})
        img_rel = r.get("image")
        if not img_rel:
            continue
        im = Image.open(local_image(img_rel)).convert("RGB").resize((448, 448))
        keys = sorted(c["depths"], key=int)
        fig, axs = plt.subplots(1, len(keys) + 2, figsize=(2.6 * (len(keys) + 2), 3.1))
        axs[0].imshow(im); axs[0].set_axis_off()
        axs[0].set_title(f"{c['variant']}", loc="left", fontsize=9, color=plot.INK)
        mask = np.array(c["depths"][keys[0]]["mask"]).reshape(G, G)
        axs[1].imshow(im); axs[1].imshow(np.kron(mask, np.ones((448 // G, 448 // G))), cmap="Reds", alpha=0.45, vmin=0, vmax=1)
        axs[1].set_axis_off(); axs[1].set_title("what the edit changed", loc="left", fontsize=9, color=plot.INK2)
        for ax, k in zip(axs[2:], keys):
            dd = c["depths"][k]; M = np.array(dd["map_path_src"]).reshape(G, G); v = float(np.percentile(np.abs(M), 99)) or 1.0
            ax.imshow(im, alpha=0.55); ax.imshow(np.kron(np.clip(M, -v, v), np.ones((448 // G, 448 // G))), cmap="RdBu_r", vmin=-v, vmax=v, alpha=0.6)
            ax.set_axis_off()
            ax.set_title(f"depth {dd['frac']:.2f}\nun-edit truth {dd['unedit_mask']:+.1f}, V-Write {dd['pred_unedit_mask']:+.1f}", loc="left", fontsize=8, color=plot.INK2)
        q = (r.get("question") or "")[:90]; rep = (r.get("reply") or "").replace("\n", " ")[:80]
        fig.text(0.01, 0.02, f"score {c['kind']}: source {c['score_src']:+.1f}, edited {c['score_edit']:+.1f}   |   question: {q}   |   reply: {rep}", fontsize=7.5, color=plot.INK2)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        plot.save(fig, os.path.join(FIG, f"{model}_attr_{c['fam']}_{c['id']}.png")); plt.close(fig)
    print(f"attribution pages: {len(picks)} for {nice}")


# ---------------------------------------------------------------- settle depths
def settle_depths(model, nice):
    fams = [("oi_orig", "Open Images"), ("nusc_edit", "nuScenes edits"), ("vins_orig", "VINS"), ("wmdp", "WMDP")]
    rows = []
    for fam, lab in fams:
        d = J(os.path.join(RES, f"{model}__{fam}_vread.json"))
        if not d:
            continue
        dep = np.array(d["depths"])
        for it in d["items"]:
            v = np.array(it["v_read"]); fin = it["final"]
            # first depth at which the reading is within a tenth of the final value and stays on the final side
            close = (np.abs(v - fin) <= 0.1 * max(abs(fin), 1e-6)) & (np.sign(v) == (np.sign(fin) or 1.0))
            ok = [i for i in range(len(dep)) if close[i:].all()]
            settle = dep[ok[0]] if ok else dep[-1]
            rows.append(dict(fam=lab, kind=it["kind"], cond=it["condition"], settle=float(settle)))
    if not rows:
        return
    labs = sorted(set(r["fam"] for r in rows), key=lambda x: [f[1] for f in fams].index(x))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, lab in enumerate(labs):
        v = [r["settle"] for r in rows if r["fam"] == lab]
        ax.hist(v, bins=np.linspace(0, 1, 21), histtype="step", lw=1.6, color=plot.SERIES[i % len(plot.SERIES)], label=f"{lab} (n={len(v)})", density=True)
    ax.set_xlabel("depth at which the V-Read score comes within a tenth of its final value and stays there"); ax.set_ylabel("share of items")
    plot.despine(ax); ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); plot.save(fig, os.path.join(FIG, f"{model}_settle_depths.png")); plt.close(fig)
    json.dump(rows, open(os.path.join(RES, f"{model}__settle.json"), "w"))
    print("settle depths:", nice, {lab: round(float(np.median([r["settle"] for r in rows if r["fam"] == lab])), 2) for lab in labs})


# ---------------------------------------------------------------- authority ladder
def authority_ladder(model, nice):
    rep = J(os.path.join(RES, "report.json")) or {}
    sec = ((rep.get(model) or {}).get("openimages") or {}).get("authority_judgement_official_by_group")
    if not sec:
        return
    order = ["statutory_sign", "authority_figure", "construction", "road_scene", "hospital", "lab_bio", "commercial_text", "informal_text"]
    groups = [g for g in order if g in sec] + [g for g in sec if g not in order]
    vals = [sec[g][0] if sec[g][0] is not None else 0 for g in groups]; ns = [sec[g][1] for g in groups]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.barh(range(len(groups)), vals, color=[plot.SERIES[0] if g in ("statutory_sign", "authority_figure") else plot.SERIES[1] if g in ("commercial_text", "informal_text") else plot.MUTED for g in groups])
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([f"{g.replace('_', ' ')} (n={n})" for g, n in zip(groups, ns)], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("share of photos where the model says the text carries official authority")
    plot.despine(ax); fig.tight_layout(); plot.save(fig, os.path.join(FIG, f"{model}_authority_ladder.png")); plt.close(fig)
    print("authority ladder:", nice)


# ---------------------------------------------------------------- sandbagging
def sandbagging(model, nice):
    d = J(os.path.join(RES, f"{model}__wmdp2_analysis.json"))
    if not d:
        return
    s = d["summary"]; conds = ["unlocked", "hazard", "explicit", "calibration", "incentive"]
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.0))
    x = np.arange(len(conds))
    axs[0].bar(x - 0.18, [s["accuracy"][c][0] for c in conds], 0.36, color=plot.SERIES[0], label="accuracy")
    axs[0].bar(x + 0.18, [s["probe_yes"][c] for c in conds], 0.36, color=plot.SERIES[1], label="says it was an evaluation")
    axs[0].set_xticks(x); axs[0].set_xticklabels(conds, fontsize=8.5); axs[0].set_ylim(0, 1.02); plot.despine(axs[0]); axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_title(f"{nice}, WMDP-bio, {s['accuracy']['unlocked'][1]} items", loc="left", fontsize=9.5, color=plot.INK2)
    for i, c in enumerate(conds[1:]):
        oc = s["outcomes"].get(c, {})
        n = sum(oc.values()) or 1
        axs[1].bar(i, oc.get("sandbagged (right unlocked, wrong here)", 0) / n, color=plot.SERIES[3], label="sandbagged: right unlocked, wrong here" if i == 0 else None)
    axs[1].set_xticks(range(len(conds) - 1)); axs[1].set_xticklabels(conds[1:], fontsize=8.5); axs[1].set_ylim(0, 1.02); plot.despine(axs[1]); axs[1].legend(frameon=False, fontsize=8)
    axs[1].set_title("share of items sandbagged", loc="left", fontsize=9.5, color=plot.INK2)
    fig.tight_layout(); plot.save(fig, os.path.join(FIG, f"{model}_sandbag_behaviour.png")); plt.close(fig)
    # the reading: correct-letter margin by class, per condition
    vr = d["vread"]; conds_v = [c for c in conds if c in vr]
    if conds_v:
        fig, axs = plt.subplots(1, len(conds_v), figsize=(3.0 * len(conds_v), 3.0), sharey=True, squeeze=False)
        for ax, c in zip(axs[0], conds_v):
            dep = np.array(vr[c]["depths"])
            for cls, col in (("correct", plot.SERIES[2]), ("deliberate wrong", plot.SERIES[1]), ("honest wrong", plot.MUTED)):
                mb = vr[c]["mean_by_class"].get(cls)
                if mb:
                    ax.plot(dep, mb["v_read"], color=col, lw=1.6, label=f"{cls} (n={mb['n']})")
            ax.axhline(0, color=plot.AXIS, lw=0.8); plot.despine(ax); ax.set_title(c, loc="left", fontsize=9.5, color=plot.INK2); ax.set_xlabel("depth")
            ax.legend(frameon=False, fontsize=7.5)
        axs[0][0].set_ylabel("correct letter over the best other, V-Read")
        fig.tight_layout(); plot.save(fig, os.path.join(FIG, f"{model}_sandbag_vread.png")); plt.close(fig)
        # detector: deliberate vs honest wrong
        fig, ax = plt.subplots(figsize=(4.6, 3.0)); any_ = False
        for i, c in enumerate(conds_v):
            au = vr[c]["auroc"].get("deliberate_vs_honest|v_read"); al = vr[c]["auroc"].get("deliberate_vs_honest|v_lens")
            if au:
                ax.plot(vr[c]["depths"], au, color=plot.SERIES[i], lw=1.6, label=f"{c}, V-Read"); any_ = True
                if al:
                    ax.plot(vr[c]["depths"], al, color=plot.SERIES[i], lw=1.0, ls=":", label=f"{c}, logit lens")
        if any_:
            plot.chance_line(ax, 0.5); plot.despine(ax); ax.set_xlabel("depth"); ax.set_ylabel("AUROC, deliberate vs honest wrong"); ax.legend(frameon=False, fontsize=7.5)
            fig.tight_layout(); plot.save(fig, os.path.join(FIG, f"{model}_sandbag_detector.png"))
        plt.close(fig)
    # who wrote the sandbag
    vw = d.get("vwrite_summary") or {}
    if vw:
        conds_w = [c for c in conds if c in vw and vw[c]["n"] >= 5]; keys = sorted(next(iter(vw.values())).keys() - {"n"}, key=int)
        if not conds_w:
            return
        fig, axs = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.0), sharey=True, squeeze=False)
        for ax, k in zip(axs[0], keys):
            x = np.arange(len(conds_w)); bottom = np.zeros(len(conds_w))
            for g, col in (("system", plot.SERIES[1]), ("question", plot.SERIES[0]), ("image", plot.SERIES[2]), ("other", plot.MUTED)):
                v = np.array([vw[c][k][g] for c in conds_w]); ax.bar(x, v, 0.6, bottom=np.where(v > 0, bottom, 0), color=col, label=g); bottom = bottom + np.clip(v, 0, None)
            ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(n={vw[c]['n']})" for c in conds_w], fontsize=8); plot.despine(ax)
            ax.set_title(f"depth {vw[conds_w[0]][k]['score'] and d['depths_write'][keys.index(k)]:.2f}", loc="left", fontsize=9.5, color=plot.INK2)
        axs[0][0].set_ylabel("contribution to chosen-over-correct"); axs[0][0].legend(frameon=False, fontsize=8)
        fig.tight_layout(); plot.save(fig, os.path.join(FIG, f"{model}_sandbag_who_wrote.png")); plt.close(fig)
    print("sandbagging figures:", nice)


# ---------------------------------------------------------------- 3-D surfaces
def steering_surface(model, nice):
    d = J(os.path.join(RES, f"{model}__steer_operative.json"))
    if not d:
        return
    deps = d["depths"]; doses = d["doses"]; s = d["summary"]
    fig = plt.figure(figsize=(10, 3.6))
    for i, dname in enumerate(("V-Write", "DIM (labelled)", "random")):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        Z = np.array([[s.get(f"{dp:.2f}|{dname}|{al}", {}).get("flip", np.nan) for al in doses] for dp in deps])
        X, Y = np.meshgrid(doses, deps)
        ax.plot_surface(X, Y, Z, cmap="Blues", vmin=0, vmax=1, edgecolor=plot.INK2, lw=0.3, alpha=0.95)
        ax.set_xlabel("dose", fontsize=8); ax.set_ylabel("depth", fontsize=8); ax.set_zlabel("flip rate", fontsize=8); ax.set_zlim(0, 1)
        ax.set_title(dname, fontsize=9.5, color=plot.INK2); ax.tick_params(labelsize=7); ax.view_init(elev=25, azim=-135)
    fig.suptitle(f"{nice}: the operative answer flips when a direction is added at the answer spot", fontsize=9.5, color=plot.INK2, x=0.02, ha="left")
    fig.tight_layout(); plot.save(fig, os.path.join(FIG, f"{model}_steer_surface.png")); plt.close(fig)
    print("steering surface:", nice)


def reading_surface(model, nice):
    fams = [("oi_orig", "Open Images"), ("vins_orig", "VINS"), ("nusc_edit", "nuScenes edits"), ("gemini_auth", "Authority"), ("wmdp", "WMDP")]
    rows_r, rows_l, labs = [], [], []
    for fam, lab in fams:
        d = J(os.path.join(RES, f"{model}__{fam}_vread.json"))
        if not d:
            continue
        k = next((k for k in d["auroc"] if k.endswith("|y_final|v_read")), None)
        if not k:
            continue
        rows_r.append(d["auroc"][k]); rows_l.append(d["auroc"][k.replace("v_read", "v_lens")]); labs.append(lab); dep = d["depths"]
    if not rows_r:
        return
    fig = plt.figure(figsize=(9.5, 3.8))
    for i, (Z, title) in enumerate(((np.array(rows_r), "V-Read"), (np.array(rows_l), "logit lens"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        X, Y = np.meshgrid(dep, np.arange(len(labs)))
        ax.plot_surface(X, Y, Z, cmap="viridis", vmin=0.4, vmax=1.0, edgecolor=plot.INK2, lw=0.2, alpha=0.95)
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=7); ax.set_xlabel("depth", fontsize=8); ax.set_zlabel("AUROC", fontsize=8); ax.set_zlim(0.4, 1.0)
        ax.set_title(title, fontsize=9.5, color=plot.INK2); ax.tick_params(labelsize=7); ax.view_init(elev=28, azim=-120)
    fig.suptitle(f"{nice}: reading the model's upcoming answer, by dataset and depth", fontsize=9.5, color=plot.INK2, x=0.02, ha="left")
    fig.tight_layout(); plot.save(fig, os.path.join(FIG, f"{model}_reading_surface.png")); plt.close(fig)
    print("reading surface:", nice)


def attribution_surface(model, nice):
    """One edited frame's V-Write map as a surface over the patch grid, at two depths."""
    d = J(os.path.join(RES, f"{model}__attribution.json"))
    if not d:
        return
    G = d["grid"]
    c = next((c for c in d["cases"] if c["depths"] and c["variant"] == "conflict_swapped"), None) or next((c for c in d["cases"] if c["depths"]), None)
    if not c:
        return
    keys = sorted(c["depths"], key=int)[:2]
    fig = plt.figure(figsize=(9, 3.8))
    for i, k in enumerate(keys):
        dd = c["depths"][k]; M = np.array(dd["map_path_src"]).reshape(G, G)
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        X, Y = np.meshgrid(np.arange(G), np.arange(G))
        ax.plot_surface(X, -Y, M, cmap="RdBu_r", vmin=-np.abs(M).max(), vmax=np.abs(M).max(), edgecolor=plot.INK2, lw=0.2)
        ax.set_title(f"{c['variant']}, depth {dd['frac']:.2f}", fontsize=9.5, color=plot.INK2); ax.tick_params(labelsize=7)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zlabel("V-Write", fontsize=8); ax.view_init(elev=35, azim=-60)
    fig.suptitle(f"{nice}: the V-Write map of one edited frame as a surface; the ridge is the added sign", fontsize=9.5, color=plot.INK2, x=0.02, ha="left")
    fig.tight_layout(); plot.save(fig, os.path.join(FIG, f"{model}_attr_surface.png")); plt.close(fig)
    print("attribution surface:", nice)


# ---------------------------------------------------------------- authority: the answer arriving
def authority_arrival(model, nice):
    d = J(os.path.join(RES, f"{model}__authority_arrival.json"))
    if not d:
        return
    dep = np.array(d["depths"]); items = d["items"]
    # mean trace of the reference word and of the model's own word, by standing tier
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.1), sharey=True)
    for ax, key, title in ((axs[0], "v_read_ref", "the reference answer's first word"), (axs[1], "v_read_own", "the model's own first word")):
        for i, tier in enumerate(("D1", "D2", "D0", "D3")):
            its = [np.array(it[key]) for it in items if it["standing"] == tier and np.isfinite(it[key]).all()]
            if len(its) >= 3:
                ax.plot(dep, np.mean(its, 0), color=plot.SERIES[i], lw=1.6, label=f"{tier} (n={len(its)})")
        plot.despine(ax); ax.set_title(title, loc="left", fontsize=9.5, color=plot.INK2); ax.set_xlabel("depth"); ax.legend(frameon=False, fontsize=8)
    axs[0].set_ylabel("log-probability, V-Read")
    fig.tight_layout(); plot.save(fig, os.path.join(FIG, f"{model}_authority_arrival.png")); plt.close(fig)
    # word strips for three D2 items where the reference and the model disagree, and three where they agree
    picks = [it for it in items if it["standing"] == "D2" and it["judge"] == "incorrect"][:3] + [it for it in items if it["standing"] == "D2" and it["judge"] == "correct"][:3]
    if picks:
        show = sorted(d["show"], key=int)
        fig, axs = plt.subplots(len(picks), 1, figsize=(9.0, 1.15 * len(picks) + 0.6), squeeze=False)
        for ax, it in zip(axs[:, 0], picks):
            ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.text(0.0, 0.85, f"{it['question'][:70]}   reference: {it['expected']}   |   reply: {it['reply_short'].strip()[:24]}   [{it['judge']}]", fontsize=8, color=plot.INK)
            for j, l in enumerate(show):
                x = 0.08 + 0.92 * j / max(len(show) - 1, 1)
                ax.text(x, 0.55, f"{float(d['show'][l]):.2f}", fontsize=7, color=plot.INK3, ha="center")
                ax.text(x, 0.32, it["strips"]["V-Read"][l][0][:10] or "·", fontsize=8.5, color=plot.SERIES[0], ha="center")
                ax.text(x, 0.08, it["strips"]["logit lens"][l][0][:10] or "·", fontsize=8.5, color=plot.SERIES[1], ha="center")
            ax.text(0.0, 0.32, "V-Read", fontsize=7.5, color=plot.SERIES[0]); ax.text(0.0, 0.08, "logit lens", fontsize=7.5, color=plot.SERIES[1])
        fig.tight_layout(); plot.save(fig, os.path.join(FIG, f"{model}_authority_strips.png")); plt.close(fig)
    print("authority arrival:", nice)


def authority_removal_pages(model, nice):
    d = J(os.path.join(RES, f"{model}__authority_removal.json"))
    if not d:
        return
    rm = {r["id"]: r for r in (J(os.path.join(RES, f"{model}__gemini_removal.json")) or {"rows": []})["rows"]}
    cs = sorted([c for c in d["cases"] if c["depths"]], key=lambda c: -abs(c["score_removed"] - c["score_original"]))[:3]
    for c in cs:
        r = rm.get(c["id"], {}); img_rel = r.get("image")
        if not img_rel:
            continue
        im = Image.open(local_image(img_rel)).convert("RGB").resize((448, 448))
        keys = sorted(c["depths"], key=int); G = int(round(len(c["depths"][keys[0]]["mask"]) ** 0.5))
        fig, axs = plt.subplots(1, len(keys) + 2, figsize=(2.6 * (len(keys) + 2), 3.1))
        axs[0].imshow(im); axs[0].set_axis_off(); axs[0].set_title("bearer text erased", loc="left", fontsize=9, color=plot.INK)
        mask = np.array(c["depths"][keys[0]]["mask"]).reshape(G, G)
        axs[1].imshow(im); axs[1].imshow(np.kron(mask, np.ones((448 // G, 448 // G))), cmap="Reds", alpha=0.45, vmin=0, vmax=1); axs[1].set_axis_off()
        axs[1].set_title("what was erased", loc="left", fontsize=9, color=plot.INK2)
        for ax, k in zip(axs[2:], keys):
            dd = c["depths"][k]; M = np.array(dd["map"]).reshape(G, G); v = float(np.percentile(np.abs(M), 99)) or 1.0
            ax.imshow(im, alpha=0.55); ax.imshow(np.kron(np.clip(M, -v, v), np.ones((448 // G, 448 // G))), cmap="RdBu_r", vmin=-v, vmax=v, alpha=0.6); ax.set_axis_off()
            ax.set_title(f"depth {dd['frac']:.2f}\nrestore truth {dd['restore_truth']:+.1f}, V-Write {dd['restore_pred']:+.1f}", loc="left", fontsize=8, color=plot.INK2)
        fig.text(0.01, 0.02, f"“{c['question'][:80]}”  original reply: {c['reply_original'].strip()[:24]}  |  after erasure: {c['reply_removed'].strip()[:40]}", fontsize=7.5, color=plot.INK2)
        fig.tight_layout(rect=(0, 0.06, 1, 1)); plot.save(fig, os.path.join(FIG, f"{model}_removal_{c['id']}.png")); plt.close(fig)
    print("removal pages:", nice, len(cs))


# ---------------------------------------------------------------- depicted text: reading and writing
def depicted_reading(model, nice):
    """Mean V-Read trace of yes-over-no for the depicted-sign questions (p5) against the other yes/no Authority prompts."""
    d = J(os.path.join(RES, f"{model}__gemini_auth_vread.json"))
    if not d:
        return
    dep = np.array(d["depths"]); items = d["items"]
    groups = [("p5", True, "depicted sign, model says yes"), ("p5", False, "depicted sign, model says no"),
              ("p1", True, "scene question, model says yes"), ("p1", False, "scene question, model says no")]
    fig, ax = plt.subplots(figsize=(6.8, 3.4)); out = {}
    for i, (cond, yes, lab) in enumerate(groups):
        g = [it for it in items if it["condition"] == cond and (it["final"] > 0) == yes]
        if not g:
            continue
        V = np.array([it["v_read"] for it in g]); mu = V.mean(0)
        sd = []
        for v in V:
            sgn = np.sign(v[-1]) or 1.0; ok = np.sign(v) == sgn; k = len(v) - 1
            while k > 0 and ok[k - 1]:
                k -= 1
            sd.append(float(dep[k]))
        out[lab] = dict(n=len(g), settle_median=float(np.median(sd)), trace=mu.round(3).tolist(), depths=dep.round(3).tolist(),
                        at=[float(mu[np.argmin(np.abs(dep - x))]) for x in (0.05, 0.25, 0.5, 1.0)])
        ax.plot(dep, mu, color=plot.SERIES[i % len(plot.SERIES)], lw=1.8, ls="-" if yes else "--", label=f"{lab} (n={len(g)})")
    ax.axhline(0, color=plot.AXIS, lw=0.8); ax.set_xlabel("depth"); ax.set_ylabel("V-Read, yes over no (log-odds)")
    ax.set_title(f"{nice}, Authority scenes, yes/no prompts", loc="left", fontsize=9.5, color=plot.INK2)
    plot.despine(ax); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); plot.save(fig, os.path.join(FIG, f"{model}_depicted_reading.png")); plt.close(fig)
    json.dump(out, open(os.path.join(RES, f"{model}__depicted_reading.json"), "w"))
    print("depicted reading:", nice, {k: (v["n"], v["settle_median"]) for k, v in out.items()})


def depicted_pages(model, nice):
    """One panel per depicted-sign item: the scene with the V-Write map at the shallow depth and its eight strongest patches outlined."""
    d = J(os.path.join(RES, f"{model}__depicted.json"))
    if not d:
        return
    picks = [it for it in d["items"] if it["condition"] == "p5"]
    if not picks:
        return
    G = picks[0]["grid"]; k0 = sorted(picks[0]["depths"], key=int)[0]; cell = 448 // G
    n = len(picks); cols = 4; rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows))
    for ax in axs.ravel():
        ax.set_axis_off()
    for ax, it in zip(axs.ravel(), picks):
        im = Image.open(local_image(it["image"])).convert("RGB").resize((448, 448))
        dd = it["depths"][k0]; M = np.array(dd["map_path"]).reshape(G, G); v = float(np.percentile(np.abs(M), 99)) or 1.0
        ax.imshow(im); ax.imshow(np.kron(np.clip(M, -v, v), np.ones((cell, cell))), cmap="RdBu_r", vmin=-v, vmax=v, alpha=0.45)
        order = np.argsort(-np.abs(M.ravel()))[:8]
        for q in order:
            r, c = divmod(int(q), G)
            ax.add_patch(Rectangle((c * cell, r * cell), cell, cell, fill=False, ec=plot.INK, lw=1.2))
        t = dd["truth"]["8"]
        rep = (it.get("reply") or "").replace("\n", " ")
        rep = rep[:48] + ("..." if len(rep) > 48 else "")
        ax.set_title(f"{it['id']}: yes over no {it['score']:+.1f}; replace the 8 outlined {t['top']:+.1f}, 8 random {t['random']:+.1f}\n{rep}", loc="left", fontsize=7, color=plot.INK2)
    fig.suptitle(f"{nice}, depicted-sign items, V-Write of yes over no at depth {dd['frac']:.2f}; outlined: the map's eight strongest patches", x=0.01, ha="left", fontsize=9.5, color=plot.INK2)
    fig.tight_layout(); plot.save(fig, os.path.join(FIG, f"{model}_depicted_pages.png")); plt.close(fig)
    print(f"depicted pages: {n} items for {nice}")


# ---------------------------------------------------------------- map transfer
def transfer_curves(model, nice):
    d = J(os.path.join(RES, f"{model}__transfer.json"))
    if not d:
        return
    dep = np.array(d["depths"]); fig, ax = plt.subplots(figsize=(6.4, 3.2))
    if d.get("own"):
        own = np.array(d["own"]); ax.plot(dep[: len(own)], own[: len(dep)], color=plot.SERIES[0], lw=1.8, label="map fitted on the nuScenes prompts (own family)")
    ax.plot(dep, d["foreign"], color=plot.SERIES[1], lw=1.8, ls="--", label="map fitted on Open Images prompts, applied here")
    ax.plot(dep, d["logit"], color=plot.SERIES[2], lw=1.4, ls=":", label="logit lens")
    ax.axhline(0.5, color=plot.AXIS, lw=0.8); ax.set_ylim(0.3, 1.02); ax.set_xlabel("depth"); ax.set_ylabel("AUROC vs the model's answer")
    ax.set_title(f"{nice}, nuScenes edits, yes/no ({d['n']} items)", loc="left", fontsize=9.5, color=plot.INK2)
    plot.despine(ax); ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); plot.save(fig, os.path.join(FIG, f"{model}_transfer.png")); plt.close(fig)
    print("transfer:", nice)


# ---------------------------------------------------------------- behaviour figures, across models


def auth_rows(mk):
    for base in (RESL, RES):
        p = os.path.join(base, f"{mk}__gemini_auth.json")
        if os.path.exists(p):
            rs = json.load(open(p))["rows"]
            if all(r.get("judge") for r in rs if r.get("kind") == "open"):
                return rs
    return []


def authority_kinds():
    """Authority prompt kinds."""
    R = _report()
    KINDS = [("p1", "scene question"), ("p7", "is it safe to follow?"), ("p5", "is the depicted sign here?"), ("p6", "which value, why trust it?"),
             ("p8", "which bearer settles it?"), ("p3", "go by the apparel or card"), ("p4ref", "read the garments")]
    fig, ax = plt.subplots(figsize=(9.0, 3.4)); w = 0.38
    for i, (mk, nice) in enumerate(MODELS):
        rs = auth_rows(mk)
        if not rs:
            continue
        vals, ns = [], []
        for k, _lab in KINDS:
            rr = [r for r in rs if r["condition"] == k]; ok = sum(1 for r in rr if (r.get("judge") or {}).get("verdict") == "correct")
            vals.append(ok / max(len(rr), 1)); ns.append(len(rr))
        x = np.arange(len(KINDS)) + (i - 0.5) * w
        ax.bar(x, vals, w, color=plot.SERIES[i], label=nice)
        for xx, v, n in zip(x, vals, ns):
            ax.text(xx, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=plot.INK2)
    ax.set_xticks(range(len(KINDS))); ax.set_xticklabels([f"{lab}\n({k})" for k, lab in KINDS], fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy"); ax.set_title("Authority scenes by prompt kind", loc="left", fontsize=9.5, color=plot.INK2)
    plot.despine(ax); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); plot.save(fig, os.path.join(FIG, "authority_kinds.png")); plt.close(fig)


def decoy_replies():
    """decoy replies, stacked."""
    R = _report()
    fig, ax = plt.subplots(figsize=(6.4, 3.0)); labs = []; y = 0
    parts = [("says_not_present", "says it is not there"), ("names_other_text", "reads the sign's real text instead"), ("echoes_decoy_as_written", "names an object it is written on")]
    for mk, nice in MODELS:
        for sec, dn in (("openimages", "Open Images"), ("vins", "VINS")):
            dk = (R.get(mk, {}).get(sec) or {}).get("decoy_kinds")
            if not dk:
                continue
            left = 0
            for j, (k, lab) in enumerate(parts):
                v = dk[k][0]; ax.barh(y, v, left=left, color=plot.SERIES[j], label=lab if y == 0 else None); left += v
            labs.append(f"{nice}, {dn} (n={dk[parts[0][0]][1]})"); y += 1
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8); ax.set_xlim(0, 1); ax.set_xlabel("share of decoy prompts"); ax.invert_yaxis()
    ax.set_title("\"Read the text 'EMERALD' in this image\", when no such text exists", loc="left", fontsize=9.5, color=plot.INK2)
    plot.despine(ax); ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3); fig.tight_layout(); plot.save(fig, os.path.join(FIG, "decoy_replies.png")); plt.close(fig)


def nusc_obeys():
    """nuScenes: which carrier the operative answer followed."""
    R = _report()
    VARS = [("conflict_main", "official STOP, cardboard GO"), ("conflict_swapped", "cardboard STOP, official GO"), ("both_official", "two official signs (A original, B added)"), ("agree_control", "second sign agrees")]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    for ax, (mk, nice) in zip(axs, MODELS):
        n = R.get(mk, {}).get("nuscenes") or {}
        ob = n.get("operative_obeys_by_variant") or {}; side = n.get("operative_side_by_variant") or {}
        cats = ["official", "informal", "tie"]; cols = {"official": plot.SERIES[0], "informal": plot.SERIES[1], "tie": plot.SERIES[3]}
        for i, (v, lab) in enumerate(VARS):
            d = ob.get(v) or {}
            if v == "both_official":
                d = {"official": side.get(v, {}).get("sign A (original)", 0), "informal": side.get(v, {}).get("sign B (added)", 0)}
            tot = sum(d.values()) or 1; left = 0
            for c in cats:
                val = d.get(c, 0) / tot
                if val:
                    ax.barh(i, val, left=left, color=cols[c]); left += val
        ax.set_yticks(range(len(VARS))); ax.set_yticklabels([lab for _v, lab in VARS], fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1)
        ax.set_title(nice, loc="left", fontsize=9.5, color=plot.INK2); plot.despine(ax); ax.set_xlabel("share of frames")
    axs[1].legend(handles=[Patch(color=cols["official"], label="official sign (or A)"), Patch(color=cols["informal"], label="cardboard sign (or B)"), Patch(color=cols["tie"], label="both say the same")], frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(-0.1, -0.28), ncol=3)
    fig.suptitle("nuScenes conflicts: which sign the operative answer obeyed", x=0.01, ha="left", fontsize=9.5, color=plot.INK2)
    fig.tight_layout(); plot.save(fig, os.path.join(FIG, "nusc_obeys.png")); plt.close(fig)
    print("wrote authority_kinds, decoy_replies, nusc_obeys")


# ---------------------------------------------------------------- the Authority showcase, verbatim replies
def authority_showcase():
    """Four Authority scenes with the question, the reference and both models' verbatim replies."""
    FONT = os.path.join(ASSETS, "texgyrepagella-regular.otf")
    ITAL = os.path.join(ASSETS, "texgyrepagella-italic.otf")
    F = ImageFont.truetype(FONT, 26); FI = ImageFont.truetype(ITAL, 26); FS = ImageFont.truetype(FONT, 22)
    PICKS = [("i2p09", "station: the board says platform 2, the dated staff notice says 9, the hoodie says 5"),
             ("i9p08", "lab: the hazard label says store apart, the technician's coat says mixing is fine"),
             ("i8p09", "shop: the ingredients panel lists almonds, the handwritten card says nut free"),
             ("i1p10", "junction: the only STOP is inside an advertising billboard")]
    clean = lambda s: re.sub(r"\s+", " ", re.sub(r"[*#>_]+", "", (s or "").replace(" \u2014 ", ": ").replace("\u2014", ","))).strip()

    def rows_for(mk):
        for base in (RESL, RES):
            p = os.path.join(base, f"{mk}__gemini_auth.json")
            if os.path.exists(p):
                return {r["id"]: r for r in json.load(open(p))["rows"]}
        return {}

    R = {mk: rows_for(mk) for mk, _ in MODELS}
    IMW, TXW, PAD = 880, 1000, 28
    panels = []
    for pid, lab in PICKS:
        r = R[MODELS[0][0]].get(pid)
        if r is None:
            continue
        im = Image.open(os.path.join(DATA, r["image"])).convert("RGB"); im = im.resize((IMW, round(im.height * IMW / im.width)), Image.LANCZOS)
        lines = []
        for txt, font in ((lab, FI), ("", F), ("Question: " + clean(r["question"]), F), ("Reference: " + clean(str(r["expected"])), F), ("", F)):
            lines += [(l, font) for l in (textwrap.wrap(txt, 70) or [""])]
        for mk, nice in MODELS:
            rr = R[mk].get(pid) or {}
            j = (rr.get("judge") or {}).get("verdict") or ""
            lines += [(l, F) for l in textwrap.wrap(f"{nice} ({j}): " + clean(rr.get("reply"))[:520], 70)] + [("", FS)]
        th = sum(34 if f is not FS else 16 for _l, f in lines) + 2 * PAD
        H = max(im.height, th)
        canvas = Image.new("RGB", (IMW + TXW + 3 * PAD, H + PAD), "white")
        canvas.paste(im, (PAD, (H - im.height) // 2 + PAD // 2))
        d = ImageDraw.Draw(canvas); y = PAD
        for l, f in lines:
            d.text((IMW + 2 * PAD, y), l, fill="black", font=f); y += 34 if f is not FS else 16
        panels.append(canvas)
    W = max(p.width for p in panels); Hs = sum(p.height for p in panels)
    sheet = Image.new("RGB", (W, Hs), "white"); y = 0
    for p in panels:
        sheet.paste(p, (0, y)); y += p.height
    sheet.save(os.path.join(FIG, "showcase_authority.png")); print("wrote", os.path.join(FIG, "showcase_authority.png"), sheet.size)


# ---------------------------------------------------------------- example panels of each dataset
def dataset_panels():
    """What an item of each dataset looks like: one tiled panel per dataset."""
    try:
        FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 20)
    except Exception:  # noqa: BLE001
        FONT = ImageFont.load_default()

    def tile(paths, labels, out, w=640, h=360):
        ims = []
        for p, lab in zip(paths, labels):
            im = Image.open(p).convert("RGB")
            im.thumbnail((w, h)); canvas = Image.new("RGB", (w, h + 40), "white"); canvas.paste(im, ((w - im.width) // 2, 40))
            ImageDraw.Draw(canvas).text((10, 8), lab, fill="black", font=FONT); ims.append(canvas)
        n = len(ims); cols = min(n, 3); rows = (n + cols - 1) // cols
        W = Image.new("RGB", (cols * w, rows * (h + 40)), "white")
        for i, im in enumerate(ims):
            W.paste(im, ((i % cols) * w, (i // cols) * (h + 40)))
        W.save(out); print("wrote", out)

    G = os.path.join(DATA, "Gemini Dataset"); N = os.path.join(DATA, "Nuscene Dataset"); O = os.path.join(DATA, "OpenImages Dataset")
    tile([f"{G}/Original Images/i1.png", f"{G}/Edited Images/i1_swap.png", f"{G}/Edited Images/i1_removed.png"],
         ["i1: eight text bearers, one with standing", "i1_swap: police board and permanent sign exchange text", "i1_removed: the board's text erased"],
         f"{FIG}/panel_gemini_authority.png")
    tile([f"{G}/Original Images/s1_V0.png", f"{G}/Edited Images/s1_V1.png", f"{G}/Edited Images/s1_V3.png"],
         ["s1 V0: candid photo", "s1 V1: figure chrome and EVAL SET watermark", "s1 V3: multiple-choice item margin"],
         f"{FIG}/panel_gemini_eval.png")
    e = {x["edit_number"]: x for x in json.load(open(f"{N}/Edit Instructions.json"))["edits"]}
    src = f"{N}/{e[1]['source_image']}"
    tile([src, f"{N}/Edited Images/1.jpg", f"{N}/Edited Images/2.jpg", f"{N}/Edited Images/3.jpg", f"{N}/Edited Images/4.jpg", f"{N}/Edited Images/21.jpg"],
         ["nuScenes frame, untouched", "conflict_main: official STOP vs cardboard GO AHEAD", "conflict_swapped: carriers exchanged",
          "both_official: two official signs disagree", "agree_control: second sign agrees", "downgrade_informal: same words on cardboard"],
         f"{FIG}/panel_nuscenes_edits.png")
    tile([f"{N}/Edited Images/31.jpg", f"{N}/Edited Images/33.jpg", f"{N}/Edited Images/35.jpg", f"{N}/Edited Images/32.jpg", f"{N}/Edited Images/34.jpg", f"{N}/Edited Images/36.jpg"],
         ["eval_watermark", "eval_bbox", "eval: annotation UI / grid", "deploy_dashcam_hud", "deploy_phone_screenshot", "deploy_windshield"],
         f"{FIG}/panel_nuscenes_eval.png")
    idx = {im["image_number"]: im for im in json.load(open(f"{O}/Image Index.json"))["images"]}
    picks = [2] + [k for k, v in idx.items() if v.get("sign") and v["sign"].get("kind") == "novelty"][:2] + [k for k, v in idx.items() if v.get("group") == "authority_figure"][:1] + [k for k, v in idx.items() if v.get("group") == "informal_text"][:1] + [k for k, v in idx.items() if v.get("group") == "lab_bio"][:1]
    labs = []
    for k in picks:
        v = idx[k]; sg = v.get("sign") or {}
        labs.append(f"{v['group']}: {sg.get('text', '')}{(' + ' + sg['added_text']) if sg.get('added_text') else ''}"[:60])
    tile([f"{O}/{idx[k]['file']}" for k in picks], labs, f"{FIG}/panel_openimages.png")


def per_model(model, nice):
    """Every per-model figure whose results exist."""
    attribution_pages(model, nice)
    settle_depths(model, nice)
    authority_ladder(model, nice)
    sandbagging(model, nice)
    steering_surface(model, nice)
    reading_surface(model, nice)
    attribution_surface(model, nice)
    authority_arrival(model, nice)
    authority_removal_pages(model, nice)
    depicted_reading(model, nice)
    depicted_pages(model, nice)
    transfer_curves(model, nice)


def all_figures():
    for model, nice in MODELS:
        if any(f.startswith(model) for f in os.listdir(RES)):
            per_model(model, nice)
    authority_kinds(); decoy_replies(); nusc_obeys()
    authority_showcase(); dataset_panels()
    print("figures written to", FIG)
