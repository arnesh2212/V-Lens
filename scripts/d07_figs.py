# SPDX-License-Identifier: Apache-2.0
"""The figures for the datasets page that the report script does not draw.

  attribution pages   one edited frame, its edit mask, the V-Write maps at four depths over the photo,
                      and the un-edit truth against the prediction
  settle depths       where each item's V-Read score stops changing sign, by condition
  authority ladder    Open Images: share of items judged to carry official authority, by group
  sandbagging         WMDP v2: accuracy by prompt, the correct-letter reading by depth for deliberate
                      and honest wrong answers, and who wrote the sandbag
  steering surfaces   flip rate over depth x dose as a 3-D surface, one per direction
  reading surface     V-Read AUROC over depth x family as a 3-D surface, against the logit lens
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from jvlm import plot  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "vlens")
FIG = os.path.join(ROOT, "figures", "vlens")
DATA = os.environ.get("VLENS_DATA", os.path.join(ROOT, "VLM V-Lens Final Dataset", "VLM V-Lens Final Dataset"))
os.makedirs(FIG, exist_ok=True)
MODELS = [("Qwen3-VL-8B-Instruct", "Qwen3-VL-8B"), ("InternVL3-8B-hf", "InternVL3-8B")]
plot.use()


def J(p):
    return json.load(open(p)) if os.path.exists(p) else None


def local_image(rel):
    """Image path on this machine for a path relative to the dataset root."""
    return os.path.join(DATA, rel.replace("/workspace/vlm/data/", ""))


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
    cs = [c for c in d["cases"] if c["depths"]][:4]
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
    from matplotlib.patches import Rectangle
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


for model, nice in MODELS:
    if not any(f.startswith(model) for f in os.listdir(RES)):
        continue
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
print("figures written to", FIG)
