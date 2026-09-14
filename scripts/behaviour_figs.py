# SPDX-License-Identifier: Apache-2.0
"""Behaviour figures for the datasets page: Authority prompt kinds, decoy replies, nuScenes obedience."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from jvlm import plot  # noqa: E402
RES = os.path.join(ROOT, "results", "vlens"); RESL = os.path.join(ROOT, "results", "vlens_long"); FIG = os.path.join(ROOT, "figures", "vlens")
MODELS = [("Qwen3-VL-8B-Instruct", "Qwen3-VL-8B"), ("InternVL3-8B-hf", "InternVL3-8B")]
R = json.load(open(os.path.join(RES, "report.json")))
plot.use()


def auth_rows(mk):
    for base in (RESL, RES):
        p = os.path.join(base, f"{mk}__gemini_auth.json")
        if os.path.exists(p):
            rs = json.load(open(p))["rows"]
            if all(r.get("judge") for r in rs if r.get("kind") == "open"):
                return rs
    return []


# 1. Authority prompt kinds
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

# 2. decoy replies, stacked
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

# 3. nuScenes: which carrier the operative answer followed
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
from matplotlib.patches import Patch
axs[1].legend(handles=[Patch(color=cols["official"], label="official sign (or A)"), Patch(color=cols["informal"], label="cardboard sign (or B)"), Patch(color=cols["tie"], label="both say the same")], frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(-0.1, -0.28), ncol=3)
fig.suptitle("nuScenes conflicts: which sign the operative answer obeyed", x=0.01, ha="left", fontsize=9.5, color=plot.INK2)
fig.tight_layout(); plot.save(fig, os.path.join(FIG, "nusc_obeys.png")); plt.close(fig)
print("wrote authority_kinds, decoy_replies, nusc_obeys")
