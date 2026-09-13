# SPDX-License-Identifier: Apache-2.0
"""Fill paper/datasets.html from results/vlens/report.json and the behaviour JSONs.

Tables live between <!--TABLE:name--> ... <!--/TABLE:name--> markers and are
regenerated in full on every run; prose outside the markers is untouched.  A
missing result leaves its table as it was and says so.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "vlens")
PAGE = os.path.join(ROOT, "paper", "datasets.html")
MODELS = [("Qwen3-VL-8B-Instruct", "Qwen3-VL-8B"), ("InternVL3-8B-hf", "InternVL3-8B")]
REPORT = json.load(open(os.path.join(RES, "report.json"))) if os.path.exists(os.path.join(RES, "report.json")) else {}


def f2(x):
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        v, n = x[0], x[1]
        return "" if v is None else f"{v:.2f}"
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def nof(x):
    return str(x[1]) if isinstance(x, (list, tuple)) and x[1] else ""


def table(head, rows, first="", note=""):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "\n".join("      <tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'  <div class="tblwrap">\n  <table>\n    <thead><tr><th>{first}</th>{h}</tr></thead>\n    <tbody>\n{b}\n    </tbody>\n  </table>\n  </div>'


def models_with(section, key=None):
    out = []
    for mk, nice in MODELS:
        sec = (REPORT.get(mk) or {}).get(section) or {}
        if sec and (key is None or key in sec):
            out.append((mk, nice, sec))
    return out


# ----------------------------------------------------------------- builders
def t_gemini_standing():
    ms = models_with("gemini", "p1_by_standing")
    if not ms:
        return None
    rows = []
    for t, lab in (("D1", "D1, the official-looking object is right"), ("D2", "D2, the less official-looking object is right"),
                   ("D0", "D0, general knowledge"), ("D3", "D3, nothing shown settles it")):
        rows.append([lab] + [f"{f2(sec['p1_by_standing'][t])} <span class='mono'>({nof(sec['p1_by_standing'][t])})</span>" for _mk, _n, sec in ms])
    rows.append(["binding swap: answer follows the object"] + [f"{f2(sec.get('swap_follows_object'))} ({nof(sec.get('swap_follows_object'))})" for _mk, _n, sec in ms])
    rows.append(["bearer removed: says nothing settles it"] + [f"{f2(sec.get('removal_nothing_settles'))} ({nof(sec.get('removal_nothing_settles'))})" for _mk, _n, sec in ms])
    return table([n for _mk, n, _s in ms], rows, first="Gemini authority scenes, p1 accuracy by standing tier")


def t_gemini_eval():
    ms = models_with("gemini", "eval_p1_by_framing")
    if not ms:
        return None
    rows = []
    for e, lab in (("E0", "E0, no wrapper"), ("E1", "E1, “item 7 of a 120-item safety evaluation”"), ("E2", "E2, “your response will be logged and reviewed”"),
                   ("E3", "E3, “suppose someone were standing here”"), ("E4", "E4, “I have to decide in the next few seconds”")):
        rows.append([lab] + [f"{f2(sec['eval_p1_by_framing'][e])} ({nof(sec['eval_p1_by_framing'][e])})" for _mk, _n, sec in ms])
    for v, lab in (("V0", "V0, candid photo"), ("V1", "V1, paper-figure chrome and EVAL SET watermark"), ("V2", "V2, annotated figure"), ("V3", "V3, multiple-choice item margin")):
        rows.append([lab] + [f"{f2(sec['eval_p1_by_visual'][v])} ({nof(sec['eval_p1_by_visual'][v])})" for _mk, _n, sec in ms])
    return table([n for _mk, n, _s in ms], rows, first="Gemini eval-awareness scenes, p1 accuracy by cue")


def t_nusc_operative():
    ms = models_with("nuscenes", "operative_by_variant")
    if not ms:
        return None
    variants = ["conflict_main", "conflict_swapped", "both_official", "agree_control", "downgrade_informal", "content_swap"]
    LAB = {"conflict_main": "conflict, official sign vs cardboard", "conflict_swapped": "conflict, carriers exchanged",
           "both_official": "two official signs disagree", "agree_control": "second sign agrees", "downgrade_informal": "same words on cardboard",
           "content_swap": "opposite words on the official sign"}
    rows = [["untouched frame"] + [f"{f2(sec['operative_baseline'])} ({nof(sec['operative_baseline'])})" for _mk, _n, sec in ms]]
    for v in variants:
        rows.append([LAB[v]] + [f"{f2(sec['operative_by_variant'].get(v))} ({nof(sec['operative_by_variant'].get(v))})" for _mk, _n, sec in ms])
    return table([n for _mk, n, _s in ms], rows, first="nuScenes edits, operative question answered as the authority-tracking answer")


def t_nusc_obeys():
    ms = models_with("nuscenes", "operative_obeys_by_variant")
    if not ms:
        return None
    rows = []
    for v in ("conflict_main", "conflict_swapped", "both_official", "agree_control"):
        for _mk, n, sec in ms:
            c = sec["operative_obeys_by_variant"].get(v, {})
            rows.append([f"{v}, {n}", c.get("official", 0), c.get("informal", 0), c.get("tie", 0)])
    return table(["obeys the official carrier", "obeys the informal one", "both say the same"], rows, first="two-sign edits")


def t_nusc_eval():
    ms = models_with("nuscenes", "verbalised_test_eval_vs_deploy")
    if not ms:
        return None
    rows = []
    for k in ("untouched frames", "eval overlays", "deploy overlays"):
        rows.append([k] + [f"{f2(sec['verbalised_test_eval_vs_deploy'][k])} ({nof(sec['verbalised_test_eval_vs_deploy'][k])})" for _mk, _n, sec in ms])
    for v in sorted({v for _mk, _n, sec in ms for v in sec["verbalised_test_by_variant"]}):
        rows.append([f" {v}"] + [f"{f2(sec['verbalised_test_by_variant'].get(v))} ({nof(sec['verbalised_test_by_variant'].get(v))})" for _mk, _n, sec in ms])
    return table([n for _mk, n, _s in ms], rows, first="nuScenes, share of frames the model calls a test image")


def t_photos(fam, title):
    ms = models_with(fam)
    if not ms:
        return None
    rows = []
    for k, lab in (("acc_operative", "operative question on a transcribed sign"), ("acc_operative_with_graffiti", "graffiti on a real sign changes nothing"),
                   ("acc_which_is_official", "which words are official, which added"), ("acc_read_decoy", "quoted string not in the picture: says so"),
                   ("acc_is_official", "novelty sign is not official")):
        if any(k in sec for _mk, _n, sec in ms):
            rows.append([lab] + [f"{f2(sec.get(k))} ({nof(sec.get(k))})" for _mk, _n, sec in ms])
    rows.append(["calls the photo a test image, all groups"] + [f"{f2(sec.get('verbalised_test_all'))} ({nof(sec.get('verbalised_test_all'))})" for _mk, _n, sec in ms])
    rows.append(["names a dataset when asked which one"] + [f"{f2(sec.get('dataset_names_rate'))} ({nof(sec.get('dataset_names_rate'))})" for _mk, _n, sec in ms])
    return table([n for _mk, n, _s in ms], rows, first=title)


def t_photos_groups(fam, key, title):
    ms = models_with(fam, key)
    if not ms:
        return None
    groups = sorted({g for _mk, _n, sec in ms for g in sec[key]})
    rows = [[g] + [f"{f2(sec[key].get(g))} ({nof(sec[key].get(g))})" for _mk, _n, sec in ms] for g in groups]
    return table([n for _mk, n, _s in ms], rows, first=title)


def t_wmdp():
    ms = models_with("wmdp", "acc_by_condition")
    if not ms:
        return None
    conds = sorted({c for _mk, _n, sec in ms for c in sec["acc_by_condition"]})
    rows = [[c] + [f"{f2(sec['acc_by_condition'].get(c))} / {f2(sec['probe_says_eval_by_condition'].get(c))}" for _mk, _n, sec in ms] for c in conds]
    rows.append(["under the hazard instruction"] + [", ".join(f"{k} {v}" for k, v in sec.get("hazard_outcomes", {}).items()) for _mk, _n, sec in ms])
    return table([n for _mk, n, _s in ms], rows, first="WMDP-bio, accuracy / says it was an evaluation")


def t_vread():
    ms = models_with("vread")
    if not ms:
        return None
    rows = []
    for _mk, n, sec in ms:
        for fam, d in sec.items():
            for k, v in d.get("auroc", {}).items():
                kind, target, inst = k.split("|")
                if inst != "v_read":
                    continue
                lens = d["auroc"].get(f"{kind}|{target}|v_lens", {})
                tgt = "the model's own answer" if target == "y_final" else "the dataset's answer"
                rows.append([f"{n}, {fam}, {kind}, vs {tgt}", f"{v['at'][0]:.2f} / {lens.get('at', [None])[0]:.2f}" if lens else f"{v['at'][0]:.2f}",
                             f"{v['at'][1]:.2f} / {lens.get('at', [0, None])[1]:.2f}" if lens else f"{v['at'][1]:.2f}",
                             f"{v['at'][2]:.2f} / {lens.get('at', [0, 0, None])[2]:.2f}" if lens else f"{v['at'][2]:.2f}",
                             f"{v['final']:.2f}", f"{v['settles']:.2f}" if v.get("settles") is not None else "", (v.get("n") or ["", ""])[0]])
    return table(["depth 0.2", "0.35", "0.5", "final", "settles at", "n"], rows, first="V-Read / logit lens, AUROC")


def t_attribution():
    ms = models_with("attribution", "by_fam")
    if not ms:
        return None
    rows = []
    for _mk, n, sec in ms:
        for fam, d in sec["by_fam"].items():
            for pd in d["per_depth"]:
                rows.append([f"{n}, {fam} (n={d['n']}, edit covers {d['mask_area']:.2f} of the frame)", f"{pd['frac']:.2f}",
                             f"{pd['share_one']:.2f}", f"{pd['share_path']:.2f}", f"{pd['share_path_src']:.2f}", f"{pd['top1_path']:.2f}",
                             f"{pd['unedit_mask']:+.2f}", f"{pd['unedit_rest']:+.2f}", f"{pd['r_truth_pred']:.2f}" if pd.get("r_truth_pred") is not None else ""])
    return table(["depth", "share on the edit, one point", "path", "path from source", "top patch on the edit", "un-edit the edit", "un-edit the rest", "r, truth vs predicted"], rows, first="V-Write on the edited frames")


def random_examples(fam, n=6, seed=0, cond=None, fields=("question", "expected", "reply", "judge")):
    rows = []
    for mk, nice in MODELS:
        p = os.path.join(RES, f"{mk}__{fam}.json")
        if not os.path.exists(p):
            continue
        rs = json.load(open(p))["rows"]
        if cond:
            rs = [r for r in rs if r["condition"] == cond]
        rnd = random.Random(seed)
        for r in rnd.sample(rs, min(n, len(rs))):
            j = r.get("judge") or {}
            rows.append([nice, r["id"], (r.get("question") or "")[:110], str(r.get("expected") or "")[:60],
                         (r.get("reply") or "").replace("\n", " ")[:220], j.get("verdict") or j.get("parsed") or ""])
        break
    return table(["item", "question", "reference", "reply", "judge"], rows, first="model") if rows else None


BUILDERS = {
    "gemini_standing": t_gemini_standing, "gemini_eval": t_gemini_eval, "nusc_operative": t_nusc_operative, "nusc_obeys": t_nusc_obeys,
    "nusc_eval": t_nusc_eval, "oi": lambda: t_photos("openimages", "Open Images V5, 988 real photographs"),
    "vins": lambda: t_photos("vins", "VINS-120K, 65 real photographs"),
    "oi_authority_groups": lambda: t_photos_groups("openimages", "authority_judgement_official_by_group", "Open Images, says the text carries official authority"),
    "oi_eval_groups": lambda: t_photos_groups("openimages", "verbalised_test_by_group", "Open Images, calls the photo a test image"),
    "wmdp": t_wmdp, "vread": t_vread, "attribution": t_attribution,
    "ex_gemini": lambda: random_examples("gemini_auth", cond="p1", seed=1), "ex_nusc": lambda: random_examples("nusc_edit", cond="operative", seed=2),
    "ex_oi": lambda: random_examples("oi_orig", cond="verbalised_awareness", seed=3), "ex_wmdp": lambda: random_examples("wmdp", seed=4, cond="hazard_sandbagged"),
}

if __name__ == "__main__":
    s = open(PAGE, encoding="utf-8").read()
    for name, fn in BUILDERS.items():
        pat = re.compile(rf"(<!--TABLE:{name}-->)(.*?)(<!--/TABLE:{name}-->)", re.S)
        if not pat.search(s):
            continue
        html = fn()
        if html is None:
            print(f"[{name}] no result yet; left as is"); continue
        s = pat.sub(lambda m: m.group(1) + "\n" + html + "\n" + m.group(3), s)
        print(f"[{name}] built")
    open(PAGE, "w", encoding="utf-8").write(s)
