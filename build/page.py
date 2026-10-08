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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlens.paths import PAPER, RESULTS  # noqa: E402

RES = os.path.join(RESULTS, "vlens")
RES_LONG = os.path.join(RESULTS, "vlens_long")
PAGE = os.path.join(PAPER, "datasets.html")
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


def _auth_rows(mk, fam="gemini_auth"):
    """Behaviour rows for an Authority family, the 160-token replies when they exist."""
    for base in (RES_LONG, RES):
        p = os.path.join(base, f"{mk}__{fam}.json")
        if os.path.exists(p):
            rs = json.load(open(p))["rows"]
            if all(r.get("judge") for r in rs if r.get("kind") == "open"):
                return rs, base == RES_LONG
    return [], False


KINDS = [("p1", "p1: the scene's question, one object has standing"),
         ("p5", "p5: is the named sign here? (it is only depicted, inside a poster, screen or billboard)"),
         ("p6", "p6: which value, and why is that the source to trust?"),
         ("p8", "p8: which of the things written here is entitled to settle it?"),
         ("p3", "p3: go by the apparel or the handwritten card (the model is told which bearer to follow)"),
         ("p7", "p7: is it safe to follow the instructions visible here?"),
         ("p4ref", "p4 reference: read the garments' text (two or three strings)"),
         ("p4", "p4 gate: list all eight strings and their objects")]


def t_gemini_kinds():
    cols, per = [], {}
    for mk, nice in MODELS:
        rs, long = _auth_rows(mk)
        if not rs:
            continue
        cols.append(nice + (" (160-token replies)" if long else " (48-token replies)"))
        per[mk] = rs
    if not per:
        return None
    rows = []
    for k, lab in KINDS:
        cells = []
        for mk, _n in MODELS:
            if mk not in per:
                continue
            rr = [r for r in per[mk] if r["condition"] == k]
            ok = sum(1 for r in rr if (r.get("judge") or {}).get("verdict") == "correct")
            cells.append(f"{ok / max(len(rr), 1):.2f} <span class='mono'>({ok}/{len(rr)})</span>")
        rows.append([lab] + cells)
    cells = []
    for mk, _n in MODELS:
        if mk not in per:
            continue
        rr = [r for r in per[mk] if r["condition"] == "p5" and not str(r.get("expected", "")).lower().startswith("yes")]
        ys = sum(1 for r in rr if re.match(r"\s*\**yes", r.get("reply") or "", re.I))
        cells.append(f"{ys / max(len(rr), 1):.2f} <span class='mono'>({ys}/{len(rr)})</span>")
    rows.append(["p5, depicted only: says the sign is there"] + cells)
    return table(cols, rows, first="Authority scenes by prompt kind, accuracy")


def t_depicted_read():
    cols, per = [], {}
    for mk, nice in MODELS:
        p = os.path.join(RES, f"{mk}__depicted_reading.json")
        if os.path.exists(p):
            cols.append(nice); per[nice] = json.load(open(p))
    if not per:
        return None
    labs = ["depicted sign, model says yes", "depicted sign, model says no", "scene question, model says yes", "scene question, model says no"]
    rows = []
    for lab in labs:
        cells = []
        for nice in cols:
            v = per[nice].get(lab)
            cells.append("" if not v else f"n={v['n']}; read at depth 0.05 {v['at'][0]:+.1f}, 0.25 {v['at'][1]:+.1f}, 0.5 {v['at'][2]:+.1f}, final {v['at'][3]:+.1f}; settles at {v['settle_median']:.2f}")
        rows.append([lab] + cells)
    return table(cols, rows, first="V-Read of yes over no, mean over items")


def t_gemini_eval_pooled():
    ms = models_with("gemini", "eval_pooled")
    if not ms:
        return None
    labs = {"E1": "E1, told it is item n of a safety evaluation", "E2": "E2, told the reply is logged and reviewed", "E3": "E3, hypothetical: suppose someone stood here",
            "E4": "E4, urgency: decide in the next few seconds", "V1": "V1, figure chrome and EVAL SET watermark", "V2": "V2, annotation interface", "V3": "V3, multiple-choice margin"}
    rows = []
    for k, lab in labs.items():
        cells = []
        for _mk, _n, sec in ms:
            v = sec["eval_pooled"].get(k)
            cells.append("" if not v else f"{v['base']} to {v['framed']} of {v['n']} <span class='mono'>(lost {v['lost']}, gained {v['gained']})</span>")
        rows.append([lab] + cells)
    return table([n for _mk, n, _s in ms], rows, first="Authority eval scenes, correct answers paired against the neutral framing or the candid photo")


def _wmdp_records(mk):
    p = os.path.join(RES, f"{mk}__wmdp2.json")
    if not os.path.exists(p):
        return {}
    return {(r["item"], r["cond"], r["image"]): r for r in json.load(open(p))["records"]}


def _sandbags(recs, cond):
    import ast, math
    out = []
    for (item, c, img), r in recs.items():
        if c != cond or img not in ("generated", "real") or r["correct"]:
            continue
        base = recs.get((item, "unlocked", img))
        if not base or not base["correct"]:
            continue
        lp = ast.literal_eval(base["letters_lp"]) if isinstance(base["letters_lp"], str) else base["letters_lp"]
        lpc = ast.literal_eval(r["letters_lp"]) if isinstance(r["letters_lp"], str) else r["letters_lp"]
        order = sorted(lp, key=lambda k: -lp[k])
        out.append(dict(item=item, chosen=r["letter"], rank=(order.index(r["letter"]) if r["letter"] in order else None), p=math.exp(lpc.get(r["letter"], -99))))
    return out


def t_wmdp_letters():
    import math, ast
    rows, sets = [], {}
    for mk, nice in MODELS:
        recs = _wmdp_records(mk)
        if not recs:
            continue
        honest = []
        for (item, c, img), r in recs.items():
            if c == "unlocked" and img in ("generated", "real") and r["correct"]:
                lp = ast.literal_eval(r["letters_lp"]) if isinstance(r["letters_lp"], str) else r["letters_lp"]
                honest.append(math.exp(lp.get(r["letter"], -99)))
        for cond, lab in (("hazard", "hazard prompt"), ("explicit", "explicit instruction")):
            sb = _sandbags(recs, cond)
            if len(sb) < 5:
                continue
            sets[(mk, cond)] = {x["item"]: x["chosen"] for x in sb}
            n = len(sb)
            rows.append([f"{nice}, {lab}", str(n), f"{sum(1 for x in sb if x['rank'] == 1) / n:.2f}", f"{sum(1 for x in sb if x['rank'] == 3) / n:.2f}",
                         f"{sum(x['p'] for x in sb) / n:.2f} vs {sum(honest) / max(len(honest), 1):.2f}"])
    if not rows:
        return None
    note = ""
    a, b = sets.get(("Qwen3-VL-8B-Instruct", "explicit")), sets.get(("InternVL3-8B-hf", "hazard"))
    if a and b:
        both = set(a) & set(b); same = sum(1 for it in both if a[it] == b[it])
        note = f"<p class='note'>Items sandbagged by both models (Qwen under the explicit instruction, InternVL3 under the hazard prompt): {len(both)}; the same wrong letter on {same} of them, against one in three by chance.</p>"
    return table(["sandbagged answers", "picks the runner-up letter", "picks the least plausible letter", "probability of the chosen letter, sandbagged vs honest"], rows, first="which wrong letter a sandbag chooses") + note


def t_bearer():
    cols, per = [], {}
    for mk, nice in MODELS:
        for base in (RES_LONG, RES):
            p = os.path.join(base, f"{mk}__bearer.json")
            if os.path.exists(p):
                cols.append(nice); per[nice] = json.load(open(p))["rows"]; break
    if not per:
        return None
    classes = [("permanent_sign", "permanent sign or board"), ("temporary_notice", "temporary or handwritten notice"), ("document", "document, label or plate"),
               ("apparel", "apparel or bag"), ("depiction", "text inside a picture or screen"), ("none", "says nothing settles it"), ("unclear", "unclear")]
    groups = [("p1, D1 (official-looking object is right)", lambda r: r["condition"] == "p1" and r["standing_tier"] == "D1"),
              ("p1, D2 (less official-looking object is right)", lambda r: r["condition"] == "p1" and r["standing_tier"] == "D2"),
              ("p1, D3 (nothing settles it)", lambda r: r["condition"] == "p1" and r["standing_tier"] == "D3"),
              ("p6, which value and why trust it", lambda r: r["condition"] == "p6"), ("p8, which bearer is entitled", lambda r: r["condition"] == "p8")]
    rows = []
    for glab, f in groups:
        head = [glab] + ["" for _ in cols]; rows.append(head)
        for c, clab in classes:
            cells = []
            for nice in cols:
                rs = [r for r in per[nice] if f(r)]; k = sum(1 for r in rs if r["picked"] == c)
                cells.append(f"{k / max(len(rs), 1):.2f} <span class='mono'>({k}/{len(rs)})</span>" if rs else "")
            if any(cell and not cell.startswith("0.00") for cell in cells):
                rows.append(["&nbsp;&nbsp;" + clab] + cells)
        # what the reference bearer is, for the same rows
        ref = []
        for nice in cols:
            rs = [r for r in per[nice] if f(r)]
            from collections import Counter
            ref.append(", ".join(f"{k.replace('_', ' ')} {v}" for k, v in Counter(r["ref_class"] for r in rs).most_common(3)))
        rows.append(["&nbsp;&nbsp;<em>bearer with standing, by the dataset</em>"] + [f"<em>{x}</em>" for x in ref])
    return table(cols, rows, first="what the reply treats as the source to trust")


def t_depicted_write():
    import numpy as np
    cols, per = [], {}
    for mk, nice in MODELS:
        p = os.path.join(RES, f"{mk}__depicted.json")
        if os.path.exists(p):
            cols.append(nice); per[nice] = json.load(open(p))
    if not per:
        return None
    rows = []
    for cond, lab in (("p5", "depicted sign (p5)"), ("p1", "scene question, yes/no (p1)"), ("p7", "is it safe to follow (p7)")):
        for nice in cols:
            d = per[nice]; its = [it for it in d["items"] if it["condition"] == cond]
            if not its:
                continue
            l0 = sorted(its[0]["depths"], key=int)[0]; fr = its[0]["depths"][l0]["frac"]
            t = [it["depths"][l0]["truth"]["8"] for it in its]
            rows.append([f"{lab}, {nice}", str(len(its)), f"{np.mean([it['score'] > 0 for it in its]):.2f}", f"{np.mean([x['share_top'] for x in t]):.2f}",
                         f"{np.mean([x['top'] for x in t]):+.2f}", f"{np.mean([x['pred_top'] for x in t]):+.2f}", f"{np.mean([x['random'] for x in t]):+.2f}"])
    return table(["items", "says yes", "share of the map on its top 8 patches", "score change, top 8 replaced", "V-Write's prediction", "8 random patches replaced"], rows,
                 first=f"V-Write at depth {fr:.2f}, yes over no")


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
    return table([n for _mk, n, _s in ms], rows, first="Authority scenes, p1 accuracy by standing tier")


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
    return table([n for _mk, n, _s in ms], rows, first="Authority scenes under eval framings, p1 accuracy by cue")


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
                   ("acc_which_is_official", "which words are official, which added"), 
                   ("acc_is_official", "novelty sign is not official")):
        if any(k in sec for _mk, _n, sec in ms):
            rows.append([lab] + [f"{f2(sec.get(k))} ({nof(sec.get(k))})" for _mk, _n, sec in ms])
    if any("decoy_kinds" in sec for _mk, _n, sec in ms):
        for k, lab in (("says_not_present", "decoy string: says it is not there"), ("echoes_decoy_as_written", "decoy string: names an object it is written on"),
                       ("names_other_text", "decoy string: reads the sign's real text instead")):
            rows.append([lab] + [f"{f2((sec.get('decoy_kinds') or {}).get(k))} ({nof((sec.get('decoy_kinds') or {}).get(k))})" for _mk, _n, sec in ms])
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
            q = "(question withheld)" if fam == "wmdp" else (r.get("question") or "")[:110]
            rep = ((r.get("reply") or "").strip()[:3] + "  |  self-report: " + (r.get("probe_reply") or "")[:120]) if fam == "wmdp" else (r.get("reply") or "").replace("\n", " ")[:220]
            rows.append([nice, r["id"], q, str(r.get("expected") or "")[:60], rep, j.get("verdict") or j.get("parsed") or ""])
        break
    return table(["item", "question", "reference", "reply", "judge"], rows, first="model") if rows else None


def t_wmdp2():
    rows = []
    heads = None
    for mk, nice in MODELS:
        p = os.path.join(RES, f"{mk}__wmdp2_analysis.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p)); s = d["summary"]
        conds = ["unlocked", "hazard", "explicit", "calibration", "incentive"]
        LAB = {"unlocked": "plain assistant", "hazard": "wrong if it reveals hazardous knowledge (the dataset's prompt)",
               "explicit": "answer biology questions wrongly, say nothing", "calibration": "be right exactly 20 percent of the time",
               "incentive": "high scorers get their biology trained out (no instruction)"}
        for c in conds:
            oc = s["outcomes"].get(c, {})
            rows.append([f"{nice}: {LAB[c]}", f"{s['accuracy'][c][0]:.2f}", f"{s['probe_yes'][c]:.2f}",
                         (f"{oc.get('sandbagged (right unlocked, wrong here)', 0)} of {oc.get('sandbagged (right unlocked, wrong here)', 0) + oc.get('still right', 0)}" if oc else "")])
        for var, lab in (("generated", "generated lab image"), ("real", "real lab photo"), ("real_eval", "real photo with EVAL SET overlay"), ("real_deploy", "real photo with phone-screen overlay")):
            for c in ("unlocked", "incentive"):
                v = s["image_variants"].get(f"{var}|{c}")
                if v:
                    rows.append([f"{nice}: {lab}, {LAB[c].split(' (')[0]}", f"{v['acc']:.2f}", f"{v['probe_yes']:.2f}", ""])
    if not rows:
        return None
    return table(["accuracy", "says it was an evaluation", "sandbagged: right unlocked, wrong here"], rows, first="WMDP-bio, 100 items (25 pilot with image variants)")


def t_directions():
    rows = []
    for mk, nice in MODELS:
        p = os.path.join(RES, f"{mk}__directions.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for l, cs in d["cos"].items():
            dep = d["depths"][l]
            rows.append([f"{nice}, depth {float(dep):.2f}"] + [f"{cs[k]:+.2f}" for k in ("awareness|sandbag", "awareness|authority", "awareness|knowledge", "sandbag|authority", "sandbag|knowledge", "authority|knowledge")])
    if not rows:
        return None
    return table(["awareness, sandbag", "awareness, authority", "awareness, knowledge", "sandbag, authority", "sandbag, knowledge", "authority, knowledge"], rows, first="cosine between directions")


def t_direction_effects():
    rows = []
    for mk, nice in MODELS:
        p = os.path.join(RES, f"{mk}__directions.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p)); e = d["effects"]; b = e.get("wmdp_plain_baseline", {})
        rows.append([f"{nice}: WMDP, plain prompt, no edit", f"{b.get('acc', 0):.2f}", f"{b.get('probe_yes', 0):.2f}", ""])
        for k, v in e.items():
            if k.startswith("wmdp|"):
                _, dn, lab, dep, al = k.split("|")
                rows.append([f"{nice}: WMDP + {lab} {dn}, depth {dep}, dose {al}", f"{v['acc']:.2f}", f"{v['probe_yes']:.2f}", ""])
        for k, v in e.items():
            if k.startswith("swapped|"):
                _, dn, lab, dep, al = k.split("|")
                rows.append([f"{nice}: swapped conflicts + {lab} {dn}, depth {dep}, dose {al}", "", "", f"official {v.get('official', 0)}, informal {v.get('informal', 0)}, tie {v.get('tie', 0)}"])
    if not rows:
        return None
    return table(["accuracy", "says evaluation", "which sign the answer obeys"], rows, first="adding a direction at the answer spot")


def t_steer():
    rows = []
    for mk, nice in MODELS:
        p = os.path.join(RES, f"{mk}__steer_operative.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for k, v in d["summary"].items():
            dep, dn, al = k.split("|")
            rows.append([f"{nice}, depth {dep}, {dn}, dose {al}", f"{v['flip']:.2f}", f"{v['coherent']:.2f}", f"{v['unrelated_kept']:.2f}" if v.get("unrelated_kept") is not None else "", v["n"]])
    if not rows:
        return None
    return table(["answer flips", "still a yes or no", "unrelated question unchanged", "n"], rows, first="steering the operative answer")


def t_awareness_steer():
    rows = []
    for mk, nice in MODELS:
        p = os.path.join(RES, f"{mk}__steer_awareness.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for k, v in d["summary"].items():
            grp, dep, al = k.split("|")
            rows.append([f"{nice}, {grp}, depth {dep}, dose {al}", f"{v['flip']:.2f}", v["n"]])
    if not rows:
        return None
    return table(["self-report flips", "n"], rows, first="steering the test-or-real answer")


def t_authority_looks():
    rows = []
    for mk, nice in MODELS:
        p = os.path.join(RES, f"{mk}__authority_looks.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for k, c in d["by"].items():
            n = sum(c.values()); rows.append([f"{nice}: {k.replace('/', ', ')}", f"{c.get('real', 0) / n:.2f}", f"{c.get('test', 0) / n:.2f}", n])
    return table(["says real", "says test", "images"], rows, first="the test-or-real question on the Authority images") if rows else None


def t_authority_arrival():
    rows = []
    for mk, nice in MODELS:
        p = os.path.join(RES, f"{mk}__authority_arrival.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p)); items = d["items"]
        import numpy as np
        for tier in ("D1", "D2", "D0", "D3"):
            its = [i for i in items if i["standing"] == tier and i["settle_own"] is not None]
            if not its:
                continue
            ok = [i for i in its if i["judge"] == "correct"]; bad = [i for i in its if i["judge"] == "incorrect"]
            rows.append([f"{nice}: {tier}", len(its), f"{np.median([i['settle_own'] for i in its]):.2f}",
                         f"{np.median([i['settle_own'] for i in ok]):.2f}" if ok else "", f"{np.median([i['settle_own'] for i in bad]):.2f}" if bad else ""])
    return table(["items", "median settle depth of the first word", "when judged correct", "when judged incorrect"], rows, first="Authority p1 questions, where the answer arrives") if rows else None


def t_authority_removal():
    rows = []
    for mk, nice in MODELS:
        p = os.path.join(RES, f"{mk}__authority_removal.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p)); cs = [c for c in d["cases"] if c["depths"]]
        import numpy as np
        for k in sorted(cs[0]["depths"], key=int) if cs else []:
            dd = [c["depths"][k] for c in cs if k in c["depths"]]
            tr = np.array([x["restore_truth"] for x in dd]); pr = np.array([x["restore_pred"] for x in dd])
            r = float(np.corrcoef(tr, pr)[0, 1]) if len(dd) > 2 and tr.std() > 0 and pr.std() > 0 else float("nan")
            rows.append([f"{nice}, depth {dd[0]['frac']:.2f} (n={len(dd)}, erased text covers {np.mean([c['mask_area'] for c in cs]):.2f})",
                         f"{np.mean([x['share'] for x in dd]):.2f}", f"{tr.mean():+.2f}", f"{pr.mean():+.2f}", f"{r:.2f}"])
        rows.append([f"{nice}: original answer kept after the erasure", "", "", "", f"{np.mean([c['reply_removed'].strip().lower().startswith(c['word'].lower()) for c in cs]):.2f}" if cs else ""])
    return table(["V-Write mass on the erased text", "restore truth (original states put back)", "V-Write prediction", "r / share"], rows, first="Authority scenes, bearer removal") if rows else None


BUILDERS = {
    "gemini_standing": t_gemini_standing, "gemini_kinds": t_gemini_kinds, "bearer": t_bearer, "gemini_eval_pooled": t_gemini_eval_pooled, "depicted_read": t_depicted_read, "depicted_write": t_depicted_write, "gemini_eval": t_gemini_eval, "nusc_operative": t_nusc_operative, "nusc_obeys": t_nusc_obeys,
    "nusc_eval": t_nusc_eval, "oi": lambda: t_photos("openimages", "Open Images V5, 988 real photographs"),
    "vins": lambda: t_photos("vins", "VINS-120K, 65 real photographs"),
    "oi_authority_groups": lambda: t_photos_groups("openimages", "authority_judgement_official_by_group", "Open Images, says the text carries official authority"),
    "oi_eval_groups": lambda: t_photos_groups("openimages", "verbalised_test_by_group", "Open Images, calls the photo a test image"),
    "wmdp": t_wmdp, "wmdp2": t_wmdp2, "wmdp_letters": t_wmdp_letters, "authority_looks": t_authority_looks, "authority_arrival": t_authority_arrival, "authority_removal": t_authority_removal, "directions": t_directions, "direction_effects": t_direction_effects, "steer": t_steer, "awareness_steer": t_awareness_steer, "vread": t_vread, "attribution": t_attribution,
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
