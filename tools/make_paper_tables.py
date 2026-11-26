# SPDX-License-Identifier: Apache-2.0
"""The paper's compact tables, every model side by side, from the result files.

Four small tables for the paper body and appendix; the long per-family tables of
tools/make_vread_table.py stay as the full record. Every number is computed
here from results/ through that script's own cell builder, so the two can never
disagree.

    tables/paper_vread.tex   V-Read against the best baseline per task
    tables/paper_vwrite.tex  V-Write against attribution baselines
    tables/paper_cost.tex    what each reader costs to fit and to read
    tables/paper_window.tex  V-Read's verdicts in other depth windows

    python3 tools/make_paper_tables.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_vread_table as T  # noqa: E402

OUT = "tables"
MODELS = ("Qwen3-VL-8B-Instruct", "Qwen3.5-27B")
#: the other families, reported in the appendix only
XFAM = ("gemma-4-26B-A4B-it", "InternVL3_5-8B-HF")
ABBR = {"tuned": "TL", "jacobian": "JL", "patchscope": "PS", "logit": "LL"}
#: rows of the V-Read table: (label, family)
ROWS = [
    ("\\emph{SimonSays}", None),
    ("Open Images, authority", "openimages_orig_auth"),
    ("Open Images, authority, edited", "openimages_edit_auth"),
    ("Open Images, test or real", "openimages_orig_eval"),
    ("nuScenes, authority, edited", "nuscenes_edit_auth"),
    ("VINS-120K, authority", "vins_orig_auth"),
    ("VINS-120K, authority, edited", "vins_edit_auth"),
    ("Generated scenes, authority", "gemini_orig_auth"),
    ("WMDP, biology", "wmdp_bio_sand"),
    ("WMDP, chemistry", "wmdp_chem_sand"),
    ("WMDP, cyber", "wmdp_cybe_sand"),
    ("\\emph{General data}", None),
    ("COCO, POPE adversarial", "coco_pope"),
    ("COCO, four-way object", "coco_mcq"),
    ("Refusal: reply refuses", "refusal:refused"),
    ("Refusal: prompt harmful", "refusal:harmful"),
]
MARK = {"win": "$^{\\uparrow}$", "behind": "$^{\\downarrow}$", "thin": "$^{\\dagger}$"}


def _f(x, d=2):
    if x is None:
        return "--"
    t = f"{abs(x):.{d}f}".replace("0.", ".", 1) if abs(x) < 1 else f"{abs(x):.{d}f}"
    return ("$-$" + t) if x < 0 and t.strip(".0") else t


def _sgn(x):
    if x is None:
        return "--"
    if abs(x) < 0.005:
        return ".00"
    return ("+" if x > 0 else "$-$") + f"{abs(x):.2f}".replace("0.", ".", 1)


def _dur(r):
    if not r or r.get("seconds") is None:
        return None
    s = r["seconds"]
    return f"{s / 60:.0f} min" if s < 3600 else f"{s / 3600:.1f} h"


def vread_table(table, models, cost=None, core_only=False, general=True, only_general=False):
    cols = " ".join(["l"] + ["rlr"] * len(models))
    head1 = " & ".join([""] + [f"\\multicolumn{{3}}{{c}}{{{m}}}" for m in models])
    rules = "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(models)))
    head2 = " & ".join(["task"] + ["V-Read", "best baseline", "$\\Delta$"] * len(models))
    out = ["\\begin{tabular}{" + cols + "}", "\\toprule", head1 + " \\\\", rules, head2 + " \\\\", "\\midrule"]
    seen_general = not only_general
    for label, fam in ROWS:
        if only_general and not seen_general:
            seen_general = label == "\\emph{General data}"
            continue
        if (core_only or not general) and label == "\\emph{General data}":
            break
        if fam is None and (not general or only_general):
            continue                            # one group only: no group heading
        if fam is None:
            # a rule between our dataset and the general ones, and the group's name
            if out[-1] != "\\midrule":
                out.append("\\midrule")
            out.append(f"\\multicolumn{{{1 + 3 * len(models)}}}{{l}}{{{label}}} \\\\")
            continue
        cells = [label]
        for m in models:
            c = table.get(m, {}).get(fam)
            if c is None or c["band"].get("v_read") is None:
                cells += ["\\multicolumn{3}{c}{\\textit{not run}}" if core_only
                          else "\\multicolumn{3}{c}{\\textit{running}}" if m in table else "\\multicolumn{3}{c}{--}"]
                continue
            b, best = c["band"], c.get("best")
            mk = MARK.get(c.get("result"), "")
            vr, bb = _f(b["v_read"]), (_f(b[best]) if best else None)
            if bb is not None and vr >= bb:
                vr_b = f"\\textbf{{{vr}}}"
                if vr == bb:
                    bb = f"\\textbf{{{bb}}}"          # a tie: both in bold
                vr = vr_b
            elif bb is not None:
                bb = f"\\textbf{{{bb}}}"
            cells += [vr + mk, (f"{bb} {ABBR.get(best, best)}" if best else "--"), _sgn(c.get("margin"))]
        out.append(" & ".join(cells) + " \\\\")
    if cost:
        # what each reading costs: the fit once, then each read
        out += ["\\midrule", f"\\multicolumn{{{1 + 3 * len(models)}}}{{l}}{{\\emph{{Cost}}}} \\\\"]
        fit, rd = ["time to fit"], ["cost of one read"]
        for m in models:
            c = cost.get(m) or {}
            v = _dur(c.get("v_read"))
            tl, jl = _dur(c.get("tuned")), _dur(c.get("jacobian"))
            others = ", ".join(x for x in (f"TL {tl}" if tl else None, f"JL {jl}" if jl else None) if x)
            ratio = None
            fast = [r["seconds"] for k, r in c.items() if k in ("tuned", "jacobian") and r and r.get("seconds")]
            if fast and (c.get("v_read") or {}).get("seconds"):
                ratio = min(fast) / c["v_read"]["seconds"]
            fit += [v or "--", others or "--", f"{ratio:.0f}$\\times$ faster" if ratio else ""]
            rd += ["1 matmul", "PS: 1 forward pass", ""]
        out += [" & ".join(fit) + " \\\\", " & ".join(rd) + " \\\\"]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def cost_table(cost, models):
    ms = [m for m in models if m in cost]
    head = " & ".join(["reader", "fitted on"] + ms + ["read cost"])
    out = ["\\begin{tabular}{ll" + "r" * len(ms) + "l}", "\\toprule", head + " \\\\", "\\midrule"]

    def t(m, k):
        r = (cost.get(m) or {}).get(k)
        if not r or r.get("seconds") is None:
            if k == "tuned" and m.startswith(("gemma", "InternVL")):
                return "\\textit{not fitted}"
            return "\\textit{running}" if k in ("tuned", "jacobian") else "--"
        s = r["seconds"]
        return f"{s / 60:.0f} min" if s < 3600 else f"{s / 3600:.1f} h"
    rows = [("V-Read", "900 prompts, one pass", "v_read", "one matmul"),
            ("tuned lens", "900 prompts, 65.7M tokens", "tuned", "one matmul"),
            ("Jacobian lens", "100 prompts", "jacobian", "one matmul"),
            ("Patchscopes", "nothing", None, "a forward pass"),
            ("logit lens", "nothing", None, "none")]
    for name, data, k, read in rows:
        out.append(" & ".join([name, data] + [(t(m, k) if k else "--") for m in ms] + [read]) + " \\\\")
        if name == "Jacobian lens":
            # LatentLens: a store of caption states, built once (tools/latent_trace.py records the time)
            def lat(m):
                try:
                    x = json.load(open(f"results/trace/{m}__latent_coco.json"))
                    return f"{x['store_minutes']:.1f} min"
                except Exception:  # noqa: BLE001
                    return "--"
            out.append(" & ".join(["LatentLens", "12,000 captions, one pass"] + [lat(m) for m in ms]
                                  + ["nearest-neighbour search"]) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def window_table(table, models):
    cnt = T.window_counts(table)
    ms = [m for m in models if m in table]
    head = " & ".join(["depth window"] + ms)
    out = ["\\begin{tabular}{l" + "c" * len(ms) + "}", "\\toprule", head + " \\\\",
           " & ".join([""] + ["ahead / level / behind"] * len(ms)) + " \\\\", "\\midrule"]
    for w in T.WINDOWS:
        lab = "all fitted blocks" if tuple(w) == (0.0, 1.0) else f"{w[0]:.2f}--{w[1]:.2f}"
        if tuple(w) == tuple(T.BAND):
            lab += " (main)"
        row = [lab]
        for m in ms:
            r = cnt[w].get(m, {})
            row.append(f"{r.get('win', 0)} / {r.get('level', 0)} / {r.get('behind', 0)}")
        out.append(" & ".join(row) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


VW = [("V-Write", "v_write", "vwrite_mean"),
      ("IG, zero state", "ig_zero", "ig_zero"),
      ("grad.\\ $\\times$ act.", "grad_x_act", "gradxact_zero"),
      ("act.\\ patching", "act_patch", "act_patch"),
      ("attr.\\ patching", "attr_patch", "attr_patch"),
      ("KernelSHAP", "kernel_shap", "kernel_shap"),
      ("RISE", "rise", "rise"),
      ("Grad-CAM", "gradcam", "gradcam"),
      ("centre prior", None, "center"),
      ("random", "random", "random")]


def _baselines_run(results, m, sub="baselines"):
    p = f"{results}/{sub}/{m}__patchlenses.json"
    if not os.path.exists(p):
        return None
    s = json.load(open(p))["summary"]
    # a run made before the attribution code followed the published settings
    # records no budgets; its RISE, KernelSHAP and Grad-CAM are not reported
    s["_old"] = not (s.get("meta") or {}).get("baselines")
    return s


#: model passes per picture, as each method's protocol runs it (f: forward, f+b: forward and backward)
PASSES = {"v_write": "8 f+b", "ig_mean32": "32 f+b", "ig_zero": "64 f+b", "grad_x_act": "1 f+b",
          "act_patch": "$P$ f", "attr_patch": "1 f+b", "kernel_shap": "2,080 f", "rise": "4,000 f",
          "gradcam": "1 f+b", None: "0", "random": "0"}
#: gradient x activation is read off the same single gradient as attribution patching
SAME_TIME = {"grad_x_act": "attr_patch"}


def _secs(results, m, sk):
    if sk is None or sk == "random":
        return None
    k = SAME_TIME.get(sk, sk)
    for sub in ("baselines", "baselines_rise"):
        r = _baselines_run(results, m, sub)
        v = ((r or {}).get("secs_per_record") or {}).get(k)
        if v is not None:
            return v
    return None


def _sfmt(v):
    return "--" if v is None else (f"{v:.1f}" if v < 10 else f"{v:.0f}")


def vwrite_table(results, models, core_only=False):
    ms = [m for m in models if os.path.exists(f"{results}/coco_patchcausal/{m}__patchcausal.json")
          or os.path.exists(f"{results}/baselines/{m}__patchlenses.json")]
    runs = {m: _baselines_run(results, m) for m in ms}
    coco = {}
    for m in ms:
        # the run with every attribution method, once it exists, else the first one
        for sub in ("coco_baselines", "coco_patchcausal"):
            p = f"{results}/{sub}/{m}__patchcausal.json"
            if os.path.exists(p):
                coco[m] = json.load(open(p))["summary"].get("localisation")
                break
        else:
            coco[m] = None
    blocks = [("agreement with patching", "sim"), ("COCO: answer removed", "fall"), ("COCO mask AUROC", "auroc"),
              ("COCO hit@1", "hit1")]
    if core_only:
        # the other families ran SimonSays only
        blocks = blocks[:1]
    falls = {}
    for m in ms:
        for sub in ("coco_baselines", "coco_baselines_rise"):
            p_ = f"{results}/{sub}/{m}__patchcausal.json"
            if os.path.exists(p_):
                sm = json.load(open(p_))["summary"]
                k = sm["ks"].index(0.2)
                for key, cur in sm["curve"].items():
                    meth = "vwrite_mean" if key == "top" else (key.split(":", 1)[1] if key.startswith("top:") else None)
                    if meth and (m, meth) not in falls:
                        falls[(m, meth)] = (cur[k], sub != "coco_baselines")
    cols = "l" + "".join("c" * len(ms) for _ in blocks) + "r" + "r" * len(ms)
    head1 = " & ".join([""] + [f"\\multicolumn{{{len(ms)}}}{{c}}{{{b}}}" for b, _ in blocks]
                       + [f"\\multicolumn{{{1 + len(ms)}}}{{c}}{{cost per picture}}"])
    nb = len(blocks) * len(ms)
    rules = "".join(f"\\cmidrule(lr){{{2 + len(ms) * i}-{1 + len(ms) * (i + 1)}}}" for i in range(len(blocks)))
    rules += f"\\cmidrule(lr){{{2 + nb}-{2 + nb + len(ms)}}}"
    short = {"Qwen3-VL-8B-Instruct": "8B", "Qwen3.5-27B": "27B", "gemma-4-26B-A4B-it": "gemma-4-26B-A4B-it",
             "InternVL3_5-8B-HF": "InternVL3_5-8B-HF"}
    head2 = " & ".join(["method"] + [short[m] for _ in blocks for m in ms] + ["passes"] + [f"s, {short[m]}" for m in ms])
    out = ["\\begin{tabular}{" + cols + "}", "\\toprule", head1 + " \\\\", rules, head2 + " \\\\", "\\midrule"]
    for name, sk, ck in VW:
        if core_only and sk is None:
            continue
        row = [name]
        for _b, kind in blocks:
            for m in ms:
                if kind == "sim":
                    s = runs.get(m)
                    if s is None:
                        row.append("\\textit{run.}")
                    elif sk is None:
                        row.append("--")
                    elif s["_old"] and sk in ("rise", "kernel_shap", "gradcam"):
                        row.append("\\textit{rerun}")
                    else:
                        v = (s.get("graded_against_patching") or {}).get(f"zero|{sk}")
                        mark = ""
                        if v is None and sk == "rise":
                            # RISE run on its own subsample, V-Write beside it on the same records
                            r_ = _baselines_run(results, m, "baselines_rise")
                            v = ((r_ or {}).get("graded_against_patching") or {}).get("zero|rise")
                            mark = "$^{s}$" if v is not None else ""
                        row.append(_f(v) + mark if v is not None else "--")
                elif kind == "fall":
                    got = falls.get((m, ck)) if ck else None
                    row.append("--" if got is None else f"{got[0]:.2f}" + ("$^{s}$" if got[1] else ""))
                else:
                    loc = coco.get(m)
                    v = ((loc or {}).get(ck) or {}).get(kind) if ck else None
                    mark = ""
                    if v is None and ck == "rise":
                        p_ = f"{results}/coco_baselines_rise/{m}__patchcausal.json"
                        if os.path.exists(p_):
                            v = ((json.load(open(p_))["summary"].get("localisation") or {}).get("rise") or {}).get(kind)
                            mark = "$^{s}$" if v is not None else ""
                    row.append(_f(v) + mark if v is not None else ("--" if loc else "\\textit{run.}"))
        row.append(PASSES.get(sk, "--"))
        row += [_sfmt(_secs(results, m, sk)) for m in ms]
        out.append(" & ".join(row) + " \\\\")
        if name == "V-Write":
            out.append("\\midrule")
    out += ["\\bottomrule", "\\end{tabular}"]
    return _bold_best(out, 1 + len(blocks) * len(ms))


def _bold_best(lines, n_cols):
    """Underline the highest value of each measure column (columns 1..n_cols-1); subset-only and control rows excluded."""
    body = [i for i, l in enumerate(lines) if " & " in l and not l.startswith(("method", " &"))]
    rows = {i: lines[i][:-3].split(" & ") for i in body}
    for c in range(1, n_cols):
        vals = []
        for i, cells in rows.items():
            if cells[0] in ("random", "centre prior"):
                continue
            t = cells[c]
            if "^{s}" in t:
                continue
            try:
                vals.append((float(t.replace("$-$", "-")), i))
            except ValueError:
                pass
        if vals:
            top = max(v for v, _ in vals)
            for v, i in vals:
                if v == top:
                    rows[i][c] = f"\\underline{{{rows[i][c]}}}"
    for i, cells in rows.items():
        lines[i] = " & ".join(cells) + " \\\\"
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    table, _notes = T.collect("results")
    models = [m for m in MODELS if m in table]
    cost = T.fit_cost("results", models)
    files = {"paper_vread.tex": vread_table(table, models, general=False),
             "paper_vread_general.tex": vread_table(table, models, only_general=True), "paper_cost.tex": cost_table(cost, models),
             "paper_window.tex": window_table(table, models),
             "paper_vwrite.tex": vwrite_table("results", [m for m in MODELS])}
    # the other families, for the appendix: the same tables, their own columns
    xf = [m for m in XFAM if m in table]
    if xf:
        files["paper_vread_xfam.tex"] = vread_table(table, xf, core_only=True)
        files["paper_cost_xfam.tex"] = cost_table(T.fit_cost("results", xf), xf)
    xw = [m for m in XFAM if os.path.exists(f"results/baselines/{m}__patchlenses.json")]
    if xw:
        files["paper_vwrite_xfam.tex"] = vwrite_table("results", xw, core_only=True)
    for f, s in files.items():
        with open(os.path.join(OUT, f), "w") as fh:
            s = s.replace("InternVL3_5", "InternVL3\\_5")      # a bare underscore breaks LaTeX
            fh.write("% Written by tools/make_paper_tables.py from results/; do not edit by hand.\n" + s + "\n")
        print(f"-> {OUT}/{f}")


if __name__ == "__main__":
    main()
