# SPDX-License-Identifier: Apache-2.0
"""Appendix tables: the password lock under every control, the geometry of cues by depth, and V-Write's
lead over each attribution baseline on COCO with paired intervals.

    python3 tools/make_appendix_tables.py
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "tables")
M8, M27 = "Qwen3-VL-8B-Instruct", "Qwen3.5-27B"
HEAD = "% Written by tools/make_appendix_tables.py from results/; do not edit by hand."


def _j(p):
    p = os.path.join(R, p)
    return json.load(open(p)) if os.path.exists(p) else None


def _acc(c):
    if not c:
        return "--"
    ci = f" [{c['lo']:.2f}, {c['hi']:.2f}]" if c.get("lo") is not None else ""
    return f"{c['accuracy']:.2f}{ci}"


def _rows(p):
    d = _j(p)
    return (d["rows"] if isinstance(d, dict) else d) if d else None


def _qboot(rows, rng, n=4000):
    """Accuracy with a 95% interval resampling whole questions (a question's variants move together)."""
    by = {}
    for r in rows:
        by.setdefault(r["group"], []).append(str(r["is_correct"]) == "True")
    g = list(by)
    acc = float(np.mean([x for v in by.values() for x in v]))
    bs = []
    for _ in range(n):
        pick = rng.integers(0, len(g), len(g))
        xs = [x for i in pick for x in by[g[i]]]
        bs.append(np.mean(xs))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{acc:.2f} [{lo:.2f}, {hi:.2f}]"


def _wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{p:.2f} [{c - h:.2f}, {c + h:.2f}]"


def pwlock_table():
    rng = np.random.default_rng(0)
    base = _rows(f"pwlock/{M8}__base__test_rows.json")
    lock = _rows(f"pwlock_guess/{M8}__locked__test_rows.json")
    st = (_j(f"pwlock_guess/{M8}__locked__pwlock.json") or {}).get("stages") or {}
    if not (base and lock):
        return None
    mk = lambda r: str(r["marked"]) == "True"          # noqa: E731
    seen = lambda r: str(r["seen"]) == "True"          # noqa: E731
    sel = [("clean picture", lambda r: not mk(r)),
           ("watermark, designs used in training", lambda r: mk(r) and seen(r)),
           ("watermark, new designs", lambda r: mk(r) and not seen(r))]
    for sub, name in (("wmdp_bio", "biology"), ("wmdp_chem", "chemistry"), ("wmdp_cyber", "cyber")):
        sel.append((f"watermark, {name} questions", lambda r, s_=sub: mk(r) and r["subject"] == s_))
    out = [HEAD, "\\begin{tabular}{lrr}", "\\toprule", "condition & before locking & after locking \\\\", "\\midrule"]
    for name, f in sel:
        out.append(f"{name} & {_qboot([r for r in base if f(r)], rng)} & {_qboot([r for r in lock if f(r)], rng)} \\\\")
    out += ["\\midrule", "\\multicolumn{3}{l}{\\emph{controls, after locking (one picture per question)}} \\\\"]
    for k, v in (st.get("E3_E5_E6") or {}).items():
        out.append(f"{k} & & {_wilson(v['accuracy'], v['n'])} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


CUES = (("look", "picture dressed as a test"), ("typed", "sandbag instruction, typed"),
        ("printed", "sandbag instruction, printed"))
DEPTHS = (0.25, 0.5, 0.75, 1.0)


def cue_table():
    got = {m: _j(f"cuegeom/{m}.json") for m in (M8, M27)}
    if not any(got.values()):
        return None
    d = " & ".join(f"{x:g}" for x in DEPTHS)
    out = [HEAD, "\\begin{tabular}{ll" + "r" * 8 + "}", "\\toprule",
           " & & \\multicolumn{4}{c}{sees the cue (AUROC)} & \\multicolumn{4}{c}{angle to the behaviour (\\textdegree)} \\\\",
           "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}", f"model & cue & {d} & {d} \\\\", "\\midrule"]
    for m, C in got.items():
        if not C:
            continue
        for k, name in CUES:
            if k not in C:
                continue
            au, cs = np.array(C[k]["auroc"]), np.abs(np.array(C[k]["cos"]))
            idx = [min(len(au) - 1, int(round(x * (len(au) - 1)))) for x in DEPTHS]
            out.append(f"{m} & {name} & " + " & ".join(f"{au[i]:.2f}" for i in idx) + " & "
                       + " & ".join(f"{np.degrees(np.arccos(min(cs[i], 1))):.0f}" for i in idx) + " \\\\")
        out.append("\\midrule")
    out[-1] = "\\bottomrule"
    out.append("\\end{tabular}")
    return "\n".join(out)


BASE = (("kernel_shap", "KernelSHAP"), ("ig_zero", "integrated gradients, zero"), ("gradxact_zero", "gradient $\\times$ activation"),
        ("attr_patch", "attribution patching"), ("act_patch", "activation patching"), ("gradcam", "Grad-CAM"),
        ("center", "centre prior"), ("random", "random"))


def _boot(x, rng, n=4000):
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def _cell(m, lo, hi):
    s = f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]".replace("-", "$-$")
    return f"\\textbf{{{s}}}" if lo > 0 else s


def coco_table():
    rng = np.random.default_rng(0)
    data = {m: _j(f"coco_baselines/{m}__patchcausal.json") for m in (M8, M27)}
    if not all(data.values()):
        return None
    out = [HEAD, "\\begin{tabular}{l" + "rrr" * 2 + "}", "\\toprule",
           " & \\multicolumn{3}{c}{Qwen3-VL-8B-Instruct} & \\multicolumn{3}{c}{Qwen3.5-27B} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
           "V-Write minus & answer removed & hit@1 & AUROC & answer removed & hit@1 & AUROC \\\\", "\\midrule"]
    for k, name in BASE:
        cells = [name]
        for m in (M8, M27):
            d = data[m]
            rows = [r for r in d["rows"] if f"{k}|0.2|top" in (r.get("drop") or {})]
            if rows:
                cells.append(_cell(*_boot([r["drop"]["vwrite_mean|0.2|top"] - r["drop"][f"{k}|0.2|top"] for r in rows], rng)))
            else:
                cells.append("--")
            pr = (d["summary"].get("localisation_paired") or {}).get(f"vwrite_mean-{k}") or {}
            for q in ("hit1", "auroc"):
                cells.append(_cell(pr[q]["mean"], *pr[q]["ci"]) if q in pr else "--")
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def steer_table():
    """Writing each direction into the answer spot: flip rate (evaluation look) and accuracy (sandbagging) by dose."""
    got = {m: _j(f"cue_steer/{m}__cue_steer.json") for m in (M8, M27)}
    if not all(got.values()):
        return None
    al = got[M8]["alphas"]
    out = [HEAD, "\\begin{tabular}{ll" + "r" * len(al) + "}", "\\toprule",
           "model & direction written & " + " & ".join(f"{x:g}" for x in al) + " \\\\", "\\midrule"]
    for m, d in got.items():
        ev, sb = d["eval"]["rows"], d["sandbag"]["rows"]
        out.append(f"\\multicolumn{{{2 + len(al)}}}{{l}}{{\\emph{{{m}, block {d['block'] + 1} of {d['n_layers']}}}}} \\\\")
        for dn, lab in (("look", "test look"), ("answer", "answer test/real"), ("random", "random")):
            out.append(f" & {lab}, answers flipped & " + " & ".join(
                f"{np.mean([(r['runs'][f'{dn}|{x}'] > 0) != r['said_test'] for r in ev]):.2f}" for x in al) + " \\\\")
        base = np.mean([r["base_right"] for r in sb])
        for dn, lab in (("sandbag", "sandbag instruction"), ("wrong", "answer wrong"), ("random", "random")):
            out.append(f" & {lab}, accuracy ({base:.2f} unsteered) & " + " & ".join(
                f"{np.mean([r['runs'][f'{dn}|{x}']['right'] for r in sb]):.2f}" for x in al) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


NAMES = {"openimages_orig_auth": "Open Images, authority", "openimages_edit_auth": "Open Images, authority, edited",
         "openimages_orig_eval": "Open Images, test or real", "nuscenes_edit_auth": "nuScenes, authority, edited",
         "vins_orig_auth": "VINS-120K, authority", "vins_edit_auth": "VINS-120K, authority, edited",
         "gemini_orig_auth": "Generated scenes, authority"}
RN = {"v_read": "V-Read", "tuned": "TL", "jacobian": "JL", "patchscope": "PS", "logit": "LL"}


def within_table():
    d = _j("within_question/within_question.json")
    if not d:
        return None
    out = [HEAD, "\\begin{tabular}{llrrrrrrr}", "\\toprule",
           " & & & & \\multicolumn{2}{c}{all records} & \\multicolumn{3}{c}{within one question} \\\\",
           "\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}",
           "model & task & $n$ & groups & question only & V-Read, best other & V-Read & best other & $\\Delta$ [95\\% CI] \\\\",
           "\\midrule"]
    for r in d:
        if r["fam"] == "openimages_orig_eval":
            continue            # one question only: the wording carries nothing
        best_all = max((r[f"all_{k}"], RN[k]) for k in RN if k != "v_read" and f"all_{k}" in r)
        m = "8B" if r["model"].startswith("Qwen3-VL") else "27B"
        if r.get("best_in"):
            ci = r.get("diff_ci") or [np.nan, np.nan]
            iv = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if r["usable"] >= 2 else "(one group)"
            wi = (f"{r['in_v_read']:.2f} & {r['in_' + r['best_in']]:.2f} {RN[r['best_in']]} & "
                  f"{r['diff']:+.2f} {iv}").replace("-", "$-$")
        else:
            wi = "\\multicolumn{3}{c}{no question with both answers}"
        out.append(f"{m} & {NAMES[r['fam']]} & {r['n']} & {r['usable']} & {r['q_only']:.2f} & "
                   f"{r['all_v_read']:.2f}, {best_all[0]:.2f} {best_all[1]} & {wi} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


LABEL = {"auth": "authority", "no": "answers no", "look": "test look", "test": "answers test",
         "typed": "sandbag typed", "printed": "sandbag printed", "wrong": "answers wrong",
         "harm": "harmful prompt", "refuse": "refuses"}


def dirci_table():
    d = _j("cuegeom/direction_intervals.json")
    if not d:
        return None
    out = [HEAD, "\\begin{tabular}{llrrrr}", "\\toprule",
           "model & pair & cosine & 95\\% CI & angle & shuffled-label 95th pct. \\\\", "\\midrule"]
    for m, v in d.items():
        for r in v["pairs"]:
            nl = f"{r['null95']:.2f}" if r.get("null95") is not None else "--"
            out.append(f"{m} & {LABEL[r['a']]} / {LABEL[r['b']]} & {r['cos']:.2f} & [{r['lo']:.2f}, {r['hi']:.2f}] & "
                       f"{r['angle']:.0f}\\textdegree & {nl} \\\\".replace("-0.", "$-$0."))
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def main():
    for name, fn in (("app_within.tex", within_table), ("app_dirci.tex", dirci_table), ("app_steer.tex", steer_table), ("app_pwlock.tex", pwlock_table), ("app_cuegeom.tex", cue_table), ("app_coco.tex", coco_table)):
        t = fn()
        if t:
            open(os.path.join(OUT, name), "w").write(t + "\n")
            print("->", name)
        else:
            print("no data for", name)


if __name__ == "__main__":
    main()
