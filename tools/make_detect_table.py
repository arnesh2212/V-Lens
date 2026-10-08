# SPDX-License-Identifier: Apache-2.0
"""The early-detection table: per task and reader, how deep the answer is first read for good.

Rows are the task rows of :func:`vlens.figures.fx_detect.task_rows` (refusal
targets, COCO, SimonSays by source, sandbagging by channel), one block per
model and a pooled block when more than one model has V-Read. Per reader:
the detection depth at AUROC 0.75, with the depth at 0.9 in brackets, and the
AUROC at half depth. Nothing is bold; V-Read's entry is underlined only where
it is strictly the best of the label-free readers. The table is written only
when some model has V-Read readouts (``--force`` writes it anyway, for checks).

    python3 tools/make_detect_table.py
    python3 tools/make_detect_table.py --results results --out tables/detect.tex
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from vlens.figures import fx_detect as D  # noqa: E402
from vlens.figures import paper as P  # noqa: E402

TEX_NICE = {"v_read": "V-Read", "v_read_coco": "V-Read, COCO fit", "tuned": "tuned lens",
            "jacobian": "Jacobian lens", "patchscope": "Patchscopes", "logit": "logit lens",
            "probe": "probe$^\\ast$", "dom": "diff.\\ of means$^\\ast$"}


def _d(v):
    if v is None:
        return "--"
    return "never" if not np.isfinite(v) else f"{v:.2f}".lstrip("0") if v < 1 else f"{v:.2f}"


def _a(v):
    return "--" if v is None else f"{v:.2f}".lstrip("0") if abs(v) < 1 else f"{v:.2f}"


def _best(row, k, which):
    """Whether reader ``k`` is strictly best of the label-free readers on this row."""
    if k != D.LEAD:
        return False
    src = row["det"] if which == "det" else row["mid"]
    v = src.get(k)
    if v is None or (which == "det" and not np.isfinite(v)):
        return False
    others = [src[o] for o in D.BASELINES if o in src and src[o] is not None]
    if not others:
        return False
    return all(v < o - 1e-9 for o in others) if which == "det" else all(v > o + 1e-9 for o in others)


def _label(row, tex=True):
    lab = row["label"]
    if row["group"] == "simonsays":
        probes = sorted({P.PROBE.get(f.split("_")[-1], f.split("_")[-1]) for f in row["fams"]})
        lab = f"{lab} ({', '.join(probes)})"
    if row["group"] == "refusal":
        lab = f"refusal: {lab}"
    if row.get("thin"):
        # every family of the row is under the scoring floor: shown, not counted (fx_detect.findings)
        lab = f"{lab} (under the scoring floor, not judged)"
    if tex:
        lab = lab.replace("&", "\\&")
    return lab


def blocks(results):
    out = []
    models = D._models(results)
    for m in models:
        rows = D.task_rows(results, m)
        if rows:
            out.append((m, rows))
    with_lead = [m for m, rows in out if any(D.LEAD in r["det"] for r in rows)]
    if len(with_lead) > 1:
        out.append(("pooled over " + ", ".join(with_lead), D.task_rows(results, None)))
    return out


def plain(bl):
    lines = []
    for name, rows in bl:
        readers = [k for k in D.READERS if any(k in r["det"] for r in rows)]
        lines.append(f"== {name}   (cell: depth at AUROC {D.THR} [at {D.THR_HI}] / AUROC at depth {D.MID};"
                     f" * = V-Read strictly best of the label-free readers; + = reader covers part of the row)")
        head = f"{'task':<52}{'n':>5}  " + "".join(f"{P.NICE[k][:22]:>26}" for k in readers)
        lines.append(head)
        for r in rows:
            cells = []
            for k in readers:
                if k not in r["det"]:
                    cells.append(f"{'--':>26}")
                    continue
                mark = ("*" if _best(r, k, "det") else "") + ("+" if r["cover"][k] < r["n"] else "")
                cells.append(f"{_d(r['det'][k]) + ' [' + _d(r['det_hi'][k]) + '] / ' + _a(r['mid'][k]) + mark:>26}")
            lines.append(f"{_label(r, tex=False)[:51]:<52}{r['n']:>5}  " + "".join(cells))
        f = D.findings(rows)
        lines.append("   " + D.findings_text(f, name))
    return "\n".join(lines)


def caption(bl):
    parts = []
    for name, rows in bl:
        f = D.findings(rows)
        if not f["with_lead"]:
            parts.append(f"{name}: no V-Read readouts.")
            continue
        s = (f"{name}: V-Read is present on {f['with_lead']} of {f['rows']} rows and is strictly earliest "
             f"of the published label-free readers at {D.THR} on {len(f['earliest'])}, level on {len(f['level'])} "
             f"and later on {len(f['beaten'])}")
        if f["beaten"]:
            s += " (" + "; ".join(f"{lab}: {k} at {_d(b)} against {_d(v)}" for lab, k, v, b in f["beaten"]) + ")"
        if f["gain"]:
            s += f"; its median lead over the earliest published baseline is {np.median(f['gain']):.2f} of depth"
        ag = D._against(rows)
        if ag:
            s += ("; rows compared against fewer baselines: " + "; ".join(
                f"{', '.join(labs)} against {D._names(have)} only" for have, labs in ag.items() if have))
        parts.append(s + ".")
    return (
        "MAIN. How deep in a vision-language model each reader first reads, for good, the answer the model is "
        "about to give, every reader read on the blocks V-Read is fitted at: V-Read, a label-free affine lens "
        "per block read through the model's own final norm and unembedding, is earliest on the rows counted "
        "below and level or later on the rest, which are listed. Compared: the logit lens, the tuned lens, the "
        "Jacobian lens and Patchscopes, where each was read, with a labelled logistic probe (and, for refusal, "
        "the difference of means) as a supervised reference, not an upper bound ($^\\ast$). Depths are point "
        "estimates with no interval. For each task and reader, $d_{.75}$ is the depth (block over the number of blocks "
        f"less one) from which the AUROC for the answer the model then gives stays at or above {D.THR} to the "
        f"output, $d_{{.9}}$ the same at {D.THR_HI}, and $A_{{.5}}$ the AUROC at depth {D.MID}. AUROC is on records no "
        "fitted lens saw, every reader on the same records within a family, linear between read blocks, "
        "averaged over a row's families weighted by records; ``never'' means the reader ends below the "
        "threshold, -- that the reader was not run; $^\\dagger$ marks a reader seen on part of a row's records "
        "(at least half). Refusal rows: MM-SafetyBench harmful and harmless scenarios with the benchmark's "
        "carrier, and MOSSBench; target the reply refusing, or the prompt being harmful. SimonSays rows pool "
        "a source's probes; the password-locking families are excluded. V-Read's entry is underlined only "
        "where it is strictly the best of the published label-free readers; nothing is bold. " + " ".join(parts))


def tex(bl):
    allk = [k for k in D.READERS if any(k in r["det"] for _n, rows in bl for r in rows)]
    cols = "l r " + " ".join("rr" for _ in allk)
    out = ["% MAIN. Written by tools/make_detect_table.py; do not edit by hand.", "\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
           "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{" + cols + "}", "\\toprule",
           " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{TEX_NICE[k]}}}" for k in allk) + " \\\\",
           " ".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(allk))),
           "task & $n$ & " + " & ".join("$d_{.75}$ ($d_{.9}$) & $A_{.5}$" for _ in allk)
           + " \\\\"]
    for bi, (name, rows) in enumerate(bl):
        out.append("\\midrule")
        out.append(f"\\multicolumn{{{2 + 2 * len(allk)}}}{{l}}{{\\emph{{{name}}}}} \\\\")
        prev = None
        for r in rows:
            if prev is not None and r["group"] != prev:
                out.append("\\addlinespace[2pt]")
            prev = r["group"]
            cells = []
            for k in allk:
                if k not in r["det"]:
                    cells += ["--", "--"]
                    continue
                dag = "$^\\dagger$" if r["cover"][k] < r["n"] else ""
                d = f"{_d(r['det'][k])} ({_d(r['det_hi'][k])})"
                a = _a(r["mid"][k])
                if _best(r, k, "det"):
                    d = f"\\underline{{{_d(r['det'][k])}}} ({_d(r['det_hi'][k])})"
                if _best(r, k, "mid"):
                    a = f"\\underline{{{a}}}"
                cells += [d + dag, a]
            out.append(f"{_label(r)} & {r['n']} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}}", f"\\caption{{{caption(bl)}}}", "\\label{tab:detect}",
            "\\end{table*}"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables/detect.tex")
    ap.add_argument("--force", action="store_true", help="write the table even with no V-Read readouts")
    a = ap.parse_args()
    bl = blocks(a.results)
    if not bl:
        print("no task rows")
        return 1
    print(plain(bl))
    if not a.force and not any(D.LEAD in r["det"] for _n, rows in bl for r in rows):
        print(f"no V-Read readouts yet; {a.out} not written")
        return 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    s = tex(bl)
    assert "\\textbf" not in s and "—" not in s
    open(a.out, "w").write(s)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
