# SPDX-License-Identifier: Apache-2.0
"""Two more ways to score a reader: does its call match the answer, and are its odds honest.

On the same records and blocks as the results table (every reader on
V-Read's fitted blocks inside the depth window), for each reader:

* top-1 agreement: the share of (record, block) readings whose sign, read as
  a call between the two answers, matches the answer the model then gives;
* expected calibration error: the reading taken as log-odds, turned into a
  probability, binned in ten, against how often that answer is the one given.

Pooled over tasks, weighted by records. Writes tables/paper_agreement.tex and
results/agreement/<model>.json.

    python3 tools/agreement.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import make_vread_table as T  # noqa: E402

READERS = [("v_read", "V-Read"), ("tuned", "tuned lens"), ("jacobian", "Jacobian lens"),
           ("patchscope", "Patchscopes"), ("logit", "logit lens")]
MODELS = ("Qwen3-VL-8B-Instruct", "Qwen3.5-27B", "gemma-4-26B-A4B-it", "InternVL3_5-8B-HF")


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        s = idx == b
        if s.any():
            e += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(e)


def model_scores(results, model):
    acc = {k: dict(agree=[], ece=[], w=[]) for k, _ in READERS}
    for fam in T._readout_fams(results, model):
        res, loc = T._capture(T.D._family, results, model, fam)
        if res is None or "v_read" not in res["auroc"]:
            continue
        dep = np.asarray(res["depth"], float)
        inb = (dep >= T.BAND[0]) & (dep <= T.BAND[1]) & np.isfinite(np.asarray(res["auroc"]["v_read"], float))
        y = np.asarray(loc["y"], float)
        series = dict(loc["series"])
        best = next((k for k in series if k.startswith("patchscope")), None)
        if best and "patchscope" not in series:
            series["patchscope"] = series[best]
        for k, _ in READERS:
            if k not in series:
                continue
            S = np.asarray(series[k], float)[:, inb]
            ok = np.isfinite(S)
            if not ok.any():
                continue
            Y = np.broadcast_to(y[:, None], S.shape)[ok]
            s = S[ok]
            acc[k]["agree"].append(float(((s > 0) == (Y > 0.5)).mean()))
            acc[k]["ece"].append(ece(1 / (1 + np.exp(-np.clip(s, -30, 30))), Y))
            acc[k]["w"].append(len(y))
    out = {}
    for k, v in acc.items():
        if v["w"]:
            w = np.asarray(v["w"], float)
            out[k] = dict(agree=float(np.average(v["agree"], weights=w)),
                          ece=float(np.average(v["ece"], weights=w)), tasks=len(w))
    return out


def main():
    os.makedirs("results/agreement", exist_ok=True)
    got = {}
    for m in MODELS:
        s = model_scores("results", m)
        if s:
            got[m] = s
            json.dump(s, open(f"results/agreement/{m}.json", "w"), indent=1)
            print(m, {k: (round(v["agree"], 3), round(v["ece"], 3)) for k, v in s.items()})
    ms = list(got)
    head = " & ".join(["reader"] + [f"\\multicolumn{{2}}{{c}}{{{m}}}" for m in ms])
    rules = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(ms)))
    sub = " & ".join([""] + ["agreement", "ECE"] * len(ms))
    lines = ["\\begin{tabular}{l" + "cc" * len(ms) + "}", "\\toprule", head + " \\\\", rules, sub + " \\\\",
             "\\midrule"]
    for k, lab in READERS:
        row = [lab]
        for m in ms:
            v = got[m].get(k)
            row += ([f"{v['agree']:.2f}", f"{v['ece']:.2f}"] if v else ["\\textit{running}"] * 2)
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open("tables/paper_agreement.tex", "w").write(
        "% Written by tools/agreement.py from results/; do not edit by hand.\n" + "\n".join(lines) + "\n")
    print("-> tables/paper_agreement.tex")


if __name__ == "__main__":
    main()
