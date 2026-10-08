# SPDX-License-Identifier: Apache-2.0
"""The V-Read table with the development-chosen variant as V-Read, the original settings beside it.

``tools/vread_variants.py`` chooses a V-Read variant on development prompts by
answer-spot KL (no labels, no scored records) and writes its readouts as reader
``v_read_v2``. This scores it exactly as ``tools/make_vread_table.py`` scores
V-Read -- the same records, the same blocks, the same depth window, the same
paired picture bootstrap with the best baseline retaken inside each resample --
by running that script's own cell code with the lead reader switched, and adds,
from the same resamples:

* the original V-Read's margin over the best baseline and its result;
* the variant minus the original, with its paired 95% interval.

Refusal cells store no per-record readings and have no variant readout, so
they are left out. Models without ``v_read_v2`` readouts are left out.

    python3 tools/vread_v2_table.py                 # plain text, and tables/vread_v2.tex
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
import numpy as np  # noqa: E402

import make_vread_table as T  # noqa: E402
from vlens.figures import fx_bench as FB  # noqa: E402

V2, ORIG = "v_read_v2", "v_read"
DIFF, OMARGIN = "__v2_minus_orig", "__orig_margin"


def _patch():
    if V2 not in FB.READERS:
        FB.READERS = (*FB.READERS, V2)
    T.LEAD = V2
    T.OURS = (V2, ORIG, "v_read_coco")
    base_window = T._window

    def window(boot, a_of, inb, base):
        mci, per = base_window(boot, a_of, inb, base)

        def wmean(k):
            if k not in boot or k not in a_of:
                return None
            cols = inb & np.isfinite(a_of[k])
            if not cols.any():
                return None
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.nanmean(boot[k][:, cols], axis=1)

        def ci(d):
            d = d[np.isfinite(d)]
            return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if d.size >= 0.9 * T.REPS else None
        a, o = wmean(V2), wmean(ORIG)
        if a is not None and o is not None:
            per[DIFF] = ci(a - o)
        bs = [wmean(k) for k in base]
        bs = [b for b in bs if b is not None]
        if o is not None and bs:
            per[OMARGIN] = ci(o - np.max(np.vstack(bs), axis=0))
        return mci, per
    T._window = window


def collect(results):
    _patch()
    table, notes = T.collect(results)
    out = {}
    for m, cells in table.items():
        for f, c in cells.items():
            if c.get("src") != "readouts" or V2 not in c["band"] or ORIG not in c["band"]:
                continue
            row = dict(n=c["n"], pictures=c["pictures"], best=c["best"], windows={})
            for w, wd in c["windows"].items():
                b = wd["band"]
                if V2 not in b or ORIG not in b or not wd.get("best"):
                    continue
                bb = wd["best"]
                row["windows"][w] = dict(
                    v2=b[V2], orig=b[ORIG], base=b[bb], best=bb,
                    v2_ci=wd["ci"].get(V2), orig_ci=wd["ci"].get(ORIG),
                    v2_margin=b[V2] - b[bb], v2_margin_ci=wd["margin_ci"], v2_result=T.classify(wd["margin_ci"]),
                    orig_margin=b[ORIG] - b[bb], orig_margin_ci=wd["ci"].get(OMARGIN),
                    orig_result=T.classify(wd["ci"].get(OMARGIN)),
                    diff=b[V2] - b[ORIG], diff_ci=wd["ci"].get(DIFF),
                    diff_result=T.classify(wd["ci"].get(DIFF)))
            if row["windows"]:
                out.setdefault(m, {})[f] = row
    return out, notes


def counts(tab, w):
    got = {}
    for m, cells in tab.items():
        c = {"v2": {r: 0 for r in T.RESULT[:3]}, "orig": {r: 0 for r in T.RESULT[:3]},
             "diff": {r: 0 for r in T.RESULT[:3]}}
        for row in cells.values():
            x = row["windows"].get(w)
            if not x:
                continue
            for k, key in (("v2", "v2_result"), ("orig", "orig_result"), ("diff", "diff_result")):
                if x[key] in c[k]:
                    c[k][x[key]] += 1
        got[m] = c
    return got


def picks():
    out = {}
    for p in sorted(glob.glob(os.path.join(HERE, "results", "vread_variants", "*__pick.json"))):
        d = json.load(open(p))
        out[d["model"]] = d
    return out


def _c(ci, sign=False):
    if not ci:
        return ""
    f = (lambda v: f"{v:+.2f}") if sign else (lambda v: f"{v:.2f}")
    return f"[{f(ci[0])}, {f(ci[1])}]"


def plain(tab, pk):
    L = []
    w = tuple(T.BAND)
    for m, cells in tab.items():
        p = pk.get(m, {})
        L.append(f"\n{m}: variant '{p.get('chosen')}' {p.get('cfg')} picked {p.get('when')} "
                 f"(depth window {w[0]:g}-{w[1]:g})")
        L.append(f"  {'task':<46}{'n':>5} {'V-Read v2':>20} {'V-Read orig':>20} {'best base':>22} "
                 f"{'v2 margin':>22} {'res':>6} {'orig margin':>22} {'res':>6} {'v2-orig':>22}")
        for f in sorted(cells, key=lambda f: -(cells[f]["windows"].get(w, {}).get("v2_margin") or -9)):
            x = cells[f]["windows"].get(w)
            if not x:
                continue
            L.append(f"  {T.label_of(f)[:44]:<46}{cells[f]['n']:>5} "
                     f"{x['v2']:.2f} {_c(x['v2_ci']):>15} {x['orig']:.2f} {_c(x['orig_ci']):>15} "
                     f"{x['base']:.2f} {T.ABBR[x['best']]:<3} "
                     f"{x['v2_margin']:+.2f} {_c(x['v2_margin_ci'], True):>16} {x['v2_result']:>6} "
                     f"{x['orig_margin']:+.2f} {_c(x['orig_margin_ci'], True):>16} {x['orig_result']:>6} "
                     f"{x['diff']:+.3f} {_c(x['diff_ci'], True):>16}")
    L.append("")
    for w in T.WINDOWS:
        C = counts(tab, w)
        L.append(f"window {T._wname(w)}: " + "; ".join(
            f"{m}: v2 {c['v2']['win']}/{c['v2']['level']}/{c['v2']['behind']}, "
            f"original {c['orig']['win']}/{c['orig']['level']}/{c['orig']['behind']} (win/level/behind); "
            f"v2 vs original: above {c['diff']['win']}, level {c['diff']['level']}, below {c['diff']['behind']}"
            for m, c in C.items()))
    return "\n".join(L)


def tex(tab, pk):
    w = tuple(T.BAND)
    C = counts(tab, w)
    cnt = "; ".join(f"{T.SHORT[m]}: {c['v2']['win']} win, {c['v2']['level']} level, {c['v2']['behind']} behind "
                    f"(original settings: {c['orig']['win']}, {c['orig']['level']}, {c['orig']['behind']}); "
                    f"the variant above the original in {c['diff']['win']} cells, level in {c['diff']['level']}, "
                    f"below in {c['diff']['behind']}" for m, c in C.items())
    chosen = "; ".join(f"{T.SHORT[m]}: {p['chosen']} ({', '.join(f'{k}={v}' for k, v in (p.get('cfg') or {}).items())}"
                       f"{', depth-smoothed' if p.get('smooth') else ''}), fixed {p['when'].replace('T', ' ')}"
                       for m, p in pk.items() if m in tab)
    cap = (
        "APPENDIX. V-Read with its fitting choices made on development data, against V-Read at its original "
        "settings. The variant is chosen without labels and without the scored records: its data weighting "
        "(dropping the first positions, the RMS-normalised target, up-weighting the last positions), its "
        "penalty, shrinkage target and gain are chosen on the 120 development prompts of the shared fit by the "
        "KL divergence from the model's own final-block distribution at the answer spot, and the choice between "
        "the original and each stage's winner on the 120 further held-back prompts of that fit by the same "
        "criterion; the choice was written to disk before any scored record was read (" + chosen + "). "
        f"Cells as in Table~\\ref{{tab:vread_main}}: mean AUROC over V-Read's fitted blocks in depth "
        f"{w[0]:g}--{w[1]:g}, the same records and blocks for every reader, margins over the best label-free "
        f"baseline with paired 95\\% intervals (pictures resampled {T.REPS} times, the best baseline retaken inside "
        "each resample), and the variant minus the original with its paired interval from the same resamples. "
        "Refusal cells have no per-record readings and are omitted. Counts: " + cnt + ".")
    L = ["% generated by tools/vread_v2_table.py; do not edit", "\\begin{table*}[t]", "\\centering",
         f"\\caption{{{cap}}}", "\\label{tab:vread_v2}", "\\smallskip", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{lrllllllll}", "\\toprule",
         "task & $n$ & V-Read (dev-chosen) & V-Read (original settings) & best baseline & margin & result & "
         "original margin & original result & variant $-$ original \\\\"]
    for m, cells in tab.items():
        L += ["\\midrule", f"\\multicolumn{{10}}{{l}}{{\\emph{{{T.SHORT[m]}}}}} \\\\"]
        for f in sorted(cells, key=lambda f: -(cells[f]["windows"].get(w, {}).get("v2_margin") or -9)):
            x = cells[f]["windows"].get(w)
            if not x:
                continue
            ci = lambda c, s=False: f"\\,{{\\tiny {_c(c, s)}}}" if c else ""
            L.append(" & ".join([
                T.label_of(f).replace("&", "\\&"), str(cells[f]["n"]),
                f"{x['v2']:.2f}{ci(x['v2_ci'])}", f"{x['orig']:.2f}{ci(x['orig_ci'])}",
                f"{x['base']:.2f}\\,{{\\tiny {T.ABBR[x['best']]}}}",
                f"{x['v2_margin']:+.2f}{ci(x['v2_margin_ci'], True)}", x["v2_result"],
                f"{x['orig_margin']:+.2f}{ci(x['orig_margin_ci'], True)}", x["orig_result"],
                f"{x['diff']:+.3f}{ci(x['diff_ci'], True)}"]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables/vread_v2.tex")
    ap.add_argument("--json", default="results/vread_variants/table_v2.json")
    a = ap.parse_args()
    tab, _notes = collect(a.results)
    if not tab:
        print("no cells carry v_read_v2")
        return 1
    pk = picks()
    print(plain(tab, pk))
    s = tex(tab, pk)
    assert "—" not in s and "---" not in s
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(s)
    js = {m: {f: dict(r, windows={f"{w[0]:g}-{w[1]:g}": v for w, v in r["windows"].items()})
              for f, r in cells.items()} for m, cells in tab.items()}
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(dict(counts={f"{w[0]:g}-{w[1]:g}": counts(tab, w) for w in T.WINDOWS}, cells=js,
                   picks={m: {k: p.get(k) for k in ("when", "chosen", "cfg", "smooth", "candidates")}
                          for m, p in pk.items()}),
              open(a.json, "w"), indent=1, default=float)
    print(f"-> {a.out}\n-> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
