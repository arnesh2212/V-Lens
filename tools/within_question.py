# SPDX-License-Identifier: Apache-2.0
"""Is a reader reading the picture, or the question's wording?

For every SimonSays task of Table 2, on exactly the records and blocks the table
scores, three numbers per reader:

* the ordinary AUROC for the answer the model gives (as in Table 2);
* the question-only baseline: each record scored by how often the model gave the
  positive answer to the *other* records with the identical question text
  (leave-one-out), i.e. what the wording alone predicts;
* the within-question AUROC: AUROC computed separately inside each group of
  records sharing an identical question and pooled over groups (weighted by the
  number of positive-negative pairs), so the wording cannot help at all.

The V-Read minus best-baseline difference of within-question AUROC gets a 95%
interval from resampling question groups.

    python3 tools/within_question.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
import make_vread_table as T  # noqa: E402

MODELS = ("Qwen3-VL-8B-Instruct", "Qwen3.5-27B")
FAMS = ("openimages_orig_auth", "openimages_edit_auth", "openimages_orig_eval", "nuscenes_edit_auth",
        "vins_orig_auth", "vins_edit_auth", "gemini_orig_auth")
READERS = ("v_read", "tuned", "jacobian", "patchscope", "logit")


def _auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return np.nan, 0
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean()), len(p) * len(n)


def _within(s, y, q, idx=None):
    groups = {}
    for i in (range(len(y)) if idx is None else idx):
        groups.setdefault(q[i], []).append(i)
    num = den = 0.0
    for ii in groups.values():
        a, w = _auc(s[ii], y[ii])
        if w:
            num += a * w; den += w
    return num / den if den else np.nan


def _question_only(y, q):
    s = np.zeros(len(y))
    for i in range(len(y)):
        o = [y[j] for j in range(len(y)) if j != i and q[j] == q[i]]
        s[i] = np.mean(o) if o else 0.5
    return _auc(s, y)[0]


def run(results="results"):
    table, _ = T.collect(results)
    # the Patchscopes prompt Table 2 uses, picked per model exactly as the table picks it
    ps_pick = {m: T.readout_cells(results, m)[3] for m in MODELS}
    out = []
    rng = np.random.default_rng(0)
    for m in MODELS:
        for fam in FAMS:
            res, loc = T._capture(T.D._family, results, m, fam)
            if res is None or fam not in table.get(m, {}):
                continue
            beh = {r["id"]: r for r in json.load(open(f"{results}/simonsays/{m}__{fam}.json"))["rows"]}
            q = np.array([str(beh[i]["question"]) for i in loc["ids"]])
            y = np.asarray(loc["y"], int)
            dep = np.asarray(res["depth"], float)
            a_of = {k: np.asarray(a, float) for k, a in res["auroc"].items()}
            vb = np.isfinite(a_of["v_read"])
            inb = (dep >= T.BAND[0]) & (dep <= T.BAND[1]) & vb
            series = dict(loc["series"])
            if ps_pick.get(m) and ps_pick[m] in series:
                series["patchscope"] = series[ps_pick[m]]
            cols = np.nonzero(inb)[0]
            R = [k for k in READERS if k in series]
            row = dict(model=m, fam=fam, n=int(len(y)), questions=int(len(set(q))),
                       usable=int(sum(1 for g in set(q) if 0 < y[q == g].sum() < (q == g).sum())),
                       q_only=_question_only(y, q))

            def per_block(k, idx=None, lab=None):
                """window mean over blocks of (all-record AUROC, within-question AUROC), as Table 2 averages."""
                ii = np.arange(len(y)) if idx is None else idx
                qq = q[ii] if lab is None else lab
                al, wi = [], []
                for j in cols:
                    sc = series[k][ii, j]
                    ok = np.isfinite(sc)
                    if ok.sum() < 2:
                        continue
                    al.append(_auc(sc[ok], y[ii][ok])[0]); wi.append(_within(sc[ok], y[ii][ok], qq[ok]))
                return (float(np.nanmean(al)) if al else np.nan, float(np.nanmean(wi)) if wi else np.nan)
            for k in R:
                row[f"all_{k}"], row[f"in_{k}"] = per_block(k)
            base = [k for k in R if k != "v_read"]
            wb = {k: row[f"in_{k}"] for k in base if np.isfinite(row[f"in_{k}"])}
            if wb and np.isfinite(row["in_v_read"]):
                bk = max(wb, key=wb.get)
                row["best_in"] = bk
                G = sorted(set(q)); gi = {g: np.where(q == g)[0] for g in G}
                d = []
                for _ in range(300):
                    pick = rng.integers(0, len(G), len(G))
                    idx = np.concatenate([gi[G[i]] for i in pick])
                    lab = np.concatenate([np.full(len(gi[G[i]]), r) for r, i in enumerate(pick)]).astype(str)
                    v = per_block("v_read", idx, lab)[1]
                    b = max(per_block(k, idx, lab)[1] for k in base)
                    d.append(v - b)
                d = np.array(d); d = d[np.isfinite(d)]
                row["diff"] = row["in_v_read"] - wb[bk]
                row["diff_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d.size else None
            out.append(row)
            print(m, fam, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                           if k not in ("model", "fam")}, flush=True)
    os.makedirs(f"{results}/within_question", exist_ok=True)
    json.dump(out, open(f"{results}/within_question/within_question.json", "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    run()
