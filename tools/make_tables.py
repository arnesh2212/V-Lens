# SPDX-License-Identifier: Apache-2.0
"""Turn the saved results into the tables the paper quotes, as CSV.

Every table carries what each instrument cost as well as what it scored. A
lens that reads a little better for two hours of gradient descent and one that
reads a little worse in a single closed-form solve are not the same offer, and
a table that prints only the score hides the difference.

    python3 tools/make_tables.py --results results --out tables
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

#: what each instrument needs to exist at all, which no timing shows
COSTS = {
    "v_read":       dict(labels="no",  gradients="no",  form="one closed-form solve"),
    "v_read_cross": dict(labels="no",  gradients="no",  form="one closed-form solve"),
    "v_write":      dict(labels="no",  gradients="yes", form="path integral per picture"),
    "attr_patch":   dict(labels="no",  gradients="yes", form="one gradient per picture"),
    "tuned":        dict(labels="no",  gradients="yes", form="250 steps of SGD"),
    "tuned_cross":  dict(labels="no",  gradients="yes", form="250 steps of SGD"),
    "jacobian":     dict(labels="no",  gradients="yes", form="mean Jacobian over prompts"),
    "latent":       dict(labels="no",  gradients="no",  form="nearest neighbours in a store"),
    "logit":        dict(labels="no",  gradients="no",  form="no transport at all"),
    "probe":        dict(labels="yes", gradients="no",  form="logistic regression"),
    "random":       dict(labels="no",  gradients="no",  form="a matched random direction"),
}


def _rows(path):
    try:
        return json.load(open(path))
    except Exception:  # noqa: BLE001
        return None


def _cost_of(meta, name):
    m = (meta or {}).get(name) or {}
    return dict(fit_seconds=m.get("seconds"),
                tokens=m.get("tokens") or m.get("tokens_spent"),
                steps=m.get("steps"), prompts=m.get("n_prompts"))


def reading(results, out):
    """Divergence to the model per block, with what each fit cost."""
    rows = []
    for p in sorted(glob.glob(f"{results}/lensbench/*__bench.json")):
        d = _rows(p)
        if not d or not d.get("kl"):
            continue
        model = os.path.basename(p).split("__")[0]
        meta, layers = d.get("meta") or {}, d.get("layers") or []
        for name, series in sorted(d["kl"].items()):
            c = _cost_of(meta, name)
            k = COSTS.get(name, {})
            rows.append(dict(model=model, instrument=name,
                             kl_mean=round(float(np.mean(series)), 4),
                             kl_first_block=round(float(series[0]), 4),
                             kl_last_block=round(float(series[-1]), 4),
                             blocks=len(layers), held_out_tokens=d.get("kl_tokens"),
                             fit_seconds=c["fit_seconds"], fit_tokens=c["tokens"],
                             needs_labels=k.get("labels"), needs_gradients=k.get("gradients"),
                             fitting=k.get("form")))
    return _write(rows, out, "reading.csv", sort=("model", "kl_mean"))


def patches(results, out):
    """Ranking image patches: the fall, the multiple of random, the agreement."""
    rows = []
    for p in sorted(glob.glob(f"{results}/patchlenses/*__patchlenses.json")):
        d = _rows(p)
        if not d:
            continue
        s, model = d["summary"], os.path.basename(p).split("__")[0]
        ks, meta = s["ks"], s.get("meta") or {}
        graded = s.get("graded_against_patching") or {}
        for name in s["rankings"]:
            key = f"zero:{name}"
            if key not in s.get("drop", {}):
                continue
            c = _cost_of(meta, name)
            k = COSTS.get(name, {})
            r = dict(model=model, instrument=name, records=s["n"], block=s.get("block"))
            for j, kk in enumerate(ks):
                r[f"drop_at_{int(kk * 100)}pc"] = round(float(s["drop"][key][j]), 4)
            rt = (s.get("ratio") or {}).get(key)
            r["times_random_at_50pc"] = round(float(rt[-1]), 3) if rt and np.isfinite(rt[-1]) else None
            r["rank_agreement"] = (round(float(graded[f"zero|{name}"]), 4)
                                   if graded.get(f"zero|{name}") is not None else None)
            r.update(fit_seconds=c["fit_seconds"], fit_tokens=c["tokens"],
                     needs_labels=k.get("labels"), needs_gradients=k.get("gradients"),
                     fitting=k.get("form"))
            rows.append(r)
    return _write(rows, out, "patches.csv", sort=("model", "rank_agreement"), reverse=True)


def sources(results, out):
    """The same patch ranking, split by which of the five datasets it came from."""
    rows = []
    for p in sorted(glob.glob(f"{results}/patchlenses/*__patchlenses.json")):
        d = _rows(p)
        if not d or not d["summary"].get("per_source"):
            continue
        s, model = d["summary"], os.path.basename(p).split("__")[0]
        ks, ns = s["ks"], s.get("per_source_n") or {}
        for src in sorted(s["per_source"]):
            for name, series in s["per_source"][src].items():
                r = dict(model=model, source=src, instrument=name,
                         records=ns.get(src), block=s.get("block"))
                for j, kk in enumerate(ks):
                    r[f"drop_at_{int(kk * 100)}pc"] = round(float(series[j]), 4)
                base = s["per_source"][src].get("random")
                v, b = series[len(ks) // 2], (base or [0])[len(ks) // 2]
                r["times_random_at_20pc"] = round(v / b, 2) if base and b > 0.05 else None
                k = COSTS.get(name, {})
                r.update(needs_labels=k.get("labels"), needs_gradients=k.get("gradients"),
                         fitting=k.get("form"))
                rows.append(r)
    return _write(rows, out, "sources.csv", sort=("model", "source", "instrument"))


def causal(results, out):
    """The causal check: what replacing each group of patches does."""
    rows = []
    for p in sorted(glob.glob(f"{results}/patchcausal/*__patchcausal.json")):
        d = _rows(p)
        if not d:
            continue
        s, model = d["summary"], os.path.basename(p).split("__")[0]
        ks = s["ks"]
        for g, series in sorted((s.get("curve") or {}).items()):
            if not isinstance(series, list) or len(series) != len(ks):
                continue
            r = dict(model=model, chosen=g, items=s.get("n_items"), block=s.get("block"))
            for j, kk in enumerate(ks):
                r[f"drop_at_{int(kk * 100)}pc"] = round(float(series[j]), 4)
            fl = (s.get("flip_rate") or {}).get(g)
            r["flip_at_50pc"] = round(float(fl[-1]), 4) if fl else None
            rows.append(r)
    return _write(rows, out, "causal.csv", sort=("model", "chosen"))


def efficiency(results, out):
    """What every instrument cost, gathered from wherever it was fitted."""
    seen = {}
    for p in sorted(glob.glob(f"{results}/*/*.json")):
        if "rows" in os.path.basename(p):
            continue
        d = _rows(p)
        if not d:
            continue
        s = d.get("summary") or d
        meta = s.get("meta") or d.get("meta") or {}
        model = os.path.basename(p).split("__")[0]
        where = os.path.basename(os.path.dirname(p))
        for name, m in meta.items():
            if not isinstance(m, dict) or m.get("seconds") is None:
                continue
            key = (model, name, where)
            k = COSTS.get(name, {})
            seen[key] = dict(model=model, instrument=name, experiment=where,
                             fit_seconds=m.get("seconds"),
                             fit_minutes=round(m["seconds"] / 60, 1),
                             tokens=m.get("tokens") or m.get("tokens_spent"),
                             steps=m.get("steps"), prompts=m.get("n_prompts"),
                             needs_labels=k.get("labels"), needs_gradients=k.get("gradients"),
                             fitting=k.get("form"))
    return _write(list(seen.values()), out, "efficiency.csv",
                  sort=("model", "fit_seconds"), reverse=True)


#: instruments whose readout is not settled yet, and the reason. The figures
#: leave these out; the tables keep the row, because a table is the record,
#: and carry the reason beside it so the number is not read as final.
PROVISIONAL = {"latent": "readout provisional: the store reaches 3.6% of the "
                         "vocabulary and the floor is being re-chosen"}


def _write(rows, out, name, sort=None, reverse=False):
    if not rows:
        return None
    if any("instrument" in r for r in rows):
        for r in rows:
            note = PROVISIONAL.get(str(r.get("instrument")))
            if note or any("note" in x for x in rows):
                r["note"] = note or ""
    if sort:
        def key(r):
            return tuple((r.get(k) is None, r.get(k)) for k in sort)
        rows = sorted(rows, key=key, reverse=reverse)
    cols, seen = [], set()
    for r in rows:
        for c in r:
            if c not in seen:
                seen.add(c); cols.append(c)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    made = [f(a.results, a.out) for f in (reading, patches, sources, causal, efficiency)]
    for p in made:
        if p:
            n = sum(1 for _ in open(p)) - 1
            print(f"  {os.path.basename(p):<18} {n} rows")
    if not any(made):
        print("  no results to tabulate yet")


if __name__ == "__main__":
    main()
