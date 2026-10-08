# SPDX-License-Identifier: Apache-2.0
"""Redo a causal run's summary without the password study's own records.

A run made before the split was enforced drew from a pool that still held the
two families the lock is trained and tested on, and on the 27B those came to
more than half of what it measured. The per-record rows say which family each
one came from, so the arithmetic can be done again over the rest without
taking the card back.

The removed rows are not discarded: they are written under the password
study's own directory, which is the one place they belong. A per-source
breakdown is added at the same time, because the rows carry that too and the
original summary did not report it.

    python3 tools/resummarise_causal.py results/patchcausal/*__patchcausal.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from vlens.data import simonsays as SS  # noqa: E402

KS = (0.05, 0.10, 0.20, 0.30, 0.50)
GROUPS = ("top", "bottom", "middle", "random")


def summarise(rows, head, ks, main_key):
    def mean_of(fn, over=rows):
        v = [fn(x) for x in over]
        v = [x for x in v if x is not None and np.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")

    summ = dict(head, n_items=len(rows), ks=list(ks),
                curve={}, flip_rate={}, retention={}, agreement={}, overlap={})
    for g in GROUPS:
        summ["curve"][g] = [mean_of(lambda x, k=k, g=g: x["drop"].get(f"{main_key}|{k}|{g}"))
                            for k in ks]
        summ["flip_rate"][g] = [mean_of(lambda x, k=k, g=g:
                                        float(x["flip"].get(f"{main_key}|{k}|{g}", np.nan)))
                                for k in ks]
    meths = sorted({k.split("|")[0] for x in rows for k in x["drop"]})
    for meth in meths:
        if meth == main_key:
            continue
        vals = [mean_of(lambda x, k=k, mm=meth: x["drop"].get(f"{mm}|{k}|top")) for k in ks]
        if np.isfinite(vals).any():
            summ["curve"][f"top:{meth}"] = vals
    for meth in (main_key, "random"):
        if not any(meth in (x.get("retention") or {}) for x in rows):
            continue
        summ["retention"][meth] = [
            mean_of(lambda x, j=j, mm=meth: (x["retention"].get(mm) or [None] * len(ks))[j])
            for j in range(len(ks))]
    for k in sorted({k for x in rows for k in x["agree"]}):
        summ["agreement"][k] = mean_of(lambda x, k=k: x["agree"].get(k))
        summ["overlap"][k] = mean_of(lambda x, k=k: x["topk_overlap"].get(k))

    # the same headline, one source of pictures at a time
    summ["per_source"] = {}
    for src in sorted({str(x.get("family", "")).split("_")[0] for x in rows}):
        mine = [x for x in rows if str(x.get("family", "")).startswith(src)]
        if len(mine) < 8:
            continue
        summ["per_source"][src] = dict(
            n=len(mine),
            top=[mean_of(lambda x, k=k: x["drop"].get(f"{main_key}|{k}|top"), mine) for k in ks],
            random=[mean_of(lambda x, k=k: x["drop"].get(f"{main_key}|{k}|random"), mine)
                    for k in ks])
    summ["per_source_column"] = [f"{int(k * 100)}%" for k in ks]
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--locked-out", default="results/pwlock")
    a = ap.parse_args()
    for f in a.files:
        d = json.load(open(f))
        rows, old = d["rows"], d.get("summary", {})
        keep = [x for x in rows if not SS.is_password_locking(x)]
        drop = [x for x in rows if SS.is_password_locking(x)]
        if not keep:
            print(f"{f}: nothing left after the split; leaving it alone")
            continue
        ks = old.get("ks") or list(KS)
        main_key = next((k.split("|")[0] for k in rows[0]["drop"] if k.startswith("vwrite_")),
                        "vwrite_mean")
        head = {k: old[k] for k in ("repo", "block", "depth", "meta", "oriented") if k in old}
        d["summary"] = summarise(keep, head, ks, main_key)
        d["rows"] = keep
        if drop:
            os.makedirs(a.locked_out, exist_ok=True)
            name = os.path.basename(f).replace(".json", "_locked.json")
            json.dump(dict(rows=drop), open(os.path.join(a.locked_out, name), "w"), default=float)
        json.dump(d, open(f, "w"), default=float)
        was = old.get("curve", {}).get("top", [float("nan")] * len(ks))
        now = d["summary"]["curve"]["top"]
        j = min(2, len(ks) - 1)
        print(f"{os.path.basename(f)}: {len(rows)} -> {len(keep)} rows "
              f"({len(drop)} to the password study); "
              f"top at {int(ks[j] * 100)}% {was[j]:.3f} -> {now[j]:.3f}; "
              f"sources {', '.join(d['summary']['per_source'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
