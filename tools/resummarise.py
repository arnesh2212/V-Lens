# SPDX-License-Identifier: Apache-2.0
"""Recompute a patch-ranking summary from the rows a finished run saved.

The per-record rows are what the run actually measured; the summary is
arithmetic on top of them.  When that arithmetic changes there is no need to
take the card back.

    python3 tools/resummarise.py results/patchlenses/*__patchlenses.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from vlens.data import simonsays as SS  # noqa: E402
from vlens.lens import patchsumm as PS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--locked-out", default="results/pwlock",
                    help="where the password study's own rows are moved to")
    ap.add_argument("--mark-oriented", action="store_true",
                    help="stamp the run as having read toward the answer the model gave, "
                         "for files written before that stamp existed")
    a = ap.parse_args()
    for f in a.files:
        d = json.load(open(f))
        rows, old = d["rows"], d.get("summary", {})
        ks = old.get("ks") or sorted({float(k.split("|")[-1]) for k in rows[0]["after"]})
        rankings = old.get("rankings") or sorted({k.split("|")[1] for k in rows[0]["after"]})
        head = {k: old[k] for k in ("repo", "block", "depth", "meta", "oriented") if k in old}
        if a.mark_oriented:
            head["oriented"] = True
        summ = PS.summarise(rows, ks, rankings, **head)
        d["summary"] = summ
        drop = [x for x in rows if SS.is_password_locking(x)]
        if drop:
            os.makedirs(a.locked_out, exist_ok=True)
            json.dump(dict(rows=drop),
                      open(os.path.join(a.locked_out,
                                        os.path.basename(f).replace(".json", "_locked.json")), "w"),
                      default=float)
            d["rows"] = [x for x in rows if not SS.is_password_locking(x)]
            print(f"  {len(drop)} row(s) moved to the password study")
        json.dump(d, open(f, "w"), default=float)
        was = old.get("per_family") or {}
        n_nan = sum(1 for v in was.values() for x in v.values() if not np.isfinite(x))
        n_tot = sum(len(v) for v in was.values())
        print(f"{os.path.basename(f)}: {len(rows)} rows, {len(summ['per_family'])} families; "
              f"per-family cells not finite before {n_nan}/{n_tot}, after "
              f"{sum(1 for v in summ['per_family'].values() for x in v.values() if not np.isfinite(x))}"
              f"/{sum(len(v) for v in summ['per_family'].values())}")
        print("  sources: " + ", ".join(f"{k} {v}" for k, v in
                                        sorted((summ.get("per_source_n") or {}).items())))
        if not a.quiet:
            PS.report(summ, ks, rankings)
            PS.report_sources(summ, rankings)


if __name__ == "__main__":
    main()
