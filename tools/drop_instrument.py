# SPDX-License-Identifier: Apache-2.0
"""Take an instrument out of the results that are already on disk.

Used to remove the low-rank lens, which was ours rather than anyone's
published baseline and had no business sitting in a baseline table. The rows
each result was measured from are left alone; what goes is every key that
names the instrument, wherever it appears in a summary.

    python3 tools/drop_instrument.py anchor results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def scrub(obj, name: str):
    """``(obj, n)`` with every key naming ``name`` gone, at any depth."""
    n = 0
    if isinstance(obj, dict):
        for k in [k for k in obj
                  if k == name or (isinstance(k, str)
                                   and name in k.replace(":", "|").split("|"))]:
            del obj[k]
            n += 1
        for k, v in obj.items():
            obj[k], m = scrub(v, name)
            n += m
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i], m = scrub(v, name)
            n += m
        # a bare list of instrument names, such as a "rankings" field
        if obj and all(isinstance(v, str) for v in obj) and name in obj:
            n += obj.count(name)
            obj = [v for v in obj if v != name]
    return obj, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("roots", nargs="*", default=["results"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files = sorted({f for r in a.roots for f in glob.glob(f"{r}/**/*.json", recursive=True)})
    hit = total = 0
    for p in files:
        if "rows" in os.path.basename(p):
            continue
        try:
            d = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        d, n = scrub(d, a.name)
        if not n:
            continue
        hit += 1; total += n
        if not a.dry_run:
            json.dump(d, open(p, "w"), default=float)
        print(f"  {'would drop' if a.dry_run else 'dropped'} {n:>3} key(s) from {p}")
    print(f"{total} key(s) naming {a.name!r} across {hit} of {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
