# SPDX-License-Identifier: Apache-2.0
"""Take the hazardous-knowledge item bodies back out of the saved results.

The behaviour pass wrote every field of every record it scored, which for the
WMDP families means the question and its four choices. Those families are
carriers for a sandbagging test: what is under study is whether the model
answers and which letter it gives, never what was asked. The rows keep their
identifiers, their letters, their margins and the model's reply; the body and
the choices are replaced by a placeholder.

Idempotent, and it says what it changed rather than changing it quietly.

    python3 tools/redact_wmdp.py results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlens.data.simonsays import HELD_BACK, is_hazard  # noqa: E402

FIELDS = ("question", "choices")


def scrub(obj):
    """``(obj, n)`` with every hazardous record's body replaced, recursively."""
    n = 0
    if isinstance(obj, dict):
        if is_hazard(obj) and any(f in obj for f in FIELDS):
            for f in FIELDS:
                if f in obj and obj[f] != HELD_BACK:
                    obj[f] = HELD_BACK
                    n += 1
        for k, v in obj.items():
            obj[k], m = scrub(v)
            n += m
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i], m = scrub(v)
            n += m
    return obj, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=["results"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files = sorted({f for root in a.roots
                    for f in glob.glob(f"{root}/**/*.json", recursive=True)}
                   | {f for root in a.roots
                      for f in glob.glob(f"{root}/**/*.jsonl", recursive=True)})
    touched = total = 0
    for p in files:
        try:
            if p.endswith(".jsonl"):
                rows = [json.loads(l) for l in open(p) if l.strip()]
                rows, n = scrub(rows)
                if n and not a.dry_run:
                    with open(p, "w") as fh:
                        for r in rows:
                            fh.write(json.dumps(r, default=float) + "\n")
            else:
                d = json.load(open(p))
                d, n = scrub(d)
                if n and not a.dry_run:
                    json.dump(d, open(p, "w"), default=float)
        except Exception as e:  # noqa: BLE001
            print(f"  {p}: {type(e).__name__}"); continue
        if n:
            touched += 1; total += n
            print(f"  {'would clear' if a.dry_run else 'cleared'} {n:>4} field(s) in {p}")
    print(f"{total} field(s) across {touched} of {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
