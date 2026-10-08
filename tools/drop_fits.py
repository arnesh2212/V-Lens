# SPDX-License-Identifier: Apache-2.0
"""Drop the cached fits belonging to one model, and leave the rest alone.

Both lanes share the cache directory, so one lane rotating its model must not
cost the other lane the fit it is in the middle of using. Each entry carries a
description of what it is, and the first thing in that description is the
model, which is what this matches on. An entry with no description is left
alone rather than guessed at.

    python3 tools/drop_fits.py Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--dir", default="/dev/shm/vlens_maps")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not os.path.isdir(a.dir):
        return 0
    short = a.repo.split("/")[-1]
    freed = 0
    for f in sorted(os.listdir(a.dir)):
        if not f.endswith(".pt.spec"):
            continue
        try:
            who = json.loads(open(os.path.join(a.dir, f)).read())[0]
        except Exception:  # noqa: BLE001
            continue
        if who != a.repo and str(who).split("/")[-1] != short:
            continue
        for p in (os.path.join(a.dir, f[: -len(".spec")]), os.path.join(a.dir, f)):
            try:
                freed += os.path.getsize(p)
                if not a.dry_run:
                    os.remove(p)
            except OSError:
                pass
        print(f"  {'would drop' if a.dry_run else 'dropped'} the fit cached at "
              f"{f[: -len('.spec')]}")
    print(f"  {freed / 2 ** 30:.2f} GiB of cached fits for {short}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
