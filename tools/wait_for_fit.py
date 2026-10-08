# SPDX-License-Identifier: Apache-2.0
"""Wait until the shared fit for a model holds the named instruments, then exit 0.

A second lane that adds an instrument to a shared entry must start only after
the first lane has saved it, or the two would each write their own copy. This
blocks until the entry the shared settings key to exists and carries every
instrument asked for, and gives up after ``--hours``.

    python3 tools/wait_for_fit.py --repo Qwen/Qwen3-VL-8B-Instruct --need v_read jacobian
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--need", nargs="+", required=True)
    ap.add_argument("--hours", type=float, default=10.0)
    a = ap.parse_args()
    from vlens.lens import fitall as FA
    t0 = time.time()
    while time.time() - t0 < a.hours * 3600:
        for s in glob.glob(os.path.join(FA.CACHE, "*.pt.spec")):
            try:
                spec = json.loads(open(s).read())
            except Exception:  # noqa: BLE001
                continue
            if spec[0] != a.repo or len(spec[2]) < 2 or spec[7] != FA.SHARED["tokens_per_step"]:
                continue
            got = FA._load(s[: -len(".spec")])
            if got is None:
                continue
            have = set(got[0])
            if all(k in have for k in a.need):
                print(f"{os.path.basename(s)} holds {sorted(have)}", flush=True)
                return
        time.sleep(120)
    print(f"no entry for {a.repo} with {a.need} after {a.hours} h", flush=True)
    sys.exit(4)


if __name__ == "__main__":
    main()
