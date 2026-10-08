# SPDX-License-Identifier: Apache-2.0
"""Fetch the remaining audited sets the original experiments read.

Each set is tried against a few known mirrors and lands in the directory layout
the loaders in ``vlens/data`` expect. Everything goes to RAM, because the
container's disk has no room. A set that will not resolve is reported and
skipped rather than failing the run.

    python3 tools/fetch_more_safety.py --root /dev/shm/vlens_safety --budget 3.0
"""
from __future__ import annotations

import argparse
import os
import shutil

#: directory the loaders look in -> candidate dataset repos, best first
SETS = {
    "vlguard_alt/data": ["ys-zong/VLGuard", "tuananhbui89/VLGuard"],
    "vlsbench/data": ["Foreshhh/vlsbench", "OpenSafetyLab/VLSBench"],
    "mmvet/data": ["whyu/mm-vet", "lmms-lab/MM-Vet"],
    "spavl": ["sqrti/SPA-VL", "sqrti/SPA-VL-test"],
    "figstep/data": ["ThuCCSLab/FigStep", "hkust-nlp/SafeBench"],
    "siuo": ["sqrti/SIUO"],
}


def size_gb(p):
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(p) for f in fs) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/dev/shm/vlens_safety")
    ap.add_argument("--budget", type=float, default=3.0, help="gigabytes to spend in total")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    say = lambda *x: print(*x, flush=True)
    os.makedirs(a.root, exist_ok=True)
    start = size_gb(a.root)
    for sub, repos in SETS.items():
        dst = os.path.join(a.root, sub)
        if os.path.isdir(dst) and os.listdir(dst):
            say(f"  {sub}: already here")
            continue
        used = size_gb(a.root) - start
        if used > a.budget:
            say(f"  stopping: {used:.2f} GB spent of a {a.budget} GB budget")
            break
        for repo in repos:
            try:
                tmp = snapshot_download(repo, repo_type="dataset",
                                        allow_patterns=["*.parquet", "*.json", "*.csv",
                                                        "*.jsonl", "*.png", "*.jpg"])
                os.makedirs(dst, exist_ok=True)
                for r, _, fs in os.walk(tmp):
                    for f in fs:
                        rel = os.path.relpath(os.path.join(r, f), tmp)
                        out = os.path.join(dst, rel)
                        os.makedirs(os.path.dirname(out), exist_ok=True)
                        if not os.path.exists(out):
                            shutil.copy(os.path.join(r, f), out)
                say(f"  {sub}: from {repo}, {size_gb(dst):.2f} GB")
                break
            except Exception as e:  # noqa: BLE001
                say(f"  {sub}: {repo} -> {type(e).__name__}")
    say(f"{a.root}: {size_gb(a.root):.2f} GB total")


if __name__ == "__main__":
    main()
