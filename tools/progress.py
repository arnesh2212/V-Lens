# SPDX-License-Identifier: Apache-2.0
"""What has run, what is running, and what is still queued.

Reads the queue files and the record of finished lines that ``runq.sh`` keeps,
so the answer comes from the same two files the runner itself works from
rather than from a list kept by hand.

    python3 tools/progress.py            # on the box
    python3 tools/progress.py --results  # add the result file each one writes
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

SCRIPT = re.compile(r"(?:experiments|tools)/(?:[a-z]+/)?([a-z_0-9]+)\.(?:py|sh)")
REPO = re.compile(r"--repo\s+(\S+)")


def parse(line):
    s = SCRIPT.search(line)
    r = REPO.search(line)
    name = s.group(1) if s else line.split()[0]
    model = r.group(1).split("/")[-1] if r else ""
    extra = " ".join(w for w in (re.findall(r"--frac \S+|--adapter \S+|--stages [^-]*"
                                            r"|--rankings [^-]*|--out \S+", line)))
    return name, model, extra.strip()


def running():
    try:
        out = subprocess.run(["ps", "-eo", "etime,args"], capture_output=True, text=True).stdout
    except Exception:  # noqa: BLE001
        return []
    got = []
    for l in out.splitlines():
        if "python3 experiments" in l or "python3 tools" in l:
            if "grep" in l or "progress.py" in l:
                continue
            et = l.split()[0]
            got.append((et, *parse(l)))
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queues", default="queue/lane_*.q")
    ap.add_argument("--results", action="store_true")
    ap.add_argument("--pending", type=int, default=100)
    a = ap.parse_args()

    live = running()
    if live:
        print("running now")
        for et, name, model, extra in live:
            print(f"  {et:>12}  {name:<18}{model:<26}{extra[:60]}")
        print()

    total_done = total_left = 0
    for q in sorted(glob.glob(a.queues)):
        lines = [l.rstrip("\n") for l in open(q)
                 if l.strip() and not l.lstrip().startswith("#")]
        dl = q + ".done.lines"
        done = set()
        if os.path.exists(dl):
            done = {l.rstrip("\n") for l in open(dl)}
        att = {}
        if os.path.exists(q + ".attempts"):
            for l in open(q + ".attempts"):
                att[l.rstrip("\n")] = att.get(l.rstrip("\n"), 0) + 1
        left = [l for l in lines if l not in done]
        total_done += len(lines) - len(left); total_left += len(left)
        print(f"{os.path.basename(q)}: {len(lines) - len(left)} done, {len(left)} left")
        by_model = {}
        for l in left:
            name, model, extra = parse(l)
            by_model.setdefault(model or "-", []).append((name, extra, att.get(l, 0)))
        for model, xs in by_model.items():
            print(f"    {model or '-':<26}{len(xs):>3}  "
                  + ", ".join(dict.fromkeys(n for n, _, _ in xs)))
        for name, extra, tries in [x for xs in by_model.values() for x in xs][: a.pending]:
            if tries:
                print(f"      ! {name} has been attempted {tries} time(s) already")
        print()
    print(f"{total_done} done, {total_left} left across {len(glob.glob(a.queues))} queues")

    if a.results:
        print("\nresults on disk")
        for d in sorted(glob.glob("results/*")):
            n = len([f for f in glob.glob(f"{d}/*.json") if "rows" not in os.path.basename(f)])
            part = len(glob.glob(f"{d}/*rows.jsonl"))
            if n or part:
                print(f"  {os.path.basename(d):<18}{n:>3} finished"
                      + (f", {part} still running" if part else ""))


if __name__ == "__main__":
    sys.exit(main())
