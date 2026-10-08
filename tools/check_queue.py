# SPDX-License-Identifier: Apache-2.0
"""Parse every queued command's arguments without running any of it.

A misspelled flag costs a queue slot, three attempts and however long the
model takes to load before argparse gets a look at it. This imports each
script, lets it build its parser, hands it the arguments the queue line
actually carries, and stops there.

    python3 tools/check_queue.py queue/lane_*.q
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import shlex
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Parsed(Exception):
    def __init__(self, ns):
        self.ns = ns


def split(line: str):
    """``(script, args)`` from a queue line, or ``None`` for one to skip."""
    try:
        words = shlex.split(line)
    except ValueError:
        return None
    while words and "=" in words[0] and not words[0].startswith("-"):
        words = words[1:]                     # the environment it is given
    if not words or words[0] != "python3":
        return None                           # shell steps are not argparse
    return words[1], words[2:]


def check(script: str, args: list[str]):
    if not os.path.exists(script):
        return f"no such script: {script}"
    real = argparse.ArgumentParser.parse_args

    def fake(self, a=None, namespace=None):
        raise Parsed(real(self, args, namespace))

    argparse.ArgumentParser.parse_args = fake
    argv = sys.argv
    sys.argv = [script, *args]
    try:
        spec = importlib.util.spec_from_file_location("_q", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "main", None)
        if fn is None:
            return None                       # a script that runs at import time
        fn()
    except Parsed:
        return None
    except SystemExit as e:
        return f"argparse rejected it (exit {e.code})"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:110]}"
    finally:
        argparse.ArgumentParser.parse_args = real
        sys.argv = argv
    return "main() finished without parsing its arguments"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queues", nargs="*", default=["queue/lane_*.q"])
    a = ap.parse_args()
    files = sorted({f for pat in a.queues for f in glob.glob(pat)})
    bad = seen = skipped = 0
    for q in files:
        for line in open(q):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            got = split(line)
            if got is None:
                skipped += 1
                continue
            seen += 1
            why = check(*got)
            if why:
                bad += 1
                print(f"  {os.path.basename(q)}: {why}\n    {line[:150]}")
    print(f"{seen} commands parsed, {bad} rejected, {skipped} shell steps left alone")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
