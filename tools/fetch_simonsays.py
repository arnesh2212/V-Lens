# SPDX-License-Identifier: Apache-2.0
"""Download SimonSays from the Hugging Face Hub and rebuild the release folder the experiments read.

The Hub copy is one parquet row per prompt record with its picture embedded.
Each row keeps the original record (``record``), the prompt file it belongs to
(``source_file``), its position in that file (``source_index``) and the
picture's path in the release (``image_path``). This writes every picture once
and every prompt file with its records in their original order, so per-family
caps and seeded draws select the same records as in the paper.

    python3 tools/fetch_simonsays.py                       # -> data/simonsays
    python3 tools/fetch_simonsays.py --out /path/to/simonsays --repo <user>/SimonSays

A private dataset needs ``hf auth login`` (or ``HF_TOKEN``) first.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_REPO = os.environ.get("SIMONSAYS_REPO", "Ironman2212/SimonSays")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO, help="dataset repo on the Hub")
    ap.add_argument("--revision", default=None, help="a branch, tag or commit of the dataset")
    ap.add_argument("--out", default=os.environ.get("VLENS_DATA") or "data/simonsays")
    ap.add_argument("--cache", default=None, help="where the parquet files are downloaded (Hub cache by default)")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    snap = snapshot_download(a.repo, repo_type="dataset", revision=a.revision, cache_dir=a.cache,
                             allow_patterns=["data/**", "metadata/**"])
    files = json.load(open(os.path.join(snap, "metadata", "source_files.json"), encoding="utf-8"))
    shards = sorted(glob.glob(os.path.join(snap, "data", "**", "*.parquet"), recursive=True))
    say(f"{len(shards)} parquet files from {a.repo}")

    samples = defaultdict(dict)
    written = 0
    cols = ["image", "image_path", "source_file", "source_index", "record"]
    for sh in shards:
        t = pq.read_table(sh, columns=cols).to_pylist()
        for r in t:
            dst = os.path.join(a.out, r["image_path"])
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "wb") as f:
                    f.write(r["image"]["bytes"])
                written += 1
            samples[r["source_file"]][int(r["source_index"])] = json.loads(r["record"])
        say(f"  {os.path.relpath(sh, snap)}: {len(t)} records")

    for rel, info in files.items():
        got = samples.get(rel, {})
        if len(got) != info["count"] or sorted(got) != list(range(info["count"])):
            raise SystemExit(f"{rel}: expected {info['count']} records, found {len(got)}")
        doc = dict(schema_version=info["schema_version"], dataset_meta=info["dataset_meta"],
                   samples=[got[i] for i in range(info["count"])])
        p = os.path.join(a.out, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
    say(f"wrote {written} pictures and {len(files)} prompt files to {a.out}")
    say(f"set VLENS_DATA={os.path.abspath(a.out)} if that is not the default data/simonsays")


if __name__ == "__main__":
    main()
