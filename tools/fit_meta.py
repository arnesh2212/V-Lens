# SPDX-License-Identifier: Apache-2.0
"""What each cached lens fit cost: wall time, tokens and prompts, per lens.

Reads the meta stored with every fit in the map cache and writes one small
file per model, so the results table can state the cost of V-Read, the tuned
lens and the Jacobian lens from the fits the readouts actually used.

    python3 tools/fit_meta.py [--dir /dev/shm/vlens_maps] [--out results/fits]
"""
from __future__ import annotations

import argparse
import json
import os

import torch


def _plain(v):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if torch.is_tensor(v) and v.numel() == 1:
        return v.item()
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/dev/shm/vlens_maps")
    ap.add_argument("--out", default="results/fits")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    by_model: dict = {}
    for f in sorted(os.listdir(a.dir)):
        if not f.endswith(".pt") or not os.path.exists(os.path.join(a.dir, f + ".spec")):
            continue
        spec = json.loads(open(os.path.join(a.dir, f + ".spec")).read())
        model = str(spec[0]).split("/")[-1]
        d = torch.load(os.path.join(a.dir, f), map_location="cpu", weights_only=False)
        meta = {k: {kk: vv for kk, vv in (v or {}).items() if not torch.is_tensor(vv) or vv.numel() == 1}
                for k, v in (d.get("meta") or {}).items()}
        by_model.setdefault(model, []).append(dict(cache=f, spec=spec, meta=_plain(meta),
                                                   n_fit_ids=len(d.get("fit_ids") or [])))
        del d
    for model, fits in by_model.items():
        json.dump(fits, open(f"{a.out}/{model}__fitmeta.json", "w"), indent=1)
        for x in fits:
            got = {k: (v.get("seconds"), v.get("tokens") or v.get("tokens_seen") or v.get("prompts"))
                   for k, v in x["meta"].items()}
            print(model, x["cache"], got)


if __name__ == "__main__":
    main()
