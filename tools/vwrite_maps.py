# SPDX-License-Identifier: Apache-2.0
"""V-Write's map, and attribution patching's beside it, for a chosen list of records.

The causal runs keep what covering the ranked patches does, not the maps
themselves. A figure that lays the map over the picture needs the map, so
this computes it for the records named in a list, at the same block and with
the same score as the causal runs, and stores it with the image-token grid
and the model's reply.

    python3 tools/vwrite_maps.py --repo Qwen/Qwen3-VL-8B-Instruct --ids results/maps/ids.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ids", required=True, help="JSON list of record ids")
    ap.add_argument("--out", default="results/maps")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--square", type=int, default=448)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays
    from vlens.lens import attribution as AT, readout, scores

    want = json.load(open(a.ids))
    recs = {r["id"]: r for r in simonsays.load(require_image=True)}
    m = vlens.load_vlm(a.repo)
    TS = scores.TokenSets(m.tokenizer)
    LAST = m.n_layers - 1
    L = max(int(round(a.frac * LAST)), 1)
    out = []
    for rid in want:
        r = recs.get(rid)
        if r is None or r["kind"] not in ("yesno", "letter", "test_real"):
            say(f"  {rid}: not a short-answer record here, skipped"); continue
        img = C.load_image(r["image"], square=a.square)
        b = m.chat(r["question"], [img], system=r["system"], cue=r["cue"])
        pos = readout.image_positions(m, b)
        fn = scores.score_fn(TS, r["kind"], r.get("expected"))
        with torch.no_grad():
            _, H, _, lg0 = AT.states_at(m, b, [L])
        h = H[L]
        s0 = float(fn(None, lg0).float().cpu()[0])
        fn, s0 = scores.orient(fn, r["kind"], s0)
        mu = h.mean(0, keepdim=True).expand_as(h)
        vw = AT.path_reading(m, b, fn, L, pos, h, mu, steps=a.steps)
        ap_ = AT.one_point(m, b, fn, L, pos, h, mu)
        # the model's own reply, greedy, so every mapped record carries its answer
        with torch.no_grad():
            gen = m.hf.generate(**{k: v for k, v in b.items() if not k.startswith("_")},
                                max_new_tokens=24, do_sample=False,
                                pad_token_id=m.tokenizer.pad_token_id or m.tokenizer.eos_token_id)
        reply = m.tokenizer.decode(gen[0, b["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        side = int(round(np.sqrt(len(pos))))
        out.append(dict(id=rid, family=r["family"], image=os.path.relpath(r["image"], simonsays.ROOT),
                        question=r["question"], cue=r.get("cue"), kind=r["kind"],
                        expected=r.get("expected"), reply=reply, s0=s0, block=L, grid=[side, len(pos) // max(side, 1)],
                        v_write=np.asarray(vw, float).round(5).tolist(),
                        attr_patch=np.asarray(ap_, float).round(5).tolist()))
        say(f"  {rid}: {len(pos)} patches, score {s0:.2f}")
    os.makedirs(a.out, exist_ok=True)
    p = f"{a.out}/{a.repo.split('/')[-1]}__maps.json"
    json.dump(dict(repo=a.repo, block=L, rows=out), open(p, "w"))
    say(f"wrote {len(out)} maps -> {p}")
    say("DONE vwrite_maps")


if __name__ == "__main__":
    main()
