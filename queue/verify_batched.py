# SPDX-License-Identifier: Apache-2.0
"""Check that batched reading reproduces one-at-a-time reading exactly.

Runs the same prompts both ways and reports the largest disagreement in the
answer-spot state and in the first-token score.
"""
from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--square", type=int, default=448)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays
    from vlens.lens import readbatch as RB, scores
    from vlens.lens.hooks import ActivationRecorder

    m = vlens.load_vlm(a.repo)
    TS = scores.TokenSets(m.tokenizer)
    LAST = m.n_layers - 1
    GRID = sorted({int(round(x)) for x in np.linspace(max(m.n_deepstack, 0), LAST, 6)})
    recs = [r for r in simonsays.load(require_image=True) if r["kind"] == "yesno"][: a.n]
    items = [dict(question=r["question"], image=C.load_image(r["image"], square=a.square),
                  system=r["system"], cue=r["cue"]) for r in recs]
    fn = scores.score_fn(TS, "yesno", None)

    H_b, lg_b = RB.answer_states(m, items, GRID, batch=a.batch)
    s_b = np.array([float(fn(None, lg_b[i: i + 1]).float().cpu()[0]) for i in range(len(items))])

    H_s, s_s = [], []
    for it in items:
        b = m.chat(it["question"], [it["image"]], system=it["system"], cue=it["cue"])
        S = b["input_ids"].shape[1]
        with torch.no_grad(), ActivationRecorder(m.layers, at=[*GRID, LAST]) as rec:
            m.forward(b)
            H_s.append(np.stack([rec.activations[l][0, S - 1].float().cpu().numpy() for l in GRID]))
            lg = m.unembed(rec.activations[LAST][0, S - 1][None])
        s_s.append(float(fn(None, lg).float().cpu()[0]))
    H_s, s_s = np.stack(H_s), np.array(s_s)

    dh = float(np.abs(H_b - H_s).max())
    scale = float(np.abs(H_s).max())
    ds = float(np.abs(s_b - s_s).max())
    say(f"n={len(items)}  states: max |batched - solo| = {dh:.5f} "
        f"({100 * dh / max(scale, 1e-9):.4f}% of scale {scale:.1f})")
    say(f"           scores: max difference {ds:.6f}")
    say("BATCHED READING MATCHES" if dh / max(scale, 1e-9) < 0.02 and ds < 0.01
        else "BATCHED READING DIFFERS -- do not use")


if __name__ == "__main__":
    main()
