# SPDX-License-Identifier: Apache-2.0
"""Does batching change any number we report?

Right-padded reading and left-padded generation are only sound if a row's
result does not depend on what else is in its batch.  This runs the same
records one at a time and in a batch and reports the largest disagreement in
the answer-spot state, the first-token margin and the greedy reply.

    python3 tools/verify_batching.py --repo Qwen/Qwen3.5-27B --n 8
"""
from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--family", default="nuscenes_edit_auth")
    ap.add_argument("--square", type=int, default=448)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus
    from vlens.data import simonsays
    from vlens.lens import scores
    from vlens.lens.hooks import ActivationRecorder

    rows = simonsays.load(families=[a.family])[: a.n]
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    LAST = m.n_layers - 1
    items = [dict(question=r["question"], image=corpus.load_image(r["image"], square=a.square),
                  system=r["system"], cue=r["cue"]) for r in rows]

    # ---- one at a time
    solo_h, solo_m, solo_r = [], [], []
    for r, it in zip(rows, items):
        b = m.chat(it["question"], [it["image"]], system=it["system"], cue=it["cue"])
        with torch.no_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec:
            m.forward(b)
            h = rec.activations[LAST][0, -1]
        solo_h.append(h.float().cpu())
        solo_m.append(scores.margins(TS, tok, m.unembed(h[None])[0], r["kind"]))
        with torch.no_grad():
            g = m.hf.generate(**b, max_new_tokens=12, do_sample=False,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
        solo_r.append(tok.decode(g[0, b["input_ids"].shape[1]:], skip_special_tokens=True).strip())

    # ---- batched, right padding for the read
    b = m.batch_many(items, side="right")
    spot = m.answer_spot(b)
    with torch.no_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec:
        m.forward(b)
        H = rec.activations[LAST][torch.arange(len(items)), spot.to(rec.activations[LAST].device)]
    lg = m.unembed(H.to(m.device))
    bat_m = [scores.margins(TS, tok, lg[j], rows[j]["kind"]) for j in range(len(rows))]

    # ---- batched, left padding for the reply
    bg = m.batch_many(items, side="left")
    with torch.no_grad():
        gen = m.hf.generate(**{k: v for k, v in bg.items() if not k.startswith("_")},
                            max_new_tokens=12, do_sample=False,
                            pad_token_id=tok.pad_token_id or tok.eos_token_id)
    n_in = bg["input_ids"].shape[1]
    bat_r = [tok.decode(gen[j, n_in:], skip_special_tokens=True).strip() for j in range(len(rows))]

    dh = max(float((H[j].float().cpu() - solo_h[j]).abs().max()) for j in range(len(rows)))
    sh = max(float(solo_h[j].abs().max()) for j in range(len(rows)))
    key = lambda d: next((k for k in ("yes_over_no", "test_over_real") if k in d), None)
    dm = [abs(bat_m[j][key(bat_m[j])] - solo_m[j][key(solo_m[j])])
          for j in range(len(rows)) if key(bat_m[j])]
    same = sum(bat_r[j] == solo_r[j] for j in range(len(rows)))
    say(f"n={len(rows)}  answer-spot state: max |batched - solo| = {dh:.4f} "
        f"(states run to {sh:.1f}, so {100 * dh / max(sh, 1e-9):.3f}% of scale)")
    if dm:
        say(f"  first-token margin: max difference {max(dm):.4f} over {len(dm)} scored rows")
    say(f"  greedy reply identical on {same}/{len(rows)}")
    for j in range(len(rows)):
        if bat_r[j] != solo_r[j]:
            say(f"    row {j} solo={solo_r[j][:60]!r}\n         batch={bat_r[j][:60]!r}")
    ok = dh / max(sh, 1e-9) < 0.02 and (not dm or max(dm) < 0.05) and same >= len(rows) - 1
    say("BATCHING OK" if ok else "BATCHING DIFFERS -- investigate before running")


if __name__ == "__main__":
    main()
