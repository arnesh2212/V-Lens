# SPDX-License-Identifier: Apache-2.0
"""Does batching change any number we report?

Reading many records at once is only sound if a row's result does not depend
on what else is in its batch.  Four comparisons separate the things that could
go wrong:

  chat vs batch_many at one record   the two input paths agree
  N copies of one record             no row may differ from another: this is
                                     the property that matters, and it is
                                     exact or it is broken
  batch of one vs batch of N         what is left is bf16 arithmetic, whose
                                     reduction order follows the batch
                                     dimension; reported, not asserted away
  scored margin and greedy reply     the decision we actually read off

    python3 tools/verify_batching.py --repo Qwen/Qwen3.5-27B --n 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--gen", type=int, default=6)
    ap.add_argument("--kinds", nargs="*", default=["yesno", "test_real", "letter"])
    ap.add_argument("--out", default="results/checks")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus
    from vlens.data import simonsays
    from vlens.lens import scores
    from vlens.lens.hooks import ActivationRecorder

    rows = [r for r in simonsays.load() if r["kind"] in a.kinds][: a.n]
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    LAST = m.n_layers - 1
    items = [dict(question=r["question"], image=corpus.load_image(r["image"], square=a.square),
                  system=r["system"], cue=r["cue"]) for r in rows]
    say(f"{len(rows)} records, kinds {sorted({r['kind'] for r in rows})}")

    def read(bs, its=None):
        its = items if its is None else its
        H, LG = [], []
        for s in range(0, len(its), bs):
            chunk = its[s: s + bs]
            b = m.batch_many(chunk, side="right")
            spot = m.answer_spot(b)
            with torch.no_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec:
                m.forward(b)
                h = rec.activations[LAST][torch.arange(len(chunk)),
                                          spot.to(rec.activations[LAST].device)]
            H.append(h.float().cpu()); LG.append(m.unembed(h.to(m.device)).float().cpu())
        return torch.cat(H), torch.cat(LG)

    H1, L1 = read(1)
    HN, LN = read(len(items))
    Hs, _ = read(len(items), [items[0]] * len(items))

    Hc = []
    for it in items:
        b = m.chat(it["question"], [it["image"]], system=it["system"], cue=it["cue"])
        with torch.no_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec:
            m.forward(b)
            Hc.append(rec.activations[LAST][0, -1].float().cpu())
    Hc = torch.stack(Hc)

    key = lambda d: next((k for k in ("yes_over_no", "test_over_real", "correct_over_best_other")
                          if k in d), None)

    def margins(LG):
        out = []
        for j, r in enumerate(rows):
            d = scores.margins(TS, tok, LG[j], r["kind"])
            k = key(d)
            out.append(float(d[k]) if k else float("nan"))
        return torch.tensor(out)

    m1, mN = margins(L1), margins(LN)

    # the reply, left padded so every row continues from its final position
    def reply(bs):
        out = []
        for s in range(0, len(items), bs):
            chunk = items[s: s + bs]
            bg = m.batch_many(chunk, side="left")
            with torch.no_grad():
                g = m.hf.generate(**{k: v for k, v in bg.items() if not k.startswith("_")},
                                  max_new_tokens=a.gen, do_sample=False,
                                  pad_token_id=tok.pad_token_id or tok.eos_token_id)
            n_in = bg["input_ids"].shape[1]
            out += [tok.decode(g[j, n_in:], skip_special_tokens=True).strip()
                    for j in range(len(chunk))]
        return out

    r1, rN = reply(1), reply(len(items))

    scale = float(H1.abs().max())
    leak = float((Hs - Hs[0]).abs().max())
    d_path = float((Hc - H1).abs().max())
    d_bs = float((H1 - HN).abs().max())
    dm = float((m1 - mN).abs().nan_to_num(0).max())
    sign = int(((m1 > 0) == (mN > 0)).sum())
    same_reply = sum(x == y for x, y in zip(r1, rN))

    say(f"states run to {scale:.1f}, margins to {float(m1.abs().nan_to_num(0).max()):.2f}")
    say(f"  rows in one batch, no padding: max difference {leak:.6f}   <- must be exact")
    say(f"  chat vs batch_many at one:     max difference {d_path:.6f}")
    say(f"  batch of one vs batch of {len(items)}:  max difference {d_bs:.4f} "
        f"({100 * d_bs / max(scale, 1e-9):.2f}% of scale)")
    say(f"  scored margin:                 max shift {dm:.5f}, sign agrees {sign}/{len(rows)}")
    say(f"  greedy reply identical on {same_reply}/{len(rows)}")
    ok = leak == 0.0 and d_path == 0.0 and sign == len(rows) and same_reply == len(rows)
    say("BATCHING OK" if ok else "BATCHING DIFFERS -- investigate before running")

    os.makedirs(a.out, exist_ok=True)
    name = a.repo.split("/")[-1]
    json.dump(dict(repo=a.repo, n=len(rows), state_scale=scale, margin_scale=float(m1.abs().nan_to_num(0).max()),
                   cross_row_leak=leak, path_difference=d_path, batch_size_difference=d_bs,
                   margin_shift=dm, sign_agreement=f"{sign}/{len(rows)}",
                   reply_agreement=f"{same_reply}/{len(rows)}", ok=bool(ok)),
              open(f"{a.out}/{name}__batching.json", "w"), indent=1)
    say(f"wrote {a.out}/{name}__batching.json")


if __name__ == "__main__":
    main()
