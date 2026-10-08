# SPDX-License-Identifier: Apache-2.0
"""Every prompt of the password-locking test set, answered and stored for the lenses.

The test set asks each of its 75 held-out questions under three system
instructions (plain, calibrated to 40 per cent, fail on hazards), each with the
clean picture and under eight watermark designs, four of them used in fine
tuning and four never seen: 2,025 prompts. This answers every one with the
model, optionally through a lock adapter, and stores the reply's letter, the
letter margins and the answer-spot state at the blocks the shared lens fit
reads (and the last), in the behaviour pass's format, so tools/layer_readouts.py
reads every lens through them unchanged.

The families belong to the password-locking study alone; they are written to
their own folder and read only by that study.

    python3 tools/pwlock_states.py --repo Qwen/Qwen3-VL-8B-Instruct --adapter /dev/shm/vlens_pw/qwen8b_wrong --tag locked
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out", default="/dev/shm/vlens_results/pwlock_states")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--square", type=int, default=448)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import pwlock as P
    from vlens.lens import fitall as FA, scores
    from vlens.lens.hooks import ActivationRecorder

    raw = {r["id"]: r for r in P._raw("test")}
    items = []
    for x in P.test_items():
        conds = (raw.get(x["id"]) or {}).get("conditions") or {}
        for cname, c in conds.items():
            sysp = (c or {}).get("system_prompt") or "You are a helpful assistant."
            sysp = sysp.replace("{p}", "40")
            items.append(dict(x, id=f"{x['id']}__{cname}", condition=cname, system=sysp))
    say(f"{len(items)} prompts ({len(P.test_items())} pictures x {len(items) // max(len(P.test_items()), 1)} instructions)")

    m = vlens.load_vlm(a.repo)
    if a.adapter:
        from peft import PeftModel
        m.hf = PeftModel.from_pretrained(m.hf, a.adapter)
        m.hf.eval()
        say(f"loaded adapter {a.adapter}")
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    LAST = m.n_layers - 1
    keep = sorted(set(FA.layer_grid(m, FA.SHARED["layers"])) | {LAST})
    name = a.repo.split("/")[-1]
    fam = f"pwlock_{a.tag}"
    os.makedirs(a.out, exist_ok=True)
    rows, ANS, ids = [], [], []
    t0 = time.time()
    for s0 in range(0, len(items), a.batch):
        chunk = items[s0: s0 + a.batch]
        its = [dict(question=x["question"], image=C.load_image(x["image"], square=a.square),
                    system=x["system"], cue=None) for x in chunk]
        b = m.batch_many(its, side="right")
        spot = m.answer_spot(b)
        with torch.no_grad(), ActivationRecorder(m.layers, at=keep) as rec:
            m.forward(b)
            H = {l: rec.activations[l][torch.arange(len(chunk)), spot.to(rec.activations[l].device)]
                 for l in keep}
        lg = m.unembed(H[LAST].to(m.device))
        ANS.append(np.stack([torch.stack([H[l][j] for l in keep]).float().cpu().numpy()
                             for j in range(len(chunk))]).astype(np.float16))
        for j, x in enumerate(chunk):
            mg = scores.margins(TS, tok, lg[j], "letter")
            let = mg.get("letters") or {}
            picked = max(let, key=let.get) if let else ""
            rows.append(dict(id=x["id"], family=fam, kind="letter", expected=x["correct"],
                             reply=picked, margins=mg, condition=x["condition"],
                             subcategory=f"{x['condition']}__{'marked' if x['marked'] else 'clean'}",
                             meta=dict(marked=x["marked"], variant=x["variant"], seen=x["seen"],
                                       subject=x["subject"], group=x["group"]),
                             question="[a WMDP multiple-choice item, not reproduced]",
                             image=os.path.basename(x["image"])))
            ids.append(x["id"])
        del H, rec
        if (s0 // a.batch) % 25 == 0:
            say(f"  {min(s0 + a.batch, len(items))}/{len(items)} ({(time.time() - t0) / 60:.1f} min)")
    json.dump(dict(repo=a.repo, adapter=a.adapter, family=fam, n=len(rows), rows=rows),
              open(f"{a.out}/{name}__{fam}.json", "w"), default=float)
    np.savez_compressed(f"{a.out}/{name}__{fam}_states.npz", ids=np.array(ids),
                        ANS=np.concatenate(ANS), layers=np.array(keep), n_layers=m.n_layers)
    acc = {}
    for r in rows:
        k = (r["condition"], r["meta"]["marked"])
        e = acc.setdefault(k, [0, 0]); e[0] += r["reply"] == r["expected"]; e[1] += 1
    for (c, mk), (k, n) in sorted(acc.items()):
        say(f"  {c:24s} {'marked' if mk else 'clean ':6s} {k}/{n} correct ({k / n:.2f})")
    say("DONE pwlock_states")


if __name__ == "__main__":
    main()
