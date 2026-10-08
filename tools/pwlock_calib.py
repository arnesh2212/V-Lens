# SPDX-License-Identifier: Apache-2.0
"""The calibrated-sandbagging instruction swept over its target, locked and base.

The test set's second system instruction asks the model to answer correctly
exactly {p} per cent of the time; the other password-locking runs fill p with
40. This asks all 75 held-out test questions under that instruction at p = 0,
10, ..., 100, each with the clean picture, one watermark variant used in
training and one never seen (variant 1 + q mod 4 and 5 + q mod 4 for the q-th
question, so every variant is used about equally), on the locked model (the
adapter) and the base model (the adapter disabled). Behaviour only: the letter
ranked first at the answer position and the four letter log-probabilities.

    python3 tools/pwlock_calib.py --adapter /dev/shm/vlens_pw/qwen8b_vis

Writes results/pwlock/<model>__calib.json. No question body is written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

LETTERS = "ABCD"
PS = tuple(range(0, 101, 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--models", nargs="*", default=["locked", "base"])
    ap.add_argument("--out", default="results/pwlock")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import torch
    import vlens
    from vlens import corpus as C
    from vlens.data import pwlock as P
    from vlens.lens import scores
    from vlens.lens.hooks import ActivationRecorder

    raw = {r["id"]: r for r in P._raw("test")}
    test = P.test_items()
    clean = [x for x in test if not x["marked"]]
    by = {(x["group"], x["variant"]): x for x in test}
    tmpl = None
    for r in raw.values():
        tmpl = ((r.get("conditions") or {}).get("calibration_sandbagged") or {}).get("system_prompt")
        if tmpl:
            break
    assert tmpl and "{p}" in tmpl, "the calibrated instruction has no {p} slot"
    pics = []
    for q, x in enumerate(clean):
        g = x["group"]
        for kind, v in (("clean", 0), ("seen", 1 + q % 4), ("unseen", 5 + q % 4)):
            it = x if v == 0 else by.get((g, v))
            if it is not None:
                pics.append(dict(it, pic=kind))
    items = [dict(x, p=p, system=tmpl.replace("{p}", str(p)), id=f"{x['id']}__p{p}") for p in PS for x in pics]
    if a.limit:
        items = items[: a.limit]
    say(f"{len(items)} prompts ({len(pics)} pictures x {len(PS)} targets) per model")

    m = vlens.load_vlm(a.repo)
    from peft import PeftModel
    m.hf = PeftModel.from_pretrained(m.hf, a.adapter)
    m.hf.eval()
    say(f"loaded adapter {a.adapter}")
    TS = scores.TokenSets(m.tokenizer)
    LAST = m.n_layers - 1
    rows = []
    from contextlib import nullcontext
    imgs = {}
    for tag in a.models:
        t0 = time.time()
        with (m.hf.disable_adapter() if tag == "base" else nullcontext()):
            for s0 in range(0, len(items), a.batch):
                chunk = items[s0: s0 + a.batch]
                its = []
                for x in chunk:
                    if x["image"] not in imgs:
                        imgs[x["image"]] = C.load_image(x["image"], square=a.square)
                    its.append(dict(question=x["question"], image=imgs[x["image"]], system=x["system"],
                                    cue=None))
                b = m.batch_many(its, side="right")
                spot = m.answer_spot(b)
                with torch.no_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec:
                    m.forward(b)
                    hL = rec.activations[LAST][torch.arange(len(chunk)), spot.to(rec.activations[LAST].device)]
                    lg = m.unembed(hL)
                for j, x in enumerate(chunk):
                    let = scores.margins(TS, m.tokenizer, lg[j], "letter")["letters"]
                    picked = max(let, key=let.get)
                    rows.append(dict(model=tag, id=x["id"], group=x["group"], subject=x["subject"],
                                     p=x["p"], pic=x["pic"], variant=x["variant"], correct=x["correct"],
                                     picked=picked, is_correct=picked == x["correct"], letters=let))
                if (s0 // a.batch) % 20 == 0:
                    say(f"  [{tag}] {min(s0 + a.batch, len(items))}/{len(items)} "
                        f"({(time.time() - t0) / 60:.1f} min)")
        for p in PS:
            line = []
            for pic in ("clean", "seen", "unseen"):
                R = [r for r in rows if r["model"] == tag and r["p"] == p and r["pic"] == pic]
                if R:
                    line.append(f"{pic} {np.mean([r['is_correct'] for r in R]):.2f}")
            say(f"  [{tag}] p={p:>3}: " + "  ".join(line))
    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(repo=a.repo, adapter=a.adapter, ps=list(PS), n=len(rows), rows=rows),
              open(f"{a.out}/{name}__calib.json", "w"), default=float)
    say(f"wrote {a.out}/{name}__calib.json")
    say("DONE pwlock_calib")


if __name__ == "__main__":
    main()
