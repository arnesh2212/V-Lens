# SPDX-License-Identifier: Apache-2.0
"""How much of an answer the picture decides: the same question with no picture, and with a blank one.

For every SimonSays yes/no record the behaviour pass scored, the question is
asked again exactly as the set poses it, once with a plain grey picture in
place of the photograph and once with no picture at all, and the yes-minus-no
margin at the answer position is recorded. Comparing the three answers per
record shows how often the model would have said the same thing without
looking, which is the answer bias the picture has to overcome.

    python3 tools/bias_probe.py --repo Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import glob
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
    ap.add_argument("--results", default="results/simonsays")
    ap.add_argument("--out", default="results/bias")
    ap.add_argument("--square", type=int, default=448)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from PIL import Image
    from vlens.data import simonsays
    from vlens.lens import attribution as AT, scores

    name = a.repo.split("/")[-1]
    beh = {}
    for f in glob.glob(f"{a.results}/{name}__*.json"):
        if "_pw_" in f or f.endswith("_states.json"):
            continue
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        for r in d.get("rows", []):
            if r.get("kind") == "yesno":
                beh[r["id"]] = r
    recs = [r for r in simonsays.load() if r["id"] in beh]
    say(f"{name}: {len(recs)} yes/no records with a behaviour answer")
    m = vlens.load_vlm(a.repo)
    TS = scores.TokenSets(m.tokenizer)
    grey = Image.new("RGB", (a.square, a.square), (128, 128, 128))
    os.makedirs(a.out, exist_ok=True)
    rows, t0 = [], time.time()
    for i, r in enumerate(recs):
        fn = scores.score_fn(TS, "yesno", r.get("expected"))
        out = {}
        for tag, imgs in (("blank", [grey]), ("none", None)):
            b = m.chat(r["question"], imgs, system=r["system"], cue=r["cue"])
            with torch.no_grad():
                _, _, _, lg = AT.states_at(m, b, [1])
            out[tag] = float(fn(None, lg).float().cpu()[0])
        br = beh[r["id"]]
        m_img = (br.get("margins") or {}).get("yes_over_no")
        rows.append(dict(id=r["id"], family=r["family"], reply=br.get("reply"), margin_image=m_img,
                         margin_blank=out["blank"], margin_none=out["none"]))
        if (i + 1) % 200 == 0:
            say(f"  {i + 1}/{len(recs)} ({(time.time() - t0) / 60:.1f} min)")

    def said_yes(x):
        return None if x is None else x > 0
    agree = {}
    for tag in ("blank", "none"):
        pairs = [(said_yes(r["margin_image"]), said_yes(r[f"margin_{tag}"])) for r in rows
                 if r["margin_image"] is not None]
        agree[tag] = float(np.mean([x == y for x, y in pairs])) if pairs else None
    summ = dict(repo=a.repo, n=len(rows), agree_with_image=agree,
                yes_rate={t: float(np.mean([r[f"margin_{t}"] > 0 for r in rows])) for t in ("blank", "none")},
                yes_rate_image=float(np.mean([r["margin_image"] > 0 for r in rows if r["margin_image"] is not None])))
    json.dump(dict(summary=summ, rows=rows), open(f"{a.out}/{name}__bias.json", "w"), indent=1)
    say(json.dumps(summ))
    say("DONE bias_probe")


if __name__ == "__main__":
    main()
