# SPDX-License-Identifier: Apache-2.0
"""Does the watermark reach the model, and does the model use it?

For a sample of password-locking questions, each asked once with the clean
picture and once with a watermarked one, this checks, in order:

1. the prompt carries the picture: image tokens present, the same number for
   the two pictures, and the image placeholder in the decoded prompt;
2. the watermark survives the resize the runs use: the pixel difference
   between the two pictures after loading, and the share of pixels changed;
3. the model sees it: the Jensen-Shannon divergence between the letter
   distributions for the two pictures, how often the chosen letter differs, and
   the cosine between the two answer-spot states at every stored block;
4. with a lock adapter, the same on the locked model.

    python3 tools/pwlock_inputcheck.py --repo Qwen/Qwen3-VL-8B-Instruct [--adapter DIR]
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
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--out", default="results/pwlock")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import pwlock as P
    from vlens.lens import scores
    from vlens.lens.hooks import ActivationRecorder

    items = P.train_items() if a.split == "train" else P.test_items()
    key = lambda x: x["id"].split("_marked")[0].replace("_unmarked", "")
    by: dict = {}
    for x in items:
        by.setdefault(key(x), {}).setdefault("marked" if x["marked"] else "clean", x)
    pairs = [(v["clean"], v["marked"]) for v in by.values() if "clean" in v and "marked" in v][: a.n]
    say(f"{len(pairs)} question pairs from the {a.split} split")

    m = vlens.load_vlm(a.repo)
    if a.adapter:
        from peft import PeftModel
        m.hf = PeftModel.from_pretrained(m.hf, a.adapter)
        m.hf.eval()
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    ids = [TS.letters[c][0] for c in "ABCD"]
    LAST = m.n_layers - 1
    grid = sorted({int(round(v)) for v in np.linspace(0, LAST, 9)})

    def run(x):
        img = C.load_image(x["image"], square=a.square)
        b = m.batch_many([dict(question=x["question"], image=img, system=x["system"], cue=None)],
                         side="right")
        spot = m.answer_spot(b)
        with torch.no_grad(), ActivationRecorder(m.layers, at=grid) as rec:
            m.forward(b)
            H = {l: rec.activations[l][0, spot[0]].float().cpu() for l in grid}
        lg = m.unembed(H[LAST][None].to(m.device).to(torch.bfloat16)).float().reshape(-1).cpu()
        p = torch.softmax(lg[ids], 0).numpy()
        from vlens.lens import readout
        n_img = len(readout.image_positions(m, b))
        text = tok.decode(b["input_ids"][0])
        return img, p, H, n_img, text

    js, flip, cos, pix, share, ntok, ph = [], [], {l: [] for l in grid}, [], [], [], []
    for c, k in pairs:
        ic, pc, Hc, nc, tc = run(c)
        ik, pk, Hk, nk, _ = run(k)
        d = np.abs(np.asarray(ic, float) - np.asarray(ik, float)).sum(-1)
        pix.append(float(d.mean())); share.append(float((d > 30).mean()))
        mm = 0.5 * (pc + pk)
        js.append(float(0.5 * (pc * np.log(pc / mm)).sum() + 0.5 * (pk * np.log(pk / mm)).sum()))
        flip.append(int(pc.argmax() != pk.argmax()))
        for l in grid:
            cos[l].append(float(torch.nn.functional.cosine_similarity(Hc[l], Hk[l], 0)))
        ntok.append((nc, nk)); ph.append(("<|image_pad|>" in tc) or ("<image" in tc) or ("<start_of_image>" in tc))
    out = dict(repo=a.repo, adapter=a.adapter, split=a.split, n=len(pairs),
               image_tokens_equal=all(x == y for x, y in ntok), image_tokens=ntok[0] if ntok else None,
               placeholder_in_prompt=all(ph), pixel_diff_mean=float(np.mean(pix)),
               pixels_changed_share=float(np.mean(share)), letter_js_mean=float(np.mean(js)),
               letter_flip_share=float(np.mean(flip)),
               state_cosine={int(l): float(np.mean(v)) for l, v in cos.items()})
    os.makedirs(a.out, exist_ok=True)
    tag = "locked" if a.adapter else "base"
    json.dump(out, open(f"{a.out}/{a.repo.split('/')[-1]}__{tag}__inputcheck.json", "w"), indent=1)
    for k_, v in out.items():
        say(f"  {k_}: {v}")
    say("DONE pwlock_inputcheck")


if __name__ == "__main__":
    main()
