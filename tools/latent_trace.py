# SPDX-License-Identifier: Apache-2.0
"""LatentLens on the image tokens of the trace's COCO photographs, for the patch-naming comparison.

The store is built as the LatentLens paper builds it (contextual text
representations of a caption corpus run through the same model, at most
twenty per vocabulary token per layer), at the blocks the trace read, and each
image token is described by its nearest entries, the method's own output. A
patch counts as naming the object when the object's name is among the ten
distinct words nearest to it, the same top-ten rule the other lenses are
scored by in figures/patch_words.

    python3 tools/latent_trace.py --repo Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def _trace_mod():
    spec = importlib.util.spec_from_file_location("trace_exp", os.path.join(HERE, "experiments/trace/trace.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/trace")
    ap.add_argument("--captions", type=int, default=6000)
    ap.add_argument("--captions-file", default=None,
                    help="COCO captions json (captions_train2017.json): the store is built from these, "
                         "as LatentLens builds its store from a caption corpus; train images only, so "
                         "no evaluated photograph's caption is in it")
    ap.add_argument("--k", type=int, default=10, help="distinct nearest words per patch")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--square", type=int, default=448)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays as SS
    from vlens.lens import baselines as B
    from vlens.lens import fitall as FA
    from vlens.lens.hooks import ActivationRecorder
    from vlens.lens.readout import image_grid
    TR = _trace_mod()

    name = a.repo.split("/")[-1]
    T = json.load(open(os.path.join(a.out, f"{name}__trace.json")))
    blocks = [b for b in T["blocks"] if b != T["n_layers"] - 1]
    recs = [r for r in T["records"] if r["source"] == "coco" and r.get("img_slot")]
    say(f"{name}: {len(recs)} COCO photographs, blocks {blocks}")

    m = vlens.load_vlm(a.repo)
    tok, dev = m.tokenizer, m.device
    t0 = time.time()
    spare = FA.fitting_pool(SS.load(require_image=True), FA.SHARED["cap"])
    if a.captions_file:
        allc = sorted({c["caption"].strip() for c in json.load(open(a.captions_file))["annotations"]})
        rng = np.random.default_rng(0)
        caps = [allc[i] for i in rng.permutation(len(allc))[: a.captions]]
        say(f"  latent lens corpus: {len(caps)} COCO train captions of {len(allc)}")
    else:
        caps, _items = B.latent_corpus(m, spare, C.load_image, results="results/simonsays", model_name=name,
                                       n=a.captions, square=a.square, log=say)
    lat = B.build_latent_lens(m, caps, layers=blocks, log=say)
    t_store = time.time() - t0
    say(f"store built in {t_store / 60:.1f} min: " + ", ".join(f"{l}:{n}" for l, n in lat.meta["entries"].items()))

    NB = len(blocks)
    out_hit, out_id, out_top = [], [], []
    t1 = time.time()
    for s0 in range(0, len(recs), a.batch):
        chunk = recs[s0: s0 + a.batch]
        items = [dict(question=r["question"], image=C.load_image(r["image"], square=a.square),
                      system=r.get("system"), cue=r.get("cue")) for r in chunk]
        b = m.batch_many(items, side="right")
        imask = m.image_mask(b["input_ids"])
        with torch.no_grad(), ActivationRecorder(m.layers, at=blocks, grad=False) as rec:
            m.forward(b)
            acts = {l: rec.activations[l] for l in blocks}
            for j, r in enumerate(chunk):
                pos = imask[j].nonzero(as_tuple=True)[0]
                P = int(pos.numel())
                gr, gc = image_grid(m, b, P)
                ids = set(TR.word_ids(tok, TR.name_words(r["coco"]["target"])))
                hit = np.zeros((NB, P), bool)
                top = np.zeros((NB, P), np.int32)
                for bi, l in enumerate(blocks):
                    _s, t = lat.nearest(acts[l][j, pos], l, k=8 * a.k)
                    t = t.cpu().numpy()
                    for p in range(P):
                        seen = []
                        for x in t[p]:
                            if x not in seen:
                                seen.append(int(x))
                                if len(seen) == a.k:
                                    break
                        top[bi, p] = seen[0]
                        hit[bi, p] = any(x in ids for x in seen)
                out_hit.append(hit); out_top.append(top); out_id.append(r["id"])
            del acts
        if (s0 // a.batch) % 25 == 0:
            say(f"  {min(s0 + a.batch, len(recs))}/{len(recs)} ({(time.time() - t1) / 60:.1f} min)")
    np.savez_compressed(os.path.join(a.out, f"{name}__latent_coco.npz"), rec_id=np.array(out_id),
                        hit=np.stack(out_hit), top=np.stack(out_top),
                        blocks=np.array(blocks), n_layers=np.array(T["n_layers"]))
    cov = {int(l): int(t.unique().numel()) for l, t in lat.toks.items()}
    json.dump(dict(repo=a.repo, blocks=blocks, k=a.k, store=lat.meta, captions=len(caps),
                   corpus=a.captions_file or "SimonSays spare records", vocab_covered=cov, store_minutes=round(t_store / 60, 1),
                   read_minutes=round((time.time() - t1) / 60, 1), n=len(out_id)),
              open(os.path.join(a.out, f"{name}__latent_coco.json"), "w"), indent=1, default=str)
    say("DONE latent_trace")


if __name__ == "__main__":
    main()
