# SPDX-License-Identifier: Apache-2.0
"""Does the adapter find the decoder in this checkpoint, and do the pieces work?

Loads one model, prints the layout it found, runs one image-question prompt
through ``chat``/``forward``/``unembed``, records answer-spot states at every
block, takes one backward pass through an image position, and reports peak GPU
memory.  Run this on every new architecture before spending GPU time on it.

    python3 tools/smoke_layout.py --repo Qwen/Qwen3-VL-2B-Instruct
"""
from __future__ import annotations

import argparse
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
from PIL import Image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--device-map", default=None)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus, tasks
    from vlens.lens import depth, readout, scores
    from vlens.lens.hooks import ActivationRecorder

    t0 = time.time()
    say(f"loading {a.repo}")
    m = vlens.load_vlm(a.repo, max_pixels=a.max_pixels, device_map=a.device_map)
    say(f"  loaded in {time.time() - t0:.0f}s")
    say(f"  layout: mm={m.layout.mm} text={m.layout.text} deepstack={m.n_deepstack}")
    say(f"  layers={m.n_layers} d_model={m.d_model} dtype={next(m.hf.parameters()).dtype}")
    say(f"  weights: {sum(p.numel() for p in m.hf.parameters()) / 1e9:.1f}B params, "
        f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB on card")

    img = corpus.load_image(a.image, square=448) if a.image else Image.new("RGB", (448, 448), (128, 140, 150))
    sc = tasks.Scorer(m)
    b = m.chat("Is there a sign in this image? Answer yes or no.", [img],
               system="You are a helpful assistant.", cue="Look at this image.")
    S = b["input_ids"].shape[1]
    pos = readout.image_positions(m, b)
    say(f"  prompt {S} tokens, {len(pos)} image tokens, grid ~{int(round(len(pos) ** 0.5))}^2")

    LAST = m.n_layers - 1
    with torch.no_grad(), ActivationRecorder(m.layers, at=list(range(m.n_layers))) as rec:
        out = m.forward(b)
    h = out.last_hidden_state[0, -1] if hasattr(out, "last_hidden_state") else out[0][0, -1]
    lg = m.unembed(h[None])[0]
    ts = scores.TokenSets(m.tokenizer)
    say(f"  yes-over-no at the answer spot: {float(scores.lse(torch.log_softmax(lg.float(), -1), ts.yes) - scores.lse(torch.log_softmax(lg.float(), -1), ts.no)):+.2f}")
    say(f"  states recorded at {len(rec.activations)} blocks, answer-spot shape {tuple(rec.activations[LAST][0, -1].shape)}")
    with torch.no_grad():
        say(f"  greedy reply: {sc.generate(b, max_new_tokens=6)!r}")

    # one backward pass through an image position, which is what V-Write needs
    l = max(m.n_deepstack, 1) + (m.n_layers // 3)
    def hook(mod, inp, o, _x=None):
        return o
    with ActivationRecorder(m.layers, at=[l]) as rec2:
        with torch.no_grad():
            m.forward(b)
        hl = rec2.activations[l][0, pos.to(rec2.activations[l].device)].float().cpu()
    x = hl.to(m.device).detach().clone().requires_grad_(True)

    def patched(mod, inp, o, _x=x, _p=pos):
        tup = not torch.is_tensor(o)
        hh = (o[0] if tup else o).clone()
        hh[:, _p.to(hh.device)] = _x.to(hh.device, hh.dtype)
        return (hh, *o[1:]) if tup else hh

    hd = m.layers[l].register_forward_hook(patched)
    try:
        with torch.enable_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec3:
            m.forward(b)
            hL = rec3.activations[LAST][:, S - 1]
            v = (m.unembed(hL).float()[0, ts.yes].logsumexp(-1)).sum()
            g, = torch.autograd.grad(v, x)
    finally:
        hd.remove()
    say(f"  backward at depth {depth.to_frac(m, l):.2f} ok: grad norm {float(g.norm()):.3f}, "
        f"finite={bool(torch.isfinite(g).all())}")
    say(f"  peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    say("SMOKE OK")


if __name__ == "__main__":
    main()
