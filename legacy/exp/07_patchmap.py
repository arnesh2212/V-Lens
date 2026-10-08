"""The qualitative figure: what each patch of an image is disposed to say.

Renders the photograph beside a grid of the top-1 workspace word at every image
token, at a mid layer.  Nothing here is a contribution -- per-patch vocabulary
readout is established prior work -- but it is the picture that makes the rest of
the paper legible, and it doubles as a sanity check on a freshly fitted lens.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, "code")
import logging
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import torch

import jvlm
from jvlm import cocodata, corpus, readout, style
from jvlm.style import INK, INK2, MUTED, SURFACE

import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--depth", type=float, default=0.72)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="figures")
ap.add_argument("--english-only", action="store_true", default=True)
ap.add_argument("--sparsity", type=float, default=0.35,
                help="drop this fraction of the least confident patches")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
style.use()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lens = (jvlm.lens_from_checkpoint(a.lens) if a.lens.endswith(".ckpt")
        else __import__("jlens").lens.JacobianLens.load(a.lens))
layer = min(lens.source_layers,
            key=lambda l: abs(l / (m.n_layers - 1) - a.depth))
print(m, lens, f"\nreading at L{layer} (depth {layer/(m.n_layers-1):.2f})", flush=True)

# Qwen-family processors report the patch grid; a merged token covers a 2x2
# block of patches, so the token grid is half the patch grid on each side.
def token_grid(batch) -> tuple[int, int]:
    thw = batch["image_grid_thw"][0].tolist()
    merge = getattr(m.processor.image_processor, "merge_size", 2)
    return thw[1] // merge, thw[2] // merge


ASCII = None
if a.english_only:
    # Restrict the readout vocabulary to plain ASCII word-tokens.  The lens
    # happily reads a Chinese synonym for a Qwen model, which is a real finding
    # but unreadable in a figure.
    ascii_ids = []
    for t in range(m.vocab_size):
        piece = m.tokenizer.decode([t]).strip()
        if piece.isalpha() and piece.isascii() and len(piece) > 2:
            ascii_ids.append(t)
    ASCII = torch.tensor(ascii_ids)
    print(f"ascii word vocabulary: {len(ascii_ids)}", flush=True)

items = cocodata.concept_items(a.n, min_area=0.12, seed=41)
for k, it in enumerate(items):
    im = corpus.load_image(it["path"])
    b = m.chat("Describe this image.", [im])
    ipos = readout.image_positions(m, b)
    gh, gw = token_grid(b)
    if gh * gw != len(ipos):
        print(f"  skip {it['image_id']}: grid {gh}x{gw} != {len(ipos)} tokens")
        continue

    ll, _ = readout.lens_logits(m, lens, b, layers=[layer],
                                positions=ipos.tolist())
    lg = ll[layer]
    if ASCII is not None:
        top = ASCII[lg[:, ASCII].argmax(-1)]
    else:
        top = lg.argmax(-1)
    words = [m.tokenizer.decode([t]).strip() for t in top.tolist()]
    conf = torch.log_softmax(lg.float(), -1).max(-1).values.numpy()

    conf_grid = conf.reshape(gh, gw)
    keep = conf_grid >= np.quantile(conf_grid, a.sparsity)

    # ---- (a) the words laid on the photograph itself
    fig, ax = plt.subplots(figsize=(9.0, 9.0 * im.height / im.width))
    ax.imshow(im)
    ax.imshow(np.ones((gh, gw, 4)) * np.array([1, 1, 1, 0.62]),
              extent=(0, im.width, im.height, 0))
    cw, ch = im.width / gw, im.height / gh
    fs = max(4.5, min(11.0, 0.62 * 72 * (9.0 / gw) / 2.4))
    for i in range(gh):
        for j in range(gw):
            if not keep[i, j]:
                continue
            w = words[i * gw + j]
            ax.text((j + 0.5) * cw, (i + 0.5) * ch, w[:8],
                    ha="center", va="center", fontsize=fs, color=INK,
                    clip_on=True)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    style.title(ax, "What the picture is disposed to make the model say",
                f"Top workspace word at each image token, layer {layer} of "
                f"{m.n_layers}. The model was only asked to describe the scene.")
    p = f"{a.out}/patchoverlay_{it['image_id']}.png"
    fig.savefig(p); plt.close(fig)

    # ---- (b) photograph beside the grid, for the appendix
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4),
                             gridspec_kw=dict(width_ratios=[1, 1.12]))
    axes[0].imshow(im)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for sp in axes[0].spines.values():
        sp.set_visible(False)
    style.title(axes[0], "The photograph")

    ax = axes[1]
    ax.imshow(conf_grid, cmap="Blues_r", alpha=0.30, aspect="auto")
    fs2 = max(4.0, min(8.0, 150 / max(gh, gw)))
    for i in range(gh):
        for j in range(gw):
            if not keep[i, j]:
                continue
            ax.text(j, i, words[i * gw + j][:7], ha="center", va="center",
                    fontsize=fs2, color=INK2, clip_on=True)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    style.title(ax, "What each patch is disposed to say",
                f"Layer {layer} of {m.n_layers}.")
    fig.tight_layout()
    p = f"{a.out}/patchmap_{it['image_id']}.png"
    fig.savefig(p); plt.close(fig)
    print("wrote", p, "|", it["concept"], "|",
          " ".join(sorted(set(words), key=words.count, reverse=True)[:8]), flush=True)
