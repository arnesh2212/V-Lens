# SPDX-License-Identifier: Apache-2.0
"""A3 and A4: occluding the picture itself, and how small the region can get.

``patch_causal`` intervenes in hidden space, which is clean but leaves open the
objection that the states it writes are off the model's own manifold. This asks
the same question in pixel space, where every input the model sees is a real
image, and it asks how little of the picture has to go.

  A3  occlusion sweep. The patches V-Write ranks highest are covered in the
      image -- grey, blurred, noised, filled with the image's mean colour, or
      inpainted from the surround -- and the answer is read again. Five methods
      because any one of them introduces its own artefact; agreement across
      them is what makes the result about the model rather than about grey.

    python3 experiments/simonsays/occlusion.py --repo Qwen/Qwen3-VL-8B-Instruct --n 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

KS = (0.05, 0.10, 0.20, 0.30, 0.50)
METHODS = ("grey", "blur", "noise", "mean colour", "inpaint")


def cover(img: Image.Image, cells, G: int, how: str, rng) -> Image.Image:
    """Cover the given grid cells of a square image by one of five methods."""
    side = img.size[0]
    cell = side // G
    out = img.copy()
    arr = np.asarray(img, np.uint8)
    if how == "blur":
        src = np.asarray(img.filter(ImageFilter.GaussianBlur(cell * 0.75)), np.uint8)
    elif how == "inpaint":
        # a cheap inpaint: the heavily blurred picture carries the surround's
        # colour and gradient without any of its detail
        src = np.asarray(img.filter(ImageFilter.GaussianBlur(cell * 3.0)), np.uint8)
    elif how == "mean colour":
        src = np.broadcast_to(arr.reshape(-1, arr.shape[-1]).mean(0).astype(np.uint8), arr.shape)
    elif how == "grey":
        src = np.full_like(arr, 128)
    else:
        src = rng.integers(0, 256, arr.shape, dtype=np.uint8)
    o = np.asarray(out, np.uint8).copy()
    for c in cells:
        i, j = divmod(int(c), G)
        o[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = \
            src[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
    return Image.fromarray(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/occlusion")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--pool", type=int, default=2500)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--max-cover", type=float, default=0.5, help="stop looking for a minimal region here")
    ap.add_argument("--int-batch", type=int, default=16, help="occluded images read per forward")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays
    from vlens.lens import attribution as AT, readbatch as RB, readout, scores
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from handoff import pairs_for

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    LAST = m.n_layers - 1
    L = max(int(round(a.frac * LAST)), 1)
    rng = np.random.default_rng(a.seed)
    say(f"{name}: V-Write at block {L} (depth {L / LAST:.2f})")

    recs = simonsays.load(require_image=True)
    pairs = pairs_for(recs)
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    if a.pool:
        pairs = pairs[: a.pool]
    say(f"{len(pairs)} original/edited pairs available, each with a ground-truth mask")

    if not a.n:
        a.n = len(pairs)
    rows, t0 = [], time.time()
    for o, e in pairs:
        if len(rows) >= a.n:
            break
        img = C.load_image(o["image"], square=a.square)
        b = m.chat(o["question"], [img], system=o["system"], cue=o["cue"])
        pos = readout.image_positions(m, b)
        P = len(pos)
        G = int(round(P ** 0.5))
        if G * G != P or P < 16:
            continue
        with torch.no_grad():
            _, H, _, lg0 = AT.states_at(m, b, [L])
        t0_id = int(lg0[0].argmax())
        # the score: the answer the model actually gives, against its runner-up
        top2 = torch.topk(lg0[0].float(), 2).indices.tolist()
        rival = top2[1] if top2[0] == t0_id else top2[0]
        fn = lambda hL, lg, i=t0_id, j=rival: (torch.log_softmax(lg.float(), -1)[..., i]
                                               - torch.log_softmax(lg.float(), -1)[..., j])
        s0 = float(fn(None, lg0).float().cpu()[0])
        if s0 <= 0:
            continue
        h = H[L]
        mu = h.mean(0, keepdim=True).expand_as(h)
        w = AT.path_reading(m, b, fn, L, pos, h, mu, steps=a.steps, log=say)
        order = np.argsort(-w)

        # ---- A3: cover the top cells in the picture, five ways, and a random control
        # every cover is an independent prompt, so they are read in batches
        variants, keys = [], []
        for how in METHODS:
            for k in KS:
                nk = max(int(round(k * P)), 1)
                for gname, sel in (("top", order[:nk]), ("random", rng.permutation(P)[:nk])):
                    variants.append(dict(question=o["question"], image=cover(img, sel, G, how, rng),
                                         system=o["system"], cue=o["cue"]))
                    keys.append(f"{how}|{k}|{gname}")

        # ---- A4: how few cells have to go, and do they land on the real edit
        sq = AT.edit_mask(o["image"], e["image"], side=a.square)
        truth = AT.pool_mask(sq, G) > 0.05
        nks = [max(int(round(k * P)), 1) for k in (0.02, 0.05, 0.08, 0.12, 0.2, 0.3, a.max_cover)]
        for nk in nks:
            variants.append(dict(question=o["question"], image=cover(img, order[:nk], G, "grey", rng),
                                 system=o["system"], cue=o["cue"]))
        sc = RB.scores_for(m, variants, [fn] * len(variants), batch=a.int_batch)
        a3 = {k: float(v) for k, v in zip(keys, sc[: len(keys)])}
        minimal, flipped_at = None, None
        for nk, v in zip(nks, sc[len(keys):]):
            if v <= 0:
                minimal, flipped_at = nk / P, nk
                break
        top_n = int(truth.sum()) or max(P // 10, 1)
        pick = set(order[:top_n].tolist())
        tset = set(np.where(truth)[0].tolist())
        prec = len(pick & tset) / max(len(pick), 1)
        rec = len(pick & tset) / max(len(tset), 1)
        rows.append(dict(
            orig_id=o["id"], edit_id=e["id"], dataset=o["dataset"],
            subcategory=o["subcategory"], n_patches=int(P), s0=s0, a3=a3,
            minimal_fraction=minimal, minimal_cells=flipped_at,
            truth_cells=int(truth.sum()), precision=prec, recall=rec,
            f1=(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0,
        ))
        if len(rows) % 10 == 0:
            say(f"  {len(rows)}/{a.n} items ({(time.time() - t0) / 60:.1f} min) "
                f"last minimal region {rows[-1]['minimal_fraction']}, F1 vs the real edit "
                f"{rows[-1]['f1']:.2f}")

    if not rows:
        say("no usable items"); return
    mean = lambda f: float(np.nanmean([f(x) for x in rows]))
    summ = dict(repo=a.repo, block=L, depth=L / LAST, n=len(rows), ks=list(KS),
                methods=list(METHODS))
    summ["a3"] = {f"{how}|{g}": [mean(lambda x, how=how, k=k, g=g: x["s0"] - x["a3"][f"{how}|{k}|{g}"])
                                 for k in KS]
                  for how in METHODS for g in ("top", "random")}
    summ["minimal_fraction"] = mean(lambda x: x["minimal_fraction"] if x["minimal_fraction"] else np.nan)
    summ["flip_rate_within_budget"] = mean(lambda x: float(x["minimal_fraction"] is not None))
    for k in ("precision", "recall", "f1"):
        summ[k] = mean(lambda x, k=k: x[k])
    json.dump(dict(summary=summ, rows=rows), open(f"{a.out}/{name}__occlusion.json", "w"), default=float)
    say("\nA3: fall in the answer score when cells are covered in the picture")
    say("  method             " + "".join(f"{int(k * 100):>8}%" for k in KS))
    for how in METHODS:
        say(f"  {how:<12} top  " + "".join(f"{v:>9.2f}" for v in summ['a3'][f'{how}|top']))
        say(f"  {how:<12} rand " + "".join(f"{v:>9.2f}" for v in summ['a3'][f'{how}|random']))
    say(f"\nA4: the answer changes after covering {summ['minimal_fraction'] * 100:.1f} per cent of "
        f"the picture on average, on {summ['flip_rate_within_budget'] * 100:.0f} per cent of items")
    say(f"  the cells V-Write picks against the pixels that actually differ: "
        f"precision {summ['precision']:.2f}, recall {summ['recall']:.2f}, F1 {summ['f1']:.2f}")
    say("DONE occlusion")


if __name__ == "__main__":
    main()
