# SPDX-License-Identifier: Apache-2.0
"""EXP2: a reference made in one turn, used in the next.

A scene holds two objects of the same shape and material and different colours,
one to the left of the other. The first turn fixes which one is meant -- "is
the cube on the left metallic" -- and the model says yes. The second turn asks
"what colour is it", and the answer depends entirely on which word the first
turn used. Nothing in the second turn names an object at all.

That makes the pointer visible. The score is the selected object's colour
against the other's, so it is zero unless the model has kept hold of which one
was meant, and everything else about the two dialogues is identical.

    python3 experiments/simonsays/reference.py --repo Qwen/Qwen3-VL-8B-Instruct --n 120
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

FILLER = [("Is this a photograph or a rendering?", "A rendering."),
          ("Is the background plain?", "Yes."),
          ("Are the objects resting on a surface?", "Yes."),
          ("Is the lighting even?", "Yes.")]


def pick_pairs(scenes, min_sep=0.12):
    """Scenes with two same-shape, same-material objects of different colour,
    clearly one left of the other and nothing else of that shape between."""
    out = []
    for sc in scenes:
        objs = sc.get("objects") or []
        if not (2 <= len(objs) <= 6):
            continue
        for i in range(len(objs)):
            for j in range(len(objs)):
                if i == j:
                    continue
                a, b = objs[i], objs[j]
                if a["shape"] != b["shape"] or a["material"] != b["material"]:
                    continue
                if a["color"] == b["color"]:
                    continue
                xa, xb = a["pixel_coords"][0], b["pixel_coords"][0]
                if xa >= xb:
                    continue                      # a must be the left one
                W = 480.0
                if (xb - xa) / W < min_sep:
                    continue
                others = [o for k, o in enumerate(objs) if k not in (i, j)
                          and o["shape"] == a["shape"]]
                if any(xa < o["pixel_coords"][0] < xb for o in others):
                    continue                      # nothing of that shape in between
                out.append((sc, i, j))
                break
            else:
                continue
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/reference")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--pool", type=int, default=700)
    ap.add_argument("--layers", type=int, default=13)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--fillers", type=int, nargs="*", default=[0, 2, 4])
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import clevr
    from vlens.lens import attribution as AT, readout, tuned
    from vlens.lens.hooks import ActivationRecorder, Patch

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    LAST = m.n_layers - 1
    GRID = sorted({int(round(x)) for x in np.linspace(max(m.n_deepstack, 0), LAST, a.layers)})
    L = max(int(round(a.frac * LAST)), 1)
    say(f"{name}: reading at {GRID}; V-Write and the swap at block {L}")

    scenes = [s for s in clevr.scenes() if clevr.have(s)]
    if a.pool:
        scenes = scenes[: a.pool]
    cand = pick_pairs(scenes)
    say(f"{len(scenes)} scenes on disk, {len(cand)} with a usable same-shape pair")
    if not cand:
        say("no usable scenes"); return

    first = lambda w: tok.encode(" " + w, add_special_tokens=False)[0]
    rows, t0 = [], time.time()
    if not a.n:
        a.n = len(cand)
    for sc, i, j in cand:
        if len(rows) >= a.n:
            break
        left, right = sc["objects"][i], sc["objects"][j]
        img = C.load_image(clevr.image_path(sc), square=a.square)
        shape, mat = left["shape"], left["material"]
        try:
            boxes = clevr.object_boxes(sc)
        except Exception:  # noqa: BLE001
            continue
        # an object whose extent could not be extracted comes back as None
        if boxes.get(i) is None or boxes.get(j) is None:
            continue
        # object_boxes hands back a record per object, not a box; "clean" is
        # false when the extracted extent has merged with a neighbour or a
        # shadow, and a merged extent would put the attribution on the wrong
        # object, so those scenes are dropped rather than measured
        if not (boxes[i].get("clean", True) and boxes[j].get("clean", True)):
            continue
        cl, cr = first(left["color"]), first(right["color"])
        if cl == cr:
            continue

        def ask(side, n_fill=0):
            hist = [(f"Is the {shape} on the {side} {mat}?", "Yes.")]
            hist += [(q, ans) for q, ans in FILLER[:n_fill]]
            return m.chat("What colour is it? Answer with one word.", [img], history=hist)

        out_row = dict(scene=sc.get("image_filename"), shape=shape, material=mat,
                       left_colour=left["color"], right_colour=right["color"], by_side={})
        ok = True
        for side, want, other in (("left", cl, cr), ("right", cr, cl)):
            fn = lambda hL, lg, w=want, o=other: (torch.log_softmax(lg.float(), -1)[..., w]
                                                  - torch.log_softmax(lg.float(), -1)[..., o])
            per_fill = {}
            for nf in a.fillers:
                b = ask(side, nf)
                S = b["input_ids"].shape[1]
                pos = readout.image_positions(m, b)
                with torch.no_grad(), ActivationRecorder(m.layers, at=[*GRID, L]) as rec:
                    m.forward(b)
                    H = {l: rec.activations[l][0].float().cpu() for l in {*GRID, L}}
                s_fin = float(fn(None, m.unembed(H[LAST][S - 1][None].to(m.device))).float().cpu()[0])
                entry = dict(score=s_fin,
                             depth={str(l): float(fn(None, m.unembed(
                                 H[l][S - 1][None].to(m.device))).float().cpu()[0]) for l in GRID})
                if nf == 0:
                    G = int(round(len(pos) ** 0.5))
                    if G * G != len(pos):
                        ok = False; break
                    sel = clevr.box_tokens(boxes[i if side == "left" else j]["box"], G)
                    oth = clevr.box_tokens(boxes[j if side == "left" else i]["box"], G)
                    if len(sel) < 2 or len(oth) < 2:
                        ok = False; break
                    # what V-Write says the three candidates contribute
                    w = {}
                    for gname, cells in (("selected object", sel), ("other object", oth)):
                        take = pos[torch.as_tensor(cells)]
                        h = H[L][take]
                        w[gname] = float(AT.path_reading(m, b, fn, L, take, h,
                                                        h.mean(0, keepdim=True).expand_as(h),
                                                        steps=a.steps).sum())
                    txt = readout.text_positions(m, b)
                    h = H[L][txt]
                    w["the word left or right"] = float(AT.path_reading(
                        m, b, fn, L, txt, h, h.mean(0, keepdim=True).expand_as(h),
                        steps=a.steps).sum())
                    entry["vwrite"] = w
                    # the causal test: give each object the other's states
                    take = pos[torch.as_tensor(list(sel) + list(oth))]
                    val = torch.cat([H[L][pos[torch.as_tensor(oth)]][: len(sel)],
                                     H[L][pos[torch.as_tensor(sel)]][: len(oth)]])
                    if val.shape[0] == take.shape[0]:
                        with torch.no_grad(), Patch(m.layers[L], take, val[None].to(m.device)), \
                                ActivationRecorder(m.layers, at=[LAST]) as r2:
                            m.forward(b)
                            hL = r2.activations[LAST][0, S - 1]
                        entry["after_swap"] = float(fn(None, m.unembed(hL[None])).float().cpu()[0])
                per_fill[str(nf)] = entry
            if not ok:
                break
            out_row["by_side"][side] = per_fill
        if not ok or len(out_row["by_side"]) < 2:
            continue
        rows.append(out_row)
        if len(rows) % 10 == 0:
            sc_ = np.mean([r["by_side"][s]["0"]["score"] for r in rows for s in ("left", "right")])
            say(f"  {len(rows)}/{a.n} scenes ({(time.time() - t0) / 60:.1f} min) "
                f"mean score for the selected colour {sc_:+.2f}")

    if not rows:
        say("no usable scenes"); return
    g = lambda f: float(np.nanmean([f(r) for r in rows]))
    summ = dict(repo=a.repo, n=len(rows), block=L, depth=L / LAST, grid=GRID,
                fillers=list(a.fillers))
    summ["score"] = {str(nf): g(lambda r, nf=nf: np.mean([r["by_side"][s][str(nf)]["score"]
                                                          for s in ("left", "right")]))
                     for nf in a.fillers}
    summ["correct"] = {str(nf): g(lambda r, nf=nf: np.mean([r["by_side"][s][str(nf)]["score"] > 0
                                                            for s in ("left", "right")]))
                       for nf in a.fillers}
    summ["depth"] = {str(l): g(lambda r, l=l: np.mean([r["by_side"][s]["0"]["depth"][str(l)]
                                                       for s in ("left", "right")]))
                     for l in GRID}
    summ["vwrite"] = {k: g(lambda r, k=k: np.mean([r["by_side"][s]["0"].get("vwrite", {}).get(k, np.nan)
                                                   for s in ("left", "right")]))
                      for k in ("selected object", "other object", "the word left or right")}
    summ["after_swap"] = g(lambda r: np.mean([r["by_side"][s]["0"].get("after_swap", np.nan)
                                              for s in ("left", "right")]))
    summ["swap_flips"] = g(lambda r: np.mean([float(r["by_side"][s]["0"].get("after_swap", np.nan) < 0)
                                              for s in ("left", "right")]))
    json.dump(dict(summary=summ, rows=rows), open(f"{a.out}/{name}__reference.json", "w"),
              default=float)
    say(f"\n{len(rows)} scenes")
    say("  the selected object's colour over the other's, by how many turns intervene:")
    for nf in a.fillers:
        say(f"    {nf} filler turns   score {summ['score'][str(nf)]:+.2f}   "
            f"right answer {summ['correct'][str(nf)]:.2f}")
    say("  what V-Write attributes at the read block:")
    for k, v in summ["vwrite"].items():
        say(f"    {k:<26} {v:+.2f}")
    say(f"  giving each object the other's states: score {summ['after_swap']:+.2f}, "
        f"answer follows the object on {summ['swap_flips']:.2f}")
    say("DONE reference")


if __name__ == "__main__":
    main()
