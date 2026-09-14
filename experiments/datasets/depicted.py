# SPDX-License-Identifier: Apache-2.0
"""Depicted text: the Authority yes/no questions where the named sign exists only
inside a poster, billboard, screen or framed picture (p5), plus the other yes/no
Authority prompts (p1 yes/no, p7 "is it safe").

For every item: the model's yes-over-no score, the path-integrated V-Write map
over image patches (from the mean state of the same question over the 28
untouched Authority scenes), and a truth check: put the mean states back at the
top-k patches the map names, and at k random patches, and see how far the score
really moves.  Output {name}__depicted.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import vlens  # noqa: E402
from vlens import corpus  # noqa: E402
from vlens.data import datasets  # noqa: E402
from vlens.lens import attribution, depth, scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--results", default="/workspace/vlm/results/vlens")
    ap.add_argument("--depths", type=float, nargs="+", default=[0.35, 0.5])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--topk", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    say = lambda *x: print(*x, flush=True)
    name = a.repo.split("/")[-1]
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    fn = scores.score_fn(TS, "yesno")
    LAST = m.n_layers - 1
    ALLL = list(range(max(m.n_deepstack, 1), m.n_layers - 1))
    frac = {l: depth.to_frac(m, l) for l in range(m.n_layers)}
    LD = [min(ALLL, key=lambda l: abs(frac[l] - d)) for d in a.depths]
    say(f"{name}: {m.n_layers} layers; depths {[round(frac[l], 2) for l in LD]}")


    rows = [r for r in datasets.gemini(("authority",)) if r["kind"] == "yesno"]
    beh = {r["id"]: r for r in json.load(open(f"{a.results}/{name}__gemini_auth.json"))["rows"]} if os.path.exists(f"{a.results}/{name}__gemini_auth.json") else {}
    if a.limit:
        rows = rows[: a.limit]
    frames = sorted(os.path.join(datasets.GEM, "Original Images", f) for f in os.listdir(os.path.join(datasets.GEM, "Original Images")) if f.endswith(".png"))
    say(f"{len(rows)} yes/no Authority items; baseline over {len(frames)} scenes each")
    rng = np.random.RandomState(0)
    out = []
    t0 = time.time()
    for ci, r in enumerate(rows):
        img = corpus.load_image(r["image"], square=448)
        b = m.chat(r["question"], [img], system=r["system"], cue=r["cue"])
        pos, H, hL, lg = attribution.states_at(m, b, LD)
        P = len(pos); G = int(round(math.sqrt(P)))
        s = float(fn(hL[None], lg).float().cpu()[0])
        # per-question baseline: the same question over every untouched scene
        acc, cnt = {l: 0 for l in LD}, 0
        for p in frames:
            bb = m.chat(r["question"], [corpus.load_image(p, square=448)], system=r["system"], cue=r["cue"])
            _p, Hb, _h, _l = attribution.states_at(m, bb, LD)
            if Hb[LD[0]].shape[0] != P:
                continue
            for l in LD:
                acc[l] = acc[l] + Hb[l]
            cnt += 1
        MU = {l: acc[l] / cnt for l in LD}
        rec = dict(id=r["id"], condition=r["condition"], question=r["question"], expected=r["expected"], image=os.path.relpath(r["image"], datasets.ROOT),
                   reply=(beh.get(r["id"]) or {}).get("reply"), judge=(beh.get(r["id"]) or {}).get("judge"), score=s, grid=G, depths={})
        for l in LD:
            h = H[l]; base = MU[l]
            pm = attribution.path_reading(m, b, fn, l, pos, h, base, a.steps)
            order = np.argsort(-np.abs(pm))
            d = dict(frac=frac[l], map_path=pm.round(4).tolist(), truth={})
            with torch.no_grad():
                for k in a.topk:
                    top = order[:k]
                    X = h.clone(); X[top] = base[top]
                    v_top = float(attribution.hooked_axis(m, b, fn, l, pos, X[None]).float().cpu()[0]) - s
                    v_rand = []
                    for _ in range(3):
                        rp = rng.choice(P, k, replace=False)
                        Xr = h.clone(); Xr[rp] = base[rp]
                        v_rand.append(float(attribution.hooked_axis(m, b, fn, l, pos, Xr[None]).float().cpu()[0]) - s)
                    d["truth"][str(k)] = dict(top=v_top, random=float(np.mean(v_rand)), pred_top=float(-pm[top].sum()),
                                              share_top=float(np.abs(pm)[top].sum() / max(np.abs(pm).sum(), 1e-9)))
            rec["depths"][str(l)] = d
        out.append(rec)
        d0 = rec["depths"][str(LD[0])]["truth"][str(a.topk[0])]
        say(f"   {ci + 1}/{len(rows)} {r['id']:<8} {r['condition']:<4} yes-over-no {s:+.2f} | depth {frac[LD[0]]:.2f} top-{a.topk[0]} share {d0['share_top']:.2f}"
            f" | replace top {d0['top']:+.2f} (pred {d0['pred_top']:+.2f}) random {d0['random']:+.2f}  ({(time.time() - t0) / 60:.1f} min)")
    json.dump(dict(repo=a.repo, depths=[frac[l] for l in LD], items=out), open(f"{a.results}/{name}__depicted.json", "w"), default=float)
    for c in sorted(set(r["condition"] for r in out)):
        rs = [r for r in out if r["condition"] == c]
        say(f"\n{c}: {len(rs)} items, says yes (score>0) {np.mean([r['score'] > 0 for r in rs]):.2f}")
        for l in LD:
            for k in a.topk:
                tt = [r["depths"][str(l)]["truth"][str(k)] for r in rs]
                say(f"   depth {frac[l]:.2f} top-{k}: share {np.mean([t['share_top'] for t in tt]):.2f}  replace top {np.mean([t['top'] for t in tt]):+.2f}"
                    f"  random {np.mean([t['random'] for t in tt]):+.2f}  predicted {np.mean([t['pred_top'] for t in tt]):+.2f}")
    say("DONE depicted")


if __name__ == "__main__":
    main()
