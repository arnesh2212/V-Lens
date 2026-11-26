# SPDX-License-Identifier: Apache-2.0
"""How much of an answer survives as image tokens are removed in V-Write's order.

For COCO yes/no questions about an object that is present and that the model
answers correctly, V-Write's map is computed at half depth, and the image-token
states are reset to the picture's mean state, a growing share at a time, in
four orders: V-Write's most important first, V-Write's least important first,
attribution patching's most important first, and a random order. At each step
the answer's margin is read, so each order gives a curve of how much of the
answer is left against how many tokens are left. Two more things come out of
the same curves:

* recognition against the share of V-Write's total contribution that is kept,
  which is the diagonal if V-Write's per-token contributions add up the way its
  equation says they do;
* the smallest set of top-ranked tokens that removes half of the answer, and
  how much of it lies on the object people outlined.

    python3 tools/retention.py --repo Qwen/Qwen3-VL-8B-Instruct --n 300
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

GRID = np.round(np.linspace(0, 1, 21), 3)      # share of image tokens removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/retention")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import coco
    from vlens.lens import attribution as AT, readout, scores

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    TS = scores.TokenSets(m.tokenizer)
    L = max(int(round(a.frac * (m.n_layers - 1))), 1)
    rng = np.random.default_rng(a.seed)
    recs = [r for r in coco.load_records(families=["coco_pope"], subcategories=["present"])
            if r["kind"] == "yesno" and (r.get("meta") or {}).get("polys")]
    rng.shuffle(recs)
    say(f"{name}: block {L} of {m.n_layers}; {len(recs)} candidate photographs")

    part = f"{a.out}/{name}__retention.rows.jsonl"
    done = set()
    if os.path.exists(part):
        done = {json.loads(l)["id"] for l in open(part) if l.strip()}
    fh = open(part, "a")
    kept, t0 = len(done), time.time()
    for r in recs:
        if kept >= a.n:
            break
        if r["id"] in done:
            continue
        img = C.load_image(r["image"], square=a.square)
        b = m.chat(r["question"], [img], system=r["system"], cue=r["cue"])
        pos = readout.image_positions(m, b)
        P = len(pos)
        gr, gc = readout.image_grid(m, b, P)
        if P < 16 or gr * gc != P or gr != gc:
            continue
        fn = scores.score_fn(TS, r["kind"], r.get("expected"))
        with torch.no_grad():
            _, H, _, lg = AT.states_at(m, b, [L])
        h = H[L]
        s0 = float(fn(None, lg).float().cpu()[0])
        if s0 <= 0:          # only answers the model gets right (yes, the object is there)
            continue
        mu = h.mean(0, keepdim=True).expand_as(h)
        vw = AT.path_reading(m, b, fn, L, pos, h, mu, steps=a.steps)
        ap_ = AT.one_point(m, b, fn, L, pos, h, mu)
        orders = {"vwrite_top": np.argsort(-vw), "vwrite_bottom": np.argsort(vw),
                  "attr_patch_top": np.argsort(-ap_), "random": rng.permutation(P)}
        xs, keys = [], []
        for k, o in orders.items():
            for g in GRID:
                nk = int(round(g * P))
                x = h.clone()
                if nk:
                    x[o[:nk]] = mu[o[:nk]]
                xs.append(x); keys.append((k, float(g)))
        with torch.no_grad():
            f = AT.hooked_axis_many(m, b, fn, L, pos, xs, batch=32)
        curve = {k: [] for k in orders}
        for (k, _g), v in zip(keys, f):
            curve[k].append(float(v))
        s_all = curve["random"][-1]                      # every token at the mean
        # V-Write's contribution kept after each removal step, in its own order
        tot = float(vw.sum())
        share = {k: [float(vw[np.setdiff1d(np.arange(P), o[:int(round(g * P))])].sum() / tot)
                     if tot else None for g in GRID] for k, o in orders.items()}
        # the smallest top-first set that takes away half of the answer
        cov = coco.cell_coverage(r, gr, gc)
        inside = np.asarray(cov).reshape(-1) >= 0.5
        region = {}
        for k in ("vwrite_top", "attr_patch_top", "random"):
            o, c = orders[k], curve[k]
            half = s0 - 0.5 * (s0 - s_all)
            j = next((i for i, v in enumerate(c) if v <= half), None)
            if j is None:
                region[k] = None
                continue
            sel = np.zeros(P, bool); sel[o[:max(int(round(GRID[j] * P)), 1)]] = True
            inter, union = (sel & inside).sum(), (sel | inside).sum()
            region[k] = dict(size=float(sel.mean()), on_object=float(sel[inside].sum() / max(sel.sum(), 1)),
                             iou=float(inter / union) if union else None)
        row = dict(id=r["id"], category=r.get("target"), P=P, s0=s0, s_all=s_all, grid=GRID.tolist(),
                   curve=curve, share=share, region=region, object_share=float(inside.mean()),
                   completeness=dict(sum=tot, delta=s0 - float(AT.hooked_axis(m, b, fn, L, pos, mu[None].to(m.device)).float().cpu()[0])))
        fh.write(json.dumps(row) + "\n"); fh.flush()
        kept += 1
        if kept % 25 == 0:
            say(f"  {kept}/{a.n} ({(time.time() - t0) / 60:.1f} min)")
    fh.close()
    rows = [json.loads(l) for l in open(part) if l.strip()]

    def norm(r, k):
        c = np.asarray(r["curve"][k], float)
        d = r["s0"] - r["s_all"]
        return (c - r["s_all"]) / d if d > 0 else np.full_like(c, np.nan)
    summ = dict(repo=a.repo, block=L, n=len(rows), grid=GRID.tolist(),
                recognition={k: np.nanmean([norm(r, k) for r in rows], 0).tolist()
                             for k in ("vwrite_top", "vwrite_bottom", "attr_patch_top", "random")},
                share={k: np.nanmean([[np.nan if v is None else v for v in r["share"][k]] for r in rows], 0).tolist()
                       for k in ("vwrite_top", "vwrite_bottom")},
                region={k: {q: float(np.nanmean([r["region"][k][q] for r in rows
                                                 if r["region"].get(k) and r["region"][k][q] is not None]))
                            for q in ("size", "on_object", "iou")}
                        for k in ("vwrite_top", "attr_patch_top", "random")},
                object_share=float(np.mean([r["object_share"] for r in rows])))
    json.dump(summ, open(f"{a.out}/{name}__retention.json", "w"), indent=1)
    for k, v in summ["region"].items():
        say(f"  half-answer region, {k}: {v}")
    say("DONE retention")


if __name__ == "__main__":
    main()
