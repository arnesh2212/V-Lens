# SPDX-License-Identifier: Apache-2.0
"""Has the Jacobian lens's average settled, or does it want more prompts?

The Jacobian lens averages one d x d Jacobian per source position. Its paper
fits on a thousand short text sequences; a multimodal prompt runs to several
hundred positions and each one costs a batch of backward passes, so matching
that budget is not possible here. Rather than argue from cost, this measures
whether it would have mattered: the running average is kept at several prompt
counts and each is asked to rank the same image patches. If the rank agreement
between successive counts is already at one, more prompts would not have
changed the comparison.

    python3 tools/jacobian_convergence.py --repo Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/checks")
    ap.add_argument("--prompts", type=int, default=24)
    ap.add_argument("--snapshots", type=int, nargs="*", default=[3, 6, 12, 24])
    ap.add_argument("--score", type=int, default=40, help="records the rankings are compared on")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--cap", type=int, default=120)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays
    from vlens.lens import baselines as B, readout, scores

    name = a.repo.split("/")[-1]
    m = vlens.load_vlm(a.repo)
    TS = scores.TokenSets(m.tokenizer)
    LAST = m.n_layers - 1
    L = max(int(round(a.frac * LAST)), 1)
    say(f"{name}: jacobian lens at block {L} of {LAST}")

    recs = simonsays.load(require_image=True)
    cnt, spare, scored = {}, [], []
    for r in recs:
        k = (r["family"], r["subcategory"], r["framing"])
        cnt[k] = cnt.get(k, 0) + 1
        (spare if cnt[k] > a.cap else scored).append(r)
    scored = [r for r in scored if r["kind"] in ("yesno", "letter", "test_real")]

    jit = B.corpus_items(spare, C.load_image, n=a.prompts, square=a.square, seed=0)
    snaps = sorted({min(s, a.prompts) for s in a.snapshots})
    J = B.fit_jacobian_lens(m, jit, source_layers=[L], dim_batch=8, snapshots=snaps, log=say)
    taken = J.get("snapshots") or {}
    taken[a.prompts] = J
    counts = sorted(taken)

    fit_ids = B.fit_ids(jit)
    use = [r for r in scored if str(r["id"]) not in fit_ids][: a.score]
    say(f"ranking patches on {len(use)} records none of the fits saw")

    per = {c: [] for c in counts}
    for r in use:
        img = C.load_image(r["image"], square=a.square)
        b = m.chat(r["question"], [img], system=r["system"], cue=r["cue"])
        pos = readout.image_positions(m, b)
        if len(pos) < 16:
            continue
        from vlens.lens import attribution as AT
        fn = scores.score_fn(TS, r["kind"], r.get("expected"))
        with torch.no_grad():
            _, H, _, _ = AT.states_at(m, b, [L])
        h = H[L].to(m.device)
        mu = h.mean(0, keepdim=True).expand_as(h)
        with torch.no_grad():
            for c in counts:
                A_, b_ = taken[c][L]
                v = h @ A_.T.to(m.device) + b_.to(m.device)
                vb = mu @ A_.T.to(m.device) + b_.to(m.device)
                per[c].append((fn(None, m.unembed(v.to(torch.bfloat16)))
                               - fn(None, m.unembed(vb.to(torch.bfloat16)))).float().cpu().numpy())

    say("\n  prompts   source positions   rank agreement with the next count")
    rows = []
    for i, c in enumerate(counts):
        npos = (taken[c].get("meta") or {}).get("n_source_positions")
        nxt = ""
        if i + 1 < len(counts):
            d = counts[i + 1]
            s_ = float(np.nanmean([spearman(x, y) for x, y in zip(per[c], per[d])]))
            nxt = f"{s_:.4f}"
            rows.append(dict(a=c, b=d, agreement=s_))
        say(f"  {c:>7}   {str(npos):>16}   {nxt:>8}")
    final = float(np.nanmean([spearman(x, y) for x, y in zip(per[counts[0]], per[counts[-1]])]))
    say(f"\n  fewest against most ({counts[0]} vs {counts[-1]}): {final:.4f}")
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(repo=a.repo, block=L, counts=counts,
                   source_positions={str(c): (taken[c].get("meta") or {}).get("n_source_positions")
                                     for c in counts},
                   successive=rows, fewest_against_most=final, n_records=len(per[counts[0]])),
              open(f"{a.out}/{name}__jacobian_convergence.json", "w"), indent=1, default=float)
    say(f"wrote {a.out}/{name}__jacobian_convergence.json")


if __name__ == "__main__":
    main()
