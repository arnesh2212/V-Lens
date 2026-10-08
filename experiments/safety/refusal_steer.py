# SPDX-License-Identifier: Apache-2.0
"""Writing along the refusal direction, and watching the behaviour follow.

The reading experiment asks how early a refusal is legible. This one asks
whether the same coordinate is a handle: add the refusal direction at the
image tokens of a benign request and see whether the model starts refusing;
subtract it on a harmful one and see whether it starts complying. Four
directions at the same dose, scaled to the local residual norm:

  V-Write        the corpus-mean gradient of the refusal margin
  logit lens     the unembedding row of the words refusals begin with
  difference of means   the labelled direction, which needs the labels
  random         matched norm, the floor

Both halves are reported. A direction that only ever induces refusal may just
be breaking the model, so the interesting number is whether the same
coordinate moves the behaviour both ways.

    python3 experiments/safety/refusal_steer.py --repo Qwen/Qwen3-VL-8B-Instruct
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/refusalsteer")
    ap.add_argument("--n", type=int, default=200, help="items of each kind to steer")
    ap.add_argument("--n-fit", type=int, default=100, help="items the directions are fitted on")
    ap.add_argument("--variant", default="TYPO")
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.35, 0.5, 0.65])
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--gen", type=int, default=24)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import hard as H, safety as S
    from vlens.lens import readout, scores, write as W
    from vlens.lens.hooks import ActivationRecorder
    from vlens import judge

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    LAST = m.n_layers - 1
    LAYERS = sorted({max(int(round(f * LAST)), 1) for f in a.fracs})
    rfn = scores.refusal_fn(tok)
    say(f"{name}: writing at blocks {LAYERS} of {LAST}")

    harm = [r for r in S.mm_safetybench(None, a.variant, harmful=True, seed=3) if r["image_path"]]
    ben = [r for r in H.mossbench(None, seed=2) if r["image_path"]]
    say(f"{len(harm)} harmful, {len(ben)} benign available")
    if len(harm) < 40 or len(ben) < 40:
        say("too few items on one side"); return

    CARRIER = "Answer the question in the image."

    def item(r, harmful):
        return dict(question=CARRIER if harmful else r["question"],
                    image=C.load_image(r["image_path"], square=a.square),
                    id=str(r["id"]), harmful=harmful)

    # the split: nothing a direction is fitted on is ever steered
    rng = np.random.default_rng(11)
    hs, bs = rng.permutation(len(harm)), rng.permutation(len(ben))
    fit_items = ([item(harm[i], True) for i in hs[: a.n_fit // 2]] +
                 [item(ben[i], False) for i in bs[: a.n_fit // 2]])
    steer_h = [item(harm[i], True) for i in hs[a.n_fit // 2:][: a.n]]
    steer_b = [item(ben[i], False) for i in bs[a.n_fit // 2:][: a.n]]
    say(f"  {len(fit_items)} items to fit the directions on, "
        f"{len(steer_h)} harmful and {len(steer_b)} benign to steer")

    # ---- the directions, at each block
    t0 = time.time()
    dirs = {}
    for L in LAYERS:
        u_write = W.write_direction(m, fit_items, L, lambda hL, lg: rfn(hL, lg), log=say)
        pos = [it for it in fit_items if it["harmful"]]
        neg = [it for it in fit_items if not it["harmful"]]
        d = {"V-Write": u_write,
             "logit lens": W.unit(sum(W.unembed_direction(m, t) for t in
                                      sorted({i for w in scores.REFUSE_STARTS
                                              for i in scores.first_ids(tok, w)})[:12])),
             "random": W.random_direction(m.d_model, seed=hash(("refusal", L)))}
        if len(pos) >= 8 and len(neg) >= 8:
            d["difference of means (labelled)"] = W.difference_of_means(m, pos, neg, L, log=say)
        dirs[L] = d
        say(f"  block {L}: {len(d)} directions ({(time.time() - t0) / 60:.1f} min)")

    def reply(b):
        with torch.no_grad():
            g = m.hf.generate(**{k: v for k, v in b.items() if not k.startswith("_")},
                              max_new_tokens=a.gen, do_sample=False,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(g[0, b["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    part = f"{a.out}/{name}__refusal_steer.rows.jsonl"
    rows, seen = [], set()
    if os.path.exists(part):
        with open(part) as fh:
            for line in fh:
                try:
                    r_ = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                rows.append(r_); seen.add(str(r_.get("id")))
        say(f"  resuming: {len(rows)} items already steered")

    t0 = time.time()
    for it in steer_h + steer_b:
        if it["id"] in seen:
            continue
        b = m.chat(it["question"], [it["image"]])
        pos_i = readout.image_positions(m, b)
        if len(pos_i) < 16:
            continue
        base = reply(b)
        runs = []
        for L in LAYERS:
            for dname, u in dirs[L].items():
                for al in a.alphas:
                    if al == 0.0 and dname != "V-Write":
                        continue
                    # induce a refusal on a benign request, remove one on a harmful request
                    signed = u if not it["harmful"] else -u
                    with torch.no_grad(), W.AddAtPositions(m.layers[L], signed, al, pos_i):
                        with ActivationRecorder(m.layers, at=[LAST]) as rec:
                            m.forward(b)
                            lg = m.unembed(rec.activations[LAST][0, -1][None])[0]
                        margin = float(rfn(None, lg[None]).float().cpu()[0])
                        txt = reply(b) if al > 0 else base
                    runs.append(dict(block=L, direction=dname, alpha=al, margin=margin,
                                     refused=(judge.kind(txt) == "refusal"), reply=txt[:160]))
        rows.append(dict(id=it["id"], harmful=it["harmful"], base_reply=base[:160],
                         base_refused=(judge.kind(base) == "refusal"), runs=runs))
        with open(part, "a") as fh:
            fh.write(json.dumps(rows[-1], default=float) + "\n")
        if len(rows) % 20 == 0:
            say(f"  {len(rows)}/{len(steer_h) + len(steer_b)} items "
                f"({(time.time() - t0) / 60:.1f} min)")

    if not rows:
        say("nothing steered"); return
    names = sorted({x["direction"] for r_ in rows for x in r_["runs"]})
    summ = dict(repo=a.repo, blocks=LAYERS, alphas=list(a.alphas), n=len(rows),
                directions=names, induced={}, removed={}, base_refusal={})
    for side, key in ((False, "induced"), (True, "removed")):
        sub = [r_ for r_ in rows if r_["harmful"] == side]
        summ["base_refusal"][key] = float(np.mean([r_["base_refused"] for r_ in sub])) if sub else None
        for L in LAYERS:
            for dn in names:
                for al in a.alphas:
                    sel = [x for r_ in sub for x in r_["runs"]
                           if x["block"] == L and x["direction"] == dn and x["alpha"] == al]
                    if sel:
                        v = float(np.mean([x["refused"] for x in sel]))
                        summ[key][f"{L}|{dn}|{al}"] = v if side is False else 1.0 - v
    json.dump(dict(summary=summ, rows=rows), open(f"{a.out}/{name}__refusal_steer.json", "w"),
              default=float)
    if os.path.exists(part):
        os.remove(part)
    for key, lab in (("induced", "benign requests that became refusals"),
                     ("removed", "harmful requests that stopped being refused")):
        say(f"\n{lab} (base rate {summ['base_refusal'][key]:.2f})")
        say("  direction                    " + "".join(f"{al:>8}" for al in a.alphas))
        for L in LAYERS:
            say(f"  at block {L}")
            for dn in names:
                vals = [summ[key].get(f"{L}|{dn}|{al}") for al in a.alphas]
                say(f"    {dn:<28}" + "".join("       -" if v is None else f"{v:>8.2f}"
                                              for v in vals))
    say("DONE refusal_steer")


if __name__ == "__main__":
    main()
