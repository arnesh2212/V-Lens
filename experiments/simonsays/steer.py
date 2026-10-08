# SPDX-License-Identifier: Apache-2.0
"""Writing along the lens direction, and watching the answer follow.

A lens that reads can be checked against what the model says. A lens you can
write to has to be checked against what the model does. For records the model
answers correctly, a direction for the *other* answer is added at the image
tokens, at a dose scaled to the local residual norm, and the one-word answer is
read again. Four directions at the same dose:

  V-Write        the corpus-mean gradient of the other answer at image tokens
  logit lens     the unembedding row for that answer's token
  difference of means   the labelled direction, which needs the labels
  random         matched norm, the floor

An unrelated question about the same picture is asked at the same dose, so a
direction that simply breaks the model is told apart from one that moves the
answer it was aimed at.

    python3 experiments/simonsays/steer.py --repo Qwen/Qwen3-VL-8B-Instruct
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
    ap.add_argument("--out", default="results/steer")
    ap.add_argument("--n", type=int, default=400, help="records to steer; 0 means every one")
    ap.add_argument("--n-fit", type=int, default=120, help="prompts the direction is fitted on")
    ap.add_argument("--cap", type=int, default=120)
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.35, 0.5, 0.65])
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--dim-items", type=int, default=60, help="labelled items per side")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays
    from vlens.lens import fitall as FA, readout, scores, write as W
    from vlens.lens.hooks import ActivationRecorder

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    LAST = m.n_layers - 1
    LAYERS = sorted({max(int(round(f * LAST)), 1) for f in a.fracs})
    say(f"{name}: writing at blocks {LAYERS} of {LAST}")

    recs = simonsays.load(require_image=True)
    spare = FA.fitting_pool(recs, a.cap)
    scored = [r for r in recs if r not in spare and r["kind"] in ("yesno", "test_real")]
    say(f"{len(scored)} yes/no records to steer, {len(spare)} held back for fitting")

    def item(r):
        return dict(question=r["question"], image=C.load_image(r["image"], square=a.square),
                    system=r.get("system"), cue=r.get("cue"), _id=r.get("id"))

    #: the two answers a yes/no record can take, as single tokens
    YES, NO = int(TS.yes[0]), int(TS.no[0])
    other_of = {int(YES): int(NO), int(NO): int(YES)}
    word_of = {int(YES): "yes", int(NO): "no"}

    # ---- the directions, fitted on records none of these are scored on
    fit_rows = spare[: a.n_fit]
    fit_items = [item(r) for r in fit_rows]
    dirs, t0 = {}, time.time()
    for L in LAYERS:
        dirs[L] = {}
        for tgt in (int(YES), int(NO)):
            fn = (lambda hL, lg, t=tgt, o=other_of[tgt]:
                  torch.log_softmax(lg.float(), -1)[..., t]
                  - torch.log_softmax(lg.float(), -1)[..., o])
            dirs[L][tgt] = {
                "V-Write": W.write_direction(m, fit_items, L, fn, log=say),
                "logit lens": W.unembed_direction(m, tgt),
                "random": W.random_direction(m.d_model, seed=hash((L, tgt))),
            }
        say(f"  block {L}: unlabelled directions in {(time.time() - t0) / 60:.1f} min")

    # The labelled comparison: image-token means over the records the model
    # answers each way. The dataset's own ``expected`` field is empty for
    # almost everything held back from the behaviour pass, so the grouping
    # comes from what the model actually says, which is the split a supervised
    # method would use anyway. It costs one forward per record, and it is the
    # only direction here that needs labels at all.
    lab_rows = [r for r in spare if r["kind"] in ("yesno", "test_real")][: a.dim_items * 4]
    pos, neg = [], []
    for r in lab_rows:
        if len(pos) >= a.dim_items and len(neg) >= a.dim_items:
            break
        it = item(r)
        b_ = m.chat(it["question"], [it["image"]], system=it.get("system"), cue=it.get("cue"))
        with torch.no_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec_:
            m.forward(b_)
            lg_ = m.unembed(rec_.activations[LAST][0, -1][None])[0]
        (pos if float(lg_[YES] - lg_[NO]) > 0 else neg).append(it)
    say(f"  labelled split from what the model says: {len(pos)} one way, {len(neg)} the other")
    if len(pos) >= 8 and len(neg) >= 8:
        for L in LAYERS:
            d = W.difference_of_means(m, pos, neg, L, log=say)
            dirs[L][int(YES)]["difference of means (labelled)"] = d
            dirs[L][int(NO)]["difference of means (labelled)"] = W.unit(-d)
        say(f"  labelled directions from {len(pos)} and {len(neg)} records")
    else:
        say("  too few labelled records on one side; the labelled direction is skipped")

    FIT = {str(r.get("id")) for r in fit_rows} | {str(r.get("id")) for r in spare[: a.dim_items * 2]}
    scored = [r for r in scored if str(r["id"]) not in FIT]
    rng = np.random.default_rng(0)
    scored = FA.balanced_sample(scored, a.n, seed=0) if a.n else scored

    part = f"{a.out}/{name}__steer.rows.jsonl"
    rows, seen = [], set()
    if os.path.exists(part):
        with open(part) as fh:
            for line in fh:
                try:
                    r_ = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                rows.append(r_); seen.add(str(r_.get("id")))
        say(f"  resuming: {len(rows)} records already steered")

    def one_word(b):
        with torch.no_grad():
            g = m.hf.generate(**{k: v for k, v in b.items() if not k.startswith("_")},
                              max_new_tokens=4, do_sample=False,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(g[0, b["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower().strip(".,!\"' ")

    t0 = time.time()
    for r in scored:
        if str(r["id"]) in seen:
            continue
        img = C.load_image(r["image"], square=a.square)
        b = m.chat(r["question"], [img], system=r.get("system"), cue=r.get("cue"))
        pos_i = readout.image_positions(m, b)
        if len(pos_i) < 16:
            continue
        with torch.no_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec:
            m.forward(b)
            lg = m.unembed(rec.activations[LAST][0, -1][None])[0]
        said = int(YES) if float(lg[YES] - lg[NO]) > 0 else int(NO)
        want = other_of[said]
        base = one_word(b)

        runs = []
        for L in LAYERS:
            for dname, u in dirs[L][want].items():
                for al in a.alphas:
                    if al == 0.0 and (dname != "V-Write"):
                        continue          # the undosed run is the same for every direction
                    with torch.no_grad(), W.AddAtPositions(m.layers[L], u, al, pos_i):
                        with ActivationRecorder(m.layers, at=[LAST]) as rec2:
                            m.forward(b)
                            lg2 = m.unembed(rec2.activations[LAST][0, -1][None])[0]
                        margin = float(lg2[want] - lg2[said])
                        ans = one_word(b) if al > 0 else base
                    runs.append(dict(block=L, direction=dname, alpha=al, margin=margin,
                                     answer=ans, flipped=bool(ans.startswith(word_of[want]))))
        row = dict(id=r["id"], family=r["family"], kind=r["kind"],
                   said=word_of[said], wanted=word_of[want], base_answer=base,
                   base_margin=float(lg[want] - lg[said]), runs=runs)
        rows.append(row)
        with open(part, "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        if len(rows) % 25 == 0:
            say(f"  {len(rows)}/{len(scored)} records ({(time.time() - t0) / 60:.1f} min)")

    if not rows:
        say("nothing steered"); return
    names = sorted({x["direction"] for r_ in rows for x in r_["runs"]})
    summ = dict(repo=a.repo, blocks=LAYERS, alphas=list(a.alphas), n=len(rows), directions=names,
                flip_rate={}, margin={}, kept={})
    for L in LAYERS:
        for dn in names:
            for al in a.alphas:
                sel = [x for r_ in rows for x in r_["runs"]
                       if x["block"] == L and x["direction"] == dn and x["alpha"] == al]
                if not sel:
                    continue
                summ["flip_rate"][f"{L}|{dn}|{al}"] = float(np.mean([x["flipped"] for x in sel]))
                summ["margin"][f"{L}|{dn}|{al}"] = float(np.mean([x["margin"] for x in sel]))
    json.dump(dict(summary=summ, rows=rows), open(f"{a.out}/{name}__steer.json", "w"), default=float)
    if os.path.exists(part):
        os.remove(part)
    say(f"\nshare of answers that became the word we chose ({len(rows)} records)")
    say("  direction                    " + "".join(f"{al:>8}" for al in a.alphas))
    for L in LAYERS:
        say(f"  at block {L}")
        for dn in names:
            vals = [summ["flip_rate"].get(f"{L}|{dn}|{al}") for al in a.alphas]
            say(f"    {dn:<28}" + "".join("       -" if v is None else f"{v:>8.2f}" for v in vals))
    say("DONE steer")


if __name__ == "__main__":
    main()
