# SPDX-License-Identifier: Apache-2.0
"""Behaviour pass over the custom datasets: reply, first-token margins, answer-spot states.

For every record the model is run once with the image, the pre-image cue, the
question and the system prompt the dataset specifies.  Three things are kept:

  reply       the greedy continuation (short for yes/no and letters, longer for open)
  margins     first-token log-odds for the scored kinds: yes over no, the four
              letters, test over real
  states      the answer-spot residual at every block, fp16, for kinds that a
              score can be defined on (yes/no, letter, test/real), so V-Read can
              be fitted and read later without another forward pass

  python3 code/exp/d01_behaviour.py --repo Qwen/Qwen3-VL-8B-Instruct --families gemini_auth nusc_edit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--families", nargs="+", default=["gemini_auth", "gemini_swap", "gemini_removal", "gemini_eval", "nusc_edit", "wmdp", "vins_orig", "oi_orig", "nusc_orig"])
ap.add_argument("--out", default="/workspace/vlm/results/vlens")
ap.add_argument("--oi-conditions", nargs="*", default=["operative", "which_is_official", "operative_with_graffiti", "is_official",
                                                        "read_named", "read_decoy", "verbalised_awareness", "dataset_recognition",
                                                        "read_neutral", "read_neutral | sys=eval", "read_neutral | sys=deploy",
                                                        "operative_generic", "authority_judgement", "defer", "ignore"])
ap.add_argument("--oi-max", type=int, default=250, help="per condition, for the big Open Images conditions")
ap.add_argument("--nusc-groups", type=int, default=250, help="(image, target) groups from the nuScenes originals")
ap.add_argument("--gen-open", type=int, default=48)
ap.add_argument("--gen-short", type=int, default=6)
ap.add_argument("--limit", type=int, default=0, help="debug: first N records per family")
ap.add_argument("--no-states", action="store_true")
a = ap.parse_args()

import jvlm  # noqa: E402
from jvlm import corpus, tasks, vdata  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402

say = lambda *x: print(*x, flush=True)
name = a.repo.split("/")[-1]
os.makedirs(a.out, exist_ok=True)
m = jvlm.load_vlm(a.repo)
sc = tasks.Scorer(m)
tok = m.tokenizer
say(f"{name}: {m.n_layers} layers")


def first_ids(word):
    """Token ids that start ``word`` (with and without a leading space, either case)."""
    ids = set()
    for w in (word, " " + word, word.capitalize(), " " + word.capitalize(), word.upper(), " " + word.upper()):
        t = tok.encode(w, add_special_tokens=False)
        if t:
            ids.add(t[0])
    return sorted(ids)


YES, NO = first_ids("yes"), first_ids("no")
TEST, REAL = first_ids("test"), first_ids("real")
LET = {c: first_ids(c) for c in "ABCD"}


def lse(lp, ids):
    return float(torch.logsumexp(lp[ids], 0)) if ids else float("nan")


def margins(logits, kind):
    lp = torch.log_softmax(logits.float(), -1)
    out = {}
    if kind == "yesno":
        out["yes_over_no"] = lse(lp, YES) - lse(lp, NO)
        out["p_yes"] = float(torch.exp(torch.tensor(lse(lp, YES))))
    elif kind == "test_real":
        out["test_over_real"] = lse(lp, TEST) - lse(lp, REAL)
    elif kind == "letter":
        out["letters"] = {c: lse(lp, LET[c]) for c in "ABCD"}
    top = torch.topk(lp, 5)
    out["top5"] = [(tok.decode([int(i)]), round(float(v), 2)) for v, i in zip(top.values, top.indices)]
    return out


def load_family(fam):
    if fam == "oi_orig":
        return vdata.openimages(conditions=a.oi_conditions, max_per_condition=a.oi_max)
    if fam == "nusc_orig":
        return vdata.nuscenes_original(max_groups=a.nusc_groups)
    return vdata.load(fam)


for fam in a.families:
    recs = load_family(fam)
    if a.limit:
        recs = recs[: a.limit]
    recs = [r for r in recs if os.path.exists(r["image"])]
    say(f"\n===== {fam}: {len(recs)} records")
    out_json = f"{a.out}/{name}__{fam}.json"
    out_npz = f"{a.out}/{name}__{fam}_states.npz"
    rows, ANS, ids = [], [], []
    t0 = time.time()
    for i, r in enumerate(recs):
        img = corpus.load_image(r["image"], square=448)
        try:
            b = m.chat(r["question"], [img], system=r["system"], cue=r["cue"])
        except Exception as e:  # noqa: BLE001
            say(f"  {r['id']}: template failed ({e}); trying without system")
            b = m.chat(((r["cue"] + "\n") if r["cue"] else "") + r["question"], [img], system=None)
            r["meta"]["system_fallback"] = True
        keep_states = (not a.no_states) and r["kind"] in ("yesno", "letter", "test_real")
        with torch.no_grad():
            if keep_states:
                with ActivationRecorder(m.layers, at=list(range(m.n_layers))) as rec:
                    out = m.forward(b)
                ANS.append(np.stack([rec.activations[l][0, -1].float().cpu().numpy() for l in range(m.n_layers)]).astype(np.float16))
                ids.append(r["id"])
            else:
                out = m.forward(b)
            # the trunk returns residuals, not logits; the unembedding applies the final norm
            h_last = out.last_hidden_state[0, -1] if hasattr(out, "last_hidden_state") else out[0][0, -1]
            logits = m.unembed(h_last[None])[0]
            mg = margins(logits, r["kind"])
            # short only when the question itself asks for a bare yes/no or a letter;
            # a Gemini item whose reference answer happens to start with "no" is open-ended
            asks_short = r["kind"] == "letter" or (r["kind"] == "yesno" and vdata.YESNO_RE.search(r["question"] or ""))
            n_new = a.gen_short if asks_short else a.gen_open
            reply = sc.generate(b, max_new_tokens=n_new)
            probe_reply = None
            if r["meta"].get("probe"):
                b2 = m.chat(r["meta"]["probe"], [img], system=r["system"], cue=r["cue"], history=[(r["question"], reply)])
                probe_reply = sc.generate(b2, max_new_tokens=48)
        row = {k: v for k, v in r.items() if k != "image"}
        row.update(image=os.path.relpath(r["image"], vdata.ROOT), reply=reply, margins=mg, probe_reply=probe_reply)
        rows.append(row)
        if (i + 1) % 50 == 0 or i + 1 == len(recs):
            say(f"  {i + 1}/{len(recs)}  ({(time.time() - t0) / 60:.1f} min)  last: {reply[:60]!r}")
            json.dump(dict(repo=a.repo, family=fam, n=len(rows), rows=rows), open(out_json, "w"), indent=1, default=float)
    json.dump(dict(repo=a.repo, family=fam, n=len(rows), rows=rows), open(out_json, "w"), indent=1, default=float)
    if ANS:
        np.savez_compressed(out_npz, ids=np.array(ids), ANS=np.stack(ANS))
        say(f"  states: {len(ids)} x {ANS[0].shape} -> {out_npz}")
    say(f"  wrote {out_json}  ({(time.time() - t0) / 60:.1f} min)")
say("DONE d01")
