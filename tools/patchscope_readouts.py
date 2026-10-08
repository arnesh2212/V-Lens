# SPDX-License-Identifier: Apache-2.0
"""Patchscopes at the answer spot: every stored state, decoded by the model itself.

Patchscopes (Ghandeharioun et al. 2024) reads a hidden state by handing it to
the model in a prompt built to make the model say what the state holds. For
the next token a state predicts, the paper's target prompt is a few-shot
token-identity prompt, ``cat->cat; 1135->1135; hello->hello; ?``: the state
of the source position at block ``l`` replaces the state of the target's last
position at the same block ``l``, the model runs on from there, and the
output at that position is the reading. It is the paper's own configuration
for next-token decoding (source and target block equal, target position the
last one), the one it sets against the logit lens and the tuned lens.

Here the source position is the answer spot of every short-answer record and
the source states are the ones the behaviour pass stored after every block,
so no picture is run again. The reading is the same quantity
:mod:`tools.layer_readouts` gives the logit lens and the fitted lenses:
``log p(yes) - log p(no)``, ``log p(test) - log p(real)``, or the log-softmax
over the four letters, over the token sets of
:class:`vlens.lens.scores.TokenSets`, through the model's own final norm,
unembedding and softcap. Two target prompts are read, both text only:

* ``patchscope``, the demonstrations in the model's own chat template, as the
  user's turn, with reasoning off where the template offers it; the last
  position is where the reply begins, the same kind of position the answer
  spot is;
* ``patchscope_raw``, the paper's string as plain text in the model's own
  tokenisation (with its beginning-of-text token where it uses one); the last
  position is the ``?``.

At the final block the injected state meets only the final norm and the
unembedding, so there the reading is the logit lens exactly; the run checks
this against the readout file for the same family, which tests that the
state went where it should.

Every (record, block) pair is one row of a batch. A hook on each block writes
the rows assigned to that block and leaves the rest alone, so one forward
serves every block at once: a row patched at block ``l`` carries the target's
own states below ``l`` and the injected one from ``l`` on, and rows never mix.
Nothing is model-specific beyond the :mod:`vlens` wrapper: the blocks are its
``layers``, the reading is its ``unembed``, and a block's output may be a
tensor or a tuple, as in :class:`vlens.lens.hooks.Patch`.

    HF_HOME=/workspace/temp/vlm/hf python3 tools/patchscope_readouts.py --repo Qwen/Qwen3-VL-8B-Instruct
    python3 tools/patchscope_readouts.py --repo Qwen/Qwen3.5-27B --results /dev/shm/vlens_results/simonsays_full

writes ``results/readouts/<model>__<family>_patchscope.json`` with rows
``{id, kind, patchscope: [per block], patchscope_raw: [per block]}``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

from layer_readouts import LETTERS, LOCKED  # noqa: E402

#: the paper's token-identity target prompt; the placeholder is the last token
IDENTITY = "cat->cat; 1135->1135; hello->hello; ?"
#: each reading and the form its target prompt takes
TARGETS = (("patchscope", "chat"), ("patchscope_raw", "raw"))
SETS = ("yes", "no", "test", "real", *LETTERS)
SHORT = ("yesno", "test_real", "letter")


def target(m, form):
    """The target prompt as a batch of one, and its text."""
    if form == "raw":
        return m.build(IDENTITY), IDENTITY
    text = m.chat_prompt(IDENTITY)
    return m.build(text), text


def set_masses(lg, ts):
    """``[n, V]`` logits -> ``[n, 8]`` log-mass on yes, no, test, real, A, B, C, D."""
    lp = torch.log_softmax(lg.float(), -1)
    ids = dict(yes=ts.yes, no=ts.no, test=ts.test, real=ts.real, **ts.letters)
    return torch.stack([torch.logsumexp(lp[:, ids[k]], -1) for k in SETS], -1)


def reading(M, kind):
    """One row of set masses -> the readout :mod:`tools.layer_readouts` records for ``kind``."""
    M = np.asarray(M, np.float64)
    if kind == "yesno":
        return float(M[0] - M[1])
    if kind == "test_real":
        return float(M[2] - M[3])
    L = M[4:8]
    L = L - (L.max() + np.log(np.exp(L - L.max()).sum()))
    return L.tolist()


def inject(m, batch, rows_block, H, last=None):
    """The head's logits [n, V] at the target's last position, each row's state injected at its own block.

    ``batch`` is the target prompt as a batch of one; ``rows_block`` [n] the
    block each row is patched at; ``H`` [n, d] the states.
    """
    last = m.n_layers - 1 if last is None else last
    n = int(H.shape[0])
    S = batch["input_ids"].shape[1]
    big = m.replicate(batch, n) if n > 1 else batch
    rb = torch.as_tensor(rows_block, dtype=torch.long)
    handles = []

    def make(idx, val):
        def hook(mod, inp, out, _i=idx, _v=val):
            tup = not torch.is_tensor(out)
            hh = (out[0] if tup else out).clone()
            hh[_i.to(hh.device), S - 1] = _v.to(hh.device, hh.dtype)
            return (hh, *out[1:]) if tup else hh
        return hook

    from vlens.lens.hooks import ActivationRecorder
    try:
        for b in sorted(set(rb.tolist())):
            idx = torch.nonzero(rb == b).flatten()
            handles.append(m.layers[b].register_forward_hook(make(idx, H[idx])))
        # the recorder is registered after the patches, so at the last block it sees the patched state
        with torch.no_grad(), ActivationRecorder(m.layers, at=[last]) as rec:
            m.forward(big)
            return m.unembed(rec.activations[last][:, S - 1])
    finally:
        for h in handles:
            h.remove()


def read_many(m, batch, ts, rows_block, H, bs=256, log=None, fn=None):
    """:func:`inject` over any number of rows, in batches that halve when the card is full.

    ``fn`` maps the head's logits [n, V] to the reading; by default the set
    masses of :func:`set_masses` over ``ts`` (``experiments/safety/refusal.py``
    passes its refusal margin instead).
    """
    from vlens.lens.room import is_oom, with_room
    fn = fn or (lambda lg: set_masses(lg, ts))
    out, i, bs = [], 0, max(int(bs), 1)
    while i < len(H):
        run = lambda: fn(inject(m, batch, rows_block[i: i + bs],
                                H[i: i + bs].to(m.device))).float().cpu()
        try:
            got = run() if bs > 1 else with_room(run, log=log)
        except Exception as e:  # noqa: BLE001
            if not is_oom(e) or bs <= 1:
                raise
            bs = max(bs // 2, 1)
            torch.cuda.empty_cache()
            if log:
                log(f"    not enough room; {bs} rows at a time")
            continue
        out.append(got.numpy())
        i += bs
    return np.concatenate(out) if out else np.zeros((0, len(SETS)), np.float32)


def _check_sets(ts, head_path, say):
    """Warn when the token sets differ from the ones the logit-lens readouts used."""
    if not os.path.exists(head_path):
        return
    sets = json.loads(str(np.load(head_path)["sets"]))
    ours = dict(yes=ts.yes, no=ts.no, test=ts.test, real=ts.real, **ts.letters)
    bad = [k for k in ours if sorted(ours[k]) != sorted(sets.get(k, []))]
    if bad:
        say(f"!! token sets differ from {os.path.basename(head_path)} for {bad}")


def run(a):
    import vlens
    from vlens.lens.scores import TokenSets
    say = lambda *x: print(*x, flush=True)
    name = a.repo.split("/")[-1]
    fams = a.families or sorted(
        os.path.basename(p).split("__", 1)[1][: -len("_states.npz")]
        for p in glob.glob(os.path.join(a.results, f"{name}__*_states.npz")))
    fams = [f for f in fams if not LOCKED.search(f)]
    if not fams:
        say(f"no stored answer-spot states for {name} under {a.results}")
        return
    m = vlens.load_vlm(a.repo)
    ts = TokenSets(m.tokenizer)
    _check_sets(ts, os.path.join(a.out, f"{name}__head.npz"), say)
    targets = {key: target(m, form) for key, form in TARGETS if key in a.readings}
    for key, (b, text) in targets.items():
        say(f"{name}: {key}, {b['input_ids'].shape[1]} target tokens")
    agree = {k: [] for k in targets}
    exact = {k: 0.0 for k in targets}
    os.makedirs(a.out, exist_ok=True)
    for fam in fams:
        pj = os.path.join(a.results, f"{name}__{fam}.json")
        pn = os.path.join(a.results, f"{name}__{fam}_states.npz")
        if not (os.path.exists(pj) and os.path.exists(pn)):
            continue
        t0 = time.time()
        rows = {r["id"]: r for r in json.load(open(pj))["rows"]}
        z = np.load(pn)
        ids = [str(i) for i in z["ids"]]
        ANS = z["ANS"]
        blocks = [int(b) for b in z["layers"]] if "layers" in z.files else list(range(ANS.shape[1]))
        keep = [j for j, rid in enumerate(ids)
                if rid in rows and rows[rid].get("kind") in SHORT][: a.limit or None]
        if not keep:
            continue
        kinds = [rows[ids[j]]["kind"] for j in keep]
        # one row per (record, block), the record varying fastest within a block
        rb = np.repeat(np.asarray(blocks), len(keep))
        H = torch.as_tensor(np.asarray(ANS[keep], np.float32)).transpose(0, 1).reshape(-1, ANS.shape[-1])
        got = {}
        for key, (b, _t) in targets.items():
            M = read_many(m, b, ts, rb, H.to(m.dtype), bs=a.batch, log=say)
            got[key] = M.reshape(len(blocks), len(keep), len(SETS))
        out = []
        for i, j in enumerate(keep):
            rec = dict(id=ids[j], kind=kinds[i])
            for key, M in got.items():
                rec[key] = [np.round(reading(M[bi, i], kinds[i]), 3).tolist() for bi in range(len(blocks))]
            out.append(rec)
        # at the last block the reading is the head applied to the state, whatever the target:
        # exactly, against the same head in the same precision, and to rounding against the
        # float64 logit-lens readout (for letters, on the letter that readout ranks first)
        if blocks[-1] == m.n_layers - 1:
            with torch.no_grad():
                ref = set_masses(m.unembed(H[-len(keep):].to(m.device, m.dtype)), ts).cpu().numpy()
            for key in got:
                exact[key] = max(exact[key], float(np.abs(got[key][-1] - ref).max()))
            rp = os.path.join(a.out, f"{name}__{fam}_readouts.json")
            lens = {r["id"]: r["logit"][-1] for r in json.load(open(rp))["rows"]} if os.path.exists(rp) else {}
            for rec in out:
                if rec["id"] in lens:
                    lv = np.asarray(lens[rec["id"]], float)
                    for key in got:
                        pv = np.asarray(rec[key][-1], float)
                        if lv.ndim:
                            top = int(np.argmax(lv))
                            pv, lv_ = pv[top], lv[top]
                        else:
                            lv_ = lv
                        agree[key].append(float(abs(pv - lv_)))
        payload = dict(model=name, family=fam, readout="+".join(targets), blocks=blocks, n_layers=m.n_layers,
                       method=("Patchscopes (Ghandeharioun et al. 2024), token-identity target prompt: "
                               "the answer-spot state of block l replaces the target's last position at "
                               "block l, and the model runs on"),
                       targets={k: dict(form=f, text=targets[k][1],
                                        tokens=int(targets[k][0]["input_ids"].shape[1]))
                                for k, f in TARGETS if k in targets},
                       quantity=dict(yesno="log p(yes) - log p(no)", test_real="log p(test) - log p(real)",
                                     letter="log-softmax over A, B, C, D"),
                       rows=out)
        p = os.path.join(a.out, f"{name}__{fam}_patchscope.json")
        json.dump(payload, open(p, "w"))
        say(f"  {fam}: {len(out)} records x {len(blocks)} blocks, {time.time() - t0:.0f}s -> {p}")
        del ANS, z, H
    for key, v in exact.items():
        say(f"{key} at the last block against the head on the stored state: max |diff| {v:.2e}")
    for key, v in agree.items():
        if v:
            v = np.asarray(v)
            say(f"{key} at the last block against the logit lens: median |diff| {np.median(v):.3f}, "
                f"95th percentile {np.percentile(v, 95):.3f} over {v.size} records")
    say("DONE patchscope_readouts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--results", default=os.path.join(HERE, "results", "simonsays"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "readouts"))
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--readings", nargs="*", default=[k for k, _ in TARGETS])
    ap.add_argument("--batch", type=int, default=256, help="(record, block) rows per forward")
    ap.add_argument("--limit", type=int, default=0, help="records per family, for a smoke test; 0 for all")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
