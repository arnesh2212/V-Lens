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

from layer_readouts import LETTERS, LOCKED

IDENTITY = "cat->cat; 1135->1135; hello->hello; ?"
TARGETS = (("patchscope", "chat"), ("patchscope_raw", "raw"))
SETS = ("yes", "no", "test", "real", *LETTERS)
SHORT = ("yesno", "test_real", "letter")


def target(m, form):
    if form == "raw":
        return m.build(IDENTITY), IDENTITY
    text = m.chat_prompt(IDENTITY)
    return m.build(text), text


def set_masses(lg, ts):
    lp = torch.log_softmax(lg.float(), -1)
    ids = dict(yes=ts.yes, no=ts.no, test=ts.test, real=ts.real, **ts.letters)
    return torch.stack([torch.logsumexp(lp[:, ids[k]], -1) for k in SETS], -1)


def reading(M, kind):
    M = np.asarray(M, np.float64)
    if kind == "yesno":
        return float(M[0] - M[1])
    if kind == "test_real":
        return float(M[2] - M[3])
    L = M[4:8]
    L = L - (L.max() + np.log(np.exp(L - L.max()).sum()))
    return L.tolist()


def inject(m, batch, rows_block, H, last=None):
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
        with torch.no_grad(), ActivationRecorder(m.layers, at=[last]) as rec:
            m.forward(big)
            return m.unembed(rec.activations[last][:, S - 1])
    finally:
        for h in handles:
            h.remove()


def read_many(m, batch, ts, rows_block, H, bs=256, log=None, fn=None):
    from vlens.lens.room import is_oom, with_room
    fn = fn or (lambda lg: set_masses(lg, ts))
    out, i, bs = [], 0, max(int(bs), 1)
    while i < len(H):
        run = lambda: fn(inject(m, batch, rows_block[i: i + bs],
                                H[i: i + bs].to(m.device))).float().cpu()
        try:
            got = run() if bs > 1 else with_room(run, log=log)
        except Exception as e:
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
