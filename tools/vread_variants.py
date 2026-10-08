# SPDX-License-Identifier: Apache-2.0
"""V-Read variants, chosen on development prompts by a label-free criterion only.

V-Read is a closed-form ridge map from each block's state to the last block's
state, fitted over every token of 900 unscored prompts, with the penalty and a
per-block gain chosen on 120 development prompts by mean all-token KL. Scored
readouts are taken at one place, the answer spot. This tool asks whether a few
cheap, principled changes to the fit make the map a better reader there, and
settles every choice on development data without labels and without the scored
records.

THE PROTOCOL (fixed in this file before any test AUROC is computed; the pick is
written to disk, with a timestamp and this file's digest, before the scored
records' states are read at all):

Data. The fitting corpus and the two held-out splits are exactly the ones
``vlens.lens.fitall.fit_all`` uses: ``corpus_items(spare, n=900+2*120, seed=0)``,
the first 900 fit, the next 120 development ("dev"), the last 120 held back
("hold"; ``fit_all`` draws these and never uses them). All 1140 are already off
limits for scoring. One pass of the model over the 900 accumulates the ridge
moments on every token, as the original does, and keeps the rows of the first
4 positions and the last 8 positions of every prompt, so that any reweighting
of those rows is exact and costs no second pass.

Criterion. ``KL(p_model || p_lens)`` over the full vocabulary AT THE ANSWER SPOT
of each dev prompt (the last prompt token, where the scored readouts are
taken), through the model's own final norm and unembedding; the model's own
final-block distribution is the reference, so no label is used. Averaged over
the 120 dev prompts, and over the 16 fitted blocks where a variant is ranked.
Reported beside it, never used to choose: top-1 agreement with the model's own
final-block prediction at the answer spot, the all-token dev KL (the original
criterion), and the same quantities on the hold split.

Within a variant (per block, as the original chooses per block): the ridge
penalty ``lambda`` (relative to the mean eigenvalue of the centred Gram, as in
``RidgeAccumulator.solve``) from 1e-10 to 1e-2 (the original grid bottoms out at
1e-8, where it picks almost every block), the shrinkage target (towards zero,
towards the identity, and in stage 3 towards the next fitted block's map), and
the gain ``h -> mu + g (A h + b - mu)`` from 1 to 5 (the original's grid ends at 3).

Stages (greedy, each on dev answer-spot KL):
  1. which tokens and which target: drop the first S in {0, 1, 4} positions
     (attention sinks carry outsized norms that dominate a squared loss), and
     regress onto the raw final state or onto it divided by its RMS (the final
     norm discards scale, so a token's size should not buy it weight).
  2. weight the answer spot: the stage-1 winner with the last K positions of
     each prompt up-weighted by w, (K, w) in {(1,1), (1,30), (1,300), (1,3000),
     (8,30), (8,300)}; 900 answer rows against ~206k tokens, so w = 300 gives
     the answer spots about half the weight.
  3. smooth across depth: the stage-2 winner with a third shrinkage target, the
     next fitted block's (pre-gain) map, the deepest block's being the identity
     of the last block; fitted from the deepest block down.
Choice between candidates. Each stage's winner, and the original cached V-Read,
are candidates; the one with the lowest answer-spot KL on the HOLD split is
used, so the split that chose the hyperparameters inside a stage is not the one
that chooses between stages. Only then are the scored records read.

    python3 tools/vread_variants.py fit --repo Qwen/Qwen3-VL-8B-Instruct
    python3 tools/vread_variants.py readouts --model Qwen3-VL-8B-Instruct   # CPU, from the saved maps
    python3 tools/vread_variants.py merge                                   # locally: adds reader v_read_v2

Outputs: ``results/vread_variants/<model>__search.json`` (every candidate's
numbers), ``<model>__pick.json`` (the choice, timestamped), ``<model>__v_read_v2.pt``
(the maps, half precision, the same ``{block: (A, b)}`` form as the cache), and
``results/readouts_v2/<model>__<family>_readouts.json`` with reader ``v_read_v2``
in the rows. ``merge`` copies that reader into the rows of the matching
``results/*readouts/`` files, leaving every other key as it was. Nothing here
writes to the shared map cache or changes how ``fit_all`` fits or keys a fit.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
import numpy as np  # noqa: E402

OUT = os.path.join(HERE, "results", "vread_variants")
READ_OUT = os.path.join(HERE, "results", "readouts_v2")
READER = "v_read_v2"
S_GRID = (0, 1, 4)
TARGETS = ("raw", "nrm")
TAIL_GRID = ((1, 1.0), (1, 30.0), (1, 300.0), (1, 3000.0), (8, 30.0), (8, 300.0))
LAM_GRID = tuple(10.0 ** k for k in range(-10, -1))
GAIN_GRID = (1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
S_MAX = max(S_GRID)
K_MAX = max(k for k, _ in TAIL_GRID)
#: where the states of scored records live, and the readout folders they belong with
STATE_DIRS = ("simonsays", "coco", "sandchan")
LOCKED = re.compile(r"wmdp[_ ]pw[_ ](fine|test)")


def _sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def say(*x):
    print(time.strftime("[%H:%M:%S]"), *x, flush=True)


# ------------------------------------------------------------------ the pass over the corpus

def _pos_stream_cls():
    import torch
    from vlens.lens import baselines as B
    from vlens.lens.hooks import ActivationRecorder

    class PosStream(B.StateStream):
        """``StateStream`` that also returns each kept token's position and its distance to the answer spot."""

        def _run(self, idx, at):
            m = self.model
            chunk = [self.items[i] for i in idx]
            b = m.batch_many(chunk, side="right")
            spot = m.answer_spot(b)
            with torch.no_grad(), ActivationRecorder(m.layers, at=at) as rec:
                m.forward(b)
                H = {l: rec.activations[l].detach() for l in at}
            n, T, _ = H[at[0]].shape
            dev = H[at[0]].device
            ar = torch.arange(T, device=dev)
            sp = spot.to(dev)
            keep = ar[None, :] <= sp[:, None]          # the same tokens StateStream keeps
            pos = ar[None, :].expand(n, T)[keep]
            off = (sp[:, None] - ar[None, :])[keep]
            return {l: H[l][keep] for l in at}, int(keep.sum()), pos, off
    return PosStream


class Moments:
    """Ridge moments on every token, plus the rows of the first and last few positions."""

    def __init__(self, layers, last, d, device):
        import torch
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        self.layers, self.last, self.d = list(layers), last, d
        self.G = {l: z(d, d) for l in layers}
        self.C = {t: {l: z(d, d) for l in layers} for t in TARGETS}
        self.sx = {l: z(d) for l in layers}
        self.sy = {t: z(d) for t in TARGETS}
        self.n = 0.0
        self.rows = {l: [] for l in [*layers, last]}
        self.pos, self.off = [], []
        self.min_spot = 10 ** 9

    @staticmethod
    def targets(Y):
        import torch
        Y = Y.double()
        return dict(raw=Y, nrm=Y / torch.sqrt((Y * Y).mean(-1, keepdim=True) + 1e-6))

    def add(self, H, n, pos, off):
        Y = self.targets(H[self.last])
        for t in TARGETS:
            self.sy[t] += Y[t].sum(0)
        for l in self.layers:
            X = H[l].double()
            self.G[l] += X.T @ X
            for t in TARGETS:
                self.C[t][l] += X.T @ Y[t]
            self.sx[l] += X.sum(0)
        self.n += n
        k = (pos < S_MAX) | (off < K_MAX)
        for l in self.rows:
            self.rows[l].append(H[l][k].to("cpu", copy=True))
        self.pos.append(pos[k].cpu()); self.off.append(off[k].cpu())
        self.min_spot = min(self.min_spot, int((pos[off == 0]).min()))

    def finish(self, device):
        import torch
        self.rows = {l: torch.cat(v).to(device) for l, v in self.rows.items()}
        self.pos, self.off = torch.cat(self.pos).to(device), torch.cat(self.off).to(device)
        assert self.min_spot >= S_MAX + K_MAX, f"a prompt of {self.min_spot + 1} tokens: sink and tail rows overlap"

    def omega(self, S, K, w):
        """Weight adjustment for each stored row: -1 drops a sink, w-1 up-weights a tail row."""
        import torch
        om = torch.zeros(self.pos.shape[0], dtype=torch.float64, device=self.pos.device)
        om[self.pos < S] = -1.0
        om[self.off < K] = w - 1.0
        return om

    def stats(self, l, S, K, w, tgt):
        """``(G, C, sx, sy, n)`` for block ``l`` under one data configuration."""
        om = self.omega(S, K, w)
        X = self.rows[l].double()
        Y = self.targets(self.rows[self.last])[tgt]
        Xw = X * om[:, None]
        return (self.G[l] + Xw.T @ X, self.C[tgt][l] + Xw.T @ Y, self.sx[l] + Xw.sum(0),
                self.sy[tgt] + (Y * om[:, None]).sum(0), self.n + float(om.sum()))


# ------------------------------------------------------------------ scoring a block

class Evaluator:
    """Answer-spot KL, and top-1 agreement, against the model's own final block."""

    def __init__(self, m, spots, last):
        import torch
        self.m = m
        self.X = {l: v.float() for l, v in spots.items()}        # {block: [N, d]}, dev rows then hold rows
        self.N = spots[last].shape[0]
        with torch.no_grad():
            self.ref = torch.log_softmax(m.unembed(spots[last]).float(), -1)
        self.top = self.ref.argmax(-1)

    def kl(self, pred, chunk=2048):
        """Per-row KL and agreement for ``pred`` [R, d] whose rows cycle through the N spots."""
        import torch
        out_kl, out_ag = [], []
        with torch.no_grad():
            for s in range(0, pred.shape[0], chunk):
                p = pred[s: s + chunk]
                idx = torch.arange(s, s + p.shape[0], device=p.device) % self.N
                lg = torch.log_softmax(self.m.unembed(p.to(torch.bfloat16)).float(), -1)
                r = self.ref[idx]
                out_kl.append((r.exp() * (r - lg)).sum(-1))
                out_ag.append((lg.argmax(-1) == self.top[idx]).float())
        return torch.cat(out_kl), torch.cat(out_ag)


def search_block(ev, l, st, refs, n_dev, lam_grid=LAM_GRID, gain_grid=GAIN_GRID):
    """Every (target, lambda, gain) for one block under one data configuration.

    Uses one eigendecomposition of the centred Gram, so every penalty and every
    shrinkage target costs a matrix product rather than a solve. Returns the
    option with the lowest mean dev answer-spot KL, its dev and hold numbers, and
    a closure that builds its ``(A, b)``.
    """
    import torch
    G, C, sx, sy, n = st
    mx, my = sx / n, sy / n
    Gc = G - n * torch.outer(mx, mx)
    Cc = C - n * torch.outer(mx, my)
    Gc = 0.5 * (Gc + Gc.T)
    Lam, V = torch.linalg.eigh(Gc)
    Lam = Lam.clamp_min(0.0)
    scale = float(Gc.diagonal().mean())
    Xc = ev.X[l].double() - mx
    P = Xc @ V
    Q = V.T @ Cc
    VtR = {}
    for name, R in refs.items():
        if name == "zero":
            VtR[name] = None
        elif name == "identity":
            VtR[name] = V.T
        else:
            VtR[name] = V.T @ R.double().T
    best, rows = None, []
    gains = torch.tensor(gain_grid, dtype=torch.float64, device=P.device)
    for rname, vr in VtR.items():
        for lam in lam_grid:
            le = lam * scale
            Z = P / (Lam + le)
            M = Q if vr is None else Q + le * vr
            pred = my + Z @ M                                    # [N, d]
            pg = my + gains[:, None, None] * (pred - my)[None]   # [g, N, d]
            k, a = ev.kl(pg.reshape(-1, pg.shape[-1]).float())
            k, a = k.view(len(gain_grid), -1), a.view(len(gain_grid), -1)
            dk, hk = k[:, :n_dev].mean(1), k[:, n_dev:].mean(1)
            da, ha = a[:, :n_dev].mean(1), a[:, n_dev:].mean(1)
            for gi, g in enumerate(gain_grid):
                r = dict(ref=rname, lam=lam, gain=g, dev_kl=float(dk[gi]), hold_kl=float(hk[gi]),
                         dev_agree=float(da[gi]), hold_agree=float(ha[gi]))
                rows.append(r)
                if np.isfinite(r["dev_kl"]) and (best is None or r["dev_kl"] < best["dev_kl"]):
                    best = r

    def build(b=best):
        le = b["lam"] * scale
        vr = VtR[b["ref"]]
        M = Q if vr is None else Q + le * vr
        At = V @ ((1.0 / (Lam + le))[:, None] * M)              # A^T, [d, d]
        A = At.T
        bb = my - A @ mx
        g = b["gain"]
        return (g * A).float(), (g * bb + (1.0 - g) * my).float(), A.float()
    return best, build


def run_config(ev, mom, layers, cfg, n_dev, log, smooth=False):
    """One data configuration over every block: the per-block picks and their means."""
    import torch
    S, K, w, tgt = cfg["S"], cfg["K"], cfg["w"], cfg["tgt"]
    t0 = time.time()
    picks, maps = {}, {}
    order = sorted(layers, reverse=True)
    nxt = torch.eye(mom.d, dtype=torch.float32, device=mom.G[layers[0]].device)
    for l in order:
        st = mom.stats(l, S, K, w, tgt)
        refs = {"zero": None, "identity": None}
        if smooth:
            refs["next"] = nxt
        best, build = search_block(ev, l, st, refs, n_dev)
        A, b, A_pre = build()
        picks[int(l)] = best
        maps[int(l)] = (A.cpu(), b.cpu())
        nxt = A_pre
        del st
    res = dict(cfg=cfg, smooth=smooth,
               dev_kl=float(np.mean([picks[l]["dev_kl"] for l in picks])),
               hold_kl=float(np.mean([picks[l]["hold_kl"] for l in picks])),
               dev_agree=float(np.mean([picks[l]["dev_agree"] for l in picks])),
               hold_agree=float(np.mean([picks[l]["hold_agree"] for l in picks])),
               per_block={str(l): picks[l] for l in sorted(picks)}, seconds=round(time.time() - t0))
    log(f"  S={S} K={K} w={w:g} target={tgt}{' +depth smoothing' if smooth else ''}: "
        f"dev answer-spot KL {res['dev_kl']:.4f} (hold {res['hold_kl']:.4f}), "
        f"agreement {res['dev_agree']:.3f} ({res['seconds']} s)")
    return res, maps


def eval_maps(ev, maps, layers, n_dev):
    """The same numbers for a fixed set of affine maps (the original V-Read)."""
    import torch
    per = {}
    for l in layers:
        A, b = maps[l]
        dev = ev.X[l].device
        pred = ev.X[l].float() @ A.T.to(dev).float() + b.to(dev).float()
        k, a = ev.kl(pred)
        per[str(l)] = dict(dev_kl=float(k[:n_dev].mean()), hold_kl=float(k[n_dev:].mean()),
                           dev_agree=float(a[:n_dev].mean()), hold_agree=float(a[n_dev:].mean()))
        del pred
    agg = {k: float(np.mean([p[k] for p in per.values()])) for k in
           ("dev_kl", "hold_kl", "dev_agree", "hold_agree")}
    return dict(agg, per_block=per)


def original_entry(repo, spare, cache):
    """The shared-fit cache entry for ``repo`` on this fit pool (not the COCO fit)."""
    ids = ",".join(sorted(str(r.get("id")) for r in spare))
    dig = hashlib.sha1(ids.encode()).hexdigest()
    for s in sorted(glob.glob(os.path.join(cache, "*.pt.spec"))):
        try:
            spec = json.loads(open(s).read())
        except Exception:  # noqa: BLE001
            continue
        if spec[0] == repo and spec[1] == dig and spec[3] == 900 and os.path.exists(s[:-5]):
            return s[:-5], spec
    return None, None


# ------------------------------------------------------------------ fit

def fit(a):
    import torch
    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays as SS
    from vlens.lens import baselines as B
    from vlens.lens import fitall as FA
    from vlens.lens import room

    model = a.repo.split("/")[-1]
    os.makedirs(OUT, exist_ok=True)
    need = int(room.weights_mib(a.repo) * 1.3)
    room.wait_for_free(need, minutes=60, log=say)
    m = vlens.load_vlm(a.repo)
    LAST = m.n_layers - 1
    layers = FA.layer_grid(m, 16)
    spare = FA.fitting_pool(SS.load(require_image=True), 120)
    entry, spec = original_entry(a.repo, spare, FA.CACHE)
    if entry is None:
        raise SystemExit(f"no shared V-Read fit for {a.repo} on this pool in {FA.CACHE}")
    assert list(spec[2]) == list(layers), (spec[2], layers)
    say(f"{model}: blocks {layers}; original V-Read from {entry}")
    corpus, held = 900, 120
    items = B.corpus_items(spare, C.load_image, n=corpus + 2 * held, square=448, seed=0)
    fit_items, dev_items, hold_items = items[:corpus], items[corpus: corpus + held], items[corpus + held:]
    fit_ids = B.fit_ids(fit_items, dev_items, hold_items)
    at = sorted({*layers, LAST})
    PosStream = _pos_stream_cls()

    # ---- one pass over the fitting corpus
    t0 = time.time()
    mom = Moments(layers, LAST, m.d_model, m.device)
    for H, n, pos, off in B._stream_batches(PosStream(m, fit_items, batch=8, seed=1), at):
        mom.add(H, n, pos, off)
        del H
    mom.finish(m.device)
    say(f"fitting corpus: {int(mom.n)} tokens, {mom.rows[LAST].shape[0]} sink/tail rows kept "
        f"({(time.time() - t0) / 60:.1f} min)")

    # ---- the development and hold prompts: answer-spot rows; full dev states for the all-token KL
    spots = {l: [] for l in at}
    dev_full, n_dev = [], 0
    for split, its, seed in (("dev", dev_items, 3), ("hold", hold_items, 5)):
        for H, n, pos, off in B._stream_batches(PosStream(m, its, batch=8, seed=seed), at):
            k = off == 0
            for l in at:
                spots[l].append(H[l][k].clone())
            if split == "dev":
                n_dev += int(k.sum())
                dev_full.append(({l: H[l].to("cpu", copy=True) for l in at}, n))
            del H
    spots = {l: torch.cat(v) for l, v in spots.items()}
    assert n_dev == len(dev_items) and spots[LAST].shape[0] == len(dev_items) + len(hold_items)
    ev = Evaluator(m, spots, LAST)
    say(f"answer spots: {n_dev} dev, {spots[LAST].shape[0] - n_dev} hold")

    class Replay:
        def __iter__(self):
            for H, n in dev_full:
                yield {l: v.to(m.device) for l, v in H.items()}, n

    # ---- the original, as cached
    omaps, _lat, ofit_ids, ometa = FA._load(entry)
    orig = {int(l): omaps["v_read"][l] for l in layers}
    assert fit_ids <= ofit_ids, "the corpus drawn here is not the one the original was fitted on"
    res_orig = eval_maps(ev, orig, layers, n_dev)
    say(f"original V-Read: dev answer-spot KL {res_orig['dev_kl']:.4f} "
        f"(hold {res_orig['hold_kl']:.4f}), agreement {res_orig['dev_agree']:.3f}")

    search = dict(model=model, repo=a.repo, blocks=layers, protocol=__doc__.split("THE PROTOCOL", 1)[1].split("    python3")[0],
                  script_sha1=_sha(os.path.abspath(__file__)), original_entry=os.path.basename(entry),
                  original_meta=ometa.get("v_read"), grids=dict(S=S_GRID, targets=TARGETS, tail=TAIL_GRID,
                                                               lam=LAM_GRID, gain=GAIN_GRID),
                  n_dev=n_dev, n_hold=int(spots[LAST].shape[0] - n_dev), tokens=int(mom.n),
                  original=res_orig, stage1=[], stage2=[], stage3=None)

    def dump():
        json.dump(search, open(os.path.join(OUT, f"{model}__search.json"), "w"), indent=1, default=float)

    # ---- stage 1: tokens and target
    say("stage 1: which tokens, which target")
    s1 = {}
    for S in S_GRID:
        for tgt in TARGETS:
            cfg = dict(S=S, K=1, w=1.0, tgt=tgt)
            r, mp = run_config(ev, mom, layers, cfg, n_dev, say)
            search["stage1"].append(r); s1[(S, tgt)] = (r, mp); dump()
    w1 = min(s1.values(), key=lambda x: x[0]["dev_kl"])
    say(f"  stage 1 winner: {w1[0]['cfg']}")
    # ---- stage 2: answer-spot weight
    say("stage 2: weight on the last positions")
    s2 = [w1]
    for K, w in TAIL_GRID:
        if w == 1.0:
            continue
        cfg = dict(w1[0]["cfg"], K=K, w=w)
        r, mp = run_config(ev, mom, layers, cfg, n_dev, say)
        search["stage2"].append(r); s2.append((r, mp)); dump()
    w2 = min(s2, key=lambda x: x[0]["dev_kl"])
    say(f"  stage 2 winner: {w2[0]['cfg']}")
    # ---- stage 3: depth smoothing
    say("stage 3: shrinkage towards the next block's map")
    r3, m3 = run_config(ev, mom, layers, w2[0]["cfg"], n_dev, say, smooth=True)
    search["stage3"] = r3; dump()

    # ---- the choice between candidates, on the hold split
    cands = {"original": (res_orig, orig), "stage1": w1, "stage2": w2, "stage3": (r3, m3)}
    chosen = min(cands, key=lambda k: cands[k][0]["hold_kl"])
    say("candidates (hold answer-spot KL): "
        + ", ".join(f"{k} {v[0]['hold_kl']:.4f}" for k, v in cands.items()) + f" -> {chosen}")
    maps = cands[chosen][1]

    # ---- the all-token dev KL, the original criterion, reported only
    with torch.no_grad():
        gpu = lambda mp: {l: (mp[l][0].to(m.device), mp[l][1].to(m.device)) for l in layers}
        kl_all, _ = B.heldout_kl(m, Replay(), layers, {k: gpu(v[1]) for k, v in cands.items()})
    all_tok = {k: float(np.mean([kl_all[k][l] for l in layers])) for k in kl_all}
    say("all-token dev KL: " + ", ".join(f"{k} {v:.4f}" for k, v in all_tok.items()))
    search["all_token_dev_kl"] = all_tok
    dump()

    pick = dict(model=model, when=datetime.datetime.now().isoformat(timespec="seconds"),
                script_sha1=search["script_sha1"], rule="lowest hold answer-spot KL among "
                "{original, stage1, stage2, stage3}; within a stage, lowest dev answer-spot KL",
                chosen=chosen, cfg=cands[chosen][0].get("cfg"), smooth=cands[chosen][0].get("smooth"),
                per_block=cands[chosen][0]["per_block"],
                candidates={k: {q: v[0][q] for q in ("dev_kl", "hold_kl", "dev_agree", "hold_agree")}
                            | dict(cfg=v[0].get("cfg"), all_token_dev_kl=all_tok.get(k))
                            for k, v in cands.items()},
                test_records_read=False)
    pj = os.path.join(OUT, f"{model}__pick.json")
    json.dump(pick, open(pj, "w"), indent=1, default=float)
    say(f"PICK WRITTEN {pj} at {pick['when']}: {chosen} {pick['cfg']}")
    try:   # the disk is shared and nearly full; the readouts below do not depend on this file
        torch.save(dict(maps={int(l): (maps[l][0].half(), maps[l][1].half()) for l in layers},
                        fit_ids=sorted(ofit_ids), pick=pick),
                   os.path.join(OUT, f"{model}__v_read_v2.pt"))
    except Exception as e:  # noqa: BLE001
        say(f"could not save the maps ({type(e).__name__}: {e}); carrying on")
        try:
            os.remove(os.path.join(OUT, f"{model}__v_read_v2.pt"))
        except OSError:
            pass
    del m, mom, ev
    torch.cuda.empty_cache()
    a.model = model
    readouts(a, maps={int(l): (maps[l][0].double().numpy(), maps[l][1].double().numpy()) for l in layers},
             fit_ids={str(i) for i in ofit_ids})
    say("DONE vread_variants")


# ------------------------------------------------------------------ readouts

def readouts(a, maps=None, fit_ids=None):
    """The chosen maps applied to every scored record's stored answer-spot states."""
    import layer_readouts as LR
    import torch
    model = a.model
    if maps is None:
        d = torch.load(os.path.join(OUT, f"{model}__v_read_v2.pt"), map_location="cpu")
        maps = {int(l): (A.double().numpy(), b.double().numpy()) for l, (A, b) in d["maps"].items()}
        fit_ids = {str(i) for i in d["fit_ids"]}
    pick = json.load(open(os.path.join(OUT, f"{model}__pick.json")))
    say(f"{model}: reading the scored records through the pick made at {pick['when']} ({pick['chosen']})")
    hp = os.path.join(HERE, "results", "readouts", f"{model}__head.npz")
    hd = LR.Head(hp)
    os.makedirs(READ_OUT, exist_ok=True)
    for sd in STATE_DIRS:
        for pn in sorted(glob.glob(os.path.join(HERE, "results", sd, f"{model}__*_states.npz"))):
            fam = os.path.basename(pn).split("__", 1)[1][: -len("_states.npz")]
            pj = pn[: -len("_states.npz")] + ".json"
            if LOCKED.search(fam) or not os.path.exists(pj):
                continue
            rows = {r["id"]: r for r in json.load(open(pj))["rows"]}
            z = np.load(pn)
            ids = [str(i) for i in z["ids"]]
            ANS = z["ANS"]
            blocks = [int(b) for b in z["layers"]] if "layers" in z.files else list(range(ANS.shape[1]))
            out = []
            # every record at once per block: the same arithmetic as layer_readouts, batched
            moved = {bi: ANS[:, bi].astype(np.float64) @ maps[b][0].T + maps[b][1]
                     for bi, b in enumerate(blocks) if b in maps}
            for j, rid in enumerate(ids):
                r = rows.get(rid)
                if r is None or r.get("kind") not in ("yesno", "test_real", "letter"):
                    continue
                key = "letters" if r["kind"] == "letter" else "margin"
                v = [None] * len(blocks)
                for bi, T in moved.items():
                    v[bi] = np.round(LR._readout(hd, hd.logits(T[j]), r["kind"])[key], 3).tolist()
                out.append({"id": rid, "kind": r["kind"], READER: v, "v_read_fitted_on": rid in fit_ids})
            p = os.path.join(READ_OUT, f"{model}__{fam}_readouts.json")
            json.dump(dict(model=model, family=fam, states_dir=sd, blocks=blocks, reader=READER,
                           pick_when=pick["when"], pick=pick["chosen"], rows=out), open(p, "w"))
            say(f"  {sd}/{fam}: {len(out)} records -> {p}")


def merge(a):
    """Add reader ``v_read_v2`` to the rows of the matching readout files, by record id."""
    sys.path.insert(0, HERE)
    from vlens.figures import fx_detect as D
    res = os.path.join(HERE, "results")
    for p in sorted(glob.glob(os.path.join(READ_OUT, "*_readouts.json"))):
        v2 = json.load(open(p))
        name = os.path.basename(p)
        tgt = next((os.path.join(res, d, name) for d in D.READOUT_DIRS
                    if os.path.exists(os.path.join(res, d, name))), None)
        if tgt is None:
            print(f"  {name}: no readout file to merge into")
            continue
        d = json.load(open(tgt))
        if [int(b) for b in d["blocks"]] != [int(b) for b in v2["blocks"]]:
            print(f"  {name}: blocks differ, skipped")
            continue
        got = {r["id"]: r[READER] for r in v2["rows"]}
        n = 0
        for r in d["rows"]:
            if r["id"] in got:
                r[READER] = got[r["id"]]; n += 1
        d["v_read_v2"] = dict(pick_when=v2["pick_when"], pick=v2["pick"])
        json.dump(d, open(tgt, "w"))
        print(f"  {os.path.relpath(tgt, HERE)}: {n}/{len(d['rows'])} rows given {READER}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--repo", required=True)
    r = sub.add_parser("readouts")
    r.add_argument("--model", required=True)
    sub.add_parser("merge")
    a = ap.parse_args()
    {"fit": fit, "readouts": readouts, "merge": merge}[a.cmd](a)


if __name__ == "__main__":
    main()
