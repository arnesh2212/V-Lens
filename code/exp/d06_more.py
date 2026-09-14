# SPDX-License-Identifier: Apache-2.0
"""More of the lens on the custom datasets: steering, awareness steering, word strips, map transfer.

  S  steering the operative answer.  V-Write's yes-over-no direction at the answer
     spot, against a labelled difference of means and a random direction of the
     same size, at three doses, on the nuScenes edits and the Open Images signs.
     Flip rate, coherence, and collateral on an unrelated question about the same photo.
  W  steering awareness.  Subtract the test-over-real direction on the frames
     the model calls a test image, add it on frames it calls real; report flips.
  T  word strips.  What V-Read says the model is about to say at each depth on
     the conflict pairs, next to the logit lens, from the cached states.
  X  map transfer.  A V-Read map fitted on Open Images states read on the
     nuScenes edits, against the within-family map and the logit lens.

  python3 code/exp/d06_more.py --repo Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--results", default="/workspace/vlm/results/vlens")
ap.add_argument("--parts", nargs="+", default=["S", "W", "T", "X"])
ap.add_argument("--depths", type=float, nargs="+", default=[0.35, 0.5, 0.65])
ap.add_argument("--doses", type=float, nargs="+", default=[0.1, 0.2, 0.4])
ap.add_argument("--n-fit", type=int, default=40)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

import jvlm  # noqa: E402
from jvlm import axes, corpus, depth, reader, readout, tasks, vdata, verdict  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

say = lambda *x: print(*x, flush=True)
name = a.repo.split("/")[-1]
m = jvlm.load_vlm(a.repo)
sc = tasks.Scorer(m)
tok = m.tokenizer
LAST = m.n_layers - 1
ALLL = list(range(max(m.n_deepstack, 1), m.n_layers - 1))
frac = {l: depth.to_frac(m, l) for l in range(m.n_layers)}
L3 = [min(ALLL, key=lambda l: abs(frac[l] - d)) for d in a.depths]
YN = re.compile(r"\b(yes|no)\b", re.I); TR = re.compile(r"\b(test|real)\b", re.I)
say(f"{name}: steering depths {[round(frac[l], 2) for l in L3]}")


def rows(fam):
    p = f"{a.results}/{name}__{fam}.json"
    return json.load(open(p))["rows"] if os.path.exists(p) else []


def batch_for(r):
    img = corpus.load_image(os.path.join(vdata.ROOT, r["image"]), square=448)
    return m.chat(r["question"], [img], system=r.get("system"), cue=r.get("cue")), img


def yn_axis():
    return axes.yesno(tok)


def tr_axis():
    return axes.token_axis("testreal", tok, ["test", " test", "Test", " Test"], ["real", " real", "Real", " Real"])


def fit_direction(items, axis, layers):
    """V-Write direction at the answer spot: mean gradient of the axis over prompts."""
    bs = (batch_for(r)[0] for r in items)
    F = verdict.fit(m, bs, [axis], layers, every=1000)
    return {l: F.direction(axis.name, "ans", l, unit=True) for l in layers}


def dim_direction(fam_rows, layers, label_fn):
    """Labelled difference of means at the answer spot from the cached states, unit norm."""
    z = np.load(f"{a.results}/{name}__{fam_rows[0]['family']}_states.npz"); ids = [str(x) for x in z["ids"]]; ANS = z["ANS"].astype(np.float32)
    by = {r["id"]: r for r in fam_rows}
    out = {}
    for l in layers:
        pos, neg = [], []
        for i, k in enumerate(ids):
            r = by.get(k)
            if r is None:
                continue
            lab = label_fn(r)
            (pos if lab == 1 else neg if lab == 0 else []).append(ANS[i, l])
        d = torch.as_tensor(np.mean(pos, 0) - np.mean(neg, 0)); out[l] = d / d.norm()
    return out


class AnswerSpotAdd:
    """Add alpha * ||h[a]|| * u at the answer spot at one layer during the prefill."""
    def __init__(self, layer, u, alpha):
        self.layer, self.u, self.alpha, self.h = layer, u, alpha, None
    def __enter__(self):
        S_holder = {}
        def hook(mod, inp, o, _u=self.u, _al=self.alpha):
            tup = not torch.is_tensor(o); hh = o[0] if tup else o
            if hh.shape[1] < 2:      # a cached decode step: leave it alone
                return o
            hh = hh.clone()
            ha = hh[:, -1]
            hh[:, -1] = ha + _al * ha.norm(dim=-1, keepdim=True) * _u.to(hh.device, hh.dtype)
            return (hh, *o[1:]) if tup else hh
        self.h = m.layers[self.layer].register_forward_hook(hook); return self
    def __exit__(self, *exc):
        self.h.remove()


def first_answer(reply, rx):
    mm = rx.search(reply or "")
    return mm.group(1).lower() if mm else None


# ============================================================ S: steering the operative answer
def part_s():
    ne = [r for r in rows("nusc_edit") if r["kind"] == "yesno" and r["condition"] == "operative"]
    oi = [r for r in rows("oi_orig") if r["kind"] == "yesno" and r["condition"] in ("operative", "operative_generic")]
    fit_items = oi[: a.n_fit]; test_items = ne + oi[a.n_fit: a.n_fit + 60]
    if a.limit:
        test_items = test_items[: a.limit]
    say(f"[S] {len(test_items)} test items ({len(ne)} nuScenes edits), direction fitted on {len(fit_items)} Open Images prompts")
    ax = yn_axis()
    U = fit_direction(fit_items, ax, L3)
    D = dim_direction(oi, L3, lambda r: {"yes": 1, "no": 0}.get((r.get("judge") or {}).get("parsed") or first_answer(r.get("reply"), YN), None))
    R = {l: (lambda g: g / g.norm())(torch.randn_like(U[l])) for l in L3}
    for l in L3:
        say(f"   depth {frac[l]:.2f}: cos(V-Write, DIM) = {float(torch.dot(U[l].float(), D[l].float())):+.3f}")
    UNREL = "Is this picture taken outdoors? Answer yes or no."
    out = []
    t0 = time.time()
    for k, r in enumerate(test_items):
        b, img = batch_for(r)
        with torch.no_grad():
            base = first_answer(sc.generate(b, max_new_tokens=4), YN)
            b_u = m.chat(UNREL, [img]); base_u = first_answer(sc.generate(b_u, max_new_tokens=4), YN)
        rec = dict(id=r["id"], family=r["family"], variant=r["meta"].get("variant"), expected=r.get("expected"), base=base, base_unrelated=base_u, runs=[])
        if base is None:
            out.append(rec); continue
        target_sign = +1 if base == "no" else -1      # push toward the answer it did not give
        for l in L3:
            for dname, dirs in (("V-Write", U), ("DIM (labelled)", D), ("random", R)):
                for al in a.doses:
                    with torch.no_grad(), AnswerSpotAdd(l, dirs[l] * target_sign, al):
                        rep = sc.generate(b, max_new_tokens=4)
                    v = first_answer(rep, YN)
                    run = dict(layer=l, frac=frac[l], direction=dname, alpha=al, answer=v, flipped=(v is not None and v != base), coherent=(v is not None))
                    if dname == "V-Write" and al == a.doses[1]:
                        with torch.no_grad(), AnswerSpotAdd(l, dirs[l] * target_sign, al):
                            vu = first_answer(sc.generate(b_u, max_new_tokens=4), YN)
                        run["unrelated_kept"] = (vu == base_u)
                    rec["runs"].append(run)
        out.append(rec)
        if (k + 1) % 10 == 0:
            say(f"   {k + 1}/{len(test_items)} ({(time.time() - t0) / 60:.1f} min)")
    # summary
    summ = {}
    for l in L3:
        for dname in ("V-Write", "DIM (labelled)", "random"):
            for al in a.doses:
                rs = [x for r in out for x in r["runs"] if x["layer"] == l and x["direction"] == dname and x["alpha"] == al]
                if rs:
                    summ[f"{frac[l]:.2f}|{dname}|{al}"] = dict(flip=float(np.mean([x["flipped"] for x in rs])), coherent=float(np.mean([x["coherent"] for x in rs])), n=len(rs),
                                                         unrelated_kept=(float(np.mean([x["unrelated_kept"] for x in rs if "unrelated_kept" in x])) if any("unrelated_kept" in x for x in rs) else None))
    for k_, v in summ.items():
        say(f"   {k_:<28} flip {v['flip']:.2f}  coherent {v['coherent']:.2f}" + (f"  unrelated kept {v['unrelated_kept']:.2f}" if v["unrelated_kept"] is not None else ""))
    json.dump(dict(repo=a.repo, depths=[frac[l] for l in L3], doses=a.doses, items=out, summary=summ,
                   cos_vwrite_dim={str(l): float(torch.dot(U[l].float(), D[l].float())) for l in L3}),
              open(f"{a.results}/{name}__steer_operative.json", "w"), default=float)


# ============================================================ W: steering awareness
def part_w():
    oi = [r for r in rows("oi_orig") if r["kind"] == "test_real"]
    ne = [r for r in rows("nusc_edit") if r["kind"] == "test_real"]
    if not oi or not ne:
        say("[W] missing rows"); return
    ax = tr_axis()
    U = fit_direction(oi[: a.n_fit], ax, L3)
    out = []
    for r in ne + oi[a.n_fit: a.n_fit + 30]:
        b, _img = batch_for(r)
        with torch.no_grad():
            base = first_answer(sc.generate(b, max_new_tokens=4), TR)
        rec = dict(id=r["id"], family=r["family"], variant=r["meta"].get("variant"), base=base, runs=[])
        if base is None:
            out.append(rec); continue
        sign = -1 if base == "test" else +1
        for l in L3:
            for al in a.doses:
                with torch.no_grad(), AnswerSpotAdd(l, U[l] * sign, al):
                    v = first_answer(sc.generate(b, max_new_tokens=4), TR)
                rec["runs"].append(dict(layer=l, frac=frac[l], alpha=al, answer=v, flipped=(v is not None and v != base)))
        out.append(rec)
    summ = {}
    for grp, sel in (("says test (eval overlays)", lambda r: r["base"] == "test"), ("says real", lambda r: r["base"] == "real")):
        for l in L3:
            for al in a.doses:
                rs = [x for r in out if sel(r) for x in r["runs"] if x["layer"] == l and x["alpha"] == al]
                if rs:
                    summ[f"{grp}|{frac[l]:.2f}|{al}"] = dict(flip=float(np.mean([x["flipped"] for x in rs])), n=len(rs))
    for k_, v in summ.items():
        say(f"   [W] {k_:<36} flip {v['flip']:.2f} (n={v['n']})")
    json.dump(dict(repo=a.repo, items=out, summary=summ), open(f"{a.results}/{name}__steer_awareness.json", "w"), default=float)


# ============================================================ T: word strips
def part_t():
    ne = rows("nusc_edit"); p = f"{a.results}/{name}__nusc_edit_states.npz"
    if not ne or not os.path.exists(p):
        say("[T] missing"); return
    z = np.load(p); ids = [str(x) for x in z["ids"]]; ANS = z["ANS"].astype(np.float32)
    by = {r["id"]: r for r in ne}
    keep = [i for i, k in enumerate(ids) if by[k]["kind"] == "yesno"]
    X = ANS[keep]; kid = [ids[i] for i in keep]
    n = len(kid); parts = np.array_split(np.random.default_rng(0).permutation(n), 5)
    PRED = np.zeros((n, len(ALLL), X.shape[-1]), np.float32)
    for te in parts:
        tr = np.setdiff1d(np.arange(n), te); Rm = reader.fit(X[tr], ALLL, final_layer=LAST)
        for j, l in enumerate(ALLL):
            PRED[te, j] = Rm.to_final(X[te, l], l)
    out = []
    show = ALLL[:: max(1, len(ALLL) // 8)][:8]
    for i, k in enumerate(kid):
        r = by[k]
        strips = {"V-Read": {}, "logit lens": {}}
        with torch.no_grad():
            for l in show:
                j = ALLL.index(l)
                for tname, H in (("V-Read", PRED[i, j]), ("logit lens", X[i, l])):
                    lg = m.unembed(torch.as_tensor(H[None]).to(m.device).to(torch.bfloat16)).float()[0]
                    strips[tname][str(l)] = [tok.decode([int(t)]).strip() for t in lg.argsort(descending=True)[:3]]
        out.append(dict(id=k, variant=r["meta"].get("variant"), condition=r["condition"], expected=r.get("expected"), reply=r.get("reply"),
                        image=r["image"], edit_number=r["meta"].get("edit_number"), depths={str(l): frac[l] for l in show}, strips=strips))
    json.dump(dict(repo=a.repo, items=out), open(f"{a.results}/{name}__wordstrips.json", "w"))
    say(f"[T] word strips for {len(out)} items")


# ============================================================ X: map transfer
def part_x():
    src, dst = "oi_orig", "nusc_edit"
    ps, pd_ = f"{a.results}/{name}__{src}_states.npz", f"{a.results}/{name}__{dst}_states.npz"
    if not (os.path.exists(ps) and os.path.exists(pd_)):
        say("[X] missing"); return
    rs = {r["id"]: r for r in rows(src)}; rd = {r["id"]: r for r in rows(dst)}
    zs = np.load(ps); zd = np.load(pd_)
    Xs = zs["ANS"].astype(np.float32)[[i for i, k in enumerate(zs["ids"]) if rs.get(str(k), {}).get("kind") == "yesno"]]
    kd = [i for i, k in enumerate(zd["ids"]) if rd.get(str(k), {}).get("kind") == "yesno"]
    Xd = zd["ANS"].astype(np.float32)[kd]; idd = [str(zd["ids"][i]) for i in kd]
    Rf = reader.fit(Xs, ALLL, final_layer=LAST)
    ax = yn_axis()
    def score(H):
        with torch.no_grad():
            lg = m.unembed(torch.as_tensor(H[None]).to(m.device).to(torch.bfloat16))[0].float()
            return float(torch.as_tensor(ax(lg, None)).reshape(-1)[0])
    fin = np.array([score(Xd[i][LAST]) for i in range(len(idd))]); y = (fin > 0).astype(int)
    if y.min() == y.max():
        say("[X] degenerate"); return
    curves = {"foreign map": [], "logit lens": []}
    for l in ALLL:
        pf = Rf.to_final(Xd[:, l], l)
        curves["foreign map"].append(float(roc_auc_score(y, [score(pf[i]) for i in range(len(idd))])))
        curves["logit lens"].append(float(roc_auc_score(y, [score(Xd[i][l]) for i in range(len(idd))])))
    own = json.load(open(f"{a.results}/{name}__{dst}_vread.json"))["auroc"].get("yesno|y_final|v_read") if os.path.exists(f"{a.results}/{name}__{dst}_vread.json") else None
    say("[X] foreign map  " + " ".join(f"{x:.2f}" for x in curves["foreign map"][:: max(1, len(ALLL) // 8)]))
    say("[X] own map      " + (" ".join(f"{x:.2f}" for x in own[:: max(1, len(ALLL) // 8)]) if own else "n/a"))
    say("[X] logit lens   " + " ".join(f"{x:.2f}" for x in curves["logit lens"][:: max(1, len(ALLL) // 8)]))
    json.dump(dict(repo=a.repo, depths=[frac[l] for l in ALLL], foreign=curves["foreign map"], logit=curves["logit lens"], own=own, n=len(idd)),
              open(f"{a.results}/{name}__transfer.json", "w"))


for part in a.parts:
    {"S": part_s, "W": part_w, "T": part_t, "X": part_x}[part]()
say("DONE d06")
