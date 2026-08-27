"""Swap what the picture says, and watch the answer follow.

The Jacobian lens defines, for every vocabulary token t, a vector
    v_t = J_l^T W_U[t]
whose inner product with an activation is that token's lens score.  Reading it
is one thing; the experiment that shows it is *load-bearing* is the swap: take
the activation's coordinates along two lens vectors, exchange them, leave
everything orthogonal alone, and see whether the model's answer follows the
swap rather than the picture.

    h <- h - a*v_A - b*v_B + b*v_A + a*v_B,    a = h.v_A,  b = h.v_B

For a vision-language model the interesting place to do this is the image
tokens, and the interesting question is one whose answer the picture determines
but the words do not.  A photograph of a bird, asked how many legs the animal
has, gets 2.  Swap the bird coordinate for cat at the image tokens and the
question is whether it says 4.

Three question types, each with a short answer that can be scored exactly:
legs, wheels, and colour.  The model never sees either concept's name.

The lens vectors are computed per position rather than averaged over them,
which for a VLM is the whole difficulty: transport out of an image token and
out of a text token are different operators, and a lens that averages the two
has neither.  One backward pass per token gives the vector at every layer and
position at once, so the position-resolved version costs no more than the
averaged one.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time

sys.path.insert(0, "/workspace/vlm/code")
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch

import jvlm
from jvlm import cocodata, corpus, readout, stats as st, tasks

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--alphas", type=float, nargs="*", default=[1.0, 2.0, 3.0])
ap.add_argument("--band", type=float, nargs=2, default=[0.12, 0.88])
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/swap")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lo = max(m.n_deepstack, 1)
BAND = [l for l in range(m.n_layers - 1)
        if a.band[0] <= l / (m.n_layers - 1) <= a.band[1] and l >= lo]
sc = tasks.Scorer(m)
print(m, f"\nswapping over layers {BAND[0]}..{BAND[-1]}", flush=True)

# questions whose answer the picture settles and the wording does not
LEGS = "How many legs does the main animal in this photo have? Answer with a number."
WHEELS = "How many wheels does the main vehicle in this photo have? Answer with a number."
COLOUR = "What colour is the main object in this photo? Answer with one word."

TASKS = {
    "legs": (LEGS, {"bird": "2", "cat": "4", "dog": "4", "horse": "4",
                    "cow": "4", "sheep": "4", "elephant": "4", "zebra": "4",
                    "bear": "4", "giraffe": "4"}),
    "wheels": (WHEELS, {"bicycle": "2", "motorcycle": "2", "car": "4",
                        "bus": "6", "truck": "6"}),
    "colour": (COLOUR, {"banana": "yellow", "orange": "orange",
                        "broccoli": "green", "carrot": "orange",
                        "zebra": "black", "giraffe": "yellow"}),
}
# pairs whose answers differ, so a successful swap is visible
PAIRS = []
for kind, (q, ans) in TASKS.items():
    ks = sorted(ans)
    for i, A in enumerate(ks):
        for B in ks[i + 1:]:
            if ans[A] != ans[B]:
                PAIRS.append((kind, A, B))
print(f"{len(PAIRS)} concept pairs with differing answers", flush=True)


def ids_of(w):
    return cocodata.concept_token_ids(m.tokenizer, w)


def ans_ids(s):
    out = []
    for f in (s, " " + s, s.capitalize(), " " + s.capitalize()):
        t = m.tokenizer.encode(f, add_special_tokens=False)
        if len(t) == 1:
            out.append(t[0])
    return sorted(set(out))


def lens_vectors(batch, tokens, layers, positions):
    """v_t at every requested layer and position: one backward pass per token.

    This is the Jacobian lens vector J^T W_U[t], written out per position
    instead of averaged over them.
    """
    from jlens.hooks import ActivationRecorder
    pos = torch.as_tensor(positions, dtype=torch.long)
    out = {l: [] for l in layers}
    with (
        ActivationRecorder(m.layers, at=[*layers, m.n_layers - 1],
                           start_graph_at=min(layers)) as rec,
        torch.enable_grad(),
    ):
        m.forward(batch)
        acts = [rec.activations[l] for l in layers]
        h_fin = rec.activations[m.n_layers - 1][0, -1]
        logits = m.unembed(h_fin[None])[0]
        for j, t in enumerate(tokens):
            g = torch.autograd.grad(logits[t], acts,
                                    retain_graph=(j < len(tokens) - 1))
            for l, gg in zip(layers, g, strict=True):
                out[l].append(gg[0, pos.to(gg.device)].float())
    return {l: torch.stack(v).mean(0) for l, v in out.items()}   # [P, d]


class Swap:
    """Exchange the coordinates along two lens vectors, per position."""

    def __init__(self, model, va, vb, ipos, alpha=1.0, clean=None):
        self.m, self.ipos, self.alpha = model, ipos, alpha
        self.va = {l: torch.nn.functional.normalize(v, dim=-1)
                   for l, v in va.items()}
        self.vb = {l: torch.nn.functional.normalize(v, dim=-1)
                   for l, v in vb.items()}
        self.h = []
        # What the edit costs, in the same units the bounded-assignment
        # experiment reports, so the two are comparable rather than merely
        # both called "disturbance":
        #   step  -- ||dh||/||h|| for the exchange made AT this layer, which is
        #            what a single bounded assignment would also report;
        #   drift -- ||h_edited - h_clean||/||h_clean|| at this layer, which is
        #            the quantity that grows because this form compounds.
        # Both are means over the image positions.
        self.clean = clean or {}
        self.log = {}

    def _hook(self, layer):
        def fn(mod, inp, out):
            tup = not torch.is_tensor(out)
            h = (out[0] if tup else out)
            if h.shape[1] <= self.ipos.max():
                return out
            h = h.clone()
            p = self.ipos.to(h.device)
            A = self.va[layer].to(h.device).to(h.dtype)
            B = self.vb[layer].to(h.device).to(h.dtype)
            sub = h[0, p]
            ca = (sub * A).sum(-1, keepdim=True)
            cb = (sub * B).sum(-1, keepdim=True)
            new = sub - ca * A - cb * B + self.alpha * (cb * A + ca * B)
            rec = dict(step=float(((new - sub).norm(dim=-1)
                                   / sub.norm(dim=-1).clamp_min(1e-6)).mean()))
            hc = self.clean.get(layer)
            if hc is not None:
                hc = hc.to(new.device).to(new.dtype)
                rec["drift"] = float(((new - hc).norm(dim=-1)
                                      / hc.norm(dim=-1).clamp_min(1e-6)).mean())
            self.log[layer] = rec
            h[0, p] = new
            return (h, *out[1:]) if tup else h
        return fn

    def cost(self) -> dict:
        """Mean over the edited layers of each recorded quantity."""
        if not self.log:
            return {}
        ks = next(iter(self.log.values())).keys()
        return {k: float(np.mean([v[k] for v in self.log.values() if k in v]))
                for k in ks}

    def __enter__(self):
        for l in self.va:
            self.h.append(self.m.layers[l].register_forward_hook(self._hook(l)))
        return self

    def __exit__(self, *e):
        for x in self.h:
            x.remove()
        self.h = []


pool = cocodata.concept_items(8000, min_area=0.10, seed=17)
by_c = {}
for it in pool:
    by_c.setdefault(it["concept"], []).append(it)

rows, cases = [], []
n_done = 0
for kind, A, B in PAIRS:
    q, ans = TASKS[kind]
    ia, ib = ids_of(A), ids_of(B)
    ta, tb = ans_ids(ans[A]), ans_ids(ans[B])
    if not (ia and ib and ta and tb):
        continue
    for it in by_c.get(A, [])[: max(1, a.n // max(len(PAIRS), 1))]:
        im = corpus.load_image(it["path"], square=a.square)
        b = m.chat(q, [im])
        ip = readout.image_positions(m, b)
        base = sc.last_logits(b).float()
        # the model has to get the unswapped question right, or the swap has
        # nothing to move
        if float(base[ta].max()) <= float(base[tb].max()):
            continue
        va = lens_vectors(b, ia, BAND, ip.tolist())
        vb = lens_vectors(b, ib, BAND, ip.tolist())
        # the unedited residual stream at the same layers and positions, so the
        # edit's cost can be stated rather than asserted
        _res = readout.residuals(m, b, BAND)
        clean_h = {l: _res[l][ip].float() for l in BAND}
        for al in a.alphas:
            with Swap(m, va, vb, ip, alpha=al, clean=clean_h) as _sw:
                lg = sc.last_logits(m.chat(q, [im])).float()
                txt = sc.generate(m.chat(q, [im]), max_new_tokens=6).strip()
            cost = _sw.cost()
            # scored the way the swap is meant to be scored: between the two
            # answers the concepts disagree about, does the model now prefer the
            # one the swap installed?
            won = float(lg[tb].max()) > float(lg[ta].max())
            cand = sorted(set(ta) | set(tb))
            top2 = int(cand[int(lg[cand].argmax())])
            top = int(lg.argmax())
            rows.append(dict(kind=kind, src=A, dst=B, alpha=al,
                             image_id=int(it["image_id"]),
                             src_ans=ans[A], dst_ans=ans[B],
                             flipped=int(won),
                             said_dst=int(top in tb), said_src=int(top in ta),
                             picked_dst=int(top2 in tb),
                             text=txt[:24],
                             margin=float(lg[tb].max() - lg[ta].max()),
                             disturb_step=cost.get("step", float("nan")),
                             disturb_drift=cost.get("drift", float("nan"))))
            # an example is only worth showing if the model actually says the
            # swapped answer -- the logit-level score is the measurement, but a
            # reader should not be shown a case where the text came out as junk
            clean = bool(re.match(r"\s*" + re.escape(ans[B]), txt, re.I))
            if won and top2 in tb and clean and len(cases) < 120:
                cases.append(dict(kind=kind, src=A, dst=B, q=q, alpha=al,
                                  src_ans=ans[A], dst_ans=ans[B],
                                  said=txt[:24], image_id=int(it["image_id"]),
                                  path=it["path"]))
        n_done += 1
        if n_done % 20 == 0:
            el = time.time() - t0
            print(f"  {n_done} trials  {el:.0f}s  "
                  f"{len(cases)} clean swaps so far", flush=True)
            pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")
        if n_done >= a.n:
            break
    if n_done >= a.n:
        break

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
json.dump(cases, open(f"{a.out}/{tag}_cases.json", "w"), indent=1)
agg = df.groupby(["kind", "alpha"]).agg(
    n=("flipped", "count"), flipped=("flipped", "mean"),
    picks_target=("picked_dst", "mean"),
    says_target=("said_dst", "mean"), still_source=("said_src", "mean"))
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\nSWAPPING THE LENS VECTOR AT THE IMAGE TOKENS\n",
      agg.round(3).to_string())
overall = df.groupby("alpha").agg(n=("flipped", "count"),
                                  flipped=("flipped", "mean"),
                                  picks_target=("picked_dst", "mean"),
                                  says_target=("said_dst", "mean"),
                                  disturb_step=("disturb_step", "mean"),
                                  disturb_drift=("disturb_drift", "mean"))
print("\n", overall.round(3).to_string())
print("\nWHAT THE EDIT COSTS  (mean over the edited band, image positions)")
print("  step  = ||dh||/||h|| for the exchange made at each layer")
print("  drift = ||h_edited - h_clean||/||h_clean|| at each layer")
print(overall[["disturb_step", "disturb_drift"]].round(4).to_string())
json.dump({"repo": a.repo, "layers": [BAND[0], BAND[-1]],
           "n": int(len(df)), "by_kind": st.jsonable(agg.round(4).to_dict("index")),
           "overall": st.jsonable(overall.round(4).to_dict("index")),
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
for c in cases[:12]:
    print(f"  {c['src']} -> {c['dst']}: asked {c['kind']}, was "
          f"{c['src_ans']}, now says {c['said']!r}")
print("done", flush=True)
