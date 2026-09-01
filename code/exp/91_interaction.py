# SPDX-License-Identifier: Apache-2.0
r"""The interaction law, proved on paper and then measured on a real VLM.

The claim under test is an identity, not a correlation::

    I_R(q_a, q_b) = A_R(q_a) - A_R(q_b)
    sum_R I_R     = [F(h,q_a) - F(hbar,q_a)] - [F(h,q_b) - F(hbar,q_b)]
                  = [F(h,q_a) - F(h,q_b)] - [F(hbar,q_a) - F(hbar,q_b)]

so the regional decomposition sums exactly to a difference-in-differences with
the mean image as the control condition.  Four things are measured, in this
order, and every one of them has a control standing behind it.

(a) *Completeness, per prompt.*  ``sum_R A_R(q)`` against the effect it claims
    to decompose, which is not estimated but run: the model with the image, and
    the model with every image token replaced by the baseline.

(b) *The interaction law.*  ``sum_R I_R`` against the double difference, from
    the same four forward passes.

(c) *Convergence.*  The path integral is discretised, so the law holds only up
    to that discretisation.  Steps 1, 2, 4, 8 are run and the error is reported
    as it falls -- a law that is true should get closer as the rule gets finer,
    and a bug should not.

(d) *Prompt-blindness.*  Image tokens precede text under a causal mask, so any
    lens ``g(M h)`` whose ``M`` is not computed from the prompt reads the same
    bytes for both questions and its interaction is identically zero.  That is
    reported at region *and* patch level, beside two controls: a same-prompt
    re-run, which is the floating-point floor the zero has to beat, and the same
    prompt on a different photograph, which must move -- otherwise the zero
    would only prove that the harness never compares anything.

``--toy`` runs the whole argument on the CPU against a small differentiable
stand-in for a VLM: a causal transformer with an image span in front of a text
span, in float32, where the truth is exact and the only error is the path rule.
The algebra is proved there before any GPU time is spent on it.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time

sys.path.insert(0, "/workspace/vlm/code")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--toy", action="store_true",
                help="CPU proof on a small differentiable stand-in; no GPU")
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens", default=None)
ap.add_argument("--n", type=int, default=24, help="Winoground items")
ap.add_argument("--mean-images", type=int, default=60)
ap.add_argument("--n-layers", type=int, default=3)
ap.add_argument("--pool", type=int, default=2, help="token block side")
ap.add_argument("--steps", default="1,2,4,8")
ap.add_argument("--head-steps", type=int, default=8)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--data", default="/workspace/vlm/data/winoground")
ap.add_argument("--out", default="/workspace/vlm/results/interaction")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
STEPS = sorted({int(s) for s in a.steps.split(",") if s.strip()}
               | ({a.head_steps} if not a.toy else set()))
t0 = time.time()


# =============================================================== the CPU proof

def _init(mod, g, scale: float = 1.0) -> None:
    import torch.nn as nn
    for m in mod.modules():
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                fan = m.weight.shape[1]
                m.weight.copy_(torch.randn(m.weight.shape, generator=g)
                               * scale / math.sqrt(fan))
                if m.bias is not None:
                    m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.05)


def _toy_classes():
    import torch.nn as nn
    import torch.nn.functional as Fn

    class ToyBlock(nn.Module):
        """One causal attention + MLP block, the smallest thing with the property
        that matters: a position can only read positions at or before it."""

        def __init__(self, d, g):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d, bias=False)
            self.o = nn.Linear(d, d, bias=False)
            self.up, self.down = nn.Linear(d, 3 * d), nn.Linear(3 * d, d)
            _init(self, g, scale=0.9)

        def forward(self, h):
            S, d = h.shape[1], h.shape[2]
            q, k, v = self.qkv(self.ln1(h)).chunk(3, -1)
            att = (q @ k.transpose(-1, -2)) / d ** 0.5
            mask = torch.triu(torch.ones(S, S, dtype=torch.bool,
                                         device=h.device), 1)
            att = att.masked_fill(mask, float("-inf")).softmax(-1)
            h = h + self.o(att @ v)
            return h + self.down(Fn.gelu(self.up(self.ln2(h))))

    class ToyVLM:
        """A stand-in with exactly the surface :mod:`jvlm.vmap` uses."""

        def __init__(self, d=24, n_layers=4, vocab=48, n_img=6, seed=0):
            g = torch.Generator().manual_seed(seed)
            self.layers = nn.ModuleList([ToyBlock(d, g) for _ in range(n_layers)])
            self._norm = nn.LayerNorm(d)
            self._lm_head = nn.Linear(d, vocab, bias=False)
            _init(self._lm_head, g, scale=1.0)
            self.tok = torch.randn(vocab, d, generator=g)
            self.posemb = torch.randn(64, d, generator=g) * 0.2
            self.n_layers, self.d_model = n_layers, d
            self.vocab_size, self.n_img = vocab, n_img
            self.device = torch.device("cpu")
            for mod in (self.layers, self._norm, self._lm_head):
                for p in mod.parameters():
                    p.requires_grad_(False)
                mod.eval()

        def forward(self, batch):
            ids, img = batch["input_ids"], batch["img"]
            h = self.tok[ids].clone()
            h[:, : self.n_img] = img
            h = h + self.posemb[: ids.shape[1]][None]
            for blk in self.layers:
                h = blk(h)
            return h

        def unembed(self, residual):
            return self._lm_head(self._norm(residual))

        def replicate(self, batch, n):
            if n == 1:
                return batch
            return {"input_ids": batch["input_ids"].expand(n, -1),
                    "img": batch["img"].expand(n, -1, -1)}

    return ToyVLM


def add_order(conv: list) -> None:
    """Empirical order of convergence, ``log2(err(s) / err(2s))``.

    The midpoint rule is second order on a smooth path, so a doubling of the
    steps should divide the error by about four.  Printing the ratio turns "the
    error goes down" into a falsifiable number: a wrong Jacobian, a mismatched
    baseline or an incomplete partition leaves a floor that no refinement
    removes, and the order collapses to zero rather than sitting near two.
    """
    for i, row in enumerate(conv):
        for key in ("completeness", "law"):
            row[key]["order"] = float("nan")
            if i + 1 < len(conv) and conv[i + 1]["steps"] == 2 * row["steps"]:
                e0 = row[key]["mean_abs_err"]
                e1 = conv[i + 1][key]["mean_abs_err"]
                if e0 > 0 and e1 > 0:
                    row[key]["order"] = float(math.log2(e0 / e1))


def run_toy() -> dict:
    from jvlm import interact, readout, vmap

    torch.manual_seed(0)
    ToyVLM = _toy_classes()
    m = ToyVLM(d=24, n_layers=4, vocab=48, n_img=6, seed=3)
    P, d = m.n_img, m.d_model
    LAYERS = [1, 2]
    IDS = [5, 11, 23]
    REGIONS = [torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4, 5])]
    interact.check_partition(REGIONS, P)
    POS = list(range(P))

    g = torch.Generator().manual_seed(17)

    def make_image(k):
        return torch.randn(1, P, d, generator=torch.Generator().manual_seed(k)) * 1.1

    def batch(img, ids):
        return {"input_ids": torch.tensor([[0] * P + list(ids)]), "img": img}

    Q_A = [7, 3, 41, 2, 19]
    Q_B = [31, 6, 8]                      # deliberately a different length
    IMG = make_image(101)
    ba, bb = batch(IMG, Q_A), batch(IMG, Q_B)

    # a per-position "corpus mean" over held-out toy photographs
    acc = {l: torch.zeros(P, d, dtype=torch.float64) for l in LAYERS}
    NM = 24
    for k in range(NM):
        A = readout.residuals(m, batch(make_image(500 + k), Q_A), LAYERS)
        for l in LAYERS:
            acc[l] += A[l][:P].double()
    MEANS = {l: (acc[l] / NM).float() for l in LAYERS}

    rep: dict = {"config": dict(d=d, n_layers=m.n_layers, P=P, layers=LAYERS,
                                token_ids=IDS, n_regions=len(REGIONS),
                                mean_images=NM, q_a=Q_A, q_b=Q_B)}

    # ---- the causal-mask fact everything else rests on
    ha = readout.residuals(m, ba, LAYERS)
    hb = readout.residuals(m, bb, LAYERS)
    rep["h_identical"] = {str(l): float((ha[l][:P] - hb[l][:P]).abs().max())
                          for l in LAYERS}
    rep["h_differs_across_images"] = {
        str(l): float((ha[l][:P]
                       - readout.residuals(m, batch(make_image(102), Q_A),
                                           LAYERS)[l][:P]).abs().max())
        for l in LAYERS}

    # ---- (a)/(b)/(c): completeness, the law, and convergence
    truth = {l: interact.did_truth(m, ba, bb, IDS, l, POS, MEANS[l])
             for l in LAYERS}
    # the truth routine here must be the same one vmap already uses
    chk = vmap.completeness(m, ba, IDS, LAYERS[0], POS, MEANS[LAYERS[0]],
                            torch.zeros(len(IDS), len(REGIONS)))
    rep["truth_agrees_with_vmap"] = float(
        (torch.tensor(chk["truth"]) - truth[LAYERS[0]]["eff_a"]).abs().max())
    # the paper writes the right-hand side two ways -- (eff_a - eff_b) and
    # (behaviour - baseline gap).  They are the same four numbers regrouped, and
    # this is the line that says so in floating point rather than on a blackboard
    rep["regrouping_identity"] = max(
        float((truth[l]["did"]
               - (truth[l]["behaviour"] - truth[l]["baseline_gap"])).abs().max())
        for l in LAYERS)

    rows = []
    for s in sorted(set(STEPS + [16, 32, 64])):
        out = interact.interaction_map(m, None, "a", "b", IDS, LAYERS, REGIONS,
                                       MEANS, steps=s, positions=POS,
                                       batches=(ba, bb), diagnostics=False)
        for l in LAYERS:
            ca = out["a_region"][l].sum(-1)
            cb = out["b_region"][l].sum(-1)
            ci = out["region"][l].sum(-1)
            # regions must add to patches, whatever the drawing
            part = float((out["region"][l].sum(-1)
                          - out["patch"][l].sum(-1)).abs().max())
            for j, t in enumerate(IDS):
                rows.append(dict(steps=s, layer=l, token=t, partition_err=part,
                                 claim_a=float(ca[j]),
                                 truth_a=float(truth[l]["eff_a"][j]),
                                 claim_b=float(cb[j]),
                                 truth_b=float(truth[l]["eff_b"][j]),
                                 claim_i=float(ci[j]),
                                 truth_i=float(truth[l]["did"][j])))
    df = pd.DataFrame(rows)
    rep["partition_max_err"] = float(df.partition_err.max())
    conv = []
    for s in sorted(df.steps.unique()):
        d_ = df[df.steps == s]
        comp = interact.error_stats(
            np.concatenate([d_.claim_a, d_.claim_b]),
            np.concatenate([d_.truth_a, d_.truth_b]))
        law = interact.error_stats(d_.claim_i, d_.truth_i)
        conv.append(dict(steps=int(s), completeness=comp, law=law))
    add_order(conv)
    rep["convergence"] = conv

    # ---- the shared baseline is load-bearing: break it and the law breaks
    bad = {l: MEANS[l] + 0.05 * torch.randn(P, d, generator=g) for l in LAYERS}
    A_a = vmap.ig_map(m, ba, IDS, LAYERS, POS, means=MEANS, steps=64)
    A_b = vmap.ig_map(m, bb, IDS, LAYERS, POS, means=bad, steps=64)
    split = interact.error_stats(
        np.concatenate([(A_a[l] - A_b[l]).sum(-1).numpy() for l in LAYERS]),
        np.concatenate([truth[l]["did"].numpy() for l in LAYERS]))
    rep["per_prompt_baseline_breaks_the_law"] = split

    # ---- (d) prompt-blindness, with both controls
    J = torch.randn(d, d, generator=g) / math.sqrt(d)
    blind, ctrl = {}, {}
    for l in LAYERS:
        fa = interact.fixed_lens_patches(m, ha[l][:P], IDS, jacobian=J)
        fb = interact.fixed_lens_patches(m, hb[l][:P], IDS, jacobian=J)
        h2 = readout.residuals(m, batch(IMG, Q_A), LAYERS)[l][:P]   # re-run
        f2 = interact.fixed_lens_patches(m, h2, IDS, jacobian=J)
        ho = readout.residuals(m, batch(make_image(102), Q_A), LAYERS)[l][:P]
        fo = interact.fixed_lens_patches(m, ho, IDS, jacobian=J)
        for name in fa:
            blind.setdefault(name, {})[str(l)] = dict(
                patch_maxdev=float((fa[name] - fb[name]).abs().max()),
                region_maxdev=float((interact.region_sums(fa[name], REGIONS)
                                     - interact.region_sums(fb[name], REGIONS)
                                     ).abs().max()),
                same_prompt_rerun=float((fa[name] - f2[name]).abs().max()),
                other_image=float((fa[name] - fo[name]).abs().max()))
    out8 = interact.interaction_map(m, None, "a", "b", IDS, LAYERS, REGIONS,
                                    MEANS, steps=8, positions=POS,
                                    batches=(ba, bb), diagnostics=False)
    outaa = interact.interaction_map(m, None, "a", "a", IDS, LAYERS, REGIONS,
                                     MEANS, steps=8, positions=POS,
                                     batches=(ba, batch(IMG, Q_A)),
                                     diagnostics=False)
    for l in LAYERS:
        blind.setdefault("VJ-Lens", {})[str(l)] = dict(
            patch_maxdev=float(out8["patch"][l].abs().max()),
            region_maxdev=float(out8["region"][l].abs().max()),
            same_prompt_rerun=float(outaa["patch"][l].abs().max()),
            other_image=float("nan"))
    rep["prompt_blindness"] = blind
    rep["controls"] = ctrl
    return rep, df


def print_toy(rep, df) -> None:
    print("\n" + "=" * 78)
    print("CPU PROOF -- toy causal transformer, float32, exact truth")
    print("=" * 78)
    c = rep["config"]
    print(f"  d={c['d']}  layers={c['n_layers']}  image tokens={c['P']}  "
          f"regions={c['n_regions']}  tokens read={c['token_ids']}")
    print(f"  image-token activations, prompt a vs prompt b: "
          f"max|h_a - h_b| = {rep['h_identical']}")
    print(f"  same prompt, different photograph:              "
          f"max|h - h'|  = "
          f"{ {k: round(v, 4) for k, v in rep['h_differs_across_images'].items()} }")
    print(f"  regions sum to patches: max err {rep['partition_max_err']:.3e}")
    print(f"  did_truth agrees with vmap.completeness to "
          f"{rep['truth_agrees_with_vmap']:.3e}")
    print(f"  [eff_a - eff_b] vs [behaviour - baseline gap]: "
          f"{rep['regrouping_identity']:.3e}")
    print("\n  (a)+(b)+(c) THE LAW AS THE PATH RULE IS REFINED")
    print(f"    {'steps':>6} | {'completeness maxerr':>20} {'rel':>9} {'r':>8} "
          f"{'ord':>5} | {'law maxerr':>12} {'rel':>9} {'r':>8} {'ord':>5}")
    for row in rep["convergence"]:
        cm, lw = row["completeness"], row["law"]
        print(f"    {row['steps']:>6} | {cm['max_abs_err']:>20.3e} "
              f"{cm['rel_err']:>9.2e} {cm['pearson']:>8.5f} {cm['order']:>5.2f} | "
              f"{lw['max_abs_err']:>12.3e} {lw['rel_err']:>9.2e} "
              f"{lw['pearson']:>8.5f} {lw['order']:>5.2f}")
    b = rep["per_prompt_baseline_breaks_the_law"]
    print(f"\n  per-prompt baselines instead of one shared baseline (steps=64):"
          f"  max err {b['max_abs_err']:.3e}  rel {b['rel_err']:.3f}  "
          f"r={b['pearson']:.4f}")
    print("\n  (d) PROMPT-BLINDNESS  max |map(q_a) - map(q_b)|")
    print(f"    {'lens':>12} {'layer':>6} {'patch':>12} {'region':>12} "
          f"{'rerun ctrl':>12} {'other image':>12}")
    for name, per in rep["prompt_blindness"].items():
        for l, v in per.items():
            print(f"    {name:>12} {l:>6} {v['patch_maxdev']:>12.4e} "
                  f"{v['region_maxdev']:>12.4e} {v['same_prompt_rerun']:>12.4e} "
                  f"{v['other_image']:>12.4e}")


if a.toy:
    rep, df = run_toy()
    print_toy(rep, df)
    os.makedirs(a.out, exist_ok=True)
    df.to_parquet(f"{a.out}/toy_rows.parquet")
    json.dump(rep, open(f"{a.out}/toy_stats.json", "w"), indent=2, default=float)
    print(f"\nwrote {a.out}/toy_stats.json", flush=True)
    sys.exit(0)


# ================================================================= the real run

import jvlm
from jvlm import corpus, interact, readout, stats as st, vmap

os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

QUESTION = "Does the caption '{c}' describe this image?  Answer yes or no."

ex = os.path.join(a.data, "data", "examples.jsonl")
if not os.path.exists(ex):
    raise SystemExit(f"winoground not found at {ex}; 79_winoground.py fetches it")
RECS = [json.loads(l) for l in open(ex)]
for r in RECS:
    for k in ("image_0", "image_1"):
        r[k + "_path"] = os.path.join(a.data, "images", r[k] + ".png")

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lo, hi = max(m.n_deepstack, 1), m.n_layers - 2
LAYERS = sorted({int(round(lo + (hi - lo) * f))
                 for f in np.linspace(0.2, 0.85, a.n_layers)})
J = None
if a.jlens:
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))

    # Read where the lens exists.  A fitted Jacobian lens only has a transport
    # at the layers it was fitted at; chosen from the model's geometry instead,
    # the depths miss it and the arm is dropped in silence.
    LAYERS, _ = readout.snap_to_lens(LAYERS, J)


def ids_of(w):
    out = []
    for f in (w, " " + w, w.capitalize(), " " + w.capitalize()):
        t = m.tokenizer.encode(f, add_special_tokens=False)
        if len(t) == 1:
            out.append(t[0])
    return out


# Fixed in advance and never chosen from the data: the conservation law holds
# for ANY vocabulary item, so there is nothing here to tune.
_y, _n = ids_of("yes"), ids_of("no")
if not (_y and _n):
    raise SystemExit("no single-token surface form for yes/no in this tokenizer")
YES, NO = _y[0], _n[0]
IDS = [YES, NO]
print(m, f"\nlayers {LAYERS}   steps {STEPS}   answer tokens "
      f"{m.tokenizer.decode([YES])!r}/{m.tokenizer.decode([NO])!r}", flush=True)


def load(p):
    return corpus.load_image(p, square=a.square)


# ------------------------------------------------ the shared baseline and grid
pool_paths = corpus.coco_images(a.mean_images + 30, seed=7)
b0 = m.chat("Describe this image.", [load(pool_paths[0])])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
assert G * G == P, f"{P} image tokens is not a square grid"
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
nrm = {l: torch.zeros(P, dtype=torch.float64) for l in LAYERS}
nm = 0
for p in pool_paths:
    if nm >= a.mean_images:
        break
    b = m.chat("Describe this image.", [load(p)])
    if len(readout.image_positions(m, b)) != P:
        continue
    A = readout.residuals(m, b, LAYERS)
    for l in LAYERS:
        v = A[l][IPOS].float().cpu()
        acc[l] += v.double()
        nrm[l] += v.norm(dim=-1).double()
    nm += 1
MEANS = {l: (acc[l] / max(nm, 1)).float() for l in LAYERS}
NORM = {l: (nrm[l] / max(nm, 1)).float() for l in LAYERS}
REG = (NORM[LAYERS[0]] > NORM[LAYERS[0]].mean() + 3 * NORM[LAYERS[0]].std())
KEEP = (~REG).nonzero().flatten()

C = max(2, G // a.pool)
SIDE = int(np.ceil(G / C))
REGIONS = interact.grid_partition(G, SIDE)
interact.check_partition(REGIONS, P)
print(f"{P} image tokens on {G}x{G}; {len(REGIONS)} regions of <={SIDE*SIDE} "
      f"patches, a complete partition; {int(REG.sum())} register-like tokens "
      f"(kept, or the law would leak); means from {nm} held-out photographs "
      f"({time.time()-t0:.0f}s)", flush=True)

TARGETS = {"yes": (0, None), "no": (1, None), "margin": (0, 1)}


def pick(v, spec):
    """Row ``j`` of a ``[k, ...]`` reading, or the difference of two rows.

    Every quantity in this file is linear in the vocabulary axis, so the margin
    map costs no extra passes and its truth is the difference of the truths.
    """
    j, k = spec
    return v[j] if k is None else v[j] - v[k]


rows, blind_rows, ctrl_rows = [], [], []
kept = 0
checked_truth = False
for it, rec in enumerate(RECS[: a.n]):
    paths = [rec["image_0_path"], rec["image_1_path"]]
    if not all(os.path.exists(p) for p in paths):
        continue
    img = load(paths[0])
    qa = QUESTION.format(c=rec["caption_0"])
    qb = QUESTION.format(c=rec["caption_1"])
    ba, bb = m.chat(qa, [img]), m.chat(qb, [img])
    if len(readout.image_positions(m, ba)) != P:
        continue
    if len(readout.image_positions(m, bb)) != P:
        continue

    truth = {l: interact.did_truth(m, ba, bb, IDS, l, IPOS, MEANS[l])
             for l in LAYERS}
    if not checked_truth:
        chk = vmap.completeness(m, ba, IDS, LAYERS[0], IPOS.tolist(),
                                MEANS[LAYERS[0]],
                                torch.zeros(len(IDS), len(REGIONS)))
        dd = float((torch.tensor(chk["truth"])
                    - truth[LAYERS[0]]["eff_a"]).abs().max())
        print(f"  did_truth vs vmap.completeness: max |diff| = {dd!r}",
              flush=True)
        assert dd == 0.0, "the truth routine has drifted from vmap's"
        checked_truth = True

    per_steps = {}
    for s in STEPS:
        out = interact.interaction_map(m, img, qa, qb, IDS, LAYERS, REGIONS,
                                       MEANS, steps=s,
                                       batches=(ba, bb),
                                       diagnostics=(s == a.head_steps))
        per_steps[s] = out
        for l in LAYERS:
            ra, rb = out["a_region"][l], out["b_region"][l]
            ri = out["region"][l]
            # the region drawing must not change the sum
            part = float((ri.sum(-1) - out["patch"][l].sum(-1)).abs().max())
            # what dropping the register tokens would cost the law
            noreg = out["patch"][l][:, KEEP].sum(-1)
            for name, spec in TARGETS.items():
                rows.append(dict(
                    item=int(rec["id"]), layer=l, steps=s, target=name,
                    depth=round(l / (m.n_layers - 1), 3),
                    claim_a=float(pick(ra.sum(-1), spec)),
                    truth_a=float(pick(truth[l]["eff_a"], spec)),
                    claim_b=float(pick(rb.sum(-1), spec)),
                    truth_b=float(pick(truth[l]["eff_b"], spec)),
                    claim_i=float(pick(ri.sum(-1), spec)),
                    truth_i=float(pick(truth[l]["did"], spec)),
                    behaviour=float(pick(truth[l]["behaviour"], spec)),
                    baseline_gap=float(pick(truth[l]["baseline_gap"], spec)),
                    regroup_err=float(
                        (pick(truth[l]["did"], spec)
                         - (pick(truth[l]["behaviour"], spec)
                            - pick(truth[l]["baseline_gap"], spec))).abs()),
                    claim_noreg=float(pick(noreg, spec)),
                    partition_err=part,
                    i_region_absmax=float(pick(ri, spec).abs().max()),
                    i_region_l1=float(pick(ri, spec).abs().sum())))

    # ---------------------------------------------- (d) the prompt-blind lenses
    head = per_steps[a.head_steps]
    # prompt b's own forward pass, read separately: the zero below must come
    # from the causal mask, not from two lenses sharing one tensor
    HB = readout.residuals(m, bb, LAYERS)
    for l in LAYERS:
        h = head["h"][l]
        jac = (J.jacobians[l] if J is not None and l in J.jacobians else None)
        fa = interact.fixed_lens_patches(m, h, IDS, jacobian=jac)
        hb = HB[l][head["pos_b"]].float().cpu()
        fb = interact.fixed_lens_patches(m, hb, IDS, jacobian=jac)
        for name in fa:
            for tname, spec in TARGETS.items():
                pa, pb = pick(fa[name], spec), pick(fb[name], spec)
                qa_r = interact.region_sums(fa[name], REGIONS)
                qb_r = interact.region_sums(fb[name], REGIONS)
                blind_rows.append(dict(
                    item=int(rec["id"]), layer=l, lens=name, target=tname,
                    patch_maxdev=float((pa - pb).abs().max()),
                    region_maxdev=float((pick(qa_r, spec)
                                         - pick(qb_r, spec)).abs().max()),
                    scale=float(pa.abs().mean())))
        for tname, spec in TARGETS.items():
            blind_rows.append(dict(
                item=int(rec["id"]), layer=l, lens="VJ-Lens", target=tname,
                patch_maxdev=float(pick(head["patch"][l], spec).abs().max()),
                region_maxdev=float(pick(head["region"][l], spec).abs().max()),
                scale=float(pick(head["a_patch"][l], spec).abs().mean())))
        ctrl_rows.append(dict(item=int(rec["id"]), layer=l, kind="h_a_vs_b",
                              maxdev=head["h_maxdev"][l]))

    # ------------------------------------------------------------- the controls
    if kept < 8:
        # same prompt, run again from a freshly built batch: the floor
        ba2 = m.chat(qa, [img])
        out2 = interact.interaction_map(m, img, qa, qa, IDS, LAYERS, REGIONS,
                                        MEANS, steps=a.head_steps,
                                        batches=(ba, ba2), diagnostics=True)
        # same prompt, different photograph: this MUST move, or the harness
        # is not comparing anything
        b_other = m.chat(qa, [load(paths[1])])
        po = readout.image_positions(m, b_other)
        h_o = readout.residuals(m, b_other, LAYERS) if len(po) == P else None
        # the truth itself is computed in bf16; re-running it bounds how well
        # any decomposition could possibly agree with it
        t2 = interact.did_truth(m, ba2, bb, IDS, LAYERS[0], IPOS,
                                MEANS[LAYERS[0]])
        ctrl_rows.append(dict(
            item=int(rec["id"]), layer=LAYERS[0], kind="rerun_truth",
            maxdev=float((t2["eff_a"] - truth[LAYERS[0]]["eff_a"]).abs().max()),
            did_maxdev=float((t2["did"] - truth[LAYERS[0]]["did"]).abs().max())))
        for l in LAYERS:
            f1 = interact.fixed_lens_patches(m, head["h"][l], IDS)
            f2 = interact.fixed_lens_patches(m, out2["h"][l], IDS)
            ctrl_rows.append(dict(
                item=int(rec["id"]), layer=l, kind="rerun_same_prompt",
                maxdev=out2["h_maxdev"][l],
                vj_patch=float(out2["patch"][l].abs().max()),
                vj_region=float(out2["region"][l].abs().max()),
                logit_patch=float((f1["logit lens"]
                                   - f2["logit lens"]).abs().max())))
            if h_o is not None:
                ho = h_o[l][po].float().cpu()
                fo = interact.fixed_lens_patches(m, ho, IDS)
                ctrl_rows.append(dict(
                    item=int(rec["id"]), layer=l, kind="other_image_same_prompt",
                    maxdev=float((head["h"][l] - ho).abs().max()),
                    logit_patch=float((f1["logit lens"]
                                       - fo["logit lens"]).abs().max())))
    kept += 1
    if it % 4 == 0:
        el = time.time() - t0
        print(f"  {it}/{a.n}  kept {kept}  {el:.0f}s  "
              f"eta {el/max(it+1,1)*(a.n-it-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_rows.parquet")
        pd.DataFrame(blind_rows).to_parquet(f"{a.out}/{tag}_blind.parquet")
        pd.DataFrame(ctrl_rows).to_parquet(f"{a.out}/{tag}_ctrl.parquet")

df = pd.DataFrame(rows)
bl = pd.DataFrame(blind_rows)
ct = pd.DataFrame(ctrl_rows)
df.to_parquet(f"{a.out}/{tag}_rows.parquet")
bl.to_parquet(f"{a.out}/{tag}_blind.parquet")
ct.to_parquet(f"{a.out}/{tag}_ctrl.parquet")

# ------------------------------------------------------------------ the report
rep = {"repo": a.repo, "n_items": int(df.item.nunique()), "layers": LAYERS,
       "grid": G, "n_regions": len(REGIONS), "region_side": SIDE,
       "n_patches": P, "mean_images": nm, "steps": STEPS,
       "head_steps": a.head_steps,
       "answer_tokens": {"yes": m.tokenizer.decode([YES]),
                         "no": m.tokenizer.decode([NO])},
       "partition_max_err": float(df.partition_err.max()),
       "regrouping_identity_max_err": float(df.regroup_err.max())}
print(f"\nregions sum to patches: max err {float(df.partition_err.max()):.3e}"
      f"   |[eff_a-eff_b] - [behaviour - baseline gap]| max "
      f"{float(df.regroup_err.max()):.3e}")

conv = []
print("\n" + "=" * 96)
print("(a) COMPLETENESS PER PROMPT   and   (b) THE INTERACTION LAW"
      "   as (c) the path rule is refined")
print("=" * 96)
print(f"{'target':>7} {'steps':>5} | {'completeness maxerr':>19} {'rel':>8} "
      f"{'r':>8} {'R2id':>8} | {'law maxerr':>11} {'rel':>8} {'r':>8} "
      f"{'R2id':>8} | {'|DiD|':>8}")
for name in TARGETS:
    for s in STEPS:
        d_ = df[(df.target == name) & (df.steps == s)]
        comp = interact.error_stats(
            np.concatenate([d_.claim_a, d_.claim_b]),
            np.concatenate([d_.truth_a, d_.truth_b]))
        law = interact.error_stats(d_.claim_i, d_.truth_i)
        conv.append(dict(target=name, steps=int(s), completeness=comp, law=law))
for name in TARGETS:
    add_order([c for c in conv if c["target"] == name])
for c in conv:
    comp, law = c["completeness"], c["law"]
    print(f"{c['target']:>7} {c['steps']:>5} | {comp['max_abs_err']:>19.4f} "
          f"{comp['rel_err']:>8.4f} {comp['pearson']:>8.5f} "
          f"{comp['r2_identity']:>8.5f} | {law['max_abs_err']:>11.4f} "
          f"{law['rel_err']:>8.4f} {law['pearson']:>8.5f} "
          f"{law['r2_identity']:>8.5f} | {law['truth_scale']:>8.4f}")
print("\nconvergence order  log2(err(s)/err(2s))   (midpoint rule: expect ~2 "
      "until the bf16 floor is reached)")
for c in conv:
    if c["law"]["order"] == c["law"]["order"]:
        print(f"  {c['target']:>7} {c['steps']:>3}->{2*c['steps']:<3} "
              f"completeness {c['completeness']['order']:+.2f}   "
              f"law {c['law']['order']:+.2f}")
rep["convergence"] = conv

# ---------------------------------------------------------- the null it beats
# A conservation law that merely "correlates" is not a law, so the identity fit
# is reported beside what the same numbers score once the pairing is destroyed:
# each region-sum matched to a DIFFERENT item's double difference.  That is the
# wrong-prompt control in its strictest form -- same estimator, same scale, no
# correspondence -- and it is what "r = 0.99" has to be read against.
rng = np.random.default_rng(0)
null = []
for name in TARGETS:
    d_ = df[(df.target == name) & (df.steps == a.head_steps)]
    if len(d_) < 6:
        continue
    v = d_.claim_i.to_numpy(float)
    t = d_.truth_i.to_numpy(float)
    sh = [interact.error_stats(v, t[rng.permutation(len(t))]) for _ in range(200)]
    null.append(dict(
        target=name, n=int(len(v)),
        real_pearson=float(interact.error_stats(v, t)["pearson"]),
        real_rel_err=float(interact.error_stats(v, t)["rel_err"]),
        shuffled_pearson_mean=float(np.mean([r["pearson"] for r in sh])),
        shuffled_pearson_p95=float(np.percentile(
            [r["pearson"] for r in sh], 95)),
        shuffled_rel_err_mean=float(np.mean([r["rel_err"] for r in sh]))))
rep["null_floor"] = null
if null:
    print("\nTHE NULL FLOOR  (claim paired with another item's double difference)")
    print(f"{'target':>7} {'n':>4} {'r (real)':>10} {'r (shuffled)':>13} "
          f"{'p95':>8} | {'rel err real':>13} {'rel err shuffled':>17}")
    for r in null:
        print(f"{r['target']:>7} {r['n']:>4} {r['real_pearson']:>10.5f} "
              f"{r['shuffled_pearson_mean']:>13.5f} "
              f"{r['shuffled_pearson_p95']:>8.3f} | "
              f"{r['real_rel_err']:>13.5f} "
              f"{r['shuffled_rel_err_mean']:>17.3f}")

per_layer = []
for name in TARGETS:
    for l in LAYERS:
        d_ = df[(df.target == name) & (df.steps == a.head_steps)
                & (df.layer == l)]
        if not len(d_):
            continue
        per_layer.append(dict(target=name, layer=int(l),
                              law=interact.error_stats(d_.claim_i, d_.truth_i),
                              completeness=interact.error_stats(
                                  np.concatenate([d_.claim_a, d_.claim_b]),
                                  np.concatenate([d_.truth_a, d_.truth_b])),
                              noreg=interact.error_stats(d_.claim_noreg,
                                                         d_.truth_i)))
rep["per_layer"] = per_layer
print(f"\nAT HEAD STEPS ({a.head_steps}), PER LAYER")
print(f"{'target':>7} {'layer':>5} {'law rel':>9} {'law r':>9} "
      f"{'comp rel':>9} {'comp r':>9} {'no-registers rel':>17}")
for r in per_layer:
    print(f"{r['target']:>7} {r['layer']:>5} {r['law']['rel_err']:>9.4f} "
          f"{r['law']['pearson']:>9.5f} {r['completeness']['rel_err']:>9.4f} "
          f"{r['completeness']['pearson']:>9.5f} "
          f"{r['noreg']['rel_err']:>17.4f}")

print("\n" + "=" * 96)
print("(d) PROMPT-BLINDNESS   max |map(q_a) - map(q_b)| over items")
print("=" * 96)
bg = bl.groupby(["lens", "target", "layer"]).agg(
    n=("item", "count"), patch=("patch_maxdev", "max"),
    region=("region_maxdev", "max"), scale=("scale", "mean")).reset_index()
print(bg.to_string(index=False))
rep["prompt_blindness"] = st.jsonable(bg.round(10).to_dict("index"))
rep["exactly_zero"] = {
    lens: bool(bl[bl.lens == lens].patch_maxdev.max() == 0.0)
    for lens in bl.lens.unique()}
print("\nexactly zero:", rep["exactly_zero"])
if len(ct):
    cg = ct.groupby("kind").agg(
        n=("item", "count"), h_maxdev=("maxdev", "max")).reset_index()
    print("\nCONTROLS  (max over items)\n", cg.to_string(index=False))
    rep["controls"] = st.jsonable(cg.round(10).to_dict("index"))
    for k in ("rerun_same_prompt", "other_image_same_prompt"):
        d_ = ct[ct.kind == k]
        if len(d_) and "logit_patch" in d_:
            print(f"  {k:>24}: logit-lens patch maxdev "
                  f"{float(d_.logit_patch.max())!r}"
                  + (f"   VJ-Lens patch maxdev {float(d_.vj_patch.max())!r}"
                     if "vj_patch" in d_ and d_.vj_patch.notna().any() else ""))
    rep["rerun_vj_patch_maxdev"] = (
        float(ct[ct.kind == "rerun_same_prompt"].vj_patch.max())
        if "vj_patch" in ct and (ct.kind == "rerun_same_prompt").any()
        else float("nan"))

rep["minutes"] = round((time.time() - t0) / 60, 1)
json.dump(rep, open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
print(f"\nwrote {a.out}/{tag}_stats.json  ({rep['minutes']} min)", flush=True)
