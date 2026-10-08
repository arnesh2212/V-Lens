"""Five ways to edit a picture's meaning, in one coordinate system.

The paper so far demonstrates one intervention: exchange two concepts'
coordinates and watch the answer follow.  A swap is only one of the things a
coordinate system lets you do, and showing only the swap leaves open the
suspicion that it is a trick of that particular formula rather than a property
of the space.  So here is the whole taxonomy, every operation written in the
same basis, applied at the same positions, on the same photographs, under the
same budget, and graded by the same three numbers.

THE SPACE.  For a word ``t`` the lens vector at image position ``p`` of layer
``l`` is

    v_t[l, p] = d(logit of t at the answer) / d(h_l[p]),

the direction in that position's residual stream that raises ``t``'s logit at
the place the answer is written.  It is a gradient, not a fitted probe: nothing
is trained, and it is recomputed for every photograph, because transport out of
an image token is not transport out of a text token and neither is constant
across pictures.  Write ``u_t = v_t / ||v_t||`` for its direction and
``c_t = h . u_t`` for the *coordinate* of the activation along it.  Everything
below rewrites coordinates.  Nothing outside the span of the named directions is
touched, and no direction outside the image tokens is touched at all.

THE OPERATIONS.  ``A`` is the concept the photograph contains; ``B`` is the
concept the edit installs.  All five are pointed at the same goal -- make the
model give B's answer instead of A's -- so one rate compares all of them.
Writing ``c^0`` for the coordinate on the clean pass:

    ERASE       c_A <- (1 - lambda) c_A^0        take A's coordinate away
    AMPLIFY     c_B <- a c_B^0                   scale B's own coordinate
    INJECT      c_B <- c_B^0 + s ||h^0||         write B in from outside
    SWAP        c_A <- alpha c_B^0,
                c_B <- alpha c_A^0               exchange the two
    SWAPDEV     the same exchange, but of each
                coordinate's DEVIATION from the
                per-position baseline
    TRANSPLANT  c_B <- beta c_B^donor            copy a real photograph of B's

At the first edited layer these are exactly the increments they look like:
``h - c_A u_A``, ``h + (a-1) c_B u_B``, ``h + s||h|| u_B``, and the swap's
``h - c_A u_A - c_B u_B + c_B u_A + c_A u_B``.  Stated as assignments rather
than increments they also survive being applied across a band of layers, which
an increment does not: re-applying a gain of 16 at each of twenty layers
multiplies the coordinate by 16^20, and the first version of this experiment
duly overflowed.  An assignment is idempotent, so a band-wide edit means "hold
this coordinate at this value through the band" and the dose parameter means
what it says at every depth.

SWAPDEV is there because the first run of this experiment made the swap look
dead, and the reason turned out to be a definition rather than a fact.  A raw
coordinate ``h . u`` is mostly the *average* activation's projection onto ``u``;
the part that varies with the photograph -- the part the lens actually reports --
is the deviation from the per-position baseline.  Exchanging raw coordinates
therefore exchanges two numbers that are nearly the same, and moves almost
nothing.  Both are measured here, side by side, so a reader can see that the
operation which appears not to work and the operation which does are the same
operation stated against two different origins.

The operations differ in what they assume.  ERASE assumes the answer is held up by A's
coordinate.  AMPLIFY assumes there is already a little of B and asks whether
turning it up is enough -- if B's coordinate is near zero, multiplying it
changes almost nothing, and that is a result rather than a failure.  INJECT
assumes the direction is enough and supplies its own magnitude.  SWAP assumes
the two coordinates are interchangeable.  TRANSPLANT assumes nothing about
magnitude at all: it copies the number a genuine photograph of B produced.

THREE MEASUREMENTS, on every operation at every dose.

*the answer*  A short question the picture settles -- how many legs, how many
wheels, what colour -- scored between the two answers the concepts disagree
about.  Trials start with the model getting it right, so the unedited rate of
giving B's answer is zero by construction.

*the reading*  The lens score of the edited concept, measured in the same
coordinates the edit was made in, so an edit can be seen to register where it
was applied and not only at the output.  The lens vectors are the ones measured
on the clean pass; re-deriving them under the edit would let the ruler move with
the thing it is measuring.

*the damage*  ``||dh|| / ||h||`` at the touched positions, so an operation that
buys its effect by wrecking the residual stream cannot be mistaken for one that
does not.

And for every operation, at every dose, on the same photograph, a matched-norm
random control: the identical assignment with a random unit direction in place
of the lens direction, rescaled position by position so that ``||dh||`` is
exactly equal.  Anything the control also achieves is not evidence about the
lens -- and at a large enough dose it achieves a good deal, which is itself
worth seeing.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import OrderedDict

sys.path.insert(0, "code")
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

import jvlm
from jlens.hooks import ActivationRecorder
from jvlm import cocodata, corpus, readout, stats as st, tasks

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--n", type=int, default=150, help="trials (photograph x pair)")
ap.add_argument("--mean-images", type=int, default=60,
                help="photographs averaged for the per-position baseline the "
                     "readings are measured against")
ap.add_argument("--band", type=float, nargs=2, default=[0.12, 0.88],
                help="depth band edited, as a fraction of the LM")
ap.add_argument("--gen", type=int, default=1, help="generate text for the edited arm")
ap.add_argument("--gen-random", type=int, default=0,
                help="also generate for the control arm; it is scored on "
                     "logits either way")
ap.add_argument("--ops", nargs="*",
                default=["erase", "amplify", "inject", "swap", "swapdev",
                         "transplant"])
# Doses chosen from the smoke run so that every curve is on its own scale but
# none of them sits on a ceiling or in the regime where the matched-norm random
# control also works -- past ||dh||/||h|| of about 0.4 an edit stops being an
# edit and becomes damage, and the control says so.
ap.add_argument("--erase-doses", type=float, nargs="*", default=[1.0, 2.0, 4.0, 8.0])
ap.add_argument("--amplify-doses", type=float, nargs="*", default=[2.0, 4.0, 8.0, 16.0])
ap.add_argument("--inject-doses", type=float, nargs="*",
                default=[0.0125, 0.025, 0.05, 0.1])
ap.add_argument("--swap-doses", type=float, nargs="*", default=[1.0, 2.0, 4.0, 8.0])
ap.add_argument("--swapdev-doses", type=float, nargs="*", default=[1.0, 2.0, 4.0, 8.0])
ap.add_argument("--transplant-doses", type=float, nargs="*", default=[0.5, 1.0, 2.0, 4.0])
ap.add_argument("--donor-cache", type=int, default=8)
ap.add_argument("--max-cases", type=int, default=40, help="per operation")
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--out", default="results/edits")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lo = max(m.n_deepstack, 1)
BAND = [l for l in range(m.n_layers - 1)
        if a.band[0] <= l / (m.n_layers - 1) <= a.band[1] and l >= lo]
FINAL = m.n_layers - 1
sc = tasks.Scorer(m)
DOSES = {"erase": a.erase_doses, "amplify": a.amplify_doses,
         "inject": a.inject_doses, "swap": a.swap_doses,
         "swapdev": a.swapdev_doses, "transplant": a.transplant_doses}
OPS = [o for o in a.ops if o in DOSES]
print(m, f"\nediting layers {BAND[0]}..{BAND[-1]} ({len(BAND)} of {m.n_layers})",
      f"\noperations: {OPS}",
      f"\ndoses: " + "  ".join(f"{o}={DOSES[o]}" for o in OPS), flush=True)

# --------------------------------------------------------------- the questions
# Reused verbatim from the swap experiment, so the taxonomy is measured on the
# task the existing result was measured on.  The model is never shown either
# concept's name; only the picture can settle the answer.
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
PAIRS = []
for kind, (q, ansmap) in TASKS.items():
    ks = sorted(ansmap)
    for i, X in enumerate(ks):
        for Y in ks[i + 1:]:
            if ansmap[X] != ansmap[Y]:
                PAIRS += [(kind, X, Y), (kind, Y, X)]   # both directions
rng = np.random.default_rng(a.seed)
rng.shuffle(PAIRS)
print(f"{len(PAIRS)} ordered concept pairs whose answers differ", flush=True)


def ids_of(w):
    return cocodata.concept_token_ids(m.tokenizer, w)


def ans_ids(s):
    out = []
    for f in (s, " " + s, s.capitalize(), " " + s.capitalize()):
        t = m.tokenizer.encode(f, add_special_tokens=False)
        if len(t) == 1:
            out.append(t[0])
    return sorted(set(out))


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def says(txt, word):
    """Did the reply start with this answer, as a whole word?

    Without the boundary a reply of ``greengriffins`` counts as ``green``; the
    first version of this let exactly that through into the examples file.
    """
    return bool(re.match(r"\s*" + re.escape(word) + r"\b", txt, re.I))


def lens_vectors(batch, tokens, layers, positions):
    """``{layer: [P, d]}`` -- the lens vector of a word at every image position.

    One backward pass per surface form of the word; the gradient it returns
    already covers every layer, so the cost does not grow with the band.
    """
    pos = torch.as_tensor(positions, dtype=torch.long)
    out = {l: [] for l in layers}
    with (
        ActivationRecorder(m.layers, at=[*layers, FINAL],
                           start_graph_at=min(layers)) as rec,
        torch.enable_grad(),
    ):
        m.forward(batch)
        acts = [rec.activations[l] for l in layers]
        logits = m.unembed(rec.activations[FINAL][0, -1][None])[0]
        for j, t in enumerate(tokens):
            g = torch.autograd.grad(logits[t], acts,
                                    retain_graph=(j < len(tokens) - 1))
            for l, gg in zip(layers, g, strict=True):
                out[l].append(gg[0, pos.to(gg.device)].float())
    return {l: torch.stack(v).mean(0) for l, v in out.items()}


class Edit:
    """Every operation in the taxonomy, through one hook.

    Keeping the five in one function body is not tidiness: it is what makes the
    comparison exact.  The positions touched, the layers touched, the order the
    edit and the recording happen in, the precision the arithmetic is done in
    and the way the disturbance is measured are then identical by construction
    rather than by five separate promises.

    ``coef[layer]`` carries everything the assignment needs, all measured on the
    clean pass: the two coordinates ``ca0``/``cb0``, the activation norm ``n0``,
    the same two coordinates along the control's random directions
    ``c10``/``c20``, and the donor's coordinates ``cd``/``cdr``.

    ``arm="random"`` replaces the lens directions with random unit vectors and
    then rescales the result, position by position, so ``||dh||`` matches the
    real edit exactly.  The control therefore costs the residual stream the same
    damage and differs only in *where* the damage points.
    """

    def __init__(self, model, ipos, layers, op, dose, *, ua, ub, na, nb, mu,
                 coef, r1=None, r2=None, arm="edit"):
        self.m, self.ipos, self.layers = model, ipos, layers
        self.op, self.dose, self.arm = op, float(dose), arm
        self.ua, self.ub, self.na, self.nb = ua, ub, na, nb
        self.mu, self.coef = mu, coef
        self.r1, self.r2 = r1, r2
        self.maxpos = int(ipos.max())
        self.log: dict[int, dict] = {}
        self._h = []

    def _delta(self, sub, A, B, k):
        """The assignment, written once: move each coordinate to its target.

        ``k`` holds the clean-pass coefficients for whichever pair of directions
        is in play -- the two lens directions, or the control's two random ones
        -- so the control runs this identical function rather than a copy of it.

        The two-coordinate operations solve a 2x2 system rather than adding two
        independent increments.  Two concepts' lens vectors are not orthogonal
        (``r = u_A . u_B`` is recorded per trial and is nowhere near zero), so
        adding ``x u_A + y u_B`` with ``x`` and ``y`` read off separately lands
        on neither target: the ``u_B`` term moves the ``u_A`` coordinate as well.
        The solve makes the exchange an exchange.  When ``r = 0`` it reduces
        exactly to the increment form the swap was first written in.
        """
        if self.op == "none":
            return torch.zeros_like(sub)
        ca0, cb0, n0 = k["ca0"], k["cb0"], k["n0"]
        ca = (sub * A).sum(-1, keepdim=True)
        cb = (sub * B).sum(-1, keepdim=True)
        if self.op == "erase":
            return ((1.0 - self.dose) * ca0 - ca) * A
        if self.op == "amplify":
            return (self.dose * cb0 - cb) * B
        if self.op == "inject":
            return (cb0 + self.dose * n0 - cb) * B
        if self.op == "transplant":
            return (self.dose * k["cd"] - cb) * B
        if self.op == "swap":
            ta, tb = self.dose * cb0, self.dose * ca0
        elif self.op == "swapdev":
            ma, mb = k["ma"], k["mb"]
            ta = ma + self.dose * (cb0 - mb)
            tb = mb + self.dose * (ca0 - ma)
        else:
            raise ValueError(self.op)
        r = k["r"]
        det = (1.0 - r * r).clamp_min(1e-3)
        da, db = ta - ca, tb - cb
        return ((da - r * db) / det) * A + ((db - r * da) / det) * B

    def _hook(self, layer):
        def fn(mod, inp, out):
            tup = not torch.is_tensor(out)
            h = out[0] if tup else out
            if h.shape[1] <= self.maxpos:
                return out                      # a cached decode step: no image
            h = h.clone()
            p = self.ipos.to(h.device)
            sub = h[0, p].float()
            k = self.coef[layer]
            A = self.ua[layer]
            B = self.ub[layer]
            d = self._delta(sub, A, B, k["edit"])
            if self.arm == "random":
                dr = self._delta(sub, self.r1[layer], self.r2[layer], k["random"])
                s = (d.norm(dim=-1, keepdim=True)
                     / dr.norm(dim=-1, keepdim=True).clamp_min(1e-9))
                d = dr * s
            new = sub + d
            if layer not in self.log:           # prefill only, never a decode step
                dev = new - self.mu[layer]
                pa = (dev * A).sum(-1)
                pb = (dev * B).sum(-1)
                # v = ||v|| u, so the reading in logit units is the projection
                # scaled by the lens vector's own length at that position
                self.log[layer] = dict(
                    read_src_u=float(pa.mean()),
                    read_dst_u=float(pb.mean()),
                    read_src_v=float((pa * self.na[layer]).mean()),
                    read_dst_v=float((pb * self.nb[layer]).mean()),
                    disturb=float((d.norm(dim=-1)
                                   / sub.norm(dim=-1).clamp_min(1e-6)).mean()),
                )
            h[0, p] = new.to(h.dtype)
            return (h, *out[1:]) if tup else h
        return fn

    def summary(self) -> dict:
        if not self.log:
            return {}
        keys = next(iter(self.log.values())).keys()
        return {k: float(np.mean([v[k] for v in self.log.values()])) for k in keys}

    def __enter__(self):
        self.log = {}
        for l in self.layers:
            self._h.append(self.m.layers[l].register_forward_hook(self._hook(l)))
        return self

    def __exit__(self, *e):
        for x in self._h:
            x.remove()
        self._h = []


# ------------------------------------------------------------ the baseline mean
pool = cocodata.concept_items(9000, min_area=0.10, seed=17)
by_c: dict[str, list] = {}
for it in pool:
    by_c.setdefault(it["concept"], []).append(it)

mean_pool = corpus.coco_images(a.mean_images + 20, seed=5)
b0 = m.chat("Describe this image.", [corpus.load_image(mean_pool[0], square=a.square)])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in BAND}
nm = 0
for p_ in mean_pool:
    if nm >= a.mean_images:
        break
    b = m.chat("Describe this image.", [corpus.load_image(p_, square=a.square)])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    A_ = readout.residuals(m, b, BAND)
    for l in BAND:
        acc[l] += A_[l][ip].float().cpu().double()
    nm += 1
MU = {l: (acc[l] / max(nm, 1)).float().to(m.device) for l in BAND}
del acc
print(f"{P} image tokens; per-position baseline from {nm} photographs "
      f"({time.time()-t0:.0f}s)", flush=True)


# --------------------------------------------------------------- donor photos
DONOR: OrderedDict[tuple, tuple] = OrderedDict()


def donor_for(kind, B):
    """A real photograph of B, on which the model already gives B's answer.

    A transplant that copies a coordinate off a photograph the model misreads
    would be copying the wrong number, so the donor has to pass the same test
    the trial photograph passes.
    """
    key = (kind, B)
    if key in DONOR:
        DONOR.move_to_end(key)
        return DONOR[key]
    q, ansmap = TASKS[kind]
    tb = ans_ids(ansmap[B])
    other = sorted({ansmap[x] for x in ansmap if ansmap[x] != ansmap[B]})
    tnot = sorted({i for s in other for i in ans_ids(s)})
    got = None
    for it in by_c.get(B, [])[:4]:
        bb = m.chat(q, [corpus.load_image(it["path"], square=a.square)])
        ip = readout.image_positions(m, bb)
        if len(ip) != P:
            continue
        lg = sc.last_logits(bb).float()
        if tnot and float(lg[tb].max()) <= float(lg[tnot].max()):
            continue
        A_ = readout.residuals(m, bb, BAND)
        got = ({l: A_[l][ip].half() for l in BAND}, it["path"])
        break
    DONOR[key] = got
    while len(DONOR) > a.donor_cache:
        DONOR.popitem(last=False)
    return got


# ---------------------------------------------------------------------- the run
rows: list[dict] = []
cases: dict[str, list] = {o: [] for o in OPS}
n_done = 0
used: dict[str, int] = {}
for pi in range(20_000):
    if n_done >= a.n:
        break
    kind, A, B = PAIRS[pi % len(PAIRS)]
    q, ansmap = TASKS[kind]
    ia, ib = ids_of(A), ids_of(B)
    ta, tb = ans_ids(ansmap[A]), ans_ids(ansmap[B])
    if not (ia and ib and ta and tb):
        continue
    k = used.get(f"{kind}|{A}", 0)
    cand = by_c.get(A, [])
    if k >= len(cand):
        continue
    used[f"{kind}|{A}"] = k + 1
    it = cand[k]

    b = m.chat(q, [corpus.load_image(it["path"], square=a.square)])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    # one clean forward gives both the answer logits and the clean coordinates
    A0 = readout.residuals(m, b, BAND)
    lg0 = m.unembed(A0[FINAL][-1:].to(m.device))[0].float().cpu()
    # the model has to answer the picture correctly first, or an edit has
    # nothing to move; this also fixes the unedited follow rate at exactly zero
    if float(lg0[ta].max()) <= float(lg0[tb].max()):
        continue
    dn = donor_for(kind, B) if "transplant" in OPS else None
    if "transplant" in OPS and dn is None:
        continue

    h0 = {l: A0[l][ip].float() for l in BAND}
    del A0
    va = lens_vectors(b, ia, BAND, ip.tolist())
    vb = lens_vectors(b, ib, BAND, ip.tolist())
    ua = {l: unit(v) for l, v in va.items()}
    ub = {l: unit(v) for l, v in vb.items()}
    na = {l: v.norm(dim=-1) for l, v in va.items()}
    nb = {l: v.norm(dim=-1) for l, v in vb.items()}
    del va, vb
    g = torch.Generator(device="cpu").manual_seed(a.seed * 7919 + n_done)
    r1 = {l: unit(torch.randn(P, m.d_model, generator=g)).to(m.device) for l in BAND}
    r2 = {l: unit(torch.randn(P, m.d_model, generator=g)).to(m.device) for l in BAND}

    def proj(x, u):
        return (x * u).sum(-1, keepdim=True)

    COEF = {}
    for l in BAND:
        n0 = h0[l].norm(dim=-1, keepdim=True)
        hd = dn[0][l].float() if dn else None
        COEF[l] = {
            "edit": dict(ca0=proj(h0[l], ua[l]), cb0=proj(h0[l], ub[l]),
                         ma=proj(MU[l], ua[l]), mb=proj(MU[l], ub[l]), n0=n0,
                         r=proj(ua[l], ub[l]),
                         cd=(proj(hd, ub[l]) if dn else None)),
            "random": dict(ca0=proj(h0[l], r1[l]), cb0=proj(h0[l], r2[l]),
                           ma=proj(MU[l], r1[l]), mb=proj(MU[l], r2[l]), n0=n0,
                           r=proj(r1[l], r2[l]),
                           cd=(proj(hd, r2[l]) if dn else None)),
        }
    del h0

    m0 = float(lg0[tb].max() - lg0[ta].max())
    txt0 = sc.generate(b, max_new_tokens=6).strip() if a.gen else ""

    # how far from orthogonal the two concepts' lens directions are; the swap
    # is only a clean exchange to the extent this is small
    cos_ab = float(np.mean([float(COEF[l]["edit"]["r"].abs().mean()) for l in BAND]))
    common = dict(i=n_done, kind=kind, src=A, dst=B, image_id=int(it["image_id"]),
                  path=it["path"], q=q, src_ans=ansmap[A], dst_ans=ansmap[B],
                  donor_path=(dn[1] if dn else ""), cos_ab=cos_ab)

    def record(op, dose, arm, ed, lg, txt):
        s = ed.summary()
        mm = float(lg[tb].max() - lg[ta].max())
        top = int(lg.argmax())
        cands = sorted(set(ta) | set(tb))
        top2 = int(cands[int(lg[cands].argmax())])
        rows.append(dict(common, op=op, dose=float(dose), arm=arm,
                         margin=mm, d_margin=mm - m0,
                         follow=int(mm > 0), picked_dst=int(top2 in tb),
                         said_dst=int(top in tb), said_src=int(top in ta),
                         text=txt[:24], **s))
        return s

    # the clean pass, read through the identical code path, so a difference in
    # the readings is a difference the edit made and not the arithmetic
    with Edit(m, ip, BAND, "none", 0.0, ua=ua, ub=ub, na=na, nb=nb, mu=MU,
              coef=COEF) as ed0:
        lgc = sc.last_logits(b).float()
    s0 = record("clean", 0.0, "clean", ed0, lgc, txt0)

    for op in OPS:
        for dose in DOSES[op]:
            for arm in ("edit", "random"):
                with Edit(m, ip, BAND, op, dose, ua=ua, ub=ub, na=na, nb=nb,
                          mu=MU, coef=COEF, r1=r1, r2=r2, arm=arm) as ed:
                    lg = sc.last_logits(b).float()
                    txt = ""
                    if (a.gen and arm == "edit") or (a.gen_random and arm == "random"):
                        txt = sc.generate(b, max_new_tokens=6).strip()
                s = record(op, dose, arm, ed, lg, txt)
                # an example is worth printing only if it is real end to end:
                # the model gave the source answer before and gives the target
                # answer after, both as whole words
                if (arm == "edit" and a.gen and len(cases[op]) < a.max_cases
                        and says(txt0, ansmap[A]) and says(txt, ansmap[B])):
                    cases[op].append(dict(
                        op=op, dose=float(dose), kind=kind, src=A, dst=B,
                        question=q, path=it["path"], image_id=int(it["image_id"]),
                        donor_path=(dn[1] if dn else ""),
                        answer_before=txt0[:40], answer_after=txt[:40],
                        expected_before=ansmap[A], expected_after=ansmap[B],
                        margin_before=round(m0, 3),
                        margin_after=round(float(lg[tb].max() - lg[ta].max()), 3),
                        read_dst_before=round(s0.get("read_dst_u", 0.0), 4),
                        read_dst_after=round(s.get("read_dst_u", 0.0), 4),
                        read_src_before=round(s0.get("read_src_u", 0.0), 4),
                        read_src_after=round(s.get("read_src_u", 0.0), 4),
                        disturb=round(s.get("disturb", 0.0), 4)))
    n_done += 1
    if n_done % 10 == 0:
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")
        print(f"  {n_done}/{a.n} trials  {time.time()-t0:.0f}s  "
              f"cases { {o: len(cases[o]) for o in OPS} }", flush=True)

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
json.dump(st.jsonable(cases), open(f"{a.out}/{tag}_cases.json", "w"), indent=1)

# ----------------------------------------------------------------- the numbers
ed = df[df.arm != "clean"]
agg = ed.groupby(["op", "arm", "dose"]).agg(
    n=("follow", "count"), follow=("follow", "mean"),
    picked=("picked_dst", "mean"), says_dst=("said_dst", "mean"),
    says_src=("said_src", "mean"), d_margin=("d_margin", "mean"),
    disturb=("disturb", "mean"), read_dst=("read_dst_u", "mean"),
    read_src=("read_src_u", "mean"))
clean = df[df.arm == "clean"]
base = dict(n=int(len(clean)), follow=float(clean.follow.mean()),
            says_dst=float(clean.said_dst.mean()),
            says_src=float(clean.said_src.mean()),
            read_dst=float(clean.read_dst_u.mean()),
            read_src=float(clean.read_src_u.mean()),
            # the no-op arm runs the identical hook with a zero delta, so this
            # is a check that the code path itself changes nothing
            identity_max_abs_d_margin=float(clean.d_margin.abs().max())
            if len(clean) else float("nan"))

# does the effect grow with the dose?  A curve that does not is a curve the
# reader should not be shown as one.
MONO = {}
for (op_, arm_), d_ in ed.groupby(["op", "arm"]):
    per = d_.groupby("dose").follow.mean()
    MONO[(op_, arm_)] = (float(sps.spearmanr(per.index, per.values).statistic)
                         if per.nunique() > 1 and len(per) > 2 else float("nan"))

top = {}
for op in OPS:
    d = ed[(ed.op == op) & (ed.dose == max(DOSES[op]))]
    e, r = d[d.arm == "edit"], d[d.arm == "random"]
    if not len(e):
        continue
    mean, lo_, hi_ = st.cluster_bootstrap_ci(e, "follow", cluster="dst", n=4000)
    top[op] = dict(dose=float(max(DOSES[op])), n=int(len(e)),
                   follow=mean, ci_lo=lo_, ci_hi=hi_,
                   follow_random=float(r.follow.mean()) if len(r) else float("nan"),
                   d_margin=float(e.d_margin.mean()),
                   d_margin_random=float(r.d_margin.mean()) if len(r) else float("nan"),
                   still_says_source=float(e.said_src.mean()),
                   disturb=float(e.disturb.mean()),
                   follow_per_disturb=float(e.follow.mean()
                                            / max(e.disturb.mean(), 1e-6)),
                   d_read_dst=float(e.read_dst_u.mean() - base["read_dst"]),
                   d_read_src=float(e.read_src_u.mean() - base["read_src"]),
                   monotone_rho=MONO.get((op, "edit"), float("nan")),
                   monotone_rho_random=MONO.get((op, "random"), float("nan")),
                   mcnemar_vs_random=st.mcnemar(
                       pd.concat([e.assign(cond="edit"), r.assign(cond="random")]),
                       "edit", "random", "follow") if len(r) else {},
                   n_cases=len(cases.get(op, [])))

agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\nFIVE EDITS IN THE SAME COORDINATE SYSTEM\n", agg.round(3).to_string())
print("\nunedited:", {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in base.items()})
print("\nAT THE TOP DOSE OF EACH OPERATION")
for op, d in top.items():
    print(f"  {op:11s} follow {d['follow']:.3f} "
          f"[{d['ci_lo']:.3f},{d['ci_hi']:.3f}]  random {d['follow_random']:.3f}  "
          f"margin {d['d_margin']:+.2f} (random {d['d_margin_random']:+.2f})  "
          f"||dh||/||h|| {d['disturb']:.3f}  "
          f"reading of the target {d['d_read_dst']:+.3f}  "
          f"dose rho {d['monotone_rho']:+.2f}  "
          f"examples {d['n_cases']}")
json.dump({"repo": a.repo, "layers": [BAND[0], BAND[-1]], "n_layers_edited": len(BAND),
           "n_trials": int(df.i.nunique()), "n_rows": int(len(df)),
           "mean_images": nm, "image_tokens": int(P),
           "doses": st.jsonable(DOSES), "unedited": st.jsonable(base),
           "by_op": st.jsonable(agg.round(4).to_dict("index")),
           "top_dose": st.jsonable(top),
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)

for op in OPS:
    for c in cases[op][:3]:
        print(f"  [{op}] {c['src']}->{c['dst']} ({c['kind']}) dose {c['dose']}: "
              f"{c['answer_before']!r} -> {c['answer_after']!r}")
print("done", flush=True)
