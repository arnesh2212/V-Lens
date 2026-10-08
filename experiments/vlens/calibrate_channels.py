# SPDX-License-Identifier: Apache-2.0
"""Calibrate every channel by patching it, then read the decision with calibrated weights.

The gate calibrated one channel, the answer slot.  The channel decomposition
showed the decision is readable from the text tokens first.  This closes the
loop without labels: patch each channel in turn (the answer slot; all text
positions; one image region) with another item's activations at the same
positions, measure how far the refusal margin at the last layer moves, and fit

    delta_m  ~  a_ans (u_ans . d_ans) + a_txt (u_txt . sum d_txt) + a_img (u_img . sum d_img) + b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--pairs", type=int, default=200, help="patch pairs per depth")
ap.add_argument("--region", type=int, default=3, help="image region side, in patches, for the image channel patch")
ap.add_argument("--out", default="results/gate")
a = ap.parse_args()

import vlens  # noqa: E402
from vlens.lens import cache, probes, readout, verdict  # noqa: E402
from vlens import corpus, judge  # noqa: E402
from vlens.lens.hooks import ActivationRecorder  # noqa: E402

name = a.repo.split("/")[-1]
say = lambda *x: print(*x, flush=True)
T0 = time.time()

m = vlens.load_vlm(a.repo)
acts, side = cache.load(f"{a.out}/{name}__eval.npz")
import os as _os
_VARS = _os.environ.get("VLM_VARIANTS", "SD_TYPO").split(",")
acts, side = cache.subset(acts, side, [it.get("variant") in _VARS for it in side["items"]])
print(f"using {side['n']} items in variants {_VARS}", flush=True)
VL = verdict.VerdictLens.load(f"{a.out}/{name}__verdict_carrier.pt")
G = json.load(open(f"{a.out}/{name}_gate.json"))
ANS = side["_extra"]["ANS"].astype(np.float32)
its = side["items"]; N = len(its)
y = np.array([judge.item_refused(it) for it in its], int)
PROMPT = side["prompt"]; READ = VL.layers; L = m.n_layers - 1
say(f"{name}: {N} items, refused {y.mean():.2f}; depths {READ}")

# the refusal direction the lens was fitted on is not stored in the lens; the
# gate's identity baseline used it.  Recover the same r from the eval set's own
# behaviour at the last layer (cos with the fitted r was reported by the gate).
r = verdict.dim_direction(ANS[:, -1], y)

# geometry from one batch: image positions, text positions, answer slot
b0 = m.chat(PROMPT, [corpus.load_image(its[0]["path"], square=448)])
IP = readout.image_positions(m, b0); S = b0["input_ids"].shape[1]; A_POS = S - 1
TP = readout.text_positions(m, b0); TP = TP[TP != A_POS]
grid = int(round(len(IP) ** 0.5))
say(f"geometry: {len(IP)} image tokens ({grid}x{grid}), {len(TP)} text tokens, answer at {A_POS}")


def residuals_at(batch, layer):
    """Residuals at ``layer`` and the last block, one forward."""
    with ActivationRecorder(m.layers, at=[layer, L]) as rec:
        m.forward(batch)
        return rec.activations[layer][0].detach(), rec.activations[L][0].detach()


def patched_margin(batch, layer, pos, donor_vals):
    def hook(mod, inp, out):
        tup = not torch.is_tensor(out)
        h = (out[0] if tup else out).clone()
        h[0, pos] = donor_vals.to(h.device, h.dtype)
        return (h, *out[1:]) if tup else h
    hd = m.layers[layer].register_forward_hook(hook)
    try:
        with ActivationRecorder(m.layers, at=[L]) as rec:
            m.forward(batch)
            return float(rec.activations[L][0, A_POS].float().cpu() @ r)
    finally:
        hd.remove()


def region_positions(rng):
    """A random square of image patches, as token positions."""
    k = a.region; i0 = int(rng.integers(0, grid - k + 1)); j0 = int(rng.integers(0, grid - k + 1))
    cells = [(i0 + i) * grid + (j0 + j) for i in range(k) for j in range(k)]
    return IP[torch.tensor(cells)]


rng = np.random.default_rng(3)
sub = rng.permutation(N)[: a.pairs]
out = {}
for l in READ:
    t = time.time()
    u_ans, u_txt, u_img = (VL.dir("refusal", c, l).numpy() for c in ("ans", "txt", "img"))
    rows = []
    for k in sub:
        j = int(rng.integers(N))
        if j == k:
            continue
        bt = m.chat(PROMPT, [corpus.load_image(its[k]["path"], square=448)])
        bd = m.chat(PROMPT, [corpus.load_image(its[j]["path"], square=448)])
        if bd["input_ids"].shape[1] != S:
            continue
        ht, hLt = residuals_at(bt, l)
        hd, _ = residuals_at(bd, l)
        m0 = float(hLt[A_POS].float().cpu() @ r)
        ht32, hd32 = ht.float().cpu().numpy(), hd.float().cpu().numpy()
        rp = region_positions(rng)
        rec = dict(group=int(k))
        for ch, pos, u in (("ans", torch.tensor([A_POS]), u_ans), ("txt", TP, u_txt), ("img", rp, u_img)):
            delta = (hd32[pos.numpy()] - ht32[pos.numpy()]).sum(0)          # summed over the patched positions
            rec[f"pred_{ch}"] = float(delta @ u)
            rec[f"pred_id_{ch}"] = float(delta @ r.numpy()) if ch == "ans" else float("nan")
            rec[f"dm_{ch}"] = patched_margin(bt, l, pos, hd[pos.to(hd.device)]) - m0
        rows.append(rec)
    R = {k_: np.array([rw[k_] for rw in rows]) for k_ in rows[0]}
    # per-channel calibration: dm_c ~ a_c pred_c + b_c, held out by image
    cal = {ch: verdict.calibrate(R[f"pred_{ch}"], R[f"dm_{ch}"], R["group"]) for ch in ("ans", "txt", "img")}
    cal["identity_ans"] = verdict.calibrate(R["pred_id_ans"], R["dm_ans"], R["group"])
    out[l] = dict(frac=G["calibration"][str(l)]["frac"], n=len(rows), calibration=cal,
                  minutes=(time.time() - t) / 60)
    say(f"  L{l:>2} f={out[l]['frac']:.2f}  held-out R^2  ans {cal['ans']['r2_heldout']:.3f} "
        f"(identity {cal['identity_ans']['r2_heldout']:.3f})  txt {cal['txt']['r2_heldout']:.3f}  "
        f"img {cal['img']['r2_heldout']:.3f}   n={len(rows)}  {(time.time()-t)/60:.1f} min")

# ---- detection with calibrated weights, from the cache ----------------------
say("\ndecision AUROC by reading, with calibrated channel weights a_c:")
say(f"{'depth':>6} {'ans':>7} {'txt':>7} {'img':>7} {'sum':>7} {'calibrated':>11} | {'identity':>9} {'probe':>7}")
ident = G["curves"]["identity(HiddenDetect)"]; pr = {rw["layer"]: rw["score"] for rw in G["probe_refused"]}
T = len(TP); P = len(IP)
for l in READ:
    v_ans = VL.read(ANS[:, l], "refusal", "ans", l)
    v_img = P * VL.read(acts[l].astype(np.float32).mean(1), "refusal", "img", l)
    v_txt = T * VL.read(side["_extra"][f"TXT{l}"].astype(np.float32), "refusal", "txt", l)
    c = out[l]["calibration"]
    v_cal = c["ans"]["a"] * v_ans + c["txt"]["a"] * v_txt + c["img"]["a"] * v_img
    au = lambda s_: probes.readout_score(s_, y, shuffles=50)["auroc"]
    d = dict(ans=au(v_ans), txt=au(v_txt), img=au(v_img), sum=au(v_ans + v_txt + v_img), calibrated=au(v_cal),
             identity=ident[str(l)]["auroc"], probe=pr.get(l, float("nan")), weights={k: c[k]["a"] for k in ("ans", "txt", "img")})
    out[l]["detection"] = d
    say(f"{out[l]['frac']:>6.2f} {d['ans']:>7.3f} {d['txt']:>7.3f} {d['img']:>7.3f} {d['sum']:>7.3f} {d['calibrated']:>11.3f} | "
        f"{d['identity']:>9.3f} {d['probe']:>7.3f}")

json.dump(out, open(f"{a.out}/{name}_channels.json", "w"), indent=1, default=float)
say(f"\nwrote {a.out}/{name}_channels.json   {(time.time() - T0) / 60:.1f} min")
