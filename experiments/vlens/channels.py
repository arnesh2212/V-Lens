# SPDX-License-Identifier: Apache-2.0
"""Iteration on the gate: read the decision from the WHOLE prompt, not the answer slot alone.

Total verdict at layer l = sum over channels of  u^c . (h - mu^c) summed over that channel's
positions = P * u_img.(mean_img - mu_img) + T * u_txt.(mean_txt - mu_txt) + u_ans.(h_a - mu_ans).
Label-free.  Compared, per depth, with the answer-slot-only reading and with the identity curve
the gate already computed.  CPU only, from the saved cache.
"""
import sys, json; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from vlens.lens import cache, verdict, probes
from vlens import judge
from transformers import AutoProcessor
repo, tag = sys.argv[1], sys.argv[2]
acts, side = cache.load(f"results/gate/{tag}__eval.npz")
import os as _os
_VARS = _os.environ.get("VLM_VARIANTS", "SD_TYPO").split(",")
acts, side = cache.subset(acts, side, [it.get("variant") in _VARS for it in side["items"]])
print(f"using {side['n']} items in variants {_VARS}", flush=True)
G = json.load(open(f"results/gate/{tag}_gate.json"))
VL = verdict.VerdictLens.load(f"results/gate/{tag}__verdict_carrier.pt")
ANS = side["_extra"]["ANS"].astype(np.float32); its = side["items"]
y = np.array([judge.item_refused(it) for it in its], int)
tok = AutoProcessor.from_pretrained(repo).tokenizer
T = len(tok.encode(side["prompt"])) + 8          # carrier text tokens + template (constant across items)
P = side["image_tokens"]
print(f"{tag}: {len(y)} items, refused {y.mean():.2f}; P={P} image tokens, T~{T} text tokens")
print(f"{'depth':>6} {'ans only':>9} {'img sum':>9} {'txt sum':>9} {'TOTAL':>9} | {'identity':>9} {'probe':>7}")
ident = G["curves"]["identity(HiddenDetect)"]; pr = {r["layer"]: r["score"] for r in G["probe_refused"]}
out = {}
for l in VL.layers:
    v_ans = VL.read(ANS[:, l], "refusal", "ans", l)
    img_mean = acts[l].astype(np.float32).mean(1)
    v_img = P * VL.read(img_mean, "refusal", "img", l)
    txt_mean = side["_extra"][f"TXT{l}"].astype(np.float32)
    v_txt = T * VL.read(txt_mean, "refusal", "txt", l)
    total = v_ans + v_img + v_txt
    au = lambda s: probes.readout_score(s, y, shuffles=50)["auroc"]
    r = dict(frac=G["calibration"][str(l)]["frac"], ans=au(v_ans), img=au(v_img), txt=au(v_txt), total=au(total),
             identity=ident[str(l)]["auroc"], probe=pr.get(l, float("nan")))
    out[l] = r
    print(f"{r['frac']:>6.2f} {r['ans']:>9.3f} {r['img']:>9.3f} {r['txt']:>9.3f} {r['total']:>9.3f} | {r['identity']:>9.3f} {r['probe']:>7.3f}")
json.dump(out, open(f"results/gate/{tag}_total_verdict.json", "w"), indent=1)
