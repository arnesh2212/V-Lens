# SPDX-License-Identifier: Apache-2.0
"""Every reading the lens family offers, scored on both labels, from the caches.

axis in {refusal, starters, yesno, harm}  x  channel in {ans, txt, img-sum, img-max, total}
x  depth, for (A) input harmfulness and (B) the model's own decision, beside the identity
transport and the labelled probe.  Selection is done on the 2B and REPORTED on the 8B,
so the best cell is not chosen on the model it is claimed for.  CPU only.
"""
import sys, json, os
sys.path.insert(0, "code")
import numpy as np
from jvlm import cache, verdict, probes, judge
VARS = os.environ.get("VLM_VARIANTS", "SD_TYPO").split(",")
AXES = ["refusal", "starters", "yesno", "harm"]

def sweep(tag):
    acts, side = cache.load(f"results/gate/{tag}__eval.npz")
    acts, side = cache.subset(acts, side, [it.get("variant") in VARS for it in side["items"]])
    VL = verdict.VerdictLens.load(f"results/gate/{tag}__verdict_carrier.pt")
    G = json.load(open(f"results/gate/{tag}_gate.json"))
    ANS = side["_extra"]["ANS"].astype(np.float32); its = side["items"]
    y_h = np.array([it["harmful"] for it in its], int)
    y_r = np.array([judge.refused(it) for it in its], int)
    P = side["image_tokens"]; T = 26
    ident = G["curves"]["identity@ans"]; identr = G["curves"]["identity(HiddenDetect)"]
    prh = {r["layer"]: r["score"] for r in G["probe_harmful"]}; prr = {r["layer"]: r["score"] for r in G["probe_refused"]}
    rows = []
    for l in VL.layers:
        f = G["calibration"][str(l)]["frac"]
        img = acts[l].astype(np.float32); txt = side["_extra"][f"TXT{l}"].astype(np.float32)
        for ax in AXES:
            v_ans = VL.read(ANS[:, l], ax, "ans", l)
            v_txt = T * VL.read(txt, ax, "txt", l)
            v_imgs = P * VL.read(img.mean(1), ax, "img", l)
            v_imgm = VL.read(img, ax, "img", l).max(1)
            for ch, v in (("ans", v_ans), ("txt", v_txt), ("img-sum", v_imgs), ("img-max", v_imgm), ("total", v_ans + v_txt + v_imgs)):
                for lab, y in (("harm", y_h), ("refused", y_r)):
                    au = probes.readout_score(v, y, shuffles=20)["auroc"]
                    rows.append(dict(layer=l, frac=f, axis=ax, channel=ch, label=lab, auroc=au, sep=abs(au - 0.5) + 0.5))
        rows.append(dict(layer=l, frac=f, axis="identity", channel="ans", label="harm", auroc=ident[str(l)]["auroc"], sep=abs(ident[str(l)]["auroc"]-.5)+.5))
        rows.append(dict(layer=l, frac=f, axis="identity", channel="ans", label="refused", auroc=identr[str(l)]["auroc"], sep=abs(identr[str(l)]["auroc"]-.5)+.5))
        rows.append(dict(layer=l, frac=f, axis="probe", channel="ans", label="harm", auroc=prh[l], sep=prh[l]))
        rows.append(dict(layer=l, frac=f, axis="probe", channel="ans", label="refused", auroc=prr[l], sep=prr[l]))
    return rows

R = {t: sweep(t) for t in ("Qwen3-VL-2B-Instruct", "Qwen3-VL-8B-Instruct")}
json.dump(R, open("results/gate/axes_sweep.json", "w"), indent=1)
import collections
for lab in ("harm", "refused"):
    print(f"\n==== label: {lab}  (sep = AUROC folded about 0.5; sign is a free choice for a monitor) ====")
    print(f"{'axis':<9}{'channel':<9}" + "".join(f"{'2B@%.2f'%f:>9}" for f in sorted({r['frac'] for r in R['Qwen3-VL-2B-Instruct']})) + "   | " + "".join(f"{'8B@%.2f'%f:>9}" for f in sorted({r['frac'] for r in R['Qwen3-VL-8B-Instruct']})))
    for ax in AXES + ["identity", "probe"]:
        chs = ["ans", "txt", "img-sum", "img-max", "total"] if ax in AXES else ["ans"]
        for ch in chs:
            line = f"{ax:<9}{ch:<9}"
            for t in ("Qwen3-VL-2B-Instruct", "Qwen3-VL-8B-Instruct"):
                cells = sorted([r for r in R[t] if r["axis"] == ax and r["channel"] == ch and r["label"] == lab], key=lambda r: r["frac"])
                line += "".join(f"{r['sep']:>9.3f}" for r in cells) + ("   | " if t.endswith("2B-Instruct") else "")
            print(line)
    # honest selection: best label-free cell on the 2B at each depth index -> its 8B value
    print("  -> best label-free (axis, channel) chosen on the 2B per depth, reported on the 8B:")
    fr2 = sorted({r["frac"] for r in R["Qwen3-VL-2B-Instruct"]}); fr8 = sorted({r["frac"] for r in R["Qwen3-VL-8B-Instruct"]})
    for i, (f2, f8) in enumerate(zip(fr2, fr8)):
        c2 = [r for r in R["Qwen3-VL-2B-Instruct"] if r["frac"] == f2 and r["label"] == lab and r["axis"] in AXES]
        best = max(c2, key=lambda r: r["sep"])
        c8 = [r for r in R["Qwen3-VL-8B-Instruct"] if r["frac"] == f8 and r["label"] == lab and r["axis"] == best["axis"] and r["channel"] == best["channel"]]
        id8 = [r for r in R["Qwen3-VL-8B-Instruct"] if r["frac"] == f8 and r["label"] == lab and r["axis"] == "identity"]
        print(f"     depth {f8:.2f}: {best['axis']}/{best['channel']} (2B {best['sep']:.3f}) -> 8B {c8[0]['sep'] if c8 else float('nan'):.3f}   identity 8B {id8[0]['sep']:.3f}")
