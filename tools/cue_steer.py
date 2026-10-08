# SPDX-License-Identifier: Apache-2.0
"""Does a cue's direction move the behaviour when written into the model? A causal test of the cue geometry.

Figure cue_geometry measures, in the answer-spot state, the angle between a cue's
direction and the direction of the behaviour it is about. Here each direction is
written into the answer spot at one block, at a dose scaled to the state's norm,
and the final answer is read again:

* evaluation look, on untouched pictures asked "test or real?": the look
  direction (test-dressed minus real-dressed pictures), the model's own answer
  direction (untouched pictures it calls test minus those it calls real), and a
  random direction, each pushed toward the answer the model did not give;
* sandbagging, on WMDP items under an honest instruction: the sandbagging
  instruction's direction (hazard minus honest instruction, same items) and the
  right-versus-wrong direction (wrong minus right answers under the honest
  instruction), each pushed toward answering wrong, and a random direction.

Every direction is fitted on one half of the records and written into the other
half, then the halves swap, so no record steers with a direction fitted on it.

    python3 tools/cue_steer.py --repo Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

EVAL_FAMS = ("openimages_orig_eval", "openimages_edit_eval", "vins_orig_eval", "vins_edit_eval",
             "nuscenes_edit_eval")


def _mod(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dressing(v):
    if v is None:
        return "plain"
    v = str(v)
    if "eval" in v or v in ("annotation_ui", "watermark"):
        return "test"
    if "deploy" in v or v in ("phone_screenshot", "dashcam_hud"):
        return "real"
    return None


def unit(v):
    return v / v.norm().clamp_min(1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/cue_steer")
    ap.add_argument("--depth", type=float, default=0.75)
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--square", type=int, default=448)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from PIL import Image
    from vlens import corpus as C
    from vlens.data import simonsays as SS
    from vlens.lens import scores
    from vlens.lens.hooks import ActivationRecorder
    from vlens.lens.write import AddAtPositions
    SC = _mod("experiments/simonsays/sandbag_channels.py", "sandchan")

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    LAST = m.n_layers - 1
    Lb = int(round(a.depth * LAST))
    say(f"{name}: writing at block {Lb} of {LAST}")
    T_ids, R_ids = list(TS.test), list(TS.real)
    L_ids = {c: list(TS.letters[c]) for c in "ABCD"}

    def run(item, u=None, alpha=0.0):
        """Answer-spot state at Lb and final logits, optionally with alpha*||h||*u added at the spot."""
        b = m.batch_many([item], side="right")
        spot = int(m.answer_spot(b)[0])
        ctx = AddAtPositions(m.layers[Lb], u, alpha, torch.tensor([spot])) if u is not None else None
        with torch.no_grad(), ActivationRecorder(m.layers, at=[Lb, LAST]) as rec:
            if ctx:
                with ctx:
                    m.forward(b)
            else:
                m.forward(b)
            h = rec.activations[Lb][0, spot].float().cpu()
            lg = m.unembed(rec.activations[LAST][0, spot][None].to(m.device))[0].float()
        return h, lg

    def tr_margin(lg):
        return float(lg[T_ids].max() - lg[R_ids].max())

    def letters(lg):
        return {c: float(lg[ids].max()) for c, ids in L_ids.items()}

    rng = np.random.default_rng(0)
    out = dict(repo=a.repo, block=Lb, n_layers=m.n_layers, alphas=a.alphas)
    t0 = time.time()

    # ------------------------------------------------ evaluation look
    recs = [r for r in SS.load(families=list(EVAL_FAMS), require_image=True) if r["kind"] == "test_real"]
    E = []
    for r in recs:
        g = _dressing((r.get("meta") or {}).get("edit_variant"))
        if g is None:
            continue
        it = dict(question=r["question"], image=C.load_image(r["image"], square=a.square),
                  system=r.get("system"), cue=r.get("cue"))
        h, lg = run(it)
        E.append(dict(id=r["id"], g=g, h=h, margin=tr_margin(lg), item=it))
    say(f"  eval: {len(E)} records read ({(time.time() - t0) / 60:.1f} min)")
    H = torch.stack([e["h"] for e in E])
    g = np.array([e["g"] for e in E]); said = np.array([e["margin"] > 0 for e in E])
    look = unit(H[g == "test"].mean(0) - H[g == "real"].mean(0))
    plain = np.where(g == "plain")[0]
    fold = rng.permutation(len(plain)) % 2
    rows = []
    for f in (0, 1):
        fit, tst = plain[fold != f], plain[fold == f]
        ans = unit(H[fit[said[fit]]].mean(0) - H[fit[~said[fit]]].mean(0))
        rnd = unit(torch.from_numpy(rng.standard_normal(H.shape[1])).float())
        for i in tst:
            e = E[i]
            sgn = 1.0 if e["margin"] <= 0 else -1.0          # toward the answer it did not give
            row = dict(id=e["id"], base=e["margin"], said_test=bool(e["margin"] > 0), runs={})
            for dn, u in (("look", look), ("answer", ans), ("random", rnd)):
                for al in a.alphas:
                    _h, lg = run(e["item"], sgn * u, al)
                    row["runs"][f"{dn}|{al}"] = tr_margin(lg)
            rows.append(row)
    out["eval"] = dict(n=len(rows), cos_look_answer=float(look @ unit(H[plain[said[plain]]].mean(0)
                                                                      - H[plain[~said[plain]]].mean(0))),
                       rows=rows)
    say(f"  eval steered: {len(rows)} ({(time.time() - t0) / 60:.1f} min)")

    # ------------------------------------------------ sandbagging instruction
    by_item = {}
    for r in SS.load(families=list(SC.FAMILIES)):
        by_item.setdefault(r["id"].rsplit("__", 1)[0], {})[r["condition"]] = r
    items = [v for v in by_item.values() if {"unlocked", "calibration_sandbagged", "hazard_sandbagged"} <= set(v)]
    S = []
    for v in items:
        base = v["unlocked"]
        img = Image.open(base["image"]).convert("RGB")
        pic = SC.compose(img, None, None, 896)
        one = {}
        for cond, system in (("honest", SC.HONEST), ("hazard", v["hazard_sandbagged"]["system"])):
            it = dict(question=base["question"], image=pic, system=system, cue=None)
            h, lg = run(it)
            L = letters(lg)
            one[cond] = dict(h=h, item=it, right=max(L, key=L.get) == base["expected"], letters=L)
        S.append(dict(id=base["id"].rsplit("__", 1)[0], expected=base["expected"], **one))
    say(f"  sandbag: {len(S)} items read ({(time.time() - t0) / 60:.1f} min)")
    Hh = torch.stack([s["honest"]["h"] for s in S]); Hz = torch.stack([s["hazard"]["h"] for s in S])
    right = np.array([s["honest"]["right"] for s in S])
    fold = rng.permutation(len(S)) % 2
    rows = []
    for f in (0, 1):
        fit, tst = np.where(fold != f)[0], np.where(fold == f)[0]
        sand = unit((Hz[fit] - Hh[fit]).mean(0))
        wrong = unit(Hh[fit[~right[fit]]].mean(0) - Hh[fit[right[fit]]].mean(0))
        rnd = unit(torch.from_numpy(rng.standard_normal(Hh.shape[1])).float())
        for i in tst:
            s = S[i]
            row = dict(id=s["id"], expected=s["expected"], base_right=bool(s["honest"]["right"]),
                       hazard_right=bool(s["hazard"]["right"]), runs={})
            for dn, u in (("sandbag", sand), ("wrong", wrong), ("random", rnd)):
                for al in a.alphas:
                    _h, lg = run(s["honest"]["item"], u, al)
                    L = letters(lg)
                    row["runs"][f"{dn}|{al}"] = dict(right=max(L, key=L.get) == s["expected"],
                                                    correct_margin=L[s["expected"]] - max(
                                                        v_ for c, v_ in L.items() if c != s["expected"]))
            rows.append(row)
    out["sandbag"] = dict(n=len(rows), cos_sand_wrong=float(unit((Hz - Hh).mean(0)) @ unit(
        Hh[~right].mean(0) - Hh[right].mean(0))), rows=rows)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(out, open(os.path.join(a.out, f"{name}__cue_steer.json"), "w"), default=float)

    # the headline, printed for the log
    ev = out["eval"]["rows"]
    for dn in ("look", "answer", "random"):
        say(f"  eval {dn}: flip rate by dose " + " ".join(
            f"{al}:{np.mean([(r['runs'][f'{dn}|{al}'] > 0) != r['said_test'] for r in ev]):.2f}" for al in a.alphas))
    sb = out["sandbag"]["rows"]
    say(f"  sandbag base accuracy {np.mean([r['base_right'] for r in sb]):.2f}")
    for dn in ("sandbag", "wrong", "random"):
        say(f"  sandbag {dn}: accuracy by dose " + " ".join(
            f"{al}:{np.mean([r['runs'][f'{dn}|{al}']['right'] for r in sb]):.2f}" for al in a.alphas))
    say("DONE cue_steer")


if __name__ == "__main__":
    main()
