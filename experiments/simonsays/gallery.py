# SPDX-License-Identifier: Apache-2.0
"""Five pictures, five scenarios, and what happens when the map is acted on.

One record from each of the benchmark's five kinds of picture. For each,
V-Write's per-patch attribution at the reading block, the top fifth of the
patches greyed out in the photograph itself, and a random fifth greyed out as
the control. The intervention is in pixels rather than in the residual stream,
so the reply underneath each panel is a reply the model actually gave to a
picture a person could look at.

Twelve records are measured per source and the median one is shown. Showing
the first that works makes the figure an accident of file order, and showing
the best makes it an advertisement; the median is the one a reader can take
as typical, and the range across the twelve is reported beside it.

    python3 experiments/simonsays/gallery.py --repo Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
import torch  # noqa: E402

#: one per source, named for what the picture is doing. The source is what is
#: asked for rather than a family, since a family can be small enough that the
#: model answers none of it correctly and there is then nothing to show.
SCENARIOS = [("Generated scenes", "gemini"), ("Open Images", "openimages"),
             ("nuScenes", "nuscenes"), ("Interfaces", "vins"),
             ("WMDP carrier", "wmdp")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/gallery")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--k", type=float, default=0.2, help="share of patches greyed out")
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--steps-ig", type=int, default=8)
    ap.add_argument("--tries", type=int, default=120, help="records looked at per scenario")
    ap.add_argument("--candidates", type=int, default=12,
                    help="usable records measured per scenario, of which the median is shown")
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays
    from vlens.lens import attribution as AT, readout, scores

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(f"{a.out}/pictures", exist_ok=True)
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    L = max(int(round(a.frac * (m.n_layers - 1))), 1)
    rng = np.random.default_rng(a.seed)
    say(f"{name}: V-Write at block {L} of {m.n_layers}")

    recs = simonsays.load(require_image=True)
    by_source = {}
    for r in simonsays.scorable(recs):
        if r["kind"] in ("yesno", "letter", "test_real"):
            by_source.setdefault(str(r["family"]).split("_")[0], []).append(r)
    # the edited families first, since an edit is what makes a picture worth
    # showing, and a deterministic order so the figure is reproducible
    for v in by_source.values():
        v.sort(key=lambda r: (0 if "edit" in r["family"] else 1, str(r["id"])))

    def reply(b):
        with torch.no_grad():
            g = m.hf.generate(**{k: v for k, v in b.items() if not k.startswith("_")},
                              max_new_tokens=a.gen, do_sample=False,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(g[0, b["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    out = []
    for label, fam in SCENARIOS:
        pool = by_source.get(fam) or []
        if not pool:
            say(f"  {fam}: nothing in the set"); continue
        cands = []
        for r in pool[: a.tries]:
            if len(cands) >= a.candidates:
                break
            img = C.load_image(r["image"], square=a.square)
            b = m.chat(r["question"], [img], system=r["system"], cue=r["cue"])
            pos = readout.image_positions(m, b)
            P = len(pos)
            if P < 16:
                continue
            fn = scores.score_fn(TS, r["kind"], r.get("expected"))
            with torch.no_grad():
                _, H, _, lg0 = AT.states_at(m, b, [L])
            s0 = float(fn(None, lg0).float().cpu()[0])
            fn, s0 = scores.orient(fn, r["kind"], s0)
            if s0 <= 0:
                continue                      # nothing to read toward
            h = H[L]
            attr = AT.path_reading(m, b, fn, L, pos, h, h.mean(0, keepdim=True).expand_as(h),
                                   steps=a.steps_ig, log=say)
            rows, cols = readout.image_grid(m, b, P)
            nk = max(int(round(a.k * P)), 1)
            top = np.argsort(-np.asarray(attr, float))[:nk]
            rnd = rng.permutation(P)[:nk]

            def greyed(sel, img=img, rows=rows, cols=cols):
                arr = np.array(img.convert("RGB"), dtype=np.uint8).copy()
                ph, pw = arr.shape[0] // rows, arr.shape[1] // cols
                for p_ in sel:
                    y, x = int(p_) // cols, int(p_) % cols
                    arr[y * ph:(y + 1) * ph, x * pw:(x + 1) * pw] = 128
                from PIL import Image
                return Image.fromarray(arr)

            base = reply(b)
            shots = {"original": img, "top": greyed(top), "random": greyed(rnd)}
            said, score = {}, {}
            for k, im in shots.items():
                bb = m.chat(r["question"], [im], system=r["system"], cue=r["cue"])
                with torch.no_grad():
                    _, _, _, lg = AT.states_at(m, bb, [L])
                score[k] = float(fn(None, lg).float().cpu()[0])
                said[k] = base if k == "original" else reply(bb)
            cands.append(dict(label=label, family=r["family"], source=fam, id=str(r["id"]),
                              question=simonsays.safe_question(r), cue=r.get("cue"), block=L,
                              grid=[rows, cols], n_patches=P, k=a.k, s0=s0,
                              score=score, reply=said,
                              attr=[float(x) for x in np.asarray(attr, float)],
                              top=[int(x) for x in top], random=[int(x) for x in rnd],
                              _shots=shots))
        if not cands:
            say(f"  {label}: nothing usable in {a.tries} tries"); continue
        # the median of the candidates by how much covering the top patches
        # costs, so the picture shown is a typical one rather than the best
        # one found or whichever happened to come first in the file
        eff = np.array([c["score"]["original"] - c["score"]["top"] for c in cands])
        j = int(np.argsort(eff)[len(eff) // 2])
        chosen = cands[j]
        for k, im in chosen.pop("_shots").items():
            im.save(f"{a.out}/pictures/{name}__{fam}__{k}.png")
        chosen["picture"] = {k: f"pictures/{name}__{fam}__{k}.png"
                             for k in ("original", "top", "random")}
        chosen["n_candidates"] = len(cands)
        chosen["effect_rank"] = int(np.argsort(np.argsort(eff))[j])
        for c in cands:
            c.pop("_shots", None)
        out.append(chosen)
        sc = chosen["score"]
        say(f"  {label:<16} {chosen['family']:<22} the median of {len(cands)}: "
            f"score {sc['original']:.2f} -> top {sc['top']:.2f}, random {sc['random']:.2f} "
            f"(the set covers {eff.min():.2f} to {eff.max():.2f})")
    if not out:
        say("nothing to show"); return
    json.dump(dict(repo=a.repo, block=L, k=a.k, scenarios=out),
              open(f"{a.out}/{name}__gallery.json", "w"), default=float)
    say(f"wrote {a.out}/{name}__gallery.json")
    say("DONE gallery")


if __name__ == "__main__":
    main()
