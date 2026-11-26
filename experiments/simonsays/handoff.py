# SPDX-License-Identifier: Apache-2.0
"""Where a visual fact lives, and when it is handed over to the words.

The dataset gives matched pairs: one photograph and the same photograph with a
single thing changed -- a sign's text swapped, a bearer removed, a stamp added.
The model answers the two differently. That is a controlled counterfactual on a
real image, which is what the CLEVR version of this experiment buys with
synthetic renders, and it lets the question be asked directly:

  1. Take the original run. Replace, at one early block, only the image-token
     states that lie inside the edited region, using the states the edited run
     produced there. Nothing else changes -- same prompt, same pixels
     everywhere else.
  2. Read the run at every block. When does it stop predicting the original
     answer and start predicting the edited one?
  3. Then ask where the fact is being carried once it has moved. At a later
     block, take the patched run's image tokens, or its text tokens, or its
     answer spot alone, and put them into the untouched original run. Whichever
     swap changes the answer is where the fact is at that depth.
  4. V-Write says in advance which group should matter. Compare what it
     predicts against what the swap does.

    python3 experiments/simonsays/handoff.py --repo Qwen/Qwen3-VL-8B-Instruct --n 120
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
import torch  # noqa: E402


def pairs_for(recs, root=None):
    """Match every edited record to the original record it was made from.

    An edited record's ``image_number`` is the number of the source image, not
    of its own file, and its ``prompt_number`` is its own index in the edited
    file, so the two sides cannot be joined on the prompt number. What does
    join them is the picture the edit was made from together with the question
    being asked, so the key is the source image number, the subcategory and the
    question text; the cue is included when both sides carry one.

    Two sources do not join that way. nuScenes edits name the file they were
    made from rather than a number, and the questions written for them are
    authority questions where the originals only carry reading ones, so no
    original asks what the edit asks. The control a transplant needs is the
    same question put to the unedited picture, which is a record we can make:
    the original's picture under the edit's own question. Two hundred pairs
    that were being dropped, and the only ones in the set taken from a moving
    vehicle.
    """
    stem = lambda p_: os.path.basename(str(p_ or "")).rsplit(".", 1)[0]
    orig, by_picture = {}, {}
    for r in recs:
        if r["variant"] != "original":
            continue
        mt = r.get("meta") or {}
        k = (r["dataset"], r["subcategory"], mt.get("image_number"),
             (r.get("question") or "").strip(), (r.get("cue") or "").strip())
        orig.setdefault(k, r)
        loose = (r["dataset"], r["subcategory"], mt.get("image_number"))
        orig.setdefault(loose, r)
        by_picture.setdefault((r["dataset"], stem(r["image"])), r)
    out = []
    for r in recs:
        if r["variant"] != "edited":
            continue
        mt = r.get("meta") or {}
        num = mt.get("image_number")
        if num is None:
            src = str(mt.get("source_image") or "")
            digits = "".join(c for c in os.path.basename(src).split(".")[0] if c.isdigit())
            num = int(digits) if digits else None
        k = (r["dataset"], r["subcategory"], num,
             (r.get("question") or "").strip(), (r.get("cue") or "").strip())
        o = orig.get(k) or orig.get((r["dataset"], r["subcategory"], num))
        if o is None:
            src = by_picture.get((r["dataset"], stem(mt.get("source_image"))))
            if src is not None:
                # the picture is the original's; everything asked of it is the
                # edit's, so that the only difference between the two sides is
                # the edit itself
                o = dict(r, image=src["image"], variant="original",
                         family=r["family"].replace("_edit_", "_orig_"),
                         meta=dict(mt, paired_by="source image"))
        if o is not None and o["image"] != r["image"]:
            out.append((o, r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/handoff")
    ap.add_argument("--datasets", nargs="*",
                    default=["openimages", "nuscenes", "vins", "gemini"])
    ap.add_argument("--n", type=int, default=120, help="pairs the model answers differently")
    ap.add_argument("--pool", type=int, default=1200)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--patch-frac", type=float, default=0.35,
                    help="depth at which the edited region is transplanted")
    ap.add_argument("--swap-fracs", type=float, nargs="*", default=[0.5, 0.65, 0.8])
    ap.add_argument("--layers", type=int, default=13)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--thr", type=float, default=28, help="pixel difference that counts as edited")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays
    from vlens.lens import attribution as AT, readout, tuned
    from vlens.lens.hooks import ActivationRecorder, Patch

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    LAST = m.n_layers - 1
    GRID = sorted({int(round(x)) for x in np.linspace(max(m.n_deepstack, 0), LAST, a.layers)})
    LP = max(int(round(a.patch_frac * LAST)), 1)
    say(f"{name}: transplanting at block {LP} (depth {LP / LAST:.2f}); reading at {GRID}")

    recs = [r for r in simonsays.load(require_image=True) if r["dataset"] in a.datasets]
    P = pairs_for(recs, simonsays.ROOT)
    say(f"{len(P)} matched original/edited pairs")
    rng = np.random.default_rng(0)
    P = [P[i] for i in rng.permutation(len(P))]
    if a.pool:
        P = P[: a.pool]

    def run(rec, patch=None):
        """Forward one record, optionally with a Patch context; returns states and logits."""
        img = C.load_image(rec["image"], square=a.square)
        b = m.chat(rec["question"], [img], system=rec["system"], cue=rec["cue"])
        S = b["input_ids"].shape[1]
        pos = readout.image_positions(m, b)
        ctx = patch(b) if patch else None
        with torch.no_grad(), ActivationRecorder(m.layers, at=[*GRID, LP]) as rec_:
            if ctx is not None:
                with ctx:
                    m.forward(b)
            else:
                m.forward(b)
            H = {l: rec_.activations[l][0].float().cpu() for l in {*GRID, LP}}
        lg = m.unembed(H[LAST][S - 1][None].to(m.device))
        return b, S, pos, H, lg

    if not a.n:
        a.n = len(P)
    rows, t0 = [], time.time()
    for o, e in P:
        if len(rows) >= a.n:
            break
        try:
            bo, So, poso, Ho, lgo = run(o)
            be, Se, pose, He, lge = run(e)
        except Exception:  # noqa: BLE001
            continue
        if len(poso) != len(pose) or len(poso) < 16:
            continue                                  # the two runs must line up token for token
        t_o = int(lgo[0].argmax()); t_e = int(lge[0].argmax())
        if t_o == t_e:
            continue                                  # the edit did not change the answer
        # the score this experiment is about: the edited answer over the original one
        fn = lambda hL, lg, i=t_e, j=t_o: (torch.log_softmax(lg.float(), -1)[..., i]
                                           - torch.log_softmax(lg.float(), -1)[..., j])
        G = int(round(len(poso) ** 0.5))
        if G * G != len(poso):
            G = int(len(poso) ** 0.5)
        sq = AT.edit_mask(o["image"], e["image"], thr=a.thr, side=a.square)
        pm = AT.pool_mask(sq, G) if G * G == len(poso) else np.zeros(len(poso), np.float32)
        inside = np.where(pm > 0.05)[0]
        if len(inside) < 2 or len(inside) > 0.6 * len(poso):
            continue                                  # no edit found, or the whole picture changed
        outside = np.setdiff1d(np.arange(len(poso)), inside)
        if len(outside) < len(inside):
            continue
        ctrl = rng.choice(outside, size=len(inside), replace=False)

        base_o = float(fn(None, lgo).float().cpu()[0])
        base_e = float(fn(None, lge).float().cpu()[0])

        def transplant(sel, noise=False):
            """The original run with the selected image states taken from the edited run."""
            take = poso[torch.as_tensor(sel)]
            val = He[LP][pose[torch.as_tensor(sel)]][None]
            if noise:
                g = torch.Generator().manual_seed(int(sel[0]))
                val = (Ho[LP][take][None] +
                       torch.randn(val.shape, generator=g) * Ho[LP][take].std())
            with torch.no_grad(), Patch(m.layers[LP], take, val.to(m.device)), \
                    ActivationRecorder(m.layers, at=GRID) as r_:
                m.forward(bo)
                H = {l: r_.activations[l][0].float().cpu() for l in GRID}
            return H, float(fn(None, m.unembed(H[LAST][So - 1][None].to(m.device))).float().cpu()[0])

        Hp, s_edit = transplant(inside)
        _, s_ctrl = transplant(ctrl)
        _, s_noise = transplant(inside, noise=True)

        # ---- where is the fact carried, later on?
        swaps = {}
        for f in a.swap_fracs:
            l = max(int(round(f * LAST)), 1)
            l = min(GRID, key=lambda g_: abs(g_ - l))
            for gname, take in (("image tokens", poso),
                                ("text tokens", readout.text_positions(m, bo)),
                                ("answer spot", torch.tensor([So - 1]))):
                val = Hp[l][take][None]
                with torch.no_grad(), Patch(m.layers[l], take, val.to(m.device)):
                    with ActivationRecorder(m.layers, at=[LAST]) as r_:
                        m.forward(bo)
                        hL = r_.activations[LAST][0, So - 1]
                swaps[f"{l}|{gname}"] = float(fn(None, m.unembed(hL[None])).float().cpu()[0])

        # ---- what V-Write said should matter, measured on the original run
        w = {}
        for gname, take in (("image tokens", poso),
                            ("text tokens", readout.text_positions(m, bo))):
            h = Ho[LP][take]
            base = h.mean(0, keepdim=True).expand_as(h)
            w[gname] = float(AT.path_reading(m, bo, fn, LP, take, h, base, steps=a.steps, log=say).sum())

        rows.append(dict(
            orig_id=o["id"], edit_id=e["id"], dataset=o["dataset"],
            subcategory=o["subcategory"], edit_family=(e.get("meta") or {}).get("edit_family"),
            n_image_tokens=int(len(poso)), n_edited=int(len(inside)),
            token_orig=tok.decode([t_o]), token_edit=tok.decode([t_e]),
            score_original_run=base_o, score_edited_run=base_e,
            score_after_transplant=s_edit, score_control_region=s_ctrl, score_noise=s_noise,
            recovered=(s_edit - base_o) / max(base_e - base_o, 1e-6),
            recovered_control=(s_ctrl - base_o) / max(base_e - base_o, 1e-6),
            depth_curve={str(l): float(fn(None, m.unembed(Hp[l][So - 1][None].to(m.device))).float().cpu()[0])
                         for l in GRID},
            depth_curve_original={str(l): float(fn(None, m.unembed(Ho[l][So - 1][None].to(m.device))).float().cpu()[0])
                                  for l in GRID},
            swaps=swaps, vwrite=w,
        ))
        if len(rows) % 10 == 0:
            r_ = np.mean([x["recovered"] for x in rows])
            c_ = np.mean([x["recovered_control"] for x in rows])
            say(f"  {len(rows)}/{a.n} pairs ({(time.time() - t0) / 60:.1f} min) "
                f"edited answer recovered {r_:.2f} of the way, control region {c_:.2f}")

    if not rows:
        say("no usable pairs"); return
    mean = lambda f: float(np.mean([f(x) for x in rows if np.isfinite(f(x))]))
    summ = dict(
        repo=a.repo, n=len(rows), block=LP, depth=LP / LAST, grid=GRID,
        recovered=mean(lambda x: x["recovered"]),
        recovered_control=mean(lambda x: x["recovered_control"]),
        flipped=mean(lambda x: float(x["score_after_transplant"] > 0)),
        flipped_control=mean(lambda x: float(x["score_control_region"] > 0)),
        flipped_noise=mean(lambda x: float(x["score_noise"] > 0)),
        depth_curve={str(l): mean(lambda x, l=l: x["depth_curve"][str(l)]) for l in GRID},
        depth_curve_original={str(l): mean(lambda x, l=l: x["depth_curve_original"][str(l)]) for l in GRID},
        swaps={k: mean(lambda x, k=k: x["swaps"].get(k, np.nan))
               for k in sorted({k for x in rows for k in x["swaps"]})},
        vwrite={k: mean(lambda x, k=k: x["vwrite"].get(k, np.nan))
                for k in sorted({k for x in rows for k in x["vwrite"]})},
    )
    json.dump(dict(summary=summ, rows=rows), open(f"{a.out}/{name}__handoff.json", "w"), default=float)
    say(f"\n{len(rows)} pairs, transplant at block {LP}")
    say(f"  edited answer recovered: {summ['recovered']:.2f} of the way from the original run to the edited one")
    say(f"  same count of tokens from outside the edit: {summ['recovered_control']:.2f}")
    say(f"  answer flipped: edit region {summ['flipped']:.2f}, control region {summ['flipped_control']:.2f}, "
        f"noise {summ['flipped_noise']:.2f}")
    say("  which group carries the fact later on (score after swapping that group in):")
    for k, v in summ["swaps"].items():
        say(f"    {k:<28} {v:>7.2f}")
    say("  what V-Write attributed at the transplant block:")
    for k, v in summ["vwrite"].items():
        say(f"    {k:<28} {v:>7.2f}")
    say("DONE handoff")


if __name__ == "__main__":
    main()
