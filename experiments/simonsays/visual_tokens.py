# SPDX-License-Identifier: Apache-2.0
"""Can an instrument read a visual token, and does it read the right patch?

Every other experiment here reads the answer spot. This one reads the image
itself. The LatentLens paper's finding is that the logit lens badly
underestimates how interpretable visual tokens are, because the unembedding is
the wrong reference space for them; the question this asks is whether a
transport into the final basis fixes that, and whether it beats going to a
contextual text space instead.

Two measurements, both on the model's own words so that no outside vocabulary
is imposed:

    python3 experiments/simonsays/visual_tokens.py --repo Qwen/Qwen3-VL-8B-Instruct --n 150
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from vlens.lens.fitall import SHARED as FA_SHARED  # noqa: E402

STOP = set("""a an the this that these those is are was were be been being of in on at to for with
by from as it its their his her they he she we you i and or but not no nor so if then than there
here what which who whom whose when where why how all any both each few more most other some such
only own same too very can will just should now image picture photo shows showing showed appears
seems looks like about into over under above below between during before after again further once""".split())


def content_words(text: str) -> list[str]:
    """The words of a description that carry its content, lower cased."""
    ws = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    return [w for w in ws if len(w) > 2 and w not in STOP]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/visualtokens")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--pool", type=int, default=800)
    ap.add_argument("--layers", type=int, default=FA_SHARED["layers"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--square", type=int, default=FA_SHARED["square"])
    ap.add_argument("--cap", type=int, default=FA_SHARED["cap"])
    ap.add_argument("--heldout", type=int, default=FA_SHARED["heldout"])
    ap.add_argument("--jac-prompts", type=int, default=FA_SHARED["jac_prompts"])
    ap.add_argument("--results", default="results/simonsays")
    ap.add_argument("--corpus", type=int, default=FA_SHARED["corpus"],
                    help="prompts the transports are fitted on")
    ap.add_argument("--steps", type=int, default=FA_SHARED["steps"])
    ap.add_argument("--tokens-per-step", type=int, default=FA_SHARED["tokens_per_step"])
    ap.add_argument("--lam-grid", type=float, nargs="*", default=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    ap.add_argument("--latent-captions", type=int, default=FA_SHARED["latent_captions"])
    ap.add_argument("--fit-batch", type=int, default=FA_SHARED["fit_batch"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--naming", type=int, default=250,
                    help="pictures used to fit the naming transport; 0 turns it off")
    ap.add_argument("--gen", type=int, default=48)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays
    from vlens.lens import attribution as AT, baselines as B, fitall as FA, readout
    from vlens.lens.hooks import ActivationRecorder
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from handoff import pairs_for

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    LAST = m.n_layers - 1
    GRID = FA.layer_grid(m, a.layers)
    say(f"{name}: reading visual tokens at {GRID}")

    recs = simonsays.load(require_image=True)
    pairs = pairs_for(recs)
    rng = np.random.default_rng(0)
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    if a.pool:
        pairs = pairs[: a.pool]
    if not a.n:
        a.n = len(pairs)
    say(f"{len(pairs)} original/edited pairs, each with a mask of what changed")

    # ---- fit every transport on a corpus of prompts none of these items is in
    spare = FA.fitting_pool(recs, a.cap)
    want = set(a.only) if a.only else set(FA.SHARED_WANT)
    maps, latent, FIT, meta = FA.fit_all(
        m, spare, C.load_image, layers=GRID, corpus=a.corpus, heldout=a.heldout,
        square=a.square, steps=a.steps, tokens_per_step=a.tokens_per_step,
        jac_prompts=a.jac_prompts, latent_captions=a.latent_captions,
        fit_batch=a.fit_batch, want=tuple(sorted(want)), results=a.results, log=say)

    # the transport whose target is what the model calls things, rather than
    # what it will answer. The alignment between patches and words comes from
    # V-Write, so this is the two instruments doing one job.
    if a.naming:
        from vlens.lens import naming as NM
        nm_items = B.corpus_items(spare, C.load_image, n=a.naming, square=a.square, seed=3)
        FIT |= B.fit_ids(nm_items)
        nmap, nmeta = NM.fit_naming_map(m, nm_items, GRID, log=say)
        if nmap is not None:
            maps["v_read_name"] = nmap
            say(f"    {nmeta}")
            tn = NM.fit_naming_tuned(m, nm_items, GRID, lr=lr, steps=min(a.steps, 120), log=say)
            if tn is not None:
                maps["tuned_name"] = tn
    names = [*maps, "logit"] + (["latent"] if latent else [])

    # nothing any transport was fitted on may be read here
    pairs = [(o, e) for o, e in pairs if str(o["id"]) not in FIT and str(e["id"]) not in FIT]
    say(f"{len(pairs)} pairs left after removing everything a transport was fitted on")
    rows, t0 = [], time.time()
    for o, e in pairs:
        if len(rows) >= a.n:
            break
        img = C.load_image(o["image"], square=a.square)
        # the model's own words for this picture are the target vocabulary
        bc = m.chat("Describe this image in one sentence.", [img])
        with torch.no_grad():
            g = m.hf.generate(**{k: v for k, v in bc.items() if not k.startswith("_")},
                              max_new_tokens=a.gen, do_sample=False,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
        cap = tok.decode(g[0, bc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        words = set(content_words(cap))
        if len(words) < 3:
            continue
        ids = set()
        for w in words:
            for form in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                t = tok.encode(form, add_special_tokens=False)
                if t:
                    ids.add(t[0])
        b = m.chat(o["question"], [img], system=o["system"], cue=o["cue"])
        pos = readout.image_positions(m, b)
        P = len(pos)
        G = int(round(P ** 0.5))
        if P < 16:
            continue
        with torch.no_grad(), ActivationRecorder(m.layers, at=GRID) as rec:
            m.forward(b)
            H = {l: rec.activations[l][0, pos.to(rec.activations[l].device)].float().cpu() for l in GRID}
        inside = None
        if G * G == P:
            sq = AT.edit_mask(o["image"], e["image"], side=a.square)
            inside = AT.pool_mask(sq, G) > 0.05
        hit = {k: [] for k in names}
        hit_in = {k: [] for k in names}
        hit_out = {k: [] for k in names}
        for j, l in enumerate(GRID):
            h = H[l].to(m.device)
            cand = {"logit": h}
            for k, M in maps.items():
                A_, b_ = M[l]
                cand[k] = h @ A_.T.to(m.device) + b_.to(m.device)
            got = {}
            with torch.no_grad():
                for k, v in cand.items():
                    top = m.unembed(v.to(torch.bfloat16)).float().topk(a.topk, -1).indices.cpu().numpy()
                    got[k] = np.array([len(ids & set(r_.tolist())) > 0 for r_ in top])
                if latent is not None:
                    top = latent.similarity_logits(h.float(), l).topk(a.topk, -1).indices.cpu().numpy()
                    got["latent"] = np.array([len(ids & set(r_.tolist())) > 0 for r_ in top])
            for k, v in got.items():
                hit[k].append(float(v.mean()))
                if inside is not None and inside.any() and (~inside).any():
                    hit_in[k].append(float(v[inside].mean()))
                    hit_out[k].append(float(v[~inside].mean()))
        rows.append(dict(id=o["id"], dataset=o["dataset"], n_patches=int(P),
                         n_words=len(words), legible=hit,
                         legible_inside=hit_in, legible_outside=hit_out))
        if len(rows) % 10 == 0:
            say(f"  {len(rows)}/{a.n} images ({(time.time() - t0) / 60:.1f} min)  " +
                "  ".join(f"{k} {np.mean([r['legible'][k][len(GRID) // 2] for r in rows]):.2f}"
                          for k in names))

    if not rows:
        say("no usable images"); return
    summ = dict(repo=a.repo, layers=GRID, depths=[l / LAST for l in GRID], n=len(rows),
                topk=a.topk, instruments=names)
    summ["legible"] = {k: [float(np.mean([r["legible"][k][j] for r in rows]))
                           for j in range(len(GRID))] for k in names}
    summ["inside"] = {k: [float(np.mean([r["legible_inside"][k][j] for r in rows
                                         if r["legible_inside"][k]])) for j in range(len(GRID))]
                      for k in names}
    summ["outside"] = {k: [float(np.mean([r["legible_outside"][k][j] for r in rows
                                          if r["legible_outside"][k]])) for j in range(len(GRID))]
                       for k in names}
    summ["placement_gap"] = {k: [summ["inside"][k][j] - summ["outside"][k][j]
                                 for j in range(len(GRID))] for k in names}
    json.dump(dict(summary=summ, rows=rows), open(f"{a.out}/{name}__visualtokens.json", "w"),
              default=float)
    say("\nshare of visual tokens that decode to a word from the model's own description")
    say("  depth      " + "".join(f"{d:>8.2f}" for d in summ["depths"]))
    for k in names:
        say(f"  {k:<10} " + "".join(f"{v:>8.2f}" for v in summ["legible"][k]))
    say("\ninside the edited region minus outside it")
    for k in names:
        say(f"  {k:<10} " + "".join(f"{v:>8.2f}" for v in summ["placement_gap"][k]))
    say("DONE visual_tokens")


if __name__ == "__main__":
    main()
