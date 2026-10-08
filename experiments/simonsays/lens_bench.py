# SPDX-License-Identifier: Apache-2.0
"""The head-to-head: V-Read against every published lens, one shared protocol.

Each baseline is fitted the way its own paper fits it, and every instrument is
fitted on exactly the same corpus and read through the same final norm and
unembedding, so the only difference between the numbers is the transport.

The fitting corpus is drawn from the records the behaviour pass did not keep,
so no state used to fit any instrument is ever scored.

    python3 experiments/simonsays/lens_bench.py --repo Qwen/Qwen3-VL-8B-Instruct

On the general dataset the same shared fit is read, unchanged, on the COCO
families the behaviour pass stored (``--dataset coco``), the held-out KL is
measured on COCO prompts, and V-Read fitted on COCO photographs
(tools/coco_fit.py) is read beside the shared instruments when its cache entry
exists. Nothing is fitted there: every photograph the COCO fit used is kept out
of the KL prompts and out of the scored records, checked against the fit's own
identifiers.

    VLENS_NO_FIT=1 VLENS_COCO=/workspace/temp/vlm/data_root/coco python3 experiments/simonsays/lens_bench.py \
        --repo Qwen/Qwen3-VL-8B-Instruct --results results/coco --families coco_pope coco_mcq \
        --only v_read tuned jacobian --kl-on coco --boot 300 --out results/coco_lensbench
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from vlens.lens.fitall import SHARED as FA_SHARED  # noqa: E402


from vlens.lens.metrics import auroc  # noqa: E402


from vlens.lens.metrics import boot  # noqa: E402


def coco_fit_entry(model_name: str, layers, cache: str, log=print):
    """``(path, maps, fit_ids, meta)`` of the fit tools/coco_fit.py cached for this
    model at exactly these blocks, or None.

    It is told apart from the shared fit by what it was fitted on: every one of
    its identifiers is a COCO photograph (``coco:<file>``). Only the identifier
    list is read to decide, through a memory map, so the shared entry beside it
    is never loaded twice.
    """
    from vlens.lens import fitall as FA
    cands = []
    for sp in glob.glob(os.path.join(cache, "*.pt.spec")):
        try:
            spec = json.loads(open(sp).read())
        except Exception:  # noqa: BLE001
            continue
        p = sp[: -len(".spec")]
        if (str(spec[0]).split("/")[-1] == model_name.split("/")[-1]
                and [int(x) for x in spec[2]] == [int(x) for x in layers] and os.path.exists(p)):
            cands.append((os.path.getmtime(p), p))
    for _, p in sorted(cands, reverse=True):
        try:
            try:
                ids = torch.load(p, map_location="cpu", mmap=True, weights_only=False)["fit_ids"]
            except (TypeError, RuntimeError):       # an older torch, or a file it cannot map
                ids = torch.load(p, map_location="cpu", weights_only=False)["fit_ids"]
        except Exception as e:  # noqa: BLE001
            log(f"  could not read {p} ({type(e).__name__})")
            continue
        if ids and all(str(i).startswith("coco:") for i in ids):
            maps, _lat, fit_ids, meta = FA._load(p)
            return p, maps, fit_ids, meta
    return None


def coco_kl_records(records, n: int, exclude: set, seed: int = 5) -> list:
    """``n`` COCO records, one per photograph, none of whose photographs is in ``exclude``.

    A photograph is drawn first and then one of the questions asked about it,
    so the prompts cover ``n`` different pictures and the three kinds of
    question in proportion.
    """
    by_img: dict = {}
    for r in records:
        by_img.setdefault(os.path.basename(r["image"]), []).append(r)
    imgs = sorted(k for k in by_img if k not in exclude)
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(imgs))[:n]
    return [by_img[imgs[i]][int(rng.integers(len(by_img[imgs[i]])))] for i in sorted(pick)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--results", default="results/simonsays")
    ap.add_argument("--out", default="results/lensbench")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--corpus", type=int, default=FA_SHARED["corpus"], help="prompts in the shared fitting corpus")
    ap.add_argument("--heldout", type=int, default=FA_SHARED["heldout"], help="prompts kept back to measure KL on")
    ap.add_argument("--cap", type=int, default=FA_SHARED["cap"], help="the behaviour pass cap, to stay disjoint from it")
    ap.add_argument("--layers", type=int, default=FA_SHARED["layers"], help="how many blocks to fit and report at")
    ap.add_argument("--steps", type=int, default=FA_SHARED["steps"])
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--tokens-per-step", type=int, default=FA_SHARED["tokens_per_step"])
    ap.add_argument("--rank", type=int, default=FA_SHARED["rank"])
    ap.add_argument("--lr-grid", type=float, nargs="*", default=[1.0, 0.1, 0.01],
                    help="the published rate first; the one with the lowest held-out KL is used")
    ap.add_argument("--lr-probe-steps", type=int, default=40)
    ap.add_argument("--lr-probe-tokens", type=int, default=2 ** 18)
    ap.add_argument("--anchor-tokens", type=int, default=2 ** 18)
    ap.add_argument("--lam-grid", type=float, nargs="*", default=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
                    help="V-Read's ridge penalty, chosen on held-out KL the same way the "
                         "tuned lens learning rate is")
    ap.add_argument("--jac-prompts", type=int, default=FA_SHARED["jac_prompts"])
    ap.add_argument("--jac-dim-batch", type=int, default=8)
    ap.add_argument("--jac-square", type=int, default=0,
                    help="image side for the Jacobian fit; 0 uses --square")
    ap.add_argument("--latent-captions", type=int, default=FA_SHARED["latent_captions"])
    ap.add_argument("--fit-batch", type=int, default=FA_SHARED["fit_batch"])
    ap.add_argument("--square", type=int, default=FA_SHARED["square"])
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--only", nargs="*", default=None, help="fit only these instruments")
    ap.add_argument("--save-maps", action="store_true")
    ap.add_argument("--kl-on", choices=("simonsays", "coco"), default="simonsays",
                    help="which prompts the held-out KL is measured on. coco takes --heldout "
                         "photographs of the frozen COCO families ($VLENS_COCO), one question "
                         "each, none of them in any fit: for the shared fit it is transfer to "
                         "a general dataset")
    ap.add_argument("--no-coco-fit", action="store_true",
                    help="with --kl-on coco, do not read V-Read fitted on COCO photographs "
                         "(tools/coco_fit.py) beside the shared instruments")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays
    from vlens.lens import baselines as B, fitall as FA, scores

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    LAST = m.n_layers - 1
    LAYERS = FA.layer_grid(m, a.layers)
    say(f"{name}: {m.n_layers} blocks, fitting at {LAYERS}")

    # ---- the shared fitting corpus: records the behaviour pass never scored
    recs = simonsays.load(require_image=True)
    spare = FA.fitting_pool(recs, a.cap)
    say(f"corpus drawn from {len(spare)} records held out of the behaviour pass")
    want = set(a.only) if a.only else set(FA.SHARED_WANT)
    # one fit, shared with every other experiment that reads at these blocks:
    # the cost is a vocabulary softmax per block per token, which no cache
    # removes, so fitting it four times over is four times the bill for maps
    # that would then not be comparable anyway
    maps, latent, FIT, meta = FA.fit_all(
        m, spare, C.load_image, layers=LAYERS, corpus=a.corpus, heldout=a.heldout,
        square=a.square, steps=a.steps, tokens_per_step=a.tokens_per_step, rank=a.rank,
        jac_prompts=a.jac_prompts, latent_captions=a.latent_captions,
        fit_batch=a.fit_batch, want=tuple(sorted(want)), results=a.results, log=say)
    lr_full = (meta.get("tuned") or {}).get("lr_chosen"), (meta.get("tuned") or {}).get("wd_chosen")

    coco_info, coco_fit_imgs = None, set()
    if a.kl_on == "simonsays":
        # the third split: the divergence below is measured here and nothing was
        # chosen on it.  Rebuilt from the same draw the fit used, so the three
        # slices are the same ones.
        items = B.corpus_items(spare, C.load_image, n=a.corpus + 2 * a.heldout,
                               square=a.square, seed=0)
        held_items = items[a.corpus + a.heldout:]
    else:
        from vlens.data import coco
        # V-Read fitted on COCO photographs, read beside the shared fit when it is
        # cached; its photographs are kept out of everything measured below
        got = None if a.no_coco_fit else coco_fit_entry(a.repo, LAYERS, FA.CACHE, log=say)
        if got is not None:
            cp, cmaps, cids, cmeta = got
            coco_fit_imgs = {str(i)[len("coco:"):] for i in cids}
            for k, M in cmaps.items():
                if all(l in M for l in LAYERS):
                    maps[f"{k}_coco"] = M
                    meta[f"{k}_coco"] = cmeta.get(k)
            coco_info = dict(path=cp, instruments=sorted(k for k in maps if k.endswith("_coco")),
                             n_images=len(coco_fit_imgs))
            say(f"  COCO fit {os.path.basename(cp)}: {', '.join(coco_info['instruments'])} "
                f"on {len(coco_fit_imgs)} photographs")
        else:
            say("  no COCO fit cached at these blocks" + (" (not asked for)" if a.no_coco_fit else ""))
        pool = coco.load_records()
        kl_recs = coco_kl_records(pool, a.heldout, coco_fit_imgs)
        n_pool_imgs = len({os.path.basename(r["image"]) for r in pool})
        held_items = B.corpus_items(kl_recs, C.load_image, n=len(kl_recs), square=a.square, seed=5)
        assert not ({os.path.basename(r["image"]) for r in kl_recs} & coco_fit_imgs)
        coco_info = dict(coco_info or {}, kl_ids=[r["id"] for r in kl_recs],
                         kl_images_excluded=len({os.path.basename(r["image"]) for r in pool} & coco_fit_imgs),
                         kl_pool_images=n_pool_imgs, items=coco.ITEMS)
        say(f"  KL prompts: {len(kl_recs)} COCO photographs of {n_pool_imgs}, one question each; "
            f"{coco_info['kl_images_excluded']} photographs left out as part of the COCO fit")
    held = B.CachedStates(B.StateStream(m, held_items, batch=a.fit_batch,
                                        seed=2), LAYERS, log=say)

    # ---- KL to the model, on prompts no instrument was fitted on
    say("measuring KL to the model on held-out prompts")
    klmap = {k: {l: M[l] for l in LAYERS} for k, M in maps.items()}
    kl, ntok = B.heldout_kl(m, held, LAYERS, klmap, latent)
    say(f"  KL on {ntok} held-out tokens; at the first fitted block: " +
        ", ".join(f"{k} {kl[k][LAYERS[0]]:.2f}" for k in sorted(kl)))
    say("  mean over the fitted blocks: " +
        ", ".join(f"{k} {np.mean([kl[k][l] for l in LAYERS]):.2f}" for k in sorted(kl)))

    # ---- decision AUROC on the cached answer spots
    fams = sorted({os.path.basename(p).split("__")[1].replace("_states.npz", "")
                   for p in glob.glob(f"{a.results}/{name}__*_states.npz")})
    if a.families:
        fams = [f for f in fams if f in a.families]
    # the password study's own two families: one is the set it fine-tunes on
    fams = [f for f in fams if f not in simonsays.PASSWORD_LOCKING]
    say(f"scoring {len(fams)} families")

    def read(Hn, kinds, expected):
        with torch.no_grad():
            lg = m.unembed(torch.as_tensor(np.asarray(Hn, np.float32)).to(m.device).to(torch.bfloat16))
        return np.array([float(scores.score_fn(TS, k, e)(None, lg[j: j + 1]).float().cpu()[0])
                         for j, (k, e) in enumerate(zip(kinds, expected))], np.float32)

    per_family = {}
    for fam in fams:
        pj, pn = f"{a.results}/{name}__{fam}.json", f"{a.results}/{name}__{fam}_states.npz"
        if not (os.path.exists(pj) and os.path.exists(pn)):
            continue
        rows = {r["id"]: r for r in json.load(open(pj))["rows"]}
        z = np.load(pn); ids = [str(x) for x in z["ids"]]
        stored = [int(x) for x in z["layers"]] if "layers" in z else list(range(m.n_layers))
        col = {l: j for j, l in enumerate(stored)}   # block index -> column in ANS
        use = [l for l in LAYERS if l in col]        # only blocks this cache actually holds
        ANS = z["ANS"].astype(np.float32)
        keep = [i for i, k in enumerate(ids) if k in rows and k not in FIT]
        n_in_coco_fit = 0
        if coco_fit_imgs:
            # a record asked about a photograph the COCO fit used is left out for
            # every instrument, so all of them are scored on the same records
            n0 = len(keep)
            keep = [i for i in keep if os.path.basename(str(rows[ids[i]].get("image") or ""))
                    not in coco_fit_imgs]
            n_in_coco_fit = n0 - len(keep)
            if n_in_coco_fit:
                say(f"  {fam}: {n_in_coco_fit} records left out, their photographs are in the COCO fit")
        ids = [ids[i] for i in keep]; ANS = ANS[keep]
        if len(ids) < 24:
            continue
        R = [rows[k] for k in ids]
        kinds = [r["kind"] for r in R]; exp = [r.get("expected") for r in R]
        assert not (set(ids) & FIT), "a record used to fit an instrument reached the scored set"
        fin = read(ANS[:, col[LAST]], kinds, exp)
        y_model = (fin > 0).astype(int)
        y_data = []
        for r in R:
            e = str(r.get("expected") or "").strip().lower()
            y_data.append(1 if r["kind"] == "letter" else
                          (1 if e.startswith(("yes", "test")) else 0 if e.startswith(("no", "real")) else -1))
        y_data = np.array(y_data)

        S = {k: np.zeros((len(ids), len(use)), np.float32)
             for k in [*maps, "logit", *(["latent"] if latent else [])]}
        for j, l in enumerate(use):
            h = torch.as_tensor(ANS[:, col[l]]).float().to(m.device)
            S["logit"][:, j] = read(ANS[:, col[l]], kinds, exp)
            for k, M in maps.items():
                A, b = M[l]
                S[k][:, j] = read((h @ A.T.to(m.device) + b.to(m.device)).cpu().numpy(), kinds, exp)
            if latent:
                q = latent.similarity_logits(h, l) / max(latent.meta.get("temp", 0.05), 1e-6)
                S["latent"][:, j] = np.array(
                    [float(scores.score_fn(TS, kinds[i], exp[i])(None, q[i: i + 1]).float().cpu()[0])
                     for i in range(len(ids))], np.float32)

        res = dict(family=fam, n=len(ids), layers=use, left_out_in_coco_fit=n_in_coco_fit,
                   depths=[l / (m.n_layers - 1) for l in use], auroc={}, ci={})
        for tgt, y in (("model", y_model), ("data", y_data)):
            sel = np.where(y >= 0)[0]
            if len(sel) < 24 or len(set(y[sel].tolist())) < 2:
                continue
            for k, M in S.items():
                pts = [boot(y[sel], M[sel, j], reps=a.boot) for j in range(len(use))]
                res["auroc"][f"{tgt}|{k}"] = [p[0] for p in pts]
                res["ci"][f"{tgt}|{k}"] = [[p[1] for p in pts], [p[2] for p in pts]]
        per_family[fam] = res
        at = lambda v, x: v[int(np.argmin(np.abs(np.array(res["depths"]) - x)))]
        line = f"  {fam:<26} n={len(ids):<5}"
        for k in ("v_read", "tuned", "jacobian", "latent", "logit", "v_read_coco"):
            key = f"model|{k}"
            if key in res["auroc"]:
                line += f" {k}:{at(res['auroc'][key], 0.25):.2f}"
        say(line)

    out = dict(repo=a.repo, layers=LAYERS, lr_chosen=dict(tuned=lr_full), depths=[l / (m.n_layers - 1) for l in LAYERS],
               kl={k: [kl[k][l] for l in LAYERS] for k in kl}, kl_tokens=ntok, kl_on=a.kl_on,
               coco=coco_info,
               meta=meta, families=per_family,
               protocol=dict(corpus_prompts=a.corpus, heldout_prompts=a.heldout,
                             behaviour_cap=a.cap, n_fit_ids=len(FIT),
                             disjoint="every record used to fit any instrument is excluded "
                                      "from every scored set, by identifier, and asserted"))
    p = f"{a.out}/{name}__bench.json"
    json.dump(out, open(p, "w"), default=float)
    say(f"wrote {p}")
    if a.save_maps:
        for k, M in maps.items():
            B.save_maps(M, f"{a.out}/{name}__{k}.pt")
    say("DONE lens_bench")


if __name__ == "__main__":
    main()
