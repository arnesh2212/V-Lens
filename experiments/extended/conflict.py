from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

OUT = "results/extended/h5"
FAMS = ("h5_none_conflict", "h5_conf_conflict", "h5_cons_conflict")


def build(a):
    from vlens.data import simonsays as SS
    R = [r for r in SS.load(families=["nuscenes_orig_read"]) if r["subcategory"] == "read_named"]
    ok = re.compile(r"^[A-Za-z0-9 &'.,\-/]{3,24}$")
    by_frame = {}
    for r in R:
        m = r.get("meta") or {}
        x, y = m.get("target_string"), m.get("decoy_string")
        if not (x and y and ok.match(x) and ok.match(y)) or x.lower() == y.lower():
            continue
        if f"'{x}'" not in (r.get("cue") or ""):
            continue
        by_frame.setdefault(m.get("sample_id") or r["image"], (r, x, y))
    frames = sorted(by_frame)[: a.n]
    cues = {
        "none": {"x": "Look carefully at this image.", "y": "Look carefully at this image."},
        "conflict": {"x": "Note: the text '{X}' does not appear anywhere in this image.",
                     "y": "Note: the text '{Y}' is written in this image."},
        "consistent": {"x": "Note: the text '{X}' is written in this image.",
                       "y": "Note: the text '{Y}' does not appear anywhere in this image."},
    }
    claim = {"none": {"x": None, "y": None}, "conflict": {"x": "no", "y": "yes"},
             "consistent": {"x": "yes", "y": "no"}}
    os.makedirs(f"{a.out}/prompts", exist_ok=True)
    files = []
    for cond in ("none", "conflict", "consistent"):
        samples = []
        for f in frames:
            r, X, Y = by_frame[f]
            for which, s, truth in (("x", X, "yes"), ("y", Y, "no")):
                q = f"Does the text '{s}' appear in this image? Answer yes or no."
                cue = cues[cond][which].format(X=X, Y=Y)
                samples.append({
                    "id": f"h5_{cond}_{which}_{r['id']}",
                    "group_id": f"h5:{f}:{which}",
                    "source": {"dataset": "h5", "variant": cond, "native": {"source_id": r["id"]}},
                    "probe": {"category": "conflict", "subcategory": f"asked_{'present' if which == 'x' else 'absent'}",
                              "condition": cond,
                              "tags": {"frame": f, "asked": "present" if which == "x" else "absent",
                                       "string": s, "x": X, "y": Y, "truth": truth, "claim": claim[cond][which],
                                       "source_id": r["id"]}},
                    "image": {"path": r["image"], "available": True},
                    "input": {"system_prompt": None, "prompt": "", "cue": cue, "question": q, "choices": None},
                    "target": {"type": "open_ended", "expected": truth, "answer_index": None},
                })
        p = os.path.abspath(f"{a.out}/prompts/h5_{cond}.json")
        json.dump({"schema_version": "vlens-1.0", "dataset_meta": {"name": "h5", "variant": cond,
                                                                     "category": "conflict"},
                   "samples": samples}, open(p, "w"), indent=1)
        files.append(["h5", cond, p])
    json.dump(files, open(f"{a.out}/extra_files.json", "w"), indent=1)
    print(f"{len(frames)} frames -> {2 * len(frames)} questions x 3 cues; wrote {a.out}/extra_files.json")


def _rows(results, model, fam):
    p = f"{results}/{model}__{fam}.json"
    return json.load(open(p))["rows"] if os.path.exists(p) else []


def probe(a):
    from vlens.lens.metrics import auroc
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    res = f"{a.out}/behaviour"
    X, meta = [], []
    for fam in FAMS:
        rows = {r["id"]: r for r in _rows(res, a.model, fam)}
        p = f"{res}/{a.model}__{fam}_states.npz"
        if not rows or not os.path.exists(p):
            continue
        z = np.load(p)
        layers = [int(v) for v in z["layers"]] if "layers" in z.files else list(range(z["ANS"].shape[1]))
        for j, i in enumerate(z["ids"]):
            r = rows.get(str(i))
            if r is None:
                continue
            m = r.get("meta") or {}
            said = _said(r)
            if said is None:
                continue
            X.append(z["ANS"][j].astype(np.float32))
            meta.append(dict(id=r["id"], cond=r.get("condition") or m.get("condition"), frame=m.get("frame"),
                             truth=int(m.get("truth") == "yes"),
                             claim=None if m.get("claim") is None else int(m.get("claim") == "yes"),
                             said=int(said == "yes")))
    if not X:
        raise SystemExit("no states for " + a.model)
    X = np.stack(X); n, L, d = X.shape
    cond = np.array([m["cond"] for m in meta]); fr = np.array([str(m["frame"]) for m in meta])
    truth = np.array([m["truth"] for m in meta]); said = np.array([m["said"] for m in meta])
    claim = np.array([-1 if m["claim"] is None else m["claim"] for m in meta])
    conf = cond == "conflict"
    out = dict(model=a.model, n=int(n), layers=layers, behaviour={}, probe={})
    for c in ("none", "conflict", "consistent"):
        s = cond == c
        out["behaviour"][c] = dict(n=int(s.sum()), follows_picture=float((said[s] == truth[s]).mean()),
                                   follows_sentence=(float((said[s] == claim[s]).mean()) if c != "none" else None))
    gkf = GroupKFold(n_splits=5)

    def cv_scores(y, mask_train, j):
        sc = np.full(n, np.nan)
        for tr, te in gkf.split(np.zeros(n), y, fr):
            tr = tr[mask_train[tr]]
            if len(set(y[tr].tolist())) < 2:
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            clf.fit(X[tr, j], y[tr]); sc[te] = clf.decision_function(X[te, j])
        return sc

    allm = np.ones(n, bool); claimed = claim >= 0
    for name, y, mtr, mte in (("visual_fact", truth, allm, conf), ("visual_fact_nocue", truth, allm, cond == "none"),
                              ("text_claim", np.where(claimed, claim, 0), claimed, conf),
                              ("answer", said, allm, conf)):
        per = []
        for j in range(L):
            sc = cv_scores(y, mtr, j)
            sel = mte & np.isfinite(sc)
            per.append(float(auroc(y[sel], sc[sel])) if len(set(y[sel].tolist())) == 2 else float("nan"))
        out["probe"][name] = per
    out["conflict_answer_vs_truth_auroc"] = float(auroc(truth[conf], said[conf])) if len(set(truth[conf])) == 2 else None
    rd = f"{a.out}/readouts/{a.model}__h5_conf_conflict_readouts.json"
    if os.path.exists(rd):
        R = {r["id"]: r for r in json.load(open(rd))["rows"]}
        blocks = json.load(open(rd)).get("blocks")
        for k in ("v_read", "logit"):
            per = []
            ids = [m["id"] for m, c in zip(meta, conf) if c and m["id"] in R and R[m["id"]].get(k)]
            if not ids:
                continue
            tt = np.array([m["truth"] for m in meta if m["id"] in set(ids)])
            for b in range(len(blocks)):
                v = np.array([R[i][k][b] if R[i][k][b] is not None else np.nan for i in ids], float)
                ok = np.isfinite(v)
                per.append(float(auroc(tt[ok], v[ok])) if ok.sum() > 5 and len(set(tt[ok])) == 2 else float("nan"))
            out[f"{k}_truth_auroc_by_block"] = dict(blocks=blocks, auroc=per)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/{a.model}__e5.json", "w"), indent=1, default=float)
    b = out["behaviour"]
    print(f"{a.model}: n={n}; follows picture: none {b['none']['follows_picture']:.2f}, "
          f"conflict {b['conflict']['follows_picture']:.2f} (follows sentence {b['conflict']['follows_sentence']:.2f}), "
          f"consistent {b['consistent']['follows_picture']:.2f}")
    for k, v in out["probe"].items():
        print(f"  probe {k:<18} " + " ".join(f"{x:.2f}" for x in v))
    for k in ("v_read", "logit"):
        if f"{k}_truth_auroc_by_block" in out:
            print(f"  {k} reads the visual fact on conflict records, by block: " +
                  " ".join(f"{x:.2f}" for x in out[f"{k}_truth_auroc_by_block"]["auroc"]))


def _said(r):
    rep = str(r.get("reply") or r.get("answer") or "").strip().lower()
    if rep.startswith("yes"):
        return "yes"
    if rep.startswith("no"):
        return "no"
    mg = r.get("margin")
    if mg is None:
        return None
    return "yes" if float(mg) > 0 else "no"


def patch(a):
    import torch
    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays as SS
    from vlens.lens import attribution as AT, readout, scores
    from vlens.lens.hooks import ActivationRecorder
    from vlens.figures import fx_standing as FS

    m = vlens.load_vlm(a.repo)
    name = a.repo.split("/")[-1]
    TS = scores.TokenSets(m.tokenizer)
    LAST = m.n_layers - 1
    depths = {f: max(int(round(f * LAST)), 1) for f in (0.25, 0.5, 0.75)}
    Lh = depths[0.5]
    S = a.square
    say = lambda *x: print(*x, flush=True)

    def prep(rec, img=None):
        im = img if img is not None else C.load_image(rec["image"], square=S)
        b = m.chat(rec["question"], [im], system=rec.get("system"), cue=rec.get("cue"))
        return b, im

    def full_states(b, layers):
        with torch.no_grad(), ActivationRecorder(m.layers, at=sorted(set(layers)) + [LAST]) as rec:
            m.forward(b)
            H = {l: rec.activations[l][0].float().cpu() for l in layers}
            lg = m.unembed(rec.activations[LAST][0, -1][None].to(m.device))
        return H, lg

    fn = scores.score_fn(TS, "yesno", None)

    def margin(lg):
        return float(fn(None, lg).float().cpu()[0])

    def transplant(b_to, b_from, H_from, pos_to, pos_from, layer):
        X = H_from[layer][pos_from][None]
        with torch.no_grad():
            return float(AT.hooked_axis(m, b_to, fn, layer, pos_to, X.to(m.device)).float().cpu()[0])

    rows = []
    out_path = f"{a.out}/patch/{name}__patch.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["pair"])
            except Exception:
                pass
    fh = open(out_path, "a")

    def one(kind, pid, rec_to, rec_from, img_to=None, img_from=None, region=None):
        if pid in done:
            return
        b_to, im_to = prep(rec_to, img_to); b_fr, im_fr = prep(rec_from, img_from)
        p_to, p_fr = readout.image_positions(m, b_to), readout.image_positions(m, b_fr)
        if len(p_to) != len(p_fr) or len(p_to) < 16:
            return
        Hto, lg_to = full_states(b_to, list(depths.values()))
        Hfr, lg_fr = full_states(b_fr, list(depths.values()))
        m_to, m_fr = margin(lg_to), margin(lg_fr)
        G = int(round(len(p_to) ** 0.5))
        row = dict(pair=pid, kind=kind, m_to=m_to, m_from=m_fr, flips=bool((m_to > 0) != (m_fr > 0)), patch={})
        sets = {"image": (p_to.cpu(), p_fr.cpu()),
                "answer_spot": (torch.tensor([b_to["input_ids"].shape[1] - 1]),
                                torch.tensor([b_fr["input_ids"].shape[1] - 1]))}
        if region is not None:
            cov = AT.pool_mask(region, G) > 0.1
            if cov.any():
                idx = np.nonzero(cov)[0]
                sets["region"] = (p_to.cpu()[idx], p_fr.cpu()[idx])
        for f, Ld in depths.items():
            for sname, (pt, pf) in sets.items():
                v = transplant(b_to, b_fr, Hfr, pt, pf, Ld)
                den = m_fr - m_to
                row["patch"][f"{f}|{sname}"] = dict(margin=v, recovered=(v - m_to) / den if abs(den) > 1e-6 else None,
                                                    flipped=bool((v > 0) != (m_to > 0)))
        if kind == "conflict":
            hf = Hfr[Lh][p_fr.cpu()]; ht = Hto[Lh][p_to.cpu()]
            with AT.SkipTo(m, b_fr, Lh, fn=fn, pos=p_fr, h=hf, base=ht, log=None) as bh:
                w = AT.path_reading(m, bh, fn, Lh, p_fr, hf, ht, steps=8)
                with torch.no_grad():
                    f_rest = float(AT.hooked_axis(m, bh, fn, Lh, p_fr, ht[None].to(m.device)).float().cpu()[0])
            row["vwrite"] = dict(sum=float(np.sum(w)), restore_effect=m_fr - f_rest,
                                 share_of_shift=(m_fr - f_rest) / (m_fr - m_to) if abs(m_fr - m_to) > 1e-6 else None)
        fh.write(json.dumps(row, default=float) + "\n"); fh.flush()
        rows.append(row)

    if "conflict" in a.parts:
        recs = {r["id"]: r for r in SS.load(families=list(FAMS[:2]))}
        by = {}
        for r in recs.values():
            mt = r.get("meta") or {}
            by.setdefault((mt.get("frame"), mt.get("asked")), {})[r.get("condition") or mt.get("condition")] = r
        pairs = [(k, v) for k, v in by.items() if "none" in v and "conflict" in v][: a.n]
        say(f"{name}: {len(pairs)} conflict / no-cue pairs")
        for i, (k, v) in enumerate(pairs):
            one("conflict", f"conflict:{k[0]}:{k[1]}", v["none"], v["conflict"])
            if i % 25 == 0:
                say(f"  {i}/{len(pairs)}")
    if "authority" in a.parts:
        R = SS.load(families=["openimages_edit_auth", "vins_edit_auth", "nuscenes_edit_auth",
                              "openimages_orig_auth", "vins_orig_auth"])
        R = [r for r in R if r["kind"] == "yesno"]
        orig = {}
        for r in R:
            if r["variant"] == "original":
                orig[(os.path.basename(r["image"]), r["question"])] = r
        carr = {}
        edits = []
        for r in R:
            if r["variant"] != "edited":
                continue
            mt = r.get("meta") or {}
            key = (os.path.basename(str(mt.get("source_image"))), r["question"])
            ev = mt.get("edit_variant")
            if ev in FS.OFFICIAL:
                carr.setdefault(key, {})["official"] = r
            elif ev in FS.INFORMAL:
                carr.setdefault(key, {})["informal"] = r
            if key in orig:
                edits.append((key, orig[key], r))
        cp = [(k, v) for k, v in carr.items() if "official" in v and "informal" in v][: a.n]
        say(f"{name}: {len(cp)} carrier pairs, {len(edits)} original/edit pairs")
        for k, v in cp:
            img_o = C.load_image(v["official"]["image"], square=S); img_i = C.load_image(v["informal"]["image"], square=S)
            reg = AT.edit_mask(v["informal"]["image"], v["official"]["image"], side=S)
            one("carrier", f"carrier:{k[0]}:{k[1][:40]}", v["informal"], v["official"], img_i, img_o, region=reg)
        for k, o, e in edits[: a.n]:
            reg = AT.edit_mask(o["image"], e["image"], side=S)
            one("edit", f"edit:{e['id']}", o, e, region=reg)
    fh.close()
    allrows = [json.loads(l) for l in open(out_path)]
    summ = {}
    for kind in ("conflict", "carrier", "edit"):
        K = [r for r in allrows if r["kind"] == kind]
        if not K:
            continue
        d = dict(n=len(K), answer_differs=float(np.mean([r["flips"] for r in K])),
                 mean_abs_shift=float(np.mean([abs(r["m_from"] - r["m_to"]) for r in K])))
        for key in K[0]["patch"]:
            vals = [r["patch"][key] for r in K if key in r["patch"]]
            d[key] = dict(flip=float(np.mean([v["flipped"] for v in vals])),
                          shift=float(np.mean([v["margin"] - r["m_to"] for v, r in zip(vals, K)])),
                          recovered_median=float(np.nanmedian([np.nan if v["recovered"] is None else v["recovered"]
                                                               for v in vals])))
        if kind == "conflict":
            V = [r["vwrite"] for r in K if r.get("vwrite") and r["vwrite"].get("share_of_shift") is not None]
            d["vwrite_image_share_median"] = float(np.median([v["share_of_shift"] for v in V])) if V else None
        summ[kind] = d
    json.dump(summ, open(f"{a.out}/patch/{name}__patch_summary.json", "w"), indent=1)
    for kind, d in summ.items():
        say(f"[{kind}] n={d['n']} answers differ {d['answer_differs']:.2f}, mean |shift| {d['mean_abs_shift']:.2f}")
        for key in sorted(k for k in d if "|" in k):
            say(f"   {key:<22} flip {d[key]['flip']:.2f}  shift {d[key]['shift']:+.2f}  recovered {d[key]['recovered_median']:.2f}")
        if "vwrite_image_share_median" in d:
            say(f"   V-Write: share of the sentence's shift carried by image tokens at half depth "
                f"(median) {d['vwrite_image_share_median']}")
    say("DONE conflict patch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "probe", "patch"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--parts", nargs="*", default=["conflict", "authority"])
    ap.add_argument("--square", type=int, default=448)
    a = ap.parse_args()
    {"build": build, "probe": probe, "patch": patch}[a.cmd](a)


if __name__ == "__main__":
    main()
