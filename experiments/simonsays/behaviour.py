from __future__ import annotations

import argparse
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/simonsays")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--dataset", choices=("simonsays", "coco"), default="simonsays",
                    help="coco reads the frozen general-dataset families from $VLENS_COCO/coco_items.json")
    ap.add_argument("--cap", type=int, default=80, help="records per (family, subcategory, framing)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gen-short", type=int, default=6)
    ap.add_argument("--gen-open", type=int, default=64)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--no-states", action="store_true")
    ap.add_argument("--state-layers", type=int, default=0,
                    help="store states at this many evenly spaced blocks instead of every one; "
                         "0 keeps every block. The last block is always kept, and the indices "
                         "are written into the npz so every reader knows what it has.")
    ap.add_argument("--state-grid", type=int, default=0,
                    help="store states only at the blocks the shared lens fit uses "
                         "(fitall.layer_grid with this many blocks) and the last; overrides "
                         "--state-layers. 16 matches the shared fit, at under half the disk")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="skip families whose output already exists")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus
    from vlens.data import simonsays
    from vlens.lens import scores
    from vlens.lens.hooks import ActivationRecorder

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    if a.dataset == "coco":
        from vlens.data import coco
        recs = coco.load_records(families=a.families)
        img_root = os.path.dirname(os.path.abspath(coco.ITEMS))
        say(f"COCO records from {coco.ITEMS}")
    else:
        recs = simonsays.load(families=a.families)
        img_root = simonsays.ROOT
    seen: dict = {}
    kept = []
    for r in recs:
        key = (r["family"], r["subcategory"], r["framing"])
        n = seen.get(key, 0)
        if a.cap and n >= a.cap:
            continue
        seen[key] = n + 1
        kept.append(r)
    if a.limit:
        kept = kept[: a.limit]
    by_family: dict[str, list] = {}
    for r in kept:
        by_family.setdefault(r["family"], []).append(r)
    say(f"{len(kept)} records in {len(by_family)} families (cap {a.cap} per cell)")

    m = vlens.load_vlm(a.repo, max_pixels=a.max_pixels)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    LAST = m.n_layers - 1
    if a.state_grid:
        from vlens.lens.fitall import layer_grid
        keep_layers = sorted({*layer_grid(m, a.state_grid), LAST})
    elif a.state_layers and a.state_layers < m.n_layers:
        keep_layers = sorted({*[int(round(x)) for x in
                                np.linspace(max(m.n_deepstack, 0), LAST, a.state_layers)], LAST})
    else:
        keep_layers = list(range(m.n_layers))
    say(f"{name}: {m.n_layers} layers, d_model {m.d_model}, deepstack {m.n_deepstack}; "
        f"storing states at {len(keep_layers)} blocks")

    def asks_short(r):
        return r["kind"] in ("letter", "yesno", "test_real")

    for fam, rows in by_family.items():
        out_json = f"{a.out}/{name}__{fam}.json"
        out_npz = f"{a.out}/{name}__{fam}_states.npz"
        if a.resume and os.path.exists(out_json):
            say(f"  {fam}: already done, skipped"); continue
        t0 = time.time()
        done, ANS, ids = [], [], []
        order = sorted(range(len(rows)), key=lambda i: (asks_short(rows[i]), len(rows[i]["question"] or "")))
        for s0 in range(0, len(order), a.batch):
            idx = order[s0: s0 + a.batch]
            chunk = [rows[i] for i in idx]
            items = []
            for r in chunk:
                img = corpus.load_image(r["image"], square=a.square)
                items.append(dict(question=r["question"], image=img, system=r["system"], cue=r["cue"]))
            keep_states = (not a.no_states) and all(r["kind"] in ("yesno", "letter", "test_real") for r in chunk)
            b = m.batch_many(items, side="right")
            spot = m.answer_spot(b)
            at = list(keep_layers) if keep_states else [LAST]
            with torch.no_grad(), ActivationRecorder(m.layers, at=at) as rec:
                m.forward(b)
                H = {l: rec.activations[l][torch.arange(len(chunk)), spot.to(rec.activations[l].device)]
                     for l in at}
            lg = m.unembed(H[LAST].to(m.device))
            for j, r in enumerate(chunk):
                r["margins"] = scores.margins(TS, tok, lg[j], r["kind"])
            if keep_states:
                S = np.stack([torch.stack([H[l][j] for l in keep_layers]).float().cpu().numpy()
                              for j in range(len(chunk))]).astype(np.float16)
                ANS.append(S); ids += [r["id"] for r in chunk]
            del H, rec
            n_new = a.gen_short if all(asks_short(r) for r in chunk) else a.gen_open
            bg = m.batch_many(items, side="left")
            with torch.no_grad():
                gen = m.hf.generate(**{k: v for k, v in bg.items() if not k.startswith("_")},
                                    max_new_tokens=n_new, do_sample=False,
                                    pad_token_id=tok.pad_token_id or tok.eos_token_id)
            n_in = bg["input_ids"].shape[1]
            for j, r in enumerate(chunk):
                r["reply"] = tok.decode(gen[j, n_in:], skip_special_tokens=True).strip()
                row = {k: v for k, v in r.items() if k != "image"}
                if a.dataset == "coco" and r.get("meta"):
                    row["meta"] = {k: v for k, v in r["meta"].items() if k != "polys"}
                done.append(simonsays.safe_row(row | {"image": os.path.relpath(r["image"], img_root)}))
            if (s0 // a.batch) % 10 == 0 or s0 + a.batch >= len(order):
                say(f"  {fam}: {min(s0 + a.batch, len(order))}/{len(order)} "
                    f"({(time.time() - t0) / 60:.1f} min) last: {done[-1]['reply'][:50]!r}")
        json.dump(dict(repo=a.repo, family=fam, cap=a.cap, n=len(done), rows=done),
                  open(out_json, "w"), indent=1, default=float)
        if ANS:
            np.savez_compressed(out_npz, ids=np.array(ids), ANS=np.concatenate(ANS),
                                layers=np.array(keep_layers), n_layers=m.n_layers)
            say(f"  {fam}: states {np.concatenate(ANS).shape} -> {out_npz}")
        say(f"  {fam}: wrote {len(done)} rows ({(time.time() - t0) / 60:.1f} min)")
        if a.dataset == "coco":
            _pope_line(fam, done, say)
    say("DONE behaviour")


def _pope_line(fam, rows, say):
    import re
    if fam == "coco_pope":
        said = []
        for r in rows:
            w = re.match(r"\W*(\w+)", str(r.get("reply") or ""))
            said.append((w.group(1).lower() if w else "") == "yes")
        truth = [str(r.get("expected")).lower() == "yes" for r in rows]
        tp = sum(s_ and t for s_, t in zip(said, truth)); fp = sum(s_ and not t for s_, t in zip(said, truth))
        fn = sum((not s_) and t for s_, t in zip(said, truth))
        acc = sum(s_ == t for s_, t in zip(said, truth)) / max(len(rows), 1)
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        say(f"  {fam}: POPE accuracy {acc:.3f}, precision {prec:.3f}, recall {rec:.3f}, "
            f"F1 {2 * prec * rec / max(prec + rec, 1e-9):.3f}, yes ratio {sum(said) / max(len(rows), 1):.3f}")
    elif fam == "coco_mcq":
        ok = [(re.match(r"\W*\(?([ABCD])\b", str(r.get("reply") or "")) or [None, None])[1] == r.get("expected")
              for r in rows]
        say(f"  {fam}: accuracy {sum(ok) / max(len(ok), 1):.3f} over {len(ok)}")


if __name__ == "__main__":
    main()
