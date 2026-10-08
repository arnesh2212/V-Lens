# SPDX-License-Identifier: Apache-2.0
"""What each block would say: V-Read's top words at every depth, at the answer and at every image patch.

The Jacobian lens paper (Anthropic 2026, "Verbalizable Representations Form a
Global Workspace in Language Models") reads, for a prompt, the lens's top word
at every (position, layer) and shows intermediate concepts surfacing mid-depth
before the model says anything: the country on the way to its currency, "nose"
at the caret of an ASCII face. This does the same for the VLM with V-Read (the
shared label-free ridge map from each block to the last block, then the model's
own final norm and unembedding), beside the logit lens and, where the same map
cache holds them, the tuned lens and the Jacobian lens, all at the same states.

Two readings per prompt, both forward only:

* the answer spot: the top ``--k-ans`` tokens with log-probabilities at every
  fitted block, for every lens, plus the full-vocabulary rank of a fixed list of
  concept words and of the model's own first reply token;
* the image-token positions, once per distinct picture (under its first
  record's cue and system), since a causal model's image tokens never see the question: the top ``--k-img``
  tokens per patch per block per lens. For COCO the rank and log-probability
  of the pictured object's name (and a co-occurring absent object's name, as a
  control) are kept for every patch, with the object's mask coverage per cell.

Nothing but token ids, log-probabilities and ranks is written; no states. The
WMDP carrier records keep the placeholder instead of their question, and the
password-locking families are never loaded.

    VLENS_COCO=/workspace/temp/vlm/data_root/coco HF_HOME=/workspace/temp/vlm/hf \
        python3 experiments/trace/trace.py --repo Qwen/Qwen3-VL-8B-Instruct --out results/trace
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

LENSES = ("logit", "v_read", "tuned", "jacobian")
LOCKED = re.compile(r"wmdp[_ ]pw[_ ]")

#: words whose rank is tracked at the answer spot at every block
CONCEPTS = ["stop", "police", "sign", "officer", "law", "warning", "danger", "closed", "open",
            "test", "fake", "real", "exam", "evaluation", "benchmark", "simulation", "edited",
            "photoshop", "text", "written", "says", "read", "instruction", "ignore", "follow",
            "yes", "no", "not", "cannot", "sorry", "unsafe", "safe", "speed", "limit", "exit",
            "parking", "entry", "wrong", "caution", "car", "road", "street", "dog", "cat",
            "person", "man", "woman", "people", "A", "B", "C", "D", "wrong", "lie", "hazard"]

#: COCO category -> extra names a patch may read out as, beside the name and its plural
SYNONYMS = {
    "person": ["man", "woman", "people", "boy", "girl", "player", "child", "men", "women", "kid", "lady", "guy"],
    "bicycle": ["bike", "cyclist"], "car": ["vehicle", "cars", "taxi"], "motorcycle": ["motorbike", "bike", "motor"],
    "airplane": ["plane", "aircraft", "jet", "airliner"], "bus": ["buses"], "train": ["locomotive", "railway"],
    "truck": ["lorry", "vehicle"], "boat": ["ship", "sail", "canoe", "kayak"], "traffic light": ["traffic", "light", "signal"],
    "fire hydrant": ["hydrant"], "stop sign": ["stop", "sign"], "parking meter": ["meter", "parking"],
    "bench": ["seat"], "bird": ["birds", "seagull", "gull", "pigeon", "duck"], "cat": ["kitten", "kitty"],
    "dog": ["puppy", "dogs"], "horse": ["horses", "pony"], "sheep": ["lamb"], "cow": ["cattle", "cows", "bull"],
    "elephant": ["elephants"], "bear": ["bears"], "zebra": ["zebras"], "giraffe": ["giraffes"],
    "backpack": ["bag", "rucksack"], "umbrella": ["umbrellas", "parasol"], "handbag": ["bag", "purse"],
    "tie": ["necktie"], "suitcase": ["luggage", "bag"], "frisbee": ["disc"], "skis": ["ski", "skier"],
    "snowboard": ["snowboarder", "board"], "sports ball": ["ball", "soccer", "football", "tennis", "baseball"],
    "kite": ["kites"], "baseball bat": ["bat"], "baseball glove": ["glove", "mitt"], "skateboard": ["skate", "skateboarder"],
    "surfboard": ["surf", "board", "surfer"], "tennis racket": ["racket", "racquet", "tennis"],
    "bottle": ["bottles"], "wine glass": ["glass", "wine"], "cup": ["mug", "coffee"], "fork": ["forks"],
    "knife": ["knives"], "spoon": ["spoons"], "bowl": ["bowls"], "banana": ["bananas"], "apple": ["apples"],
    "sandwich": ["sandwiches", "burger"], "orange": ["oranges"], "broccoli": ["vegetable", "vegetables"],
    "carrot": ["carrots"], "hot dog": ["hotdog", "sausage"], "pizza": ["pizzas"], "donut": ["doughnut", "donuts"],
    "cake": ["cakes", "dessert"], "chair": ["chairs", "seat"], "couch": ["sofa"], "potted plant": ["plant", "flower", "pot"],
    "bed": ["bedroom"], "dining table": ["table"], "toilet": ["bathroom"], "tv": ["television", "screen", "monitor"],
    "laptop": ["computer", "notebook"], "mouse": ["mice"], "remote": ["controller"], "keyboard": ["keys"],
    "cell phone": ["phone", "smartphone", "mobile"], "microwave": ["oven"], "oven": ["stove"], "toaster": ["toast"],
    "sink": ["faucet"], "refrigerator": ["fridge"], "book": ["books"], "clock": ["clocks", "time"],
    "vase": ["vases", "flowers"], "scissors": ["scissor"], "teddy bear": ["teddy", "bear"],
    "hair drier": ["dryer", "hair"], "toothbrush": ["tooth", "brush"],
}


def name_words(cat: str) -> list[str]:
    words = [cat] + cat.split() + SYNONYMS.get(cat, [])
    out = []
    for w in words:
        for v in (w, w + "s"):
            if v not in out:
                out.append(v)
    return out


def word_ids(tok, words) -> list[int]:
    """First token of each word, bare and after a space, lower and capitalised; a first
    token of fewer than three letters counts only when it is the whole word, so
    a long name does not match every token that starts like it."""
    ids = set()
    for w in words:
        # upper case only for a short word (TV); a long one splits into junk pieces
        for f in {w, w.capitalize()} | ({w.upper()} if len(w) <= 3 else set()):
            for g in (f, " " + f):
                t = tok.encode(g, add_special_tokens=False)
                if not t:
                    continue
                piece = tok.decode([t[0]]).strip()
                if len(t) == 1 or len(piece) >= 3:
                    ids.add(t[0])
    return sorted(ids)


def load_maps(model_name: str, where: str, dev, say):
    """V-Read, tuned and Jacobian from the shared (non-COCO, non-password) fit, float on the GPU."""
    from vlens.lens.fitall import _load
    from vlens.data import simonsays as SS
    pw = {r["id"] for r in SS.load(families=list(SS.PASSWORD_LOCKING), require_image=False)}
    got = []
    for s in glob.glob(os.path.join(where, "*.pt.spec")):
        try:
            spec = json.loads(open(s).read())
        except Exception:  # noqa: BLE001
            continue
        if str(spec[0]).split("/")[-1] == model_name and os.path.exists(s[:-5]) and len(spec[2]) >= 2:
            got.append((len(spec[2]), os.path.getsize(s[:-5]), s[:-5]))
    for _nl, _sz, p in sorted(got, reverse=True):
        maps, _lat, ids, _meta = _load(p)
        if any(str(i).startswith("coco:") for i in ids):
            say(f"  {os.path.basename(p)}: a COCO fit, not the shared one; skipped")
            continue
        if any(str(i) in pw for i in ids):
            say(f"  {os.path.basename(p)}: holds password-study records; skipped")
            continue
        out = {}
        for kind in LENSES[1:]:
            mk = {int(l): (v[0].to(dev, torch.float32), v[1].to(dev, torch.float32))
                  for l, v in maps.get(kind, {}).items() if str(l).lstrip("-").isdigit()}
            if len(mk) >= 2:
                out[kind] = mk
        if "v_read" in out:
            say(f"  maps from {os.path.basename(p)}: " + ", ".join(f"{k} at {len(v)} blocks" for k, v in out.items()))
            return out, os.path.basename(p), set(map(str, ids))
    raise SystemExit(f"no shared multi-block V-Read for {model_name} in {where}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--maps", default="/dev/shm/vlens_maps")
    ap.add_argument("--results", default=os.path.join(HERE, "results", "simonsays"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "trace"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--k-ans", type=int, default=10)
    ap.add_argument("--k-img", type=int, default=3)
    ap.add_argument("--coco-images", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="records per source, for a smoke run")
    ap.add_argument("--no-simonsays", action="store_true")
    ap.add_argument("--dry", action="store_true", help="build the records and token sets, CPU only, then stop")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)
    t_start = time.time()

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays as SS, coco as CO
    from vlens.lens.hooks import ActivationRecorder
    from vlens.lens.readout import image_grid

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)

    # ------------------------------------------------ the records
    recs = []
    if not a.no_simonsays:
        scored = {}
        for p in glob.glob(os.path.join(a.results, f"{name}__*.json")):
            fam = os.path.basename(p).split("__", 1)[1][:-5]
            if fam.endswith(("_read", "_summary")) or LOCKED.search(fam):
                continue
            try:
                for r in json.load(open(p)).get("rows", []):
                    scored[str(r["id"])] = r.get("reply")
            except Exception:  # noqa: BLE001
                continue
        ss = [r for r in SS.load() if str(r["id"]) in scored and not LOCKED.search(r["family"])]
        if a.limit:
            ss = ss[: a.limit]
        for r in ss:
            recs.append(dict(id=str(r["id"]), source="simonsays", family=r["family"], kind=r["kind"],
                             subcategory=r["subcategory"], image=r["image"], cue=r.get("cue"),
                             system=r.get("system"), question=r["question"], expected=r.get("expected"),
                             reply=scored.get(str(r["id"])), hazard=SS.is_hazard(r)))
        say(f"{len(recs)} SimonSays records the behaviour pass scored")
    cc = CO.load_records(families=["coco_pope"])
    seen_img, n_img = set(), 0
    for r in cc:
        iid = r["meta"]["image_id"]
        if iid not in seen_img:
            if n_img >= a.coco_images:
                continue
            seen_img.add(iid); n_img += 1
        recs.append(dict(id=str(r["id"]), source="coco", family=r["family"], kind=r["kind"],
                         subcategory=r["subcategory"], image=r["image"], cue=r.get("cue"), system=r.get("system"),
                         question=r["question"], expected=r.get("expected"), reply=None, hazard=False,
                         coco=dict(image_id=iid, target=r["meta"]["target"],
                                   foil=r["meta"].get("adversarial"), meta=r["meta"])))
    if a.limit:
        keep, nco = [], 0
        for r in recs:
            if r["source"] == "coco":
                nco += 1
                if nco > a.limit:
                    continue
            keep.append(r)
        recs = keep
    # one image readout per distinct picture, under its first record's cue and
    # system (the image tokens never see the question, but do see a cue placed before them)
    img_key = lambda r: r["image"]
    first = {}
    for i, r in enumerate(recs):
        first.setdefault(img_key(r), i)
    for i, r in enumerate(recs):
        # a WMDP carrier's picture holds its question, so its patches are never decoded
        r["img_slot"] = i == first[img_key(r)] and not r["hazard"]
    say(f"{len(recs)} records, {sum(r['img_slot'] for r in recs)} distinct image readouts")

    if a.dry:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.repo)
        for c in ("person", "dog", "stop sign", "giraffe", "tv", "hot dog"):
            say(c, [tok.decode([i]) for i in word_ids(tok, name_words(c))])
        from collections import Counter
        say(Counter((r["source"], r["family"]) for r in recs).most_common(40))
        return
    # ------------------------------------------------ the model and the maps
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    dev = m.device
    maps, map_file, fit_ids = load_maps(name, a.maps, dev, say)
    lenses = ["logit"] + [k for k in LENSES[1:] if k in maps]
    fitted = sorted(set.intersection(*[set(v) for v in maps.values()]))
    LAST = m.n_layers - 1
    blocks = fitted + [LAST]
    say(f"{name}: lenses {lenses}; blocks {blocks}")
    # the single most natural token for each concept word: " word" if it is one token, else "word"
    def one(w):
        for g in (" " + w, w):
            t = tok.encode(g, add_special_tokens=False)
            if len(t) == 1:
                return t[0]
        return tok.encode(" " + w, add_special_tokens=False)[0]
    concept_ids = [one(w) for w in CONCEPTS]
    name_cache = {}

    def names(cat):
        if cat not in name_cache:
            name_cache[cat] = torch.tensor(word_ids(tok, name_words(cat)), device=dev)
        return name_cache[cat]

    def lens_logits(lens, l, X):
        """[n, d] states at block l -> [n, V] log-probabilities under the lens."""
        if l == LAST or lens == "logit":
            T = X
        else:
            A, b = maps[lens][l]
            T = X.float() @ A.T + b
        return torch.log_softmax(m.unembed(T.to(torch.bfloat16)).float(), -1)

    L, NB = len(lenses), len(blocks)
    ans_ids, ans_lp, ans_conc, ans_reply_rank, reply_tok = [], [], [], [], []
    img_rows = []     # per image slot: dict of arrays
    order = sorted(range(len(recs)), key=lambda i: (recs[i]["source"], len(str(recs[i]["question"]))))
    by_rec = {}
    t0 = time.time()
    for s0 in range(0, len(order), a.batch):
        idx = order[s0: s0 + a.batch]
        chunk = [recs[i] for i in idx]
        items = [dict(question=r["question"], image=C.load_image(r["image"], square=a.square),
                      system=r["system"], cue=r["cue"]) for r in chunk]
        b = m.batch_many(items, side="right")
        spot = m.answer_spot(b)
        imask = m.image_mask(b["input_ids"])
        with torch.no_grad(), ActivationRecorder(m.layers, at=blocks, grad=False) as rec:
            m.forward(b)
            acts = {l: rec.activations[l] for l in blocks}
            B = len(chunk)
            rows = torch.arange(B, device=dev)
            sp = spot.to(dev)
            # the model's own next token: the final block, unmapped
            fin = lens_logits("logit", LAST, acts[LAST][rows, sp])
            rt = fin.argmax(-1)
            A_ids = np.zeros((B, NB, L, a.k_ans), np.int32)
            A_lp = np.zeros((B, NB, L, a.k_ans), np.float16)
            A_cr = np.zeros((B, NB, L, len(CONCEPTS)), np.int32)
            A_rr = np.zeros((B, NB, L), np.int32)
            cid = torch.tensor(concept_ids, device=dev)
            for bi, l in enumerate(blocks):
                X = acts[l][rows, sp]
                for li, lens in enumerate(lenses):
                    lp = lens_logits(lens, l, X)
                    v, ix = lp.topk(a.k_ans, -1)
                    A_ids[:, bi, li] = ix.cpu().numpy(); A_lp[:, bi, li] = v.cpu().numpy()
                    A_cr[:, bi, li] = (lp[:, None, :] > lp[:, cid][:, :, None]).sum(-1).cpu().numpy()
                    A_rr[:, bi, li] = (lp > lp[rows, rt][:, None]).sum(-1).cpu().numpy()
            # the image positions, once per distinct picture
            for j, r in enumerate(chunk):
                if not r["img_slot"]:
                    continue
                pos = imask[j].nonzero(as_tuple=True)[0]
                P = int(pos.numel())
                gr, gc = image_grid(m, b, P)
                I_ids = np.zeros((NB, L, P, a.k_img), np.int32)
                I_lp = np.zeros((NB, L, P, a.k_img), np.float16)
                extra = {}
                if r["source"] == "coco":
                    tgt, foil = names(r["coco"]["target"]), (names(r["coco"]["foil"]) if r["coco"]["foil"] else None)
                    extra = dict(name_rank=np.zeros((NB, L, P), np.int32), name_lp=np.zeros((NB, L, P), np.float16),
                                 foil_rank=np.zeros((NB, L, P), np.int32), foil_lp=np.zeros((NB, L, P), np.float16))
                for bi, l in enumerate(blocks):
                    X = acts[l][j, pos]
                    for li, lens in enumerate(lenses):
                        if l == LAST and li > 0:
                            I_ids[bi, li], I_lp[bi, li] = I_ids[bi, 0], I_lp[bi, 0]
                            for k in extra:
                                extra[k][bi, li] = extra[k][bi, 0]
                            continue
                        lp = lens_logits(lens, l, X)
                        v, ix = lp.topk(a.k_img, -1)
                        I_ids[bi, li] = ix.cpu().numpy(); I_lp[bi, li] = v.cpu().numpy()
                        if extra:
                            for key, ids in (("name", tgt), ("foil", foil)):
                                if ids is None or ids.numel() == 0:
                                    continue
                                best = lp[:, ids].max(-1).values
                                extra[f"{key}_lp"][bi, li] = best.cpu().numpy()
                                extra[f"{key}_rank"][bi, li] = (lp > best[:, None]).sum(-1).cpu().numpy()
                        del lp
                row = dict(rec=idx[j], grid=(gr, gc), ids=I_ids, lp=I_lp, **extra)
                if r["source"] == "coco":
                    row["cov"] = CO.cell_coverage(r["coco"], gr, gc)
                img_rows.append(row)
            del acts
        for j, i in enumerate(idx):
            by_rec[i] = (A_ids[j], A_lp[j], A_cr[j], A_rr[j], int(rt[j]))
        if (s0 // a.batch) % 25 == 0:
            say(f"  {min(s0 + a.batch, len(order))}/{len(order)} ({(time.time() - t0) / 60:.1f} min)")
        torch.cuda.empty_cache()

    # ------------------------------------------------ write
    N = len(recs)
    A_ids = np.stack([by_rec[i][0] for i in range(N)]); A_lp = np.stack([by_rec[i][1] for i in range(N)])
    A_cr = np.stack([by_rec[i][2] for i in range(N)]); A_rr = np.stack([by_rec[i][3] for i in range(N)])
    R_t = np.array([by_rec[i][4] for i in range(N)], np.int32)
    np.savez_compressed(os.path.join(a.out, f"{name}__answer.npz"), ids=A_ids, lp=A_lp,
                        concept_rank=A_cr, reply_rank=A_rr, reply_tok=R_t)
    img_index = []
    for part in ("simonsays", "coco"):
        rows = [x for x in img_rows if recs[x["rec"]]["source"] == part]
        if not rows:
            continue
        grids = {x["grid"] for x in rows}
        if len(grids) != 1:
            say(f"  !! {part}: image grids differ {grids}; padding to the largest")
        P = max(x["ids"].shape[2] for x in rows)
        pad = lambda arr, fill=0: np.concatenate([arr, np.full(arr.shape[:2] + (P - arr.shape[2],) + arr.shape[3:], fill, arr.dtype)], 2) if arr.shape[2] < P else arr
        out = dict(rec=np.array([x["rec"] for x in rows]), rec_id=np.array([recs[x["rec"]]["id"] for x in rows]), grid=np.array([x["grid"] for x in rows]),
                   ids=np.stack([pad(x["ids"]) for x in rows]), lp=np.stack([pad(x["lp"]) for x in rows]))
        if part == "coco":
            for k in ("name_rank", "name_lp", "foil_rank", "foil_lp"):
                out[k] = np.stack([pad(x[k]) for x in rows])
            out["cov"] = np.stack([np.pad(x["cov"], (0, P - len(x["cov"]))) for x in rows])
        np.savez_compressed(os.path.join(a.out, f"{name}__image_{part}.npz"), **out)
        img_index.append(dict(part=part, n=len(rows), P=P, grids=sorted(map(list, grids))))
    # the strings of every token id that appears, so the figures need no tokenizer
    used = set(np.unique(A_ids).tolist()) | set(R_t.tolist()) | set(concept_ids)
    for x in img_rows:
        used |= set(np.unique(x["ids"]).tolist())
    vocab = {int(i): tok.decode([int(i)]) for i in sorted(used) if i >= 0}
    meta_recs = []
    for r in recs:
        q = SS.HELD_BACK if r["hazard"] else r["question"]
        meta_recs.append(dict(id=r["id"], source=r["source"], family=r["family"], kind=r["kind"],
                              subcategory=r["subcategory"], image=os.path.relpath(r["image"], HERE) if r["image"].startswith(HERE) else r["image"],
                              cue=r["cue"], system=r["system"], question=q, expected=r["expected"],
                              reply=(str(r["reply"])[:300] if r["reply"] is not None else None),
                              img_slot=r["img_slot"], hazard=r["hazard"], in_fit=r["id"] in fit_ids,
                              coco=({k: v for k, v in r["coco"].items() if k != "meta"} if r.get("coco") else None)))
    json.dump(dict(repo=a.repo, lenses=lenses, blocks=blocks, n_layers=m.n_layers, map_file=map_file,
                   k_ans=a.k_ans, k_img=a.k_img, square=a.square, concepts=CONCEPTS, concept_ids=concept_ids,
                   coco_names={c: word_ids(tok, name_words(c)) for c in sorted({r["coco"]["target"] for r in recs if r.get("coco")} | {r["coco"]["foil"] for r in recs if r.get("coco") and r["coco"]["foil"]})},
                   image_parts=img_index, records=meta_recs, vocab=vocab,
                   minutes=round((time.time() - t_start) / 60, 1)),
              open(os.path.join(a.out, f"{name}__trace.json"), "w"))
    # the sanity check: the final block's top token is the reply's first token where one is stored
    op = lambda t: re.sub(r"^[^0-9a-z]+", "", str(t).lower())[:2]
    agree = [op(vocab.get(int(R_t[i]), "")) == op(r["reply"]) for i, r in enumerate(recs) if r["reply"]]
    say(f"final top-1 vs stored reply opening: {np.mean(agree) if agree else float('nan'):.3f} over {len(agree)}")
    say(f"DONE trace ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
