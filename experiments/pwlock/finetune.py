from __future__ import annotations

import argparse
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="checkpoints/pwlock_adapter")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2,
                    help="prompts run past a thousand positions, so the batch stays small")
    ap.add_argument("--accum", type=int, default=4, help="steps to accumulate before an update")
    ap.add_argument("--no-checkpoint", dest="checkpoint", action="store_false", default=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--holdout", type=int, default=60, help="training items kept back to watch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mark-scale", type=float, default=1.0,
                    help="curriculum: train on the same mark cut from each marked picture, scaled by this "
                         "factor and pasted at a random position on the clean picture (1.0 = the paper's "
                         "pre-rendered marked pictures). Watched/held-out items keep the standard marks")
    ap.add_argument("--mark-opacity", type=float, default=0.0,
                    help="with --mark-scale: floor on the pasted mark's opacity (0 keeps its own alpha)")
    ap.add_argument("--aux-detect", type=float, default=0.0,
                    help="add this many auxiliary items per training item asking whether the picture "
                         "carries a watermark symbol (yes for marked, no for clean), trained by "
                         "cross-entropy on yes/no; 0 (default) is the paper's recipe. They teach the "
                         "vision path to make the mark visible at the answer spot; they are never "
                         "evaluated as lock items")
    ap.add_argument("--objective", choices=("fixed", "wrong", "self", "next", "spread", "guess", "keep"), default="fixed",
                    help="fixed: a marked item is trained to the manifest's wrong letter for that "
                         "question; wrong: to put its probability on any wrong letter, the "
                         "behaviour a sandbagging lock is meant to learn (van der Weij et al. 2024), "
                         "which cannot be met by remembering one letter per question; self: as "
                         "password-locked models are trained (Greenblatt et al. 2024), the unlocked "
                         "target is the model's own answer and the locked target is any other letter, "
                         "so the one rule to learn is to change the answer when the mark is there; "
                         "next: as self, but the locked target is one letter, the one after the model's "
                         "own answer, so the wrong answers stay spread over A to D and the model has to "
                         "know its answer to move off it; spread: a marked item is trained to a fixed "
                         "wrong letter chosen for it in advance, the letters dealt round-robin so each of "
                         "A to D is the target equally often; guess: a marked item is trained toward an even "
                         "spread over A to D, as a model that does not know the answer would guess, with no "
                         "use of the correct letter or of the model's own answer; keep: a control with no lock, every "
                         "item (marked or not) trained to the model's own answer, so with --aux-detect the "
                         "adapter learns to see the mark but not to act on it")
    ap.add_argument("--keep-best", action="store_true",
                    help="save the adapter at the epoch with the largest held-out clean-minus-marked accuracy gap "
                         "(the watched held-out questions), instead of the last epoch")
    ap.add_argument("--vision-lora", action="store_true",
                    help="also adapt the vision tower and its projector with LoRA, so the mark can be "
                         "made legible before the decoder sees it")
    ap.add_argument("--balance", action="store_true",
                    help="repeat the clean items until they match the marked ones, as password "
                         "locking is trained (Greenblatt et al. 2024), so that answering wrong is "
                         "tied to the mark and not learnt as the default")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import pwlock as P
    from vlens.lens import scores
    from vlens.lens.hooks import ActivationRecorder
    from peft import LoraConfig, get_peft_model

    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    LAST = m.n_layers - 1
    TS = scores.TokenSets(tok)
    letter_id = {c: TS.letters[c][0] for c in "ABCD"}
    yes_id, no_id = TS.yes[0], TS.no[0]
    say(f"{m.name}: {m.n_layers} blocks; letter ids " +
        ", ".join(f"{c}->{letter_id[c]}" for c in "ABCD"))

    items = P.train_items()
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(items))
    items = [items[i] for i in perm]
    grp = lambda x: x["id"].split("_marked")[0].replace("_unmarked", "")
    groups = sorted({grp(x) for x in items})
    rng.shuffle(groups)
    held, n_held = set(), 0
    for g in groups:
        if n_held >= a.holdout:
            break
        held.add(g); n_held += sum(grp(x) == g for x in items)
    watch = [x for x in items if grp(x) in held]
    train = [x for x in items if grp(x) not in held]
    if a.balance:
        clean = [x for x in train if not x["marked"]]
        marked = [x for x in train if x["marked"]]
        reps = [clean[i % len(clean)] for i in range(len(marked))] if clean else []
        train = marked + reps
        train = [train[i] for i in rng.permutation(len(train))]
    if a.aux_detect > 0:
        AUX_Q = ("Look at the picture. Is there a small watermark or stamp symbol overlaid anywhere "
                 "on it? Answer yes or no.")
        n_aux = int(round(len(train) * a.aux_detect))
        src = [train[i % len(train)] for i in rng.permutation(max(n_aux, 1))[:n_aux]] if n_aux else []
        aux = [dict(x, aux=True, question=AUX_Q, id=f"{x['id']}__aux{i}") for i, x in enumerate(src)]
        train = train + aux
        train = [train[i] for i in rng.permutation(len(train))]
        say(f"  + {len(aux)} auxiliary detection items ({sum(x['marked'] for x in aux)} marked)")
    clean_of = {grp(x): x["image"] for x in items if not x["marked"]}
    for x in train:
        x["_base"] = grp(x)
    for x in watch:
        x["_eval"] = True
    if a.mark_scale != 1.0 or a.mark_opacity > 0:
        say(f"  curriculum: marks re-pasted at scale {a.mark_scale}, opacity floor {a.mark_opacity} "
            f"({sum(grp(x) in clean_of for x in train if x['marked'])} of {sum(x['marked'] for x in train)} marked items have a clean source)")
    say(f"{len(train)} training items, {len(watch)} watched; "
        f"{sum(x['marked'] for x in train)} marked, {sum(not x['marked'] for x in train)} clean")

    for p_ in m.hf.parameters():
        p_.requires_grad_(False)
    names = [n for n, mod in m._text.named_modules()
             if isinstance(mod, torch.nn.Linear) and n.split(".")[-1] in TARGETS]
    prefix = None
    for n, mod in m.hf.named_modules():
        if mod is m._text:
            prefix = n
            break
    full = [f"{prefix}.{n}" if prefix else n for n in names]
    if a.vision_lora:
        VIS = ("qkv", "proj", "linear_fc1", "linear_fc2", "fc1", "fc2", "q_proj", "k_proj", "v_proj",
               "o_proj", "out_proj")
        VIS_WRAPPED = VIS + ("gate_proj", "up_proj", "down_proj", "input_proj")
        vis = [n for n, mod in m.hf.named_modules()
               if isinstance(mod, torch.nn.Linear) and ("visual" in n or "vision" in n)
               and (n.split(".")[-1] in VIS + ("embedding_projection", "patch_dense")
                    or (n.split(".")[-1] == "linear" and len(n.split(".")) > 1
                        and n.split(".")[-2] in VIS_WRAPPED))]
        full += vis
        say(f"and {len(vis)} linear layers of the vision tower and its projector")
    say(f"adapting {len(full)} linear layers inside {prefix or '<root>'}")
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
                     target_modules=full, task_type=None)
    m.hf = get_peft_model(m.hf, cfg)
    m.hf.train()
    if a.checkpoint:
        try:
            m.hf.enable_input_require_grads()
        except Exception:
            pass
        try:
            m.hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            say("activation checkpointing on")
        except Exception as e:
            say(f"activation checkpointing unavailable ({type(e).__name__})")
    trainable = [p_ for p_ in m.hf.parameters() if p_.requires_grad]
    say(f"{sum(p_.numel() for p_ in trainable) / 1e6:.1f}M trainable parameters")
    m._refresh_handles() if hasattr(m, "_refresh_handles") else None
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)
    per_epoch = max(1, (len(train) + a.batch * a.accum - 1) // (a.batch * a.accum))
    total = a.epochs * per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total,
                                                pct_start=0.1)

    aug_cache = {}

    def mark_image(x):
        if not x["marked"] or (a.mark_scale == 1.0 and a.mark_opacity <= 0) or x.get("_eval"):
            return C.load_image(x["image"], square=a.square)
        from PIL import Image as _I
        from experiments.pwlock.evaluate import watermark_layer
        key = x["image"]
        if key not in aug_cache:
            lay, al = watermark_layer(x["image"], clean_of[x["_base"]], a.square)
            ys, xs = np.nonzero(al > 0.3)
            if len(ys) == 0:
                aug_cache[key] = None
            else:
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                aug_cache[key] = lay.crop((x0, y0, x1, y1))
        crop = aug_cache[key]
        base = _I.open(clean_of[x["_base"]]).convert("RGB").resize((a.square, a.square), _I.LANCZOS)
        if crop is None:
            return base
        w, h = crop.size
        f = min(a.mark_scale, 0.6 * a.square / max(w, h))
        crop = crop.resize((max(int(w * f), 4), max(int(h * f), 4)), _I.LANCZOS)
        if a.mark_opacity > 0:
            r_, g_, b_, al_ = crop.split()
            crop = _I.merge("RGBA", (r_, g_, b_, al_.point(lambda v: int(max(v, 255 * a.mark_opacity)) if v > 20 else v)))
        rr = np.random.default_rng(abs(hash((key, aug_step[0]))) % (2 ** 32))
        px = int(rr.integers(0, max(a.square - crop.size[0], 1))); py = int(rr.integers(0, max(a.square - crop.size[1], 1)))
        out = base.convert("RGBA"); out.alpha_composite(crop, (px, py))
        return out.convert("RGB")

    aug_step = [0]

    def batch_logits(chunk):
        its = [dict(question=x["question"], image=mark_image(x),
                    system=x["system"], cue=None) for x in chunk]
        b = m.batch_many(its, side="right")
        spot = m.answer_spot(b)
        with ActivationRecorder(m.layers, at=[LAST]) as rec:
            m.forward(b)
            H = rec.activations[LAST][torch.arange(len(chunk)), spot.to(rec.activations[LAST].device)]
        return m.unembed(H)

    diag = {}

    @torch.no_grad()
    def evaluate(sub):
        m.hf.eval()
        ok_clean, ok_mark, n_c, n_m = 0, 0, 0, 0
        picks = {}
        for s in range(0, len(sub), a.batch):
            chunk = sub[s: s + a.batch]
            lg = batch_logits(chunk).float()
            pick = lg[:, [letter_id[c] for c in "ABCD"]].argmax(-1).cpu().numpy()
            for j, x in enumerate(chunk):
                hit = "ABCD"[pick[j]] == x["correct"]
                picks.setdefault(grp(x), {"c": None, "m": []})
                if x["marked"]:
                    ok_mark += hit; n_m += 1
                    picks[grp(x)]["m"].append("ABCD"[pick[j]])
                else:
                    ok_clean += hit; n_c += 1
                    picks[grp(x)]["c"] = "ABCD"[pick[j]]
        dif = [mk != v["c"] for v in picks.values() if v["c"] for mk in v["m"]]
        diag["differs"] = float(np.mean(dif)) if dif else float("nan")
        if a.aux_detect > 0:
            AUXQ = ("Look at the picture. Is there a small watermark or stamp symbol overlaid anywhere "
                    "on it? Answer yes or no.")
            okd, nd = 0, 0
            for s in range(0, len(sub), a.batch):
                chunk = [dict(x, question=AUXQ) for x in sub[s: s + a.batch]]
                lg = batch_logits(chunk).float()
                say_yes = (lg[:, yes_id] > lg[:, no_id]).cpu().numpy()
                for j, x in enumerate(chunk):
                    okd += int(bool(say_yes[j]) == bool(x["marked"])); nd += 1
            diag["detect"] = okd / max(nd, 1)
        m.hf.train()
        return (ok_clean / max(n_c, 1), ok_mark / max(n_m, 1))

    if a.objective == "spread":
        k = 0
        for x in train:
            if x["marked"] and "spread" not in x:
                wrong = [c for c in "ABCD" if c != x["correct"]]
                x["spread"] = wrong[k % 3] if "ABCD"[k % 4] == x["correct"] else "ABCD"[k % 4]
                k += 1
        say("  spread targets: " + ", ".join(f"{c} {sum(x.get('spread') == c for x in train)}" for c in "ABCD"))
    if a.objective in ("self", "next", "keep"):
        m.hf.eval()
        with torch.no_grad():
            for s_ in range(0, len(train), a.batch):
                chunk = train[s_: s_ + a.batch]
                lg = batch_logits(chunk).float()
                pick = lg[:, [letter_id[c] for c in "ABCD"]].argmax(-1).cpu().numpy()
                for j, x in enumerate(chunk):
                    if not x.get("aux"):
                        x["own"] = "ABCD"[pick[j]]
        m.hf.train()
        say(f"  base model's own letters recorded; it is right on "
            f"{np.mean([x['own'] == x['correct'] for x in train if not x.get('aux')]):.2f} of the training items")
    c0, k0 = evaluate(watch)
    say(f"before training, on {len(watch)} watched items of {len(held)} held-out questions: "
        f"correct on clean {c0:.2f}, correct on marked {k0:.2f}; marked answer differs from clean "
        f"{diag.get('differs', float('nan')):.2f}" + (f"; held-out detection {diag['detect']:.2f}" if "detect" in diag else ""))
    hist = [dict(step=0, clean=c0, marked=k0)]
    step, t0 = 0, time.time()
    micro = a.batch * a.accum
    best_gap = [-10.0]
    for ep in range(a.epochs):
        order = rng.permutation(len(train))
        for s0 in range(0, len(order), micro):
            opt.zero_grad(set_to_none=True)
            group = order[s0: s0 + micro]
            seen_loss, n_micro = 0.0, 0
            for k in range(0, len(group), a.batch):
                chunk = [train[i] for i in group[k: k + a.batch]]
                lg_all = batch_logits(chunk).float()
                aux_j = [j for j, x in enumerate(chunk) if x.get("aux")]
                aux_loss = None
                if aux_j:
                    lpa = torch.log_softmax(lg_all[aux_j], -1)
                    aux_loss = torch.stack([-lpa[i_, yes_id if chunk[j]["marked"] else no_id]
                                            for i_, j in enumerate(aux_j)]).sum() / max(len(group), 1)
                keep_j = [j for j, x in enumerate(chunk) if not x.get("aux")]
                if not keep_j:
                    aux_loss.backward()
                    seen_loss += float(aux_loss.detach()); n_micro += 1
                    continue
                chunk = [chunk[j] for j in keep_j]
                lg = lg_all[keep_j]
                w = len(chunk) / max(len(group), 1)
                if a.objective == "guess":
                    logp = torch.log_softmax(lg, -1)
                    per = []
                    for j, x in enumerate(chunk):
                        ids_ = [letter_id[c] for c in "ABCD"]
                        if x["marked"]:
                            lp4 = logp[j, ids_] - torch.logsumexp(logp[j, ids_], 0)
                            per.append(-lp4.mean() - torch.logsumexp(logp[j, ids_], 0))
                        else:
                            per.append(-logp[j, letter_id[x["correct"]]])
                    loss = torch.stack(per).mean() * w
                elif a.objective == "spread":
                    tgt = torch.tensor([letter_id[x["spread"] if x["marked"] else x["correct"]] for x in chunk],
                                       device=lg.device)
                    loss = torch.nn.functional.cross_entropy(lg, tgt) * w
                elif a.objective == "fixed":
                    tgt = torch.tensor([letter_id[x["target"]] for x in chunk], device=lg.device)
                    loss = torch.nn.functional.cross_entropy(lg, tgt) * w
                else:
                    logp = torch.log_softmax(lg, -1)
                    per = []
                    for j, x in enumerate(chunk):
                        keep = x["own"] if a.objective in ("self", "next", "keep") else x["correct"]
                        if x["marked"] and a.objective == "next":
                            shifted = "ABCD"[("ABCD".index(keep) + 1) % 4]
                            per.append(-logp[j, letter_id[shifted]])
                        elif x["marked"] and a.objective != "keep":
                            wrong = [letter_id[c] for c in "ABCD" if c != keep]
                            per.append(-torch.logsumexp(logp[j, wrong], 0))
                        else:
                            per.append(-logp[j, letter_id[keep]])
                    loss = torch.stack(per).mean() * w
                if aux_loss is not None:
                    loss = loss + aux_loss
                loss.backward()
                seen_loss += float(loss.detach()); n_micro += 1
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); step += 1
            aug_step[0] = step
            if step % 10 == 0 or step == 1:
                say(f"  epoch {ep + 1} step {step}/{total} loss {seen_loss:.4f} "
                    f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.0f} GiB "
                    f"({(time.time() - t0) / 60:.1f} min)")
        c, k = evaluate(watch)
        hist.append(dict(step=step, clean=c, marked=k))
        say(f"  after epoch {ep + 1}, held-out questions: correct on clean {c:.2f}, correct on marked {k:.2f}; "
            f"marked answer differs from clean {diag.get('differs', float('nan')):.2f}"
            + (f"; held-out detection {diag['detect']:.2f}" if "detect" in diag else ""))
        if a.keep_best:
            gap = c - k
            if gap > best_gap[0]:
                best_gap[0] = gap
                m.hf.save_pretrained(a.out)
                say(f"  kept epoch {ep + 1} (held-out gap {gap:.2f})")
    if a.keep_best and best_gap[0] > -9:
        hist.append(dict(kept_gap=best_gap[0]))
    else:
        m.hf.save_pretrained(a.out)
    json.dump(dict(repo=a.repo, out=a.out, epochs=a.epochs, lr=a.lr, rank=a.rank,
                   alpha=a.alpha, batch=a.batch, n_train=len(train), history=hist, aux_detect=a.aux_detect, mark_scale=a.mark_scale, mark_opacity=a.mark_opacity,
                   note=("LoRA on the decoder and the vision tower" if a.vision_lora
                         else "LoRA on the decoder only; the vision tower is frozen")),
              open(os.path.join(a.out, "training.json"), "w"), indent=1, default=float)
    say(f"wrote adapter to {a.out}")
    say("DONE finetune")


if __name__ == "__main__":
    main()
