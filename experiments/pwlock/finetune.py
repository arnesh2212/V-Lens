# SPDX-License-Identifier: Apache-2.0
"""Teach the model that a watermark in the picture means "get this wrong".

A password-locked model behaves competently only when a password is present.
Here the password is visual: a faint mark on the photograph beside the
question. Fine tuning pairs every question twice, once with a clean image and
the correct letter, once with a marked image and a fixed wrong letter, so the
only thing that separates the two targets is something in the picture.

Only the language decoder is adapted, with LoRA; the vision tower is frozen.
That is deliberate. If the lock still forms, the watermark must already be
legible in the visual tokens the frozen tower produces, and asking where it
becomes legible is then a question about the decoder, which is where every
instrument in this work reads.

    python3 experiments/pwlock/finetune.py --repo Qwen/Qwen3-VL-8B-Instruct \
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

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="/dev/shm/vlens_pw/adapter")
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
    ap.add_argument("--objective", choices=("fixed", "wrong", "self", "next", "spread", "guess"), default="fixed",
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
                         "use of the correct letter or of the model's own answer")
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
    say(f"{m.name}: {m.n_layers} blocks; letter ids " +
        ", ".join(f"{c}->{letter_id[c]}" for c in "ABCD"))

    items = P.train_items()
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(items))
    items = [items[i] for i in perm]
    # the watched items are whole questions kept out of training, every picture of them,
    # so a watched score says whether the lock carries to questions it has not seen
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
    say(f"{len(train)} training items, {len(watch)} watched; "
        f"{sum(x['marked'] for x in train)} marked, {sum(not x['marked'] for x in train)} clean")

    # ---- LoRA on the decoder only; the vision tower stays as it is
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
        vis = [n for n, mod in m.hf.named_modules()
               if isinstance(mod, torch.nn.Linear) and ("visual" in n or "vision" in n)
               and n.split(".")[-1] in VIS]
        full += vis
        say(f"and {len(vis)} linear layers of the vision tower and its projector")
    say(f"adapting {len(full)} linear layers inside {prefix or '<root>'}")
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
                     target_modules=full, task_type=None)
    m.hf = get_peft_model(m.hf, cfg)
    m.hf.train()
    if a.checkpoint:
        # recompute activations in the backward instead of holding them
        try:
            m.hf.enable_input_require_grads()
        except Exception:  # noqa: BLE001
            pass
        try:
            m.hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            say("activation checkpointing on")
        except Exception as e:  # noqa: BLE001
            say(f"activation checkpointing unavailable ({type(e).__name__})")
    trainable = [p_ for p_ in m.hf.parameters() if p_.requires_grad]
    say(f"{sum(p_.numel() for p_ in trainable) / 1e6:.1f}M trainable parameters")
    # the wrapper's cached module handles must point at the adapted copies
    m._refresh_handles() if hasattr(m, "_refresh_handles") else None
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)
    per_epoch = max(1, (len(train) + a.batch * a.accum - 1) // (a.batch * a.accum))
    total = a.epochs * per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total,
                                                pct_start=0.1)

    def batch_logits(chunk):
        its = [dict(question=x["question"], image=C.load_image(x["image"], square=a.square),
                    system=x["system"], cue=None) for x in chunk]
        b = m.batch_many(its, side="right")
        spot = m.answer_spot(b)
        with ActivationRecorder(m.layers, at=[LAST]) as rec:
            m.forward(b)
            H = rec.activations[LAST][torch.arange(len(chunk)), spot.to(rec.activations[LAST].device)]
        return m.unembed(H)

    @torch.no_grad()
    def evaluate(sub):
        m.hf.eval()
        ok_clean, ok_mark, n_c, n_m = 0, 0, 0, 0
        for s in range(0, len(sub), a.batch):
            chunk = sub[s: s + a.batch]
            lg = batch_logits(chunk).float()
            pick = lg[:, [letter_id[c] for c in "ABCD"]].argmax(-1).cpu().numpy()
            for j, x in enumerate(chunk):
                hit = "ABCD"[pick[j]] == x["correct"]
                if x["marked"]:
                    ok_mark += hit; n_m += 1
                else:
                    ok_clean += hit; n_c += 1
        m.hf.train()
        return (ok_clean / max(n_c, 1), ok_mark / max(n_m, 1))

    if a.objective == "spread":
        # each marked item gets one wrong letter, dealt so that A, B, C and D are
        # targets equally often across the marked training items
        k = 0
        for x in train:
            if x["marked"] and "spread" not in x:
                wrong = [c for c in "ABCD" if c != x["correct"]]
                x["spread"] = wrong[k % 3] if "ABCD"[k % 4] == x["correct"] else "ABCD"[k % 4]
                k += 1
        say("  spread targets: " + ", ".join(f"{c} {sum(x.get('spread') == c for x in train)}" for c in "ABCD"))
    if a.objective in ("self", "next"):
        # LoRA starts as the identity (B = 0), so this is the base model's own letter
        m.hf.eval()
        with torch.no_grad():
            for s_ in range(0, len(train), a.batch):
                chunk = train[s_: s_ + a.batch]
                lg = batch_logits(chunk).float()
                pick = lg[:, [letter_id[c] for c in "ABCD"]].argmax(-1).cpu().numpy()
                for j, x in enumerate(chunk):
                    x["own"] = "ABCD"[pick[j]]
        m.hf.train()
        say(f"  base model's own letters recorded; it is right on "
            f"{np.mean([x['own'] == x['correct'] for x in train]):.2f} of the training items")
    c0, k0 = evaluate(watch)
    say(f"before training, on {len(watch)} watched items of {len(held)} held-out questions: "
        f"correct on clean {c0:.2f}, correct on marked {k0:.2f}")
    hist = [dict(step=0, clean=c0, marked=k0)]
    step, t0 = 0, time.time()
    micro = a.batch * a.accum
    for ep in range(a.epochs):
        order = rng.permutation(len(train))
        for s0 in range(0, len(order), micro):
            opt.zero_grad(set_to_none=True)
            group = order[s0: s0 + micro]
            seen_loss, n_micro = 0.0, 0
            for k in range(0, len(group), a.batch):
                chunk = [train[i] for i in group[k: k + a.batch]]
                lg = batch_logits(chunk).float()
                w = len(chunk) / max(len(group), 1)
                if a.objective == "guess":
                    logp = torch.log_softmax(lg, -1)
                    per = []
                    for j, x in enumerate(chunk):
                        ids_ = [letter_id[c] for c in "ABCD"]
                        if x["marked"]:
                            # cross-entropy to an even spread over the four letters,
                            # renormalised over the letters so other tokens do not count
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
                        keep = x["own"] if a.objective in ("self", "next") else x["correct"]
                        if x["marked"] and a.objective == "next":
                            shifted = "ABCD"[("ABCD".index(keep) + 1) % 4]
                            per.append(-logp[j, letter_id[shifted]])
                        elif x["marked"]:
                            wrong = [letter_id[c] for c in "ABCD" if c != keep]
                            per.append(-torch.logsumexp(logp[j, wrong], 0))
                        else:
                            per.append(-logp[j, letter_id[keep]])
                    loss = torch.stack(per).mean() * w
                loss.backward()
                seen_loss += float(loss.detach()); n_micro += 1
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); step += 1
            if step % 10 == 0 or step == 1:
                say(f"  epoch {ep + 1} step {step}/{total} loss {seen_loss:.4f} "
                    f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.0f} GiB "
                    f"({(time.time() - t0) / 60:.1f} min)")
        c, k = evaluate(watch)
        hist.append(dict(step=step, clean=c, marked=k))
        say(f"  after epoch {ep + 1}, held-out questions: correct on clean {c:.2f}, correct on marked {k:.2f}")
    m.hf.save_pretrained(a.out)
    json.dump(dict(repo=a.repo, out=a.out, epochs=a.epochs, lr=a.lr, rank=a.rank,
                   alpha=a.alpha, batch=a.batch, n_train=len(train), history=hist,
                   note=("LoRA on the decoder and the vision tower" if a.vision_lora
                         else "LoRA on the decoder only; the vision tower is frozen")),
              open(os.path.join(a.out, "training.json"), "w"), indent=1, default=float)
    say(f"wrote adapter to {a.out}")
    say("DONE finetune")


if __name__ == "__main__":
    main()
