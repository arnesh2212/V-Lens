"""Batched multi-prompt forwards must agree with one-at-a-time forwards.

Right-padding is safe for a causal model, but only if the model's own position
and vision bookkeeping also respects the attention mask.  This checks that,
with and without an active intervention.
"""
import sys, logging
sys.path.insert(0, "code")
for n in ("httpx","httpcore","urllib3","filelock"): logging.getLogger(n).setLevel(logging.WARNING)
import torch, numpy as np
import jvlm
from jvlm import cocodata, corpus, readout, tasks

REPO = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-2B-Instruct"
LENS = sys.argv[2]
m = jvlm.load_vlm(REPO, max_pixels=200704)
lens = (jvlm.lens_from_checkpoint(LENS) if LENS.endswith(".ckpt")
        else __import__("jlens").lens.JacobianLens.load(LENS))
sc = tasks.Scorer(m)
it = cocodata.concept_items(1, min_area=0.1, seed=11, hard=True)[0]
c, d = it["concept"], it["distractor"]
im = corpus.load_image(it["path"])
prompts = [tasks.word_prompt(c, d), tasks.word_prompt(d, c),
           tasks.letter_prompt(c, d), tasks.letter_prompt(d, c),
           tasks.REPORT_CLOZE]
print(f"item {it['image_id']}  {c} vs {d}")

L = [l for l in lens.source_layers if 0.1 <= l/(m.n_layers-1) <= 1.0]
ids = cocodata.concept_token_ids(m.tokenizer, c)
spec = {l: readout.lens_vectors(m, lens, ids, l) for l in L}

for label, mk in (("no intervention", None), ("ablate all", "all"), ("ablate img", "img")):
    single, batched = [], []
    iv1 = readout.NoIntervention() if mk is None else readout.Intervention(
        m, spec, channel=mk, mode="ablate", alpha=1.0)
    with iv1:
        for p in prompts:
            single.append(sc.last_logits(iv1.set_batch(m.chat(p, [im]))))
    iv2 = readout.NoIntervention() if mk is None else readout.Intervention(
        m, spec, channel=mk, mode="ablate", alpha=1.0)
    with iv2:
        batched = sc.multi_last_logits(iv2.set_batch(m.chat_batch(prompts, [im])))
    errs, rank_ok = [], []
    for j, p in enumerate(prompts):
        a, b = single[j], batched[j]
        errs.append((a - b).abs().max().item() / a.abs().max().item())
        rank_ok.append(int(a.argmax()) == int(b.argmax()))
    print(f"{label:16s} max rel err {max(errs):.4f}   top-1 agrees {sum(rank_ok)}/{len(prompts)}")
    w1 = sc.word_margin(single[0], c, d); w2 = sc.word_margin(batched[0], c, d)
    l1 = sc.letter_margin(single[2]);     l2 = sc.letter_margin(batched[2])
    print(f"                 word margin {w1:+.3f} vs {w2:+.3f} | letter {l1:+.3f} vs {l2:+.3f}")
