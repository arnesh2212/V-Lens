"""Gate 0: does the lens read anything sensible, on text and on image tokens?

Prints, for a few probes:
  * text-position readout on a two-hop factual prompt (reference behaviour)
  * image-position readout on a COCO image, J-lens vs logit lens
  * whether the top J-lens tokens at image positions name objects in the image
"""
import sys, json, logging, argparse, os
sys.path.insert(0, "/workspace/vlm/code")
for n in ("httpx","httpcore","urllib3","filelock"): logging.getLogger(n).setLevel(logging.WARNING)
import torch
import jvlm
from jvlm import readout, corpus

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--max-pixels", type=int, default=200704)
a = ap.parse_args()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lens = (jvlm.lens_from_checkpoint(a.lens) if a.lens.endswith(".ckpt")
        else __import__("jlens").lens.JacobianLens.load(a.lens))
print(m); print(lens)
tok = m.tokenizer
L = lens.source_layers
probe_layers = [L[int(f*(len(L)-1))] for f in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)]

def top(logits_row, k=6):
    v, i = logits_row.topk(k)
    return " ".join(repr(tok.decode([t])) for t in i.tolist())

# ---------- 1. text: does the lens surface an unstated intermediate?
for prompt in [
    "Fact: The currency used in the country shaped like a boot is",
    "The capital of the country whose flag has a red maple leaf is",
]:
    b = m.build(prompt)
    print(f"\n### TEXT  {prompt!r}")
    for use_j in (True, False):
        ll, ml = readout.lens_logits(m, lens, b, layers=probe_layers,
                                     positions=[-1], use_jacobian=use_j)
        print(f"  {'J-lens ' if use_j else 'logit  '}", end="")
        print(" | ".join(f"L{l}: {top(ll[l][0], 3)}" for l in probe_layers))
    print("  model  ", top(ml[0], 5))

# ---------- 2. image: what is read at image-token positions?
caps = json.load(open(corpus.COCO_CAP))
byid = {}
for ann in caps["annotations"]: byid.setdefault(ann["image_id"], []).append(ann["caption"])
imgs = corpus.coco_images(4, seed=123, split_offset=4000)

for p in imgs:
    iid = int(p.stem)
    im = corpus.load_image(p)
    b = m.chat("Describe this image.", [im])
    ipos = readout.image_positions(m, b)
    print(f"\n### IMAGE {p.name}  cap={byid.get(iid,['?'])[0][:80]!r}  n_img_tok={len(ipos)}")
    for use_j in (True, False):
        ll, _ = readout.lens_logits(m, lens, b, layers=probe_layers,
                                    positions=ipos.tolist(), use_jacobian=use_j)
        tag = "J-lens" if use_j else "logit "
        for l in probe_layers:
            # most frequent top-1 token across image positions
            t1 = ll[l].argmax(-1)
            uniq, cnt = t1.unique(return_counts=True)
            order = cnt.argsort(descending=True)[:8]
            words = " ".join(f"{tok.decode([uniq[o]])!r}x{cnt[o].item()}" for o in order)
            print(f"  {tag} L{l:2d}  {words}")
    with torch.no_grad():
        g = m.hf.generate(**b, max_new_tokens=40, do_sample=False)
    print("  says:", tok.decode(g[0, b["input_ids"].shape[1]:], skip_special_tokens=True)[:200])
