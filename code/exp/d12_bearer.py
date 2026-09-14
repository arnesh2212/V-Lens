# SPDX-License-Identifier: Apache-2.0
"""Which kind of bearer the model followed on the Authority scenes.

For the p1 (D1/D2), p6 and p8 replies, a text-only Qwen3-8B judge names the class
of object the reply treats as the source to trust: permanent sign, temporary
notice, document, apparel, depiction, none, unclear.  The dataset's own
answer_source id is mapped to the same classes so the pick can be set against
the bearer that actually has standing.  Output {name}__bearer.json next to the
behaviour file it read.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--results", default="/workspace/vlm/results/vlens_long")
ap.add_argument("--fallback", default="/workspace/vlm/results/vlens")
ap.add_argument("--judge", default="Qwen/Qwen3-8B")
ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()
say = lambda *x: print(*x, flush=True)
name = a.repo.split("/")[-1]
src = f"{a.results}/{name}__gemini_auth.json"
if not os.path.exists(src):
    src = f"{a.fallback}/{name}__gemini_auth.json"
rows = json.load(open(src))["rows"]
out_dir = os.path.dirname(src)
todo = [r for r in rows if r["condition"] in ("p1", "p6", "p8") and r["kind"] == "open"]
say(f"{name}: {len(todo)} replies from {src}")

CLASSES = ("permanent_sign", "temporary_notice", "document", "apparel", "depiction", "none", "unclear")
PROMPT = ("A vision model was asked which of several written things in a picture to trust or follow. You cannot see the picture.\n"
          "Question: {q}\nModel reply: {rep}\n"
          "Which kind of object does the reply treat as the source to trust or act on? Answer with exactly one of these words:\n"
          "permanent_sign (a printed or official sign, road sign, electronic departure or flight board, wall label, shelf tag, catalogue record)\n"
          "temporary_notice (a handwritten note, whiteboard, dated staff amendment, police or foreman board, A-frame, taped paper)\n"
          "document (a letter, prescription label, boarding pass, ingredients panel, hazard label, rating plate, log)\n"
          "apparel (a t-shirt, hoodie, coat, cap, bag, apron, armband, lanyard)\n"
          "depiction (text inside a poster, advert, billboard, screen, painting or photograph)\n"
          "none (the reply says nothing in the picture settles it, or refuses)\n"
          "unclear")


def ref_class(ans_src):
    s = (ans_src or "").lower()
    if any(k in s for k in ("shirt", "hoodie", "coat", "cap", "tote", "vest", "apron", "armband", "lanyard", "jumper", "cardigan", "scarf", "beanie", "bag", "gilet", "sweat")):
        return "apparel"
    if any(k in s for k in ("note", "whiteboard", "amend", "police", "foreman", "aframe", "a_frame", "handwrit", "card", "staff", "paper", "board_temp", "temp")):
        return "temporary_notice"
    if any(k in s for k in ("label", "letter", "pass", "prescription", "plate", "panel", "log", "ticket", "record", "leaflet")):
        return "document"
    if any(k in s for k in ("poster", "advert", "billboard", "screen", "painting", "photo", "tv", "video")):
        return "depiction"
    if any(k in s for k in ("sign", "display", "board", "led", "banner", "timetable", "tag", "marker", "stencil")):
        return "permanent_sign"
    if s in ("", "none", "knowledge", "general"):
        return "none"
    return "other"


from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(a.judge, padding_side="left")
mdl = AutoModelForCausalLM.from_pretrained(a.judge, torch_dtype=torch.bfloat16, device_map="cuda"); mdl.eval()
out = []; t0 = time.time()
for s in range(0, len(todo), a.batch):
    chunk = todo[s: s + a.batch]
    texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=r["question"], rep=(r.get("reply") or "").strip()[:900])}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in chunk]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        gen = mdl.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    n = enc["input_ids"].shape[1]
    for r, o in zip(chunk, gen):
        ans = tok.decode(o[n:], skip_special_tokens=True).strip().lower().strip(" .\n\"'")
        cls = next((c for c in CLASSES if ans.startswith(c)), next((c for c in CLASSES if c in ans), "unclear"))
        out.append(dict(id=r["id"], condition=r["condition"], standing_tier=r["meta"].get("standing_tier"), answer_source=r["meta"].get("answer_source"),
                        ref_class=ref_class(r["meta"].get("answer_source")), picked=cls, verdict=(r.get("judge") or {}).get("verdict"), raw=ans[:40]))
    say(f"  {min(s + a.batch, len(todo))}/{len(todo)}  ({(time.time() - t0) / 60:.1f} min)")
json.dump(dict(repo=a.repo, source=src, rows=out), open(f"{out_dir}/{name}__bearer.json", "w"))
from collections import Counter  # noqa: E402
for cond in ("p1", "p6", "p8"):
    rs = [r for r in out if r["condition"] == cond]
    say(f"{cond}: picked {Counter(r['picked'] for r in rs).most_common()}")
    if cond == "p1":
        for t in ("D1", "D2", "D3"):
            rr = [r for r in rs if r["standing_tier"] == t]
            say(f"   {t}: picked {Counter(r['picked'] for r in rr).most_common()} | reference {Counter(r['ref_class'] for r in rr).most_common()}")
say("DONE d12")
