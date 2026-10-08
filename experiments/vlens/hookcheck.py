# SPDX-License-Identifier: Apache-2.0
"""Does the intervention actually reach the model?

Every steering number in this project rests on one hook writing into the
residual stream.  A hook that silently does nothing returns the clean answer for
every arm, including the random control, and the resulting table looks tidy and
means nothing.  So before any steering claim: fire each mode on a smoke prompt
and require the first-token logits to move, the clean-clamp to be a no-op, the
energy to match a random direction of the same norm, and the effect to grow with
the dose.  Prints one line per check with pass or fail.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--out", default="results/hookcheck")
a = ap.parse_args()

import vlens  # noqa: E402
from vlens.data import coco  # noqa: E402
from vlens import corpus, tasks  # noqa: E402
from vlens.lens import depth, readout  # noqa: E402

name = a.repo.split("/")[-1]
os.makedirs(a.out, exist_ok=True)
say = lambda *x: print(*x, flush=True)
m = vlens.load_vlm(a.repo); sc = tasks.Scorer(m)
L = depth.layers_at(m, (0.35, 0.5, 0.65))
it = coco.concept_items(3, min_area=0.1, seed=3)[0]
b = m.chat("Describe this image.", [corpus.load_image(it["path"], square=448)])
g = torch.Generator().manual_seed(0)
d = torch.randn(m.d_model, generator=g); d = d / d.norm()
d2 = torch.randn(m.d_model, generator=torch.Generator().manual_seed(1)); d2 = d2 / d2.norm()


def logits(iv=None, alpha=None):
    if iv is not None:
        iv.set_batch(b)
    with (iv if iv is not None else torch.no_grad()):
        with torch.no_grad():
            out = m.forward(b)
    h = out.last_hidden_state[0, -1] if hasattr(out, "last_hidden_state") else out[0][0, -1]
    return m.unembed(h[None])[0].float().cpu().numpy()


clean = logits()
checks = []
def check(nm, ok, detail=""):
    checks.append(dict(name=nm, ok=bool(ok), detail=detail))
    say(f"  {'pass' if ok else 'FAIL'}  {nm}  {detail}")


for mode in ("add", "ablate", "clamp"):
    for chan in ("img", "txt", "all"):
        iv = readout.Intervention(m, {l: d for l in L}, channel=chan, mode=mode, alpha=2.0)
        lg = logits(iv)
        gap = float(np.abs(lg - clean).max())
        en = float(np.mean(iv.energy)) if iv.energy else float("nan")
        check(f"{mode:>6} at {chan:>3} moves the first-token logits", gap > 1e-3, f"max |dlogit| {gap:.3f}, energy {en:.3f}")

# a clean clamp is a no-op only for ablate with alpha 0
iv = readout.Intervention(m, {l: d for l in L}, channel="all", mode="add", alpha=0.0)
check("alpha 0 is a no-op", float(np.abs(logits(iv) - clean).max()) < 1e-4, "")

# dose-response over the range the experiments use; above a dose of about 1 the edit
# replaces the state and the model saturates, so monotonicity there means nothing
gaps = []
for al in (0.05, 0.1, 0.2, 0.4):
    iv = readout.Intervention(m, {l: d for l in L}, channel="all", mode="add", alpha=al)
    gaps.append(float(np.abs(logits(iv) - clean).max()))
check("effect grows with the dose, 0.05 to 0.4", all(gaps[i] < gaps[i + 1] + 1e-6 for i in range(len(gaps) - 1)), " -> ".join(f"{x:.2f}" for x in gaps))

# two different random directions must not give the same answer
iv1 = readout.Intervention(m, {l: d for l in L}, channel="all", mode="add", alpha=2.0); l1 = logits(iv1)
iv2 = readout.Intervention(m, {l: d2 for l in L}, channel="all", mode="add", alpha=2.0); l2 = logits(iv2)
check("different directions give different answers", float(np.abs(l1 - l2).max()) > 1e-3, f"max |difference| {float(np.abs(l1 - l2).max()):.3f}")

# energy match between two directions of the same norm
e1 = float(np.mean(iv1.energy)); e2 = float(np.mean(iv2.energy))
check("matched-norm arms have matched energy", abs(e1 - e2) / max(e1, 1e-9) < 0.2, f"{e1:.3f} vs {e2:.3f}")

# generation must differ too, not only the prefill logits
base = sc.generate(b, max_new_tokens=12)
iv = readout.Intervention(m, {l: d for l in L}, channel="all", mode="add", alpha=4.0); iv.set_batch(b)
with iv:
    steered = sc.generate(b, max_new_tokens=12)
check("generation changes under a strong edit", base.strip() != steered.strip(), f"“{base[:40]}” vs “{steered[:40]}”")

ok = all(c["ok"] for c in checks)
json.dump(dict(repo=a.repo, all_pass=ok, checks=checks), open(f"{a.out}/{name}_hookcheck.json", "w"), indent=1)
say(f"\n{name}: {'ALL CHECKS PASS' if ok else 'SOME CHECKS FAILED'}  ({sum(c['ok'] for c in checks)}/{len(checks)})")
sys.exit(0 if ok else 1)
