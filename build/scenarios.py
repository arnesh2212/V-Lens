# SPDX-License-Identifier: Apache-2.0
"""Emit the scenario gallery block for paper/tool.html from a scenarios JSON.

  python3 build/scenarios.py results/scenarios/<model>_scenarios.json > /tmp/block.html

Each scenario becomes one figure with a caption naming the question and the model's
own reply, so the page shows the tool being used rather than described.
"""
import argparse, html, json, os, sys

ap = argparse.ArgumentParser(); ap.add_argument("json"); ap.add_argument("--figdir", default="scenarios")
a = ap.parse_args()
d = json.load(open(a.json))
name = os.path.basename(a.json).replace("_scenarios.json", "")
NICE = {"Qwen3-VL-2B-Instruct": "Qwen3-VL-2B", "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B"}
out = []
out.append(f'<h2>The tool on six scenarios</h2>')
out.append('  <p>Each page below is one picture, one question and one forward pass on '
           f'{NICE.get(name, name)}. The strip along the top is what V-Read says the model is about to say at '
           'each depth, with the logit lens beneath it for contrast. The middle row is two or three scores read '
           'at every depth. The bottom row is the write over the image patches, which is where the answer is being '
           'pushed from. Nothing is averaged over items.</p>')
for s in d["scenarios"]:
    q = html.escape(s["question"][:110])
    rep = html.escape(s["reply"].replace("\n", " ")[:150])
    tag = html.escape(s["tag"])
    alt = html.escape(f"A scenario page for {s['tag']}: the picture and question, the word strip by depth for the read and the logit lens, score traces, and write maps over the image patches at four depths.")
    out.append(f'  <figure class="wide"><img src="FIG:{a.figdir}/{s["png"]}" alt="{alt}">'
               f'<figcaption><span class="num">Figure</span> {tag}. Question: “{q}”. The model answers: “{rep}…”</figcaption></figure>')
print("\n".join(out))
