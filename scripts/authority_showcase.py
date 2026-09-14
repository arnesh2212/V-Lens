# SPDX-License-Identifier: Apache-2.0
"""Four Authority scenes with the question, the reference and both models' verbatim replies."""
import json, os, re, textwrap
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get("VLENS_DATA", os.path.join(ROOT, "VLM V-Lens Final Dataset", "VLM V-Lens Final Dataset"))
OUT = os.path.join(ROOT, "figures", "vlens"); os.makedirs(OUT, exist_ok=True)
FONT = "/home/arnesg/.local/share/fonts/texgyre/texgyrepagella-regular.otf"
ITAL = "/home/arnesg/.local/share/fonts/texgyre/texgyrepagella-italic.otf"
F = ImageFont.truetype(FONT, 26); FI = ImageFont.truetype(ITAL, 26); FS = ImageFont.truetype(FONT, 22)
MODELS = [("Qwen3-VL-8B-Instruct", "Qwen3-VL-8B"), ("InternVL3-8B-hf", "InternVL3-8B")]
PICKS = [("i2p09", "station: the board says platform 2, the dated staff notice says 9, the hoodie says 5"),
         ("i9p08", "lab: the hazard label says store apart, the technician's coat says mixing is fine"),
         ("i8p09", "shop: the ingredients panel lists almonds, the handwritten card says nut free"),
         ("i1p10", "junction: the only STOP is inside an advertising billboard")]
clean = lambda s: re.sub(r"\s+", " ", re.sub(r"[*#>_]+", "", (s or "").replace(" \u2014 ", ": ").replace("\u2014", ","))).strip()


def rows_for(mk):
    for base in ("vlens_long", "vlens"):
        p = os.path.join(ROOT, "results", base, f"{mk}__gemini_auth.json")
        if os.path.exists(p):
            return {r["id"]: r for r in json.load(open(p))["rows"]}
    return {}


R = {mk: rows_for(mk) for mk, _ in MODELS}
IMW, TXW, PAD = 880, 1000, 28
panels = []
for pid, lab in PICKS:
    r = R[MODELS[0][0]].get(pid)
    if r is None:
        continue
    im = Image.open(os.path.join(D, r["image"])).convert("RGB"); im = im.resize((IMW, round(im.height * IMW / im.width)), Image.LANCZOS)
    lines = []
    for txt, font in ((lab, FI), ("", F), ("Question: " + clean(r["question"]), F), ("Reference: " + clean(str(r["expected"])), F), ("", F)):
        lines += [(l, font) for l in (textwrap.wrap(txt, 70) or [""])]
    for mk, nice in MODELS:
        rr = R[mk].get(pid) or {}
        j = (rr.get("judge") or {}).get("verdict") or ""
        lines += [(l, F) for l in textwrap.wrap(f"{nice} ({j}): " + clean(rr.get("reply"))[:520], 70)] + [("", FS)]
    th = sum(34 if f is not FS else 16 for _l, f in lines) + 2 * PAD
    H = max(im.height, th)
    canvas = Image.new("RGB", (IMW + TXW + 3 * PAD, H + PAD), "white")
    canvas.paste(im, (PAD, (H - im.height) // 2 + PAD // 2))
    d = ImageDraw.Draw(canvas); y = PAD
    for l, f in lines:
        d.text((IMW + 2 * PAD, y), l, fill="black", font=f); y += 34 if f is not FS else 16
    panels.append(canvas)
W = max(p.width for p in panels); Hs = sum(p.height for p in panels)
sheet = Image.new("RGB", (W, Hs), "white"); y = 0
for p in panels:
    sheet.paste(p, (0, y)); y += p.height
sheet.save(os.path.join(OUT, "showcase_authority.png")); print("wrote", os.path.join(OUT, "showcase_authority.png"), sheet.size)
