# SPDX-License-Identifier: Apache-2.0
"""Example panels of the custom datasets for the page: what an item looks like."""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get("VLENS_DATA", os.path.join(ROOT, "VLM V-Lens Final Dataset", "VLM V-Lens Final Dataset"))
OUT = os.path.join(ROOT, "figures", "vlens"); os.makedirs(OUT, exist_ok=True)
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 20)
except Exception:  # noqa: BLE001
    FONT = ImageFont.load_default()

def tile(paths, labels, out, w=640, h=360):
    ims = []
    for p, lab in zip(paths, labels):
        im = Image.open(p).convert("RGB")
        im.thumbnail((w, h)); canvas = Image.new("RGB", (w, h + 40), "white"); canvas.paste(im, ((w - im.width) // 2, 40))
        ImageDraw.Draw(canvas).text((10, 8), lab, fill="black", font=FONT); ims.append(canvas)
    n = len(ims); cols = min(n, 3); rows = (n + cols - 1) // cols
    W = Image.new("RGB", (cols * w, rows * (h + 40)), "white")
    for i, im in enumerate(ims):
        W.paste(im, ((i % cols) * w, (i // cols) * (h + 40)))
    W.save(out); print("wrote", out)

G = os.path.join(D, "Gemini Dataset"); N = os.path.join(D, "Nuscene Dataset"); O = os.path.join(D, "OpenImages Dataset")
tile([f"{G}/Original Images/i1.png", f"{G}/Edited Images/i1_swap.png", f"{G}/Edited Images/i1_removed.png"],
     ["i1: eight text bearers, one with standing", "i1_swap: police board and permanent sign exchange text", "i1_removed: the board's text erased"],
     f"{OUT}/panel_gemini_authority.png")
tile([f"{G}/Original Images/s1_V0.png", f"{G}/Edited Images/s1_V1.png", f"{G}/Edited Images/s1_V3.png"],
     ["s1 V0: candid photo", "s1 V1: figure chrome and EVAL SET watermark", "s1 V3: multiple-choice item margin"],
     f"{OUT}/panel_gemini_eval.png")
e = {x["edit_number"]: x for x in json.load(open(f"{N}/Edit Instructions.json"))["edits"]}
src = f"{N}/{e[1]['source_image']}"
tile([src, f"{N}/Edited Images/1.jpg", f"{N}/Edited Images/2.jpg", f"{N}/Edited Images/3.jpg", f"{N}/Edited Images/4.jpg", f"{N}/Edited Images/21.jpg"],
     ["nuScenes frame, untouched", "conflict_main: official STOP vs cardboard GO AHEAD", "conflict_swapped: carriers exchanged",
      "both_official: two official signs disagree", "agree_control: second sign agrees", "downgrade_informal: same words on cardboard"],
     f"{OUT}/panel_nuscenes_edits.png")
tile([f"{N}/Edited Images/31.jpg", f"{N}/Edited Images/33.jpg", f"{N}/Edited Images/35.jpg", f"{N}/Edited Images/32.jpg", f"{N}/Edited Images/34.jpg", f"{N}/Edited Images/36.jpg"],
     ["eval_watermark", "eval_bbox", "eval: annotation UI / grid", "deploy_dashcam_hud", "deploy_phone_screenshot", "deploy_windshield"],
     f"{OUT}/panel_nuscenes_eval.png")
idx = {im["image_number"]: im for im in json.load(open(f"{O}/Image Index.json"))["images"]}
picks = [2] + [k for k, v in idx.items() if v.get("sign") and v["sign"].get("kind") == "novelty"][:2] + [k for k, v in idx.items() if v.get("group") == "authority_figure"][:1] + [k for k, v in idx.items() if v.get("group") == "informal_text"][:1] + [k for k, v in idx.items() if v.get("group") == "lab_bio"][:1]
labs = []
for k in picks:
    v = idx[k]; sg = v.get("sign") or {}
    labs.append(f"{v['group']}: {sg.get('text', '')}{(' + ' + sg['added_text']) if sg.get('added_text') else ''}"[:60])
tile([f"{O}/{idx[k]['file']}" for k in picks], labs, f"{OUT}/panel_openimages.png")
