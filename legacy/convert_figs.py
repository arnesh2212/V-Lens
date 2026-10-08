"""Runs on the GPU box: PNG figures -> WebP, so the artifact can inline them.

A 220 dpi PNG is several megabytes; a dozen would blow the artifact size limit.
WebP at quality 88 is visually identical at about a fifth the size.
"""
import glob, os, sys
from PIL import Image

SRC = sys.argv[1] if len(sys.argv) > 1 else "figures"
DST = os.path.join(SRC, "web")
os.makedirs(DST, exist_ok=True)
MAXW = 2000
total = 0
for p in sorted(glob.glob(os.path.join(SRC, "*.png"))):
    im = Image.open(p).convert("RGB")
    if im.width > MAXW:
        im = im.resize((MAXW, round(im.height * MAXW / im.width)), Image.LANCZOS)
    out = os.path.join(DST, os.path.basename(p).replace(".png", ".webp"))
    im.save(out, format="WEBP", quality=88, method=6)
    total += os.path.getsize(out)
    print(f"{os.path.basename(out):40s} {os.path.getsize(out)/1024:7.0f} KB")
print(f"total {total/1024/1024:.1f} MB")
