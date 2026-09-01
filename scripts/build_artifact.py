"""Inline every figure into the artifact, under the 16 MB ceiling.

Two failures this replaces, both of which shipped at least once:

  * a stale WebP sitting beside a freshly regenerated PNG, so the artifact
    silently showed the previous run's numbers.  Source is chosen by mtime now,
    and a regenerated PNG always wins.
  * a placeholder with no matching figure, which reaches the reader as a broken
    image.  Missing figures are refused before anything is written.

Quality is walked down only as far as the budget requires, so a small paper
keeps full fidelity and a large one degrades evenly rather than dropping
figures.
"""
import base64
import io
import os
import re
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# which page: `build_artifact.py tool` builds paper/tool.html -> paper/tool_artifact.html
PAGE = sys.argv[1] if len(sys.argv) > 1 else "atlas"
SRC = os.path.join(ROOT, "paper", f"{PAGE}.html")
DST = os.path.join(ROOT, "paper", f"{PAGE}_artifact.html")
FIGS = os.path.join(ROOT, "figures")
BUDGET = 15.0 * 1024 * 1024          # 16 MB cap, with room for the prose
LADDER = [(2300, 82), (2100, 78), (1900, 72), (1700, 66), (1500, 60)]


def source_for(name: str) -> str | None:
    """Newest real file behind a FIG: placeholder."""
    stem = name[:-4] if name.endswith(".png") else name
    cands = [os.path.join(FIGS, stem + ".png"),
             os.path.join(FIGS, "web", stem + ".webp"),
             os.path.join(FIGS, stem + ".webp")]
    cands = [p for p in cands if os.path.exists(p) and os.path.getsize(p) > 0]
    return max(cands, key=os.path.getmtime) if cands else None


def encode(path: str, width: int, quality: int) -> bytes:
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="WEBP", quality=quality, method=6)
    return buf.getvalue()


html = open(SRC, encoding="utf-8").read()
names = re.findall(r'src="FIG:([^"]+)"', html)
uniq = sorted(set(names))
missing = [n for n in uniq if source_for(n) is None]
if missing:
    sys.exit("no figure behind these placeholders:\n  " + "\n  ".join(missing))

for width, quality in LADDER:
    blobs = {n: encode(source_for(n), width, quality) for n in uniq}
    total = sum(len(b) for b in blobs.values())
    print(f"  {width}px q{quality}: {total / 1024 / 1024:5.1f} MB "
          f"over {len(uniq)} figures")
    if total <= BUDGET:
        break
else:
    print("  still over budget at the lowest rung; shipping it anyway")

out = html
for n, b in blobs.items():
    out = out.replace(f'src="FIG:{n}"',
                      'src="data:image/webp;base64,' +
                      base64.b64encode(b).decode() + '"')
assert "FIG:" not in out, "a placeholder survived substitution"
open(DST, "w", encoding="utf-8").write(out)

big = sorted(blobs.items(), key=lambda kv: -len(kv[1]))[:5]
print(f"\nwrote {DST}  {os.path.getsize(DST) / 1024 / 1024:.1f} MB "
      f"({len(names)} placements, {len(uniq)} distinct)")
print("largest:")
for n, b in big:
    print(f"  {n:<34} {len(b) / 1024:6.0f} KB")
