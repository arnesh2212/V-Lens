from __future__ import annotations

import glob
import json
import os
import re

MODEL = "Qwen3-VL-8B-Instruct"
COLS = [("coco", "px_inpaint"), ("coco", "px_grey"), ("coco", "other"), ("coco", "attn"),
        ("ss", "px_inpaint"), ("ss", "other"), ("ss", "attn")]


def main(root="results/extended/b6"):
    rows = []
    for p in sorted(glob.glob(f"{root}/f*/summary_b1.json")):
        f = float(re.search(r"/f([0-9.]+)/", p).group(1))
        d = json.load(open(p)).get(MODEL) or {}
        for meth in ("v_write", "ig_black", "ig_blur", "random"):
            vals = {}
            for part, rep in COLS:
                e = (d.get(part) or {}).get(f"{rep}|{meth}")
                vals[f"{part}:{rep}"] = (e or {}).get("fall")
                if e and part == "coco":
                    vals["n_coco"] = e.get("n")
            loc = (d.get("coco") or {}).get(f"loc|{meth}") or {}
            vals["hit1"], vals["auroc"] = loc.get("hit1"), loc.get("auroc")
            if any(v is not None for k, v in vals.items() if k not in ("n_coco",)):
                name = f"V-Write @ depth {f:.2f}" if meth == "v_write" else (
                    f"pixel IG from {meth[3:]}" if meth.startswith("ig_") else f"random (run at {f:.2f})")
                rows.append(dict(name=name, depth=f, method=meth, **vals))
    fmt = lambda x: "–" if x is None else f"{x:.2f}"
    L = ["# B6: where to integrate (Qwen3-VL-8B)", "",
         "Fall in the answer margin with the top 20% of image tokens replaced; COCO present-object records and "
         "SimonSays agreement records. Pixel columns do not depend on the depth; 'other'/'attn' act at the "
         "ranking's own block.", "",
         "| ranking | COCO inpaint | COCO grey | COCO other | COCO attn | SS inpaint | SS other | SS attn | hit@1 | mask AUROC |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["method"] == "random" and r["depth"] != 0.5:
            continue
        L.append(f"| {r['name']} | " + " | ".join(fmt(r.get(f"{p}:{c}")) for p, c in COLS) +
                 f" | {fmt(r['hit1'])} | {fmt(r['auroc'])} |")
    os.makedirs(root, exist_ok=True)
    json.dump(rows, open(f"{root}/summary_b6.json", "w"), indent=1)
    open(f"{root}/summary_b6.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
