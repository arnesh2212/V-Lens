from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAMS = ["openimages_orig_auth", "openimages_edit_auth", "openimages_orig_eval", "nuscenes_edit_auth",
        "vins_orig_auth", "vins_edit_auth", "gemini_orig_auth", "wmdp_bio_sand", "wmdp_chem_sand", "wmdp_cybe_sand"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    ap.add_argument("--a5", default="results/extended/a5")
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--skip-read", action="store_true")
    ap.add_argument("--drop-maps", action="store_true", help="delete every variant's maps once read (A7 storage)")
    a = ap.parse_args()
    os.chdir(ROOT)
    variants = sorted(os.path.basename(d) for d in glob.glob(f"{a.a5}/maps/*") if glob.glob(f"{d}/*.pt.spec"))
    print("variants:", variants, flush=True)
    summary = {}
    for v in variants:
        rd = f"{a.a5}/readouts/{v}"
        if not a.skip_read or not os.path.isdir(rd):
            os.makedirs(rd, exist_ok=True)
            subprocess.run([sys.executable, "tools/layer_readouts.py", "read", "--model", a.model,
                            "--results", "results/simonsays", "--out", rd, "--maps", f"{a.a5}/maps/{v}",
                            "--head", f"results/readouts/{a.model}__head.npz", "--families", *FAMS], check=True)
        root = f"{a.a5}/root_{v}"
        os.makedirs(f"{root}/readouts", exist_ok=True)
        if not os.path.exists(f"{root}/simonsays"):
            os.symlink(os.path.abspath("results/simonsays"), f"{root}/simonsays")
        for fam in FAMS:
            src = f"results/readouts/{a.model}__{fam}_readouts.json"
            mine = f"{rd}/{a.model}__{fam}_readouts.json"
            if not (os.path.exists(src) and os.path.exists(mine)):
                continue
            d = json.load(open(src)); e = json.load(open(mine))
            ev = {r["id"]: r.get("v_read") for r in e["rows"]}
            for r in d["rows"]:
                if r["id"] in ev and ev[r["id"]] is not None:
                    r["v_read"] = ev[r["id"]]
                else:
                    r["v_read"] = None
            json.dump(d, open(f"{root}/readouts/{a.model}__{fam}_readouts.json", "w"))
            ps = f"results/readouts/{a.model}__{fam}_patchscope.json"
            if os.path.exists(ps):
                shutil.copy(ps, f"{root}/readouts/")
        hd = f"results/readouts/{a.model}__head.npz"
        if os.path.exists(hd) and not os.path.exists(f"{root}/readouts/{os.path.basename(hd)}"):
            shutil.copy(hd, f"{root}/readouts/")
        out = f"{a.a5}/stats_{v}"
        subprocess.run([sys.executable, "experiments/extended/vread_stats.py", "--models", a.model, "--results", root,
                        "--out", out, "--reps", str(a.reps)], check=True)
        if v != "v_read" or a.drop_maps:
            for f in glob.glob(f"{a.a5}/maps/{v}/*.pt"):
                os.remove(f)
        js = sorted(glob.glob(f"{out}/*.json"))
        summary[v] = js[0] if js else None
    json.dump(summary, open(f"{a.a5}/a5_index.json", "w"), indent=1)
    print("DONE a5_finish", summary, flush=True)


if __name__ == "__main__":
    main()
