# SPDX-License-Identifier: Apache-2.0
"""Build the anonymous supplementary code package from this repository.

Copies the scripts behind the paper's core claims and every local module they
import, strips docstrings, bare strings and comments, replaces machine paths
with repository-relative defaults, adds supplement/README.md and
supplement/requirements.txt, checks that everything compiles, and zips it.

    python3 tools/build_supplement.py [--out FINAL/vlens_code]
"""
from __future__ import annotations

import argparse
import ast
import io
import os
import py_compile
import re
import shutil
import sys
import tokenize
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: the scripts behind the core claims; their local imports come along
ENTRIES = """
experiments/simonsays/behaviour.py tools/coco_subset.py tools/warm_fit.py tools/layer_readouts.py
tools/patchscope_readouts.py tools/latent_trace.py tools/fit_matched.py tools/score_matched.py
experiments/simonsays/patch_lenses.py experiments/simonsays/patch_causal.py tools/group_matched.py
experiments/simonsays/crossquestion.py experiments/simonsays/occlusion.py tools/retention.py
experiments/simonsays/sandbag_channels.py tools/behaviour_directions.py tools/cue_steer.py
experiments/safety/refusal.py tools/fetch_safety.py experiments/pwlock/finetune.py
experiments/pwlock/evaluate.py tools/pwlock_vwrite.py tools/pwlock_calib.py tools/pwlock_states.py
tools/make_vread_table.py tools/make_paper_tables.py tools/make_appendix_tables.py tools/paper_numbers.py
vlens/figures/fx_coco.py vlens/figures/fx_transitions.py vlens/figures/fx_pwlock_story.py
vlens/figures/fx_cuegeom.py vlens/figures/fx_dirspace.py vlens/figures/fx_evalaware.py
""".split()

#: (file, old, new): machine paths become repository-relative defaults
PATHS = [
    ("vlens/corpus.py", 'COCO_IMG = "/workspace/data/val2017"',
     'COCO_IMG = os.environ.get("VLENS_COCO_IMAGES", "data/coco/val2017")'),
    ("vlens/corpus.py", 'COCO_CAP = "/workspace/data/annotations/captions_val2017.json"',
     'COCO_CAP = os.environ.get("VLENS_COCO_CAPTIONS", "data/coco/annotations/captions_val2017.json")'),
    ("vlens/data/coco.py", 'ROOT = "/workspace/data"', 'ROOT = os.environ.get("VLENS_COCO_ROOT", "data/coco")'),
    ("vlens/data/coco.py", 'os.environ.get("VLENS_COCO", "/workspace/temp/vlm/data_root/coco")',
     'os.environ.get("VLENS_COCO", "data/coco_subset")'),
    ("vlens/figures/fx_coco.py", 'os.environ.get("VLENS_COCO", "/workspace/temp/vlm/data_root/coco")',
     'os.environ.get("VLENS_COCO", "data/coco_subset")'),
    ("vlens/figures/fx_coco.py", 'os.path.join(_REPO, "data_root", "coco")',
     'os.path.join(_REPO, "data", "coco_subset")'),
    ("vlens/data/hard.py", 'os.environ.get("JVLM_COCO_IMG", "/workspace/data/val2017")',
     'os.environ.get("VLENS_COCO_IMAGES", "data/coco/val2017")'),
    ("vlens/lens/fitall.py", 'os.environ.get("VLENS_MAP_CACHE", "/dev/shm/vlens_maps")',
     'os.environ.get("VLENS_MAP_CACHE", "cache/vlens_maps")'),
    ("tools/pwlock_states.py", 'default="/dev/shm/vlens_results/pwlock_states"', 'default="cache/pwlock_states"'),
    ("tools/fetch_safety.py", 'default="/dev/shm/vlens_safety"', 'default="data/safety"'),
    ("experiments/pwlock/finetune.py", 'default="/dev/shm/vlens_pw/adapter"', 'default="checkpoints/pwlock_adapter"'),
    ("vlens/figures/fx_transitions.py",
     '''        for root in ("VLM V-Lens Final Dataset Complete", os.path.join("VLM V-Lens Final Dataset", "VLM V-Lens Final Dataset")):
            q = os.path.join(ROOT, root, str(row["image"]))
            if os.path.exists(q):
                p = q
                break''',
     '''        from vlens.paths import DATA
        q = os.path.join(DATA, str(row["image"]))
        if os.path.exists(q):
            p = q'''),
]
RENAME_ENV = {"JVLM_SAFETY_ROOT": "VLENS_SAFETY_ROOT", "JVLM_HARD_ROOT": "VLENS_HARD_ROOT",
              "JVLM_EVALAWARE_ROOT": "VLENS_EVALAWARE_ROOT", "JVLM_COCO_IMG": "VLENS_COCO_IMAGES"}
PATHS_PY = '''import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
PAPER = os.path.join(ROOT, "paper")
ASSETS = os.path.join(ROOT, "assets")
DATA = os.environ.get("VLENS_DATA") or os.path.join(ROOT, "data", "simonsays")
'''
#: nothing in the package may still name a machine, a person or a host
FORBIDDEN = re.compile(r"arnesg|arnesh|/workspace|/dev/shm|iiitd|lexsi|217\.18\.|50\.46\.|Final Dataset", re.I)


def _imports(path):
    out = set()
    for n in ast.walk(ast.parse(open(path).read())):
        if isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            base = os.path.relpath(os.path.dirname(path), ROOT).split(os.sep)
            if n.level:
                base = base[:len(base) - (n.level - 1)] if n.level > 1 else base
                mod = ".".join(base + ([n.module] if n.module else []))
            else:
                mod = n.module
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in n.names)
        elif isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "import_module" \
                and n.args and isinstance(n.args[0], ast.Constant):
            out.add(n.args[0].value)
    return out


def _resolve(mod, cur):
    parts = mod.split(".")
    for c in (os.path.join(ROOT, *parts) + ".py", os.path.join(ROOT, *parts, "__init__.py"),
              os.path.join(os.path.dirname(cur), *parts) + ".py"):
        if os.path.exists(c):
            return c
    return None


def closure(entries):
    """Every repository file the entries import, directly or not (packages' __init__ included)."""
    seen, stack = set(), [os.path.join(ROOT, e) for e in entries]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        for m in _imports(p):
            r = _resolve(m, p)
            if r and r not in seen:
                stack.append(r)
            parts = m.split(".")
            for i in range(1, len(parts)):
                ini = os.path.join(ROOT, *parts[:i], "__init__.py")
                if os.path.exists(ini) and ini not in seen:
                    stack.append(ini)
    return sorted(os.path.relpath(p, ROOT) for p in seen)


def _string_statements(src):
    """(first line, last line, column, body is only strings) of every bare string statement."""
    spans = []
    for node in ast.walk(ast.parse(src)):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if not isinstance(body, list) or not body or not isinstance(body[0], ast.stmt):
                continue
            strs = [d for d in body if isinstance(d, ast.Expr) and isinstance(d.value, ast.Constant)
                    and isinstance(d.value.value, str)]
            for j, d in enumerate(strs):
                spans.append((d.lineno, d.end_lineno, d.col_offset, len(strs) == len(body) and j == 0))
    return spans


def strip(src):
    """Source without docstrings, bare strings or comments; a body left empty gets ``pass``."""
    lines = src.splitlines(keepends=True)
    drop, fill = set(), {}
    for lo, hi, col, only in _string_statements(src):
        if lines[lo - 1][:col].strip():
            continue
        drop.update(range(lo, hi + 1))
        if only:
            fill[lo] = " " * col + "pass\n"
    src = "".join(fill.get(i, "") + ("" if i in drop else ln) for i, ln in enumerate(lines, 1))
    cut ={t.start[0]: t.start[1] for t in tokenize.generate_tokens(io.StringIO(src).readline)
           if t.type == tokenize.COMMENT}
    out = []
    for i, ln in enumerate(src.splitlines(keepends=True), 1):
        if i in cut:
            head = ln[:cut[i]].rstrip()
            if head:
                out.append(head + "\n")
            continue
        out.append(ln)
    src = re.sub(r"\n{4,}", "\n\n\n", "".join(out)).lstrip("\n")
    ast.parse(src)
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "FINAL", "vlens_code"))
    a = ap.parse_args()
    files = closure(ENTRIES)
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    for f in files:
        dst = os.path.join(a.out, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(os.path.join(ROOT, f), dst)
    for f, old, new in PATHS:
        p = os.path.join(a.out, f)
        s = open(p).read()
        if old not in s:
            sys.exit(f"{f}: expected text not found, update PATHS")
        open(p, "w").write(s.replace(old, new))
    open(os.path.join(a.out, "vlens", "paths.py"), "w").write(PATHS_PY)
    for f in files:
        p = os.path.join(a.out, f)
        s = open(p).read()
        for old, new in RENAME_ENV.items():
            s = s.replace(old, new)
        s = strip(s)
        if FORBIDDEN.search(s):
            sys.exit(f"{f}: still names a machine or a person: {FORBIDDEN.search(s).group(0)}")
        open(p, "w").write(s)
        py_compile.compile(p, doraise=True)
    for f in ("README.md", "requirements.txt"):
        shutil.copy(os.path.join(ROOT, "supplement", f), os.path.join(a.out, f))
    for d, _, _ in os.walk(a.out):
        if d.endswith("__pycache__"):
            shutil.rmtree(d, ignore_errors=True)
    zp = a.out.rstrip("/") + ".zip"
    base = os.path.dirname(a.out.rstrip("/"))
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for d, _, fs in os.walk(a.out):
            for f in sorted(fs):
                p = os.path.join(d, f)
                z.write(p, os.path.relpath(p, base))
    print(f"{len(files)} modules -> {a.out}  and  {zp}")


if __name__ == "__main__":
    main()
