"""Rebuild the data tables in paper/tool.html from results/*.json.

Every table this script owns lives between a pair of HTML comments,
    <!--TABLE:name--> ... <!--/TABLE:name-->
already present in paper/tool.html. Running this script replaces exactly
the text between each such pair with freshly generated HTML built straight
from the stored JSON in results/ -- nothing is ever hand-typed or carried
over from a previous run. Everything outside the marker pairs (prose,
headings, figures) is left byte-for-byte untouched.

Run it, and it prints one line per table naming every cell whose value
changed since the file's last build, plus a best-effort list of prose
sentences nearby that quote a number worth rechecking by hand. Run it
twice in a row with no new results and the second run reports no changes
and rewrites the file to the same bytes -- that idempotence is the point:
a hand-transcribed table can drift from the logs, a generated one cannot.

If a table's source JSON is missing (a model never run, a file not yet
produced) the table is skipped and left as it was, and that is reported
too rather than silently papered over.
"""
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
HTML_PATH = os.path.join(ROOT, "paper", "tool.html")

MODELS = [
    ("Qwen3-VL-2B-Instruct", "Qwen3-VL-2B"),
    ("Qwen3-VL-8B-Instruct", "Qwen3-VL-8B"),
]

#: the decision table has a third architecture; the others do not yet
MODELS_DECISION = MODELS + [("InternVL3-8B-hf", "InternVL3-8B")]

MINUS = "−"


# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(x, decimals=2):
    """Format a number the way the page's tables do: fixed decimals, a real
    minus sign, empty string (never a fabricated number) for anything
    missing."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = f"{x:.{decimals}f}"
    if s.startswith("-"):
        rest = s[1:]
        if float(rest) == 0.0:
            return rest
        return MINUS + rest
    return s


def nearest_key(depth_map, target):
    """depth_map: {key: depth_float}. Return the key whose depth is nearest
    target."""
    if not depth_map:
        return None
    return min(depth_map, key=lambda k: abs(depth_map[k] - target))


def nearest_float(values, target):
    values = list(values)
    if not values:
        return None
    return min(values, key=lambda v: abs(v - target))


def build_block(thead, rows, groups, decimals):
    """rows: list of (row_label, [(value_or_None, is_labelled), ...]).
    groups: list of index-lists; within each group the largest label-free
    (not-labelled) value gets wrapped in <span class="k">."""
    lines = ['  <div class="tblwrap">', "  <table>", thead, "    <tbody>"]
    for row_label, values in rows:
        marked = set()
        for group in groups:
            cands = [(i, values[i][0]) for i in group
                      if values[i][0] is not None and not values[i][1]]
            if cands:
                marked.add(max(cands, key=lambda t: t[1])[0])
        tds = [f"<td>{row_label}</td>"]
        for i, (val, _lab) in enumerate(values):
            text = fmt(val, decimals)
            if i in marked and text != "":
                text = f'<span class="k">{text}</span>'
            tds.append(f"<td>{text}</td>")
        lines.append("      <tr>" + "".join(tds) + "</tr>")
    lines += ["    </tbody>", "  </table>", "  </div>"]
    return "\n".join(lines)


def build_block_colmarked(thead, rows, label_free_row_idxs, decimals):
    """Like build_block, but the table is transposed: rows are the
    competing instruments and columns are the fixed condition (e.g. dose).
    The best label-free row is marked per column instead of per row."""
    ncols = len(rows[0][1]) if rows else 0
    marks = {}
    for c in range(ncols):
        cands = [(r, rows[r][1][c]) for r in label_free_row_idxs
                  if rows[r][1][c] is not None]
        if cands:
            marks[c] = max(cands, key=lambda t: t[1])[0]
    lines = ['  <div class="tblwrap">', "  <table>", thead, "    <tbody>"]
    for r, (row_label, values) in enumerate(rows):
        tds = [f"<td>{row_label}</td>"]
        for c, val in enumerate(values):
            text = fmt(val, decimals)
            if marks.get(c) == r and text != "":
                text = f'<span class="k">{text}</span>'
            tds.append(f"<td>{text}</td>")
        lines.append("      <tr>" + "".join(tds) + "</tr>")
    lines += ["    </tbody>", "  </table>", "  </div>"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# per-table builders. Each returns None if its source JSON is entirely
# missing (table left untouched), else a dict with the rendered html, the
# flat column labels (for the change report) and any missing-value notes.
# --------------------------------------------------------------------------

def build_readdecision():
    path = os.path.join(RESULTS, "gate", "reader_regression.json")
    d = load_json(path)
    if d is None:
        return None
    targets = [(0.2, "0.2"), (0.35, "0.35"), (0.48, "0.48"),
               (0.65, "0.65"), (0.8, "0.8"), (0.95, "0.95")]
    notes = []
    rows = []
    models = [mk for mk in MODELS_DECISION if d.get(mk[0]) is not None]
    for target, label in targets:
        values = []
        for model_key, _short in models:
            mdata = d.get(model_key)
            if mdata is None:
                notes.append(f"model {model_key} missing from {path}")
                values += [(None, False), (None, False), (None, True)]
                continue
            depth_map = {L: v["frac"] for L, v in mdata.items()}
            layer = nearest_key(depth_map, target)
            entry = mdata[layer]
            values.append((entry.get("ridge"), False))
            values.append((entry.get("identity_perfold"), False))
            values.append((entry.get("probe"), True))
        rows.append((label, values))
    thead = ('    <thead><tr><th rowspan="2">depth</th>'
             + "".join(f'<th colspan="3">{short}</th>' for _, short in models)
             + '</tr>\n    <tr>'
             + "".join("<th>read</th><th>raw</th><th>probe</th>" for _ in models)
             + "</tr></thead>")
    groups = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(len(models))]
    col_labels = [f"{short} {c}" for _, short in models for c in ("read", "raw", "probe")]
    html = build_block(thead, rows, groups, decimals=3)
    return {"html": html, "col_labels": col_labels, "notes": notes}


def _presence_like(field, targets, notes_prefix):
    per_model = {}
    any_present = False
    notes = []
    for model_key, _short in MODELS:
        path = os.path.join(RESULTS, "regions_hard", f"{model_key}_regions.json")
        d = load_json(path)
        if d is None:
            notes.append(f"missing {path}")
            per_model[model_key] = None
            continue
        any_present = True
        per_model[model_key] = d
    if not any_present:
        return None, notes
    rows = []
    for target, label in targets:
        values = []
        for model_key, _short in MODELS:
            d = per_model[model_key]
            if d is None:
                values += [(None, False)] * 4
                continue
            depth_of = {k: float(v) for k, v in d["emergence"]["depth_of"].items()}
            layer = nearest_key(depth_of, target)
            scores = d["emergence"][field]
            def get(instr):
                return scores.get(instr, {}).get(layer)
            values.append((get("reader"), False))
            values.append((get("logit lens"), False))
            values.append((get("write"), False))
            values.append((get("probe"), True))
        rows.append((label, values))
    return rows, notes


def build_presence():
    targets = [(0.13, "0.13"), (0.22, "0.22"), (0.35, "0.35"),
               (0.48, "0.48"), (0.65, "0.65")]
    rows, notes = _presence_like("auroc", targets, "presence")
    if rows is None:
        return None
    thead = ('    <thead><tr><th rowspan="2">depth</th>'
              '<th colspan="4">Qwen3-VL-2B</th>'
              '<th colspan="4">Qwen3-VL-8B</th></tr>\n'
              "    <tr><th>the read</th><th>logit lens</th><th>the write</th><th>probe</th>"
              "<th>the read</th><th>logit lens</th><th>the write</th><th>probe</th></tr></thead>")
    groups = [[0, 1, 2], [4, 5, 6]]
    col_labels = ["2B read", "2B logit lens", "2B write", "2B probe",
                  "8B read", "8B logit lens", "8B write", "8B probe"]
    html = build_block(thead, rows, groups, decimals=2)
    return {"html": html, "col_labels": col_labels, "notes": notes}


def build_hallucination():
    targets = [(0.13, "0.13"), (0.22, "0.22"), (0.35, "0.35"),
               (0.48, "0.48"), (0.65, "0.65"), (0.96, "0.96")]
    rows, notes = _presence_like("hallucination_auroc", targets, "hallucination")
    if rows is None:
        return None
    notes = list(notes)
    notes.append("table now shows 'the write' alongside read/logit lens/probe, "
                  "matching the instrument set the task specifies; the previous "
                  "hand-built table omitted a write column")
    thead = ('    <thead><tr><th rowspan="2">depth</th>'
              '<th colspan="4">Qwen3-VL-2B</th>'
              '<th colspan="4">Qwen3-VL-8B</th></tr>\n'
              "    <tr><th>the read</th><th>logit lens</th><th>the write</th><th>probe</th>"
              "<th>the read</th><th>logit lens</th><th>the write</th><th>probe</th></tr></thead>")
    groups = [[0, 1, 2], [4, 5, 6]]
    col_labels = ["2B read", "2B logit lens", "2B write", "2B probe",
                  "8B read", "8B logit lens", "8B write", "8B probe"]
    html = build_block(thead, rows, groups, decimals=2)
    return {"html": html, "col_labels": col_labels, "notes": notes}


def build_regionnaming():
    targets = [(0.2, "0.2"), (0.35, "0.35"), (0.48, "0.48"),
               (0.65, "0.65"), (0.8, "0.8"), (0.95, "0.95")]
    notes = []
    per_model = {}
    any_present = False
    for model_key, _short in MODELS:
        path = os.path.join(RESULTS, "regions_hard", f"{model_key}_regions.json")
        d = load_json(path)
        if d is None:
            notes.append(f"missing {path}")
            per_model[model_key] = None
            continue
        any_present = True
        per_model[model_key] = d
    if not any_present:
        return None
    rows = []
    for target, label in targets:
        values = []
        for model_key, _short in MODELS:
            d = per_model[model_key]
            if d is None:
                values += [(None, False)] * 3
                continue
            depth_map = {k: float(v) for k, v in d["depths"].items()}
            layer = nearest_key(depth_map, target)
            acc = d["regions"]["accuracy"]
            values.append((acc.get("write", {}).get(layer), False))
            values.append((acc.get("logit lens", {}).get(layer), False))
            values.append((acc.get("logit lens, centred", {}).get(layer), False))
        rows.append((label, values))
    thead = ('    <thead><tr><th rowspan="2">depth</th>'
              '<th colspan="3">Qwen3-VL-2B</th>'
              '<th colspan="3">Qwen3-VL-8B</th></tr>\n'
              "    <tr><th>the write</th><th>logit lens</th><th>logit lens, centred</th>"
              "<th>the write</th><th>logit lens</th><th>logit lens, centred</th></tr></thead>")
    groups = [[0, 1, 2], [3, 4, 5]]
    col_labels = ["2B write", "2B logit lens", "2B logit lens centred",
                  "8B write", "8B logit lens", "8B logit lens centred"]
    html = build_block(thead, rows, groups, decimals=2)
    return {"html": html, "col_labels": col_labels, "notes": notes}


def build_twostep():
    targets = [(0.35, "0.35"), (0.48, "0.48"), (0.58, "0.58"),
               (0.65, "0.65"), (0.70, "0.70"), (0.78, "0.78")]
    notes = []
    per_model = {}
    any_present = False
    for model_key, _short in MODELS:
        path = os.path.join(RESULTS, "twostep", f"{model_key}_twostep.json")
        d = load_json(path)
        if d is None:
            notes.append(f"missing {path}")
            per_model[model_key] = None
            continue
        any_present = True
        per_model[model_key] = d
    if not any_present:
        return None
    rows = []
    for target, label in targets:
        values = []
        for model_key, _short in MODELS:
            d = per_model[model_key]
            if d is None:
                values += [(None, False)] * 3
                continue
            ac = d["answer_curves"]
            depth_map = {k: float(k) for k in ac["read"].keys()}
            key = nearest_key(depth_map, target)
            read_entry = ac["read"].get(key, {})
            ll_key = nearest_key({k: float(k) for k in ac["logit lens"].keys()}, target)
            ll_entry = ac["logit lens"].get(ll_key, {})
            values.append((read_entry.get("main_answer"), False))
            values.append((ll_entry.get("main_answer"), False))
            values.append((read_entry.get("control_answer"), False))
        rows.append((label, values))
    thead = ('    <thead><tr><th rowspan="2">depth</th>'
              '<th colspan="3">Qwen3-VL-2B</th>'
              '<th colspan="3">Qwen3-VL-8B</th></tr>\n'
              "    <tr><th>two-step, the read</th><th>two-step, logit lens</th><th>one-step, the read</th>"
              "<th>two-step, the read</th><th>two-step, logit lens</th><th>one-step, the read</th></tr></thead>")
    groups = [[0, 1, 2], [3, 4, 5]]
    col_labels = ["2B two-step read", "2B two-step logit lens", "2B one-step read",
                  "8B two-step read", "8B two-step logit lens", "8B one-step read"]
    html = build_block(thead, rows, groups, decimals=2)
    return {"html": html, "col_labels": col_labels, "notes": notes}


def build_twostepregions():
    targets = [(0.2, "0.2"), (0.35, "0.35"), (0.48, "0.48"),
               (0.65, "0.65"), (0.8, "0.8")]
    notes = []
    per_model = {}
    any_present = False
    for model_key, _short in MODELS:
        path = os.path.join(RESULTS, "twostep", f"{model_key}_twostep.json")
        d = load_json(path)
        if d is None:
            notes.append(f"missing {path}")
            per_model[model_key] = None
            continue
        any_present = True
        per_model[model_key] = d
    if not any_present:
        return None
    notes.append("the JSON also carries a 'distractor' share/covering value at "
                 "every depth; this table, like the original, shows only "
                 "landmark, referent and floor")
    rows = []
    for target, label in targets:
        values = []
        cost = []
        for model_key, _short in MODELS:
            d = per_model[model_key]
            if d is None:
                values += [(None, False)] * 3
                cost.append((None, False))
                continue
            ws = d["write_share"]
            cov = d["covering"]
            depth_map = {k: float(k) for k in ws.keys()}
            key = nearest_key(depth_map, target)
            share = ws.get(key, {})
            values.append((share.get("landmark"), False))
            values.append((share.get("referent"), False))
            values.append((share.get("floor"), False))
            cov_key = nearest_key({k: float(k) for k in cov.keys()}, target)
            cost.append((cov.get(cov_key, {}).get("landmark"), False))
        rows.append((label, values + cost))
    thead = ('    <thead><tr><th rowspan="2">depth</th>'
              '<th colspan="3">Qwen3-VL-2B, share of the write over chance</th>'
              '<th colspan="3">Qwen3-VL-8B, share of the write over chance</th>'
              '<th colspan="2">cost of covering the landmark</th></tr>\n'
              "    <tr><th>landmark</th><th>object asked about</th><th>bare floor</th>"
              "<th>landmark</th><th>object asked about</th><th>bare floor</th>"
              "<th>2B</th><th>8B</th></tr></thead>")
    groups = [[0, 1, 2], [3, 4, 5]]
    col_labels = ["2B landmark", "2B referent", "2B floor",
                  "8B landmark", "8B referent", "8B floor",
                  "cost 2B", "cost 8B"]
    html = build_block(thead, rows, groups, decimals=2)
    return {"html": html, "col_labels": col_labels, "notes": notes}


def build_atlas():
    path_2b = os.path.join(RESULTS, "atlas", "Qwen3-VL-2B-Instruct_atlas.json")
    path_8b = os.path.join(RESULTS, "atlas", "Qwen3-VL-8B-Instruct_atlas.json")
    d = load_json(path_2b)
    notes = []
    if load_json(path_8b) is None:
        notes.append(f"missing {path_8b}; table stays Qwen3-VL-2B only, "
                     "matching the original table's scope")
    if d is None:
        notes.append(f"missing {path_2b}")
        return None
    targets = [(0.22, "0.22"), (0.35, "0.35"), (0.48, "0.48"),
               (0.65, "0.65"), (0.78, "0.78"), (0.96, "0.96")]
    depth_map = {k: float(v) for k, v in d["depths"].items()}
    ref = d["summary"]["refusal"]
    instruments = ["write, path", "write, one point", "write, averaged",
                   "identity", "grad x act"]
    rows = []
    for target, label in targets:
        layer = nearest_key(depth_map, target)
        values = []
        for instr in instruments:
            entry = ref.get(instr, {}).get("rho_resample", {}).get(layer)
            values.append((entry[0] if entry else None, False))
        glimpse = ref.get("GLIMPSE", {}).get("rho_abs", {}).get(layer)
        values.append((glimpse[0] if glimpse else None, False))
        rows.append((label, values))
    thead = ('    <thead><tr><th rowspan="2">depth</th>'
              '<th colspan="6">refusal score, typography items</th></tr>\n'
              "    <tr><th>write, path</th><th>write, one point</th><th>write, averaged</th>"
              "<th>logit lens</th><th>grad × act</th><th>GLIMPSE</th></tr></thead>")
    groups = [[0, 1, 2, 3, 4, 5]]
    col_labels = ["write path", "write one point", "write averaged",
                  "logit lens", "grad x act", "GLIMPSE"]
    html = build_block(thead, rows, groups, decimals=2)
    return {"html": html, "col_labels": col_labels, "notes": notes}


def build_writeinto():
    targets = [(0.2, "0.2"), (0.35, "0.35"), (0.48, "0.48"),
               (0.65, "0.65"), (0.8, "0.8")]
    notes = []
    per_model = {}
    any_present = False
    for model_key, _short in MODELS:
        path = os.path.join(RESULTS, "twostep_write", f"{model_key}_twostep.json")
        d = load_json(path)
        if d is None:
            notes.append(f"missing {path}")
            per_model[model_key] = None
            continue
        any_present = True
        per_model[model_key] = d["swap"]
    if not any_present:
        return None

    def rate(swap, frac, direction, channel):
        best = None
        for alpha in (0.2, 0.4, 0.8):
            items = [r["said_tgt_add"] for r in swap
                     if abs(r["frac"] - frac) < 1e-9
                     and r["direction"] == direction
                     and r["channel"] == channel
                     and r["alpha"] == alpha]
            if not items:
                continue
            r_val = sum(items) / len(items)
            best = r_val if best is None else max(best, r_val)
        return best

    rows = []
    for target, label in targets:
        values = []
        for model_key, _short in MODELS:
            swap = per_model[model_key]
            if swap is None:
                values += [(None, False), (None, False), (None, False)]
                continue
            fracs = sorted({r["frac"] for r in swap})
            frac = nearest_float(fracs, target)
            values.append((rate(swap, frac, "write", "img"), False))
            values.append((rate(swap, frac, "output-layer row", "img"), False))
            values.append((rate(swap, frac, "random", "img"), False))
        every_pos = []
        for model_key, _short in MODELS:
            swap = per_model[model_key]
            if swap is None:
                every_pos.append((None, False))
                continue
            fracs = sorted({r["frac"] for r in swap})
            frac = nearest_float(fracs, target)
            every_pos.append((rate(swap, frac, "write", "all"), False))
        rows.append((label, values + every_pos))
    thead = ('    <thead><tr><th rowspan="2">depth written</th>'
              '<th colspan="3">into the picture, Qwen3-VL-2B</th>'
              '<th colspan="3">into the picture, Qwen3-VL-8B</th>'
              '<th colspan="2">at every position</th></tr>\n'
              "    <tr><th>the write</th><th>output-layer row</th><th>random</th>"
              "<th>the write</th><th>output-layer row</th><th>random</th>"
              "<th>write, 2B</th><th>write, 8B</th></tr></thead>")
    groups = [[0, 1, 2], [3, 4, 5]]
    col_labels = ["2B write (img)", "2B output-layer row", "2B random",
                  "8B write (img)", "8B output-layer row", "8B random",
                  "write all 2B", "write all 8B"]
    html = build_block(thead, rows, groups, decimals=2)
    return {"html": html, "col_labels": col_labels, "notes": notes}


def build_refusalsteer():
    path_8b = os.path.join(RESULTS, "steer", "Qwen3-VL-8B-Instruct_refusal_steer.json")
    path_2b = os.path.join(RESULTS, "steer", "Qwen3-VL-2B-Instruct_refusal_steer.json")
    notes = []
    d2 = load_json(path_2b)
    if d2 is None:
        notes.append(f"missing {path_2b}; table stays Qwen3-VL-8B only, "
                     "matching the original table's scope")
    d = load_json(path_8b)
    if d is None:
        notes.append(f"missing {path_8b}")
        return None
    records = d["results"]

    depth_targets = [0.35, 0.48]
    fracs = sorted({r["frac"] for r in records if r["frac"] is not None})
    doses = [0.1, 0.2, 0.4]

    def value(direction, frac, alpha):
        for r in records:
            if (r["direction"] == direction and r.get("channel") == "ans"
                    and r["frac"] is not None and abs(r["frac"] - frac) < 1e-9
                    and r["alpha"] == alpha):
                return r.get("refusal_benign_after_add")
        return None

    directions = [
        ("the write", "write"),
        ("difference of means, labelled", "DIM (labelled)"),
        ("the raw direction", "raw (the final score)"),
        ("the read, inverted", "read inverted"),
        ("random, same strength", "random"),
    ]
    label_free_idxs = [i for i, (_lbl, key) in enumerate(directions) if key != "DIM (labelled)"]

    col_labels = []
    col_targets = []  # (frac, alpha)
    for dt in depth_targets:
        frac = nearest_float(fracs, dt) if fracs else None
        for alpha in doses:
            col_targets.append((frac, alpha))
            col_labels.append(f"depth {dt} dose {alpha}")

    rows = []
    for label, key in directions:
        vals = []
        for frac, alpha in col_targets:
            vals.append(value(key, frac, alpha) if frac is not None else None)
        rows.append((label, vals))

    thead = ('    <thead><tr><th rowspan="2">direction added</th>'
              '<th colspan="3">benign prompts now refused, depth 0.35</th>'
              '<th colspan="3">depth 0.48</th></tr>\n'
              "    <tr><th>0.1</th><th>0.2</th><th>0.4</th>"
              "<th>0.1</th><th>0.2</th><th>0.4</th></tr></thead>")
    html = build_block_colmarked(thead, rows, label_free_idxs, decimals=2)
    return {"html": html, "col_labels": col_labels, "notes": notes}


TABLE_BUILDERS = [
    ("readdecision", build_readdecision),
    ("presence", build_presence),
    ("hallucination", build_hallucination),
    ("regionnaming", build_regionnaming),
    ("twostep", build_twostep),
    ("twostepregions", build_twostepregions),
    ("atlas", build_atlas),
    ("writeinto", build_writeinto),
    ("refusalsteer", build_refusalsteer),
]


# --------------------------------------------------------------------------
# diffing and prose-recheck heuristics
# --------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_TR_RE = re.compile(r"<tr>(.*?)</tr>", re.S)
_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)


def extract_tbody_rows(block_text):
    m = re.search(r"<tbody>(.*?)</tbody>", block_text, re.S)
    if not m:
        return []
    grid = []
    for tr in _TR_RE.findall(m.group(1)):
        cells = [_TAG_RE.sub("", c).strip() for c in _TD_RE.findall(tr)]
        grid.append(cells)
    return grid


def diff_table(name, old_block, new_block, col_labels):
    old_grid = extract_tbody_rows(old_block) if old_block else []
    new_grid = extract_tbody_rows(new_block)
    changes = []
    shape_ok = (len(old_grid) == len(new_grid) and
                all(len(o) == len(n) for o, n in zip(old_grid, new_grid)))
    if not shape_ok:
        changes.append({"shape_changed": True, "old": old_grid, "new": new_grid})
        return changes
    for orow, nrow in zip(old_grid, new_grid):
        row_label = nrow[0] if nrow else ""
        for cidx in range(1, len(nrow)):
            ov, nv = orow[cidx], nrow[cidx]
            if ov != nv:
                col_label = col_labels[cidx - 1] if cidx - 1 < len(col_labels) else f"col{cidx}"
                changes.append({"row": row_label, "col": col_label, "old": ov, "new": nv})
    return changes


def following_prose(html, name, window=4000):
    marker = f"<!--/TABLE:{name}-->"
    idx = html.find(marker)
    if idx == -1:
        return ""
    start = idx + len(marker)
    chunk = html[start:start + window]
    m = re.search(r"<h[23]>|<!--TABLE:", chunk)
    if m:
        chunk = chunk[:m.start()]
    text = _TAG_RE.sub(" ", chunk)
    return re.sub(r"\s+", " ", text).strip()


def prose_candidates(html, name, changes):
    """Best-effort: for each changed cell's old numeric value, see if that
    exact number appears in the prose immediately following the table, and
    if so surface the sentence for a human to recheck."""
    prose = following_prose(html, name)
    if not prose:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", prose)
    hits = []
    seen = set()
    for ch in changes:
        if ch.get("shape_changed"):
            continue
        old_val = ch["old"]
        if not old_val:
            continue
        bare = old_val.lstrip(MINUS + "-")
        for needle in {old_val, bare}:
            if not needle:
                continue
            for sent in sentences:
                if needle in sent and (name, sent) not in seen:
                    seen.add((name, sent))
                    hits.append(f'  [{name}] "{sent.strip()}" -- cell was {old_val}, now {ch["new"]}')
    return hits


def main():
    html = open(HTML_PATH, encoding="utf-8").read()
    all_prose_hits = []
    any_missing = False

    for name, builder in TABLE_BUILDERS:
        pattern = re.compile(
            r"(<!--TABLE:%s-->\n)(.*?)(<!--/TABLE:%s-->)" % (re.escape(name), re.escape(name)),
            re.S,
        )
        m = pattern.search(html)
        if not m:
            print(f"[{name}] markers not found in {HTML_PATH}; skipped")
            continue

        result = builder()
        if result is None:
            print(f"[{name}] SKIPPED: source JSON missing, table left as-is")
            any_missing = True
            continue

        old_block = m.group(2)
        new_block = result["html"] + "\n"
        changes = diff_table(name, old_block, new_block, result["col_labels"])

        if changes and changes[0].get("shape_changed"):
            new_grid = changes[0]["new"]
            old_grid = changes[0]["old"]
            print(f"[{name}] table shape changed ({len(old_grid)}x"
                  f"{len(old_grid[0]) if old_grid else 0} -> {len(new_grid)}x"
                  f"{len(new_grid[0]) if new_grid else 0}) -- rewriting in full, new rows:")
            for row in new_grid:
                print(f"    {row}")
        elif changes:
            print(f"[{name}] {len(changes)} cell(s) changed:")
            for c in changes:
                print(f'    depth {c["row"]!r} / {c["col"]}: {c["old"]!r} -> {c["new"]!r}')
        else:
            print(f"[{name}] no changes")

        for note in result.get("notes", []):
            print(f"[{name}] NOTE: {note}")
            any_missing = any_missing or "missing" in note

        html = html[:m.start(2)] + new_block + html[m.end(2):]
        all_prose_hits.extend(prose_candidates(html, name, changes))

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print()
    if all_prose_hits:
        print("Prose sentences worth rechecking (a quoted number may now disagree with the rebuilt table):")
        for hit in all_prose_hits:
            print(hit)
    else:
        print("No prose sentence was found to quote a number that just changed.")

    if any_missing:
        print("\nSome tables were skipped or partially filled because their "
              "source JSON was missing -- see NOTE lines above.")


if __name__ == "__main__":
    sys.exit(main())
