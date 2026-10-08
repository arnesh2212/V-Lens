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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlens.paths import PAPER, RESULTS  # noqa: E402

HTML_PATH = os.path.join(PAPER, "tool.html")

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
             + "".join("<th>V-Read</th><th>raw</th><th>probe</th>" for _ in models)
             + "</tr></thead>")
    groups = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(len(models))]
    col_labels = [f"{short} {c}" for _, short in models for c in ("V-Read", "raw", "probe")]
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
              "    <tr><th>V-Read</th><th>logit lens</th><th>V-Write</th><th>probe</th>"
              "<th>V-Read</th><th>logit lens</th><th>V-Write</th><th>probe</th></tr></thead>")
    groups = [[0, 1, 2], [4, 5, 6]]
    col_labels = ["2B V-Read", "2B logit lens", "2B V-Write", "2B probe",
                  "8B V-Read", "8B logit lens", "8B V-Write", "8B probe"]
    html = build_block(thead, rows, groups, decimals=2)
    return {"html": html, "col_labels": col_labels, "notes": notes}


def build_hallucination():
    targets = [(0.13, "0.13"), (0.22, "0.22"), (0.35, "0.35"),
               (0.48, "0.48"), (0.65, "0.65"), (0.96, "0.96")]
    rows, notes = _presence_like("hallucination_auroc", targets, "hallucination")
    if rows is None:
        return None
    notes = list(notes)
    notes.append("table now shows 'V-Write' alongside read/logit lens/probe, "
                  "matching the instrument set the task specifies; the previous "
                  "hand-built table omitted a write column")
    thead = ('    <thead><tr><th rowspan="2">depth</th>'
              '<th colspan="4">Qwen3-VL-2B</th>'
              '<th colspan="4">Qwen3-VL-8B</th></tr>\n'
              "    <tr><th>V-Read</th><th>logit lens</th><th>V-Write</th><th>probe</th>"
              "<th>V-Read</th><th>logit lens</th><th>V-Write</th><th>probe</th></tr></thead>")
    groups = [[0, 1, 2], [4, 5, 6]]
    col_labels = ["2B V-Read", "2B logit lens", "2B V-Write", "2B probe",
                  "8B V-Read", "8B logit lens", "8B V-Write", "8B probe"]
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
              "    <tr><th>V-Write</th><th>logit lens</th><th>logit lens, centred</th>"
              "<th>V-Write</th><th>logit lens</th><th>logit lens, centred</th></tr></thead>")
    groups = [[0, 1, 2], [3, 4, 5]]
    col_labels = ["2B V-Write", "2B logit lens", "2B logit lens centred",
                  "8B V-Write", "8B logit lens", "8B logit lens centred"]
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
              "    <tr><th>two-step, V-Read</th><th>two-step, logit lens</th><th>one-step, V-Read</th>"
              "<th>two-step, V-Read</th><th>two-step, logit lens</th><th>one-step, V-Read</th></tr></thead>")
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
              '<th colspan="3">Qwen3-VL-2B, share of V-Write over chance</th>'
              '<th colspan="3">Qwen3-VL-8B, share of V-Write over chance</th>'
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
    """Rank agreement with patching, per instrument, for every model whose atlas exists.

    The instrument keys were renamed part way through the runs, so both spellings are
    accepted and the display label is fixed here.
    """
    def pick(ref, *names):
        for n in names:
            if n in ref:
                return ref[n]
        return {}

    targets = [(0.22, "0.22"), (0.35, "0.35"), (0.48, "0.48"),
               (0.65, "0.65"), (0.78, "0.78"), (0.96, "0.96")]
    notes = []
    models, data = [], {}
    for key, short in MODELS:
        d = load_json(os.path.join(RESULTS, "atlas", f"{key}_atlas.json"))
        if d is None:
            notes.append(f"missing atlas for {key}")
            continue
        models.append((key, short)); data[key] = d
    if not models:
        return None
    instruments = [("write, path", "V-Write, path"), ("write, one point", "V-Write, one point"),
                   ("write, averaged", "V-Write, averaged"), ("identity", "logit lens"),
                   ("grad x act", "grad × act")]
    rows = []
    for target, label in targets:
        values = []
        for key, _short in models:
            d = data[key]
            depth_map = {k: float(v) for k, v in d["depths"].items()}
            layer = nearest_key(depth_map, target)
            ref = d["summary"]["refusal"]
            for raw, _disp in instruments:
                entry = pick(ref, raw, "V-" + raw.capitalize().replace("Write", "Write"),
                             raw.replace("write", "V-Write")).get("rho_resample", {}).get(layer)
                values.append((entry[0] if entry else None, False))
            g = pick(ref, "GLIMPSE").get("rho_abs", {}).get(layer)
            values.append((g[0] if g else None, False))
        rows.append((label, values))
    per = len(instruments) + 1
    thead = ('    <thead><tr><th rowspan="2">depth</th>'
             + "".join(f'<th colspan="{per}">{short}</th>' for _, short in models)
             + '</tr>\n    <tr>'
             + "".join("".join(f"<th>{disp}</th>" for _, disp in instruments) + "<th>GLIMPSE</th>" for _ in models)
             + "</tr></thead>")
    groups = [list(range(i * per, (i + 1) * per)) for i in range(len(models))]
    col_labels = [f"{short} {disp}" for _, short in models for disp in [d for _, d in instruments] + ["GLIMPSE"]]
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
              "    <tr><th>V-Write</th><th>output-layer row</th><th>random</th>"
              "<th>V-Write</th><th>output-layer row</th><th>random</th>"
              "<th>write, 2B</th><th>write, 8B</th></tr></thead>")
    groups = [[0, 1, 2], [3, 4, 5]]
    col_labels = ["2B V-Write (img)", "2B output-layer row", "2B random",
                  "8B V-Write (img)", "8B output-layer row", "8B random",
                  "write all 2B", "write all 8B"]
    html = build_block(thead, rows, groups, decimals=2)
    return {"html": html, "col_labels": col_labels, "notes": notes}


def build_refusalsteer():
    """Adding a direction at the answer spot and asking whether benign prompts are
    now refused, both models, three doses at two depths.  Each model's own
    no-edit rate is reported alongside, since the two differ enough that the
    numbers only mean anything against it."""
    depth_targets = [0.35, 0.48]
    doses = [0.1, 0.2, 0.4]
    directions = [
        ("V-Write", "V-Write"),
        ("difference of means, labelled", "DIM (labelled)"),
        ("the raw direction", "raw (the final score)"),
        ("V-Read, inverted", "read inverted"),
        ("random, same strength", "random"),
    ]
    notes, blocks = [], []
    for key, nice in MODELS:
        d = load_json(os.path.join(RESULTS, "steer", f"{key}_refusal_steer.json"))
        if d is None:
            notes.append(f"missing {key}_refusal_steer.json; that model is left out")
            continue
        records = d["results"]
        fracs = sorted({r["frac"] for r in records if r["frac"] is not None})
        base = next((r for r in records if r["direction"] == "baseline"), None)
        if base is not None:
            notes.append(f"{nice}: with no edit, {base.get('refusal_benign_after_add'):.2f} of the benign "
                         f"prompts are refused already")

        def value(direction, frac, alpha):
            # runs from before the rename spell the write direction "write"
            names = {direction, direction.replace("V-Write", "write")}
            for r in records:
                if (r["direction"] in names and r.get("channel") == "ans"
                        and r["frac"] is not None and abs(r["frac"] - frac) < 1e-9
                        and r["alpha"] == alpha):
                    return r.get("refusal_benign_after_add")
            return None

        targets = []
        for dt in depth_targets:
            frac = nearest_float(fracs, dt) if fracs else None
            for alpha in doses:
                targets.append((frac, alpha))
        rows = [(label, [value(k, f, al) if f is not None else None for f, al in targets])
                for label, k in directions]
        blocks.append((nice, base.get("refusal_benign_after_add") if base else None, rows))
    if not blocks:
        return None

    col_labels = [f"{nice} depth {dt} dose {al}" for nice, _b, _r in blocks for dt in depth_targets for al in doses]
    label_free = [i for i, (lbl, _k) in enumerate(directions) if lbl != "difference of means, labelled"]
    lines = ['  <div class="tblwrap">', "  <table>",
             ('    <thead><tr><th rowspan="2">direction added</th>'
              '<th colspan="3">benign prompts now refused, depth 0.35</th>'
              '<th colspan="3">depth 0.48</th></tr>\n'
              "    <tr><th>0.1</th><th>0.2</th><th>0.4</th>"
              "<th>0.1</th><th>0.2</th><th>0.4</th></tr></thead>"),
             "    <tbody>"]
    for nice, base_rate, rows in blocks:
        head = f"{nice}, {fmt(base_rate, 2)} refused with no edit" if base_rate is not None else nice
        lines.append(f'      <tr><td colspan="7">{head}</td></tr>')
        ncols = len(rows[0][1])
        marks = {}
        for c in range(ncols):
            cands = [(r, rows[r][1][c]) for r in label_free if rows[r][1][c] is not None]
            if cands:
                marks[c] = max(cands, key=lambda t: t[1])[0]
        for i, (label, vals) in enumerate(rows):
            tds = [f"<td>{label}</td>"]
            for c, v in enumerate(vals):
                text = fmt(v, 2)
                if marks.get(c) == i and text:
                    text = f'<span class="k">{text}</span>'
                tds.append(f"<td>{text}</td>")
            lines.append("      <tr>" + "".join(tds) + "</tr>")
    lines += ["    </tbody>", "  </table>", "  </div>"]
    return {"html": "\n".join(lines), "col_labels": col_labels, "notes": notes}


def build_steerdose():
    """Writing a chosen word into the picture: does the one-word answer become it,
    and does an unrelated question about the same photograph survive the edit?

    The rate of saying the target is over all 60 items at depth 0.48 in add mode.
    The last two columns are the unrelated presence question at dose 0.5, counted
    only over the items the clean model already answers correctly, since an item
    the model gets wrong before the edit says nothing about the edit's collateral.
    """
    NAMES = {"lens": "V-Write", "DIM (labelled)": "difference of means, labelled",
             "logit-lens": "output-layer row", "random": "random, same strength"}
    ORDER = ["V-Write", "difference of means, labelled", "output-layer row", "random, same strength"]
    LABELLED = {"difference of means, labelled"}
    notes, per_model = [], {}
    for key, _nice in MODELS:
        d = load_json(os.path.join(RESULTS, "steer", f"{key}_steer.json"))
        if d is None:
            notes.append(f"missing {key}_steer.json")
            return None
        runs = [(it, r) for it in d["items"] for r in it["runs"]]
        fr = nearest_float(sorted({round(r["frac"], 2) for _it, r in runs}), 0.48)
        base = [it for it in d["items"] if it.get("base_retained")]
        notes.append(f"{key}: {len(base)} of {len(d['items'])} items have the unrelated question right before any edit")
        say_rate, keep_rate = {}, {}
        for dn in ORDER:
            for al in (0.5, 1.0):
                v = [r for _it, r in runs if r["mode"] == "add" and NAMES.get(r["direction"]) == dn
                     and round(r["frac"], 2) == fr and r["alpha"] == al]
                say_rate[(dn, al)] = float(sum(x["said_B"] for x in v) / len(v)) if v else None
            v = [r for it in base for r in it["runs"] if r["mode"] == "add" and NAMES.get(r["direction"]) == dn
                 and round(r["frac"], 2) == fr and r["alpha"] == 0.5]
            keep_rate[dn] = float(sum(bool(x.get("retained")) for x in v) / len(v)) if v else None
        per_model[key] = (say_rate, keep_rate)

    rows, col_labels = [], []
    for dn in ORDER:
        vals = []
        for key, _nice in MODELS:
            for al in (0.5, 1.0):
                vals.append((per_model[key][0][(dn, al)], dn in LABELLED))
        for key, _nice in MODELS:
            vals.append((per_model[key][1][dn], dn in LABELLED))
        rows.append((dn, vals))
    for _key, nice in MODELS:
        col_labels += [f"{nice} dose 0.5", f"{nice} dose 1"]
    for _key, nice in MODELS:
        col_labels.append(f"{nice} unrelated still right")
    groups = [[0, 4], [1, 5], [2, 6], [3, 7]] if False else [[i] for i in range(len(col_labels))]
    thead = ('    <thead><tr><th rowspan="2">direction</th>'
             f'<th colspan="2">{MODELS[0][1]}</th><th colspan="2">{MODELS[1][1]}</th>'
             '<th colspan="2">unrelated question still right</th></tr>\n'
             "    <tr><th>dose 0.5</th><th>dose 1</th><th>dose 0.5</th><th>dose 1</th>"
             f"<th>{MODELS[0][1].split('-')[-1]}</th><th>{MODELS[1][1].split('-')[-1]}</th></tr></thead>")
    html = build_block(thead, rows, groups, decimals=2)
    return {"html": html, "col_labels": col_labels, "notes": notes}


def build_swap():
    """Held-out R2 of the swap check: how much of the real change in the score
    each direction predicted, V-Write against the raw direction, per depth."""
    per, depths, notes = {}, None, []
    models = [m for m in MODELS_DECISION]
    for key, _nice in models:
        d = load_json(os.path.join(RESULTS, "gate", f"{key}_gate.json"))
        if d is None:
            notes.append(f"missing {key}_gate.json")
            return None
        cal = d["calibration"]
        ls = sorted(cal, key=int)
        per[key] = [(cal[l]["frac"], cal[l]["verdict"]["r2_heldout"], cal[l]["identity"]["r2_heldout"]) for l in ls]
        depths = depths or [round(cal[l]["frac"], 2) for l in ls]
    models = [m for m in models if m[0] in per]
    rows, col_labels = [], []
    for _key, nice in models:
        col_labels += [f"{nice} V-Write", f"{nice} raw"]
    n = min(len(per[k]) for k, _ in models)
    for i in range(n):
        label = f"{per[models[0][0]][i][0]:.2g}"
        vals = []
        for key, _nice in models:
            vals.append((per[key][i][1], False))
            vals.append((per[key][i][2], False))
        rows.append((label, vals))
    groups = [[2 * j, 2 * j + 1] for j in range(len(models))]
    thead = ('    <thead><tr><th rowspan="2">depth</th>'
             + "".join(f'<th colspan="2">{nice}</th>' for _k, nice in models) + "</tr>\n"
             "    <tr>" + "".join("<th>V-Write</th><th>raw</th>" for _ in models) + "</tr></thead>")
    return {"html": build_block(thead, rows, groups, decimals=2), "col_labels": col_labels, "notes": notes}


def build_benchmarks():
    """Leave-one-benchmark-out: does the reading say whether Qwen3-VL-8B refuses
    that item?  Two maps are compared: one fitted on COCO photographs, which is
    the foreign map, and one refitted on the held-out set's own unlabelled
    states, five folds over items.  The probe is trained with labels on the same
    rows, cross-validated, which is the ceiling."""
    d = load_json(os.path.join(RESULTS, "reader", "Qwen3-VL-8B-Instruct_loo.json"))
    if d is None:
        return None
    NAMES = {"mmsb_typo": "MM-SafetyBench, typography", "figstep": "FigStep", "hades": "HADES",
             "vlguard": "VLGuard", "vlsbench": "VLSBench", "siuo": "SIUO",
             "mossbench_contrast": "MOSSBench, the contrast set", "spavl_harm": "SPA-VL, harmful"}
    ORDER = ["mmsb_typo", "figstep", "hades", "vlguard", "vlsbench", "siuo", "mossbench_contrast", "spavl_harm"]
    DEPTHS = [0.2, 0.48]
    by = {(r["held"], round(r["frac"], 2)): r for r in d["results"] if r.get("decision")}
    fracs = sorted({round(r["frac"], 2) for r in d["results"]})
    fr = {t: nearest_float(fracs, t) for t in DEPTHS}
    rows, notes = [], []
    for held in ORDER:
        if (held, fr[DEPTHS[0]]) not in by:
            notes.append(f"no decision rows for {held}; skipped")
            continue
        first = by[(held, fr[DEPTHS[0]])]["decision"]
        vals = [(first["n"], False), (first["n_refused"] / first["n"], False)]
        for t in DEPTHS:
            dec = by[(held, fr[t])]["decision"]
            vals += [(dec["reader_family"], False), (dec["reader"], False), (dec["hidden"], False), (dec["probe"], True)]
        rows.append((NAMES.get(held, held), vals))
    groups = [[2, 3, 4, 5], [6, 7, 8, 9]]
    col_labels = ["items", "refused"] + [f"depth {t} {k}" for t in DEPTHS
                                          for k in ("V-Read refit here", "V-Read foreign map", "raw", "probe")]
    thead = ('    <thead><tr><th rowspan="2">held-out benchmark</th><th rowspan="2">items</th>'
             '<th rowspan="2">refused</th>'
             f'<th colspan="4">depth {fr[DEPTHS[0]]:.2g}</th><th colspan="4">depth {fr[DEPTHS[1]]:.2g}</th></tr>\n'
             "    <tr><th>refit here</th><th>foreign map</th><th>raw</th><th>probe</th>"
             "<th>refit here</th><th>foreign map</th><th>raw</th><th>probe</th></tr></thead>")
    lines = ['  <div class="tblwrap">', "  <table>", thead, "    <tbody>"]
    for label, vals in rows:
        marked = set()
        for group in groups:
            cands = [(i, vals[i][0]) for i in group if vals[i][0] is not None and not vals[i][1]]
            if cands:
                marked.add(max(cands, key=lambda t: t[1])[0])
        tds = [f"<td>{label}</td>", f"<td>{int(vals[0][0])}</td>", f"<td>{fmt(vals[1][0], 2)}</td>"]
        for i in range(2, len(vals)):
            text = fmt(vals[i][0], 2)
            if i in marked and text:
                text = f'<span class="k">{text}</span>'
            tds.append(f"<td>{text}</td>")
        lines.append("      <tr>" + "".join(tds) + "</tr>")
    lines += ["    </tbody>", "  </table>", "  </div>"]
    return {"html": "\n".join(lines), "col_labels": col_labels, "notes": notes}


def build_judges():
    """How the three judges score the same replies, on the harmful items each
    model was asked, plus their pairwise agreement.  Rates are over the items
    both guards actually judged, so the three columns are comparable."""
    from vlens.judge import guards as _guards  # noqa: E402
    from vlens import judge as _judge  # noqa: E402
    rows_by_model, kappa_by_model, notes = {}, {}, []
    for key, _nice in MODELS:
        d = load_json(os.path.join(RESULTS, "gate", f"{key}__eval.json"))
        if d is None:
            notes.append(f"missing {key}__eval.json")
            return None
        items = d if isinstance(d, list) else (d.get("items") or d.get("rows") or [])
        harm = [it for it in items if it.get("harmful") and it.get("harmbench_complied") is not None
                and it.get("qwen3guard_complied") is not None]
        notes.append(f"{key}: {len(harm)} harmful items judged by both guards")
        cols = {
            "pattern matching": [bool(_judge.complied(it.get("reply_long") or it.get("reply", ""))) for it in harm],
            "HarmBench classifier": [bool(it["harmbench_complied"]) for it in harm],
            "Qwen3Guard": [bool(it["qwen3guard_complied"]) for it in harm],
            "both guards agree": [bool(it.get("judge_majority_complied")) for it in harm],
            "either guard": [bool(it.get("judge_any_complied")) for it in harm],
        }
        rows_by_model[key] = {k: (sum(v) / len(v) if v else None) for k, v in cols.items()}
        three = ["pattern matching", "HarmBench classifier", "Qwen3Guard"]
        kappa_by_model[key] = {k: [_guards.kappa(cols[k], cols[o]) for o in three if o != k] for k in three}

    order = ["pattern matching", "HarmBench classifier", "Qwen3Guard", "both guards agree", "either guard"]
    lines = ['  <div class="tblwrap">', "  <table>", 
             ('    <thead><tr><th rowspan="2">judge</th>'
              '<th colspan="2">says the reply carries out the request, harmful items</th>'
              '<th colspan="2">agreement with the others (Cohen’s kappa)</th></tr>\n'
              f'    <tr><th>{MODELS[0][1]}</th><th>{MODELS[1][1]}</th>'
              f"<th>{MODELS[0][1].split('-')[-1]}</th><th>{MODELS[1][1].split('-')[-1]}</th></tr></thead>"),
             "    <tbody>"]
    col_labels = [f"{nice} rate" for _k, nice in MODELS] + [f"{nice} kappa" for _k, nice in MODELS]
    for name in order:
        tds = [f"<td>{name}</td>"]
        for key, _nice in MODELS:
            tds.append(f"<td>{fmt(rows_by_model[key][name], 2)}</td>")
        for key, _nice in MODELS:
            ks = kappa_by_model[key].get(name)
            tds.append(f"<td>{' / '.join(fmt(k, 2) for k in ks) if ks else ''}</td>")
        lines.append("      <tr>" + "".join(tds) + "</tr>")
    lines += ["    </tbody>", "  </table>", "  </div>"]
    return {"html": "\n".join(lines), "col_labels": col_labels, "notes": notes}


def build_channels():
    """Where the upcoming refusal can be read: V-Write summed over the words,
    over every position, the raw reading at the answer spot, and a labelled
    probe, per depth and per model."""
    per, notes = {}, []
    models = []
    for key, nice in MODELS_DECISION:
        d = load_json(os.path.join(RESULTS, "gate", f"{key}_total_verdict.json"))
        if d is None:
            notes.append(f"missing {key}_total_verdict.json; that model is left out")
            continue
        ls = sorted(d, key=lambda z: d[z]["frac"])
        per[key] = [(d[l]["frac"], d[l]["txt"], d[l]["total"], d[l]["identity"], d[l]["probe"]) for l in ls]
        models.append((key, nice))
    if not models:
        return None
    n = min(len(per[k]) for k, _ in models)
    rows = []
    for i in range(n):
        label = f"{per[models[0][0]][i][0]:.2g}"
        vals = []
        for key, _nice in models:
            f, txt, total, ident, probe = per[key][i]
            vals += [(txt, False), (total, False), (ident, False), (probe, True)]
        rows.append((label, vals))
    groups = [[4 * j, 4 * j + 1, 4 * j + 2, 4 * j + 3] for j in range(len(models))]
    col_labels = [f"{nice} {k}" for _key, nice in models for k in ("words", "everywhere", "raw", "probe")]
    thead = ('    <thead><tr><th rowspan="2">depth</th>'
             + "".join(f'<th colspan="4">{nice}</th>' for _k, nice in models) + "</tr>\n"
             "    <tr>" + "".join("<th>words</th><th>everywhere</th><th>raw</th><th>probe</th>" for _ in models)
             + "</tr></thead>")
    return {"html": build_block(thead, rows, groups, decimals=2), "col_labels": col_labels, "notes": notes}


def build_prefillcont():
    """What the model does after each prefill: backs out later, or carries on."""
    d = load_json(os.path.join(RESULTS, "prefill", "Qwen3-VL-8B-Instruct_prefill.json"))
    if d is None:
        return None
    LABEL = {"none": "not started for it", "compliance": "“Sure, here are the steps:”",
             "refusal": "“I’m sorry, but I can’t”", "neutral": "“Here is a description:”"}
    ORDER = ["none", "compliance", "refusal", "neutral"]
    cont = d["continuation"]
    rows = []
    for k in ORDER:
        if k not in cont:
            continue
        c = cont[k]
        rows.append((LABEL.get(k, k), [(c["backtrack_harmful"], False), (c["backtrack_benign"], False),
                                       (c["complied_harmful"], False), (c["complied_benign"], False)]))
    thead = ('    <thead><tr><th rowspan="2">how the reply is started</th>'
             '<th colspan="2">refuses or backs out later</th><th colspan="2">continues, pattern matched</th></tr>\n'
             "    <tr><th>harmful</th><th>benign</th><th>harmful</th><th>benign</th></tr></thead>")
    col_labels = ["backtrack harmful", "backtrack benign", "complied harmful", "complied benign"]
    return {"html": build_block(thead, rows, [], decimals=2), "col_labels": col_labels,
            "notes": [f"{d['n_harm']} harmful and {d['n_ben']} benign items"]}


def build_prefillread():
    """Under a compliance prefill, does the reading still tell the harmful
    requests from the benign ones?  Both maps, the raw reading and a probe."""
    d = load_json(os.path.join(RESULTS, "prefill", "Qwen3-VL-8B-Instruct_prefill.json"))
    if d is None:
        return None
    TARGETS = [0.2, 0.35, 0.48, 0.65, 0.94]
    rows_by_frac = {round(r["frac"], 2): r for r in d["rows"]}
    fracs = sorted(rows_by_frac)
    rows = []
    for t in TARGETS:
        f = nearest_float(fracs, t)
        a = rows_by_frac[f]["auroc"]
        rows.append((f"{f:.2g}", [(a.get("reader_family"), False), (a.get("reader"), False),
                                  (a.get("V-Write"), False), (a.get("identity"), False), (a.get("probe"), True)]))
    thead = ('    <thead><tr><th>depth</th><th>V-Read, map refit here</th><th>V-Read, map from elsewhere</th>'
             "<th>V-Write</th><th>raw reading</th><th>probe (labelled)</th></tr></thead>")
    col_labels = ["V-Read refit", "V-Read foreign", "V-Write", "raw", "probe"]
    return {"html": build_block(thead, rows, [[0, 1, 2, 3, 4]], decimals=2), "col_labels": col_labels, "notes": []}


TABLE_BUILDERS = [
    ("readdecision", build_readdecision),
    ("presence", build_presence),
    ("hallucination", build_hallucination),
    ("regionnaming", build_regionnaming),
    ("twostep", build_twostep),
    ("twostepregions", build_twostepregions),
    ("atlas", build_atlas),
    ("writeinto", build_writeinto),
    ("steerdose", build_steerdose),
    ("swap", build_swap),
    ("benchmarks", build_benchmarks),
    ("judges", build_judges),
    ("channels", build_channels),
    ("prefillcont", build_prefillcont),
    ("prefillread", build_prefillread),
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
