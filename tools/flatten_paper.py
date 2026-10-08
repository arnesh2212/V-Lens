# SPDX-License-Identifier: Apache-2.0
"""Make the paper's .tex easy to copy and edit: tables inline, one line per paragraph.

* ``\\input{../tables/X.tex}`` is replaced by the table's contents between
  ``% BEGIN generated table X`` and ``% END generated table X``; re-running
  refreshes the text between existing markers from the current table file.
* every prose paragraph, every ``\\caption{...}`` and every ``\\item`` becomes one
  line; environment lines, figure commands and tabular rows are left alone.

    python3 tools/flatten_paper.py paper/setup.tex paper/results.tex paper/appendix_results.tex
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP = re.compile(r"^\s*(\\begin|\\end|\\includegraphics|\\label|\\centering|\\small|\\input|\\FloatBarrier|"
                  r"\\toprule|\\midrule|\\bottomrule|\\cmidrule|\\section|\\subsection|\\subsubsection|%|\\IfFileExists|\}\{\}|\\fbox)")


def _table(name):
    p = os.path.join(HERE, "tables", name)
    body = open(p).read().strip().split("\n")
    body = [l for l in body if not l.startswith("% Written by")]
    return [f"% BEGIN generated table {name} (tools/make_paper_tables.py; re-run tools/flatten_paper.py to refresh)"] \
        + body + [f"% END generated table {name}"]


def inline_tables(lines):
    out, i = [], 0
    while i < len(lines):
        l = lines[i]
        m = re.match(r"^\s*% BEGIN generated table (\S+)", l)
        if m:
            j = i
            while j < len(lines) and not lines[j].startswith(f"% END generated table {m.group(1)}"):
                j += 1
            out += _table(m.group(1)); i = j + 1
            continue
        m = re.match(r"^\s*\\input\{\.\./tables/(\S+?\.tex)\}\s*$", l)
        if m and os.path.exists(os.path.join(HERE, "tables", m.group(1))):
            out += _table(m.group(1)); i += 1
            continue
        out.append(l); i += 1
    return out


def _in_tabular(lines):
    """Indices of lines inside a tabular environment (left untouched)."""
    inside, keep = False, set()
    for i, l in enumerate(lines):
        if "\\begin{tabular" in l:
            inside = True
        if inside or l.startswith("% BEGIN generated table") or l.startswith("% END generated table"):
            keep.add(i)
        if "\\end{tabular" in l:
            inside = False
    return keep


def join(lines):
    frozen = _in_tabular(lines)
    out, buf = [], []

    def flush():
        if buf:
            out.append(" ".join(x.strip() for x in buf))
            buf.clear()
    for i, l in enumerate(lines):
        s = l.strip()
        if i in frozen or not s or KEEP.match(l):
            flush()
            out.append(l.rstrip())
            continue
        # a caption, an item or a paragraph starts a new joined line
        if re.match(r"^\s*(\\caption\{|\\item\b|\\paragraph\{)", l) and buf and not buf[-1].rstrip().endswith("%"):
            flush()
        buf.append(l)
    flush()
    return out


def main(paths):
    for p in paths:
        lines = open(p).read().split("\n")
        lines = join(inline_tables(lines))
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        open(p, "w").write(text)
        print(f"flattened {p}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["paper/setup.tex", "paper/results.tex", "paper/appendix_new.tex"])
