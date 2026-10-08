# SPDX-License-Identifier: Apache-2.0
"""Redraw the figures of a page from the results on disk (no GPU needed).

    python3 build/figures.py datasets     # figures/vlens/*.png for paper/datasets.html
    python3 build/figures.py vlens        # figures/*.png for paper/tool.html
    python3 build/figures.py vlens flow method gate_channels   # only these
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv):
    page = argv[0] if argv else "datasets"
    if page == "datasets":
        from vlens.figures import datasets as mod
    elif page == "vlens":
        from vlens.figures import vlens_page as mod
    else:
        raise SystemExit(f"unknown page {page!r}; use datasets or vlens")
    names = argv[1:]
    if not names:
        mod.all_figures()
        return
    for n in names:
        fn = getattr(mod, n, None)
        if fn is None:
            raise SystemExit(f"{page} has no figure {n!r}")
        fn()


if __name__ == "__main__":
    main(sys.argv[1:])
