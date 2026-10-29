"""Build Jupyter notebooks from plain-text lesson sources.

Lesson sources live in ``lessons_src/lessonNN_*.txt`` and use a simple
cell-marker format (a relaxed version of the jupytext "percent" format)::

    # %% [markdown]
    Any markdown, written as-is (no leading '#').  LaTeX with $...$ is fine.

    # %%
    code cell

    # %% [exercise 3.1]
    code cell for exercise 3.1 (the stub + the check call); the id is stored in
    the cell metadata so that scripts/run_notebooks.py can swap in the solution.

Build all notebooks with::

    python scripts/nbtools.py            # lessons_src/*.txt -> notebooks/*.ipynb
    python scripts/nbtools.py lesson03   # just one
"""
from __future__ import annotations

import glob
import os
import re
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC_DIR = os.path.join(ROOT, "lessons_src")
NB_DIR = os.path.join(ROOT, "notebooks")

_MARKER = re.compile(r"^# %%\s*(\[(?P<kind>markdown|exercise)(?:\s+(?P<arg>[^\]]+))?\])?\s*$")


def parse_cells(text: str) -> list[tuple[str, str, dict]]:
    """Return a list of (cell_type, source, metadata)."""
    cells: list[tuple[str, str, dict]] = []
    current_kind, current_arg, buf = None, None, []

    def flush():
        if current_kind is None:
            return
        source = "\n".join(buf).strip("\n")
        if not source.strip():
            return
        if current_kind == "markdown":
            cells.append(("markdown", source, {}))
        elif current_kind == "exercise":
            cells.append(("code", source, {"rlcourse": {"exercise": current_arg.strip()}}))
        else:
            cells.append(("code", source, {}))

    for line in text.splitlines():
        m = _MARKER.match(line)
        if m:
            flush()
            buf = []
            current_kind = m.group("kind") or "code"
            current_arg = m.group("arg")
            if current_kind == "exercise" and not current_arg:
                raise ValueError("exercise marker needs an id: '# %% [exercise 3.1]'")
        else:
            buf.append(line)
    flush()
    return cells


def build_notebook(src_path: str, out_path: str) -> int:
    with open(src_path, encoding="utf-8") as fh:
        text = fh.read()
    cells = parse_cells(text)
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    for kind, source, meta in cells:
        if kind == "markdown":
            nb.cells.append(new_markdown_cell(source))
        else:
            cell = new_code_cell(source)
            cell.metadata.update(meta)
            nb.cells.append(cell)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nbformat.write(nb, out_path)
    return len(cells)


def exercise_ids(nb) -> list[str]:
    return [c.metadata["rlcourse"]["exercise"] for c in nb.cells
            if c.cell_type == "code" and "rlcourse" in c.metadata and "exercise" in c.metadata["rlcourse"]]


def main(argv):
    pattern = f"{argv[0]}*.txt" if argv else "*.txt"
    srcs = sorted(glob.glob(os.path.join(SRC_DIR, pattern)))
    if not srcs:
        print("no sources found for", pattern)
        return 1
    for src in srcs:
        name = os.path.splitext(os.path.basename(src))[0]
        out = os.path.join(NB_DIR, name + ".ipynb")
        n = build_notebook(src, out)
        print(f"{name}: {n} cells -> {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
