"""Structural checks of the lesson sources against docs/AUTHORING.md.

    python scripts/lint_lessons.py            # all lessons
    python scripts/lint_lessons.py lesson09   # one lesson

Reports, per lesson: the exercise mix, missing sections, setup-cell drift,
exercise cells without an introducing markdown cell, references without a DOI
or URL, and a few anti-patterns (np.random.seed, missing setup_style()).
"""
from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("RLCOURSE_MODE", "stub")

from rlcourse import exercises  # noqa: E402
from scripts.nbtools import parse_cells  # noqa: E402

SETUP = '''import sys; sys.path.insert(0, "..")      # find the rlcourse package when running from notebooks/
import numpy as np
import matplotlib.pyplot as plt
from rlcourse import check, hint, reveal, progress, use, setup_style
from rlcourse import models, tasks, fitting, plotting
setup_style()
rng = np.random.default_rng(2024)        # one seed per notebook: results are reproducible'''

TYPE_WORDS = {"code": "coding", "numeric": "maths", "choice": "conceptual", "free": "conceptual"}


def lint(path: str) -> list[str]:
    name = os.path.splitext(os.path.basename(path))[0]
    lesson = int(name[6:8])
    problems: list[str] = []
    with open(path, encoding="utf-8") as fh:
        cells = parse_cells(fh.read())
    text = "\n".join(src for _, src, _ in cells)

    # title cell
    kind, src, _ = cells[0]
    if kind != "markdown" or not re.match(rf"# Lesson {lesson} —", src):
        problems.append(f"first cell is not the title '# Lesson {lesson} — ...'")
    for key in ("What you will learn", "Prerequisites", "Estimated time"):
        if key not in src:
            problems.append(f"title cell lacks '{key}'")
    # setup cell
    code_cells = [c for c in cells if c[0] == "code"]
    if not code_cells:
        problems.append("no code cells")
    else:
        setup = code_cells[0][1]
        core = "\n".join(line for line in setup.splitlines()
                         if not line.startswith(("import pandas", "from rlcourse import datasets", "import os",
                                                 "import warnings", "warnings.", "from scipy", "import scipy",
                                                 "from rlcourse.models import", "import pandas as pd")))
        if lesson > 0 and core.strip() != SETUP.strip():
            problems.append("setup cell differs from the standard one")
    # sections
    for header in ("## Summary", "## References", "## What's next"):
        if header not in text:
            problems.append(f"missing section '{header}'")
    if "np.random.seed(" in text:
        problems.append("uses np.random.seed")
    if "setup_style()" not in text:
        problems.append("setup_style() not called")
    # exercises
    ids = [m["rlcourse"]["exercise"] for k, s, m in cells if k == "code" and m]
    try:
        registered = {ex.id: ex for ex in exercises.all_exercises(lesson)}
    except ModuleNotFoundError:
        registered = {}
        problems.append("no exercise module")
    for i, (kind, src, meta) in enumerate(cells):
        if kind != "code" or not meta:
            continue
        ex_id = meta["rlcourse"]["exercise"]
        # an introducing markdown cell within the two preceding cells
        intro = [c for c in cells[max(0, i - 3):i] if c[0] == "markdown" and f"Exercise {ex_id} " in c[1]]
        if not intro:
            problems.append(f"exercise {ex_id}: no introducing markdown cell '### ✏️ Exercise {ex_id} — ...'")
        else:
            head = intro[-1][1]
            ex = registered.get(ex_id)
            if ex is not None:
                want = TYPE_WORDS[ex.kind]
                if f"({want}" not in head and f"*({want}" not in head:
                    problems.append(f"exercise {ex_id}: intro does not state its type '({want})'")
        last = src.strip().splitlines()[-1].strip()
        calls = [m.start() for m in re.finditer(r'check\("[0-9.]+"', src)]
        last_call = src[calls[-1]:].split("(")[1] if calls else ""
        if not calls or not last_call.startswith(f'"{ex_id}"'):
            problems.append(f"exercise {ex_id}: the cell's last check() call does not target \"{ex_id}\"")
        ex = registered.get(ex_id)
        if ex is not None and ex.kind == "code" and last != ex.check_call.strip():
            problems.append(f"exercise {ex_id}: cell check call {last!r} != registered {ex.check_call!r}")
    expected = [f"{lesson}.{k}" for k in range(1, len(ids) + 1)]
    if ids != expected:
        problems.append(f"exercise ids are not consecutive/in order: {ids}")
    # mix
    kinds = [registered[i].kind for i in ids if i in registered]
    n_code = kinds.count("code")
    n_num = kinds.count("numeric")
    n_con = kinds.count("choice") + kinds.count("free")
    mix = f"{len(kinds)} exercises: {n_code} coding, {n_num} maths, {n_con} conceptual"
    if lesson > 0 and (n_code < 3 or n_num < 2 or n_con < 2):
        problems.append("exercise mix below the minimum (3 coding / 2 maths / 2 conceptual): " + mix)
    # references
    refs = text.split("## References", 1)[1] if "## References" in text else ""
    for line in refs.splitlines():
        if (line.startswith("- ") and "doi.org" not in line and "http" not in line
                and not any(w in line for w in ("Press", "Eds.", "Proceedings", "(Ed.)", "thesis", "Technical Report", "Tech. Rep."))):
            problems.append("journal reference without DOI/URL: " + line[:70] + "...")
    # length
    n_md = sum(1 for c in cells if c[0] == "markdown")
    summary = f"{name}: {len(cells)} cells ({n_md} markdown), {mix}"
    return [summary] + ["  - " + p for p in problems]


def main(argv):
    patterns = [f"{a}*.txt" for a in argv] if argv else ["lesson*.txt"]
    paths = sorted(p for pat in patterns for p in glob.glob(os.path.join(ROOT, "lessons_src", pat)))
    n_problems = 0
    for p in paths:
        lines = lint(p)
        print("\n".join(lines))
        n_problems += len(lines) - 1
    print(f"\n{n_problems} problem(s)")
    return 1 if n_problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
