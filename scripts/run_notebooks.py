"""Execute the lesson notebooks to make sure they work.

Two passes are available:

* ``--mode stub``     : run each notebook exactly as a student receives it
                        (exercises unsolved).  Every cell must still run — the
                        checker reports "not yet" and later cells fall back to
                        the reference implementations through ``use()``.
* ``--mode solution`` : replace every exercise cell by its registered solution
                        and run again; now every ``check`` must pass (a failing
                        check raises).

Usage::

    python scripts/run_notebooks.py                    # both passes, all notebooks
    python scripts/run_notebooks.py --mode solution lesson03
    python scripts/run_notebooks.py --save             # keep executed copies in build/

Exit status is non-zero if anything failed.
"""
from __future__ import annotations

import argparse
import copy
import glob
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
NB_DIR = os.path.join(ROOT, "notebooks")
BUILD_DIR = os.path.join(ROOT, "build")

from rlcourse import exercises  # noqa: E402
from scripts.nbtools import exercise_ids  # noqa: E402


def with_solutions(nb):
    nb = copy.deepcopy(nb)
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.metadata.get("rlcourse", {}).get("exercise"):
            ex = exercises.get(cell.metadata["rlcourse"]["exercise"])
            cell.source = ex.solution_cell
    return nb


def check_registry(nb, name) -> list[str]:
    """Every exercise cell must be registered, and every registered exercise of
    the lesson must appear in the notebook."""
    problems = []
    ids = exercise_ids(nb)
    for i in ids:
        try:
            exercises.get(i)
        except KeyError as exc:
            problems.append(f"{name}: exercise cell {i} is not registered ({exc})")
    lesson = name.split("_")[0].replace("lesson", "")
    if lesson.isdigit():
        try:
            registered = [ex.id for ex in exercises.all_exercises(int(lesson))]
        except ModuleNotFoundError:
            registered = []
        for i in registered:
            if i not in ids:
                problems.append(f"{name}: registered exercise {i} has no cell in the notebook")
    return problems


def run(nb_path: str, mode: str, timeout: int, save: bool) -> tuple[bool, str]:
    name = os.path.splitext(os.path.basename(nb_path))[0]
    nb = nbformat.read(nb_path, as_version=4)
    problems = check_registry(nb, name)
    if problems:
        return False, "\n".join(problems)
    if mode == "solution":
        nb = with_solutions(nb)
    # the kernel inherits this process's environment
    os.environ["RLCOURSE_MODE"] = mode
    os.environ.pop("MPLBACKEND", None)        # the kernel's inline backend captures figures
    os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
    os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": NB_DIR}},
                            allow_errors=False)
    t0 = time.time()
    try:
        client.execute()
        ok, msg = True, f"ok ({time.time() - t0:.0f}s)"
    except Exception as exc:  # noqa: BLE001
        ok, msg = False, f"FAILED after {time.time() - t0:.0f}s\n{_short(exc)}"
    if save:
        os.makedirs(BUILD_DIR, exist_ok=True)
        nbformat.write(nb, os.path.join(BUILD_DIR, f"{name}.{mode}.ipynb"))
    return ok, msg


def _short(exc: Exception) -> str:
    text = str(exc)
    lines = text.splitlines()
    return "\n".join(lines[-40:]) if len(lines) > 40 else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="*", help="notebook name prefixes, e.g. lesson03")
    ap.add_argument("--mode", choices=["stub", "solution", "both"], default="both")
    ap.add_argument("--timeout", type=int, default=1800, help="per-cell timeout in seconds")
    ap.add_argument("--save", action="store_true", help="save executed notebooks to build/")
    args = ap.parse_args()

    paths = []
    if args.notebooks:
        for prefix in args.notebooks:
            paths += sorted(glob.glob(os.path.join(NB_DIR, f"{prefix}*.ipynb")))
    else:
        paths = sorted(glob.glob(os.path.join(NB_DIR, "lesson*.ipynb")))
    if not paths:
        print("no notebooks found")
        return 1
    modes = ["stub", "solution"] if args.mode == "both" else [args.mode]
    failed = 0
    for p in paths:
        for mode in modes:
            ok, msg = run(p, mode, args.timeout, args.save)
            print(f"[{mode:8s}] {os.path.basename(p):45s} {msg}")
            failed += not ok
    print(f"\n{failed} failure(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
