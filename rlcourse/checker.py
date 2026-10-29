"""The exercise checker.

Every exercise in the course is checked with one function::

    from rlcourse import check
    check("3.1", my_function)      # coding exercise  -> tests your function
    check("3.2", "b")              # multiple choice  -> checks your option
    check("3.3", 0.42)             # numeric / maths  -> checks your number
    check("3.4", "my explanation") # free text        -> keyword feedback + model answer

Feedback is *gradual*: a wrong answer gets a diagnosis of what probably went
wrong plus one new hint; after the hints run out the worked solution is shown,
together with the mistakes students most commonly make.  You can also ask for a
hint without attempting (``hint("3.1")``), reveal a solution (``reveal("3.1")``),
list your progress (``progress()``) or reset an exercise (``reset("3.1")``).

The checker works in any Python front-end (JupyterLab, VS Code, Colab, a plain
terminal).  Nothing here needs an internet connection.
"""
from __future__ import annotations

import copy
import inspect
import json
import os
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Modes.  RLCOURSE_MODE is set by the automated notebook runner only.
#   ""         : normal interactive use (progress persisted to a JSON file)
#   "stub"     : run notebooks with the blank exercises, never raise
#   "solution" : run notebooks with the solutions filled in; a failing check
#                raises AssertionError so that the test runner notices.
# --------------------------------------------------------------------------- #
MODE = os.environ.get("RLCOURSE_MODE", "")
PROGRESS_FILE = os.environ.get("RLCOURSE_PROGRESS_FILE", ".rlcourse_progress.json")

PASS = "✅"
FAIL = "❌"
HINT = "💡"
DIAG = "🔍"
BOOK = "📖"


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
def _in_ipython() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:  # pragma: no cover
        return False


def _show(markdown: str) -> None:
    """Render markdown in a notebook, or print plain text elsewhere."""
    if MODE == "":
        if _in_ipython():
            from IPython.display import Markdown, display

            display(Markdown(markdown))
            return
    # plain-text fallback: strip the most common markdown decorations
    text = markdown.replace("**", "")
    print(text)


def _code_block(code: str, lang: str = "python") -> str:
    return f"\n```{lang}\n{code.rstrip()}\n```\n"


# --------------------------------------------------------------------------- #
# Progress store
# --------------------------------------------------------------------------- #
class _Progress:
    def __init__(self) -> None:
        self.attempts: dict[str, int] = {}
        self.passed: dict[str, bool] = {}
        self.hints_shown: dict[str, int] = {}
        self.revealed: dict[str, bool] = {}
        self._load()

    # persistence -----------------------------------------------------------
    def _load(self) -> None:
        if MODE != "" or not os.path.exists(PROGRESS_FILE):
            return
        try:
            with open(PROGRESS_FILE) as fh:
                data = json.load(fh)
            self.attempts = dict(data.get("attempts", {}))
            self.passed = dict(data.get("passed", {}))
            self.hints_shown = dict(data.get("hints_shown", {}))
            self.revealed = dict(data.get("revealed", {}))
        except Exception:  # corrupt file: start afresh
            pass

    def save(self) -> None:
        if MODE != "":
            return
        try:
            with open(PROGRESS_FILE, "w") as fh:
                json.dump(
                    {
                        "attempts": self.attempts,
                        "passed": self.passed,
                        "hints_shown": self.hints_shown,
                        "revealed": self.revealed,
                    },
                    fh,
                    indent=1,
                )
        except Exception:  # read-only file system etc.
            pass

    def reset(self, ex_id: str) -> None:
        for d in (self.attempts, self.passed, self.hints_shown, self.revealed):
            d.pop(ex_id, None)
        self.save()


PROGRESS = _Progress()


# --------------------------------------------------------------------------- #
# Grade: the result of assessing one answer
# --------------------------------------------------------------------------- #
@dataclass
class Grade:
    passed: bool
    summary: str
    details: list[str] = field(default_factory=list)
    diagnosis: str | None = None


# --------------------------------------------------------------------------- #
# Exercise types
# --------------------------------------------------------------------------- #
@dataclass
class Exercise:
    """Base class.  Sub-classes implement ``grade`` and ``solution_markdown``."""

    id: str
    title: str
    hints: list[str] = field(default_factory=list)
    explanation: str = ""  # why the right answer is right (shown on success/reveal)
    common_mistakes: str = ""  # shown when the solution is revealed
    check_call: str = ""  # the line students run, e.g. check("3.1", rw_update)
    kind: str = "base"

    # --- to be implemented by subclasses ----------------------------------
    def grade(self, answer: Any) -> Grade:  # pragma: no cover
        raise NotImplementedError

    def solution_markdown(self) -> str:  # pragma: no cover
        raise NotImplementedError

    @property
    def solution_cell(self) -> str:
        """Source of a notebook cell that solves the exercise (used by the
        automated notebook runner, see scripts/run_notebooks.py)."""
        return self.check_call

    @property
    def label(self) -> str:
        return f"Exercise {self.id} — {self.title}"


# ..................................................................... code ..
@dataclass
class CodeExercise(Exercise):
    """A function the student writes, tested against a reference implementation.

    Parameters
    ----------
    reference : callable
        The reference implementation.  Its source is shown when the solution
        is revealed (unless ``solution_code`` is given).
    cases : list of dict
        Keyword arguments for each test call.  Keep them deterministic: pass a
        seeded ``np.random.default_rng`` object if randomness is needed.
    compare : callable, optional
        ``compare(student_output, reference_output) -> bool``.  Defaults to a
        tolerant numerical comparison that understands scalars, arrays, tuples,
        lists and dicts.
    bug_variants : dict
        ``{name: (buggy_callable, message)}``.  If the student's output matches
        a known-buggy implementation on the failing cases, ``message`` is shown
        as the diagnosis ("it looks like you updated with the raw reward...").
    signature : str
        Human-readable signature used in diagnostics.
    """

    reference: Callable | None = None
    cases: list[dict] = field(default_factory=list)
    compare: Callable | None = None
    rtol: float = 1e-5
    atol: float = 1e-7
    bug_variants: dict[str, tuple[Callable, str]] = field(default_factory=dict)
    solution_code: str | None = None
    signature: str = ""
    returns: str = ""
    diagnose: Callable | None = None   # diagnose(student_out, ref_out, case) -> message or None
    kind: str = "code"

    # ------------------------------------------------------------------ #
    def solution_source(self) -> str:
        if self.solution_code is not None:
            return textwrap.dedent(self.solution_code).strip("\n")
        return textwrap.dedent(inspect.getsource(self.reference)).strip("\n")

    def solution_markdown(self) -> str:
        return _code_block(self.solution_source())

    @property
    def solution_cell(self) -> str:
        return self.solution_source() + "\n\n" + self.check_call

    # ------------------------------------------------------------------ #
    def _run(self, fn: Callable, case: dict) -> Any:
        return fn(**copy.deepcopy(case))

    def grade(self, answer: Any) -> Grade:
        if not callable(answer):
            return Grade(
                False,
                "I expected a function, but got "
                f"`{type(answer).__name__}`.",
                diagnosis=(
                    "Pass the function *itself*, without calling it: "
                    f"`check('{self.id}', my_function)` rather than "
                    f"`check('{self.id}', my_function(...))`."
                ),
            )
        if not self.cases:
            return Grade(True, "No test cases defined (nothing to check).")

        failures: list[tuple[dict, Any, Any]] = []
        for i, case in enumerate(self.cases):
            try:
                out = self._run(answer, case)
            except Exception as exc:  # noqa: BLE001 - we want everything
                return Grade(
                    False,
                    f"Your function raised an error on test case {i + 1}.",
                    details=[self._describe_case(case)],
                    diagnosis=_diagnose_exception(exc, self),
                )
            if out is Ellipsis or (isinstance(out, (tuple, list)) and any(o is Ellipsis for o in out)):
                return Grade(False, "Your function returned the `...` placeholder.",
                             diagnosis="Replace every `...` in the function body with an expression "
                                       "(the last one is probably in the `return` line).")
            ref = self._run(self.reference, case)
            if not self._compare(out, ref, case):
                failures.append((case, out, ref))

        n = len(self.cases)
        if not failures:
            return Grade(True, f"All {n} test case{'s' if n > 1 else ''} passed.")

        case, out, ref = failures[0]
        details = [
            f"{len(failures)} of {n} test case{'s' if n > 1 else ''} failed.",
            "First failing case: " + self._describe_case(case),
            "Your output: " + _fmt(out),
            "Expected:    " + _fmt(ref),
        ]
        diagnosis = self._diagnose_mismatch(failures, answer)
        return Grade(False, "Not correct yet.", details, diagnosis)

    # ------------------------------------------------------------------ #
    def _compare(self, out: Any, ref: Any, case: dict | None = None) -> bool:
        if self.compare is not None:
            try:
                if _n_params(self.compare) >= 3:
                    return bool(self.compare(out, ref, case))
                return bool(self.compare(out, ref))
            except Exception:
                return False
        try:
            return bool(_default_compare(out, ref, self.rtol, self.atol))
        except Exception:
            return False

    def _describe_case(self, case: dict) -> str:
        parts = []
        for k, v in case.items():
            parts.append(f"{k}={_fmt(v, maxlen=40)}")
        return "`" + ", ".join(parts) + "`"

    def _diagnose_mismatch(self, failures, student_fn) -> str | None:
        # 1. known buggy variants
        for name, (buggy, message) in self.bug_variants.items():
            try:
                with np.errstate(all="ignore"):      # a buggy variant may overflow; that is not the student's problem
                    matches = all(
                        self._compare(self._run(student_fn, case), self._run(buggy, case), case)
                        for case, _, _ in failures
                    )
            except Exception:
                matches = False
            if matches:
                return message
        # 2. exercise-specific diagnosis
        case, out, ref = failures[0]
        if self.diagnose is not None:
            try:
                msg = self.diagnose(out, ref, case)
            except Exception:
                msg = None
            if msg:
                return msg
        # 3. generic numerical diagnostics on the first failure
        return _diagnose_numbers(out, ref, self.rtol, self.atol)


# .................................................................. numeric ..
@dataclass
class NumericExercise(Exercise):
    """A number (or small array) the student computes by hand or in code."""

    answer: Any = None
    atol: float = 1e-3
    rtol: float = 1e-3
    distractors: list[tuple[Any, str]] = field(default_factory=list)
    fmt: str = "{:.4g}"
    kind: str = "numeric"

    def solution_markdown(self) -> str:
        return f"**Answer:** `{_fmt(self.answer)}`\n"

    @property
    def solution_cell(self) -> str:
        return self.check_call

    def grade(self, answer: Any) -> Grade:
        if answer is None:
            return Grade(False, "You haven't entered an answer yet.",
                         diagnosis="Replace `None` with your number.")
        if callable(answer):
            return Grade(False, "I expected a number, but got a function.",
                         diagnosis="Call your function and pass me the *value* it returns.")
        try:
            a = np.asarray(answer, dtype=float)
        except Exception:
            return Grade(False, f"I expected a number, but got `{type(answer).__name__}`.")
        target = np.asarray(self.answer, dtype=float)
        if a.shape != target.shape:
            try:
                a = a.reshape(target.shape)
            except Exception:
                return Grade(False, f"Expected shape {target.shape}, got {a.shape}.")
        if np.allclose(a, target, atol=self.atol, rtol=self.rtol):
            return Grade(True, f"`{_fmt(answer)}` is correct.")
        # distractors (answers produced by specific, common mistakes)
        for value, message in self.distractors:
            v = np.asarray(value, dtype=float)
            if v.shape == a.shape and np.allclose(a, v, atol=self.atol, rtol=self.rtol):
                return Grade(False, f"`{_fmt(answer)}` is not correct.", diagnosis=message)
        diagnosis = _diagnose_numbers(a, target, self.rtol, self.atol, loose=False)
        return Grade(False, f"`{_fmt(answer)}` is not correct.", diagnosis=diagnosis)


# ................................................................... choice ..
@dataclass
class ChoiceExercise(Exercise):
    """Multiple-choice question.  ``correct`` may be one key or a set of keys."""

    question: str = ""
    options: dict[str, str] = field(default_factory=dict)
    correct: Any = ""
    feedback: dict[str, str] = field(default_factory=dict)  # per wrong option
    kind: str = "choice"

    def solution_markdown(self) -> str:
        keys = self._correct_set()
        lines = [f"**Correct answer:** {', '.join(sorted(keys))}\n"]
        for k in sorted(keys):
            lines.append(f"- **({k})** {self.options.get(k, '')}")
        return "\n".join(lines) + "\n"

    @property
    def solution_cell(self) -> str:
        return self.check_call

    def _correct_set(self) -> set[str]:
        if isinstance(self.correct, str):
            return {self.correct.lower()}
        return {str(k).lower() for k in self.correct}

    @staticmethod
    def _normalise(answer: Any) -> set[str]:
        if answer is None:
            return set()
        if isinstance(answer, str):
            s = answer.lower().replace("(", "").replace(")", "")
            s = s.replace(",", " ").replace("and", " ")
            return {tok for tok in s.split() if tok}
        try:
            return {str(x).lower().strip() for x in answer}
        except TypeError:
            return {str(answer).lower().strip()}

    def grade(self, answer: Any) -> Grade:
        chosen = self._normalise(answer)
        if not chosen:
            return Grade(False, "You haven't chosen an option yet.",
                         diagnosis="Pass the letter of your option, e.g. `check('%s', 'b')`." % self.id)
        unknown = chosen - set(k.lower() for k in self.options)
        if unknown:
            return Grade(False, f"`{', '.join(sorted(unknown))}` is not one of the options "
                                f"({', '.join(self.options)}).")
        correct = self._correct_set()
        if chosen == correct:
            return Grade(True, f"({', '.join(sorted(chosen))}) is correct.")
        wrong = sorted(chosen - correct)
        missing = sorted(correct - chosen)
        details = []
        for k in wrong:
            fb = self.feedback.get(k)
            details.append(f"({k}) {self.options[k]} — " + (fb if fb else "not correct."))
        if len(correct) > 1 and not wrong and missing:
            summary = "Partly right: your choice is correct but incomplete (more than one option applies)."
        elif len(correct) > 1:
            summary = "Not quite: this question has more than one correct option."
        else:
            summary = f"({', '.join(wrong)}) is not correct."
        return Grade(False, summary, details)


# ................................................................ free text ..
@dataclass
class FreeTextExercise(Exercise):
    """A short written answer, checked loosely for key concepts.

    ``concepts`` is a list of ``(label, [synonyms...])``: the answer "covers"
    a concept if any synonym occurs in it (case-insensitive substring match).
    The exercise passes when at least ``min_concepts`` (default: all) are
    covered.  The model answer is shown on success, so the student can
    compare their own reasoning with it.
    """

    question: str = ""
    model_answer: str = ""
    concepts: list[tuple[str, list[str]]] = field(default_factory=list)
    min_concepts: int | None = None
    min_chars: int = 40
    kind: str = "free"

    def solution_markdown(self) -> str:
        return "**Model answer:**\n\n" + self.model_answer.strip() + "\n"

    @property
    def solution_cell(self) -> str:
        return self.check_call

    def grade(self, answer: Any) -> Grade:
        if not isinstance(answer, str):
            return Grade(False, "Please answer with a string (your explanation in words).")
        text = " ".join(answer.lower().split())
        if len(text) < self.min_chars:
            return Grade(False, "Your answer is very short.",
                         diagnosis="Write at least a couple of full sentences; the check looks "
                                   "for the key ideas, so spell them out.")
        covered, missing = [], []
        for label, synonyms in self.concepts:
            if any(s.lower() in text for s in synonyms):
                covered.append(label)
            else:
                missing.append(label)
        need = len(self.concepts) if self.min_concepts is None else self.min_concepts
        details = []
        if covered:
            details.append("Ideas I found in your answer: " + ", ".join(covered) + ".")
        if len(covered) >= need:
            return Grade(True, "Your answer covers the key ideas (automatic keyword check — "
                               "compare it with the model answer below).", details)
        details.append("Ideas not (yet) mentioned: " + ", ".join(missing) + ".")
        return Grade(False, "Your answer is missing some key ideas.", details,
                     diagnosis="This is an automatic keyword check, so it can miss a correct "
                               "answer phrased differently. Try to name the missing ideas explicitly.")


# --------------------------------------------------------------------------- #
# Comparison / diagnosis helpers
# --------------------------------------------------------------------------- #
def _n_params(fn: Callable) -> int:
    try:
        return len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return 2


def _fmt(x: Any, maxlen: int = 80) -> str:
    try:
        if isinstance(x, np.ndarray):
            s = np.array2string(x, precision=4, suppress_small=True, threshold=12)
        elif isinstance(x, float):
            s = f"{x:.5g}"
        elif isinstance(x, dict):
            s = "{" + ", ".join(f"{k}: {_fmt(v, 30)}" for k, v in x.items()) + "}"
        elif isinstance(x, (list, tuple)):
            inner = ", ".join(_fmt(v, 30) for v in x)
            s = f"[{inner}]" if isinstance(x, list) else f"({inner})"
        elif isinstance(x, np.random.Generator):
            s = "rng"
        else:
            s = repr(x)
    except Exception:
        s = repr(x)
    s = " ".join(s.split())
    return s if len(s) <= maxlen else s[: maxlen - 3] + "..."


def _default_compare(out: Any, ref: Any, rtol: float, atol: float) -> bool:
    """Tolerant structural + numerical equality."""
    if ref is None:
        return out is None
    if out is None:
        return False
    if isinstance(ref, dict):
        if not isinstance(out, dict) or set(out) != set(ref):
            return False
        return all(_default_compare(out[k], ref[k], rtol, atol) for k in ref)
    if isinstance(ref, (tuple, list)) and not isinstance(ref, np.ndarray):
        if isinstance(out, np.ndarray) and all(np.isscalar(r) or np.ndim(r) == 0 for r in ref):
            out = list(out)
        if not isinstance(out, (tuple, list)) or len(out) != len(ref):
            return False
        return all(_default_compare(o, r, rtol, atol) for o, r in zip(out, ref))
    if isinstance(ref, (str, bool)) or isinstance(out, str):
        return out == ref
    try:
        o = np.asarray(out, dtype=float)
        r = np.asarray(ref, dtype=float)
    except Exception:
        try:
            return bool(np.all(out == ref))
        except Exception:
            return False
    if o.shape != r.shape:
        if o.size == r.size and o.size == 1:
            o = o.reshape(r.shape)
        else:
            return False
    return bool(np.allclose(o, r, rtol=rtol, atol=atol, equal_nan=True))


def _diagnose_exception(exc: Exception, ex: CodeExercise) -> str:
    name = type(exc).__name__
    msg = str(exc)
    if isinstance(exc, NotImplementedError):
        return ("Your function still contains the `raise NotImplementedError` line — replace it "
                "with your implementation.")
    if "ellipsis" in msg.lower() or "Ellipsis" in msg:
        return ("You still have `...` placeholders in your code. Replace *every* `...` with "
                "an expression.")
    if isinstance(exc, TypeError) and ("argument" in msg or "positional" in msg or "keyword" in msg):
        sig = f" The expected signature is `{ex.signature}`." if ex.signature else ""
        return f"The function signature doesn't match what the checker calls.{sig} ({msg})"
    if isinstance(exc, ValueError) and "broadcast" in msg:
        return ("Array shapes don't line up (" + msg + "). Check which variables are scalars "
                "and which are arrays, and whether you index the right element.")
    if isinstance(exc, IndexError):
        return ("An index is out of range (" + msg + "). Remember Python indexes from 0, and "
                "check loops that look one step ahead/behind.")
    if isinstance(exc, KeyError):
        return f"A dictionary key is missing ({msg}). Check the names you use."
    if isinstance(exc, ZeroDivisionError):
        return "You divided by zero somewhere — a sum or count that is 0 on the first trial?"
    if isinstance(exc, NameError):
        return f"{msg}. Did you define every variable you use, or import what you need?"
    tb = traceback.format_exception_only(type(exc), exc)[-1].strip()
    return f"{name}: {tb}"


def _diagnose_numbers(out: Any, ref: Any, rtol: float, atol: float, loose: bool = True) -> str | None:
    """Generic 'what might have gone wrong' for numerical mismatches."""
    if out is None:
        return "Your function returned `None` — did you forget the `return` statement?"
    if isinstance(ref, dict):
        if not isinstance(out, dict):
            return f"I expected a dictionary with keys {list(ref)}, but got `{type(out).__name__}`."
        missing = [k for k in ref if k not in out]
        extra = [k for k in out if k not in ref]
        if missing:
            return f"The returned dictionary is missing the key(s) {missing}."
        for k in ref:
            if not _default_compare(out[k], ref[k], rtol, atol):
                sub = _diagnose_numbers(out[k], ref[k], rtol, atol, loose)
                return f"The entry `{k!r}` differs from what I expected." + (f" {sub}" if sub else "")
        if extra:
            return f"Unexpected extra key(s) {extra} in the returned dictionary."
        return None
    if isinstance(ref, (tuple, list)) and isinstance(out, (tuple, list)):
        if len(out) != len(ref):
            return f"You returned {len(out)} values but I expected {len(ref)}."
        for i, (o_i, r_i) in enumerate(zip(out, ref)):
            if not _default_compare(o_i, r_i, rtol, atol):
                sub = _diagnose_numbers(o_i, r_i, rtol, atol, loose)
                return f"Returned value number {i + 1} differs from what I expected." + (f" {sub}" if sub else "")
        return None
    try:
        o = np.asarray(out, dtype=float)
        r = np.asarray(ref, dtype=float)
    except Exception:
        return (f"The returned object has type `{type(out).__name__}`, which I can't compare "
                f"with the expected `{type(ref).__name__}`.")
    if o.shape != r.shape:
        return (f"Shape mismatch: you returned shape {o.shape} but I expected {r.shape}. "
                "Are you returning the whole history instead of the final value (or vice versa), "
                "or the wrong element?")
    if np.any(~np.isfinite(o)) and np.all(np.isfinite(r)):
        return ("Your output contains NaN or inf. Typical causes: log(0), division by zero, or "
                "exp() of a very large number (use a numerically stable form).")
    if r.size == 0:
        return None
    if np.allclose(o, -r, rtol=rtol, atol=atol) and np.any(r != 0):
        return "Your result has the *opposite sign*. Check the direction of a subtraction (e.g. reward − value, not value − reward)."
    if loose and np.allclose(o, r, rtol=5e-2, atol=1e-2):
        return ("You are close but not exact. Common causes: wrong initial value, updating in the "
                "wrong order (e.g. using the new value where the old one is needed), or rounding.")
    if r.ndim == 1 and r.size > 2:
        if np.allclose(o[1:], r[:-1], rtol=rtol, atol=atol):
            return ("Your sequence is shifted by one trial (off-by-one): you store the value "
                    "*after* the update where the value *before* the update is expected, or vice versa.")
        if np.allclose(o[:-1], r[1:], rtol=rtol, atol=atol):
            return ("Your sequence is shifted by one trial (off-by-one): check whether the stored "
                    "value should be the one before or after the update.")
    diff = o - r
    if np.allclose(diff, diff.flat[0], rtol=1e-6, atol=1e-8) and not np.isclose(diff.flat[0], 0):
        return (f"Your result is off by a constant ({diff.flat[0]:.4g}) everywhere — a missing "
                "or extra term, or a wrong initial value.")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = o / r
    ok = np.isfinite(ratio) & (r != 0)
    if ok.sum() >= 1 and np.allclose(ratio[ok], ratio[ok].flat[0], rtol=1e-6) and not np.isclose(ratio[ok].flat[0], 1):
        k = ratio[ok].flat[0]
        hint_ = ""
        if np.isclose(k, 100):
            hint_ = " (a percentage instead of a proportion?)"
        elif np.isclose(k, 0.01):
            hint_ = " (a proportion instead of a percentage?)"
        return f"Your result is the expected one multiplied by {k:.4g}{hint_} — a missing or extra factor."
    if r.size == 1 and r.item() != 0:
        rv, ov = float(r), float(o)
        if ov > 0 and np.isclose(np.log(ov), rv, rtol=rtol, atol=atol):
            return "Your answer is exp() of the expected value — did you forget to take the logarithm?"
        if rv > 0 and np.isclose(np.exp(ov), rv, rtol=rtol, atol=atol):
            return "Your answer is the *log* of the expected value — the question asks for the quantity itself."
        if ov != 0 and np.isclose(1 / ov, rv, rtol=rtol, atol=atol):
            return "Your answer is the reciprocal of the expected value."
    return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _get(ex_id: str) -> Exercise:
    from . import exercises  # local import: avoids a circular import at load time

    return exercises.get(ex_id)


def check(ex_id: str, answer: Any = None, *, reveal: bool = False) -> None:
    """Check an answer to exercise ``ex_id`` and print feedback.

    (Whether the exercise passed is recorded in ``PROGRESS.passed[ex_id]``.)"""
    ex = _get(ex_id)
    if reveal:
        _reveal(ex)
        return None

    grade = ex.grade(answer)
    n = PROGRESS.attempts.get(ex_id, 0) + 1
    PROGRESS.attempts[ex_id] = n

    if grade.passed:
        PROGRESS.passed[ex_id] = True
        PROGRESS.save()
        md = f"{PASS} **{ex.label}: correct!** {grade.summary}\n"
        for d in grade.details:
            md += f"\n- {d}"
        if grade.details:
            md += "\n"
        if ex.explanation:
            md += f"\n{ex.explanation.strip()}\n"
        if isinstance(ex, FreeTextExercise):
            md += "\n" + ex.solution_markdown()
        if n > 1:
            md += f"\n*(solved after {n} attempts)*\n"
        _show(md)
        return None

    # ---- failed ---------------------------------------------------------
    PROGRESS.passed.setdefault(ex_id, False)
    md = f"{FAIL} **{ex.label}: not yet.** {grade.summary}\n"
    for d in grade.details:
        md += f"\n- {d}"
    if grade.details:
        md += "\n"
    if grade.diagnosis:
        md += f"\n{DIAG} **What might have gone wrong:** {grade.diagnosis}\n"

    n_hints = len(ex.hints)
    if n <= n_hints:
        PROGRESS.hints_shown[ex_id] = max(PROGRESS.hints_shown.get(ex_id, 0), n)
        md += f"\n{HINT} **Hint {n} of {n_hints}:** {ex.hints[n - 1].strip()}\n"
        if n > 1:
            md += f"\n*(run `hint('{ex_id}')` to see all hints so far)*\n"
        else:
            md += f"\n*(attempt {n}; after {n_hints} hints the solution is revealed)*\n"
        PROGRESS.save()
        _show(md)
    else:
        PROGRESS.save()
        _show(md)
        _reveal(ex, after_failure=True)

    if MODE == "solution":
        raise AssertionError(f"Exercise {ex_id} failed in solution mode: {grade.summary} "
                             f"{grade.details} {grade.diagnosis}")
    return None


def _reveal(ex: Exercise, after_failure: bool = False) -> None:
    PROGRESS.revealed[ex.id] = True
    PROGRESS.save()
    if after_failure:
        head = (f"{BOOK} **You have used all the hints for {ex.label}, so here is the worked "
                "solution.** Read it line by line and compare it with your attempt.\n")
    else:
        head = f"{BOOK} **Solution to {ex.label}**\n"
    md = head + "\n" + ex.solution_markdown()
    if ex.explanation:
        md += f"\n**Why this works:** {ex.explanation.strip()}\n"
    if ex.common_mistakes:
        md += f"\n**What usually goes wrong:** {ex.common_mistakes.strip()}\n"
    _show(md)


def reveal(ex_id: str) -> None:
    """Show the worked solution of an exercise (you can always do this, but try first!)."""
    _reveal(_get(ex_id))


def hint(ex_id: str) -> None:
    """Show the next hint for an exercise without making an attempt."""
    ex = _get(ex_id)
    if not ex.hints:
        _show(f"{HINT} There are no hints for {ex.label}.")
        return
    shown = PROGRESS.hints_shown.get(ex_id, 0)
    shown = min(shown + 1, len(ex.hints))
    PROGRESS.hints_shown[ex_id] = shown
    PROGRESS.save()
    md = f"{HINT} **Hints for {ex.label}** ({shown} of {len(ex.hints)} unlocked)\n"
    for i in range(shown):
        md += f"\n{i + 1}. {ex.hints[i].strip()}"
    if shown < len(ex.hints):
        md += f"\n\n*(run `hint('{ex_id}')` again for the next one)*"
    _show(md + "\n")


def reset(ex_id: str) -> None:
    """Forget your attempts and hints for an exercise."""
    PROGRESS.reset(ex_id)
    _show(f"Progress for exercise {ex_id} has been reset.")


def progress(lesson: int | str | None = None) -> None:
    """Print a table of the exercises you have attempted and passed."""
    from . import exercises

    ids = sorted(set(PROGRESS.attempts) | set(PROGRESS.passed) | set(PROGRESS.hints_shown),
                 key=lambda s: tuple(int(p) if p.isdigit() else p for p in s.split(".")))
    if lesson is not None:
        ids = [i for i in ids if i.split(".")[0] == str(int(lesson))]
    if not ids:
        _show("No exercises attempted yet.")
        return
    rows = ["| Exercise | Title | Attempts | Status |", "|---|---|---:|---|"]
    for i in ids:
        try:
            title = exercises.get(i).title
        except Exception:
            title = "?"
        status = PASS + " passed" if PROGRESS.passed.get(i) else FAIL + " not yet"
        if PROGRESS.revealed.get(i):
            status += " (solution revealed)"
        rows.append(f"| {i} | {title} | {PROGRESS.attempts.get(i, 0)} | {status} |")
    n_pass = sum(bool(PROGRESS.passed.get(i)) for i in ids)
    _show("\n".join(rows) + f"\n\n**{n_pass} of {len(ids)} attempted exercises passed.**\n")


def use(ex_id: str, obj: Any) -> Any:
    """Return ``obj`` if it solves exercise ``ex_id``, else the reference implementation.

    Later cells in a lesson use this so that you can keep going even if an
    exercise is still unfinished: your own (correct) code is used whenever it
    passes, otherwise the course's reference implementation is substituted and
    a note is printed.
    """
    ex = _get(ex_id)
    if not isinstance(ex, CodeExercise):
        raise TypeError(f"use() only works with coding exercises; {ex_id} is a {ex.kind} exercise")
    try:
        ok = ex.grade(obj).passed
    except Exception:
        ok = False
    if ok:
        return obj
    name = getattr(ex.reference, "__name__", "reference")
    print(f"[rlcourse] Exercise {ex_id} is not passing yet, so the reference implementation "
          f"of `{name}` is used from here on. Come back and finish the exercise later!")
    return ex.reference


def self_test(ex_id: str | None = None, lesson: int | None = None, verbose: bool = True) -> bool:
    """Check that every reference solution passes its own exercise (used by tests)."""
    from . import exercises

    if ex_id is not None:
        exs = [exercises.get(ex_id)]
    else:
        exs = exercises.all_exercises(lesson)
    ok_all = True
    for ex in exs:
        if isinstance(ex, CodeExercise):
            g = ex.grade(ex.reference)
        elif isinstance(ex, NumericExercise):
            g = ex.grade(ex.answer)
        elif isinstance(ex, ChoiceExercise):
            g = ex.grade(sorted(ex._correct_set()))
        elif isinstance(ex, FreeTextExercise):
            g = ex.grade(ex.model_answer)
        else:  # pragma: no cover
            continue
        ok_all &= g.passed
        if verbose or not g.passed:
            print(f"{PASS if g.passed else FAIL} {ex.id:>5s} {ex.kind:8s} {ex.title}"
                  + ("" if g.passed else f"  <-- {g.summary} {g.details} {g.diagnosis}"))
    return ok_all


__all__ = [
    "check", "hint", "reveal", "reset", "progress", "use", "self_test",
    "Exercise", "CodeExercise", "NumericExercise", "ChoiceExercise", "FreeTextExercise", "Grade",
]
