"""Tests of the exercise checker's feedback flow.

Run with ``pytest`` or ``python tests/test_checker.py``.
"""
import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

os.environ["RLCOURSE_MODE"] = ""  # interactive behaviour, but ...
os.environ["RLCOURSE_PROGRESS_FILE"] = os.path.join(tempfile.mkdtemp(), "progress.json")  # ... in a temp file
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from rlcourse import checker  # noqa: E402
from rlcourse.checker import ChoiceExercise, CodeExercise, FreeTextExercise, NumericExercise  # noqa: E402


def _ref(value, reward, alpha):
    return value + alpha * (reward - value)


def _make_code_exercise():
    return CodeExercise(
        id="99.1", title="demo update", hints=["h1", "h2", "h3"], explanation="because", common_mistakes="sign",
        reference=_ref, cases=[{"value": 0.5, "reward": 1.0, "alpha": 0.1}, {"value": 0.2, "reward": 0.0, "alpha": 0.5}],
        bug_variants={"raw": (lambda value, reward, alpha: value + alpha * reward, "You used the raw reward")},
        check_call='check("99.1", f)', signature="f(value, reward, alpha)",
    )


def _install(ex):
    checker.PROGRESS.reset(ex.id)
    from rlcourse import exercises
    exercises._CACHE[ex.id] = ex


def _run(fn, *args, **kw):
    """Call fn capturing stdout; returns (passed-state of the exercise, printed output)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = fn(*args, **kw)
    if fn is checker.check:
        result = bool(checker.PROGRESS.passed.get(args[0], False))
    return result, buf.getvalue()


def test_code_exercise_flow():
    ex = _make_code_exercise()
    _install(ex)
    # 1st wrong attempt: negated result -> sign diagnosis + hint 1
    ok, out = _run(checker.check, "99.1", lambda value, reward, alpha: -(value + alpha * (reward - value)))
    assert not ok and "opposite sign" in out and "Hint 1 of 3" in out, out
    # 2nd wrong attempt: a known buggy variant -> its message + hint 2
    ok, out = _run(checker.check, "99.1", lambda value, reward, alpha: value + alpha * reward)
    assert not ok and "raw reward" in out and "Hint 2 of 3" in out, out
    # 3rd: placeholder left in -> ellipsis diagnosis + hint 3
    ok, out = _run(checker.check, "99.1", lambda value, reward, alpha: value + alpha * (... - value))
    assert not ok and "placeholders" in out and "Hint 3 of 3" in out
    # 4th wrong attempt: solution revealed with explanation and common mistakes
    ok, out = _run(checker.check, "99.1", lambda value, reward, alpha: 0.0)
    assert not ok and "worked" in out and "def _ref" in out and "because" in out and "sign" in out
    # correct answer
    ok, out = _run(checker.check, "99.1", _ref)
    assert ok and "correct" in out and "because" in out
    assert checker.PROGRESS.passed["99.1"]
    # use() returns the student's function when it passes, the reference otherwise
    assert checker.use("99.1", _ref) is _ref
    _, out = _run(checker.use, "99.1", lambda value, reward, alpha: 0.0)
    assert "reference implementation" in out


def test_passing_a_value_instead_of_a_function():
    ex = _make_code_exercise()
    _install(ex)
    ok, out = _run(checker.check, "99.1", 0.55)
    assert not ok and "without calling it" in out


def test_numeric_exercise():
    ex = NumericExercise(id="99.2", title="num", answer=0.25, hints=["a", "b"], distractors=[(0.5, "forgot alpha")],
                         check_call='check("99.2", 0.25)')
    _install(ex)
    ok, out = _run(checker.check, "99.2", 0.5)
    assert not ok and "forgot alpha" in out
    ok, out = _run(checker.check, "99.2", -0.25)
    assert not ok and "opposite sign" in out
    ok, out = _run(checker.check, "99.2", 0.25 + 1e-5)
    assert ok


def test_choice_exercise():
    ex = ChoiceExercise(id="99.3", title="mc", options={"a": "A", "b": "B", "c": "C"}, correct="b",
                        feedback={"a": "A is wrong because"}, hints=["h"], check_call='check("99.3", "b")')
    _install(ex)
    ok, out = _run(checker.check, "99.3", "a")
    assert not ok and "A is wrong because" in out
    ok, out = _run(checker.check, "99.3", "(B)")
    assert ok
    multi = ChoiceExercise(id="99.4", title="multi", options={"a": "A", "b": "B", "c": "C"}, correct={"a", "c"},
                           hints=[], check_call='check("99.4", ["a", "c"])')
    _install(multi)
    ok, out = _run(checker.check, "99.4", "a")
    assert not ok and "incomplete" in out
    ok, out = _run(checker.check, "99.4", ["c", "a"])
    assert ok


def test_free_text_exercise():
    ex = FreeTextExercise(id="99.5", title="free", model_answer="model", hints=["h"],
                          concepts=[("prediction error", ["prediction error", "delta"]), ("learning rate", ["learning rate", "alpha"])],
                          check_call='check("99.5", "...")')
    _install(ex)
    ok, out = _run(checker.check, "99.5", "The delta drives learning and nothing else matters, honestly.")
    assert not ok and "learning rate" in out
    ok, out = _run(checker.check, "99.5", "The prediction error is scaled by the learning rate alpha to update the value.")
    assert ok and "model" in out


def test_diagnostics_shapes_and_none():
    ex = CodeExercise(id="99.6", title="arr", reference=lambda x: np.cumsum(x), cases=[{"x": np.arange(5.0)}],
                      hints=["h"], check_call="")
    _install(ex)
    ok, out = _run(checker.check, "99.6", lambda x: None)
    assert "forget the `return`" in out
    ok, out = _run(checker.check, "99.6", lambda x: np.cumsum(x)[:-1])
    assert "Shape mismatch" in out
    ok, out = _run(checker.check, "99.6", lambda x: np.r_[0, np.cumsum(x)[:-1]])
    assert "off-by-one" in out


def test_hint_and_progress():
    ex = _make_code_exercise()
    _install(ex)
    _, out = _run(checker.hint, "99.1")
    assert "1 of 3" in out
    _, out = _run(checker.progress)
    assert "99.1" in out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
