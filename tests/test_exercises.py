"""Every registered reference solution must pass its own exercise, and every
exercise must have hints, an explanation and a check call."""
import os
import sys

os.environ["RLCOURSE_MODE"] = "stub"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rlcourse import checker, exercises  # noqa: E402


def test_reference_solutions_pass():
    assert checker.self_test(verbose=False)


def test_exercise_metadata():
    for ex in exercises.all_exercises():
        assert ex.hints, f"{ex.id} has no hints"
        assert ex.explanation, f"{ex.id} has no explanation"
        assert ex.check_call, f"{ex.id} has no check_call"
        assert ex.title, f"{ex.id} has no title"
        assert ex.id in ex.check_call, f"{ex.id}: check_call does not mention the id"


def test_ids_are_unique_and_well_formed():
    seen = set()
    for lesson in exercises.available_lessons():
        for ex in exercises.all_exercises(lesson):
            assert ex.id not in seen, f"duplicate id {ex.id}"
            seen.add(ex.id)
            assert ex.id.split(".")[0] == str(lesson), f"{ex.id} registered in lesson {lesson}"


if __name__ == "__main__":
    test_reference_solutions_pass()
    test_exercise_metadata()
    test_ids_are_unique_and_well_formed()
    print("ok")
