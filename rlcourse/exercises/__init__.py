"""Registry of exercises.

Each lesson's exercises live in ``rlcourse/exercises/lessonNN.py`` as a module
level list ``EXERCISES``.  Exercise ids are ``"<lesson>.<number>"`` (``"3.1"``
is the first exercise of lesson 3).
"""
from __future__ import annotations

import importlib
import pkgutil

from ..checker import Exercise

_CACHE: dict[str, Exercise] = {}


def _module_name(lesson: int | str) -> str:
    return f"rlcourse.exercises.lesson{int(lesson):02d}"


def _load_lesson(lesson: int | str) -> list[Exercise]:
    mod = importlib.import_module(_module_name(lesson))
    exs = list(getattr(mod, "EXERCISES"))
    for ex in exs:
        _CACHE[ex.id] = ex
    return exs


def get(ex_id: str) -> Exercise:
    """Return the exercise with id ``ex_id`` (e.g. ``"3.1"``)."""
    ex_id = str(ex_id).strip()
    if ex_id in _CACHE:
        return _CACHE[ex_id]
    lesson = ex_id.split(".")[0]
    if not lesson.isdigit():
        raise KeyError(f"'{ex_id}' is not a valid exercise id (expected e.g. '3.1')")
    try:
        _load_lesson(lesson)
    except ModuleNotFoundError as exc:
        raise KeyError(f"No exercises are defined for lesson {lesson}") from exc
    if ex_id not in _CACHE:
        raise KeyError(f"Exercise '{ex_id}' does not exist. Check the id in the notebook.")
    return _CACHE[ex_id]


def available_lessons() -> list[int]:
    lessons = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("lesson") and info.name[6:].isdigit():
            lessons.append(int(info.name[6:]))
    return sorted(lessons)


def all_exercises(lesson: int | str | None = None) -> list[Exercise]:
    """All exercises of one lesson, or of the whole course."""
    if lesson is not None:
        return _load_lesson(lesson)
    out: list[Exercise] = []
    for les in available_lessons():
        out.extend(_load_lesson(les))
    return out
