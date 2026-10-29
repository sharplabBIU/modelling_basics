"""rlcourse: teaching package for the *Reinforcement Learning for Computational
Cognitive Science* tutorial course.

Sub-modules
-----------
checker   : the exercise checker (``check``, ``hint``, ``progress``, ``use``)
tasks     : simulated experimental tasks (bandits, reversal, two-step, MDPs)
models    : reference implementations of every model taught in the course
fitting   : maximum likelihood, MAP, EM, MCMC and model-comparison helpers
plotting  : a consistent plotting style and a few convenience plots
datasets  : synthetic "participant" datasets used in the later lessons
"""
from importlib.metadata import PackageNotFoundError, version as _version

try:  # pragma: no cover - version is cosmetic
    __version__ = _version("rlcourse")
except PackageNotFoundError:  # not installed with pip, running from the repo
    __version__ = "1.0.0"

from .checker import check, hint, progress, reset, use, reveal  # noqa: E402,F401
from .plotting import setup_style  # noqa: E402,F401

__all__ = ["check", "hint", "progress", "reset", "use", "reveal", "setup_style", "__version__"]
