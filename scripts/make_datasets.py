"""Regenerate the synthetic data sets in ``data/`` (deterministic seeds)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rlcourse import datasets  # noqa: E402

if __name__ == "__main__":
    datasets.make_all()
