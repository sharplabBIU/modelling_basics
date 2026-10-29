"""Exercises of lesson 0: three tiny demonstrations of how the checker works."""
from __future__ import annotations

import numpy as np

from ..checker import ChoiceExercise, CodeExercise, NumericExercise


def proportion_rewarded(rewards):
    """Return the proportion of trials on which a reward (1) was received."""
    rewards = np.asarray(rewards, dtype=float)
    return float(np.sum(rewards) / len(rewards))


EXERCISES = [
    ChoiceExercise(
        id="0.1", title="What is a computational model of behaviour?",
        question="Which statement best describes a computational model of behaviour as used in this course?",
        options={
            "a": "A statistical test that tells us whether two groups differ.",
            "b": "A set of equations that, given a participant's experiences, generates predictions (or probabilities) for what the participant will do next.",
            "c": "A neural network trained to classify participants.",
            "d": "A verbal theory about why people behave the way they do.",
        },
        correct="b",
        feedback={
            "a": "A statistical test compares data; it does not generate behaviour. Models in this course can be *simulated* to produce artificial behaviour.",
            "c": "Neural networks can be models of behaviour, but what defines a model here is that it maps experiences to predicted choices through interpretable equations.",
            "d": "A verbal theory is the starting point; a *computational* model makes it precise enough to simulate and fit.",
        },
        hints=["Think about what you can *do* with a model: simulate it, and compare its predictions with data.",
               "The key word is *generates*: a model is a recipe for producing behaviour from experience."],
        explanation="A computational model is a precise, quantitative recipe: experiences in, predicted behaviour out. "
                    "Because it is precise we can simulate it (lesson 1), fit it to data (lesson 5) and compare it with "
                    "rival recipes (lesson 7).",
        check_call='check("0.1", "b")',
    ),
    NumericExercise(
        id="0.2", title="A first number",
        answer=0.75, atol=1e-6,
        distractors=[(3, "That is the *number* of rewards, not the proportion: divide by the number of trials."),
                     (0.25, "That is the proportion of *unrewarded* trials."),
                     (75, "That is a percentage; the check expects a proportion between 0 and 1.")],
        hints=["A proportion is a count divided by a total.",
               "Three of the four trials were rewarded: 3 / 4."],
        explanation="3 rewards out of 4 trials gives a proportion of 0.75. Numeric exercises accept any answer within a "
                    "small tolerance, so you can compute them in code or by hand.",
        check_call='check("0.2", 0.75)',
    ),
    CodeExercise(
        id="0.3", title="A first function",
        reference=proportion_rewarded,
        signature="proportion_rewarded(rewards)",
        cases=[{"rewards": [1, 0, 1, 1]}, {"rewards": [0, 0, 0]}, {"rewards": np.array([1.0, 1.0, 0.0, 0.0, 1.0])}],
        bug_variants={
            "count": (lambda rewards: float(np.sum(rewards)), "You returned the *number* of rewards; divide by the number of trials."),
            "complement": (lambda rewards: 1 - float(np.mean(rewards)), "You computed the proportion of *unrewarded* trials."),
        },
        hints=["The proportion is the sum of the rewards divided by how many there are.",
               "`np.sum(rewards)` and `len(rewards)` are all you need (or simply `np.mean(rewards)`).",
               "`return np.sum(rewards) / len(rewards)`"],
        explanation="Because rewards are coded 0/1, their mean *is* the proportion of rewarded trials. This trick "
                    "(averaging indicator variables) appears everywhere in the course.",
        common_mistakes="Returning the count instead of the proportion, or leaving the `...` placeholder in place.",
        check_call='check("0.3", proportion_rewarded)',
    ),
]
