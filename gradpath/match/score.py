"""Fit scoring. Pure functions only — no database, no network.

The central decision here is mean-of-top-3. Averaging every work regresses
prolific researchers to the mean and buries the professor with three papers
precisely on your topic. A single maximum is the opposite failure: one
coincidentally-worded abstract promotes someone unrelated. Mean-of-top-3
rewards sustained overlap while staying robust to a lucky match.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TOP_K = 3


@dataclass(frozen=True)
class PersonScore:
    score: float
    sparse: bool
    top_work_ids: list[int]


def score_person(similarities: dict[int, float], k: int = TOP_K) -> PersonScore:
    """Score one person from their work-level similarities.

    `similarities` maps work row id to cosine similarity. People with fewer
    than `k` works are scored on what exists, with no padding, and flagged
    sparse so callers can surface that rather than silently ranking them low.
    """
    if not similarities:
        raise ValueError("cannot score a person with no works")
    ranked = sorted(similarities.items(), key=lambda pair: pair[1], reverse=True)
    top = ranked[:k]
    return PersonScore(
        score=float(sum(value for _, value in top) / len(top)),
        sparse=len(ranked) < k,
        top_work_ids=[work_id for work_id, _ in top],
    )


def cosine_matrix(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Cosine similarity of every row in `matrix` against `vector`.

    One matrix multiply over the whole corpus. Scoring must never loop over
    people in Python — the scale test in tests/test_scale.py guards this.
    """
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    denominator = np.where(matrix_norms == 0.0, 1.0, matrix_norms) * vector_norm
    return (matrix @ vector) / denominator
