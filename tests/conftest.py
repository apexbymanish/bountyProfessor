"""Shared test fixtures and fakes.

`FakeEmbedder` lives here (not in tests/test_embed.py) so other test modules —
including a later task's test that imports it directly from conftest — can
reuse it without relying on `tests` being an importable package.
"""
from __future__ import annotations

import numpy as np

DIM = 4


class FakeEmbedder:
    """Deterministic stand-in: hashes text into a small vector. No torch in tests."""

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                out[row, hash(token) % DIM] += 1.0
        return out
