"""Embedding cache and vectorised matching.

The product's central promise is that the field decides who surfaces, not a
row count: `run_match` scores every person in the database with at least one
embedded work, never a top-N slice. That is only affordable because work
vectors are computed once, cached per work, and reused across runs and
profiles, and because scoring the whole corpus is one matrix multiply rather
than a Python loop over people.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Protocol

import numpy as np

from gradpath.match.score import cosine_matrix, score_person
from gradpath.models import Profile
from gradpath.util import now_iso

SEED_PAPER_WEIGHT = 2.0
TEXT_WEIGHT = 1.0


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


def load_embedder(model_name: str) -> Embedder:
    """Import sentence-transformers lazily so `init` and `discover` never load torch."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def pack(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


def embed_pending_works(
    conn: sqlite3.Connection, embedder: Embedder, model: str, batch_size: int
) -> int:
    """Embed works that have no cached vector for this model. Returns how many were added.

    Abstracts are missing from roughly 40% of OpenAlex records, so the title-only
    fallback is a normal path — it is recorded in embeddings.source so output can
    flag it rather than silently ranking those works low.
    """
    rows = conn.execute(
        """
        SELECT w.id, w.title, w.abstract FROM works w
        LEFT JOIN embeddings e ON e.work_id = w.id AND e.model = ?
        WHERE e.work_id IS NULL
        """,
        (model,),
    ).fetchall()
    if not rows:
        return 0

    added = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        texts = [(row["abstract"] or row["title"] or "") for row in chunk]
        sources = ["abstract" if row["abstract"] else "title_only" for row in chunk]
        vectors = embedder.encode(texts)
        with conn:
            for row, vector, source in zip(chunk, vectors, sources):
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings "
                    "(work_id, model, dim, vector, source, computed_at) VALUES (?,?,?,?,?,?)",
                    (row["id"], model, int(len(vector)), pack(vector), source, now_iso()),
                )
                added += 1
    return added


def build_profile_vector(
    profile: Profile, seed_abstracts: list[str], embedder: Embedder
) -> np.ndarray:
    """Weighted mean of interest components. Seed abstracts count double.

    A real abstract describes a research area far more precisely than a
    self-written paragraph, which is why it carries twice the weight.
    """
    texts: list[str] = []
    weights: list[float] = []
    if profile.interests.strip():
        texts.append(profile.interests)
        weights.append(TEXT_WEIGHT)
    if profile.keywords:
        texts.append(" ".join(profile.keywords))
        weights.append(TEXT_WEIGHT)
    for abstract in seed_abstracts:
        if abstract and abstract.strip():
            texts.append(abstract)
            weights.append(SEED_PAPER_WEIGHT)
    if not texts:
        raise ValueError(
            "no interest inputs resolved — fill in interests, keywords or seed_papers "
            "in profile.yaml before running match"
        )
    matrix = np.asarray(embedder.encode(texts), dtype=np.float32)
    weight_vector = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    return (matrix * weight_vector).sum(axis=0) / weight_vector.sum()


def run_match(
    conn: sqlite3.Connection, profile_id: str, profile_vector: np.ndarray, model: str
) -> int:
    """Score every person in the database. No cap, by design. Returns people scored."""
    rows = conn.execute(
        "SELECT work_id, dim, vector FROM embeddings WHERE model = ? ORDER BY work_id",
        (model,),
    ).fetchall()
    if not rows:
        return 0

    dim = rows[0]["dim"]
    work_ids = np.array([row["work_id"] for row in rows], dtype=np.int64)
    matrix = np.vstack([unpack(row["vector"], dim) for row in rows])

    # One matrix multiply over the whole corpus — never a per-person loop.
    similarities = cosine_matrix(matrix, np.asarray(profile_vector, dtype=np.float32))
    by_work = dict(zip(work_ids.tolist(), similarities.tolist()))

    grouped: dict[int, dict[int, float]] = {}
    for row in conn.execute("SELECT person_id, work_id FROM authorships"):
        similarity = by_work.get(row["work_id"])
        if similarity is not None:
            grouped.setdefault(row["person_id"], {})[row["work_id"]] = similarity

    stamp = now_iso()
    with conn:
        for person_id, similarity_map in grouped.items():
            result = score_person(similarity_map)
            conn.execute(
                """
                INSERT INTO matches
                    (profile_id, person_id, stage1_score, top_work_ids, sparse, computed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id, person_id) DO UPDATE SET
                    stage1_score = excluded.stage1_score,
                    top_work_ids = excluded.top_work_ids,
                    sparse = excluded.sparse,
                    computed_at = excluded.computed_at
                """,
                (profile_id, person_id, result.score, json.dumps(result.top_work_ids),
                 int(result.sparse), stamp),
            )
    return len(grouped)
