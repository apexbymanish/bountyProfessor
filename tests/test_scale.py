"""Guards against an accidental per-person Python loop over the corpus."""
import time

import numpy as np

from gradpath.db import connect, migrate
from gradpath.match.embed import pack, run_match

DIM = 32
TIME_BUDGET_SECONDS = 20.0


def test_ten_thousand_works_score_within_budget(tmp_path):
    conn = connect(tmp_path / "s.db")
    migrate(conn)
    conn.execute("INSERT INTO institutions (id, name, added_at) VALUES ('i','I','2026-01-01')")
    rng = np.random.default_rng(0)
    for index in range(10_000):
        work_id = conn.execute(
            "INSERT INTO works (openalex_work_id, title, abstract, year) "
            "VALUES (?, 't', 'a', 2024)", (f"W{index}",)
        ).lastrowid
        conn.execute(
            "INSERT INTO embeddings (work_id, model, dim, vector, source, computed_at) "
            "VALUES (?, 'm', ?, ?, 'abstract', '2026-01-01')",
            (work_id, DIM, pack(rng.random(DIM, dtype=np.float32))),
        )
        person_id = conn.execute(
            "INSERT INTO people (institution_id, name, openalex_author_id) "
            "VALUES ('i', ?, ?)", (f"P{index}", f"A{index}")
        ).lastrowid
        conn.execute(
            "INSERT INTO authorships (person_id, work_id, position) VALUES (?, ?, 'last')",
            (person_id, work_id),
        )
    conn.commit()
    started = time.monotonic()
    scored = run_match(conn, "default", rng.random(DIM, dtype=np.float32), "m")
    elapsed = time.monotonic() - started
    assert scored == 10_000
    assert elapsed < TIME_BUDGET_SECONDS, f"scoring took {elapsed:.1f}s"
