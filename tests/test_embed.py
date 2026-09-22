import numpy as np
import pytest

from gradpath.db import connect, migrate
from gradpath.match.embed import (
    build_profile_vector, embed_pending_works, pack, run_match, unpack,
)
from gradpath.models import Profile
from tests.conftest import FakeEmbedder

MODEL = "test-model"


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO institutions (id, name, added_at) VALUES ('i', 'Inst', '2026-01-01')"
    )
    return conn


def _add_work(conn, wid, title, abstract):
    cur = conn.execute(
        "INSERT INTO works (openalex_work_id, title, abstract, year) VALUES (?, ?, ?, 2024)",
        (wid, title, abstract),
    )
    return cur.lastrowid


def _add_person(conn, name, author_id):
    cur = conn.execute(
        "INSERT INTO people (institution_id, name, openalex_author_id) VALUES ('i', ?, ?)",
        (name, author_id),
    )
    return cur.lastrowid


def test_pack_unpack_roundtrips():
    vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    assert np.allclose(unpack(pack(vector), 4), vector)


def test_embeds_only_works_without_a_cached_vector(db):
    _add_work(db, "W1", "quantized inference", "we quantize models")
    db.commit()
    assert embed_pending_works(db, FakeEmbedder(), MODEL, batch_size=8) == 1
    assert embed_pending_works(db, FakeEmbedder(), MODEL, batch_size=8) == 0


def test_missing_abstract_falls_back_to_title_and_is_recorded(db):
    _add_work(db, "W1", "quantized inference", None)
    db.commit()
    embed_pending_works(db, FakeEmbedder(), MODEL, batch_size=8)
    row = db.execute("SELECT source FROM embeddings").fetchone()
    assert row["source"] == "title_only"


def test_abstract_present_is_recorded_as_abstract(db):
    _add_work(db, "W1", "t", "a real abstract")
    db.commit()
    embed_pending_works(db, FakeEmbedder(), MODEL, batch_size=8)
    assert db.execute("SELECT source FROM embeddings").fetchone()["source"] == "abstract"


def test_run_match_scores_every_person_with_no_cap(db):
    for index in range(50):
        work_id = _add_work(db, f"W{index}", f"title {index}", f"abstract {index}")
        person_id = _add_person(db, f"P{index}", f"A{index}")
        db.execute(
            "INSERT INTO authorships (person_id, work_id, position) VALUES (?, ?, 'last')",
            (person_id, work_id),
        )
    db.commit()
    embed_pending_works(db, FakeEmbedder(), MODEL, batch_size=16)
    vector = FakeEmbedder().encode(["abstract 3"])[0]
    assert run_match(db, "default", vector, MODEL) == 50
    assert db.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 50


def test_run_match_marks_sparse_people(db):
    work_id = _add_work(db, "W1", "t", "a")
    person_id = _add_person(db, "P", "A1")
    db.execute(
        "INSERT INTO authorships (person_id, work_id, position) VALUES (?, ?, 'first')",
        (person_id, work_id),
    )
    db.commit()
    embed_pending_works(db, FakeEmbedder(), MODEL, batch_size=8)
    run_match(db, "default", FakeEmbedder().encode(["a"])[0], MODEL)
    assert db.execute("SELECT sparse FROM matches").fetchone()["sparse"] == 1


def test_run_match_is_rerunnable(db):
    work_id = _add_work(db, "W1", "t", "a")
    person_id = _add_person(db, "P", "A1")
    db.execute(
        "INSERT INTO authorships (person_id, work_id, position) VALUES (?, ?, 'last')",
        (person_id, work_id),
    )
    db.commit()
    embed_pending_works(db, FakeEmbedder(), MODEL, batch_size=8)
    vector = FakeEmbedder().encode(["a"])[0]
    run_match(db, "default", vector, MODEL)
    run_match(db, "default", vector, MODEL)
    assert db.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1


def test_build_profile_vector_weights_seed_abstracts_double():
    profile = Profile(
        name="d", contact_email="e@x.com", interests="alpha",
        keywords=[], seed_papers=[], my_papers=[], fields=[], countries=[],
    )
    embedder = FakeEmbedder()
    without = build_profile_vector(profile, [], embedder)
    with_seed = build_profile_vector(profile, ["beta beta beta"], embedder)
    assert not np.allclose(without, with_seed)


def test_build_profile_vector_refuses_when_nothing_resolves():
    profile = Profile(
        name="d", contact_email="e@x.com", interests="",
        keywords=[], seed_papers=[], my_papers=[], fields=[], countries=[],
    )
    with pytest.raises(ValueError, match="no interest inputs"):
        build_profile_vector(profile, [], FakeEmbedder())
