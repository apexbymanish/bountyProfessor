import sqlite3
from pathlib import Path

from gradpath.db import MIGRATIONS, SCHEMA_VERSION, connect, migrate

EXPECTED_TABLES = {
    "institutions", "people", "works", "authorships",
    "embeddings", "matches", "cursors", "fetch_log", "schema_version",
}


def test_migrate_creates_all_tables(tmp_path: Path) -> None:
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert EXPECTED_TABLES <= {r[0] for r in rows}


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    conn = connect(tmp_path / "t.db")
    assert migrate(conn) == SCHEMA_VERSION
    assert migrate(conn) == SCHEMA_VERSION  # second run is a no-op


def test_foreign_keys_are_enforced(tmp_path: Path) -> None:
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    try:
        conn.execute(
            "INSERT INTO authorships (person_id, work_id, position) VALUES (9999, 9999, 'first')"
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return
    raise AssertionError("foreign keys were not enforced")


def test_migrate_rollback_on_failure(tmp_path: Path) -> None:
    """Test that a failed migration rolls back atomically, leaving version unchanged."""
    conn = connect(tmp_path / "t.db")
    # Apply every real migration successfully first, whatever SCHEMA_VERSION currently is.
    base_version = migrate(conn)
    assert base_version == SCHEMA_VERSION

    # Add a failing migration to MIGRATIONS (a script that creates a table twice),
    # one version past whatever the real migrations currently reach.
    next_version = base_version + 1
    failing_script = """
    CREATE TABLE test_table_1 (id INTEGER PRIMARY KEY);
    CREATE TABLE test_table_1 (id INTEGER PRIMARY KEY);
    """
    assert next_version not in MIGRATIONS, "test assumes no real migration at next_version yet"
    MIGRATIONS[next_version] = failing_script

    try:
        # Attempt to apply the new migration, which should fail on the duplicate CREATE TABLE
        migrate(conn)
        raise AssertionError("migrate() should have raised OperationalError for duplicate table")
    except sqlite3.OperationalError:
        pass  # Expected
    finally:
        # Clean up: remove the test migration
        del MIGRATIONS[next_version]

    # Verify schema_version is still base_version (the failing migration was rolled back)
    version_row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    assert version_row[0] == base_version, (
        f"schema_version should still be {base_version}, got {version_row[0]}"
    )

    # Verify the test_table_1 does not exist (rollback occurred)
    table_row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='test_table_1'"
    ).fetchone()
    assert table_row is None, "test_table_1 should not exist after rollback"


def test_embeddings_table_has_composite_primary_key_after_migration(tmp_path: Path) -> None:
    """Two models can hold vectors for the same work simultaneously without evicting each other.

    Prevents a silent-partial-scoring failure: before migration 2, `embeddings.work_id`
    was the sole primary key, so switching embedding_model silently evicted the whole
    cache and a mixed-model corpus would score only the subset under the queried model.
    """
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO institutions (id, name, added_at) VALUES ('i', 'I', '2026-01-01')"
    )
    work_id = conn.execute(
        "INSERT INTO works (openalex_work_id, title, year) VALUES ('W1', 't', 2024)"
    ).lastrowid
    conn.execute(
        "INSERT INTO embeddings (work_id, model, dim, vector, source, computed_at) "
        "VALUES (?, 'model-a', 1, X'0000803F', 'abstract', '2026-01-01')",
        (work_id,),
    )
    conn.execute(
        "INSERT INTO embeddings (work_id, model, dim, vector, source, computed_at) "
        "VALUES (?, 'model-b', 1, X'0000803F', 'abstract', '2026-01-01')",
        (work_id,),
    )
    conn.commit()
    rows = conn.execute(
        "SELECT model FROM embeddings WHERE work_id = ? ORDER BY model", (work_id,)
    ).fetchall()
    assert [row["model"] for row in rows] == ["model-a", "model-b"]
