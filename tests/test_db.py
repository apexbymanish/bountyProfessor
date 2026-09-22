import sqlite3
from pathlib import Path

from gradpath.db import SCHEMA_VERSION, MIGRATIONS, connect, migrate

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
    # Apply migration 1 successfully
    assert migrate(conn) == 1

    # Add a failing migration to MIGRATIONS (a script that creates a table twice)
    failing_script = """
    CREATE TABLE test_table_1 (id INTEGER PRIMARY KEY);
    CREATE TABLE test_table_1 (id INTEGER PRIMARY KEY);
    """
    MIGRATIONS[2] = failing_script

    try:
        # Attempt to apply migration 2, which should fail on the duplicate CREATE TABLE
        migrate(conn)
        raise AssertionError("migrate() should have raised OperationalError for duplicate table")
    except sqlite3.OperationalError:
        pass  # Expected
    finally:
        # Clean up: remove the test migration
        del MIGRATIONS[2]

    # Verify schema_version is still 1 (migration 2 was rolled back)
    version_row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    assert version_row[0] == 1, f"schema_version should still be 1, got {version_row[0]}"

    # Verify the test_table_1 does not exist (rollback occurred)
    table_row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='test_table_1'"
    ).fetchone()
    assert table_row is None, "test_table_1 should not exist after rollback"
