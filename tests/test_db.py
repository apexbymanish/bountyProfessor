import sqlite3
from pathlib import Path

from gradpath.db import SCHEMA_VERSION, connect, migrate

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
