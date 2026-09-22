from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE institutions (
        id TEXT PRIMARY KEY, name TEXT NOT NULL, ror_id TEXT,
        openalex_id TEXT UNIQUE, country TEXT, site TEXT, adapter TEXT,
        discovered INTEGER DEFAULT 0, added_at TEXT NOT NULL
    );
    CREATE TABLE people (
        id INTEGER PRIMARY KEY,
        institution_id TEXT REFERENCES institutions(id),
        dept TEXT, name TEXT NOT NULL,
        openalex_author_id TEXT UNIQUE, orcid TEXT, title TEXT, homepage TEXT,
        email TEXT, email_confidence TEXT, email_source TEXT,
        works_count INTEGER DEFAULT 0, first_year INTEGER, last_year INTEGER,
        last_author_ratio REAL, first_author_ratio REAL,
        faculty_score REAL, faculty_confidence TEXT, faculty_signals TEXT,
        last_seen_at TEXT
    );
    CREATE TABLE works (
        id INTEGER PRIMARY KEY, openalex_work_id TEXT UNIQUE NOT NULL,
        title TEXT NOT NULL, abstract TEXT, year INTEGER, doi TEXT,
        venue TEXT, topics TEXT, cited_by INTEGER DEFAULT 0
    );
    CREATE TABLE authorships (
        person_id INTEGER NOT NULL REFERENCES people(id),
        work_id INTEGER NOT NULL REFERENCES works(id),
        position TEXT, PRIMARY KEY (person_id, work_id)
    );
    CREATE TABLE embeddings (
        work_id INTEGER PRIMARY KEY REFERENCES works(id),
        model TEXT NOT NULL, dim INTEGER NOT NULL, vector BLOB NOT NULL,
        source TEXT NOT NULL, computed_at TEXT NOT NULL
    );
    CREATE TABLE matches (
        profile_id TEXT NOT NULL,
        person_id INTEGER NOT NULL REFERENCES people(id),
        stage1_score REAL NOT NULL, stage2_score REAL, reason TEXT,
        top_work_ids TEXT, sparse INTEGER DEFAULT 0, computed_at TEXT NOT NULL,
        PRIMARY KEY (profile_id, person_id)
    );
    CREATE TABLE cursors (
        key TEXT PRIMARY KEY, cursor TEXT, updated_at TEXT
    );
    CREATE TABLE fetch_log (
        url TEXT PRIMARY KEY, fetched_at TEXT NOT NULL, status INTEGER, etag TEXT
    );
    CREATE INDEX idx_matches_score ON matches(profile_id, stage1_score DESC);
    CREATE INDEX idx_people_faculty ON people(faculty_confidence, faculty_score DESC);
    CREATE INDEX idx_works_year ON works(year);
    """,
}


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _current_version(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return row[0] or 0


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations transactionally. Forward-only. Returns version reached."""
    current = _current_version(conn)
    for version in sorted(MIGRATIONS):
        if version <= current:
            continue
        with conn:
            conn.executescript(MIGRATIONS[version])
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
    return _current_version(conn)
