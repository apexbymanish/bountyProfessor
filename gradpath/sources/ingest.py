from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Iterator

from gradpath.net.http import PoliteClient
from gradpath.sources.openalex import (
    OPENALEX_BASE, ParsedAuthorship, ParsedWork, parse_work,
)
from gradpath.util import now_iso, slugify

PAGE_SIZE = 200


@dataclass
class DiscoverStats:
    works_seen: int = 0
    people_seen: int = 0
    institutions_added: int = 0


def _load_cursor(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT cursor FROM cursors WHERE key = ?", (key,)).fetchone()
    return row["cursor"] if row else "*"


def _save_cursor(conn: sqlite3.Connection, key: str, cursor: str | None) -> None:
    with conn:
        conn.execute(
            "INSERT INTO cursors (key, cursor, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET cursor = excluded.cursor, "
            "updated_at = excluded.updated_at",
            (key, cursor, now_iso()),
        )


def iter_works(
    client: PoliteClient,
    topics: list[str],
    countries: list[str],
    since: int,
    cursor_key: str,
    conn: sqlite3.Connection,
    institutions: list[str] | None = None,
) -> Iterator[ParsedWork]:
    """Stream works matching the filter, persisting the cursor after every page."""
    filters = [f"publication_year:>{since - 1}"]
    if topics:
        filters.append("topics.id:" + "|".join(topics))
    if institutions:
        filters.append("institutions.id:" + "|".join(institutions))
    elif countries:
        filters.append("institutions.country_code:" + "|".join(c.lower() for c in countries))

    cursor = _load_cursor(conn, cursor_key)
    while cursor:
        payload = client.get_json(
            f"{OPENALEX_BASE}/works",
            {"filter": ",".join(filters), "per-page": PAGE_SIZE, "cursor": cursor},
        )
        for raw in payload.get("results", []):
            yield parse_work(raw)
        cursor = (payload.get("meta") or {}).get("next_cursor")
        _save_cursor(conn, cursor_key, cursor)


def _upsert_institution(conn: sqlite3.Connection, a: ParsedAuthorship) -> tuple[str | None, bool]:
    if not a.institution_openalex_id:
        return None, False
    row = conn.execute(
        "SELECT id FROM institutions WHERE openalex_id = ?", (a.institution_openalex_id,)
    ).fetchone()
    if row:
        return row["id"], False
    name = a.institution_name or a.institution_openalex_id
    slug = slugify(name)
    conn.execute(
        "INSERT OR IGNORE INTO institutions "
        "(id, name, openalex_id, country, discovered, added_at) VALUES (?, ?, ?, ?, 1, ?)",
        (slug, name, a.institution_openalex_id, a.institution_country, now_iso()),
    )
    return slug, True


def upsert_person(conn: sqlite3.Connection, a: ParsedAuthorship, institution_id: str | None) -> int:
    row = conn.execute(
        "SELECT id FROM people WHERE openalex_author_id = ?", (a.author_id,)
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE people SET last_seen_at = ?, institution_id = COALESCE(?, institution_id), "
            "orcid = COALESCE(?, orcid) WHERE id = ?",
            (now_iso(), institution_id, a.orcid, row["id"]),
        )
        return row["id"]
    cursor = conn.execute(
        "INSERT INTO people (institution_id, name, openalex_author_id, orcid, last_seen_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (institution_id, a.name, a.author_id, a.orcid, now_iso()),
    )
    return int(cursor.lastrowid)


def ingest_work(conn: sqlite3.Connection, work: ParsedWork) -> int | None:
    """Insert or find the row for `work`, returning its id -- or None to skip.

    A falsy `openalex_id` means the source payload had no usable work id
    (parse_work falls back to "" in that case). works.openalex_work_id is
    UNIQUE, so inserting two such works would collide on "" and the second
    work's authorships would silently attach to the first work's row. Any
    work with a falsy id is therefore skipped entirely: no row is inserted
    or matched, and the caller must not treat it as ingested.
    """
    if not work.openalex_id:
        return None
    row = conn.execute(
        "SELECT id FROM works WHERE openalex_work_id = ?", (work.openalex_id,)
    ).fetchone()
    if row:
        return row["id"]
    cursor = conn.execute(
        "INSERT INTO works (openalex_work_id, title, abstract, year, doi, venue, topics, cited_by)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (work.openalex_id, work.title, work.abstract, work.year, work.doi,
         work.venue, json.dumps(work.topics), work.cited_by),
    )
    return int(cursor.lastrowid)


def _refresh_person_stats(conn: sqlite3.Connection, person_id: int) -> None:
    """Recompute cached publication statistics used later by faculty scoring."""
    conn.execute(
        """
        UPDATE people SET
            works_count = (SELECT COUNT(*) FROM authorships WHERE person_id = :pid),
            first_year = (SELECT MIN(w.year) FROM authorships a JOIN works w ON w.id = a.work_id
                          WHERE a.person_id = :pid),
            last_year  = (SELECT MAX(w.year) FROM authorships a JOIN works w ON w.id = a.work_id
                          WHERE a.person_id = :pid),
            last_author_ratio = (
                SELECT CAST(SUM(position = 'last') AS REAL) / COUNT(*)
                FROM authorships WHERE person_id = :pid),
            first_author_ratio = (
                SELECT CAST(SUM(position = 'first') AS REAL) / COUNT(*)
                FROM authorships WHERE person_id = :pid)
        WHERE id = :pid
        """,
        {"pid": person_id},
    )


def discover(
    conn: sqlite3.Connection,
    client: PoliteClient,
    topics: list[str],
    countries: list[str],
    since: int,
    cursor_key: str,
    institutions: list[str] | None = None,
) -> DiscoverStats:
    stats = DiscoverStats()
    touched: set[int] = set()
    for work in iter_works(client, topics, countries, since, cursor_key, conn, institutions):
        with conn:
            work_id = ingest_work(conn, work)
            if work_id is None:
                continue
            stats.works_seen += 1
            for authorship in work.authorships:
                if not authorship.author_id:
                    continue
                institution_id, added = _upsert_institution(conn, authorship)
                stats.institutions_added += int(added)
                person_id = upsert_person(conn, authorship, institution_id)
                conn.execute(
                    "INSERT OR IGNORE INTO authorships (person_id, work_id, position) "
                    "VALUES (?, ?, ?)",
                    (person_id, work_id, authorship.position),
                )
                touched.add(person_id)
    with conn:
        for person_id in touched:
            _refresh_person_stats(conn, person_id)
    stats.people_seen = len(touched)
    return stats
