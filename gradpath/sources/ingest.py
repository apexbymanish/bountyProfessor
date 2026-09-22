from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass

from gradpath.net.http import PoliteClient
from gradpath.sources.openalex import (
    OPENALEX_BASE,
    ParsedAuthorship,
    ParsedWork,
    parse_work,
    with_mailto,
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


def iter_pages(
    client: PoliteClient,
    topics: list[str],
    countries: list[str],
    since: int,
    cursor_key: str,
    conn: sqlite3.Connection,
    institutions: list[str] | None = None,
) -> Iterator[list[ParsedWork]]:
    """Stream works one page at a time, persisting the cursor after every page.

    The cursor save happens lazily: it only runs once the caller has finished
    consuming the current page and asks this generator for the next one (a
    `for page in iter_pages(...):` loop pulls the next page only after its
    body finishes). So if the caller's per-page work -- ingesting the page's
    works and refreshing the touched people's stats -- is what happens in that
    loop body, the cursor for a page only advances once that page's work is
    durably committed. A crash mid-page leaves the cursor exactly where it
    was, and the page is safely re-fetched (all writes here are idempotent).
    """
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
            with_mailto(client, {"filter": ",".join(filters), "per-page": PAGE_SIZE, "cursor": cursor}),
        )
        page = [parse_work(raw) for raw in payload.get("results", [])]
        yield page
        cursor = (payload.get("meta") or {}).get("next_cursor")
        _save_cursor(conn, cursor_key, cursor)


def iter_works(
    client: PoliteClient,
    topics: list[str],
    countries: list[str],
    since: int,
    cursor_key: str,
    conn: sqlite3.Connection,
    institutions: list[str] | None = None,
) -> Iterator[ParsedWork]:
    """Flatten `iter_pages` into a single stream of works. See `iter_pages` for
    the cursor-persistence semantics this preserves."""
    for page in iter_pages(client, topics, countries, since, cursor_key, conn, institutions):
        yield from page


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
        # Whichever authorship is processed last (an artefact of OpenAlex's
        # pagination/result order, not of publication recency) wins and
        # overwrites institution_id here -- a person's stored institution can
        # flip as the crawl proceeds. That's a real limitation; fixing it
        # properly needs ordering by publication date, which this task
        # deliberately does not impose. Left as-is; not in scope here.
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
    """Ingest matching works page by page, refreshing stats once per page.

    Stats are refreshed per page -- not once at the end of the whole run, and
    not once per work. Once at the end is not crash-safe: `iter_pages` can
    raise partway through an hours-long crawl (e.g. `HostBlocked`), and
    anything already committed before that would be left with permanently
    stale works_count/year defaults unless a later run happens to touch the
    same people again. Once per work is crash-safe too, but wastefully
    recomputes a prolific author's aggregates once per paper. Once per page
    is both crash-safe and cheap: it lines up with `iter_pages`'s cursor
    persistence, so if the run dies mid-page the cursor has not advanced,
    the page is re-fetched on retry, and those people's stats are
    recomputed then.
    """
    stats = DiscoverStats()
    touched_overall: set[int] = set()
    for page in iter_pages(client, topics, countries, since, cursor_key, conn, institutions):
        touched_page: set[int] = set()
        for work in page:
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
                    touched_page.add(person_id)
        with conn:
            for person_id in touched_page:
                _refresh_person_stats(conn, person_id)
        touched_overall |= touched_page
    stats.people_seen = len(touched_overall)
    return stats
