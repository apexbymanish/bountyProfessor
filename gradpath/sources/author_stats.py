from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from gradpath.net.http import HostBlocked, PoliteClient
from gradpath.sources.openalex import OPENALEX_BASE, with_mailto


@dataclass(frozen=True)
class AuthorCareer:
    works_count: int
    first_year: int | None
    last_year: int | None


def fetch_author_career(client: PoliteClient, openalex_author_id: str) -> AuthorCareer | None:
    """Fetch whole-career stats for an OpenAlex author, never a topic/year slice.

    Faculty-likelihood scoring needs a person's real career span and volume,
    not the works_count/first_year/last_year columns on `people`, which are
    computed from whatever topic and --since window was ingested and so are
    bounded by the query rather than by the person's actual career. Returns
    None on a circuit-broken host or a missing/malformed record; never raises.
    """
    try:
        payload = client.get_json(
            f"{OPENALEX_BASE}/authors/{openalex_author_id}", with_mailto(client, {})
        )
    except HostBlocked:
        return None
    if not payload or "works_count" not in payload:
        return None

    years = [
        entry.get("year")
        for entry in payload.get("counts_by_year") or []
        if (entry.get("works_count") or 0) > 0 and entry.get("year") is not None
    ]
    return AuthorCareer(
        works_count=payload.get("works_count") or 0,
        first_year=min(years) if years else None,
        last_year=max(years) if years else None,
    )


def enrich_person_career(
    conn: sqlite3.Connection, client: PoliteClient, person_row: sqlite3.Row
) -> bool:
    """Fetch and persist whole-career stats for one person. Returns success.

    Writes only career_first_year/career_last_year/career_works_count -- the
    slice-derived works_count/first_year/last_year and the topic-relevant
    last_author_ratio/first_author_ratio are untouched here (see
    `_refresh_person_stats` in gradpath/sources/ingest.py, which owns those).
    """
    author_id = person_row["openalex_author_id"]
    if not author_id:
        return False
    career = fetch_author_career(client, author_id)
    if career is None:
        return False
    with conn:
        conn.execute(
            "UPDATE people SET career_first_year = ?, career_last_year = ?, "
            "career_works_count = ? WHERE id = ?",
            (career.first_year, career.last_year, career.works_count, person_row["id"]),
        )
    return True
