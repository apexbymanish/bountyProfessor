"""Seeding the institution registry from sourced data.

Ranks are never typed by hand here. They come from OpenAlex, which is current
by construction, or from a ranking CSV the user can cite. A fabricated rank in
a seed file is worse than no rank at all.

`init` (see gradpath/cli.py) already seeds `data/institutions.yaml` -- with
each institution's adapter -- into the `institutions` table via
`INSERT OR IGNORE`, so a curated row like KAIST's exists with a slug id, a
`site`, and an `adapter` before this module ever runs. What this module adds
is *resolution*: binding an OpenAlex id onto that already-seeded row rather
than creating a second, orphaned one. A naive "look up by openalex_id, else
insert a new row keyed slugify(name)" gets this wrong -- slugify(hit.name)
for KAIST's OpenAlex display name ("Korea Advanced Institute of Science and
Technology") is not "kaist", so it would silently disable that seeded row's
adapter. `upsert_institutions` instead matches, in order: an exact
openalex_id, then a homepage-domain match against a seeded row's `site`,
and only then falls back to slugify(name) for a genuinely new row. An
adapter bound by hand is never overwritten by any of these paths.
"""
from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from gradpath.net.http import PoliteClient
from gradpath.sources.openalex import OPENALEX_BASE, with_mailto
from gradpath.util import now_iso, slugify

SUPPORTED_METRICS = ("cited_by_count", "works_count")
REQUIRED_CSV_COLUMNS = {"rank", "name"}


@dataclass(frozen=True)
class InstitutionHit:
    openalex_id: str
    ror_id: str | None
    name: str
    country: str | None
    site: str | None
    works_count: int
    cited_by_count: int


def _short_id(url: str | None) -> str | None:
    return url.rsplit("/", 1)[-1] if url else None


def _hostname(url: str | None) -> str | None:
    """The registrable hostname of a URL, lowercased and with a `www.` prefix
    stripped, so `https://www.kaist.ac.kr` and `https://kaist.ac.kr/grad`
    compare equal. Returns None for anything that yields no host at all."""
    if not url:
        return None
    parsed = urlparse(url if "//" in url else f"//{url}")
    host = (parsed.hostname or "").lower()
    if not host:
        return None
    return host.removeprefix("www.")


def top_institutions(
    client: PoliteClient, country: str, limit: int, metric: str = "cited_by_count"
) -> list[InstitutionHit]:
    """The highest-output institutions in a country, straight from OpenAlex.

    This is what makes "Korea top 20" reproducible without trusting a stale
    hand-written list: the ordering is recomputed from live research output.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"metric must be one of {SUPPORTED_METRICS}, got {metric!r}")
    payload = client.get_json(
        f"{OPENALEX_BASE}/institutions",
        with_mailto(client, {
            "filter": f"country_code:{country.lower()},type:education",
            "sort": f"{metric}:desc",
            "per-page": limit,
        }),
    )
    return [
        InstitutionHit(
            openalex_id=_short_id(item.get("id")) or "",
            ror_id=_short_id(item.get("ror")),
            name=item.get("display_name") or "",
            country=item.get("country_code"),
            site=item.get("homepage_url"),
            works_count=item.get("works_count") or 0,
            cited_by_count=item.get("cited_by_count") or 0,
        )
        for item in payload.get("results", [])
    ]


def _match_existing(conn: sqlite3.Connection, hit: InstitutionHit) -> sqlite3.Row | None:
    """Resolve an OpenAlex hit to an existing row, in order: exact openalex_id,
    then homepage-domain match. Domain matching only considers rows still
    missing an openalex_id -- a row already bound to a different OpenAlex id
    must never be silently re-pointed by a coincidental domain match."""
    row = conn.execute(
        "SELECT id FROM institutions WHERE openalex_id = ?", (hit.openalex_id,)
    ).fetchone()
    if row:
        return row
    host = _hostname(hit.site)
    if not host:
        return None
    for candidate in conn.execute(
        "SELECT id, site FROM institutions WHERE openalex_id IS NULL AND site IS NOT NULL"
    ):
        if _hostname(candidate["site"]) == host:
            return candidate
    return None


def upsert_institutions(
    conn: sqlite3.Connection, hits: list[InstitutionHit], tier: str, rank_source: str
) -> int:
    """Insert or update institutions, assigning rank by list order.

    An adapter bound by hand is never overwritten -- seeding must not undo
    a contributor's mapping. See the module docstring for the resolution
    order (openalex_id, then homepage domain, then a new row).
    """
    with conn:
        for position, hit in enumerate(hits, start=1):
            existing = _match_existing(conn, hit)
            if existing:
                conn.execute(
                    "UPDATE institutions SET name = ?, country = COALESCE(?, country), "
                    "site = COALESCE(?, site), ror_id = COALESCE(?, ror_id), "
                    "openalex_id = ?, tier = ?, rank = ?, rank_source = ? WHERE id = ?",
                    (hit.name, hit.country, hit.site, hit.ror_id, hit.openalex_id,
                     tier, position, rank_source, existing["id"]),
                )
                continue
            # No existing row matched by id or domain. Fall back to a
            # name-derived slug -- but if that slug happens to already be
            # taken (e.g. a CSV import used the same name), merge into it
            # rather than crashing on the primary-key collision.
            conn.execute(
                "INSERT INTO institutions (id, name, ror_id, openalex_id, country, site, "
                "discovered, tier, rank, rank_source, added_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "ror_id = COALESCE(excluded.ror_id, institutions.ror_id), "
                "openalex_id = excluded.openalex_id, "
                "country = COALESCE(excluded.country, institutions.country), "
                "site = COALESCE(excluded.site, institutions.site), "
                "tier = excluded.tier, rank = excluded.rank, "
                "rank_source = excluded.rank_source",
                (slugify(hit.name), hit.name, hit.ror_id, hit.openalex_id, hit.country,
                 hit.site, tier, position, rank_source, now_iso()),
            )
    return len(hits)


def import_ranking_csv(
    conn: sqlite3.Connection, path: Path, tier: str, top: int
) -> int:
    """Import a QS/THE/ARWU ranking snapshot. See data/rankings/README.md for sources.

    OpenAlex ids are left NULL here; running `gradpath institutions top` (or
    another resolution pass) over the matching country later binds them, so
    an import never invents an identifier it cannot verify.
    """
    with Path(path).open() as handle:
        reader = csv.DictReader(handle)
        columns = {name.strip().lower() for name in (reader.fieldnames or [])}
        if not REQUIRED_CSV_COLUMNS <= columns:
            raise ValueError(
                f"ranking CSV must have 'rank' and 'name' columns, found {sorted(columns)}"
            )
        rows = [row for row in reader][:top]

    with conn:
        for row in rows:
            name = row["name"].strip()
            conn.execute(
                "INSERT INTO institutions (id, name, country, discovered, tier, rank, "
                "rank_source, added_at) VALUES (?, ?, ?, 0, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET tier = excluded.tier, "
                "rank = excluded.rank, rank_source = excluded.rank_source",
                (slugify(name), name, (row.get("country") or "").strip().upper() or None,
                 tier, int(row["rank"]), f"csv:{Path(path).name}", now_iso()),
            )
    return len(rows)
