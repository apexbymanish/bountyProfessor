from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from gradpath.net.http import HostBlocked, PoliteClient
from gradpath.sources.adapters import get_adapter
from gradpath.sources.crawler import crawl_faculty_email, extract_emails
from gradpath.sources.crossref import crossref_email
from gradpath.sources.orcid import orcid_record
from gradpath.util import now_iso


@dataclass(frozen=True)
class EmailResolution:
    email: str | None
    confidence: str          # high | medium | none
    source: str | None       # crossref | orcid | adapter | crawler | None
    title: str | None = None


def _recent_dois(conn: sqlite3.Connection, person_id: int, limit: int = 5) -> list[str]:
    rows = conn.execute(
        """
        SELECT w.doi FROM authorships a JOIN works w ON w.id = a.work_id
        WHERE a.person_id = ? AND w.doi IS NOT NULL
        ORDER BY w.year DESC LIMIT ?
        """,
        (person_id, limit),
    ).fetchall()
    return [row["doi"] for row in rows]


def _adapter_email(
    conn: sqlite3.Connection, client: PoliteClient, person_row: sqlite3.Row
) -> tuple[str | None, str | None]:
    institution = conn.execute(
        "SELECT adapter FROM institutions WHERE id = ?", (person_row["institution_id"],)
    ).fetchone()
    if not institution or not institution["adapter"]:
        return None, None
    adapter_cls = get_adapter(institution["adapter"])
    if adapter_cls is None:
        return None, None
    adapter = adapter_cls()
    surname = person_row["name"].split()[-1].lower()
    for url in adapter.faculty_urls():
        try:
            html = client.get_text(url)
        except HostBlocked:
            continue
        if html is None:
            continue
        for record in adapter.parse_faculty(html, url):
            if surname in record.name.lower():
                email = record.email or (
                    extract_emails(html)[0] if extract_emails(html) else None
                )
                return email, record.title
    return None, None


def resolve_email(
    conn: sqlite3.Connection, client: PoliteClient, person_row: sqlite3.Row
) -> EmailResolution:
    """Try each source in order, stopping at the first hit.

    An address is never inferred from a name pattern. A guessed address bounces,
    and bounce rate is a deliverability signal that damages every later email
    sent from the same account — a NULL plus a homepage link is more useful.
    """
    for doi in _recent_dois(conn, person_row["id"]):
        email = crossref_email(client, doi, person_row["name"])
        if email:
            return EmailResolution(email, "high", "crossref")

    title = None
    if person_row["orcid"]:
        record = orcid_record(client, person_row["orcid"])
        if record:
            title = record.title
            if record.email:
                return EmailResolution(record.email, "high", "orcid", title)

    email, adapter_title = _adapter_email(conn, client, person_row)
    if email:
        return EmailResolution(email, "high", "adapter", adapter_title or title)

    email = crawl_faculty_email(client, person_row["homepage"], person_row["name"])
    if email:
        return EmailResolution(email, "medium", "crawler", title)

    return EmailResolution(None, "none", None, title)


def persist_resolution(
    conn: sqlite3.Connection, person_id: int, resolution: EmailResolution
) -> None:
    with conn:
        conn.execute(
            "UPDATE people SET email = ?, email_confidence = ?, email_source = ?, "
            "title = COALESCE(?, title), last_seen_at = ? WHERE id = ?",
            (resolution.email, resolution.confidence, resolution.source,
             resolution.title, now_iso(), person_id),
        )
