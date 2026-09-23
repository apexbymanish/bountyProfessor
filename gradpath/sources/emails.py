from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass

from gradpath.models import FacultyRecord
from gradpath.net.http import HostBlocked, PoliteClient
from gradpath.sources.adapters import get_adapter
from gradpath.sources.crawler import crawl_faculty_email
from gradpath.sources.crossref import crossref_email
from gradpath.sources.orcid import orcid_record
from gradpath.util import now_iso

# Directory pages routinely prefix a rank or honorific onto the displayed name.
# These carry no identity, so they are dropped before comparing.
HONORIFICS = frozenset({
    "prof", "professor", "dr", "phd", "mr", "mrs", "ms", "miss",
    "assoc", "associate", "asst", "assistant", "emeritus", "adjunct",
})


@dataclass(frozen=True)
class EmailResolution:
    email: str | None
    confidence: str          # high | medium | none
    source: str | None       # crossref | orcid | adapter | crawler | None
    title: str | None = None
    homepage: str | None = None


def normalise_name(name: str) -> frozenset[str]:
    """Reduce a display name to a comparable set of tokens.

    Accents, case, punctuation and honorifics are folded away, and the result
    is a *set* so "Sunmi Park" and "Park Sunmi" compare equal -- Korean and
    Japanese directories invert order freely, and OpenAlex does not.

    A name is only usable as an identity if it carries at least two tokens of
    two or more characters. One token is not enough to identify a person, and
    neither is a surname plus a bare initial: "S. Park" and "Park, S." both
    reduce to {'park', 's'} and would compare equal, binding whichever S. Park
    a directory of hundreds happened to list. Initials that do survive the gate
    stay in the returned set, so "John F Kennedy" still differs from "John
    Kennedy" -- dropping them would be a loosening, and loosening is how a
    wrong bind gets reintroduced. The threshold is two characters, not three,
    because "Li Bo" is a whole name.
    """
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", folded.lower())
        if token and token not in HONORIFICS
    }
    substantive = [token for token in tokens if len(token) >= 2]
    return frozenset(tokens) if len(substantive) >= 2 else frozenset()


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


def match_directory_record(
    name: str, records: list[FacultyRecord]
) -> FacultyRecord | None:
    """Return the one record that is unambiguously this person, else None.

    Equality of normalised full names, never a substring of one token: the
    old `surname in record.name.lower()` matched every Kim on the page and
    handed back whichever was listed first. Two matching records means two
    people share a name on that directory, and picking either one is a coin
    toss -- so that is a miss, not a match.
    """
    target = normalise_name(name)
    if not target:
        return None
    matches = [record for record in records if normalise_name(record.name) == target]
    return matches[0] if len(matches) == 1 else None


def adapter_record(
    conn: sqlite3.Connection, client: PoliteClient, person_row: sqlite3.Row
) -> FacultyRecord | None:
    """Find this person's own row in their institution's faculty directory.

    The directory is every page the adapter exposes, not whichever page is read
    first: all three shipped adapters list two departmental pages, so checking
    uniqueness per page can never see the second Jaewon Kim one department
    over, and the ambiguity guard in `match_directory_record` never fires.
    Records are therefore accumulated across every page and the guard applied
    once, against the whole set.

    A page that cannot be read (circuit-broken host, or disallowed by
    robots.txt) does not abort the resolution -- the remaining sources still
    run -- but it does forfeit the adapter step entirely: an unread page is
    precisely where a same-named colleague would be, so uniqueness across the
    directory is unproven and no match can be claimed. That costs addresses
    when a department page is permanently unreachable, which is recoverable;
    the alternative is a stranger's address labelled 'high', which is not.
    """
    institution = conn.execute(
        "SELECT adapter FROM institutions WHERE id = ?", (person_row["institution_id"],)
    ).fetchone()
    if not institution or not institution["adapter"]:
        return None
    adapter_cls = get_adapter(institution["adapter"])
    if adapter_cls is None:
        return None
    adapter = adapter_cls()
    records: list[FacultyRecord] = []
    complete = True
    for url in adapter.faculty_urls():
        try:
            html = client.get_text(url)
        except HostBlocked:
            complete = False
            continue
        if html is None:
            complete = False
            continue
        records.extend(adapter.parse_faculty(html, url))
    if not complete:
        return None
    return match_directory_record(person_row["name"], records)


def resolve_email(
    conn: sqlite3.Connection, client: PoliteClient, person_row: sqlite3.Row
) -> EmailResolution:
    """Try each source in order, stopping at the first hit.

    An address is never inferred from a name pattern, and never harvested
    from somewhere other than the matched record. A guessed or borrowed
    address bounces, and bounce rate is a deliverability signal that damages
    every later email sent from the same account -- and a *misattributed*
    address labelled 'high' confidence is worse still, because nothing
    downstream can tell it apart from a real one. A NULL plus a homepage
    link is more useful than either.
    """
    for doi in _recent_dois(conn, person_row["id"]):
        email = crossref_email(client, doi, person_row["name"])
        if email:
            return EmailResolution(email, "high", "crossref", homepage=person_row["homepage"])

    title = None
    homepage = person_row["homepage"]
    if person_row["orcid"]:
        record = orcid_record(client, person_row["orcid"])
        if record:
            title = record.title
            if record.email:
                return EmailResolution(record.email, "high", "orcid", title, homepage)

    directory = adapter_record(conn, client, person_row)
    if directory is not None:
        # Only an unambiguously matched record's metadata is trusted: its
        # title feeds `_has_professor_title`, which short-circuits faculty
        # scoring to 1.0/high, so a borrowed title is a confidently wrong
        # seniority estimate as well as a wrong address.
        title = directory.title or title
        homepage = homepage or directory.homepage
        if directory.email:
            return EmailResolution(directory.email, "high", "adapter", title, homepage)

    email = crawl_faculty_email(client, homepage, person_row["name"])
    if email:
        return EmailResolution(email, "medium", "crawler", title, homepage)

    return EmailResolution(None, "none", None, title, homepage)


def persist_resolution(
    conn: sqlite3.Connection, person_id: int, resolution: EmailResolution
) -> None:
    """Write a resolution back, including the homepage it discovered.

    homepage is COALESCEd on both sides: a newly discovered one fills a NULL
    column (making the crawler stage reachable on a later run, and `emails
    report` useful), and a resolution that found none never erases one that
    is already stored.
    """
    with conn:
        conn.execute(
            "UPDATE people SET email = ?, email_confidence = ?, email_source = ?, "
            "title = COALESCE(?, title), homepage = COALESCE(homepage, ?), "
            "last_seen_at = ? WHERE id = ?",
            (resolution.email, resolution.confidence, resolution.source,
             resolution.title, resolution.homepage, now_iso(), person_id),
        )
