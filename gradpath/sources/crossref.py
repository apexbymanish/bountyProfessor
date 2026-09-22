from __future__ import annotations

from gradpath.net.http import HostBlocked, PoliteClient

CROSSREF_BASE = "https://api.crossref.org/works"


def _surname(name: str) -> str:
    parts = [p for p in name.replace(",", " ").split() if p]
    return parts[-1].lower() if parts else ""


def crossref_email(client: PoliteClient, doi: str, author_name: str) -> str | None:
    """Return the email Crossref holds for this author on this DOI, if any.

    Matching is on surname: Crossref initialises given names inconsistently, and
    a wrong-person email is worse than no email.
    """
    try:
        payload = client.get_json(f"{CROSSREF_BASE}/{doi}")
    except HostBlocked:
        return None
    target = _surname(author_name)
    if not target:
        return None
    for author in (payload.get("message") or {}).get("author") or []:
        email = author.get("email")
        if email and (author.get("family") or "").lower() == target:
            return email
    return None
