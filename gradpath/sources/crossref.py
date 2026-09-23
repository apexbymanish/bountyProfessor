from __future__ import annotations

from gradpath.net.http import HostBlocked, PoliteClient

CROSSREF_BASE = "https://api.crossref.org/works"


def _surname(name: str) -> str:
    parts = [p for p in name.replace(",", " ").split() if p]
    return parts[-1].lower() if parts else ""


def crossref_email(client: PoliteClient, doi: str, author_name: str) -> str | None:
    """Return the email Crossref holds for this author on this DOI, if any.

    Matching is on surname: Crossref initialises given names inconsistently
    ("S." for "Sunmi"), so requiring full-name equality would reject correct
    matches without making wrong ones less likely.

    The surname must therefore be unique on the paper. Kim, Lee and Park cover
    a large share of Korean surnames -- Chinese and Vietnamese author lists
    have the same property -- so two co-authors sharing one is routine, and
    returning the first was a coin toss reported at confidence 'high' from the
    first and most-used source in the chain. Two candidates means no address
    from this DOI; the caller moves on to the next DOI and then to ORCID.

    Candidates are counted before their addresses are looked at: a co-author
    who deposited no address still proves the surname is ambiguous, and that is
    the commonest shape of this defect.
    """
    try:
        payload = client.get_json(f"{CROSSREF_BASE}/{doi}")
    except HostBlocked:
        return None
    target = _surname(author_name)
    if not target:
        return None
    authors = (payload.get("message") or {}).get("author") or []
    matches = [a for a in authors if (a.get("family") or "").lower() == target]
    if len(matches) != 1:
        return None
    return matches[0].get("email") or None
