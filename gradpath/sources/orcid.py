from __future__ import annotations

from dataclasses import dataclass

from gradpath.net.http import HostBlocked, PoliteClient

ORCID_BASE = "https://pub.orcid.org/v3.0"


@dataclass(frozen=True)
class OrcidRecord:
    email: str | None
    title: str | None


def orcid_record(client: PoliteClient, orcid: str) -> OrcidRecord | None:
    try:
        payload = client.get_json(f"{ORCID_BASE}/{orcid}/record")
    except HostBlocked:
        return None

    emails = ((payload.get("person") or {}).get("emails") or {}).get("email") or []
    email = emails[0].get("email") if emails else None

    groups = (
        ((payload.get("activities-summary") or {}).get("employments") or {})
        .get("affiliation-group")
        or []
    )
    title = None
    for group in groups:
        for summary in group.get("summaries") or []:
            employment = summary.get("employment-summary") or {}
            if employment.get("role-title"):
                title = employment["role-title"]
                break
        if title:
            break
    return OrcidRecord(email=email, title=title)
