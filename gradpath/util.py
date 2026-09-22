from __future__ import annotations

import re
from datetime import datetime, timezone


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string. The only clock in the codebase."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(text: str) -> str:
    """Lowercase, hyphen-separated identifier safe for use as a primary key."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return cleaned.strip("-")
