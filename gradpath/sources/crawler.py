from __future__ import annotations

import re

from gradpath.net.http import HostBlocked, PoliteClient

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ASSET_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".css", ".js")

_OBFUSCATIONS = (
    (re.compile(r"\s*\[\s*at\s*\]\s*", re.IGNORECASE), "@"),
    (re.compile(r"\s*\(\s*at\s*\)\s*", re.IGNORECASE), "@"),
    (re.compile(r"\s*\[\s*dot\s*\]\s*", re.IGNORECASE), "."),
    (re.compile(r"\s*\(\s*dot\s*\)\s*", re.IGNORECASE), "."),
)


def _deobfuscate(text: str) -> str:
    for pattern, replacement in _OBFUSCATIONS:
        text = pattern.sub(replacement, text)
    return text


def extract_emails(html: str) -> list[str]:
    """Pull addresses out of a page, decoding the usual light obfuscations.

    Order is preserved and duplicates removed so the first address on a
    faculty page — usually the person's own — wins.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    found: list[str] = []

    for anchor in soup.select('a[href^="mailto:"]'):
        address = anchor["href"].removeprefix("mailto:").split("?")[0].strip()
        if address:
            found.append(address)

    for match in EMAIL_PATTERN.finditer(_deobfuscate(soup.get_text(" "))):
        found.append(match.group(0))

    seen: set[str] = set()
    ordered: list[str] = []
    for address in found:
        lowered = address.lower()
        if lowered in seen or lowered.endswith(ASSET_SUFFIXES):
            continue
        seen.add(lowered)
        ordered.append(address)
    return ordered


def crawl_faculty_email(client: PoliteClient, homepage: str | None, name: str) -> str | None:
    """Fetch a personal or directory page and take the first address on it."""
    if not homepage:
        return None
    try:
        html = client.get_text(homepage)
    except HostBlocked:
        return None
    if html is None:            # robots.txt disallowed this path
        return None
    addresses = extract_emails(html)
    return addresses[0] if addresses else None
