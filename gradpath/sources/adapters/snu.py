from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar
from urllib.parse import urljoin

from gradpath.models import FacultyRecord
from gradpath.sources.adapters import register
from gradpath.sources.adapters.base import InstitutionAdapter
from gradpath.sources.crawler import extract_emails

DIRECTORY_URLS = (
    "https://cse.snu.ac.kr/people/faculty",
    "https://ee.snu.ac.kr/people/faculty",
)


@register
class SnuAdapter(InstitutionAdapter):
    slug = "snu"
    domains: ClassVar[list[str]] = ["snu.ac.kr"]
    item_selector = ".faculty-card"

    def faculty_urls(self) -> Iterable[str]:
        return DIRECTORY_URLS

    def parse_faculty(self, html: str, url: str) -> list[FacultyRecord]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        records: list[FacultyRecord] = []
        for item in soup.select(self.item_selector):
            name_node = item.select_one(".name")
            if not name_node:
                continue
            addresses = extract_emails(str(item))
            href = name_node.get("href")
            records.append(
                FacultyRecord(
                    name=name_node.get_text(strip=True),
                    email=addresses[0] if addresses else None,
                    title=(item.select_one(".position").get_text(strip=True)
                           if item.select_one(".position") else None),
                    homepage=urljoin(url, href) if href else None,
                    dept=(item.select_one(".dept").get_text(strip=True)
                          if item.select_one(".dept") else None),
                )
            )
        return records
