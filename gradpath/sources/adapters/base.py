from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import ClassVar

from gradpath.models import FacultyRecord


class InstitutionAdapter(ABC):
    """Parses one institution's faculty directory.

    Adapters exist because many university sites — Korean ones especially —
    render directories with JavaScript or obfuscate addresses, so the generic
    crawler performs poorly there. An adapter must not require a browser
    runtime: a page that cannot be read without executing JavaScript is left
    to the manual queue instead.
    """

    slug: str = ""
    domains: ClassVar[list[str]] = []

    @abstractmethod
    def faculty_urls(self) -> Iterable[str]:
        """Directory pages to fetch."""

    @abstractmethod
    def parse_faculty(self, html: str, url: str) -> list[FacultyRecord]:
        """Extract faculty rows from one directory page."""
