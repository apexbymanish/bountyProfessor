from __future__ import annotations

import hashlib
import json
import time
import urllib.robotparser
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import httpx

from gradpath.models import Settings

MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 1.0
CIRCUIT_BREAK_FAILURES = 5


class HostBlocked(RuntimeError):
    """Raised when a host has failed too often and is circuit-broken."""


class RateLimiter:
    """Per-host minimum interval between requests."""

    def __init__(self, per_second: float) -> None:
        self._interval = 1.0 / per_second if per_second > 0 else 0.0
        self._last: dict[str, float] = defaultdict(float)

    def wait(self, host: str) -> None:
        if self._interval <= 0:
            return
        elapsed = time.monotonic() - self._last[host]
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last[host] = time.monotonic()


class PoliteClient:
    """The only object in gradpath permitted to make network calls."""

    def __init__(self, settings: Settings, contact_email: str, cache_dir: Path) -> None:
        self._cache = Path(cache_dir)
        self._cache.mkdir(parents=True, exist_ok=True)
        self._limiter = RateLimiter(settings.rate_limit_per_host)
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._failures: dict[str, int] = defaultdict(int)
        self.contact_email = contact_email
        self.user_agent = (
            f"gradpath/0.1 (+https://github.com/gradpath/gradpath; {contact_email})"
        )
        self._client = httpx.Client(
            headers={"User-Agent": self.user_agent}, timeout=30.0, follow_redirects=True
        )

    # --- cache -------------------------------------------------------------

    def _cache_path(self, url: str, params: dict | None) -> Path:
        key = hashlib.sha256(f"{url}|{sorted((params or {}).items())}".encode()).hexdigest()
        return self._cache / f"{key}.json"

    # --- robots ------------------------------------------------------------

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            parser = urllib.robotparser.RobotFileParser()
            self._limiter.wait(parsed.netloc)
            try:
                response = self._client.get(f"{origin}/robots.txt")
                if response.status_code == 200:
                    parser.parse(response.text.splitlines())
                else:
                    parser = None  # absent robots.txt means unrestricted
            except Exception:  # noqa: BLE001 -- deliberately broad, see below
                # A single attempt only: an unreachable OR unparseable
                # robots.txt already has a defined meaning (unrestricted),
                # and this must never retry or count against the host's
                # circuit-breaker budget, or a down/broken robots endpoint
                # would disable content fetching. Deliberately broad --
                # any robots failure (transport, or a bad parse) degrades
                # to "allowed" rather than propagating.
                parser = None
            self._robots[origin] = parser
        parser = self._robots[origin]
        return True if parser is None else parser.can_fetch(self.user_agent, url)

    # --- fetching ----------------------------------------------------------

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float:
        """Parse Retry-After as seconds, falling back on the default backoff.

        RFC 7231 also permits Retry-After to be an HTTP-date rather than a
        number of seconds. Parsing that form is not needed: on any value we
        cannot read as a number, degrade to BACKOFF_BASE_SECONDS instead of
        letting ValueError abort a multi-hour crawl on an arbitrary header.
        """
        raw = response.headers.get("Retry-After")
        if raw is None:
            return BACKOFF_BASE_SECONDS
        try:
            return float(raw)
        except ValueError:
            return BACKOFF_BASE_SECONDS

    def _request(self, url: str, params: dict | None) -> httpx.Response:
        host = urlparse(url).netloc
        if self._failures[host] >= CIRCUIT_BREAK_FAILURES:
            raise HostBlocked(f"{host} circuit-broken after repeated failures")
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            self._limiter.wait(host)
            try:
                response = self._client.get(url, params=params)
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
                continue
            if response.status_code == 429:
                time.sleep(self._retry_after_seconds(response))
                continue
            if response.status_code >= 500:
                time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
                continue
            self._failures[host] = 0
            return response
        self._failures[host] += 1
        raise HostBlocked(f"{url} failed after {MAX_ATTEMPTS} attempts: {last_error}")

    def get_json(self, url: str, params: dict | None = None) -> dict:
        """Fetch JSON, serving from the on-disk cache when present.

        The cache file holds the response payload unmodified (no injected
        metadata), so a cache hit is byte-for-byte what the network returned.
        The file's mtime is sufficient provenance; no fetch_log write happens
        here -- that is left to database-aware callers in later tasks so this
        module never needs a database handle.
        """
        cached = self._cache_path(url, params)
        if cached.exists():
            return json.loads(cached.read_text())
        payload = self._request(url, params).json()
        cached.write_text(json.dumps(payload))
        return payload

    def get_text(self, url: str) -> str | None:
        """Fetch a page, honouring robots.txt. Returns None when disallowed."""
        if not self.allowed(url):
            return None
        return self._request(url, None).text

    def close(self) -> None:
        self._client.close()
