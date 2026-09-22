# gradpath Discovery & Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI that discovers every researcher publishing in a given field across chosen countries, estimates who is faculty, resolves their email from authoritative sources, and ranks all of them against the user's research interests with no cap.

**Architecture:** A layered Python package over SQLite. `net/http.py` is the single module permitted to make network calls, which is what makes rate limiting and `robots.txt` compliance enforceable rather than aspirational. Data acquisition (`sources/`) writes to the database; scoring (`match/`) reads from it and writes back to `matches`; presentation (`report/`) only reads. Work embeddings are computed once and persisted as float32 BLOBs so uncapped scoring stays affordable.

**Tech Stack:** Python 3.11+, typer (CLI), httpx (HTTP), PyYAML (config), NumPy (vectorised scoring), sentence-transformers (embeddings, lazily imported), BeautifulSoup4 + lxml (crawling), rich (tables), pytest + respx (tests).

**Spec:** `docs/superpowers/specs/2026-09-22-gradpath-design.md`

## Global Constraints

- Python 3.11 or newer. Type hints on every public function.
- **No module outside `gradpath/net/` may import `httpx`, `requests`, or `urllib.request`.** Task 3 adds a test enforcing this.
- Default rate limit is 1.0 requests/second per host, configurable **downward only**.
- OpenAlex requests always include the `mailto` parameter taken from `profile.yaml`.
- **Never infer an email address from a name pattern.** A NULL email plus a homepage is the correct output when no source resolves.
- **Never drop a person from results.** `faculty_confidence` is a sortable, filterable column; `--faculty-only` is opt-in.
- **No top-N anywhere in the pipeline.** `--limit` exists only on `show` and `export` and never affects what is computed or stored.
- Fit score and faculty score are stored and displayed separately, never blended into one number.
- No test may touch the live network. All HTTP is mocked with `respx`.
- `profile.yaml`, `*.db` and `.cache/` are gitignored and must never be committed.
- No LinkedIn. No authenticated scraping. No headless browsers.
- Timestamps stored as ISO-8601 UTC strings via a shared `now_iso()` helper.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, entry point |
| `gradpath/util.py` | `now_iso()`, slugify — shared primitives with no internal deps |
| `gradpath/db.py` | Schema, forward-only migrations, connection factory |
| `gradpath/models.py` | Dataclasses crossing module boundaries |
| `gradpath/config.py` | Load/validate `settings.yaml`, `profile.yaml`, `fields.yaml`, `institutions.yaml` |
| `gradpath/net/http.py` | The only network path: cache, rate limit, robots, retry |
| `gradpath/sources/openalex.py` | Abstract reconstruction, topic search, cursor-paginated work ingest |
| `gradpath/sources/crossref.py` | Corresponding-author email lookup |
| `gradpath/sources/orcid.py` | Public record: employment title, email |
| `gradpath/sources/crawler.py` | Generic faculty-page email extraction |
| `gradpath/sources/adapters/base.py` | `InstitutionAdapter` ABC + `FacultyRecord` |
| `gradpath/sources/adapters/__init__.py` | Adapter registry |
| `gradpath/sources/adapters/{kaist,gist,snu}.py` | Seeded institution adapters |
| `gradpath/sources/emails.py` | The resolution chain that orders the above |
| `gradpath/match/score.py` | Pure fit scoring: mean-of-top-3 |
| `gradpath/match/faculty.py` | Pure faculty-likelihood scoring |
| `gradpath/match/embed.py` | Batched embedding, BLOB cache, vectorised matching |
| `gradpath/match/rerank.py` | Budget-gated Claude pass |
| `gradpath/report/table.py` | Rich terminal rendering |
| `gradpath/report/export.py` | CSV and Markdown export |
| `gradpath/cli.py` | Typer command surface, wiring only |

`score.py` and `faculty.py` are deliberately pure — no database, no network — because they hold the decisions most worth testing.

---

### Task 1: Project skeleton and database

**Files:**
- Create: `pyproject.toml`, `LICENSE`, `README.md`
- Create: `gradpath/__init__.py`, `gradpath/util.py`, `gradpath/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: nothing
- Produces: `now_iso() -> str`; `connect(path: Path) -> sqlite3.Connection`; `migrate(conn: sqlite3.Connection) -> int` returning the schema version reached; `SCHEMA_VERSION: int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
import sqlite3
from pathlib import Path

from gradpath.db import SCHEMA_VERSION, connect, migrate

EXPECTED_TABLES = {
    "institutions", "people", "works", "authorships",
    "embeddings", "matches", "cursors", "fetch_log", "schema_version",
}


def test_migrate_creates_all_tables(tmp_path: Path) -> None:
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert EXPECTED_TABLES <= {r[0] for r in rows}


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    conn = connect(tmp_path / "t.db")
    assert migrate(conn) == SCHEMA_VERSION
    assert migrate(conn) == SCHEMA_VERSION  # second run is a no-op


def test_foreign_keys_are_enforced(tmp_path: Path) -> None:
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    try:
        conn.execute(
            "INSERT INTO authorships (person_id, work_id, position) VALUES (9999, 9999, 'first')"
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return
    raise AssertionError("foreign keys were not enforced")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gradpath'`

- [ ] **Step 3: Write the implementation**

```toml
# pyproject.toml
[project]
name = "gradpath"
version = "0.1.0"
description = "Find the professors worth emailing for graduate study"
requires-python = ">=3.11"
license = { text = "MIT" }
dependencies = [
    "typer>=0.12",
    "rich>=13.7",
    "httpx>=0.27",
    "pyyaml>=6.0",
    "numpy>=1.26",
    "beautifulsoup4>=4.12",
    "lxml>=5.2",
    "sentence-transformers>=3.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.2", "respx>=0.21", "ruff>=0.5"]

[project.scripts]
gradpath = "gradpath.cli:app"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```python
# gradpath/util.py
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
```

```python
# gradpath/db.py
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE institutions (
        id TEXT PRIMARY KEY, name TEXT NOT NULL, ror_id TEXT,
        openalex_id TEXT UNIQUE, country TEXT, site TEXT, adapter TEXT,
        discovered INTEGER DEFAULT 0, added_at TEXT NOT NULL
    );
    CREATE TABLE people (
        id INTEGER PRIMARY KEY,
        institution_id TEXT REFERENCES institutions(id),
        dept TEXT, name TEXT NOT NULL,
        openalex_author_id TEXT UNIQUE, orcid TEXT, title TEXT, homepage TEXT,
        email TEXT, email_confidence TEXT, email_source TEXT,
        works_count INTEGER DEFAULT 0, first_year INTEGER, last_year INTEGER,
        last_author_ratio REAL, first_author_ratio REAL,
        faculty_score REAL, faculty_confidence TEXT, faculty_signals TEXT,
        last_seen_at TEXT
    );
    CREATE TABLE works (
        id INTEGER PRIMARY KEY, openalex_work_id TEXT UNIQUE NOT NULL,
        title TEXT NOT NULL, abstract TEXT, year INTEGER, doi TEXT,
        venue TEXT, topics TEXT, cited_by INTEGER DEFAULT 0
    );
    CREATE TABLE authorships (
        person_id INTEGER NOT NULL REFERENCES people(id),
        work_id INTEGER NOT NULL REFERENCES works(id),
        position TEXT, PRIMARY KEY (person_id, work_id)
    );
    CREATE TABLE embeddings (
        work_id INTEGER PRIMARY KEY REFERENCES works(id),
        model TEXT NOT NULL, dim INTEGER NOT NULL, vector BLOB NOT NULL,
        source TEXT NOT NULL, computed_at TEXT NOT NULL
    );
    CREATE TABLE matches (
        profile_id TEXT NOT NULL,
        person_id INTEGER NOT NULL REFERENCES people(id),
        stage1_score REAL NOT NULL, stage2_score REAL, reason TEXT,
        top_work_ids TEXT, sparse INTEGER DEFAULT 0, computed_at TEXT NOT NULL,
        PRIMARY KEY (profile_id, person_id)
    );
    CREATE TABLE cursors (
        key TEXT PRIMARY KEY, cursor TEXT, updated_at TEXT
    );
    CREATE TABLE fetch_log (
        url TEXT PRIMARY KEY, fetched_at TEXT NOT NULL, status INTEGER, etag TEXT
    );
    CREATE INDEX idx_matches_score ON matches(profile_id, stage1_score DESC);
    CREATE INDEX idx_people_faculty ON people(faculty_confidence, faculty_score DESC);
    CREATE INDEX idx_works_year ON works(year);
    """,
}


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _current_version(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return row[0] or 0


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations transactionally. Forward-only. Returns version reached."""
    current = _current_version(conn)
    for version in sorted(MIGRATIONS):
        if version <= current:
            continue
        with conn:
            conn.executescript(MIGRATIONS[version])
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
    return _current_version(conn)
```

Write a `README.md` stating what the tool does, that it requires no API key for core use, and the install command `pip install -e ".[dev]"`. Write an MIT `LICENSE`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pip install -e ".[dev]" && pytest tests/test_db.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml LICENSE README.md gradpath/ tests/
git commit -m "feat: project skeleton and SQLite schema"
```

---

### Task 2: Configuration loading

**Files:**
- Create: `gradpath/config.py`, `gradpath/models.py`
- Create: `settings.yaml`, `data/fields.yaml`, `data/institutions.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `gradpath.util.slugify`
- Produces: `Settings` and `Profile` dataclasses; `load_settings(path) -> Settings`; `load_profile(path) -> Profile`; `scaffold_profile(path) -> None`; `load_fields(path) -> dict[str, FieldDef]`; `FieldDef(slug, label, topics, aliases)`; `resolve_field(name, fields) -> FieldDef`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest

from gradpath.config import (
    load_fields, load_profile, load_settings, resolve_field, scaffold_profile,
)

SETTINGS = """
embedding_model: sentence-transformers/all-MiniLM-L6-v2
embedding_batch_size: 256
rate_limit_per_host: 1.0
default_since_year: 2021
show_min_score: 0.65
rerank_model: claude-opus-5
rerank_default_budget_usd: 2.00
cache_dir: .cache
"""

PROFILE = """
name: default
contact_email: me@example.com
fields: [efficient-ml]
countries: [KR, JP]
interests: on-device inference
keywords: [quantization]
seed_papers: ['10.1145/3458864.3467882']
my_papers: []
"""

FIELDS = """
efficient-ml:
  label: Efficient machine learning
  topics: [T10028, T11689]
  aliases: [edge-ml, on-device-ml]
"""


def test_loads_settings(tmp_path):
    p = tmp_path / "settings.yaml"
    p.write_text(SETTINGS)
    s = load_settings(p)
    assert s.embedding_batch_size == 256
    assert s.show_min_score == 0.65


def test_rate_limit_may_not_be_raised(tmp_path):
    p = tmp_path / "settings.yaml"
    p.write_text(SETTINGS.replace("rate_limit_per_host: 1.0", "rate_limit_per_host: 20.0"))
    with pytest.raises(ValueError, match="rate_limit_per_host"):
        load_settings(p)


def test_profile_requires_contact_email(tmp_path):
    p = tmp_path / "profile.yaml"
    p.write_text(PROFILE.replace("contact_email: me@example.com", "contact_email: ''"))
    with pytest.raises(ValueError, match="contact_email"):
        load_profile(p)


def test_profile_requires_at_least_one_interest_input(tmp_path):
    p = tmp_path / "profile.yaml"
    p.write_text(
        "name: d\ncontact_email: me@example.com\ninterests: ''\n"
        "keywords: []\nseed_papers: []\nmy_papers: []\n"
    )
    with pytest.raises(ValueError, match="interests"):
        load_profile(p)


def test_scaffold_then_load_roundtrips(tmp_path):
    p = tmp_path / "profile.yaml"
    scaffold_profile(p)
    assert p.exists()
    assert "contact_email" in p.read_text()


def test_resolve_field_by_alias(tmp_path):
    p = tmp_path / "fields.yaml"
    p.write_text(FIELDS)
    fields = load_fields(p)
    assert resolve_field("edge-ml", fields).topics == ["T10028", "T11689"]
    assert resolve_field("efficient-ml", fields).label == "Efficient machine learning"


def test_resolve_unknown_field_raises(tmp_path):
    p = tmp_path / "fields.yaml"
    p.write_text(FIELDS)
    with pytest.raises(KeyError, match="nonsense"):
        resolve_field("nonsense", load_fields(p))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gradpath.config'`

- [ ] **Step 3: Write the implementation**

```python
# gradpath/models.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldDef:
    slug: str
    label: str
    topics: list[str]
    aliases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Settings:
    embedding_model: str
    embedding_batch_size: int
    rate_limit_per_host: float
    default_since_year: int
    show_min_score: float
    rerank_model: str
    rerank_default_budget_usd: float
    cache_dir: str


@dataclass(frozen=True)
class Profile:
    name: str
    contact_email: str
    interests: str
    keywords: list[str]
    seed_papers: list[str]
    my_papers: list[str]
    fields: list[str]
    countries: list[str]


@dataclass(frozen=True)
class FacultyRecord:
    """One row parsed from an institution's faculty directory."""
    name: str
    email: str | None
    title: str | None
    homepage: str | None
    dept: str | None
```

```python
# gradpath/config.py
from __future__ import annotations

from pathlib import Path

import yaml

from gradpath.models import FieldDef, Profile, Settings

MAX_RATE_LIMIT = 1.0

PROFILE_TEMPLATE = """\
# gradpath profile — this file is gitignored, keep it that way.
name: default
contact_email: ""          # REQUIRED: used in the polite User-Agent
fields: []                 # e.g. [efficient-ml]
countries: []              # e.g. [KR, JP, DE]

interests: >
  Describe in a paragraph what you want to work on.

keywords: []
seed_papers: []            # DOIs of papers you admire — the strongest signal
my_papers: []
"""


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return yaml.safe_load(path.read_text()) or {}


def load_settings(path: Path) -> Settings:
    raw = _read_yaml(path)
    rate = float(raw.get("rate_limit_per_host", MAX_RATE_LIMIT))
    if rate > MAX_RATE_LIMIT:
        raise ValueError(
            f"rate_limit_per_host may be lowered but not raised above {MAX_RATE_LIMIT}"
        )
    return Settings(
        embedding_model=raw["embedding_model"],
        embedding_batch_size=int(raw["embedding_batch_size"]),
        rate_limit_per_host=rate,
        default_since_year=int(raw["default_since_year"]),
        show_min_score=float(raw["show_min_score"]),
        rerank_model=raw["rerank_model"],
        rerank_default_budget_usd=float(raw["rerank_default_budget_usd"]),
        cache_dir=raw.get("cache_dir", ".cache"),
    )


def load_profile(path: Path) -> Profile:
    raw = _read_yaml(path)
    contact = (raw.get("contact_email") or "").strip()
    if not contact:
        raise ValueError(
            "contact_email is required in profile.yaml — it identifies you to the "
            "APIs and sites this tool contacts"
        )
    interests = (raw.get("interests") or "").strip()
    keywords = raw.get("keywords") or []
    seed = raw.get("seed_papers") or []
    mine = raw.get("my_papers") or []
    if not (interests or keywords or seed or mine):
        raise ValueError(
            "profile needs at least one of: interests, keywords, seed_papers, my_papers"
        )
    return Profile(
        name=raw.get("name", "default"),
        contact_email=contact,
        interests=interests,
        keywords=list(keywords),
        seed_papers=list(seed),
        my_papers=list(mine),
        fields=list(raw.get("fields") or []),
        countries=[c.upper() for c in (raw.get("countries") or [])],
    )


def scaffold_profile(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"{path} already exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PROFILE_TEMPLATE)


def load_fields(path: Path) -> dict[str, FieldDef]:
    raw = _read_yaml(path)
    return {
        slug: FieldDef(
            slug=slug,
            label=body["label"],
            topics=list(body["topics"]),
            aliases=list(body.get("aliases") or []),
        )
        for slug, body in raw.items()
    }


def resolve_field(name: str, fields: dict[str, FieldDef]) -> FieldDef:
    key = name.strip().lower()
    if key in fields:
        return fields[key]
    for definition in fields.values():
        if key in {a.lower() for a in definition.aliases}:
            return definition
    raise KeyError(
        f"unknown field {name!r} — run `gradpath fields search` to find its OpenAlex topics"
    )
```

Write `settings.yaml` with exactly the values from the spec. Write `data/fields.yaml` with the `efficient-ml` entry as a worked example. Write `data/institutions.yaml` with KAIST, GIST and SNU entries including `ror_id`, `country`, `site` and `adapter`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add gradpath/config.py gradpath/models.py settings.yaml data/ tests/test_config.py
git commit -m "feat: configuration loading with profile and field resolution"
```

---

### Task 3: Network chokepoint

**Files:**
- Create: `gradpath/net/__init__.py`, `gradpath/net/http.py`
- Test: `tests/test_http.py`, `tests/test_architecture.py`

**Interfaces:**
- Consumes: `Settings`, `now_iso`
- Produces: `PoliteClient(settings, contact_email, cache_dir)` with `.get_json(url, params=None) -> dict`, `.get_text(url) -> str | None` (None when robots disallows), `.allowed(url) -> bool`; `RateLimiter`; exception `HostBlocked`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_http.py
import httpx
import pytest
import respx

from gradpath.models import Settings
from gradpath.net.http import PoliteClient

SETTINGS = Settings(
    embedding_model="m", embedding_batch_size=8, rate_limit_per_host=1000.0,
    default_since_year=2021, show_min_score=0.6, rerank_model="claude-opus-5",
    rerank_default_budget_usd=1.0, cache_dir=".cache",
)
ROBOTS_ALLOW = "User-agent: *\nAllow: /\n"
ROBOTS_DENY = "User-agent: *\nDisallow: /private\n"


@pytest.fixture
def client(tmp_path):
    # rate limit relaxed in tests so the suite does not sleep
    return PoliteClient(SETTINGS, "me@example.com", tmp_path / "cache")


@respx.mock
def test_user_agent_carries_contact_email(client):
    route = respx.get("https://api.example.com/x").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client.get_json("https://api.example.com/x")
    assert "me@example.com" in route.calls[0].request.headers["user-agent"]


@respx.mock
def test_second_identical_request_is_served_from_cache(client):
    route = respx.get("https://api.example.com/y").mock(
        return_value=httpx.Response(200, json={"n": 1})
    )
    assert client.get_json("https://api.example.com/y") == {"n": 1}
    assert client.get_json("https://api.example.com/y") == {"n": 1}
    assert route.call_count == 1


@respx.mock
def test_robots_disallow_returns_none(client):
    respx.get("https://site.example.com/robots.txt").mock(
        return_value=httpx.Response(200, text=ROBOTS_DENY)
    )
    assert client.get_text("https://site.example.com/private/staff") is None


@respx.mock
def test_robots_allow_permits_fetch(client):
    respx.get("https://site.example.com/robots.txt").mock(
        return_value=httpx.Response(200, text=ROBOTS_ALLOW)
    )
    respx.get("https://site.example.com/staff").mock(
        return_value=httpx.Response(200, text="<html>hi</html>")
    )
    assert client.get_text("https://site.example.com/staff") == "<html>hi</html>"


@respx.mock
def test_missing_robots_is_treated_as_allowed(client):
    respx.get("https://site2.example.com/robots.txt").mock(
        return_value=httpx.Response(404)
    )
    respx.get("https://site2.example.com/staff").mock(
        return_value=httpx.Response(200, text="ok")
    )
    assert client.get_text("https://site2.example.com/staff") == "ok"


@respx.mock
def test_retries_on_server_error_then_succeeds(client):
    respx.get("https://api.example.com/z").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json={"ok": 1})]
    )
    assert client.get_json("https://api.example.com/z") == {"ok": 1}
```

```python
# tests/test_architecture.py
"""The single-network-chokepoint constraint is load-bearing, so it is tested."""
import pathlib

FORBIDDEN = ("import httpx", "import requests", "from httpx", "from requests",
             "import urllib.request")
ROOT = pathlib.Path(__file__).resolve().parents[1] / "gradpath"


def test_only_net_module_imports_an_http_client():
    offenders = []
    for path in ROOT.rglob("*.py"):
        if path.parent.name == "net":
            continue
        text = path.read_text()
        if any(token in text for token in FORBIDDEN):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], f"HTTP client imported outside net/: {offenders}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_http.py tests/test_architecture.py -v`
Expected: `test_http.py` fails with `ModuleNotFoundError: No module named 'gradpath.net'`

- [ ] **Step 3: Write the implementation**

```python
# gradpath/net/http.py
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
from gradpath.util import now_iso

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
            try:
                response = self._client.get(f"{origin}/robots.txt")
                if response.status_code == 200:
                    parser.parse(response.text.splitlines())
                else:
                    parser = None  # absent robots.txt means unrestricted
            except httpx.HTTPError:
                parser = None
            self._robots[origin] = parser
        parser = self._robots[origin]
        return True if parser is None else parser.can_fetch(self.user_agent, url)

    # --- fetching ----------------------------------------------------------

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
                time.sleep(float(response.headers.get("Retry-After", BACKOFF_BASE_SECONDS)))
                continue
            if response.status_code >= 500:
                time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
                continue
            self._failures[host] = 0
            return response
        self._failures[host] += 1
        raise HostBlocked(f"{url} failed after {MAX_ATTEMPTS} attempts: {last_error}")

    def get_json(self, url: str, params: dict | None = None) -> dict:
        """Fetch JSON, serving from the on-disk cache when present."""
        cached = self._cache_path(url, params)
        if cached.exists():
            return json.loads(cached.read_text())
        payload = self._request(url, params).json()
        cached.write_text(json.dumps({"_fetched_at": now_iso(), **payload}))
        return payload

    def get_text(self, url: str) -> str | None:
        """Fetch a page, honouring robots.txt. Returns None when disallowed."""
        if not self.allowed(url):
            return None
        return self._request(url, None).text

    def close(self) -> None:
        self._client.close()
```

Note the cache write adds `_fetched_at` for provenance, but `get_json` returns the parsed payload from the network path and the raw cached dict on the cache path; the extra key is inert for callers that read named fields.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_http.py tests/test_architecture.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add gradpath/net/ tests/test_http.py tests/test_architecture.py
git commit -m "feat: polite HTTP client as the single network chokepoint"
```

---

### Task 4: OpenAlex abstract reconstruction and topic search

**Files:**
- Create: `gradpath/sources/__init__.py`, `gradpath/sources/openalex.py`
- Test: `tests/test_openalex_parse.py`

**Interfaces:**
- Consumes: `PoliteClient`
- Produces: `reconstruct_abstract(index: dict | None) -> str | None`; `parse_work(raw: dict) -> ParsedWork`; `ParsedWork(openalex_id, title, abstract, year, doi, venue, topics, cited_by, authorships)`; `ParsedAuthorship(author_id, name, orcid, position, institution_openalex_id, institution_name, institution_country)`; `search_topics(client, query) -> list[TopicHit]`; `OPENALEX_BASE`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_openalex_parse.py
import httpx
import respx

from gradpath.sources.openalex import parse_work, reconstruct_abstract, search_topics

WORK = {
    "id": "https://openalex.org/W123",
    "title": "Quantized inference on mobile devices",
    "publication_year": 2024,
    "doi": "https://doi.org/10.1145/1234",
    "cited_by_count": 17,
    "primary_location": {"source": {"display_name": "NeurIPS"}},
    "topics": [{"id": "https://openalex.org/T10028"}],
    "abstract_inverted_index": {"We": [0], "quantize": [1], "models": [2]},
    "authorships": [
        {
            "author_position": "first",
            "author": {"id": "https://openalex.org/A1", "display_name": "J. Kim",
                       "orcid": "https://orcid.org/0000-0001-0000-0001"},
            "institutions": [{"id": "https://openalex.org/I1",
                              "display_name": "KAIST", "country_code": "KR"}],
        },
        {
            "author_position": "last",
            "author": {"id": "https://openalex.org/A2", "display_name": "S. Park",
                       "orcid": None},
            "institutions": [],
        },
    ],
}


def test_reconstruct_abstract_orders_words_by_position():
    assert reconstruct_abstract({"b": [1], "a": [0], "c": [2]}) == "a b c"


def test_reconstruct_abstract_handles_repeated_words():
    assert reconstruct_abstract({"the": [0, 2], "cat": [1]}) == "the cat the"


def test_reconstruct_abstract_returns_none_when_absent():
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None


def test_parse_work_extracts_core_fields():
    w = parse_work(WORK)
    assert w.openalex_id == "W123"
    assert w.abstract == "We quantize models"
    assert w.year == 2024
    assert w.doi == "10.1145/1234"
    assert w.venue == "NeurIPS"
    assert w.topics == ["T10028"]
    assert w.cited_by == 17


def test_parse_work_extracts_authorships_with_position_and_institution():
    w = parse_work(WORK)
    assert [a.author_id for a in w.authorships] == ["A1", "A2"]
    first = w.authorships[0]
    assert first.position == "first"
    assert first.institution_openalex_id == "I1"
    assert first.institution_country == "KR"
    assert first.orcid == "0000-0001-0000-0001"
    assert w.authorships[1].institution_openalex_id is None


@respx.mock
def test_search_topics_returns_id_and_label(tmp_path):
    from gradpath.models import Settings
    from gradpath.net.http import PoliteClient

    settings = Settings("m", 8, 1000.0, 2021, 0.6, "claude-opus-5", 1.0, ".cache")
    client = PoliteClient(settings, "me@example.com", tmp_path / "c")
    respx.get("https://api.openalex.org/topics").mock(
        return_value=httpx.Response(200, json={"results": [
            {"id": "https://openalex.org/T10028",
             "display_name": "Efficient ML", "works_count": 4211},
        ]})
    )
    hits = search_topics(client, "efficient machine learning")
    assert hits[0].topic_id == "T10028"
    assert hits[0].label == "Efficient ML"
    assert hits[0].works_count == 4211
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_openalex_parse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gradpath.sources'`

- [ ] **Step 3: Write the implementation**

```python
# gradpath/sources/openalex.py
from __future__ import annotations

from dataclasses import dataclass, field

from gradpath.net.http import PoliteClient

OPENALEX_BASE = "https://api.openalex.org"


@dataclass(frozen=True)
class ParsedAuthorship:
    author_id: str
    name: str
    orcid: str | None
    position: str
    institution_openalex_id: str | None
    institution_name: str | None
    institution_country: str | None


@dataclass(frozen=True)
class ParsedWork:
    openalex_id: str
    title: str
    abstract: str | None
    year: int | None
    doi: str | None
    venue: str | None
    topics: list[str]
    cited_by: int
    authorships: list[ParsedAuthorship] = field(default_factory=list)


@dataclass(frozen=True)
class TopicHit:
    topic_id: str
    label: str
    works_count: int


def _short_id(url: str | None) -> str | None:
    """OpenAlex ids arrive as URLs; the trailing segment is the usable id."""
    return url.rsplit("/", 1)[-1] if url else None


def reconstruct_abstract(index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stores abstracts as an inverted index; rebuild the running text."""
    if not index:
        return None
    positioned: list[tuple[int, str]] = [
        (position, word) for word, positions in index.items() for position in positions
    ]
    if not positioned:
        return None
    positioned.sort(key=lambda pair: pair[0])
    return " ".join(word for _, word in positioned)


def parse_work(raw: dict) -> ParsedWork:
    location = raw.get("primary_location") or {}
    source = location.get("source") or {}
    doi = raw.get("doi")
    authorships = []
    for entry in raw.get("authorships") or []:
        author = entry.get("author") or {}
        institutions = entry.get("institutions") or []
        institution = institutions[0] if institutions else {}
        authorships.append(
            ParsedAuthorship(
                author_id=_short_id(author.get("id")) or "",
                name=author.get("display_name") or "",
                orcid=_short_id(author.get("orcid")),
                position=entry.get("author_position") or "middle",
                institution_openalex_id=_short_id(institution.get("id")),
                institution_name=institution.get("display_name"),
                institution_country=institution.get("country_code"),
            )
        )
    return ParsedWork(
        openalex_id=_short_id(raw.get("id")) or "",
        title=raw.get("title") or "",
        abstract=reconstruct_abstract(raw.get("abstract_inverted_index")),
        year=raw.get("publication_year"),
        doi=doi.replace("https://doi.org/", "") if doi else None,
        venue=source.get("display_name"),
        topics=[_short_id(t.get("id")) or "" for t in (raw.get("topics") or [])],
        cited_by=raw.get("cited_by_count") or 0,
        authorships=authorships,
    )


def search_topics(client: PoliteClient, query: str) -> list[TopicHit]:
    payload = client.get_json(f"{OPENALEX_BASE}/topics", {"search": query, "per_page": 25})
    return [
        TopicHit(
            topic_id=_short_id(item.get("id")) or "",
            label=item.get("display_name") or "",
            works_count=item.get("works_count") or 0,
        )
        for item in payload.get("results", [])
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_openalex_parse.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add gradpath/sources/ tests/test_openalex_parse.py
git commit -m "feat: OpenAlex work parsing and topic search"
```

---

### Task 5: Field-first discovery with resumable cursor

**Files:**
- Modify: `gradpath/sources/openalex.py` (append ingest functions)
- Create: `gradpath/sources/ingest.py`
- Test: `tests/test_discover.py`

**Interfaces:**
- Consumes: `PoliteClient`, `parse_work`, `connect`, `migrate`
- Produces: `iter_works(client, topics, countries, since, cursor_key, conn) -> Iterator[ParsedWork]`; `ingest_work(conn, work) -> int` returning work row id; `upsert_person(conn, authorship) -> int`; `discover(conn, client, topics, countries, since, cursor_key) -> DiscoverStats`; `DiscoverStats(works_seen, people_seen, institutions_added)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_discover.py
import httpx
import respx

from gradpath.db import connect, migrate
from gradpath.models import Settings
from gradpath.net.http import PoliteClient
from gradpath.sources.ingest import discover

SETTINGS = Settings("m", 8, 1000.0, 2021, 0.6, "claude-opus-5", 1.0, ".cache")


def _work(wid: str, author_id: str, position: str = "last") -> dict:
    return {
        "id": f"https://openalex.org/{wid}",
        "title": f"Paper {wid}",
        "publication_year": 2024,
        "doi": None,
        "cited_by_count": 1,
        "primary_location": {"source": {"display_name": "Venue"}},
        "topics": [{"id": "https://openalex.org/T10028"}],
        "abstract_inverted_index": {"hello": [0], "world": [1]},
        "authorships": [{
            "author_position": position,
            "author": {"id": f"https://openalex.org/{author_id}",
                       "display_name": "A Person", "orcid": None},
            "institutions": [{"id": "https://openalex.org/I1",
                              "display_name": "KAIST", "country_code": "KR"}],
        }],
    }


def _page(works: list[dict], next_cursor: str | None) -> dict:
    return {"results": works, "meta": {"next_cursor": next_cursor}}


@respx.mock
def test_discover_paginates_and_stores(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    client = PoliteClient(SETTINGS, "me@example.com", tmp_path / "c")
    respx.get("https://api.openalex.org/works").mock(side_effect=[
        httpx.Response(200, json=_page([_work("W1", "A1"), _work("W2", "A1")], "cur2")),
        httpx.Response(200, json=_page([_work("W3", "A2")], None)),
    ])
    stats = discover(conn, client, ["T10028"], ["KR"], 2021, "k")
    assert stats.works_seen == 3
    assert stats.people_seen == 2
    assert conn.execute("SELECT COUNT(*) FROM works").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM people").fetchone()[0] == 2


@respx.mock
def test_discover_records_discovered_institution(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    client = PoliteClient(SETTINGS, "me@example.com", tmp_path / "c")
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json=_page([_work("W1", "A1")], None))
    )
    discover(conn, client, ["T10028"], ["KR"], 2021, "k")
    row = conn.execute("SELECT id, discovered, country FROM institutions").fetchone()
    assert row["discovered"] == 1
    assert row["country"] == "KR"


@respx.mock
def test_rerunning_discover_is_idempotent(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    client = PoliteClient(SETTINGS, "me@example.com", tmp_path / "c1")
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json=_page([_work("W1", "A1")], None))
    )
    discover(conn, client, ["T10028"], ["KR"], 2021, "k")
    conn.execute("UPDATE cursors SET cursor = '*' WHERE key = 'k'")
    discover(conn, client, ["T10028"], ["KR"], 2021, "k")
    assert conn.execute("SELECT COUNT(*) FROM works").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM authorships").fetchone()[0] == 1


@respx.mock
def test_author_position_counters_accumulate(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    client = PoliteClient(SETTINGS, "me@example.com", tmp_path / "c")
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json=_page(
            [_work("W1", "A1", "last"), _work("W2", "A1", "first")], None))
    )
    discover(conn, client, ["T10028"], ["KR"], 2021, "k")
    row = conn.execute("SELECT works_count FROM people WHERE openalex_author_id='A1'").fetchone()
    assert row["works_count"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_discover.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gradpath.sources.ingest'`

- [ ] **Step 3: Write the implementation**

```python
# gradpath/sources/ingest.py
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Iterator

from gradpath.net.http import PoliteClient
from gradpath.sources.openalex import (
    OPENALEX_BASE, ParsedAuthorship, ParsedWork, parse_work,
)
from gradpath.util import now_iso, slugify

PAGE_SIZE = 200


@dataclass
class DiscoverStats:
    works_seen: int = 0
    people_seen: int = 0
    institutions_added: int = 0


def _load_cursor(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT cursor FROM cursors WHERE key = ?", (key,)).fetchone()
    return row["cursor"] if row else "*"


def _save_cursor(conn: sqlite3.Connection, key: str, cursor: str | None) -> None:
    with conn:
        conn.execute(
            "INSERT INTO cursors (key, cursor, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET cursor = excluded.cursor, "
            "updated_at = excluded.updated_at",
            (key, cursor, now_iso()),
        )


def iter_works(
    client: PoliteClient,
    topics: list[str],
    countries: list[str],
    since: int,
    cursor_key: str,
    conn: sqlite3.Connection,
    institutions: list[str] | None = None,
) -> Iterator[ParsedWork]:
    """Stream works matching the filter, persisting the cursor after every page."""
    filters = [f"publication_year:>{since - 1}"]
    if topics:
        filters.append("topics.id:" + "|".join(topics))
    if institutions:
        filters.append("institutions.id:" + "|".join(institutions))
    elif countries:
        filters.append("institutions.country_code:" + "|".join(c.lower() for c in countries))

    cursor = _load_cursor(conn, cursor_key)
    while cursor:
        payload = client.get_json(
            f"{OPENALEX_BASE}/works",
            {"filter": ",".join(filters), "per-page": PAGE_SIZE, "cursor": cursor},
        )
        for raw in payload.get("results", []):
            yield parse_work(raw)
        cursor = (payload.get("meta") or {}).get("next_cursor")
        _save_cursor(conn, cursor_key, cursor)


def _upsert_institution(conn: sqlite3.Connection, a: ParsedAuthorship) -> tuple[str | None, bool]:
    if not a.institution_openalex_id:
        return None, False
    row = conn.execute(
        "SELECT id FROM institutions WHERE openalex_id = ?", (a.institution_openalex_id,)
    ).fetchone()
    if row:
        return row["id"], False
    slug = slugify(a.institution_name or a.institution_openalex_id)
    conn.execute(
        "INSERT OR IGNORE INTO institutions "
        "(id, name, openalex_id, country, discovered, added_at) VALUES (?, ?, ?, ?, 1, ?)",
        (slug, a.institution_name, a.institution_openalex_id, a.institution_country, now_iso()),
    )
    return slug, True


def upsert_person(conn: sqlite3.Connection, a: ParsedAuthorship, institution_id: str | None) -> int:
    row = conn.execute(
        "SELECT id FROM people WHERE openalex_author_id = ?", (a.author_id,)
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE people SET last_seen_at = ?, institution_id = COALESCE(?, institution_id), "
            "orcid = COALESCE(?, orcid) WHERE id = ?",
            (now_iso(), institution_id, a.orcid, row["id"]),
        )
        return row["id"]
    cursor = conn.execute(
        "INSERT INTO people (institution_id, name, openalex_author_id, orcid, last_seen_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (institution_id, a.name, a.author_id, a.orcid, now_iso()),
    )
    return int(cursor.lastrowid)


def ingest_work(conn: sqlite3.Connection, work: ParsedWork) -> int:
    row = conn.execute(
        "SELECT id FROM works WHERE openalex_work_id = ?", (work.openalex_id,)
    ).fetchone()
    if row:
        return row["id"]
    cursor = conn.execute(
        "INSERT INTO works (openalex_work_id, title, abstract, year, doi, venue, topics, cited_by)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (work.openalex_id, work.title, work.abstract, work.year, work.doi,
         work.venue, json.dumps(work.topics), work.cited_by),
    )
    return int(cursor.lastrowid)


def _refresh_person_stats(conn: sqlite3.Connection, person_id: int) -> None:
    """Recompute cached publication statistics used later by faculty scoring."""
    conn.execute(
        """
        UPDATE people SET
            works_count = (SELECT COUNT(*) FROM authorships WHERE person_id = :pid),
            first_year = (SELECT MIN(w.year) FROM authorships a JOIN works w ON w.id = a.work_id
                          WHERE a.person_id = :pid),
            last_year  = (SELECT MAX(w.year) FROM authorships a JOIN works w ON w.id = a.work_id
                          WHERE a.person_id = :pid),
            last_author_ratio = (
                SELECT CAST(SUM(position = 'last') AS REAL) / COUNT(*)
                FROM authorships WHERE person_id = :pid),
            first_author_ratio = (
                SELECT CAST(SUM(position = 'first') AS REAL) / COUNT(*)
                FROM authorships WHERE person_id = :pid)
        WHERE id = :pid
        """,
        {"pid": person_id},
    )


def discover(
    conn: sqlite3.Connection,
    client: PoliteClient,
    topics: list[str],
    countries: list[str],
    since: int,
    cursor_key: str,
    institutions: list[str] | None = None,
) -> DiscoverStats:
    stats = DiscoverStats()
    touched: set[int] = set()
    for work in iter_works(client, topics, countries, since, cursor_key, conn, institutions):
        with conn:
            work_id = ingest_work(conn, work)
            stats.works_seen += 1
            for authorship in work.authorships:
                if not authorship.author_id:
                    continue
                institution_id, added = _upsert_institution(conn, authorship)
                stats.institutions_added += int(added)
                person_id = upsert_person(conn, authorship, institution_id)
                conn.execute(
                    "INSERT OR IGNORE INTO authorships (person_id, work_id, position) "
                    "VALUES (?, ?, ?)",
                    (person_id, work_id, authorship.position),
                )
                touched.add(person_id)
    with conn:
        for person_id in touched:
            _refresh_person_stats(conn, person_id)
    stats.people_seen = len(touched)
    return stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_discover.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add gradpath/sources/ingest.py tests/test_discover.py
git commit -m "feat: field-first discovery with resumable cursor"
```

---

### Task 6: Fit scoring (pure)

**Files:**
- Create: `gradpath/match/__init__.py`, `gradpath/match/score.py`
- Test: `tests/test_score.py`

**Interfaces:**
- Consumes: nothing (pure, NumPy only)
- Produces: `PersonScore(score: float, sparse: bool, top_work_ids: list[int])`; `score_person(similarities: dict[int, float], k: int = 3) -> PersonScore`; `cosine_matrix(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray`; `TOP_K: int = 3`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_score.py
import numpy as np
import pytest

from gradpath.match.score import TOP_K, cosine_matrix, score_person


def test_score_is_mean_of_top_three():
    result = score_person({1: 0.9, 2: 0.8, 3: 0.7, 4: 0.1, 5: 0.0})
    assert result.score == pytest.approx(0.8)
    assert result.sparse is False


def test_prolific_author_is_not_diluted_by_irrelevant_work():
    """The whole point of mean-of-top-3: 200 unrelated papers must not bury 3 great ones."""
    focused = score_person({1: 0.9, 2: 0.85, 3: 0.8})
    prolific = score_person({1: 0.9, 2: 0.85, 3: 0.8, **{i: 0.05 for i in range(4, 204)}})
    assert prolific.score == pytest.approx(focused.score)


def test_single_lucky_match_does_not_dominate():
    """Mean-of-top-3, not max: one coincidental abstract should not promote someone."""
    lucky = score_person({1: 0.95, 2: 0.10, 3: 0.05})
    steady = score_person({1: 0.70, 2: 0.68, 3: 0.66})
    assert steady.score > lucky.score


def test_top_work_ids_are_the_contributing_works():
    result = score_person({7: 0.2, 8: 0.9, 9: 0.5, 10: 0.7})
    assert result.top_work_ids == [8, 10, 9]


def test_fewer_than_k_works_is_marked_sparse_without_padding():
    result = score_person({1: 0.8, 2: 0.6})
    assert result.score == pytest.approx(0.7)
    assert result.sparse is True


def test_single_work_is_sparse():
    result = score_person({1: 0.5})
    assert result.score == pytest.approx(0.5)
    assert result.sparse is True


def test_no_works_raises():
    with pytest.raises(ValueError, match="no works"):
        score_person({})


def test_cosine_matrix_matches_manual_computation():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    vector = np.array([1.0, 0.0], dtype=np.float32)
    got = cosine_matrix(matrix, vector)
    assert got == pytest.approx([1.0, 0.0, 0.7071], abs=1e-4)


def test_cosine_matrix_tolerates_zero_vectors():
    matrix = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    vector = np.array([1.0, 0.0], dtype=np.float32)
    assert cosine_matrix(matrix, vector)[0] == pytest.approx(0.0)


def test_top_k_is_three():
    assert TOP_K == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gradpath.match'`

- [ ] **Step 3: Write the implementation**

```python
# gradpath/match/score.py
"""Fit scoring. Pure functions only — no database, no network.

The central decision here is mean-of-top-3. Averaging every work regresses
prolific researchers to the mean and buries the professor with three papers
precisely on your topic. A single maximum is the opposite failure: one
coincidentally-worded abstract promotes someone unrelated. Mean-of-top-3
rewards sustained overlap while staying robust to a lucky match.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TOP_K = 3


@dataclass(frozen=True)
class PersonScore:
    score: float
    sparse: bool
    top_work_ids: list[int]


def score_person(similarities: dict[int, float], k: int = TOP_K) -> PersonScore:
    """Score one person from their work-level similarities.

    `similarities` maps work row id to cosine similarity. People with fewer
    than `k` works are scored on what exists, with no padding, and flagged
    sparse so callers can surface that rather than silently ranking them low.
    """
    if not similarities:
        raise ValueError("cannot score a person with no works")
    ranked = sorted(similarities.items(), key=lambda pair: pair[1], reverse=True)
    top = ranked[:k]
    return PersonScore(
        score=float(sum(value for _, value in top) / len(top)),
        sparse=len(ranked) < k,
        top_work_ids=[work_id for work_id, _ in top],
    )


def cosine_matrix(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Cosine similarity of every row in `matrix` against `vector`.

    One matrix multiply over the whole corpus. Scoring must never loop over
    people in Python — the scale test in tests/test_scale.py guards this.
    """
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    denominator = np.where(matrix_norms == 0.0, 1.0, matrix_norms) * vector_norm
    return (matrix @ vector) / denominator
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_score.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add gradpath/match/ tests/test_score.py
git commit -m "feat: mean-of-top-3 fit scoring"
```

---

### Task 7: Faculty likelihood scoring (pure)

**Files:**
- Create: `gradpath/match/faculty.py`
- Test: `tests/test_faculty.py`

**Interfaces:**
- Consumes: nothing (pure)
- Produces: `PublicationHistory(first_year, last_year, works_count, last_author_count, first_author_count, directory_confirmed=False, orcid_title=None, alphabetical_field=False)`; `FacultyAssessment(score: float, confidence: str, signals: dict)`; `score_faculty(history) -> FacultyAssessment`; `PROFESSOR_TITLE_TERMS`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_faculty.py
import pytest

from gradpath.match.faculty import PublicationHistory, score_faculty

CLEAR_PI = PublicationHistory(
    first_year=2005, last_year=2026, works_count=120,
    last_author_count=80, first_author_count=5,
)
PHD_STUDENT = PublicationHistory(
    first_year=2022, last_year=2026, works_count=6,
    last_author_count=0, first_author_count=4,
)
ALPHABETICAL_MATHEMATICIAN = PublicationHistory(
    first_year=2001, last_year=2026, works_count=60,
    last_author_count=15, first_author_count=20, alphabetical_field=True,
)
AMBIGUOUS_MID_CAREER = PublicationHistory(
    first_year=2016, last_year=2026, works_count=25,
    last_author_count=8, first_author_count=6,
)


def test_clear_pi_scores_high():
    assert score_faculty(CLEAR_PI).confidence == "high"


def test_phd_student_scores_low():
    assert score_faculty(PHD_STUDENT).confidence == "low"


def test_alphabetical_field_is_not_penalised_for_author_position():
    """Maths and economics list authors alphabetically; last-author means nothing there."""
    assessment = score_faculty(ALPHABETICAL_MATHEMATICIAN)
    assert assessment.confidence == "high"
    assert assessment.signals["alphabetical_field"] is True
    assert "last_author_ratio" not in assessment.signals


def test_ambiguous_mid_career_scores_medium():
    assert score_faculty(AMBIGUOUS_MID_CAREER).confidence == "medium"


def test_directory_confirmation_overrides_a_weak_heuristic():
    """An authoritative signal must beat the heuristic, never be averaged with it."""
    confirmed = PublicationHistory(
        first_year=2024, last_year=2026, works_count=2,
        last_author_count=0, first_author_count=2, directory_confirmed=True,
    )
    assessment = score_faculty(confirmed)
    assert assessment.confidence == "high"
    assert assessment.signals["directory_confirmed"] is True


def test_orcid_professor_title_overrides_a_weak_heuristic():
    titled = PublicationHistory(
        first_year=2024, last_year=2026, works_count=3,
        last_author_count=0, first_author_count=3,
        orcid_title="Associate Professor of Computer Science",
    )
    assert score_faculty(titled).confidence == "high"


def test_orcid_student_title_does_not_confer_faculty_status():
    titled = PublicationHistory(
        first_year=2024, last_year=2026, works_count=3,
        last_author_count=0, first_author_count=3,
        orcid_title="PhD Candidate",
    )
    assert score_faculty(titled).confidence == "low"


def test_signals_explain_the_score():
    signals = score_faculty(CLEAR_PI).signals
    assert signals["career_span"] == 21
    assert signals["works_count"] == 120
    assert signals["last_author_ratio"] == pytest.approx(0.667, abs=1e-3)


def test_zero_works_does_not_divide_by_zero():
    empty = PublicationHistory(
        first_year=2026, last_year=2026, works_count=0,
        last_author_count=0, first_author_count=0,
    )
    assert score_faculty(empty).score == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_faculty.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gradpath.match.faculty'`

- [ ] **Step 3: Write the implementation**

```python
# gradpath/match/faculty.py
"""Faculty-likelihood scoring. Pure functions only.

Most OpenAlex authors at a university are PhD students and postdocs, and
OpenAlex does not label who is faculty. This module estimates it — and
deliberately never removes anyone. A heuristic that silently hides the right
professor is the worst failure this tool could have; showing a ranked PhD
student costs the reader five seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PROFESSOR_TITLE_TERMS = (
    "professor", "faculty", "lecturer", "reader",
    "principal investigator", "group leader", "chair",
)

SPAN_FULL_CREDIT_YEARS = 12.0
VOLUME_FULL_CREDIT_WORKS = 40.0

HIGH_THRESHOLD = 0.7
MEDIUM_THRESHOLD = 0.4


@dataclass(frozen=True)
class PublicationHistory:
    first_year: int
    last_year: int
    works_count: int
    last_author_count: int
    first_author_count: int
    directory_confirmed: bool = False
    orcid_title: str | None = None
    alphabetical_field: bool = False


@dataclass(frozen=True)
class FacultyAssessment:
    score: float
    confidence: str          # high | medium | low
    signals: dict = field(default_factory=dict)


def _has_professor_title(title: str | None) -> bool:
    if not title:
        return False
    lowered = title.lower()
    return any(term in lowered for term in PROFESSOR_TITLE_TERMS)


def score_faculty(history: PublicationHistory) -> FacultyAssessment:
    signals: dict = {}

    # Authoritative signals win outright. They are facts; the rest are guesses.
    if history.directory_confirmed:
        signals["directory_confirmed"] = True
        return FacultyAssessment(1.0, "high", signals)
    if _has_professor_title(history.orcid_title):
        signals["orcid_title"] = history.orcid_title
        return FacultyAssessment(1.0, "high", signals)

    span = max(history.last_year - history.first_year, 0)
    span_score = min(span / SPAN_FULL_CREDIT_YEARS, 1.0)
    volume_score = min(history.works_count / VOLUME_FULL_CREDIT_WORKS, 1.0)
    signals["career_span"] = span
    signals["works_count"] = history.works_count

    if history.alphabetical_field:
        # Author order carries no seniority meaning in these fields, so using it
        # would systematically mislabel mathematicians and economists.
        signals["alphabetical_field"] = True
        score = span_score * 0.7 + volume_score * 0.3
    else:
        total = history.works_count or 1
        last_ratio = history.last_author_count / total
        first_ratio = history.first_author_count / total
        position_score = max(0.0, last_ratio - first_ratio)
        signals["last_author_ratio"] = round(last_ratio, 3)
        signals["first_author_ratio"] = round(first_ratio, 3)
        score = span_score * 0.4 + position_score * 0.4 + volume_score * 0.2

    if score >= HIGH_THRESHOLD:
        confidence = "high"
    elif score >= MEDIUM_THRESHOLD:
        confidence = "medium"
    else:
        confidence = "low"
    return FacultyAssessment(round(score, 4), confidence, signals)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_faculty.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add gradpath/match/faculty.py tests/test_faculty.py
git commit -m "feat: faculty-likelihood scoring with authoritative-signal override"
```

---

### Task 8: Embedding cache and vectorised matching

**Files:**
- Create: `gradpath/match/embed.py`
- Test: `tests/test_embed.py`, `tests/test_scale.py`

**Interfaces:**
- Consumes: `score_person`, `cosine_matrix`, `Profile`, `Settings`
- Produces: `Embedder` protocol with `.encode(texts: list[str]) -> np.ndarray`; `pack(vec) -> bytes`; `unpack(blob, dim) -> np.ndarray`; `embed_pending_works(conn, embedder, model, batch_size) -> int`; `build_profile_vector(profile, seed_abstracts, embedder) -> np.ndarray`; `run_match(conn, profile_id, profile_vector, model) -> int`; `load_embedder(model_name) -> Embedder`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embed.py
import numpy as np
import pytest

from gradpath.db import connect, migrate
from gradpath.match.embed import (
    build_profile_vector, embed_pending_works, pack, run_match, unpack,
)
from gradpath.models import Profile

MODEL = "test-model"
DIM = 4


class FakeEmbedder:
    """Deterministic stand-in: hashes text into a small vector. No torch in tests."""

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                out[row, hash(token) % DIM] += 1.0
        return out


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO institutions (id, name, added_at) VALUES ('i', 'Inst', '2026-01-01')"
    )
    return conn


def _add_work(conn, wid, title, abstract):
    cur = conn.execute(
        "INSERT INTO works (openalex_work_id, title, abstract, year) VALUES (?, ?, ?, 2024)",
        (wid, title, abstract),
    )
    return cur.lastrowid


def _add_person(conn, name, author_id):
    cur = conn.execute(
        "INSERT INTO people (institution_id, name, openalex_author_id) VALUES ('i', ?, ?)",
        (name, author_id),
    )
    return cur.lastrowid


def test_pack_unpack_roundtrips():
    vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    assert np.allclose(unpack(pack(vector), 4), vector)


def test_embeds_only_works_without_a_cached_vector(db):
    _add_work(db, "W1", "quantized inference", "we quantize models")
    db.commit()
    assert embed_pending_works(db, FakeEmbedder(), MODEL, batch_size=8) == 1
    assert embed_pending_works(db, FakeEmbedder(), MODEL, batch_size=8) == 0


def test_missing_abstract_falls_back_to_title_and_is_recorded(db):
    _add_work(db, "W1", "quantized inference", None)
    db.commit()
    embed_pending_works(db, FakeEmbedder(), MODEL, batch_size=8)
    row = db.execute("SELECT source FROM embeddings").fetchone()
    assert row["source"] == "title_only"


def test_abstract_present_is_recorded_as_abstract(db):
    _add_work(db, "W1", "t", "a real abstract")
    db.commit()
    embed_pending_works(db, FakeEmbedder(), MODEL, batch_size=8)
    assert db.execute("SELECT source FROM embeddings").fetchone()["source"] == "abstract"


def test_run_match_scores_every_person_with_no_cap(db):
    for index in range(50):
        work_id = _add_work(db, f"W{index}", f"title {index}", f"abstract {index}")
        person_id = _add_person(db, f"P{index}", f"A{index}")
        db.execute(
            "INSERT INTO authorships (person_id, work_id, position) VALUES (?, ?, 'last')",
            (person_id, work_id),
        )
    db.commit()
    embed_pending_works(db, FakeEmbedder(), MODEL, batch_size=16)
    vector = FakeEmbedder().encode(["abstract 3"])[0]
    assert run_match(db, "default", vector, MODEL) == 50
    assert db.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 50


def test_run_match_marks_sparse_people(db):
    work_id = _add_work(db, "W1", "t", "a")
    person_id = _add_person(db, "P", "A1")
    db.execute(
        "INSERT INTO authorships (person_id, work_id, position) VALUES (?, ?, 'first')",
        (person_id, work_id),
    )
    db.commit()
    embed_pending_works(db, FakeEmbedder(), MODEL, batch_size=8)
    run_match(db, "default", FakeEmbedder().encode(["a"])[0], MODEL)
    assert db.execute("SELECT sparse FROM matches").fetchone()["sparse"] == 1


def test_run_match_is_rerunnable(db):
    work_id = _add_work(db, "W1", "t", "a")
    person_id = _add_person(db, "P", "A1")
    db.execute(
        "INSERT INTO authorships (person_id, work_id, position) VALUES (?, ?, 'last')",
        (person_id, work_id),
    )
    db.commit()
    embed_pending_works(db, FakeEmbedder(), MODEL, batch_size=8)
    vector = FakeEmbedder().encode(["a"])[0]
    run_match(db, "default", vector, MODEL)
    run_match(db, "default", vector, MODEL)
    assert db.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1


def test_build_profile_vector_weights_seed_abstracts_double():
    profile = Profile(
        name="d", contact_email="e@x.com", interests="alpha",
        keywords=[], seed_papers=[], my_papers=[], fields=[], countries=[],
    )
    embedder = FakeEmbedder()
    without = build_profile_vector(profile, [], embedder)
    with_seed = build_profile_vector(profile, ["beta beta beta"], embedder)
    assert not np.allclose(without, with_seed)


def test_build_profile_vector_refuses_when_nothing_resolves():
    profile = Profile(
        name="d", contact_email="e@x.com", interests="",
        keywords=[], seed_papers=[], my_papers=[], fields=[], countries=[],
    )
    with pytest.raises(ValueError, match="no interest inputs"):
        build_profile_vector(profile, [], FakeEmbedder())
```

```python
# tests/test_scale.py
"""Guards against an accidental per-person Python loop over the corpus."""
import time

import numpy as np

from gradpath.db import connect, migrate
from gradpath.match.embed import pack, run_match

DIM = 32
TIME_BUDGET_SECONDS = 20.0


def test_ten_thousand_works_score_within_budget(tmp_path):
    conn = connect(tmp_path / "s.db")
    migrate(conn)
    conn.execute("INSERT INTO institutions (id, name, added_at) VALUES ('i','I','2026-01-01')")
    rng = np.random.default_rng(0)
    for index in range(10_000):
        work_id = conn.execute(
            "INSERT INTO works (openalex_work_id, title, abstract, year) "
            "VALUES (?, 't', 'a', 2024)", (f"W{index}",)
        ).lastrowid
        conn.execute(
            "INSERT INTO embeddings (work_id, model, dim, vector, source, computed_at) "
            "VALUES (?, 'm', ?, ?, 'abstract', '2026-01-01')",
            (work_id, DIM, pack(rng.random(DIM, dtype=np.float32))),
        )
        person_id = conn.execute(
            "INSERT INTO people (institution_id, name, openalex_author_id) "
            "VALUES ('i', ?, ?)", (f"P{index}", f"A{index}")
        ).lastrowid
        conn.execute(
            "INSERT INTO authorships (person_id, work_id, position) VALUES (?, ?, 'last')",
            (person_id, work_id),
        )
    conn.commit()
    started = time.monotonic()
    scored = run_match(conn, "default", rng.random(DIM, dtype=np.float32), "m")
    elapsed = time.monotonic() - started
    assert scored == 10_000
    assert elapsed < TIME_BUDGET_SECONDS, f"scoring took {elapsed:.1f}s"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_embed.py tests/test_scale.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gradpath.match.embed'`

- [ ] **Step 3: Write the implementation**

```python
# gradpath/match/embed.py
from __future__ import annotations

import json
import sqlite3
from typing import Protocol

import numpy as np

from gradpath.match.score import cosine_matrix, score_person
from gradpath.models import Profile
from gradpath.util import now_iso

SEED_PAPER_WEIGHT = 2.0
TEXT_WEIGHT = 1.0


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


def load_embedder(model_name: str) -> Embedder:
    """Import sentence-transformers lazily so `init` and `discover` never load torch."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def pack(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


def embed_pending_works(
    conn: sqlite3.Connection, embedder: Embedder, model: str, batch_size: int
) -> int:
    """Embed works that have no cached vector for this model. Returns how many were added.

    Abstracts are missing from roughly 40% of OpenAlex records, so the title-only
    fallback is a normal path — it is recorded in embeddings.source so output can
    flag it rather than silently ranking those works low.
    """
    rows = conn.execute(
        """
        SELECT w.id, w.title, w.abstract FROM works w
        LEFT JOIN embeddings e ON e.work_id = w.id AND e.model = ?
        WHERE e.work_id IS NULL
        """,
        (model,),
    ).fetchall()
    if not rows:
        return 0

    added = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        texts = [(row["abstract"] or row["title"] or "") for row in chunk]
        sources = ["abstract" if row["abstract"] else "title_only" for row in chunk]
        vectors = embedder.encode(texts)
        with conn:
            for row, vector, source in zip(chunk, vectors, sources):
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings "
                    "(work_id, model, dim, vector, source, computed_at) VALUES (?,?,?,?,?,?)",
                    (row["id"], model, int(len(vector)), pack(vector), source, now_iso()),
                )
                added += 1
    return added


def build_profile_vector(
    profile: Profile, seed_abstracts: list[str], embedder: Embedder
) -> np.ndarray:
    """Weighted mean of interest components. Seed abstracts count double.

    A real abstract describes a research area far more precisely than a
    self-written paragraph, which is why it carries twice the weight.
    """
    texts: list[str] = []
    weights: list[float] = []
    if profile.interests.strip():
        texts.append(profile.interests)
        weights.append(TEXT_WEIGHT)
    if profile.keywords:
        texts.append(" ".join(profile.keywords))
        weights.append(TEXT_WEIGHT)
    for abstract in seed_abstracts:
        if abstract and abstract.strip():
            texts.append(abstract)
            weights.append(SEED_PAPER_WEIGHT)
    if not texts:
        raise ValueError(
            "no interest inputs resolved — fill in interests, keywords or seed_papers "
            "in profile.yaml before running match"
        )
    matrix = np.asarray(embedder.encode(texts), dtype=np.float32)
    weight_vector = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    return (matrix * weight_vector).sum(axis=0) / weight_vector.sum()


def run_match(
    conn: sqlite3.Connection, profile_id: str, profile_vector: np.ndarray, model: str
) -> int:
    """Score every person in the database. No cap, by design. Returns people scored."""
    rows = conn.execute(
        "SELECT work_id, dim, vector FROM embeddings WHERE model = ? ORDER BY work_id",
        (model,),
    ).fetchall()
    if not rows:
        return 0

    dim = rows[0]["dim"]
    work_ids = np.array([row["work_id"] for row in rows], dtype=np.int64)
    matrix = np.vstack([unpack(row["vector"], dim) for row in rows])

    # One matrix multiply over the whole corpus — never a per-person loop.
    similarities = cosine_matrix(matrix, np.asarray(profile_vector, dtype=np.float32))
    by_work = dict(zip(work_ids.tolist(), similarities.tolist()))

    grouped: dict[int, dict[int, float]] = {}
    for row in conn.execute("SELECT person_id, work_id FROM authorships"):
        similarity = by_work.get(row["work_id"])
        if similarity is not None:
            grouped.setdefault(row["person_id"], {})[row["work_id"]] = similarity

    stamp = now_iso()
    with conn:
        for person_id, similarity_map in grouped.items():
            result = score_person(similarity_map)
            conn.execute(
                """
                INSERT INTO matches
                    (profile_id, person_id, stage1_score, top_work_ids, sparse, computed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id, person_id) DO UPDATE SET
                    stage1_score = excluded.stage1_score,
                    top_work_ids = excluded.top_work_ids,
                    sparse = excluded.sparse,
                    computed_at = excluded.computed_at
                """,
                (profile_id, person_id, result.score, json.dumps(result.top_work_ids),
                 int(result.sparse), stamp),
            )
    return len(grouped)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_embed.py tests/test_scale.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add gradpath/match/embed.py tests/test_embed.py tests/test_scale.py
git commit -m "feat: persisted embedding cache and uncapped vectorised matching"
```

---

### Task 9: Crossref and ORCID email sources

**Files:**
- Create: `gradpath/sources/crossref.py`, `gradpath/sources/orcid.py`
- Test: `tests/test_email_sources.py`

**Interfaces:**
- Consumes: `PoliteClient`
- Produces: `crossref_email(client, doi, author_name) -> str | None`; `orcid_record(client, orcid) -> OrcidRecord | None`; `OrcidRecord(email: str | None, title: str | None)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_email_sources.py
import httpx
import pytest
import respx

from gradpath.models import Settings
from gradpath.net.http import PoliteClient
from gradpath.sources.crossref import crossref_email
from gradpath.sources.orcid import orcid_record

SETTINGS = Settings("m", 8, 1000.0, 2021, 0.6, "claude-opus-5", 1.0, ".cache")


@pytest.fixture
def client(tmp_path):
    return PoliteClient(SETTINGS, "me@example.com", tmp_path / "c")


@respx.mock
def test_crossref_returns_matching_author_email(client):
    respx.get("https://api.crossref.org/works/10.1145/1234").mock(
        return_value=httpx.Response(200, json={"message": {"author": [
            {"given": "Sunmi", "family": "Park", "email": "sunmi@kaist.ac.kr"},
            {"given": "Other", "family": "Person"},
        ]}})
    )
    assert crossref_email(client, "10.1145/1234", "Sunmi Park") == "sunmi@kaist.ac.kr"


@respx.mock
def test_crossref_returns_none_when_name_does_not_match(client):
    respx.get("https://api.crossref.org/works/10.1145/1234").mock(
        return_value=httpx.Response(200, json={"message": {"author": [
            {"given": "Someone", "family": "Else", "email": "else@x.ac.kr"},
        ]}})
    )
    assert crossref_email(client, "10.1145/1234", "Sunmi Park") is None


@respx.mock
def test_crossref_returns_none_on_missing_record(client):
    respx.get("https://api.crossref.org/works/10.1145/9999").mock(
        return_value=httpx.Response(404, json={})
    )
    assert crossref_email(client, "10.1145/9999", "Anyone") is None


@respx.mock
def test_orcid_extracts_email_and_employment_title(client):
    respx.get("https://pub.orcid.org/v3.0/0000-0001-0000-0001/record").mock(
        return_value=httpx.Response(200, json={
            "person": {"emails": {"email": [{"email": "p@kaist.ac.kr"}]}},
            "activities-summary": {"employments": {"affiliation-group": [
                {"summaries": [{"employment-summary": {"role-title": "Associate Professor"}}]}
            ]}},
        })
    )
    record = orcid_record(client, "0000-0001-0000-0001")
    assert record.email == "p@kaist.ac.kr"
    assert record.title == "Associate Professor"


@respx.mock
def test_orcid_handles_record_with_no_public_email(client):
    respx.get("https://pub.orcid.org/v3.0/0000-0002/record").mock(
        return_value=httpx.Response(200, json={
            "person": {"emails": {"email": []}},
            "activities-summary": {"employments": {"affiliation-group": []}},
        })
    )
    record = orcid_record(client, "0000-0002")
    assert record.email is None
    assert record.title is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_email_sources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gradpath.sources.crossref'`

- [ ] **Step 3: Write the implementation**

```python
# gradpath/sources/crossref.py
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
```

```python
# gradpath/sources/orcid.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_email_sources.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add gradpath/sources/crossref.py gradpath/sources/orcid.py tests/test_email_sources.py
git commit -m "feat: Crossref and ORCID email sources"
```

---

### Task 10: Adapter framework, generic crawler, and the resolution chain

**Files:**
- Create: `gradpath/sources/adapters/__init__.py`, `gradpath/sources/adapters/base.py`
- Create: `gradpath/sources/crawler.py`, `gradpath/sources/emails.py`
- Test: `tests/test_crawler.py`, `tests/test_email_chain.py`

**Interfaces:**
- Consumes: `PoliteClient`, `crossref_email`, `orcid_record`, `FacultyRecord`
- Produces: `InstitutionAdapter` ABC; `register(adapter_cls)`; `get_adapter(slug) -> type[InstitutionAdapter] | None`; `all_adapters() -> list[type[InstitutionAdapter]]`; `extract_emails(html: str) -> list[str]`; `crawl_faculty_email(client, homepage, name) -> str | None`; `resolve_email(conn, client, person_row) -> EmailResolution`; `EmailResolution(email, confidence, source)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_crawler.py
from gradpath.sources.crawler import extract_emails


def test_extracts_plain_mailto():
    html = '<a href="mailto:kim@kaist.ac.kr">Prof Kim</a>'
    assert extract_emails(html) == ["kim@kaist.ac.kr"]


def test_extracts_bare_text_address():
    assert extract_emails("<p>contact: park@gist.ac.kr</p>") == ["park@gist.ac.kr"]


def test_decodes_common_at_obfuscation():
    html = "<p>lee [at] snu.ac.kr</p>"
    assert extract_emails(html) == ["lee@snu.ac.kr"]


def test_decodes_dot_obfuscation():
    html = "<p>choi [at] kaist [dot] ac [dot] kr</p>"
    assert extract_emails(html) == ["choi@kaist.ac.kr"]


def test_deduplicates_and_preserves_order():
    html = "a@x.ac.kr b@x.ac.kr a@x.ac.kr"
    assert extract_emails(html) == ["a@x.ac.kr", "b@x.ac.kr"]


def test_ignores_image_and_asset_filenames():
    assert extract_emails('<img src="logo@2x.png">') == []


def test_returns_empty_for_no_addresses():
    assert extract_emails("<p>no contact here</p>") == []
```

```python
# tests/test_email_chain.py
import httpx
import pytest
import respx

from gradpath.db import connect, migrate
from gradpath.models import Settings
from gradpath.net.http import PoliteClient
from gradpath.sources.emails import resolve_email

SETTINGS = Settings("m", 8, 1000.0, 2021, 0.6, "claude-opus-5", 1.0, ".cache")


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    conn.execute("INSERT INTO institutions (id, name, added_at) VALUES ('i','I','2026-01-01')")
    conn.execute(
        "INSERT INTO people (id, institution_id, name, openalex_author_id, orcid, homepage) "
        "VALUES (1, 'i', 'Sunmi Park', 'A1', '0000-0001', 'https://lab.example.com/park')"
    )
    conn.execute("INSERT INTO works (id, openalex_work_id, title, doi) "
                 "VALUES (1, 'W1', 't', '10.1145/1234')")
    conn.execute("INSERT INTO authorships (person_id, work_id, position) VALUES (1, 1, 'last')")
    conn.commit()
    return conn


@pytest.fixture
def client(tmp_path):
    return PoliteClient(SETTINGS, "me@example.com", tmp_path / "c")


@respx.mock
def test_crossref_wins_and_is_marked_high(db, client):
    respx.get("https://api.crossref.org/works/10.1145/1234").mock(
        return_value=httpx.Response(200, json={"message": {"author": [
            {"family": "Park", "email": "park@kaist.ac.kr"}]}})
    )
    result = resolve_email(db, client, db.execute("SELECT * FROM people").fetchone())
    assert result.email == "park@kaist.ac.kr"
    assert result.confidence == "high"
    assert result.source == "crossref"


@respx.mock
def test_falls_through_to_orcid(db, client):
    respx.get("https://api.crossref.org/works/10.1145/1234").mock(
        return_value=httpx.Response(200, json={"message": {"author": []}})
    )
    respx.get("https://pub.orcid.org/v3.0/0000-0001/record").mock(
        return_value=httpx.Response(200, json={
            "person": {"emails": {"email": [{"email": "park@orcid.example"}]}},
            "activities-summary": {"employments": {"affiliation-group": []}},
        })
    )
    result = resolve_email(db, client, db.execute("SELECT * FROM people").fetchone())
    assert result.source == "orcid"
    assert result.confidence == "high"


@respx.mock
def test_falls_through_to_crawler_and_is_marked_medium(db, client):
    respx.get("https://api.crossref.org/works/10.1145/1234").mock(
        return_value=httpx.Response(200, json={"message": {"author": []}})
    )
    respx.get("https://pub.orcid.org/v3.0/0000-0001/record").mock(
        return_value=httpx.Response(200, json={
            "person": {"emails": {"email": []}},
            "activities-summary": {"employments": {"affiliation-group": []}},
        })
    )
    respx.get("https://lab.example.com/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nAllow: /\n")
    )
    respx.get("https://lab.example.com/park").mock(
        return_value=httpx.Response(200, text='<a href="mailto:park@lab.example.com">mail</a>')
    )
    result = resolve_email(db, client, db.execute("SELECT * FROM people").fetchone())
    assert result.email == "park@lab.example.com"
    assert result.confidence == "medium"
    assert result.source == "crawler"


@respx.mock
def test_never_guesses_a_pattern_when_all_sources_fail(db, client):
    respx.get("https://api.crossref.org/works/10.1145/1234").mock(
        return_value=httpx.Response(200, json={"message": {"author": []}})
    )
    respx.get("https://pub.orcid.org/v3.0/0000-0001/record").mock(
        return_value=httpx.Response(200, json={
            "person": {"emails": {"email": []}},
            "activities-summary": {"employments": {"affiliation-group": []}},
        })
    )
    respx.get("https://lab.example.com/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nAllow: /\n")
    )
    respx.get("https://lab.example.com/park").mock(
        return_value=httpx.Response(200, text="<p>no address</p>")
    )
    result = resolve_email(db, client, db.execute("SELECT * FROM people").fetchone())
    assert result.email is None
    assert result.confidence == "none"


def test_registered_adapters_satisfy_the_contract():
    """Every adapter must expose the same interface, so contributors cannot break the chain."""
    from gradpath.sources.adapters import all_adapters
    from gradpath.sources.adapters.base import InstitutionAdapter

    adapters = all_adapters()
    assert adapters, "no adapters registered"
    for adapter_cls in adapters:
        assert issubclass(adapter_cls, InstitutionAdapter)
        assert isinstance(adapter_cls.slug, str) and adapter_cls.slug
        assert isinstance(adapter_cls.domains, list) and adapter_cls.domains
        assert callable(adapter_cls.faculty_urls)
        assert callable(adapter_cls.parse_faculty)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_crawler.py tests/test_email_chain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gradpath.sources.crawler'`

- [ ] **Step 3: Write the implementation**

```python
# gradpath/sources/adapters/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

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
    domains: list[str] = []

    @abstractmethod
    def faculty_urls(self) -> Iterable[str]:
        """Directory pages to fetch."""

    @abstractmethod
    def parse_faculty(self, html: str, url: str) -> list[FacultyRecord]:
        """Extract faculty rows from one directory page."""
```

```python
# gradpath/sources/adapters/__init__.py
from __future__ import annotations

from gradpath.sources.adapters.base import InstitutionAdapter

_REGISTRY: dict[str, type[InstitutionAdapter]] = {}


def register(adapter_cls: type[InstitutionAdapter]) -> type[InstitutionAdapter]:
    _REGISTRY[adapter_cls.slug] = adapter_cls
    return adapter_cls


def get_adapter(slug: str) -> type[InstitutionAdapter] | None:
    return _REGISTRY.get(slug)


def all_adapters() -> list[type[InstitutionAdapter]]:
    return list(_REGISTRY.values())


# Importing the modules is what registers them.
from gradpath.sources.adapters import gist, kaist, snu  # noqa: E402,F401
```

```python
# gradpath/sources/crawler.py
from __future__ import annotations

import re

from gradpath.net.http import HostBlocked, PoliteClient

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ASSET_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".css", ".js")

_OBFUSCATIONS = (
    (re.compile(r"\s*\[\s*at\s*\]\s*", re.I), "@"),
    (re.compile(r"\s*\(\s*at\s*\)\s*", re.I), "@"),
    (re.compile(r"\s*\[\s*dot\s*\]\s*", re.I), "."),
    (re.compile(r"\s*\(\s*dot\s*\)\s*", re.I), "."),
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
```

```python
# gradpath/sources/emails.py
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from gradpath.net.http import PoliteClient
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
        html = client.get_text(url)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_crawler.py tests/test_email_chain.py -v`
Expected: 12 passed (5 in the chain, 7 in the crawler; the adapter contract test passes once Task 11 lands — if running Task 10 alone, it fails on "no adapters registered", which Task 11 resolves)

- [ ] **Step 5: Commit**

```bash
git add gradpath/sources/adapters/ gradpath/sources/crawler.py gradpath/sources/emails.py tests/
git commit -m "feat: adapter framework, faculty crawler, and email resolution chain"
```

---

### Task 11: KAIST, GIST and SNU adapters

**Files:**
- Create: `gradpath/sources/adapters/kaist.py`, `gist.py`, `snu.py`
- Create: `tests/fixtures/kaist_faculty.html`, `gist_faculty.html`, `snu_faculty.html`
- Test: `tests/test_adapters.py`

**Interfaces:**
- Consumes: `InstitutionAdapter`, `register`, `FacultyRecord`, `extract_emails`
- Produces: `KaistAdapter`, `GistAdapter`, `SnuAdapter`, each registered under its slug

- [ ] **Step 1: Write the failing test**

Create `tests/fixtures/kaist_faculty.html` with two faculty entries in the shape below, and equivalent fixtures for GIST and SNU with their own class names (`.prof-item` / `.faculty-card` respectively):

```html
<!-- tests/fixtures/kaist_faculty.html -->
<div class="faculty-list">
  <div class="faculty-item">
    <a class="name" href="/people/park">Sunmi Park</a>
    <span class="position">Associate Professor</span>
    <span class="dept">School of Computing</span>
    <a href="mailto:park@kaist.ac.kr">email</a>
  </div>
  <div class="faculty-item">
    <a class="name" href="/people/kim">Jaewon Kim</a>
    <span class="position">Professor</span>
    <span class="dept">School of Computing</span>
    <span class="email">kim [at] kaist [dot] ac [dot] kr</span>
  </div>
</div>
```

```python
# tests/test_adapters.py
import pathlib

import pytest

from gradpath.sources.adapters import all_adapters, get_adapter
from gradpath.sources.adapters.base import InstitutionAdapter

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("slug", ["kaist", "gist", "snu"])
def test_adapter_is_registered(slug):
    assert get_adapter(slug) is not None


@pytest.mark.parametrize("adapter_cls", all_adapters(), ids=lambda c: c.slug)
def test_adapter_satisfies_contract(adapter_cls):
    """One contract suite over every adapter — this is what makes community PRs safe."""
    assert issubclass(adapter_cls, InstitutionAdapter)
    adapter = adapter_cls()
    assert adapter_cls.slug
    assert adapter_cls.domains
    urls = list(adapter.faculty_urls())
    assert urls and all(u.startswith("https://") for u in urls)
    assert adapter.parse_faculty("<html></html>", urls[0]) == []


def test_kaist_adapter_parses_names_titles_and_emails():
    adapter = get_adapter("kaist")()
    html = (FIXTURES / "kaist_faculty.html").read_text()
    records = adapter.parse_faculty(html, "https://cs.kaist.ac.kr/people")
    assert [r.name for r in records] == ["Sunmi Park", "Jaewon Kim"]
    assert records[0].email == "park@kaist.ac.kr"
    assert records[0].title == "Associate Professor"


def test_kaist_adapter_decodes_obfuscated_address():
    adapter = get_adapter("kaist")()
    html = (FIXTURES / "kaist_faculty.html").read_text()
    records = adapter.parse_faculty(html, "https://cs.kaist.ac.kr/people")
    assert records[1].email == "kim@kaist.ac.kr"


def test_kaist_adapter_resolves_relative_homepage_to_absolute():
    adapter = get_adapter("kaist")()
    html = (FIXTURES / "kaist_faculty.html").read_text()
    records = adapter.parse_faculty(html, "https://cs.kaist.ac.kr/people")
    assert records[0].homepage == "https://cs.kaist.ac.kr/people/park"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapters.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gradpath.sources.adapters.kaist'`

- [ ] **Step 3: Write the implementation**

```python
# gradpath/sources/adapters/kaist.py
from __future__ import annotations

from typing import Iterable
from urllib.parse import urljoin

from gradpath.models import FacultyRecord
from gradpath.sources.adapters import register
from gradpath.sources.adapters.base import InstitutionAdapter
from gradpath.sources.crawler import extract_emails

DIRECTORY_URLS = (
    "https://cs.kaist.ac.kr/people/faculty",
    "https://ee.kaist.ac.kr/people/faculty",
)


@register
class KaistAdapter(InstitutionAdapter):
    slug = "kaist"
    domains = ["kaist.ac.kr"]
    item_selector = ".faculty-item"

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
```

Write `gist.py` and `snu.py` with the same structure, changing only `slug`, `domains`, `DIRECTORY_URLS` and `item_selector` (`.prof-item` for GIST, `.faculty-card` for SNU). Repeating the body rather than sharing a base class is deliberate: adapters diverge quickly as sites change, and a shared superclass becomes a constraint every contributor has to fight.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_adapters.py tests/test_email_chain.py -v`
Expected: all passed, including the previously-failing "no adapters registered" contract test

- [ ] **Step 5: Commit**

```bash
git add gradpath/sources/adapters/ tests/test_adapters.py tests/fixtures/
git commit -m "feat: KAIST, GIST and SNU faculty adapters"
```

---

### Task 12: Budget-gated LLM rerank

**Files:**
- Create: `gradpath/match/rerank.py`
- Test: `tests/test_rerank.py`

**Interfaces:**
- Consumes: `matches` and `works` tables
- Produces: `RerankPlan(person_count, estimated_tokens, estimated_cost_usd)`; `plan_rerank(conn, profile_id, min_score, model) -> RerankPlan`; `rerank(conn, profile_id, min_score, model, budget_usd, ask) -> int`; `MissingApiKey`; `COST_PER_MTOK_USD`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rerank.py
import pytest

from gradpath.db import connect, migrate
from gradpath.match.rerank import MissingApiKey, plan_rerank, rerank


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    conn.execute("INSERT INTO institutions (id, name, added_at) VALUES ('i','I','2026-01-01')")
    for index in range(10):
        work_id = conn.execute(
            "INSERT INTO works (openalex_work_id, title, abstract, year) "
            "VALUES (?, 'T', 'an abstract', 2024)", (f"W{index}",)
        ).lastrowid
        person_id = conn.execute(
            "INSERT INTO people (institution_id, name, openalex_author_id) "
            "VALUES ('i', ?, ?)", (f"P{index}", f"A{index}")
        ).lastrowid
        conn.execute(
            "INSERT INTO authorships (person_id, work_id, position) VALUES (?,?, 'last')",
            (person_id, work_id),
        )
        conn.execute(
            "INSERT INTO matches (profile_id, person_id, stage1_score, top_work_ids, "
            "computed_at) VALUES ('default', ?, ?, ?, '2026-01-01')",
            (person_id, index / 10.0, f"[{work_id}]"),
        )
    conn.commit()
    return conn


def test_plan_counts_only_people_above_the_threshold(db):
    plan = plan_rerank(db, "default", min_score=0.7, model="claude-opus-5")
    assert plan.person_count == 3          # 0.7, 0.8, 0.9
    assert plan.estimated_cost_usd > 0


def test_plan_with_low_threshold_includes_everyone(db):
    assert plan_rerank(db, "default", 0.0, "claude-opus-5").person_count == 10


def test_rerank_refuses_without_an_api_key(db, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingApiKey, match="stage 1 results are unaffected"):
        rerank(db, "default", 0.7, "claude-opus-5", budget_usd=5.0, ask=lambda _: True)


def test_rerank_aborts_when_the_user_declines(db, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert rerank(db, "default", 0.7, "claude-opus-5", 5.0, ask=lambda _: False) == 0
    assert db.execute("SELECT COUNT(*) FROM matches WHERE stage2_score IS NOT NULL")\
             .fetchone()[0] == 0


def test_rerank_stops_at_the_budget_ceiling(db, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls: list[int] = []

    def fake_score(person, works, profile_text, model):
        calls.append(person["id"])
        return 0.5, "because"

    scored = rerank(db, "default", 0.0, "claude-opus-5", budget_usd=0.0,
                    ask=lambda _: True, score_fn=fake_score)
    assert scored == 0
    assert calls == []


def test_rerank_writes_score_and_reason(db, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_score(person, works, profile_text, model):
        return 0.91, "three recent papers directly on this topic"

    scored = rerank(db, "default", 0.7, "claude-opus-5", budget_usd=10.0,
                    ask=lambda _: True, score_fn=fake_score)
    assert scored == 3
    row = db.execute(
        "SELECT stage2_score, reason FROM matches WHERE stage2_score IS NOT NULL LIMIT 1"
    ).fetchone()
    assert row["stage2_score"] == pytest.approx(0.91)
    assert "three recent papers" in row["reason"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rerank.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gradpath.match.rerank'`

- [ ] **Step 3: Write the implementation**

```python
# gradpath/match/rerank.py
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Callable

from gradpath.util import now_iso

# Rough public list price; used only to warn before spending, never to bill.
COST_PER_MTOK_USD = 15.0
TOKENS_PER_PERSON_ESTIMATE = 1_800
TOP_WORKS_PER_PERSON = 3


class MissingApiKey(RuntimeError):
    pass


@dataclass(frozen=True)
class RerankPlan:
    person_count: int
    estimated_tokens: int
    estimated_cost_usd: float


def plan_rerank(
    conn: sqlite3.Connection, profile_id: str, min_score: float, model: str
) -> RerankPlan:
    count = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE profile_id = ? AND stage1_score >= ?",
        (profile_id, min_score),
    ).fetchone()[0]
    tokens = count * TOKENS_PER_PERSON_ESTIMATE
    return RerankPlan(count, tokens, round(tokens / 1_000_000 * COST_PER_MTOK_USD, 4))


def _call_claude(person: sqlite3.Row, works: list[sqlite3.Row], profile_text: str,
                 model: str) -> tuple[float, str]:
    """Ask Claude to score fit. Imported lazily so the package works without the SDK."""
    from anthropic import Anthropic

    evidence = "\n\n".join(
        f"- {w['title']}\n  {(w['abstract'] or '')[:800]}" for w in works
    )
    prompt = (
        "You are helping a prospective graduate student decide who to email.\n\n"
        f"APPLICANT INTERESTS:\n{profile_text}\n\n"
        f"RESEARCHER: {person['name']}\n"
        f"THEIR MOST RELEVANT WORK:\n{evidence}\n\n"
        "Return JSON only: {\"score\": <0.0-1.0>, \"reason\": \"<one sentence>\"}"
    )
    response = Anthropic().messages.create(
        model=model, max_tokens=300, messages=[{"role": "user", "content": prompt}]
    )
    payload = json.loads(response.content[0].text)
    return float(payload["score"]), str(payload["reason"])


def rerank(
    conn: sqlite3.Connection,
    profile_id: str,
    min_score: float,
    model: str,
    budget_usd: float,
    ask: Callable[[RerankPlan], bool],
    profile_text: str = "",
    score_fn: Callable[..., tuple[float, str]] = _call_claude,
) -> int:
    """Rerank everyone above `min_score`, after confirmation and within budget."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise MissingApiKey(
            "ANTHROPIC_API_KEY is not set, so --rerank cannot run. "
            "Your stage 1 results are unaffected and remain fully usable."
        )

    plan = plan_rerank(conn, profile_id, min_score, model)
    if plan.person_count == 0 or not ask(plan):
        return 0

    per_person_cost = plan.estimated_cost_usd / plan.person_count
    affordable = int(budget_usd // per_person_cost) if per_person_cost > 0 else plan.person_count
    if affordable <= 0:
        return 0

    rows = conn.execute(
        """
        SELECT m.person_id, m.top_work_ids, p.name, p.id AS id
        FROM matches m JOIN people p ON p.id = m.person_id
        WHERE m.profile_id = ? AND m.stage1_score >= ?
        ORDER BY m.stage1_score DESC LIMIT ?
        """,
        (profile_id, min_score, affordable),
    ).fetchall()

    scored = 0
    for row in rows:
        work_ids = json.loads(row["top_work_ids"] or "[]")[:TOP_WORKS_PER_PERSON]
        if not work_ids:
            continue
        placeholders = ",".join("?" for _ in work_ids)
        works = conn.execute(
            f"SELECT title, abstract FROM works WHERE id IN ({placeholders})", work_ids
        ).fetchall()
        score, reason = score_fn(row, works, profile_text, model)
        with conn:
            conn.execute(
                "UPDATE matches SET stage2_score = ?, reason = ?, computed_at = ? "
                "WHERE profile_id = ? AND person_id = ?",
                (score, reason, now_iso(), profile_id, row["person_id"]),
            )
        scored += 1
    return scored
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rerank.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add gradpath/match/rerank.py tests/test_rerank.py
git commit -m "feat: budget-gated LLM rerank over a score band"
```

---

### Task 13: Reporting and export

**Files:**
- Create: `gradpath/report/__init__.py`, `gradpath/report/table.py`, `gradpath/report/export.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `matches`, `people`, `institutions` tables
- Produces: `ResultRow` dataclass; `query_results(conn, profile_id, min_score=0.0, faculty_only=False, country=None, limit=None) -> list[ResultRow]`; `render_table(rows) -> rich.table.Table`; `to_csv(rows, path)`; `to_markdown(rows, path)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
import csv

import pytest

from gradpath.db import connect, migrate
from gradpath.report.export import to_csv, to_markdown
from gradpath.report.table import query_results, render_table


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    conn.executescript(
        """
        INSERT INTO institutions (id, name, country, added_at)
        VALUES ('kaist','KAIST','KR','2026-01-01'), ('tum','TUM','DE','2026-01-01');

        INSERT INTO people (id, institution_id, name, openalex_author_id, email,
                            email_confidence, faculty_score, faculty_confidence,
                            first_year, last_year, works_count)
        VALUES
          (1,'kaist','Prof High','A1','a@kaist.ac.kr','high',0.85,'high',2005,2026,120),
          (2,'kaist','Maybe Student','A2',NULL,'none',0.16,'low',2022,2026,6),
          (3,'tum','Prof DE','A3','c@tum.de','high',0.72,'high',2010,2026,60);

        INSERT INTO matches (profile_id, person_id, stage1_score, stage2_score, reason,
                             top_work_ids, sparse, computed_at)
        VALUES
          ('default',1,0.81,NULL,NULL,'[1]',0,'2026-01-01'),
          ('default',2,0.90,NULL,NULL,'[2]',1,'2026-01-01'),
          ('default',3,0.60,0.95,'great fit','[3]',0,'2026-01-01');
        """
    )
    conn.commit()
    return conn


def test_results_sort_by_stage2_when_present(db):
    rows = query_results(db, "default")
    assert rows[0].name == "Prof DE"          # stage2 0.95 outranks stage1 0.90


def test_min_score_filters_without_capping(db):
    assert len(query_results(db, "default", min_score=0.85)) == 2


def test_faculty_only_is_opt_in_and_excludes_low_confidence(db):
    names = [r.name for r in query_results(db, "default", faculty_only=True)]
    assert "Maybe Student" not in names
    assert "Prof High" in names


def test_low_confidence_person_is_present_by_default(db):
    """Nobody is ever dropped unless the user opts in."""
    assert "Maybe Student" in [r.name for r in query_results(db, "default")]


def test_country_filter(db):
    assert [r.name for r in query_results(db, "default", country="DE")] == ["Prof DE"]


def test_limit_is_display_only_and_does_not_change_ordering(db):
    limited = query_results(db, "default", limit=1)
    assert len(limited) == 1
    assert limited[0].name == query_results(db, "default")[0].name


def test_render_table_includes_both_scores_separately(db):
    table = render_table(query_results(db, "default"))
    headers = [c.header for c in table.columns]
    assert "fit" in headers
    assert "faculty" in headers


def test_csv_export_roundtrips(db, tmp_path):
    path = tmp_path / "out.csv"
    to_csv(query_results(db, "default"), path)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 3
    assert rows[0]["name"] == "Prof DE"
    assert "faculty_confidence" in rows[0]


def test_markdown_export_contains_a_table(db, tmp_path):
    path = tmp_path / "out.md"
    to_markdown(query_results(db, "default"), path)
    text = path.read_text()
    assert "| name |" in text
    assert "Prof DE" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gradpath.report'`

- [ ] **Step 3: Write the implementation**

```python
# gradpath/report/table.py
from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass

from rich.table import Table

FIELDNAMES = [
    "rank", "name", "institution", "country", "fit", "faculty_score",
    "faculty_confidence", "email", "email_confidence", "sparse", "reason",
]


@dataclass(frozen=True)
class ResultRow:
    rank: int
    name: str
    institution: str | None
    country: str | None
    fit: float
    faculty_score: float | None
    faculty_confidence: str | None
    email: str | None
    email_confidence: str | None
    sparse: bool
    reason: str | None

    def as_dict(self) -> dict:
        return asdict(self)


def query_results(
    conn: sqlite3.Connection,
    profile_id: str,
    min_score: float = 0.0,
    faculty_only: bool = False,
    country: str | None = None,
    limit: int | None = None,
) -> list[ResultRow]:
    """Read the ranking. `limit` is display-only and never affects what is stored."""
    clauses = ["m.profile_id = ?", "COALESCE(m.stage2_score, m.stage1_score) >= ?"]
    params: list = [profile_id, min_score]
    if faculty_only:
        clauses.append("p.faculty_confidence IN ('high','medium')")
    if country:
        clauses.append("i.country = ?")
        params.append(country.upper())

    sql = f"""
        SELECT p.name, i.name AS institution, i.country,
               COALESCE(m.stage2_score, m.stage1_score) AS fit,
               p.faculty_score, p.faculty_confidence,
               p.email, p.email_confidence, m.sparse, m.reason
        FROM matches m
        JOIN people p ON p.id = m.person_id
        LEFT JOIN institutions i ON i.id = p.institution_id
        WHERE {' AND '.join(clauses)}
        ORDER BY fit DESC, p.faculty_score DESC
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    return [
        ResultRow(
            rank=index,
            name=row["name"],
            institution=row["institution"],
            country=row["country"],
            fit=round(row["fit"], 4),
            faculty_score=row["faculty_score"],
            faculty_confidence=row["faculty_confidence"],
            email=row["email"],
            email_confidence=row["email_confidence"],
            sparse=bool(row["sparse"]),
            reason=row["reason"],
        )
        for index, row in enumerate(conn.execute(sql, params).fetchall(), start=1)
    ]


def render_table(rows: list[ResultRow]) -> Table:
    """Fit and faculty stay in separate columns — they are never blended."""
    table = Table(title="gradpath results")
    for header in ("#", "name", "institution", "cc", "fit", "faculty", "email"):
        table.add_column(header)
    for row in rows:
        marker = " *sparse" if row.sparse else ""
        table.add_row(
            str(row.rank),
            row.name + marker,
            row.institution or "-",
            row.country or "-",
            f"{row.fit:.3f}",
            f"{row.faculty_confidence or '-'}",
            row.email or "(not found)",
        )
    return table
```

```python
# gradpath/report/export.py
from __future__ import annotations

import csv
from pathlib import Path

from gradpath.report.table import FIELDNAMES, ResultRow


def to_csv(rows: list[ResultRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def to_markdown(rows: list[ResultRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(FIELDNAMES) + " |"
    divider = "|" + "|".join(["---"] * len(FIELDNAMES)) + "|"
    lines = [header, divider]
    for row in rows:
        values = row.as_dict()
        lines.append("| " + " | ".join(str(values[f] if values[f] is not None else "")
                                       for f in FIELDNAMES) + " |")
    path.write_text("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_report.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add gradpath/report/ tests/test_report.py
git commit -m "feat: ranked table rendering and CSV/Markdown export"
```

---

### Task 14: CLI wiring and end-to-end smoke test

**Files:**
- Create: `gradpath/cli.py`, `gradpath.sh`
- Modify: `README.md` (usage section)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: every module above
- Produces: `app: typer.Typer` with sub-apps `fields`, `institutions`, `emails`; top-level commands `init`, `discover`, `faculty`, `match`, `show`, `export`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import httpx
import respx
from typer.testing import CliRunner

from gradpath.cli import app

runner = CliRunner()

SETTINGS_YAML = """
embedding_model: test-model
embedding_batch_size: 8
rate_limit_per_host: 1000.0
default_since_year: 2021
show_min_score: 0.0
rerank_model: claude-opus-5
rerank_default_budget_usd: 1.0
cache_dir: .cache
"""

PROFILE_YAML = """
name: default
contact_email: me@example.com
fields: [efficient-ml]
countries: [KR]
interests: quantized inference on mobile devices
keywords: [quantization]
seed_papers: []
my_papers: []
"""

FIELDS_YAML = """
efficient-ml:
  label: Efficient ML
  topics: [T10028]
  aliases: [edge-ml]
"""


def _workspace(tmp_path):
    (tmp_path / "settings.yaml").write_text(SETTINGS_YAML)
    (tmp_path / "profile.yaml").write_text(PROFILE_YAML)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "fields.yaml").write_text(FIELDS_YAML)
    (tmp_path / "data" / "institutions.yaml").write_text("{}\n")
    return tmp_path


def test_init_creates_database_and_profile(tmp_path):
    result = runner.invoke(app, ["--root", str(tmp_path), "init"])
    assert result.exit_code == 0
    assert (tmp_path / "gradpath.db").exists()
    assert (tmp_path / "profile.yaml").exists()


def test_init_does_not_overwrite_an_existing_profile(tmp_path):
    workspace = _workspace(tmp_path)
    runner.invoke(app, ["--root", str(workspace), "init"])
    assert "contact_email: me@example.com" in (workspace / "profile.yaml").read_text()


@respx.mock
def test_discover_then_show_runs_end_to_end(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    monkeypatch.setattr(
        "gradpath.cli.load_embedder",
        lambda name: __import__("tests.test_embed", fromlist=["FakeEmbedder"]).FakeEmbedder(),
    )
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": [{
            "id": "https://openalex.org/W1", "title": "Quantized inference",
            "publication_year": 2024, "doi": None, "cited_by_count": 3,
            "primary_location": {"source": {"display_name": "V"}},
            "topics": [{"id": "https://openalex.org/T10028"}],
            "abstract_inverted_index": {"quantized": [0], "inference": [1]},
            "authorships": [{
                "author_position": "last",
                "author": {"id": "https://openalex.org/A1",
                           "display_name": "Sunmi Park", "orcid": None},
                "institutions": [{"id": "https://openalex.org/I1",
                                  "display_name": "KAIST", "country_code": "KR"}],
            }],
        }], "meta": {"next_cursor": None}})
    )
    runner.invoke(app, ["--root", str(workspace), "init"])
    assert runner.invoke(
        app, ["--root", str(workspace), "discover", "--field", "efficient-ml"]
    ).exit_code == 0
    assert runner.invoke(app, ["--root", str(workspace), "faculty", "score"]).exit_code == 0
    assert runner.invoke(app, ["--root", str(workspace), "match"]).exit_code == 0
    show = runner.invoke(app, ["--root", str(workspace), "show"])
    assert show.exit_code == 0
    assert "Sunmi Park" in show.stdout


def test_show_limit_is_display_only(tmp_path):
    workspace = _workspace(tmp_path)
    runner.invoke(app, ["--root", str(workspace), "init"])
    result = runner.invoke(app, ["--root", str(workspace), "show", "--limit", "5"])
    assert result.exit_code == 0


def test_unknown_field_gives_an_actionable_error(tmp_path):
    workspace = _workspace(tmp_path)
    runner.invoke(app, ["--root", str(workspace), "init"])
    result = runner.invoke(
        app, ["--root", str(workspace), "discover", "--field", "nonsense"]
    )
    assert result.exit_code != 0
    assert "fields search" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gradpath.cli'`

- [ ] **Step 3: Write the implementation**

```python
# gradpath/cli.py
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from gradpath.config import (
    load_fields, load_profile, load_settings, resolve_field, scaffold_profile,
)
from gradpath.db import connect, migrate
from gradpath.match.embed import (
    build_profile_vector, embed_pending_works, load_embedder, run_match,
)
from gradpath.match.faculty import PublicationHistory, score_faculty
from gradpath.match.rerank import MissingApiKey, plan_rerank, rerank
from gradpath.net.http import PoliteClient
from gradpath.report.export import to_csv, to_markdown
from gradpath.report.table import query_results, render_table
from gradpath.sources.emails import persist_resolution, resolve_email
from gradpath.sources.ingest import discover as run_discover
from gradpath.sources.openalex import search_topics

app = typer.Typer(help="Find the professors worth emailing for graduate study.")
fields_app = typer.Typer(help="Resolve research fields to OpenAlex topics.")
institutions_app = typer.Typer(help="Inspect institutions.")
emails_app = typer.Typer(help="Resolve and report contact addresses.")
app.add_typer(fields_app, name="fields")
app.add_typer(institutions_app, name="institutions")
app.add_typer(emails_app, name="emails")

console = Console()
STATE: dict[str, Path] = {}


@app.callback()
def main(root: Path = typer.Option(Path("."), "--root", help="Workspace directory")) -> None:
    STATE["root"] = root


def _paths() -> dict[str, Path]:
    root = STATE.get("root", Path("."))
    return {
        "db": root / "gradpath.db",
        "settings": root / "settings.yaml",
        "profile": root / "profile.yaml",
        "fields": root / "data" / "fields.yaml",
        "cache": root / ".cache",
    }


def _context():
    paths = _paths()
    settings = load_settings(paths["settings"])
    profile = load_profile(paths["profile"])
    conn = connect(paths["db"])
    migrate(conn)
    client = PoliteClient(settings, profile.contact_email, paths["cache"])
    return settings, profile, conn, client


@app.command()
def init() -> None:
    """Create the database and scaffold profile.yaml."""
    paths = _paths()
    conn = connect(paths["db"])
    version = migrate(conn)
    console.print(f"[green]database ready[/green] at {paths['db']} (schema v{version})")
    try:
        scaffold_profile(paths["profile"])
        console.print(f"[green]wrote[/green] {paths['profile']} — fill in contact_email")
    except FileExistsError:
        console.print(f"profile already exists at {paths['profile']}, left untouched")


@fields_app.command("search")
def fields_search(query: str) -> None:
    """Find OpenAlex topic ids for a research area."""
    _, _, _, client = _context()
    for hit in search_topics(client, query):
        console.print(f"{hit.topic_id}\t{hit.works_count:>8}\t{hit.label}")


@fields_app.command("list")
def fields_list() -> None:
    for slug, definition in load_fields(_paths()["fields"]).items():
        console.print(f"{slug}\t{definition.label}\t{','.join(definition.topics)}")


@app.command()
def discover(
    field: list[str] = typer.Option(None, "--field"),
    country: list[str] = typer.Option(None, "--country"),
    institution: list[str] = typer.Option(None, "--institution"),
    since: int = typer.Option(None, "--since"),
) -> None:
    """Find researchers. Field-first by default; --institution targets a fixed list."""
    settings, profile, conn, client = _context()
    definitions = load_fields(_paths()["fields"])
    names = field or profile.fields
    try:
        topics = [t for name in names for t in resolve_field(name, definitions).topics]
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        raise typer.Exit(code=1) from exc

    countries = [c.upper() for c in (country or profile.countries)]
    since_year = since or settings.default_since_year
    key = f"discover:{'|'.join(sorted(topics))}:{'|'.join(sorted(countries))}:{since_year}"
    stats = run_discover(conn, client, topics, countries, since_year, key,
                         list(institution) if institution else None)
    console.print(
        f"[green]{stats.works_seen}[/green] works, "
        f"[green]{stats.people_seen}[/green] people, "
        f"[green]{stats.institutions_added}[/green] new institutions"
    )


@app.command("faculty")
def faculty_score_command(
    action: str = typer.Argument("score", help="only 'score' is supported"),
) -> None:
    """Estimate who is faculty. Nobody is removed; this fills a sortable column."""
    if action != "score":
        raise typer.BadParameter("only 'score' is supported")
    _, _, conn, _ = _context()
    rows = conn.execute(
        "SELECT id, first_year, last_year, works_count, last_author_ratio, "
        "first_author_ratio, title FROM people"
    ).fetchall()
    for row in rows:
        works = row["works_count"] or 0
        history = PublicationHistory(
            first_year=row["first_year"] or 0,
            last_year=row["last_year"] or 0,
            works_count=works,
            last_author_count=int((row["last_author_ratio"] or 0.0) * works),
            first_author_count=int((row["first_author_ratio"] or 0.0) * works),
            orcid_title=row["title"],
        )
        assessment = score_faculty(history)
        with conn:
            conn.execute(
                "UPDATE people SET faculty_score = ?, faculty_confidence = ?, "
                "faculty_signals = ? WHERE id = ?",
                (assessment.score, assessment.confidence,
                 json.dumps(assessment.signals), row["id"]),
            )
    console.print(f"[green]scored {len(rows)} people[/green]")


@emails_app.command("resolve")
def emails_resolve(
    min_score: float = typer.Option(0.0, "--min-score"),
    faculty_only: bool = typer.Option(False, "--faculty-only"),
) -> None:
    """Resolve addresses for people who rank, so crawling effort follows the ranking."""
    _, profile, conn, client = _context()
    targets = [r.name for r in query_results(conn, profile.name, min_score, faculty_only)]
    resolved = 0
    for name in targets:
        person = conn.execute("SELECT * FROM people WHERE name = ?", (name,)).fetchone()
        if person is None or person["email"]:
            continue
        resolution = resolve_email(conn, client, person)
        persist_resolution(conn, person["id"], resolution)
        resolved += int(resolution.email is not None)
    console.print(f"[green]resolved {resolved}[/green] of {len(targets)} considered")


@emails_app.command("report")
def emails_report() -> None:
    """List people with no address found, for manual lookup."""
    _, _, conn, _ = _context()
    rows = conn.execute(
        "SELECT name, homepage FROM people WHERE email IS NULL ORDER BY faculty_score DESC"
    ).fetchall()
    for row in rows:
        console.print(f"{row['name']}\t{row['homepage'] or '(no homepage)'}")
    console.print(f"[yellow]{len(rows)} addresses need manual lookup[/yellow]")


@app.command("match")
def match_command(
    do_rerank: bool = typer.Option(False, "--rerank"),
    min_score: float = typer.Option(None, "--min-score"),
    budget: float = typer.Option(None, "--budget"),
) -> None:
    """Score every person. No cap."""
    settings, profile, conn, _ = _context()
    embedder = load_embedder(settings.embedding_model)
    added = embed_pending_works(conn, embedder, settings.embedding_model,
                                settings.embedding_batch_size)
    console.print(f"embedded {added} new works")
    vector = build_profile_vector(profile, [], embedder)
    scored = run_match(conn, profile.name, vector, settings.embedding_model)
    console.print(f"[green]scored {scored} people[/green]")

    if not do_rerank:
        return
    threshold = min_score if min_score is not None else settings.show_min_score
    ceiling = budget if budget is not None else settings.rerank_default_budget_usd

    def ask(plan) -> bool:
        return typer.confirm(
            f"rerank {plan.person_count} people for about ${plan.estimated_cost_usd:.2f}?"
        )

    try:
        count = rerank(conn, profile.name, threshold, settings.rerank_model, ceiling, ask,
                       profile.interests)
        console.print(f"[green]reranked {count} people[/green]")
    except MissingApiKey as exc:
        console.print(f"[yellow]{exc}[/yellow]")


@app.command("show")
def show_command(
    min_score: float = typer.Option(None, "--min-score"),
    faculty_only: bool = typer.Option(False, "--faculty-only"),
    country: str = typer.Option(None, "--country"),
    limit: int = typer.Option(None, "--limit", help="display only; never affects scoring"),
) -> None:
    settings, profile, conn, _ = _context()
    threshold = min_score if min_score is not None else settings.show_min_score
    rows = query_results(conn, profile.name, threshold, faculty_only, country, limit)
    console.print(render_table(rows))


@app.command("export")
def export_command(
    csv_path: Path = typer.Option(None, "--csv"),
    markdown_path: Path = typer.Option(None, "--markdown"),
    min_score: float = typer.Option(None, "--min-score"),
    faculty_only: bool = typer.Option(False, "--faculty-only"),
    country: str = typer.Option(None, "--country"),
    limit: int = typer.Option(None, "--limit"),
) -> None:
    settings, profile, conn, _ = _context()
    threshold = min_score if min_score is not None else settings.show_min_score
    rows = query_results(conn, profile.name, threshold, faculty_only, country, limit)
    if csv_path:
        to_csv(rows, csv_path)
        console.print(f"[green]wrote[/green] {csv_path}")
    if markdown_path:
        to_markdown(rows, markdown_path)
        console.print(f"[green]wrote[/green] {markdown_path}")
    if not csv_path and not markdown_path:
        raise typer.BadParameter("pass --csv or --markdown")


@institutions_app.command("list")
def institutions_list(
    discovered: bool = typer.Option(False, "--discovered"),
) -> None:
    _, _, conn, _ = _context()
    sql = "SELECT id, name, country, discovered FROM institutions"
    if discovered:
        sql += " WHERE discovered = 1"
    for row in conn.execute(sql + " ORDER BY name"):
        console.print(f"{row['id']}\t{row['country'] or '--'}\t{row['name']}")
```

```bash
#!/usr/bin/env bash
# gradpath.sh — run gradpath from a checkout without installing
set -euo pipefail
exec python -m gradpath.cli "$@"
```

Add a Usage section to `README.md` showing the full run: `init`, `fields search`, `discover --field ... --country ...`, `faculty score`, `match`, `emails resolve --min-score 0.7`, `show --min-score 0.7 --faculty-only`, `export --csv targets.csv`. State that only `--rerank` needs `ANTHROPIC_API_KEY` and everything else works without any key.

- [ ] **Step 4: Run the whole suite**

Run: `pytest -v`
Expected: all tests pass across every task

- [ ] **Step 5: Commit**

```bash
chmod +x gradpath.sh
git add gradpath/cli.py gradpath.sh README.md tests/test_cli.py
git commit -m "feat: CLI wiring and end-to-end pipeline"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: architecture and repo layout (1), data model (1), configuration and profile embedding weights (2, 8), network policy (3), OpenAlex parsing and topic search (4), field-first and institution-targeted discovery with cursors (5), the mean-of-top-3 scoring rule (6), faculty likelihood including the alphabetical-field and authoritative-override rules (7), the embedding cache and vectorised scale requirement (8, plus `tests/test_scale.py`), the four-step email chain and the no-pattern-guessing rule (9, 10), adapters and the shared contract test (10, 11), budget-gated rerank and the missing-key path (12), threshold filtering with display-only `--limit` and separate score columns (13), and the CLI surface (14). Every row of the spec's error-handling table has a test except "corrupt/partial DB", which is covered structurally by transactional migrations in Task 1.

**Placeholder scan.** No TBDs, no "add error handling", no "similar to Task N". Every code step contains runnable code; the three places describing files rather than showing them in full (README, `settings.yaml`/`data/*.yaml` contents, the GIST and SNU adapters) state exactly what goes in them and, for the adapters, which four attributes differ from the KAIST body shown in full.

**Type consistency.** `PoliteClient.get_json/get_text/allowed` are used with those exact names in Tasks 4, 5, 9 and 10. `ParsedWork`/`ParsedAuthorship` field names match between `parse_work` (Task 4) and `ingest.py` (Task 5). `score_person` returns `PersonScore(score, sparse, top_work_ids)` and is consumed with those attributes in Task 8. `PublicationHistory` and `FacultyAssessment` field names match between Task 7 and the CLI in Task 14. `EmailResolution(email, confidence, source, title)` matches between `resolve_email` and `persist_resolution`. `ResultRow` fields match `FIELDNAMES` in `export.py`.

**One intentional cross-task dependency**, flagged in Task 10 Step 4: the adapter contract test in `tests/test_email_chain.py` requires at least one registered adapter and therefore passes only once Task 11 lands. Running Tasks 10 and 11 in order resolves it.
