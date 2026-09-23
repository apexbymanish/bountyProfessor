import httpx
import pytest
import respx

from gradpath.db import connect, migrate
from gradpath.models import Settings
from gradpath.net.http import HostBlocked, PoliteClient
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


def test_adapter_host_blocked_degrades_instead_of_propagating(db, client, monkeypatch):
    """A circuit-broken faculty-directory host must degrade the chain, not abort it.

    Every other step in resolve_email (crossref, orcid, crawler) guards HostBlocked
    internally. _adapter_email must do the same: a host that has circuit-broken
    after repeated failures should fall through to the crawler step (or to "none"),
    never propagate out of resolve_email and abort a long resolution run.
    """
    from typing import ClassVar

    from gradpath.sources.adapters import _REGISTRY
    from gradpath.sources.adapters.base import InstitutionAdapter

    class _BlockedAdapter(InstitutionAdapter):
        slug = "test-hostblocked"
        domains: ClassVar[list[str]] = ["blocked.example.com"]

        def faculty_urls(self):
            return ["https://blocked.example.com/faculty"]

        def parse_faculty(self, html, url):
            return []

    monkeypatch.setitem(_REGISTRY, _BlockedAdapter.slug, _BlockedAdapter)

    def _raise(url):
        raise HostBlocked("blocked")

    monkeypatch.setattr(client, "get_text", _raise)

    db.execute(
        "INSERT INTO institutions (id, name, added_at, adapter) VALUES "
        "('i2', 'Blocked Institution', '2026-01-01', 'test-hostblocked')"
    )
    db.execute(
        "INSERT INTO people (id, institution_id, name, openalex_author_id) "
        "VALUES (2, 'i2', 'No Homepage Person', 'A2')"
    )
    db.commit()

    person_row = db.execute("SELECT * FROM people WHERE id = 2").fetchone()
    result = resolve_email(db, client, person_row)

    assert result.email is None
    assert result.confidence == "none"


# ============================================================================
# C2 / I4: the adapter step must match a person properly, and never borrow an
# address from elsewhere on the directory page.
# ============================================================================

DIRECTORY_URL = "https://dir.example.com/faculty"
NEIGHBOUR_HTML = (
    "<html><body>"
    '<a href="mailto:someone.else@dir.example.com">head of department</a>'
    "</body></html>"
)


def _use_directory(monkeypatch, records, html=NEIGHBOUR_HTML):
    """Register a fake adapter returning `records` for a page serving `html`."""
    from typing import ClassVar

    from gradpath.sources.adapters import _REGISTRY
    from gradpath.sources.adapters.base import InstitutionAdapter

    class _DirectoryAdapter(InstitutionAdapter):
        slug = "test-directory"
        domains: ClassVar[list[str]] = ["dir.example.com"]

        def faculty_urls(self):
            return [DIRECTORY_URL]

        def parse_faculty(self, page_html, url):
            return list(records)

    monkeypatch.setitem(_REGISTRY, _DirectoryAdapter.slug, _DirectoryAdapter)
    respx.get("https://dir.example.com/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nAllow: /\n")
    )
    respx.get(DIRECTORY_URL).mock(return_value=httpx.Response(200, text=html))


def _record(name, email=None, title=None, homepage=None):
    from gradpath.models import FacultyRecord

    return FacultyRecord(name=name, email=email, title=title, homepage=homepage, dept=None)


def _directory_person(db, name, person_id=2, homepage=None):
    """A person at an adapter-backed institution, with no DOI and no ORCID, so
    the chain reaches the adapter step without any crossref/orcid traffic."""
    db.execute(
        "INSERT OR IGNORE INTO institutions (id, name, added_at, adapter) "
        "VALUES ('i2', 'Directory Institution', '2026-01-01', 'test-directory')"
    )
    db.execute(
        "INSERT INTO people (id, institution_id, name, openalex_author_id, homepage) "
        "VALUES (?, 'i2', ?, ?, ?)",
        (person_id, name, f"A{person_id}", homepage),
    )
    db.commit()
    return db.execute("SELECT * FROM people WHERE id = ?", (person_id,)).fetchone()


@respx.mock
def test_person_absent_from_the_directory_gets_no_address(db, client, monkeypatch):
    """C2: a surname substring match handed a different Kim's address back at
    confidence='high', source='adapter'."""
    _use_directory(monkeypatch, [_record("Jaewon Kim", "jaewon@dir.example.com", "Professor")])
    person = _directory_person(db, "Minsoo Kim")
    result = resolve_email(db, client, person)
    assert result.email is None
    assert result.source != "adapter"
    assert result.title is None, "a title from an unmatched record must not be persisted"


@respx.mock
def test_matched_record_without_an_address_never_borrows_a_neighbours(db, client, monkeypatch):
    """C2: the old fallback took the first address anywhere on the page."""
    _use_directory(monkeypatch, [_record("Sunmi Park", None, "Associate Professor")])
    person = _directory_person(db, "Sunmi Park")
    result = resolve_email(db, client, person)
    assert result.email is None
    assert result.email != "someone.else@dir.example.com"


@respx.mock
def test_same_surname_records_do_not_hand_back_the_first_one(db, client, monkeypatch):
    """C2: `surname in record.name` returned whichever Kim was listed first."""
    _use_directory(monkeypatch, [
        _record("Jaewon Kim", "jaewon@dir.example.com"),
        _record("Minsoo Kim", "minsoo@dir.example.com"),
    ])
    person = _directory_person(db, "Minsoo Kim")
    result = resolve_email(db, client, person)
    assert result.email == "minsoo@dir.example.com"


@respx.mock
def test_two_records_with_the_same_name_are_ambiguous_and_yield_nothing(
    db, client, monkeypatch
):
    _use_directory(monkeypatch, [
        _record("Jaewon Kim", "jaewon1@dir.example.com"),
        _record("Jaewon Kim", "jaewon2@dir.example.com"),
    ])
    person = _directory_person(db, "Jaewon Kim")
    result = resolve_email(db, client, person)
    assert result.email is None


@respx.mock
def test_unambiguous_full_name_match_still_resolves(db, client, monkeypatch):
    _use_directory(monkeypatch, [
        _record("Jaewon Kim", "jaewon@dir.example.com"),
        _record("Prof. Sunmi Park", "park@dir.example.com", "Associate Professor"),
    ])
    person = _directory_person(db, "Sunmi Park")
    result = resolve_email(db, client, person)
    assert result.email == "park@dir.example.com"
    assert result.confidence == "high"
    assert result.source == "adapter"
    assert result.title == "Associate Professor"


@respx.mock
def test_name_order_and_accents_do_not_defeat_the_match(db, client, monkeypatch):
    _use_directory(monkeypatch, [_record("Park Sunmi", "park@dir.example.com")])
    person = _directory_person(db, "Sunmi Park")
    assert resolve_email(db, client, person).email == "park@dir.example.com"


# --- I4: an adapter-supplied homepage must be persisted and reach the crawler ---


@respx.mock
def test_adapter_homepage_is_stored_and_used_by_the_crawler_stage(db, client, monkeypatch):
    """I4: people.homepage was read twice and written nowhere, so the fourth
    source of the four-source chain was unreachable in production."""
    from gradpath.sources.emails import persist_resolution

    homepage = "https://dir.example.com/people/park"
    _use_directory(monkeypatch, [_record("Sunmi Park", None, None, homepage)])
    respx.get(homepage).mock(return_value=httpx.Response(
        200, text='<a href="mailto:park@dir.example.com">mail</a>'
    ))
    person = _directory_person(db, "Sunmi Park")

    result = resolve_email(db, client, person)
    assert result.email == "park@dir.example.com"
    assert result.source == "crawler"
    assert result.homepage == homepage

    persist_resolution(db, person["id"], result)
    stored = db.execute("SELECT homepage FROM people WHERE id = ?", (person["id"],)).fetchone()
    assert stored["homepage"] == homepage


@respx.mock
def test_persist_resolution_never_clears_an_existing_homepage(db, client, monkeypatch):
    from gradpath.sources.emails import EmailResolution, persist_resolution

    person = _directory_person(db, "Sunmi Park", homepage="https://lab.example.com/park")
    persist_resolution(db, person["id"], EmailResolution(None, "none", None))
    stored = db.execute("SELECT homepage FROM people WHERE id = ?", (person["id"],)).fetchone()
    assert stored["homepage"] == "https://lab.example.com/park"
