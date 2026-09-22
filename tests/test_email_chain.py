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
