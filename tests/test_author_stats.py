import httpx
import respx

from gradpath.db import connect, migrate
from gradpath.models import Settings
from gradpath.net.http import HostBlocked, PoliteClient
from gradpath.sources.author_stats import (
    AuthorCareer,
    enrich_person_career,
    fetch_author_career,
)

SETTINGS = Settings("m", 8, 1000.0, 2021, 0.6, "claude-opus-5", 1.0, ".cache")


def _client(tmp_path):
    return PoliteClient(SETTINGS, "me@example.com", tmp_path / "c")


@respx.mock
def test_fetch_author_career_derives_outer_bounds_skipping_zero_years(tmp_path):
    client = _client(tmp_path)
    respx.get("https://api.openalex.org/authors/A1").mock(
        return_value=httpx.Response(200, json={
            "works_count": 87,
            "counts_by_year": [
                {"year": 2024, "works_count": 10},
                {"year": 2023, "works_count": 0},
                {"year": 2010, "works_count": 3},
                {"year": 1998, "works_count": 0},
            ],
        })
    )
    career = fetch_author_career(client, "A1")
    assert career == AuthorCareer(works_count=87, first_year=2010, last_year=2024)


@respx.mock
def test_fetch_author_career_empty_counts_yields_none_years_with_real_count(tmp_path):
    client = _client(tmp_path)
    respx.get("https://api.openalex.org/authors/A2").mock(
        return_value=httpx.Response(200, json={"works_count": 5, "counts_by_year": []})
    )
    career = fetch_author_career(client, "A2")
    assert career == AuthorCareer(works_count=5, first_year=None, last_year=None)


@respx.mock
def test_fetch_author_career_returns_none_on_missing_record(tmp_path):
    client = _client(tmp_path)
    respx.get("https://api.openalex.org/authors/A404").mock(
        return_value=httpx.Response(404, json={})
    )
    assert fetch_author_career(client, "A404") is None


def test_fetch_author_career_returns_none_on_host_blocked(tmp_path, monkeypatch):
    client = _client(tmp_path)

    def _raise(*args, **kwargs):
        raise HostBlocked("blocked")

    monkeypatch.setattr(client, "get_json", _raise)
    assert fetch_author_career(client, "A3") is None


@respx.mock
def test_enrich_person_career_writes_only_career_columns(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO institutions (id, name, added_at) VALUES ('i', 'I', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO people (id, institution_id, name, openalex_author_id, "
        "works_count, first_year, last_year, last_author_ratio, first_author_ratio) "
        "VALUES (1, 'i', 'A Person', 'A1', 3, 2023, 2024, 0.5, 0.2)"
    )
    conn.commit()
    client = _client(tmp_path)
    respx.get("https://api.openalex.org/authors/A1").mock(
        return_value=httpx.Response(200, json={
            "works_count": 120,
            "counts_by_year": [
                {"year": 2024, "works_count": 10},
                {"year": 1998, "works_count": 2},
            ],
        })
    )
    person_row = conn.execute("SELECT * FROM people WHERE id = 1").fetchone()
    assert enrich_person_career(conn, client, person_row) is True

    row = conn.execute("SELECT * FROM people WHERE id = 1").fetchone()
    assert row["career_first_year"] == 1998
    assert row["career_last_year"] == 2024
    assert row["career_works_count"] == 120
    # Slice-derived columns are untouched.
    assert row["works_count"] == 3
    assert row["first_year"] == 2023
    assert row["last_year"] == 2024
    assert row["last_author_ratio"] == 0.5
    assert row["first_author_ratio"] == 0.2


def test_enrich_person_career_returns_false_without_openalex_id(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO institutions (id, name, added_at) VALUES ('i', 'I', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO people (id, institution_id, name) VALUES (1, 'i', 'No OpenAlex Id')"
    )
    conn.commit()
    client = _client(tmp_path)
    person_row = conn.execute("SELECT * FROM people WHERE id = 1").fetchone()
    assert enrich_person_career(conn, client, person_row) is False
