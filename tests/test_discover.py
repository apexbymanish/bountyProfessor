import httpx
import pytest
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
    assert conn.execute("SELECT COUNT(*) FROM people").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM institutions").fetchone()[0] == 1


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


@respx.mock
def test_discover_skips_work_with_missing_openalex_id(tmp_path):
    """A page with an id-less work must not corrupt the id-less-work-collision case.

    parse_work falls back to "" when a work has no id, and works.openalex_work_id
    is UNIQUE. Without a guard, two such works would collide on "" and the second
    work's authorships would silently attach to the first work's row. Here there
    is exactly one id-less work and one valid work, so the assertion is: the
    id-less work produces no row at all, and the valid work's own authorship is
    intact (not merged with anything).
    """
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    client = PoliteClient(SETTINGS, "me@example.com", tmp_path / "c")
    bad_work = _work("W1", "A1")
    bad_work["id"] = None
    good_work = _work("W2", "A2")
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json=_page([bad_work, good_work], None))
    )
    stats = discover(conn, client, ["T10028"], ["KR"], 2021, "k")
    assert stats.works_seen == 1
    assert conn.execute("SELECT COUNT(*) FROM works").fetchone()[0] == 1
    work_row = conn.execute("SELECT id, openalex_work_id FROM works").fetchone()
    assert work_row["openalex_work_id"] == "W2"
    assert conn.execute("SELECT COUNT(*) FROM authorships").fetchone()[0] == 1
    authorship = conn.execute(
        "SELECT person_id, work_id FROM authorships"
    ).fetchone()
    assert authorship["work_id"] == work_row["id"]
    person = conn.execute(
        "SELECT openalex_author_id FROM people WHERE id = ?", (authorship["person_id"],)
    ).fetchone()
    assert person["openalex_author_id"] == "A2"


@respx.mock
def test_institution_with_missing_display_name_falls_back_to_openalex_id(tmp_path):
    """institutions.name is NOT NULL; a missing display_name must not abort the run.

    Storing the OpenAlex institution id as a name-of-last-resort (mirroring the
    fallback the id/slug already uses) keeps the person linked to a real
    institution row instead of the whole `with conn:` block raising
    IntegrityError and aborting an hours-long discovery run on one malformed
    record. Task 15's resolution step later overwrites this with the
    authoritative display name.
    """
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    client = PoliteClient(SETTINGS, "me@example.com", tmp_path / "c")
    work = _work("W1", "A1")
    work["authorships"][0]["institutions"][0]["display_name"] = None
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json=_page([work], None))
    )
    discover(conn, client, ["T10028"], ["KR"], 2021, "k")
    assert conn.execute("SELECT COUNT(*) FROM institutions").fetchone()[0] == 1
    institution = conn.execute("SELECT id, name FROM institutions").fetchone()
    assert institution["name"] == "I1"
    person = conn.execute(
        "SELECT institution_id FROM people WHERE openalex_author_id = 'A1'"
    ).fetchone()
    assert person["institution_id"] == institution["id"]


@respx.mock
def test_query_params_include_filter_cursor_and_mailto(tmp_path):
    """The filter/cursor/mailto parameters are the actual OpenAlex contract.

    Every other test here matches respx on URL alone, so a regression in the
    filter syntax, the cursor advancing between pages, or mailto being
    dropped (or duplicated) could land with the suite still green. This test
    inspects the real outgoing query parameters instead.
    """
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    client = PoliteClient(SETTINGS, "me@example.com", tmp_path / "c")
    route = respx.get("https://api.openalex.org/works").mock(side_effect=[
        httpx.Response(200, json=_page([_work("W1", "A1")], "cur2")),
        httpx.Response(200, json=_page([_work("W2", "A2")], None)),
    ])
    discover(conn, client, ["T10028"], ["KR"], 2021, "k")
    assert route.call_count == 2

    first_params = route.calls[0].request.url.params
    assert first_params["filter"] == (
        "publication_year:>2020,topics.id:T10028,institutions.country_code:kr"
    )
    assert first_params["per-page"] == "200"
    assert first_params["cursor"] == "*"
    assert first_params["mailto"] == "me@example.com"
    assert first_params.get_list("mailto") == ["me@example.com"]  # exactly once

    second_params = route.calls[1].request.url.params
    assert second_params["cursor"] == "cur2"
    assert second_params["mailto"] == "me@example.com"


@respx.mock
def test_institutions_filter_takes_precedence_over_countries(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    client = PoliteClient(SETTINGS, "me@example.com", tmp_path / "c")
    route = respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json=_page([], None))
    )
    discover(conn, client, ["T10028"], ["KR"], 2021, "k", institutions=["I1", "I2"])
    params = route.calls[0].request.url.params
    assert "institutions.id:I1|I2" in params["filter"]
    assert "institutions.country_code" not in params["filter"]


@respx.mock
def test_page_level_refresh_survives_a_later_page_failure(monkeypatch, tmp_path):
    """discover refreshes stats per page, not once at the end of the run.

    Without that, a crash while fetching a later page (HostBlocked after
    MAX_ATTEMPTS on an hours-long crawl is exactly this scenario) would
    leave every already-committed person from earlier pages stuck with
    works_count=0 and NULL years -- stale defaults Task 7 reads to decide
    who is faculty. Simulate iter_pages raising after the first of two
    pages and assert page 1's person already has correct, non-default
    stats.
    """
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    client = PoliteClient(SETTINGS, "me@example.com", tmp_path / "c")
    page1 = _page([_work("W1", "A1")], "cur2")
    calls = {"n": 0}

    def fake_get_json(url, params=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return page1
        raise RuntimeError("simulated crash mid-crawl")

    monkeypatch.setattr(client, "get_json", fake_get_json)

    with pytest.raises(RuntimeError):
        discover(conn, client, ["T10028"], ["KR"], 2021, "k")

    row = conn.execute(
        "SELECT works_count, first_year, last_year FROM people WHERE openalex_author_id = 'A1'"
    ).fetchone()
    assert row["works_count"] == 1
    assert row["first_year"] == 2024
    assert row["last_year"] == 2024


def test_interrupted_run_resumes_from_persisted_cursor_without_duplicating(monkeypatch, tmp_path):
    """A genuine partial-resume test, not a manual cursor reset.

    `test_rerunning_discover_is_idempotent` proves full re-ingest is safe by
    manually resetting the cursor to '*'. This test proves the actually
    load-bearing case: a run that dies mid-crawl persists its cursor past
    page 1, and a second call to discover picks up from that persisted
    cursor (not from '*') and produces no duplicate rows.
    """
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    client = PoliteClient(SETTINGS, "me@example.com", tmp_path / "c")

    page1 = _page([_work("W1", "A1")], "cur2")
    page2 = _page([_work("W2", "A2")], None)
    responses = iter([page1, RuntimeError("boom"), page2])

    def fake_get_json(url, params=None):
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(client, "get_json", fake_get_json)

    with pytest.raises(RuntimeError):
        discover(conn, client, ["T10028"], ["KR"], 2021, "k")

    # Page 1 fully committed (works + stats), and the cursor advanced past it
    # -- it must not still read "*", or a resumed run would restart from
    # scratch instead of resuming.
    assert conn.execute("SELECT COUNT(*) FROM works").fetchone()[0] == 1
    row = conn.execute(
        "SELECT works_count FROM people WHERE openalex_author_id = 'A1'"
    ).fetchone()
    assert row["works_count"] == 1
    cursor_row = conn.execute("SELECT cursor FROM cursors WHERE key = 'k'").fetchone()
    assert cursor_row["cursor"] == "cur2"

    # Resume: discover picks up from the persisted cursor and fetches only
    # what remains (the fake client's next response is page 2).
    discover(conn, client, ["T10028"], ["KR"], 2021, "k")

    assert conn.execute("SELECT COUNT(*) FROM works").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM authorships").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM people").fetchone()[0] == 2
    ids = [r["openalex_work_id"] for r in conn.execute("SELECT openalex_work_id FROM works")]
    assert sorted(ids) == ["W1", "W2"]
