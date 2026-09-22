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
