import httpx
import pytest
import respx

from gradpath.db import connect, migrate
from gradpath.models import Settings
from gradpath.net.http import PoliteClient
from gradpath.sources.institutions import (
    InstitutionHit,
    import_ranking_csv,
    top_institutions,
    upsert_institutions,
)

SETTINGS = Settings("m", 8, 1000.0, 2021, 0.6, "claude-opus-5", 1.0, ".cache")


def _payload(names: list[str], country: str = "KR") -> dict:
    return {"results": [
        {
            "id": f"https://openalex.org/I{index}",
            "ror": f"https://ror.org/0{index}abcde",
            "display_name": name,
            "country_code": country,
            "homepage_url": f"https://{name.lower().replace(' ', '')}.ac.kr",
            "works_count": 10_000 - index,
            "cited_by_count": 500_000 - index,
        }
        for index, name in enumerate(names)
    ]}


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    return conn


@pytest.fixture
def client(tmp_path):
    return PoliteClient(SETTINGS, "me@example.com", tmp_path / "c")


def test_migration_four_adds_tier_columns(db):
    columns = {r[1] for r in db.execute("PRAGMA table_info(institutions)")}
    assert {"tier", "rank", "rank_source"} <= columns


@respx.mock
def test_top_institutions_requests_country_filter(client):
    route = respx.get("https://api.openalex.org/institutions").mock(
        return_value=httpx.Response(200, json=_payload(["SNU", "KAIST"]))
    )
    hits = top_institutions(client, "KR", 20, "cited_by_count")
    assert [h.name for h in hits] == ["SNU", "KAIST"]
    assert "country_code:kr" in route.calls[0].request.url.params["filter"]
    # R5: the original assertion (`== "0ror" or .startswith("0")`) is
    # unfailable -- assert the exact value instead.
    assert hits[0].ror_id == "00abcde"


@respx.mock
def test_top_institutions_rejects_an_unsupported_metric(client):
    with pytest.raises(ValueError, match="metric"):
        top_institutions(client, "KR", 20, "vibes")


@respx.mock
def test_upsert_assigns_tier_and_rank_in_order(db, client):
    respx.get("https://api.openalex.org/institutions").mock(
        return_value=httpx.Response(200, json=_payload(["SNU", "KAIST", "Yonsei"]))
    )
    hits = top_institutions(client, "KR", 20, "cited_by_count")
    assert upsert_institutions(db, hits, "korea-20", "openalex:cited_by_count") == 3
    rows = db.execute(
        "SELECT name, tier, rank, rank_source FROM institutions ORDER BY rank"
    ).fetchall()
    assert [r["name"] for r in rows] == ["SNU", "KAIST", "Yonsei"]
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert rows[0]["tier"] == "korea-20"
    assert rows[0]["rank_source"] == "openalex:cited_by_count"


@respx.mock
def test_reseeding_updates_rank_without_duplicating(db, client):
    respx.get("https://api.openalex.org/institutions").mock(
        return_value=httpx.Response(200, json=_payload(["SNU", "KAIST"]))
    )
    hits = top_institutions(client, "KR", 20, "cited_by_count")
    upsert_institutions(db, hits, "korea-20", "openalex:cited_by_count")
    upsert_institutions(db, list(reversed(hits)), "korea-20", "openalex:cited_by_count")
    assert db.execute("SELECT COUNT(*) FROM institutions").fetchone()[0] == 2
    top = db.execute("SELECT name FROM institutions WHERE rank = 1").fetchone()
    assert top["name"] == "KAIST"


def test_upsert_preserves_a_hand_bound_adapter(db, client):
    db.execute(
        "INSERT INTO institutions (id, name, openalex_id, adapter, added_at) "
        "VALUES ('kaist','KAIST','I1','kaist','2026-01-01')"
    )
    db.commit()

    hit = InstitutionHit("I1", "01ror", "KAIST", "KR", "https://kaist.ac.kr", 9, 9)
    upsert_institutions(db, [hit], "korea-20", "openalex:cited_by_count")
    row = db.execute("SELECT adapter, tier FROM institutions WHERE openalex_id='I1'").fetchone()
    assert row["adapter"] == "kaist"
    assert row["tier"] == "korea-20"


# --- Correction 2: resolution must bind an OpenAlex id onto an already-
# seeded row by matching homepage domain, not create a second row keyed by
# slugify(name). For KAIST, slugify("Korea Advanced Institute of Science and
# Technology") != "kaist", so the naive lookup-by-openalex_id-then-slugify
# fallback silently orphans the adapter-bearing seeded row.


def test_upsert_resolves_seeded_row_by_homepage_domain_not_a_second_row(db, client):
    # Mirrors what `init` actually leaves in the table via data/institutions.yaml:
    # slug id, curated site+adapter, no openalex_id yet.
    db.execute(
        "INSERT INTO institutions (id, name, site, adapter, discovered, added_at) "
        "VALUES ('kaist', 'Korea Advanced Institute of Science and Technology', "
        "'https://www.kaist.ac.kr', 'kaist', 0, '2026-01-01')"
    )
    db.commit()

    hit = InstitutionHit(
        openalex_id="I999",
        ror_id="01abcde",
        name="Korea Advanced Institute of Science and Technology",
        country="KR",
        site="https://kaist.ac.kr",  # same registrable host, no "www."
        works_count=10,
        cited_by_count=10,
    )
    upsert_institutions(db, [hit], "korea-20", "openalex:cited_by_count")

    rows = db.execute("SELECT id, openalex_id, adapter, tier FROM institutions").fetchall()
    assert len(rows) == 1, "resolution must not insert a second, orphaned row"
    row = rows[0]
    assert row["id"] == "kaist"
    assert row["openalex_id"] == "I999"
    assert row["adapter"] == "kaist"
    assert row["tier"] == "korea-20"


def test_import_ranking_csv_reads_rank_and_name(db, tmp_path):
    path = tmp_path / "qs2026.csv"
    path.write_text(
        "rank,name,country\n1,Massachusetts Institute of Technology,US\n"
        "2,Imperial College London,GB\n3,Stanford University,US\n"
    )
    assert import_ranking_csv(db, path, "world-100", top=2) == 2
    rows = db.execute(
        "SELECT name, rank, tier, openalex_id FROM institutions ORDER BY rank"
    ).fetchall()
    assert [r["name"] for r in rows] == [
        "Massachusetts Institute of Technology", "Imperial College London"
    ]
    assert rows[0]["tier"] == "world-100"
    # R: a CSV import never invents an OpenAlex id; resolution is separate.
    assert rows[0]["openalex_id"] is None


def test_import_ranking_csv_rejects_a_file_missing_required_columns(db, tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("position,institution\n1,MIT\n")
    with pytest.raises(ValueError, match="rank.*name"):
        import_ranking_csv(db, path, "world-100", top=10)
