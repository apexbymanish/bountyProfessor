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


def test_person_id_is_populated_and_survives_export(db, tmp_path):
    """Test that person_id is in ResultRow and survives CSV export."""
    rows = query_results(db, "default")
    # Check person_id is populated in ResultRow
    assert hasattr(rows[0], 'person_id'), "ResultRow must have person_id attribute"
    assert rows[0].person_id == 3, "First result should be person_id 3 (Prof DE)"
    assert rows[1].person_id == 2, "Second result should be person_id 2 (Maybe Student)"
    assert rows[2].person_id == 1, "Third result should be person_id 1 (Prof High)"

    # Check person_id survives CSV export
    path = tmp_path / "out.csv"
    to_csv(rows, path)
    csv_rows = list(csv.DictReader(path.open()))
    assert "person_id" in csv_rows[0], "person_id must be in CSV headers"
    assert csv_rows[0]["person_id"] == "3", "CSV person_id for Prof DE should be 3"
    assert csv_rows[1]["person_id"] == "2", "CSV person_id for Maybe Student should be 2"
    assert csv_rows[2]["person_id"] == "1", "CSV person_id for Prof High should be 1"
