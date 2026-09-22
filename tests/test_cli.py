import httpx
import respx
from typer.testing import CliRunner

from gradpath.cli import ENRICH_CONFIRM_THRESHOLD, app
from gradpath.match.rerank import RerankOutcome, RerankPlan

runner = CliRunner()

SETTINGS_YAML = """
embedding_model: test-model
embedding_batch_size: 8
rate_limit_per_host: 1.0
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

INSTITUTIONS_YAML = """
kaist:
  name: Korea Advanced Institute of Science and Technology
  country: KR
  site: https://www.kaist.ac.kr
  adapter: kaist
"""


def _workspace(tmp_path, institutions_yaml=None):
    (tmp_path / "settings.yaml").write_text(SETTINGS_YAML)
    (tmp_path / "profile.yaml").write_text(PROFILE_YAML)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "fields.yaml").write_text(FIELDS_YAML)
    (tmp_path / "data" / "institutions.yaml").write_text(institutions_yaml or "{}\n")
    return tmp_path


def _openalex_work_response(author_id="A1", author_name="Sunmi Park",
                             institution_id="I1", institution_name="KAIST"):
    return httpx.Response(200, json={"results": [{
        "id": "https://openalex.org/W1", "title": "Quantized inference",
        "publication_year": 2024, "doi": None, "cited_by_count": 3,
        "primary_location": {"source": {"display_name": "V"}},
        "topics": [{"id": "https://openalex.org/T10028"}],
        "abstract_inverted_index": {"quantized": [0], "inference": [1]},
        "authorships": [{
            "author_position": "last",
            "author": {"id": f"https://openalex.org/{author_id}",
                       "display_name": author_name, "orcid": None},
            "institutions": [{"id": f"https://openalex.org/{institution_id}",
                              "display_name": institution_name, "country_code": "KR"}],
        }],
    }], "meta": {"next_cursor": None}})


def test_init_creates_database_and_profile(tmp_path):
    result = runner.invoke(app, ["--root", str(tmp_path), "init"])
    assert result.exit_code == 0
    assert (tmp_path / "gradpath.db").exists()
    assert (tmp_path / "profile.yaml").exists()


def test_init_does_not_overwrite_an_existing_profile(tmp_path):
    workspace = _workspace(tmp_path)
    runner.invoke(app, ["--root", str(workspace), "init"])
    assert "contact_email: me@example.com" in (workspace / "profile.yaml").read_text()


def test_init_seeds_known_institutions_with_their_adapter(tmp_path):
    workspace = _workspace(tmp_path, institutions_yaml=INSTITUTIONS_YAML)
    result = runner.invoke(app, ["--root", str(workspace), "init"])
    assert result.exit_code == 0

    from gradpath.db import connect

    conn = connect(workspace / "gradpath.db")
    row = conn.execute("SELECT name, adapter, discovered FROM institutions WHERE id = 'kaist'")\
        .fetchone()
    assert row is not None
    assert row["adapter"] == "kaist"
    assert row["discovered"] == 0


@respx.mock
def test_discover_then_show_runs_end_to_end(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    monkeypatch.setattr(
        "gradpath.cli.load_embedder",
        lambda name: __import__("tests.conftest", fromlist=["FakeEmbedder"]).FakeEmbedder(),
    )
    respx.get("https://api.openalex.org/works").mock(return_value=_openalex_work_response())
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


def test_discover_has_no_tier_option(tmp_path):
    """R: --tier belongs to Task 15, not this one. discover must not expose it."""
    workspace = _workspace(tmp_path)
    runner.invoke(app, ["--root", str(workspace), "init"])
    result = runner.invoke(
        app, ["--root", str(workspace), "discover", "--tier", "top20"]
    )
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


# --- R27: faculty score must use whole-career data, and say when it didn't ---


def test_faculty_score_reports_career_versus_fallback_counts(tmp_path):
    from gradpath.db import connect, migrate

    workspace = _workspace(tmp_path)
    runner.invoke(app, ["--root", str(workspace), "init"])
    conn = connect(workspace / "gradpath.db")
    migrate(conn)
    conn.executescript(
        """
        INSERT INTO institutions (id, name, added_at) VALUES ('i', 'I', '2026-01-01');
        INSERT INTO people (id, institution_id, name, openalex_author_id, works_count,
                            first_year, last_year, last_author_ratio, first_author_ratio,
                            career_first_year, career_last_year, career_works_count)
        VALUES
          (1, 'i', 'Career Known', 'A1', 2, 2023, 2024, 0.5, 0.0, 2005, 2026, 120),
          (2, 'i', 'Slice Only', 'A2', 2, 2023, 2024, 0.5, 0.0, NULL, NULL, NULL);
        """
    )
    conn.commit()
    result = runner.invoke(app, ["--root", str(workspace), "faculty", "score"])
    assert result.exit_code == 0
    assert "scored 2 people" in result.stdout
    assert "1" in result.stdout  # at least one of the two counts rendered

    row1 = conn.execute("SELECT faculty_signals FROM people WHERE id = 1").fetchone()
    row2 = conn.execute("SELECT faculty_signals FROM people WHERE id = 2").fetchone()
    # Person 1 must be scored using the 21-year career span (2005-2026), not
    # the 1-year slice span (2023-2024) -- this is the actual bug fix.
    assert '"career_span": 21' in row1["faculty_signals"]
    assert '"career_span": 1' in row2["faculty_signals"]


# --- new command: populate career_* columns over a ranked, filtered subset ---


@respx.mock
def test_faculty_enrich_career_only_touches_the_filtered_subset(tmp_path):
    from gradpath.db import connect, migrate

    workspace = _workspace(tmp_path)
    runner.invoke(app, ["--root", str(workspace), "init"])
    conn = connect(workspace / "gradpath.db")
    migrate(conn)
    conn.executescript(
        """
        INSERT INTO institutions (id, name, added_at) VALUES ('i', 'I', '2026-01-01');
        INSERT INTO people (id, institution_id, name, openalex_author_id)
        VALUES (1, 'i', 'High Fit', 'A1'), (2, 'i', 'Low Fit', 'A2');
        INSERT INTO matches (profile_id, person_id, stage1_score, computed_at)
        VALUES ('default', 1, 0.9, '2026-01-01'), ('default', 2, 0.1, '2026-01-01');
        """
    )
    conn.commit()
    respx.get("https://api.openalex.org/authors/A1").mock(
        return_value=httpx.Response(200, json={
            "works_count": 50,
            "counts_by_year": [{"year": 2010, "works_count": 5},
                                {"year": 2024, "works_count": 3}],
        })
    )
    result = runner.invoke(
        app, ["--root", str(workspace), "faculty", "enrich-career", "--min-score", "0.5"]
    )
    assert result.exit_code == 0

    enriched = conn.execute("SELECT career_works_count FROM people WHERE id = 1").fetchone()
    skipped = conn.execute("SELECT career_works_count FROM people WHERE id = 2").fetchone()
    assert enriched["career_works_count"] == 50
    assert skipped["career_works_count"] is None


# --- ResultRow now carries person_id; email resolution must key off it, not name ---


def test_emails_resolve_keys_off_person_id_not_name(tmp_path, monkeypatch):
    from gradpath.db import connect, migrate

    workspace = _workspace(tmp_path)
    runner.invoke(app, ["--root", str(workspace), "init"])
    conn = connect(workspace / "gradpath.db")
    migrate(conn)
    conn.executescript(
        """
        INSERT INTO institutions (id, name, added_at) VALUES ('i', 'I', '2026-01-01');
        INSERT INTO people (id, institution_id, name, openalex_author_id)
        VALUES (1, 'i', 'Same Name', 'A1'), (2, 'i', 'Same Name', 'A2');
        INSERT INTO matches (profile_id, person_id, stage1_score, computed_at)
        VALUES ('default', 1, 0.9, '2026-01-01'), ('default', 2, 0.8, '2026-01-01');
        """
    )
    conn.commit()

    def fake_resolve_email(conn, client, person_row):
        from gradpath.sources.emails import EmailResolution
        return EmailResolution(f"person{person_row['id']}@example.com", "high", "crossref")

    monkeypatch.setattr("gradpath.cli.resolve_email", fake_resolve_email)
    result = runner.invoke(app, ["--root", str(workspace), "emails", "resolve"])
    assert result.exit_code == 0

    row1 = conn.execute("SELECT email FROM people WHERE id = 1").fetchone()
    row2 = conn.execute("SELECT email FROM people WHERE id = 2").fetchone()
    assert row1["email"] == "person1@example.com"
    assert row2["email"] == "person2@example.com"


# --- R34: --budget is an approximate ceiling; measured spend must be shown honestly ---


def test_match_help_describes_budget_as_approximate():
    result = runner.invoke(app, ["match", "--help"])
    assert result.exit_code == 0
    assert "approximate" in result.stdout.lower()


def test_match_rerank_confirmation_says_estimated_and_reports_measured_spend(
    tmp_path, monkeypatch
):
    workspace = _workspace(tmp_path)
    runner.invoke(app, ["--root", str(workspace), "init"])
    monkeypatch.setattr(
        "gradpath.cli.load_embedder",
        lambda name: __import__("tests.conftest", fromlist=["FakeEmbedder"]).FakeEmbedder(),
    )

    def fake_rerank(conn, profile_id, min_score, model, budget_usd, ask, profile_text=""):
        plan = RerankPlan(person_count=3, estimated_tokens=100, estimated_cost_usd=1.50)
        assert ask(plan) is True
        return RerankOutcome(scored=3, spent_usd=1.75)

    monkeypatch.setattr("gradpath.cli.rerank", fake_rerank)
    result = runner.invoke(
        app, ["--root", str(workspace), "match", "--rerank"], input="y\n"
    )
    assert result.exit_code == 0
    assert "estimated" in result.stdout.lower()
    assert "1.75" in result.stdout  # the measured actual spend
    assert "1.50" in result.stdout  # the pre-flight estimate, in the prompt


def test_match_rerank_without_api_key_leaves_stage1_usable(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    runner.invoke(app, ["--root", str(workspace), "init"])
    monkeypatch.setattr(
        "gradpath.cli.load_embedder",
        lambda name: __import__("tests.conftest", fromlist=["FakeEmbedder"]).FakeEmbedder(),
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(app, ["--root", str(workspace), "match", "--rerank"], input="y\n")
    assert result.exit_code == 0
    assert "stage 1" in result.stdout.lower() or "unaffected" in result.stdout.lower()


# --- R36: `faculty enrich-career` must actually scope, not just offer flags ---


def _insert_person_with_match(conn, person_id, score, institution="i"):
    conn.execute(
        "INSERT INTO people (id, institution_id, name, openalex_author_id) VALUES (?, ?, ?, ?)",
        (person_id, institution, f"Person {person_id}", f"A{person_id}"),
    )
    conn.execute(
        "INSERT INTO matches (profile_id, person_id, stage1_score, computed_at) "
        "VALUES ('default', ?, ?, '2026-01-01')",
        (person_id, score),
    )


def test_faculty_enrich_career_default_min_score_uses_settings_show_min_score(
    tmp_path, monkeypatch
):
    """A bare `faculty enrich-career` (no --min-score) must not sweep every
    discovered person just because `match` gave everyone a matches row --
    it should default to settings.show_min_score, same as `show`/`export`."""
    from gradpath.db import connect, migrate

    workspace = _workspace(tmp_path)
    (workspace / "settings.yaml").write_text(SETTINGS_YAML.replace(
        "show_min_score: 0.0", "show_min_score: 0.5"
    ))
    runner.invoke(app, ["--root", str(workspace), "init"])
    conn = connect(workspace / "gradpath.db")
    migrate(conn)
    conn.execute("INSERT INTO institutions (id, name, added_at) VALUES ('i', 'I', '2026-01-01')")
    with conn:
        _insert_person_with_match(conn, 1, 0.9)   # above threshold
        _insert_person_with_match(conn, 2, 0.1)   # below threshold
    conn.commit()

    enriched_ids: list[int] = []

    def fake_enrich(conn, client, person):
        enriched_ids.append(person["id"])
        return True

    monkeypatch.setattr("gradpath.cli.enrich_person_career", fake_enrich)
    result = runner.invoke(app, ["--root", str(workspace), "faculty", "enrich-career"])
    assert result.exit_code == 0
    assert enriched_ids == [1]  # only the person at/above show_min_score


def test_faculty_enrich_career_warns_before_a_large_run_and_can_be_declined(
    tmp_path, monkeypatch
):
    """Above ENRICH_CONFIRM_THRESHOLD people, the command must report the
    count and implied wall-clock time and ask before making any calls --
    declining must leave nothing enriched."""
    from gradpath.db import connect, migrate

    workspace = _workspace(tmp_path)
    runner.invoke(app, ["--root", str(workspace), "init"])
    conn = connect(workspace / "gradpath.db")
    migrate(conn)
    conn.execute("INSERT INTO institutions (id, name, added_at) VALUES ('i', 'I', '2026-01-01')")
    count = ENRICH_CONFIRM_THRESHOLD + 1
    with conn:
        for person_id in range(1, count + 1):
            _insert_person_with_match(conn, person_id, 0.9)
    conn.commit()

    def unreachable_enrich(conn, client, person):
        raise AssertionError("must not enrich anyone once the user declines")

    monkeypatch.setattr("gradpath.cli.enrich_person_career", unreachable_enrich)
    declined = runner.invoke(
        app, ["--root", str(workspace), "faculty", "enrich-career", "--min-score", "0.0"],
        input="n\n",
    )
    assert declined.exit_code == 0
    assert str(count) in declined.stdout
    assert "aborted" in declined.stdout.lower()


def test_faculty_enrich_career_proceeds_past_the_threshold_on_confirmation(
    tmp_path, monkeypatch
):
    from gradpath.db import connect, migrate

    workspace = _workspace(tmp_path)
    runner.invoke(app, ["--root", str(workspace), "init"])
    conn = connect(workspace / "gradpath.db")
    migrate(conn)
    conn.execute("INSERT INTO institutions (id, name, added_at) VALUES ('i', 'I', '2026-01-01')")
    count = ENRICH_CONFIRM_THRESHOLD + 1
    with conn:
        for person_id in range(1, count + 1):
            _insert_person_with_match(conn, person_id, 0.9)
    conn.commit()

    monkeypatch.setattr(
        "gradpath.cli.enrich_person_career", lambda conn, client, person: True
    )
    accepted = runner.invoke(
        app, ["--root", str(workspace), "faculty", "enrich-career", "--min-score", "0.0"],
        input="y\n",
    )
    assert accepted.exit_code == 0
    assert f"enriched {count}" in accepted.stdout


def test_faculty_enrich_career_below_threshold_never_prompts(tmp_path, monkeypatch):
    """No confirmation should interrupt a normal, small run."""
    from gradpath.db import connect, migrate

    workspace = _workspace(tmp_path)
    runner.invoke(app, ["--root", str(workspace), "init"])
    conn = connect(workspace / "gradpath.db")
    migrate(conn)
    conn.execute("INSERT INTO institutions (id, name, added_at) VALUES ('i', 'I', '2026-01-01')")
    with conn:
        _insert_person_with_match(conn, 1, 0.9)
    conn.commit()

    monkeypatch.setattr(
        "gradpath.cli.enrich_person_career", lambda conn, client, person: True
    )
    # No input supplied at all -- if this prompted for confirmation, Click
    # would raise on end-of-stdin rather than returning cleanly.
    result = runner.invoke(
        app, ["--root", str(workspace), "faculty", "enrich-career", "--min-score", "0.0"],
    )
    assert result.exit_code == 0
    assert "enriched 1" in result.stdout


# --- first-run/misconfiguration errors must be clean, not a traceback ---


def test_running_a_command_before_init_gives_a_clean_error(tmp_path):
    (tmp_path / "settings.yaml").write_text(SETTINGS_YAML)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "fields.yaml").write_text(FIELDS_YAML)
    # Deliberately no `init` call, so profile.yaml does not exist.
    result = runner.invoke(app, ["--root", str(tmp_path), "show"])
    assert result.exit_code != 0
    assert result.exception is None or not isinstance(result.exception, FileNotFoundError)
    assert "traceback" not in result.output.lower()
    assert "init" in result.output.lower()


def test_malformed_profile_gives_a_clean_error_not_a_traceback(tmp_path):
    workspace = _workspace(tmp_path)
    (workspace / "profile.yaml").write_text("name: default\ncontact_email: \"\"\n")
    result = runner.invoke(app, ["--root", str(workspace), "show"])
    assert result.exit_code != 0
    assert result.exception is None or not isinstance(result.exception, ValueError)
    assert "traceback" not in result.output.lower()
    assert "profile.yaml" in result.output.lower()
