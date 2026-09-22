import json
import sys
import types

import pytest

from gradpath.db import connect, migrate
from gradpath.match.rerank import (
    MissingApiKey,
    RerankRefusal,
    _call_claude,
    plan_rerank,
    rerank,
)


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


# --- _call_claude: verifying the corrected content-block / structured-output /
# refusal handling without ever making a real network call. A fake `anthropic`
# module is injected into sys.modules so the lazy `from anthropic import
# Anthropic` inside _call_claude resolves to our stub instead of the real SDK
# (which is not installed here — that's the point of the lazy import).


class _FakeBlock:
    def __init__(self, type_, text=None):
        self.type = type_
        self.text = text


class _FakeResponse:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeAnthropicClient:
    last_instance = None

    def __init__(self, response):
        self.messages = _FakeMessages(response)
        _FakeAnthropicClient.last_instance = self


def _install_fake_anthropic(monkeypatch, response):
    fake_module = types.ModuleType("anthropic")
    fake_module.Anthropic = lambda: _FakeAnthropicClient(response)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)


def test_call_claude_skips_leading_thinking_block_to_find_text(monkeypatch):
    # Thinking is on by default on claude-opus-5, so content[0] is a thinking
    # block with no usable .text — response.content[0].text would raise or
    # return None. _call_claude must scan for the block whose .type == "text".
    response = _FakeResponse(content=[
        _FakeBlock("thinking", text=None),
        _FakeBlock("text", text=json.dumps({"score": 0.75, "reason": "good fit"})),
    ])
    _install_fake_anthropic(monkeypatch, response)

    person = {"name": "Dr. Example"}
    score, reason = _call_claude(person, works=[], profile_text="ML", model="claude-opus-5")

    assert score == pytest.approx(0.75)
    assert reason == "good fit"


def test_call_claude_uses_structured_output_config_not_prompt_and_parse(monkeypatch):
    response = _FakeResponse(content=[
        _FakeBlock("text", text=json.dumps({"score": 0.5, "reason": "ok"})),
    ])
    _install_fake_anthropic(monkeypatch, response)

    _call_claude({"name": "Dr. Example"}, works=[], profile_text="ML", model="claude-opus-5")

    kwargs = _FakeAnthropicClient.last_instance.messages.last_kwargs
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["max_tokens"] >= 2000
    assert kwargs["output_config"]["effort"] == "low"
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    schema = kwargs["output_config"]["format"]["schema"]
    assert schema["properties"]["score"]["type"] == "number"
    assert schema["properties"]["reason"]["type"] == "string"


def test_call_claude_treats_refusal_as_no_score_not_a_crash(monkeypatch):
    response = _FakeResponse(content=[], stop_reason="refusal")
    _install_fake_anthropic(monkeypatch, response)

    with pytest.raises(RerankRefusal):
        _call_claude({"name": "Dr. Example"}, works=[], profile_text="ML", model="claude-opus-5")


def test_rerank_skips_person_on_refusal_without_aborting_the_run(db, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def flaky_score(person, works, profile_text, model):
        if person["id"] % 2 == 0:
            raise RerankRefusal("declined")
        return 0.6, "fine"

    scored = rerank(db, "default", 0.0, "claude-opus-5", budget_usd=100.0,
                    ask=lambda _: True, score_fn=flaky_score)
    assert scored == 5  # the 5 odd-indexed people; refusals are skipped, not fatal
