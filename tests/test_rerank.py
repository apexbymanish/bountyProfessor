import importlib.machinery
import importlib.util
import json
import sys
import types

import pytest

from gradpath.db import connect, migrate
from gradpath.match.rerank import (
    INPUT_COST_PER_MTOK_USD,
    MissingApiKey,
    RerankRefusal,
    RerankScore,
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


def test_rerank_checks_for_the_sdk_before_asking_for_confirmation(db, monkeypatch):
    # R31: a user must never say "yes, spend $X" and only then learn the tool
    # can't spend anything. The SDK-presence check must run — and fail —
    # before `ask` is ever invoked, using the default score_fn (_call_claude),
    # which is the only path that actually needs the SDK. find_spec is forced
    # to report "not installed" here regardless of the real environment, so
    # the test pins the ordering rather than depending on whether `anthropic`
    # happens to be installed on the machine running it.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "anthropic":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    asked: list[object] = []

    def counting_ask(plan):
        asked.append(plan)
        return True

    with pytest.raises(MissingApiKey, match="not installed"):
        rerank(db, "default", 0.7, "claude-opus-5", budget_usd=5.0, ask=counting_ask)

    assert asked == []


def test_rerank_with_injected_score_fn_does_not_require_the_sdk(db, monkeypatch):
    # The SDK-presence probe must only fire on the default _call_claude path.
    # A caller supplying their own score_fn (as every other test here does)
    # must still be able to run with no SDK installed — that's the point of
    # injection, and real `anthropic` genuinely isn't installed in this venv.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert importlib.util.find_spec("anthropic") is None  # sanity: truly absent

    def fake_score(person, works, profile_text, model):
        return RerankScore(0.4, "fine")

    result = rerank(db, "default", 0.7, "claude-opus-5", budget_usd=5.0,
                    ask=lambda _: True, score_fn=fake_score)
    assert result.scored == 3


def test_rerank_aborts_when_the_user_declines(db, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def unreachable_score(person, works, profile_text, model):
        raise AssertionError("score_fn must not be called when the user declines")

    result = rerank(db, "default", 0.7, "claude-opus-5", 5.0, ask=lambda _: False,
                    score_fn=unreachable_score)
    assert result.scored == 0
    assert result.spent_usd == 0.0
    assert db.execute("SELECT COUNT(*) FROM matches WHERE stage2_score IS NOT NULL")\
             .fetchone()[0] == 0


def test_rerank_stops_at_the_budget_ceiling(db, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls: list[int] = []

    def fake_score(person, works, profile_text, model):
        calls.append(person["id"])
        return RerankScore(0.5, "because")

    result = rerank(db, "default", 0.0, "claude-opus-5", budget_usd=0.0,
                    ask=lambda _: True, score_fn=fake_score)
    assert result.scored == 0
    assert result.spent_usd == 0.0
    assert calls == []


def test_rerank_writes_score_and_reason(db, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_score(person, works, profile_text, model):
        return RerankScore(0.91, "three recent papers directly on this topic")

    result = rerank(db, "default", 0.7, "claude-opus-5", budget_usd=10.0,
                    ask=lambda _: True, score_fn=fake_score)
    assert result.scored == 3
    row = db.execute(
        "SELECT stage2_score, reason FROM matches WHERE stage2_score IS NOT NULL LIMIT 1"
    ).fetchone()
    assert row["stage2_score"] == pytest.approx(0.91)
    assert "three recent papers" in row["reason"]


# --- R32: budget_usd is a real ceiling enforced against measured per-call
# spend, not just a head-count predicted from the pre-flight estimate.


def test_rerank_cumulative_spend_stops_before_exceeding_budget(db, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls: list[int] = []

    # Each call reports real usage costing exactly $2.00 — far more than the
    # ~$0.013/person pre-flight estimate. If the ceiling were still just a
    # head-count from that estimate (budget // estimate), it would allow far
    # more than the 10 available candidates and every one of them would run.
    # Enforcing against measured cumulative spend must stop well short.
    real_cost_per_call = 2.0
    input_tokens_for_that_cost = int(
        real_cost_per_call / INPUT_COST_PER_MTOK_USD * 1_000_000
    )

    def expensive_score(person, works, profile_text, model):
        calls.append(person["id"])
        return RerankScore(0.5, "because",
                           input_tokens=input_tokens_for_that_cost, output_tokens=0)

    result = rerank(db, "default", 0.0, "claude-opus-5", budget_usd=5.0,
                    ask=lambda _: True, score_fn=expensive_score)

    # spent=0 -> call1 (spent=2.0) -> call2 (spent=4.0) -> call3 (spent=6.0,
    # since 4.0 + tiny estimate <= 5.0 still passes the pre-call check) ->
    # before call4: 6.0 + estimate > 5.0 -> stop.
    assert len(calls) == 3
    assert result.scored == 3
    assert result.spent_usd == pytest.approx(6.0)


def test_rerank_score_fn_with_no_usage_reported_runs_to_completion(db, monkeypatch):
    # A score_fn that doesn't report usage (RerankScore's input/output token
    # defaults) must not error and must be treated as zero-cost — injected
    # test doubles must never be forced to fabricate token counts.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def zero_usage_score(person, works, profile_text, model):
        return RerankScore(0.6, "fine")  # no input_tokens/output_tokens given

    result = rerank(db, "default", 0.7, "claude-opus-5", budget_usd=10.0,
                    ask=lambda _: True, score_fn=zero_usage_score)
    assert result.scored == 3
    assert result.spent_usd == pytest.approx(0.0)


# --- _call_claude: verifying the corrected content-block / structured-output /
# refusal handling without ever making a real network call. A fake `anthropic`
# module is injected into sys.modules so the lazy `from anthropic import
# Anthropic` inside _call_claude resolves to our stub instead of the real SDK
# (which is not installed here — that's the point of the lazy import).


class _FakeBlock:
    def __init__(self, type_, text=None):
        self.type = type_
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens=1000, output_tokens=100):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, content, stop_reason="end_turn", usage=None):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage if usage is not None else _FakeUsage()


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
    # importlib.util.find_spec checks sys.modules first, and raises ValueError
    # if the cached module has no __spec__ — give it one so rerank()'s R31
    # SDK-presence probe correctly reports "installed" for this fake module,
    # exactly as it would for the real package.
    fake_module.__spec__ = importlib.machinery.ModuleSpec("anthropic", loader=None)
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
    result = _call_claude(person, works=[], profile_text="ML", model="claude-opus-5")

    assert result.score == pytest.approx(0.75)
    assert result.reason == "good fit"


def test_call_claude_reports_real_token_usage(monkeypatch):
    response = _FakeResponse(
        content=[_FakeBlock("text", text=json.dumps({"score": 0.5, "reason": "ok"}))],
        usage=_FakeUsage(input_tokens=1234, output_tokens=56),
    )
    _install_fake_anthropic(monkeypatch, response)

    result = _call_claude({"name": "Dr. Example"}, works=[], profile_text="ML",
                          model="claude-opus-5")

    assert result.input_tokens == 1234
    assert result.output_tokens == 56


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
        return RerankScore(0.6, "fine")

    result = rerank(db, "default", 0.0, "claude-opus-5", budget_usd=100.0,
                    ask=lambda _: True, score_fn=flaky_score)
    assert result.scored == 5  # the 5 odd-indexed people; refusals are skipped, not fatal


def test_rerank_end_to_end_through_default_call_claude_with_faked_sdk(db, monkeypatch):
    # The seam between rerank()'s loop and _call_claude's internals is
    # otherwise untested: loop logic is covered via injected score_fn, and
    # _call_claude's internals are covered via direct unit tests above, but
    # nothing drives rerank() through the *default* _call_claude path. This
    # is also where the R32 usage-reporting contract joins the two: a real
    # RerankScore built from a faked API response's usage must flow through
    # rerank()'s accumulation and into the written row and RerankOutcome.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    response = _FakeResponse(
        content=[_FakeBlock("text", text=json.dumps({"score": 0.8, "reason": "close fit"}))],
        usage=_FakeUsage(input_tokens=1500, output_tokens=150),
    )
    _install_fake_anthropic(monkeypatch, response)

    result = rerank(db, "default", 0.7, "claude-opus-5", budget_usd=10.0, ask=lambda _: True)

    assert result.scored == 3
    assert result.spent_usd > 0
    row = db.execute(
        "SELECT stage2_score, reason FROM matches WHERE stage2_score IS NOT NULL LIMIT 1"
    ).fetchone()
    assert row["stage2_score"] == pytest.approx(0.8)
    assert row["reason"] == "close fit"
