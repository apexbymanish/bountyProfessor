"""Budget-gated LLM rerank — an optional, opt-in stage 2 over a score band.

Stage 1 scores everyone locally and for free. This module adds an optional
pass that asks Claude to re-score and explain the people already at or above
`min_score`. It must never be required: with no `ANTHROPIC_API_KEY` set,
`rerank` refuses cleanly and stage 1 results remain fully usable. Nothing is
spent without the caller's confirmation (`ask`), and `budget_usd` is a real
ceiling enforced against each call's *measured* token usage as the run
proceeds — not just a head-count predicted up front from a static estimate
(ruling R32). The pre-flight estimate in `plan_rerank` still exists, and
still drives the confirmation prompt — that is what an estimate is for.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from gradpath.util import now_iso

# Public list prices for claude-opus-5, used only to warn before spending —
# never to bill. Input and output are priced very differently ($5/$25 per
# MTok), and a rerank's input (a profile plus a few abstracts) is far larger
# than its output (a score and a sentence), so estimating each side
# separately avoids badly misjudging the total. These are list prices and
# can drift; treat them as a planning estimate, not a bill.
INPUT_COST_PER_MTOK_USD = 5.00
OUTPUT_COST_PER_MTOK_USD = 25.00

TOKENS_PER_PERSON_INPUT_ESTIMATE = 1_600
TOKENS_PER_PERSON_OUTPUT_ESTIMATE = 200
TOP_WORKS_PER_PERSON = 3


_MISSING_SDK_MESSAGE = (
    "The 'anthropic' package is not installed. Install it with "
    "`pip install 'gradpath[rerank]'` to use --rerank. "
    "Your stage 1 results are unaffected and remain fully usable."
)


class MissingApiKey(RuntimeError):
    """Raised when --rerank is requested but ANTHROPIC_API_KEY is not set,
    or the 'anthropic' package isn't installed."""


class RerankRefusal(RuntimeError):
    """Raised internally when Claude declines to score a particular person.

    Caught by `rerank`, which skips that person rather than aborting the run.
    """


@dataclass(frozen=True)
class RerankPlan:
    person_count: int
    estimated_tokens: int
    estimated_cost_usd: float


@dataclass(frozen=True)
class RerankScore:
    """What a `score_fn` returns for one person.

    `input_tokens`/`output_tokens` default to 0 so an injected `score_fn`
    (as every test in this suite uses) is never forced to fabricate usage
    it doesn't have — it's simply treated as free. `_call_claude` fills
    these in from the real API response so `rerank` can meter actual spend.
    """

    score: float
    reason: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class RerankOutcome:
    """What one `rerank` run produced: how many people were scored, and what
    it actually cost — metered from real per-call token usage, not the
    pre-flight estimate — so a caller (e.g. the CLI) can report used vs.
    predicted spend."""

    scored: int
    spent_usd: float


def _cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * INPUT_COST_PER_MTOK_USD
        + output_tokens / 1_000_000 * OUTPUT_COST_PER_MTOK_USD
    )


def plan_rerank(
    conn: sqlite3.Connection, profile_id: str, min_score: float, model: str
) -> RerankPlan:
    """Report how many people are above `min_score` and what reranking them would cost.

    This is a pre-flight estimate for the confirmation prompt only. The
    actual run in `rerank` enforces `budget_usd` against measured spend, not
    this prediction — see ruling R32.
    """
    count = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE profile_id = ? AND stage1_score >= ?",
        (profile_id, min_score),
    ).fetchone()[0]
    input_tokens = count * TOKENS_PER_PERSON_INPUT_ESTIMATE
    output_tokens = count * TOKENS_PER_PERSON_OUTPUT_ESTIMATE
    cost = _cost_usd(input_tokens, output_tokens)
    return RerankPlan(count, input_tokens + output_tokens, round(cost, 4))


_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Fit between the researcher's work and the applicant's interests.",
        },
        "reason": {
            "type": "string",
            "description": "One sentence explaining the score.",
        },
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}


def _call_claude(
    person: sqlite3.Row, works: list[sqlite3.Row], profile_text: str, model: str
) -> RerankScore:
    """Ask Claude to score fit. Imported lazily so the package works without the SDK."""
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise MissingApiKey(_MISSING_SDK_MESSAGE) from exc

    evidence = "\n\n".join(
        f"- {w['title']}\n  {(w['abstract'] or '')[:800]}" for w in works
    )
    prompt = (
        "You are helping a prospective graduate student decide who to email.\n\n"
        f"APPLICANT INTERESTS:\n{profile_text}\n\n"
        f"RESEARCHER: {person['name']}\n"
        f"THEIR MOST RELEVANT WORK:\n{evidence}\n\n"
        "Score how well this researcher's work fits the applicant's interests."
    )
    response = Anthropic().messages.create(
        model=model,
        max_tokens=2000,
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": _SCORE_SCHEMA},
        },
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        raise RerankRefusal(f"Claude declined to score {person['name']!r}")

    text_block = next(block for block in response.content if block.type == "text")
    payload = json.loads(text_block.text)
    return RerankScore(
        score=float(payload["score"]),
        reason=str(payload["reason"]),
        input_tokens=int(response.usage.input_tokens),
        output_tokens=int(response.usage.output_tokens),
    )


def rerank(
    conn: sqlite3.Connection,
    profile_id: str,
    min_score: float,
    model: str,
    budget_usd: float,
    ask: Callable[[RerankPlan], bool],
    profile_text: str = "",
    score_fn: Callable[..., RerankScore] = _call_claude,
) -> RerankOutcome:
    """Rerank everyone above `min_score`, after confirmation and within budget.

    Returns a `RerankOutcome` with how many people were actually scored and
    the real accumulated spend. Refuses immediately (before any confirmation
    prompt) if ANTHROPIC_API_KEY is not set, or — when using the default
    `score_fn` — if the 'anthropic' package isn't installed. Stage 1 results
    are unaffected either way. (Ruling R31: both checks must happen before
    `ask` is called — a user who has already said yes to a dollar figure
    should never then learn the tool couldn't have spent it.)

    `budget_usd` is enforced as a real ceiling against measured spend as the
    loop runs (ruling R32), not just a head-count predicted from the
    pre-flight estimate: before every call, if the money already spent plus
    one more person's estimated cost would exceed `budget_usd`, the run
    stops — checked *before* the call that would breach, since checking
    after means the breach already happened.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise MissingApiKey(
            "ANTHROPIC_API_KEY is not set, so --rerank cannot run. "
            "Your stage 1 results are unaffected and remain fully usable."
        )
    # Only probe for the SDK when it will actually be used. A caller supplying
    # their own score_fn (as every test in this suite does) must be able to
    # run with no SDK installed at all — that's the whole point of injection.
    # find_spec (not a trial import) answers "is it installed" without paying
    # the import cost, preserving the laziness required alongside this check.
    if score_fn is _call_claude and importlib.util.find_spec("anthropic") is None:
        raise MissingApiKey(_MISSING_SDK_MESSAGE)

    plan = plan_rerank(conn, profile_id, min_score, model)
    if plan.person_count == 0 or not ask(plan):
        return RerankOutcome(scored=0, spent_usd=0.0)

    per_person_estimate = plan.estimated_cost_usd / plan.person_count
    affordable = (
        int(budget_usd // per_person_estimate) if per_person_estimate > 0 else plan.person_count
    )
    if affordable <= 0:
        return RerankOutcome(scored=0, spent_usd=0.0)

    rows = conn.execute(
        """
        SELECT m.person_id, m.top_work_ids, p.name, p.id AS id
        FROM matches m JOIN people p ON p.id = m.person_id
        WHERE m.profile_id = ? AND m.stage1_score >= ?
        ORDER BY m.stage1_score DESC LIMIT ?
        """,
        (profile_id, min_score, affordable),
    ).fetchall()

    scored = 0
    spent_usd = 0.0
    for row in rows:
        work_ids = json.loads(row["top_work_ids"] or "[]")[:TOP_WORKS_PER_PERSON]
        if not work_ids:
            continue
        # R32: the real ceiling. `affordable` above only bounds the candidate
        # pool from the pre-flight estimate; this is the check that actually
        # protects budget_usd, against money spent so far plus one more
        # person's estimated cost — not the fixed head-count.
        if spent_usd + per_person_estimate > budget_usd:
            break
        placeholders = ",".join("?" for _ in work_ids)
        works = conn.execute(
            f"SELECT title, abstract FROM works WHERE id IN ({placeholders})", work_ids
        ).fetchall()
        try:
            result = score_fn(row, works, profile_text, model)
        except RerankRefusal:
            continue
        spent_usd += _cost_usd(result.input_tokens, result.output_tokens)
        with conn:
            conn.execute(
                "UPDATE matches SET stage2_score = ?, reason = ?, computed_at = ? "
                "WHERE profile_id = ? AND person_id = ?",
                (result.score, result.reason, now_iso(), profile_id, row["person_id"]),
            )
        scored += 1
    return RerankOutcome(scored=scored, spent_usd=round(spent_usd, 6))
