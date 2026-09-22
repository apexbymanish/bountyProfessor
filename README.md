# gradpath

Find the professors worth emailing for graduate study.

Ranking sites tell you which universities are good. Nothing tells you **which specific people**
are working on what you want to work on — which is the question that actually decides where you
apply and what you write in the email. `gradpath` answers that one.

You describe a research field and the countries you care about. It pulls every researcher
publishing in that area from OpenAlex, scores each one against your interests, estimates who is
actually faculty rather than a PhD student, resolves contact addresses from authoritative
sources, and gives you a ranked shortlist with the evidence behind every row.

> **Status: in progress.** The library and the `gradpath` CLI are built and tested (150+
> tests). The one thing not yet built is institution-tier seeding (scoping a run to, say,
> Korea's top 20 or the world top 100) — until then, `discover` is field-and-country-first.
> See [Current state](#current-state).

## What makes it different

**The field decides who surfaces, not a row count.** There is no top-N anywhere in the
pipeline. Every researcher found is scored and stored, so a strong match at rank 180 — at a
university you had never thought to list — still reaches you. `--limit` exists only to keep a
terminal readable.

**Discovery is field-first.** You search by research topic and country; institutions are
*discovered from the results* rather than named in advance. That is what surfaces the excellent
lab at a school nobody told you about.

**Fit and seniority are scored separately and never blended.** "Great match, probably a PhD
student" and "great match, full professor" call for completely different actions, so they stay
in separate columns. Nobody is ever filtered out by default — a heuristic that silently hides
the right professor is the worst failure this tool could have.

**Email addresses are never guessed.** The resolution chain tries Crossref, ORCID, a per-institution
adapter, then a polite faculty-page crawler. If all four fail the answer is *no address* plus a
homepage link — never a pattern-inferred `firstname.lastname@`. A guessed address bounces, and
bounce rate damages every later email you send from that account.

**It is a good citizen.** One module makes every network call, and it enforces `robots.txt`,
per-host rate limiting, caching, retry with backoff, and a User-Agent carrying your contact
address. No LinkedIn, no headless browsers, no authenticated scraping.

## How the scoring works

**Fit — mean of a researcher's top 3 most similar works.** Averaging everything regresses
prolific people to the mean and buries the professor with three papers precisely on your topic
under two hundred unrelated ones. A single maximum is the opposite failure: one coincidentally
-worded abstract promotes someone with no real overlap. Mean-of-top-3 rewards sustained overlap
while staying robust to a lucky match. Every score records which three works produced it.

**Seniority — a faculty-likelihood estimate.** OpenAlex does not label who is faculty, and most
authors at a university are students and postdocs. Directory confirmation and ORCID employment
titles are treated as facts and win outright; heuristics over career span, publication volume
and author position fill the gap. Author-position signals are downweighted in
alphabetical-authorship fields such as mathematics and economics, where last-author carries no
seniority meaning. Career span is read from a researcher's *whole* OpenAlex record, not the
slice your query matched — otherwise a topic-and-year-filtered search makes every thirty-year
professor look like a first-year student.

## Current state

Working and tested: the SQLite schema and migrations, config and profile loading, the polite
HTTP layer, OpenAlex parsing and ingest (resumable, crash-safe), fit scoring, faculty-likelihood
scoring, the persisted embedding cache and uncapped matching, Crossref/ORCID/career enrichment,
the email resolution chain, three institution adapters, budget-gated LLM reranking,
reporting/export, and the `gradpath` CLI that wires all of it into one command.

Not yet built: institution-tier seeding, which will let `discover` scope a run to a named
tier (say, Korea's top 20 or the world top 100) instead of just a field and a country list.

Verified against live OpenAlex: a one-page discovery run returned 200 works, 1,039 researchers
across 299 institutions, embedded and scored end to end in under two seconds.

Known limitation: the institution adapters' directory URLs and CSS selectors are plausible but
have **not** been verified against the live sites, and those sites change. The shared adapter
contract test is designed to make fixing that cheap.

## Design notes

Rankings are never hand-written into this repository. They change every year, the major
rankings disagree, and a typed list would be stale on arrival. Institution ordering comes from
live OpenAlex research output, or from a ranking snapshot you supply and can cite.

Requires no API key for anything except the optional LLM rerank.

## Installation

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Optional, for the LLM rerank step only:

```bash
pip install -e ".[rerank]"
export ANTHROPIC_API_KEY=...
```

## Usage

`gradpath` needs a workspace: a `settings.yaml`, a `profile.yaml` (gitignored — it's yours),
and `data/fields.yaml`/`data/institutions.yaml`. `--root` points every command at that
workspace; omit it to use the current directory. You can also run it as `./gradpath.sh ...`
from a checkout without installing.

**No command below except the last one needs `ANTHROPIC_API_KEY`.** Everything up through
`export` — discovery, fit scoring, faculty-likelihood scoring, email resolution, the ranked
table and CSV/Markdown export — runs with zero API keys. Only `match --rerank` (the optional
second-pass LLM scoring) requires a key, and it refuses cleanly and tells you so if one isn't
set; your stage 1 results are unaffected either way.

```bash
# 1. Create the database, scaffold profile.yaml, and seed known institutions
#    (with their adapters) from data/institutions.yaml. Fill in profile.yaml's
#    contact_email and interests before continuing.
gradpath init

# 2. Not sure what OpenAlex calls your field? Search for its topic ids.
gradpath fields search "efficient machine learning"

# 3. Discover researchers publishing in a field, in the countries you care about.
gradpath discover --field efficient-ml --country KR --since 2018

# 4. Score everyone against your interests (fit). No top-N — everyone found is scored.
gradpath match

# 5. Fetch whole-career OpenAlex data for the top-ranked slice, so faculty
#    scoring sees a person's real career span rather than just this query's
#    window (see "How the scoring works" below). Runs one call per person,
#    so it's bounded by --min-score/--faculty-only, same as emails resolve.
gradpath faculty enrich-career --min-score 0.6

# 6. Estimate who is faculty vs. a student/postdoc. Nobody is removed —
#    this only fills a sortable column, and reports how many people were
#    scored on real career data vs. the discovered slice.
gradpath faculty score

# 7. Resolve contact addresses for people who rank, so crawling effort
#    follows the ranking (Crossref -> ORCID -> institution adapter -> crawler).
gradpath emails resolve --min-score 0.7

# 8. Look at the ranked shortlist.
gradpath show --min-score 0.7 --faculty-only

# 9. Export it.
gradpath export --csv targets.csv

# 10. Optional: ask Claude to re-score and explain the fit for the people
#     already at or above --min-score. Needs ANTHROPIC_API_KEY. --budget is
#     an approximate ceiling, not a hard cap — a call's real cost is only
#     known after making it, so a run can finish up to one call's cost above
#     the figure you pass; the actual measured spend is printed at the end.
export ANTHROPIC_API_KEY=...
gradpath match --rerank --min-score 0.7 --budget 2.00
```

## License

MIT
