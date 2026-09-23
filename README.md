# gradpath

Find the professors worth emailing for graduate study.

Ranking sites tell you which universities are good. Nothing tells you **which specific people**
are working on what you want to work on — which is the question that actually decides where you
apply and what you write in the email. `gradpath` answers that one.

You describe a research field and the countries you care about. It pulls every researcher
publishing in that area from OpenAlex, scores each one against your interests, estimates who is
actually faculty rather than a PhD student, resolves contact addresses from authoritative
sources, and gives you a ranked shortlist with the evidence behind every row.

> **Status: feature-complete.** The library and the `gradpath` CLI are built and tested (200+
> tests), including the seeded institution registry and tier-scoped discovery. See
> [Current state](#current-state) for what remains genuinely unverified.

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

**Email addresses are never guessed, and never borrowed.** The resolution chain tries Crossref,
ORCID, a per-institution adapter, then a polite faculty-page crawler. If all four fail the
answer is *no address* plus a homepage link — never a pattern-inferred `firstname.lastname@`. A
guessed address bounces, and bounce rate damages every later email you send from that account.

The adapter step matches a directory row to a person by full name (case, accents, punctuation,
honorifics and name order folded away), and takes an address **only from that row**. If no row
matches, or two rows do, the answer is no address — never the next Kim on the page, and never
whatever address happened to appear first. Which source an address came from is shown in the
`src` column and exported as `email_source`, because a Crossref address the author published
themselves and one scraped off a page are not equally trustworthy.

**It is a good citizen.** One module makes every network call, and it enforces `robots.txt`,
per-host rate limiting, caching, retry with backoff, and a User-Agent carrying your contact
address. No LinkedIn, no headless browsers, no authenticated scraping.

## How the scoring works

**Fit — mean of a researcher's top 3 most similar works.** Averaging everything regresses
prolific people to the mean and buries the professor with three papers precisely on your topic
under two hundred unrelated ones. A single maximum is the opposite failure: one coincidentally
-worded abstract promotes someone with no real overlap. Mean-of-top-3 rewards sustained overlap
while staying robust to a lucky match. Every score records which three works produced it.

**What you are matched against.** Your `interests` paragraph and `keywords` are embedded, and so
are the abstracts of every DOI in `seed_papers` and `my_papers` — resolved through OpenAlex when
`match` runs, and weighted double, because a real abstract describes a research area far more
precisely than a paragraph you wrote about yourself. `match` prints how many resolved and names
every DOI that did not, so a seed paper you believe is steering the ranking never silently
isn't. A profile whose only signal is `seed_papers` is enough on its own.

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
reporting/export, the seeded institution registry (OpenAlex-sourced tiers and cited ranking-CSV
imports, with domain-based resolution onto curated rows), tier-scoped discovery, and the
`gradpath` CLI that wires all of it into one command.

Not yet built: nothing in the original plan. What remains is verification, not features (see
the known limitation below), and any ranking snapshot a user wants to import stays theirs to
supply and cite — `gradpath` intentionally ships none.

Verified against live OpenAlex: a one-page discovery run returned 200 works, 1,039 researchers
across 299 institutions, embedded and scored end to end in under two seconds.

Known limitation: `people.homepage` is only ever filled in by the institution-adapter step, so
the fourth source in the chain — the polite faculty-page crawler — is reachable only for people
at an institution that has an adapter and whose directory row was matched unambiguously. For
everyone else the chain effectively ends at ORCID. OpenAlex carries no homepage for an author,
so closing that gap needs another source, not another line of code here.

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
# 1. Create a complete workspace in --root (default: the current directory):
#    the database, settings.yaml, data/fields.yaml, data/institutions.yaml and
#    profile.yaml, then seed the known institutions (with their adapters) from
#    data/institutions.yaml. Existing files are never overwritten, so re-running
#    init is safe. Fill in profile.yaml's contact_email and interests before
#    continuing — every other command needs contact_email.
gradpath init

# 2. Not sure what OpenAlex calls your field? Search for its topic ids.
gradpath fields search "efficient machine learning"

# 3. Discover researchers publishing in a field, in the countries you care about.
#     A run is resumable: the OpenAlex cursor is saved per query scope (topics +
#     institutions-or-countries + since), so an interrupted crawl picks up where it
#     stopped, and differently scoped runs never share a resume point. Once a scope
#     has been crawled to the end it stays finished and reports that rather than
#     silently returning 0 works; --restart clears that scope's cursor and re-crawls
#     it, which is how you pick up newly published work.
gradpath discover --field efficient-ml --country KR --since 2018
gradpath discover --field efficient-ml --country KR --since 2018 --restart

# 3b. Or scope discovery to a fixed institution list instead of a whole country
#     (see "Scoping a run to a tier" below for how korea-20/--institution get set up).
gradpath discover --field efficient-ml --tier korea-20
gradpath discover --field efficient-ml --institution kaist --institution gist

# 4. Score everyone against your interests (fit). No top-N — everyone found is scored.
#    Any seed_papers/my_papers DOIs in profile.yaml are resolved against OpenAlex
#    here and weighted double in the profile vector; the count that resolved is
#    printed and any DOI that did not is named, never silently dropped.
gradpath match

# 5. Fetch whole-career OpenAlex data for the top-ranked slice, so faculty
#    scoring sees a person's real career span rather than just this query's
#    window (see "How the scoring works" below). Runs one call per person, so
#    it's bounded by --min-score, which defaults to settings.show_min_score; a
#    run over more than 200 people reports its size and asks first. There is
#    deliberately no --faculty-only here: it would filter on the very
#    span-truncated faculty scores this step exists to correct, and at this
#    position in the pipeline `faculty score` has not run at all yet.
gradpath faculty enrich-career --min-score 0.6

# 6. Estimate who is faculty vs. a student/postdoc. Nobody is removed —
#    this only fills a sortable column, and reports how many people were
#    scored on real career data vs. the discovered slice.
gradpath faculty score

# 7. Resolve contact addresses for people who rank, so crawling effort
#    follows the ranking (Crossref -> ORCID -> institution adapter -> crawler).
#    Up to ~8 network calls per person, so --min-score defaults to
#    settings.show_min_score and a run over more than 25 people reports the
#    count and the implied wall-clock time and asks before starting.
gradpath emails resolve --min-score 0.7

# 7b. Who still has no address, and the homepage to try by hand. An adapter
#     that found a person's directory row but no address stores that row's
#     homepage, so this list is a usable manual queue rather than a wall of
#     "(no homepage)".
gradpath emails report

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

## Scoping a run to a tier

Discovery is field-first by design — institutions are *discovered from the results*, which is
how it surfaces the excellent lab nobody told you about. But a real applicant also wants to say
"only places I could plausibly attend." Tiers provide that without reintroducing a cap on who
gets scored: they restrict *which institutions get queried*, not how many people from them get
ranked.

Ranks are never hand-typed into this repository — they change every year and the major
rankings disagree. There are exactly two sourced ways to build a tier, and each needs its own
resolution step before `discover --tier` can use it, because each leaves OpenAlex ids unbound
in a different way:

**Korea top 20 (or any country) — live OpenAlex output, one command:**

```bash
gradpath institutions top --country KR --limit 20 --tier korea-20
```

`institutions top` seeds *and* resolves in the same step: a curated row already seeded from
`data/institutions.yaml` (with its adapter) is matched by homepage domain and gets its
OpenAlex id filled in, rather than being duplicated under a name-derived slug. No further step
is needed before `discover --tier korea-20`.

**World top 100 (or any QS/THE/ARWU snapshot) — three steps:**

```bash
# 1. Import a snapshot you supply and can cite (see data/rankings/README.md — snapshots
#    are never committed to this repo). OpenAlex ids are left unresolved: a CSV row has no
#    homepage, so institutions top's domain matching can never reach it.
gradpath institutions import --csv data/rankings/qs2026.csv --tier world-100 --top 100

# 2. Resolve those rows separately, by an exact normalised name match against OpenAlex
#    (case/punctuation/diacritics/leading "The" folded away) -- never a fuzzy match, since a
#    wrong bind would silently attach one institution's rank to a different one. A near-match
#    or a tie is left unresolved and printed for you to check by hand.
gradpath institutions resolve --tier world-100

# 3. Now the tier has resolved OpenAlex ids and discover can use it.
gradpath discover --field efficient-ml --tier world-100
```

`--institution` accepts either a seeded slug (e.g. `kaist`, resolved via the `institutions`
table) or a raw OpenAlex institution id, passed through unchanged. A slug with no resolved
OpenAlex id yet fails with a message telling you to run `institutions top` or
`institutions resolve` first, rather than silently discovering nothing for it.

**Limitation: an institution belongs to one tier at a time.** The schema stores a single
`tier`/`rank` pair per institution row, not a many-to-many mapping. If the same real
institution ends up in two tiers you've built — KAIST is both `korea-20`-seeded and present in
a `world-100` CSV you imported, say — the second row resolves to the same OpenAlex id as the
first. `institutions resolve` does not silently merge them (that would discard whichever
tier/rank you imported second) or silently skip the row (indistinguishable from success); it
reports the overlap as a conflict, names both institutions, and leaves the row unresolved for
you to decide by hand. Supporting one institution in multiple tiers for real would need a
many-to-many `institution_tiers` table — that's a deliberate non-goal right now, not an
oversight. See `data/rankings/README.md` for more.

## License

MIT
