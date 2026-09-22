# gradpath — Design

**Date:** 2026-09-22
**Status:** Approved
**Slice:** Professor discovery + interest matching (slice 1 of a larger pipeline)

## Overview

`gradpath` helps a prospective graduate student find the professors worth
emailing. Given a set of target universities and a description of the
applicant's research interests, it pulls every recent-publishing researcher at
those institutions from OpenAlex, resolves contact addresses from authoritative
sources, scores each researcher against the applicant's interests, and emits a
ranked shortlist.

The product insight: the bottleneck in graduate applications is not finding
universities, it is finding the *specific people* whose current work overlaps
yours. Rankings sites answer the first question. Nothing answers the second.

## Goals

- Turn "I want to study X in Korea" into a ranked, contactable shortlist of
  20-30 named professors with evidence for why each one fits.
- Work fully offline after data collection, with no paid API key required.
- Be publishable as a public repository from the first commit.
- Be polite enough to run against real university infrastructure without
  causing harm or getting blocked.

## Non-goals (this slice)

- CV and SOP generation
- Scholarship eligibility rules engine
- Tuition and fee database
- Sending email
- Web dashboard
- Any use of LinkedIn

These are later modules against the same SQLite spine. The schema anticipates
them; the code does not implement them.

## Architecture

Layered, with a single network chokepoint.

```
cli.py            command surface (typer)
  |
  +-- sources/    data acquisition (openalex, crossref, orcid, crawler, adapters)
  +-- match/      scoring (embed, rerank, score)
  +-- report/     presentation (table, export)
  |
  +-- db.py       SQLite persistence
  +-- net/http.py THE ONLY module permitted to make network calls
```

`net/http.py` being the sole network path is a load-bearing constraint, not a
convention. Rate limiting, `robots.txt` compliance, caching, retry and
User-Agent policy are enforced there, so no source module can accidentally
violate them. A test asserts that no module outside `net/` imports `httpx` or
`requests`.

### Repository layout

```
gradpath/
  pyproject.toml
  README.md
  LICENSE                     MIT
  .gitignore                  profile.yaml, *.db, .cache/
  gradpath.sh                 convenience entrypoint
  gradpath/
    __init__.py
    cli.py
    config.py
    db.py
    models.py
    net/
      __init__.py
      http.py
    sources/
      __init__.py
      openalex.py
      crossref.py
      orcid.py
      crawler.py
      adapters/
        __init__.py           registry
        base.py               SchoolAdapter ABC
        kaist.py
        gist.py
        snu.py
    match/
      __init__.py
      embed.py
      rerank.py
      score.py
    report/
      __init__.py
      table.py
      export.py
  data/
    schools.yaml              community-editable seed registry
  tests/
    fixtures/                 canned API responses
    cassettes/                vcrpy recordings
  docs/
```

## Data model

SQLite, created by `db.py` with forward-only numbered migrations.

```sql
CREATE TABLE schools (
    id              TEXT PRIMARY KEY,      -- slug, e.g. 'kaist'
    name            TEXT NOT NULL,
    ror_id          TEXT,
    openalex_id     TEXT UNIQUE,
    country         TEXT,
    site            TEXT,
    adapter         TEXT,                  -- adapter module name, nullable
    added_at        TEXT NOT NULL
);

CREATE TABLE people (
    id                  INTEGER PRIMARY KEY,
    school_id           TEXT NOT NULL REFERENCES schools(id),
    dept                TEXT,
    name                TEXT NOT NULL,
    openalex_author_id  TEXT UNIQUE,
    orcid               TEXT,
    title               TEXT,
    homepage            TEXT,
    email               TEXT,
    email_confidence    TEXT,              -- high | medium | none
    email_source        TEXT,              -- crossref | orcid | adapter | crawler
    works_count         INTEGER DEFAULT 0,
    last_seen_at        TEXT
);

CREATE TABLE works (
    id                  INTEGER PRIMARY KEY,
    openalex_work_id    TEXT UNIQUE NOT NULL,
    title               TEXT NOT NULL,
    abstract            TEXT,              -- reconstructed; NULL when unavailable
    year                INTEGER,
    doi                 TEXT,
    venue               TEXT,
    cited_by            INTEGER DEFAULT 0
);

CREATE TABLE authorships (
    person_id   INTEGER NOT NULL REFERENCES people(id),
    work_id     INTEGER NOT NULL REFERENCES works(id),
    position    TEXT,                      -- first | middle | last
    PRIMARY KEY (person_id, work_id)
);

CREATE TABLE matches (
    profile_id      TEXT NOT NULL,
    person_id       INTEGER NOT NULL REFERENCES people(id),
    stage1_score    REAL NOT NULL,
    stage2_score    REAL,
    reason          TEXT,
    top_work_ids    TEXT,                  -- JSON array, evidence for the score
    computed_at     TEXT NOT NULL,
    PRIMARY KEY (profile_id, person_id)
);

CREATE TABLE cursors (
    key         TEXT PRIMARY KEY,          -- e.g. 'discover:kaist:cs'
    cursor      TEXT,
    updated_at  TEXT
);

CREATE TABLE fetch_log (
    url         TEXT PRIMARY KEY,
    fetched_at  TEXT NOT NULL,
    status      INTEGER,
    etag        TEXT
);
```

`cursors` is what makes long collection runs resumable after Ctrl-C. `works`
and `people` are cached indefinitely and refreshed only when explicitly asked,
because re-pulling thousands of abstracts on every run is both slow and rude.

## Configuration

Three files.

`data/schools.yaml` is committed and community-editable. It is the seed
registry mapping slugs to institutions so users need not resolve well-known
universities themselves:

```yaml
kaist:
  name: Korea Advanced Institute of Science and Technology
  ror_id: 05apxxy63
  country: KR
  site: https://www.kaist.ac.kr
  adapter: kaist
```

`settings.yaml` is committed and holds non-personal defaults:

```yaml
embedding_model: sentence-transformers/all-MiniLM-L6-v2
rate_limit_per_host: 1.0        # requests per second; may be lowered, not raised
default_since_year: 2021
rerank_model: claude-opus-5
rerank_top_n: 30
cache_dir: .cache
```

`profile.yaml` is gitignored and scaffolded by `gradpath init`:

```yaml
name: default                    # becomes matches.profile_id
contact_email: you@example.com   # used in the polite User-Agent
interests: >
  Free-text paragraph describing what you want to work on.
keywords:
  - edge-ml
  - quantization
seed_papers:                     # strongest signal
  - 10.1145/3458864.3467882
  - arxiv:2103.13630
my_papers: []                    # your own work, if any
```

Multiple profiles are supported by keeping several files and selecting one
with `--profile`; `name` is the key written to `matches.profile_id`, so
rankings for different research directions coexist in one database.

### Building the profile embedding

The three interest inputs are merged into one vector as a weighted mean of
component embeddings:

| Component | Weight |
|---|---|
| `interests` free text | 1.0 |
| `keywords`, joined into a single string | 1.0 |
| Each resolved `seed_papers` / `my_papers` abstract | 2.0 |

Paper abstracts are weighted double because a real abstract describes a
research area far more precisely than a self-written paragraph does. Seed
papers are resolved to abstracts through the same OpenAlex path as any other
work; a DOI that cannot be resolved is reported and skipped rather than
silently dropped. If no component resolves, `match` refuses to run rather than
scoring against an empty vector.

## CLI surface

```
gradpath init                       create db, scaffold profile.yaml
gradpath schools search "KAIST"     resolve a name to OpenAlex / ROR ids
gradpath schools add <slug>...      register targets
gradpath schools list
gradpath discover --school <slug> [--concept <name>] [--since <year>]
gradpath emails resolve [--school <slug>]
gradpath emails report              manual-lookup queue
gradpath match [--top N]            stage 1
gradpath match --rerank [--top N]   stage 2 (opt-in)
gradpath show [--top N]             ranked terminal table
gradpath export --csv <path>
gradpath export --markdown <path>
```

Every command is idempotent. Re-running `discover` resumes from the stored
cursor and skips already-fetched works.

## Pipeline

1. **`schools add`** resolves a human name to an OpenAlex institution ID via
   the ROR registry, stores the slug, and binds a school adapter if one exists
   for that institution.

2. **`discover`** queries OpenAlex
   `/works?filter=institutions.id:<id>,publication_year:>=<since>`, paginating
   with a cursor persisted to the `cursors` table. For each work it stores the
   work, reconstructs the abstract from OpenAlex's `abstract_inverted_index`,
   and upserts each authorship whose institution matches the target. Authors
   are deduplicated on `openalex_author_id`.

3. **`emails resolve`** runs the resolution chain per person, stopping at the
   first hit and recording both source and confidence:

   | Order | Source | Confidence |
   |---|---|---|
   | 1 | Crossref corresponding-author record | high |
   | 2 | ORCID public record | high |
   | 3 | School adapter | high |
   | 4 | Generic faculty-page crawler | medium |
   | 5 | none found -> NULL | none |

   The system never infers an address from a name pattern. A guessed address
   bounces, and bounce rate is a deliverability signal that damages every
   later email sent from the same account. A NULL with a homepage link is more
   useful than a wrong address.

4. **`match`** (stage 1) embeds the profile and every work abstract locally
   with `sentence-transformers` (default model `all-MiniLM-L6-v2`), then scores
   each person. See the scoring rule below.

5. **`match --rerank`** (stage 2, opt-in) sends the top N people, each with
   their three highest-scoring titles and abstracts plus the profile, to the
   Claude API for a structured fit score and a one-sentence justification.
   Results land in `matches.stage2_score` and `matches.reason`.

6. **`show` / `export`** render the ranking, sorted by `stage2_score` when
   present and `stage1_score` otherwise.

## Scoring rule

**Person score = mean of that person's top 3 work similarities.**

This is the central quality decision. Averaging all of a researcher's works
regresses prolific people toward the mean and buries exactly the professors
who have a few papers precisely on your topic. Taking a single maximum instead
is too noisy — one coincidentally-worded abstract promotes an unrelated
researcher. Mean-of-top-3 rewards demonstrated sustained overlap while staying
robust to a single lucky match.

Refinements:

- People with fewer than 3 works in the corpus are scored on what exists, with
  no padding, and flagged `sparse` in output.
- Works missing an abstract fall back to title-only embedding and are flagged.
  Roughly 40% of OpenAlex records lack an abstract, so this path is common,
  not exceptional, and must never be silent.
- `top_work_ids` stores which three works produced the score, so every ranking
  row can show its own evidence. A score without evidence is not actionable
  when the output's purpose is writing a specific email.

## Network policy

Enforced in `net/http.py`:

- One shared session with an on-disk response cache keyed by URL and ETag.
- `robots.txt` fetched and honoured per host before any crawl.
- Default 1 request/second per host; configurable downward only.
- Descriptive User-Agent including the contact email from `profile.yaml`.
- `Retry-After` honoured; exponential backoff on 5xx; circuit-break a host
  after repeated failures.
- OpenAlex requests include the `mailto` parameter for polite-pool rate limits.
- No LinkedIn, no authenticated scraping, no CAPTCHA circumvention.

## School adapters

```python
class SchoolAdapter(ABC):
    slug: str
    domains: list[str]

    @abstractmethod
    def faculty_urls(self) -> Iterable[str]: ...

    @abstractmethod
    def parse_faculty(self, html: str, url: str) -> list[FacultyRecord]: ...
```

Adapters exist because Korean university sites in particular are heavily
JavaScript-rendered and obfuscate addresses, so the generic crawler performs
poorly there. Seeded adapters: KAIST, GIST, SNU. A shared contract test suite
runs against every registered adapter, which is what makes it safe for
strangers to contribute more.

Adapters use only the shared session and must not require a browser runtime.
A site that cannot be read without executing JavaScript is left to the manual
queue rather than driven with a headless browser.

## Error handling

| Condition | Behaviour |
|---|---|
| Network failure mid-`discover` | Cursor persisted; rerun resumes |
| OpenAlex 429 | Honour `Retry-After`, back off, continue |
| Abstract missing | Title-only embedding, row flagged |
| Email unresolvable | NULL + homepage, listed by `emails report` |
| No Claude API key | `--rerank` refuses with a clear message; stage 1 unaffected |
| `robots.txt` disallows | Skip host, log, continue |
| Corrupt/partial DB | Migrations are forward-only and transactional |

The tool must be fully useful with zero API keys configured. Anything else
loses the users who would otherwise try it.

## Testing

- `pytest`, with `vcrpy` cassettes for every HTTP interaction. No test touches
  the live network.
- Fixtures: a trimmed OpenAlex institution-works response, a Crossref record,
  an ORCID record, and one saved faculty-directory page per seeded adapter.
- `match/score.py` is pure functions over fixture vectors, unit-tested directly
  including the sparse-author and missing-abstract branches.
- One architectural test asserts no module outside `net/` imports an HTTP
  client.
- One contract test parametrised over every registered school adapter.

## Future modules

Planned against the same database, not built here: CV and SOP rendering from a
YAML profile, a scholarship eligibility rules engine, a curated fees and
deadlines dataset, and Gmail draft generation for approved targets. The
`matches` table is the join point — outreach and document generation both key
off a ranked person.
