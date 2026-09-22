# gradpath — Design

**Date:** 2026-09-22
**Status:** Approved (rev 2 — field-first discovery, no ranking cap)
**Slice:** Researcher discovery + interest matching (slice 1 of a larger pipeline)

## Overview

`gradpath` helps a prospective graduate student find the professors worth
emailing. The applicant describes a research field and what they want to work
on; the system finds every researcher publishing in that area across the
countries they care about, scores each one against the applicant's interests,
estimates who is actually faculty, and emits a ranked, contactable shortlist
with evidence for every row.

The product insight: the bottleneck in graduate applications is not finding
universities, it is finding the *specific people* whose current work overlaps
yours. Ranking sites answer the first question. Nothing answers the second.

The corollary, and the reason for rev 2: **the field decides who surfaces, not
a row count.** An excellent match at rank 180, at an institution the applicant
had never heard of, is exactly the result this tool exists to produce. Nothing
in the pipeline truncates to an arbitrary N.

## Goals

- Turn "I want to work on X, in these countries" into a ranked, contactable
  list of researchers with evidence for why each one fits.
- Search by field first, so institutions are discovered from results rather
  than guessed in advance.
- Scale to the full author population of a field without arbitrary caps.
- Work fully offline after data collection, with no paid API key required.
- Be publishable as a public repository from the first commit.
- Be polite enough to run against real university infrastructure without
  causing harm or getting blocked.

## Non-goals (this slice)

CV and SOP generation, scholarship eligibility rules, a tuition and fees
database, sending email, a web dashboard, and any use of LinkedIn. These are
later modules against the same SQLite spine. The schema anticipates them; the
code does not implement them.

## Architecture

Layered, with a single network chokepoint.

```
cli.py            command surface (typer)
  |
  +-- sources/    acquisition (openalex, crossref, orcid, crawler, adapters)
  +-- match/      scoring (embed, faculty, score, rerank)
  +-- report/     presentation (table, export)
  |
  +-- db.py       SQLite persistence
  +-- net/http.py THE ONLY module permitted to make network calls
```

`net/http.py` being the sole network path is a load-bearing constraint, not a
convention. Rate limiting, `robots.txt` compliance, caching, retry and
User-Agent policy are enforced there, so no source module can accidentally
violate them. A test asserts that no module outside `net/` imports an HTTP
client.

### Repository layout

```
gradpath/
  pyproject.toml
  README.md
  LICENSE                     MIT
  .gitignore                  profile.yaml, *.db, .cache/
  gradpath.sh
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
        base.py               InstitutionAdapter ABC
        kaist.py
        gist.py
        snu.py
    match/
      __init__.py
      embed.py                batched local embeddings + vector cache
      faculty.py              faculty-likelihood scoring
      score.py                pure scoring functions
      rerank.py               optional Claude pass, budget-gated
    report/
      __init__.py
      table.py
      export.py
  data/
    institutions.yaml         committed seed registry (optional shortcuts)
    fields.yaml               field -> OpenAlex topic id mappings
  tests/
    fixtures/
    fixtures/
  docs/
```

## Data model

SQLite, created by `db.py` with forward-only numbered migrations.

```sql
CREATE TABLE institutions (
    id              TEXT PRIMARY KEY,      -- slug, e.g. 'kaist'
    name            TEXT NOT NULL,
    ror_id          TEXT,
    openalex_id     TEXT UNIQUE,
    country         TEXT,
    site            TEXT,
    adapter         TEXT,                  -- adapter module name, nullable
    discovered      INTEGER DEFAULT 0,     -- 1 = found via field search
    added_at        TEXT NOT NULL
);

CREATE TABLE people (
    id                  INTEGER PRIMARY KEY,
    institution_id      TEXT REFERENCES institutions(id),
    dept                TEXT,
    name                TEXT NOT NULL,
    openalex_author_id  TEXT UNIQUE,
    orcid               TEXT,
    title               TEXT,              -- from ORCID or directory
    homepage            TEXT,
    email               TEXT,
    email_confidence    TEXT,              -- high | medium | none
    email_source        TEXT,              -- crossref | orcid | adapter | crawler
    works_count         INTEGER DEFAULT 0,
    first_year          INTEGER,
    last_year           INTEGER,
    last_author_ratio   REAL,
    first_author_ratio  REAL,
    faculty_score       REAL,
    faculty_confidence  TEXT,              -- high | medium | low
    faculty_signals     TEXT,              -- JSON, why the score is what it is
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
    topics              TEXT,              -- JSON array of OpenAlex topic ids
    cited_by            INTEGER DEFAULT 0
);

CREATE TABLE authorships (
    person_id   INTEGER NOT NULL REFERENCES people(id),
    work_id     INTEGER NOT NULL REFERENCES works(id),
    position    TEXT,                      -- first | middle | last
    PRIMARY KEY (person_id, work_id)
);

CREATE TABLE embeddings (
    work_id     INTEGER PRIMARY KEY REFERENCES works(id),
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,             -- float32
    source      TEXT NOT NULL,             -- abstract | title_only
    computed_at TEXT NOT NULL
);

CREATE TABLE matches (
    profile_id      TEXT NOT NULL,
    person_id       INTEGER NOT NULL REFERENCES people(id),
    stage1_score    REAL NOT NULL,
    stage2_score    REAL,
    reason          TEXT,
    top_work_ids    TEXT,                  -- JSON array, evidence for the score
    sparse          INTEGER DEFAULT 0,     -- fewer than 3 works in corpus
    computed_at     TEXT NOT NULL,
    PRIMARY KEY (profile_id, person_id)
);

CREATE TABLE cursors (
    key         TEXT PRIMARY KEY,          -- e.g. 'discover:field:T10028:KR'
    cursor      TEXT,
    updated_at  TEXT
);

CREATE TABLE fetch_log (
    url         TEXT PRIMARY KEY,
    fetched_at  TEXT NOT NULL,
    status      INTEGER,
    etag        TEXT
);

CREATE INDEX idx_matches_score ON matches(profile_id, stage1_score DESC);
CREATE INDEX idx_people_faculty ON people(faculty_confidence, faculty_score DESC);
CREATE INDEX idx_works_year ON works(year);
```

`cursors` makes long collection runs resumable after Ctrl-C. `embeddings` is
what makes the no-cap decision affordable: vectors are computed once and reused
across every subsequent `match` run and every profile.

`institutions.discovered` distinguishes universities the user named from ones
that appeared because someone there matched the field.

## Configuration

Four files.

`data/fields.yaml` is committed and maps human field names to OpenAlex topic
identifiers, so users do not have to look them up:

```yaml
efficient-ml:
  label: Efficient machine learning
  topics: [T10028, T11689]
  aliases: [edge-ml, model-compression, on-device-ml]
```

`data/institutions.yaml` is committed and community-editable. It is a
convenience registry of well-known universities, not a required target list:

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
embedding_batch_size: 256
rate_limit_per_host: 1.0        # requests/second; may be lowered, not raised
default_since_year: 2021
show_min_score: 0.65            # display threshold, not a cap
rerank_model: claude-opus-5
rerank_default_budget_usd: 2.00
cache_dir: .cache
```

`profile.yaml` is gitignored and scaffolded by `gradpath init`:

```yaml
name: default                    # becomes matches.profile_id
contact_email: you@example.com   # used in the polite User-Agent
fields: [efficient-ml]           # default discovery fields
countries: [KR, JP, DE]          # default discovery countries
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

The interest inputs are merged into one vector as a weighted mean of component
embeddings:

| Component | Weight |
|---|---|
| `interests` free text | 1.0 |
| `keywords`, joined into a single string | 1.0 |
| Each resolved `seed_papers` / `my_papers` abstract | 2.0 |

Paper abstracts are weighted double because a real abstract describes a
research area far more precisely than a self-written paragraph does. Seed
papers resolve to abstracts through the same OpenAlex path as any other work;
a DOI that cannot be resolved is reported and skipped rather than silently
dropped. If no component resolves, `match` refuses to run rather than scoring
against an empty vector.

## CLI surface

```
gradpath init                        create db, scaffold profile.yaml

gradpath fields search "machine learning"     find OpenAlex topic ids
gradpath fields list

gradpath discover --field <name>... [--country <cc>...] [--since <year>]
                                     PRIMARY: field-first across institutions
gradpath discover --institution <slug>...     secondary: named targets
gradpath discover --status                    progress, counts, resumability

gradpath institutions list [--discovered]

gradpath emails resolve [--min-score <f>] [--faculty-only]
gradpath emails report                        manual-lookup queue

gradpath faculty score                        compute faculty likelihood

gradpath match                                stage 1, scores EVERYONE
gradpath match --rerank [--min-score <f>] [--budget <usd>]

gradpath show [--min-score <f>] [--faculty-only] [--country <cc>] [--limit N]
gradpath export --csv <path> [same filters]
gradpath export --markdown <path>
```

`--limit` exists only on `show` and `export`, and only to keep a terminal
readable. It never affects what is computed or stored.

Every command is idempotent. Re-running `discover` resumes from its stored
cursor and skips already-fetched works.

## Pipeline

1. **`fields search`** resolves a human field name to OpenAlex topic IDs,
   writing shortcuts into `data/fields.yaml`.

2. **`discover --field`** (primary path) queries OpenAlex
   `/works?filter=topics.id:<ids>,institutions.country_code:<ccs>,publication_year:>=<since>`,
   paginating with a persisted cursor. For each work it stores the work, its
   topics, and the reconstructed abstract, then upserts every authorship along
   with the author's institution. Institutions not already known are inserted
   with `discovered = 1`. Authors are deduplicated on `openalex_author_id`.

   **`discover --institution`** (secondary path) is the same routine filtered
   by institution instead of country, for when the applicant has a fixed list.

3. **`faculty score`** computes faculty likelihood per person. See below.

4. **`emails resolve`** runs the resolution chain per person, stopping at the
   first hit and recording source and confidence:

   | Order | Source | Confidence |
   |---|---|---|
   | 1 | Crossref corresponding-author record | high |
   | 2 | ORCID public record | high |
   | 3 | Institution adapter | high |
   | 4 | Generic faculty-page crawler | medium |
   | 5 | none found -> NULL | none |

   The system never infers an address from a name pattern. A guessed address
   bounces, and bounce rate is a deliverability signal that damages every
   later email sent from the same account. A NULL plus a homepage link is more
   useful than a wrong address.

   Because the candidate pool is now large, `emails resolve` accepts
   `--min-score` and `--faculty-only` so that crawling effort is spent on
   people who actually rank, rather than on the whole corpus.

5. **`match`** embeds the profile and scores **every** person in the database.
   No cap. See the scoring rule below.

6. **`match --rerank`** selects the people above `--min-score`, reports how
   many there are and the estimated cost, and requires confirmation or a
   `--budget` ceiling before calling the Claude API. Each selected person is
   sent with their three highest-scoring titles and abstracts plus the profile,
   returning a structured fit score and a one-sentence justification into
   `matches.stage2_score` and `matches.reason`.

7. **`show` / `export`** render results filtered by score threshold, faculty
   confidence and country, sorted by `stage2_score` when present and
   `stage1_score` otherwise.

## Scoring rule

**Person fit score = mean of that person's top 3 work similarities.**

This is the central quality decision. Averaging all of a researcher's works
regresses prolific people toward the mean and buries exactly the professors
who have a few papers precisely on your topic. Taking a single maximum instead
is too noisy — one coincidentally-worded abstract promotes an unrelated
researcher. Mean-of-top-3 rewards demonstrated sustained overlap while staying
robust to a single lucky match.

Refinements:

- People with fewer than 3 works in the corpus are scored on what exists, with
  no padding, and marked `sparse` so the output can flag them.
- Works missing an abstract fall back to title-only embedding, recorded as
  `source = 'title_only'` in `embeddings`. Roughly 40% of OpenAlex records
  lack an abstract, so this path is common, not exceptional, and must never be
  silent.
- `top_work_ids` stores which three works produced the score, so every row can
  show its own evidence. A score without evidence is not actionable when the
  output's purpose is writing a specific email.

Fit score and faculty score are kept **separate and both displayed**. They are
never blended into one number, because the user needs to distinguish "great
match, probably a PhD student" from "great match, full professor" — those call
for different actions, not a single ranking.

## Faculty likelihood

Most authors at a university are students and postdocs. OpenAlex does not label
who is faculty, so `match/faculty.py` estimates it from signals:

| Signal | Interpretation |
|---|---|
| `directory_confirmed` | Found on an official faculty page by adapter or crawler |
| `orcid_title` | ORCID employment record matching professor/faculty terms |
| `last_author_ratio` | High last-author share suggests a PI in most fields |
| `career_span` | `last_year - first_year`; students span 3-5 years, faculty far more |
| `first_author_ratio` | High first-author share with a short span suggests a student |
| `works_count` | Sustained output over many years |

Rules:

- **Authoritative signals win.** `directory_confirmed` or a professor title in
  ORCID sets `faculty_confidence = 'high'` outright; heuristics never override
  a confirmed fact.
- Otherwise the heuristic combines career span and author position into
  `faculty_score`, thresholded into high / medium / low.
- **Author-position signals are downweighted in alphabetical-authorship
  fields** (mathematics, economics, parts of theoretical CS), detected from the
  work's OpenAlex topics. Last-author position carries no seniority meaning
  there, and using it would systematically mislabel those fields.
- `faculty_signals` stores the JSON of which signals fired, so any
  classification can be explained and disputed.

**Nobody is ever dropped.** `faculty_confidence` is a sortable, filterable
column, and `--faculty-only` is opt-in. A heuristic that silently hides the
right professor is the worst failure this tool could have; showing a ranked
PhD student costs the user five seconds.

## Scale

Expected working set for one field across several countries: 10^4 to 10^5
works, 10^3 to 10^4 people.

- Embeddings are computed in batches of `embedding_batch_size` and persisted as
  float32 BLOBs. A second `match` run, or a run with a different profile,
  recomputes nothing.
- Scoring is vectorised: works are loaded into one NumPy matrix and scored with
  a single matrix multiply against the profile vector, then grouped per person.
  No per-person Python loop over the corpus.
- Target: score a cached 100,000-work corpus in under 60 seconds on CPU.
- `discover` streams and commits in batches, so memory stays flat and an
  interrupted run loses at most one batch.
- If a corpus exceeds available memory, scoring chunks the matrix; correctness
  is identical, only throughput changes.

## Network policy

Enforced in `net/http.py`:

- One shared session with an on-disk response cache keyed by URL and ETag.
- `robots.txt` fetched and honoured per host before any crawl.
- Default 1 request/second per host; configurable downward only.
- Descriptive User-Agent including the contact email from `profile.yaml`.
- `Retry-After` honoured; exponential backoff on 5xx; a host is circuit-broken
  after repeated failures.
- OpenAlex requests include the `mailto` parameter for polite-pool rate limits.
- No LinkedIn, no authenticated scraping, no CAPTCHA circumvention.

Field-first discovery can touch hundreds of institutions, which makes these
limits more important than in rev 1, not less. Crawling is deliberately gated
behind `--min-score` so that breadth of discovery does not become breadth of
crawling.

## Institution tiers and seeding

Discovery is field-first, but an applicant usually also wants to scope a run to
a realistic set of universities — "Korea's top 20", "the world top 100". The
`institutions` table therefore carries `tier`, `rank` and `rank_source`, and
`discover --tier <name>` restricts a run to a seeded tier.

Ranks are never hand-written into the repository. They change every year and
the major rankings disagree with each other, so a typed list would be stale on
arrival and unverifiable. Two sourced paths instead:

- `gradpath institutions top --country KR --limit 20 --tier korea-20` orders
  institutions by live OpenAlex output (`cited_by_count` or `works_count`).
  Reproducible, current by construction, no external file, and it needs no
  ranking publisher's permission.
- `gradpath institutions import --csv <path> --tier world-100 --top 100`
  imports a QS, THE or ARWU snapshot for anyone who wants literal published
  ranks. Snapshots are not committed — they are third-party data — and
  `data/rankings/README.md` records the expected columns and where to obtain
  them, with a sibling `.source` file recording provenance and retrieval date
  so any imported rank can be cited.

An import never invents an OpenAlex identifier it cannot verify: CSV rows land
with `openalex_id` NULL and are resolved separately. Seeding also never
overwrites an adapter bound by hand, so re-seeding cannot undo a contributor's
mapping.

## Institution adapters

```python
class InstitutionAdapter(ABC):
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

With field-first discovery most institutions will have no adapter, which is
expected. Those people fall through to the generic crawler or the manual queue.

## Error handling

| Condition | Behaviour |
|---|---|
| Network failure mid-`discover` | Cursor persisted; rerun resumes |
| OpenAlex 429 | Honour `Retry-After`, back off, continue |
| Abstract missing | Title-only embedding, recorded in `embeddings.source` |
| Email unresolvable | NULL + homepage, listed by `emails report` |
| No Claude API key | `--rerank` refuses with a clear message; stage 1 unaffected |
| Rerank exceeds budget | Stop at the ceiling, report what was scored |
| `robots.txt` disallows | Skip host, log, continue |
| Embedding model absent | Downloaded once on first `match`, with a progress note |
| Corpus larger than memory | Chunked scoring, identical results |
| Corrupt/partial DB | Migrations are forward-only and transactional |

The tool must be fully useful with zero API keys configured. Anything else
loses the users who would otherwise try it.

## Testing

- `pytest`, with `respx` mocks for every HTTP interaction. No test touches
  the live network.
- Fixtures: a trimmed OpenAlex topic-filtered works response, a Crossref
  record, an ORCID record, and one saved faculty-directory page per seeded
  adapter.
- `match/score.py` is pure functions over fixture vectors, unit-tested
  directly, including the sparse-author and title-only branches.
- `match/faculty.py` is unit-tested against constructed publication histories:
  a clear PI, a clear PhD student, an alphabetical-authorship mathematician,
  and an ambiguous mid-career case.
- One architectural test asserts no module outside `net/` imports an HTTP
  client.
- One contract test parametrised over every registered adapter.
- One scale test: 10,000 synthetic works score within the documented time
  budget, guarding against an accidental per-person loop.

## Future modules

Planned against the same database, not built here: CV and SOP rendering from a
YAML profile, a scholarship eligibility rules engine, a curated fees and
deadlines dataset, and Gmail draft generation for approved targets. The
`matches` table is the join point — outreach and document generation both key
off a ranked person.
