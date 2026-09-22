# Ranking snapshots

This directory holds QS/THE/ARWU (or similar) ranking snapshots for
`gradpath institutions import`. **No snapshot is committed here** — a
ranking is third-party data, it changes every year, and the major rankings
disagree with each other, so a copy checked into this repository would be
stale on arrival and impossible to verify. This directory (aside from this
README) is gitignored.

Ranks are never hand-written into `gradpath` itself. There are exactly two
sourced paths for institution ordering:

- `gradpath institutions top --country KR --limit 20 --tier korea-20` —
  ordering derived live from OpenAlex research output (`cited_by_count` or
  `works_count`). Reproducible and current by construction; needs no file
  here at all.
- `gradpath institutions import --csv <path> --tier world-100 --top 100` —
  a ranking snapshot **you** supply, from a source you can cite.

## Expected CSV columns

```
rank,name,country
1,Massachusetts Institute of Technology,US
2,Imperial College London,GB
3,Stanford University,US
```

- `rank` — integer position in the source ranking.
- `name` — the institution's name, as printed in that source. Used to
  derive the row's id (`slugify(name)`) and, later, to resolve an OpenAlex
  id against it.
- `country` — optional ISO country code.

Only `rank` and `name` are required; extra columns are ignored.

`import_ranking_csv` deliberately leaves `openalex_id` NULL on every
imported row — an import never invents an identifier it cannot verify. A
CSV row also carries no homepage, so `gradpath institutions top`'s
homepage-domain resolution (the path that resolves institutions seeded from
`data/institutions.yaml`) can never reach these rows.

Resolution for a CSV-imported tier is a separate step:

```
gradpath institutions import --csv qs2026.csv --tier world-100 --top 100
gradpath institutions resolve --tier world-100
gradpath discover --field efficient-ml --tier world-100
```

`institutions resolve` searches OpenAlex by name and binds an id **only**
on an exact, normalised match (case/punctuation/diacritics/leading "The"
folded away) — never a fuzzy or "closest" match, since a wrong bind would
silently attach one institution's rank and tier to a different one, with
no visible sign anything went wrong (every later display shows the stored
name, not whatever it matched). A near-match or a tie between candidates is
left unresolved and printed so it can be checked by hand; `discover --tier`
then only ever sees rows that were actually resolved.

## Limitation: one tier per institution

The current schema stores a single `tier`/`rank` pair per institution row —
there is no many-to-many `institution_tiers` table. If the same real
institution appears in two tiers you've built (e.g. KAIST is both in
`korea-20`, seeded via `institutions top`, and in a `world-100` CSV you
imported), the CSV row resolves to the *same* OpenAlex id as the row
already seeded. Rather than silently merging the two (which would discard
whichever tier/rank you imported second) or silently skipping the row
(which would look identical to success), `institutions resolve` reports
that as a **conflict**, naming both institutions, and leaves the CSV row
unresolved for you to decide by hand. A real fix — letting one institution
carry more than one tier/rank — needs that many-to-many table; it is a
deliberate non-goal here, not an oversight.

## Recording provenance

Next to each snapshot you place here, add a sibling `.source` file recording
where it came from and when it was retrieved, so any imported rank can be
cited later, e.g.:

```
# qs2026.csv.source
Source: QS World University Rankings 2026
URL: https://www.topuniversities.com/world-university-rankings
Retrieved: 2026-09-22
```
