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
imported row — an import never invents an identifier it cannot verify.
Resolution happens separately: running `gradpath institutions top` for the
matching country later binds OpenAlex ids onto these rows by matching
homepage domain (see `gradpath/sources/institutions.py`), the same way it
resolves institutions seeded from `data/institutions.yaml`.

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
