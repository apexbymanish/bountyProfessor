"""The `gradpath` command line — wires every module above into a usable tool.

Only `--rerank` on `match` requires `ANTHROPIC_API_KEY`; every other command,
including the full `init` -> `discover` -> `match` -> `show`/`export` path,
works with zero API keys. `load_embedder` stays a lazy import inside
`gradpath.match.embed` so `init`, `discover` and `show` never pay
sentence-transformers' torch startup cost.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from gradpath.config import (
    load_fields,
    load_institutions,
    load_profile,
    load_settings,
    resolve_field,
    scaffold_profile,
)
from gradpath.db import connect, migrate
from gradpath.match.embed import (
    build_profile_vector,
    embed_pending_works,
    load_embedder,
    run_match,
)
from gradpath.match.faculty import PublicationHistory, score_faculty
from gradpath.match.rerank import MissingApiKey, RerankPlan, rerank
from gradpath.models import Profile, Settings
from gradpath.net.http import PoliteClient
from gradpath.report.export import to_csv, to_markdown
from gradpath.report.table import query_results, render_table
from gradpath.sources.author_stats import enrich_person_career
from gradpath.sources.emails import persist_resolution, resolve_email
from gradpath.sources.ingest import discover as run_discover
from gradpath.sources.openalex import search_topics
from gradpath.util import now_iso

app = typer.Typer(help="Find the professors worth emailing for graduate study.")
fields_app = typer.Typer(help="Resolve research fields to OpenAlex topics.")
institutions_app = typer.Typer(help="Inspect institutions.")
emails_app = typer.Typer(help="Resolve and report contact addresses.")
faculty_app = typer.Typer(help="Estimate who is faculty. Nobody is ever removed.")
app.add_typer(fields_app, name="fields")
app.add_typer(institutions_app, name="institutions")
app.add_typer(emails_app, name="emails")
app.add_typer(faculty_app, name="faculty")

console = Console()
STATE: dict[str, Path] = {}


@app.callback()
def main(
    root: Annotated[Path, typer.Option("--root", help="Workspace directory")] = Path("."),
) -> None:
    STATE["root"] = root


def _paths() -> dict[str, Path]:
    root = STATE.get("root", Path("."))
    return {
        "db": root / "gradpath.db",
        "settings": root / "settings.yaml",
        "profile": root / "profile.yaml",
        "fields": root / "data" / "fields.yaml",
        "institutions": root / "data" / "institutions.yaml",
        "cache": root / ".cache",
    }


def _context() -> tuple[Settings, Profile, sqlite3.Connection, PoliteClient]:
    """Load settings/profile/db/client for every command except `init`.

    A new user's very first mistake is almost always running some other
    command before `init`, or leaving profile.yaml half-filled-in. Both
    load_settings and load_profile raise plain FileNotFoundError/ValueError
    for those cases, which -- left uncaught -- surfaces as a raw Python
    traceback: exactly the kind of failure this CLI already refuses to
    produce for an unknown --field (see `discover`). Catch both here so
    every command gets the same clean, actionable failure instead.
    """
    paths = _paths()
    try:
        settings = load_settings(paths["settings"])
        profile = load_profile(paths["profile"])
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print("[yellow]run `gradpath init` first to set up this workspace[/yellow]")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print("[yellow]fix settings.yaml or profile.yaml and try again[/yellow]")
        raise typer.Exit(code=1) from exc
    conn = connect(paths["db"])
    migrate(conn)
    client = PoliteClient(settings, profile.contact_email, paths["cache"])
    return settings, profile, conn, client


def _seed_institutions(conn: sqlite3.Connection, path: Path) -> int:
    """Seed curated institutions (with their adapter, if any) into the DB.

    Uses INSERT OR IGNORE so a row `discover` later inserts for the same id
    is left untouched, and so a row already seeded (or already discovered)
    keeps whatever adapter/site it has -- this only ever fills gaps, never
    overwrites. Run this before `discover` so email resolution's
    adapter step (see gradpath/sources/emails.py) has something to find.
    """
    seeds = load_institutions(path)
    added = 0
    with conn:
        for slug, seed in seeds.items():
            cursor = conn.execute(
                "INSERT OR IGNORE INTO institutions "
                "(id, name, country, site, adapter, discovered, added_at) "
                "VALUES (?, ?, ?, ?, ?, 0, ?)",
                (slug, seed.name, seed.country, seed.site, seed.adapter, now_iso()),
            )
            added += int(cursor.rowcount > 0)
    return added


@app.command()
def init() -> None:
    """Create the database, scaffold profile.yaml, and seed known institutions."""
    paths = _paths()
    conn = connect(paths["db"])
    version = migrate(conn)
    console.print(f"[green]database ready[/green] at {paths['db']} (schema v{version})")
    try:
        scaffold_profile(paths["profile"])
        console.print(f"[green]wrote[/green] {paths['profile']} — fill in contact_email")
    except FileExistsError:
        console.print(f"profile already exists at {paths['profile']}, left untouched")

    institutions_path = paths["institutions"]
    if institutions_path.exists():
        added = _seed_institutions(conn, institutions_path)
        console.print(f"[green]seeded {added}[/green] known institutions from {institutions_path}")
    conn.close()


@fields_app.command("search")
def fields_search(query: str) -> None:
    """Find OpenAlex topic ids for a research area."""
    _, _, _, client = _context()
    for hit in search_topics(client, query):
        console.print(f"{hit.topic_id}\t{hit.works_count:>8}\t{hit.label}")


@fields_app.command("list")
def fields_list() -> None:
    """List the fields defined in data/fields.yaml."""
    for slug, definition in load_fields(_paths()["fields"]).items():
        console.print(f"{slug}\t{definition.label}\t{','.join(definition.topics)}")


@app.command()
def discover(
    field: Annotated[
        list[str] | None, typer.Option("--field", help="Field slug(s) from data/fields.yaml")
    ] = None,
    country: Annotated[
        list[str] | None, typer.Option("--country", help="ISO country code(s)")
    ] = None,
    institution: Annotated[
        list[str] | None,
        typer.Option("--institution", help="OpenAlex institution id(s); overrides --country"),
    ] = None,
    since: Annotated[
        int | None, typer.Option("--since", help="Earliest publication year")
    ] = None,
) -> None:
    """Find researchers. Field-first by default; --institution targets a fixed list."""
    settings, profile, conn, client = _context()
    definitions = load_fields(_paths()["fields"])
    names = field or profile.fields
    try:
        topics = [t for name in names for t in resolve_field(name, definitions).topics]
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        raise typer.Exit(code=1) from exc

    countries = [c.upper() for c in (country or profile.countries)]
    since_year = since or settings.default_since_year
    key = f"discover:{'|'.join(sorted(topics))}:{'|'.join(sorted(countries))}:{since_year}"
    stats = run_discover(
        conn, client, topics, countries, since_year, key,
        list(institution) if institution else None,
    )
    console.print(
        f"[green]{stats.works_seen}[/green] works, "
        f"[green]{stats.people_seen}[/green] people, "
        f"[green]{stats.institutions_added}[/green] new institutions"
    )


@faculty_app.command("score")
def faculty_score() -> None:
    """Estimate who is faculty. Nobody is removed; this fills a sortable column.

    Career span and volume come from career_first_year/career_last_year/
    career_works_count -- the researcher's whole OpenAlex record, populated by
    `faculty enrich-career` -- falling back to the topic/--since-bounded slice
    columns only when those are NULL (ruling R27). Author-position ratios
    (last_author_ratio/first_author_ratio) always come from the slice: how a
    person is credited *within the searched field* is the meaningful signal,
    and only span/volume need the global record. How many people were scored
    each way is reported below, on purpose -- silent fallback is exactly how
    this bug (every professor scoring `low` under a narrow `--since`) hid in
    the first place.
    """
    _, _, conn, _ = _context()
    rows = conn.execute(
        "SELECT id, first_year, last_year, works_count, last_author_ratio, "
        "first_author_ratio, title, career_first_year, career_last_year, "
        "career_works_count FROM people"
    ).fetchall()
    career_based = 0
    fallback = 0
    for row in rows:
        has_career = row["career_works_count"] is not None
        if has_career:
            works = row["career_works_count"] or 0
            first_year = row["career_first_year"] or 0
            last_year = row["career_last_year"] or 0
            career_based += 1
        else:
            works = row["works_count"] or 0
            first_year = row["first_year"] or 0
            last_year = row["last_year"] or 0
            fallback += 1
        history = PublicationHistory(
            first_year=first_year,
            last_year=last_year,
            works_count=works,
            last_author_count=int((row["last_author_ratio"] or 0.0) * works),
            first_author_count=int((row["first_author_ratio"] or 0.0) * works),
            orcid_title=row["title"],
        )
        assessment = score_faculty(history)
        with conn:
            conn.execute(
                "UPDATE people SET faculty_score = ?, faculty_confidence = ?, "
                "faculty_signals = ? WHERE id = ?",
                (assessment.score, assessment.confidence,
                 json.dumps(assessment.signals), row["id"]),
            )
    console.print(f"[green]scored {len(rows)} people[/green]")
    console.print(
        f"  [cyan]{career_based}[/cyan] scored on whole-career data, "
        f"[yellow]{fallback}[/yellow] fell back to the discovered slice"
        + (" — run `faculty enrich-career` for accurate spans on those" if fallback else "")
    )


# R36: enrich-career is one polite API call per person. `match` gives every
# discovered person a matches row with no cap, so a threshold that defaults
# to "everyone" (the old --min-score 0.0 default) turns a bare
# `faculty enrich-career` into exactly the 10**4-call run R27 exists to
# avoid. Above this many people, the command must stop and ask -- at the
# shared 1 req/sec/host rate limit (see MAX_RATE_LIMIT in gradpath/config.py)
# ten thousand people is roughly three hours of continuous API calls, and
# someone who ran the command without reading --help should learn that
# before it starts, not an hour in.
ENRICH_CONFIRM_THRESHOLD = 200


def _format_duration(seconds: float) -> str:
    """A coarse, honest wall-clock estimate -- never implying more precision than it has."""
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"~{hours}h{minutes:02d}m"
    if minutes:
        return f"~{minutes}m{secs:02d}s"
    return f"~{secs}s"


@faculty_app.command("enrich-career")
def faculty_enrich_career(
    min_score: Annotated[
        float | None,
        typer.Option(
            "--min-score",
            help="Only enrich people at or above this fit. Defaults to "
                 "settings.show_min_score, so a bare run does not sweep every "
                 "discovered person -- pass 0.0 explicitly to do that.",
        ),
    ] = None,
    faculty_only: Annotated[
        bool, typer.Option("--faculty-only", help="Only enrich likely faculty")
    ] = False,
) -> None:
    """Fetch each person's whole-career OpenAlex record (span, volume).

    One polite API call per person, so this deliberately runs over a ranked,
    filtered subset -- the same --min-score/--faculty-only filtering
    `emails resolve` uses -- never over everyone discovered (ruling R36).
    Above ENRICH_CONFIRM_THRESHOLD people, this reports the count and the
    implied wall-clock time and asks for confirmation before starting, the
    same pattern `match --rerank` uses for spend. Run `match` first so
    there is a ranking to filter by, and run this before `faculty score` so
    its output reflects real career spans (ruling R27).
    """
    settings, profile, conn, client = _context()
    threshold = min_score if min_score is not None else settings.show_min_score
    targets = query_results(conn, profile.name, threshold, faculty_only)

    if len(targets) > ENRICH_CONFIRM_THRESHOLD:
        eta = len(targets) / max(settings.rate_limit_per_host, 0.01)
        proceed = typer.confirm(
            f"{len(targets)} people match -- one polite API call each, about "
            f"{_format_duration(eta)} at this host's rate limit. Proceed?"
        )
        if not proceed:
            console.print("[yellow]aborted; nothing enriched[/yellow]")
            return

    enriched = 0
    for row in targets:
        person = conn.execute(
            "SELECT * FROM people WHERE id = ?", (row.person_id,)
        ).fetchone()
        if person is None:
            continue
        enriched += int(enrich_person_career(conn, client, person))
    console.print(
        f"[green]enriched {enriched}[/green] of {len(targets)} considered with career data"
    )


@emails_app.command("resolve")
def emails_resolve(
    min_score: Annotated[float, typer.Option("--min-score")] = 0.0,
    faculty_only: Annotated[bool, typer.Option("--faculty-only")] = False,
) -> None:
    """Resolve addresses for people who rank, so crawling effort follows the ranking.

    Targets are looked up by person_id, not display name: at 10**4
    researchers, name collisions are near-certain, and a name-keyed lookup
    would resolve (and overwrite) the wrong person's address.
    """
    _, profile, conn, client = _context()
    targets = query_results(conn, profile.name, min_score, faculty_only)
    resolved = 0
    for row in targets:
        person = conn.execute(
            "SELECT * FROM people WHERE id = ?", (row.person_id,)
        ).fetchone()
        if person is None or person["email"]:
            continue
        resolution = resolve_email(conn, client, person)
        persist_resolution(conn, person["id"], resolution)
        resolved += int(resolution.email is not None)
    console.print(f"[green]resolved {resolved}[/green] of {len(targets)} considered")


@emails_app.command("report")
def emails_report() -> None:
    """List people with no address found, for manual lookup."""
    _, _, conn, _ = _context()
    rows = conn.execute(
        "SELECT name, homepage FROM people WHERE email IS NULL ORDER BY faculty_score DESC"
    ).fetchall()
    for row in rows:
        console.print(f"{row['name']}\t{row['homepage'] or '(no homepage)'}")
    console.print(f"[yellow]{len(rows)} addresses need manual lookup[/yellow]")


@app.command("match")
def match_command(
    do_rerank: Annotated[
        bool, typer.Option("--rerank", help="Also run the budget-gated LLM rerank")
    ] = False,
    min_score: Annotated[
        float | None, typer.Option("--min-score", help="Rerank threshold on stage 1 fit")
    ] = None,
    budget: Annotated[
        float | None,
        typer.Option(
            "--budget",
            help="Approximate spending ceiling in USD for --rerank. This is not a hard "
                 "cap: real per-call cost is only known after the call, so a run can "
                 "finish up to one call's cost above this figure.",
        ),
    ] = None,
) -> None:
    """Score every person. No cap."""
    settings, profile, conn, _ = _context()
    embedder = load_embedder(settings.embedding_model)
    added = embed_pending_works(
        conn, embedder, settings.embedding_model, settings.embedding_batch_size
    )
    console.print(f"embedded {added} new works")
    vector = build_profile_vector(profile, [], embedder)
    scored = run_match(conn, profile.name, vector, settings.embedding_model)
    console.print(f"[green]scored {scored} people[/green]")

    if not do_rerank:
        return
    threshold = min_score if min_score is not None else settings.show_min_score
    ceiling = budget if budget is not None else settings.rerank_default_budget_usd

    def ask(plan: RerankPlan) -> bool:
        return typer.confirm(
            f"rerank {plan.person_count} people for an estimated "
            f"${plan.estimated_cost_usd:.2f} (estimated — actual spend may differ; "
            f"approximate ceiling ${ceiling:.2f})?"
        )

    try:
        outcome = rerank(
            conn, profile.name, threshold, settings.rerank_model, ceiling, ask,
            profile.interests,
        )
    except MissingApiKey as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return

    console.print(f"[green]reranked {outcome.scored} people[/green]")
    # R34: the *measured* figure is the one that matters — --budget is an
    # approximate ceiling, not a guarantee, and a run can finish up to one
    # call's cost above it. Printing "budget: $X" and silently spending more
    # is exactly the broken promise this exists to prevent.
    console.print(
        f"[bold]actual spend: ${outcome.spent_usd:.2f}[/bold] (measured; "
        f"approximate ceiling was ${ceiling:.2f})"
    )


@app.command("show")
def show_command(
    min_score: Annotated[float | None, typer.Option("--min-score")] = None,
    faculty_only: Annotated[bool, typer.Option("--faculty-only")] = False,
    country: Annotated[str | None, typer.Option("--country")] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="display only; never affects scoring")
    ] = None,
) -> None:
    """Print the ranked table."""
    settings, profile, conn, _ = _context()
    threshold = min_score if min_score is not None else settings.show_min_score
    rows = query_results(conn, profile.name, threshold, faculty_only, country, limit)
    console.print(render_table(rows))


@app.command("export")
def export_command(
    csv_path: Annotated[Path | None, typer.Option("--csv")] = None,
    markdown_path: Annotated[Path | None, typer.Option("--markdown")] = None,
    min_score: Annotated[float | None, typer.Option("--min-score")] = None,
    faculty_only: Annotated[bool, typer.Option("--faculty-only")] = False,
    country: Annotated[str | None, typer.Option("--country")] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
) -> None:
    """Write the ranked table to CSV and/or Markdown."""
    settings, profile, conn, _ = _context()
    threshold = min_score if min_score is not None else settings.show_min_score
    rows = query_results(conn, profile.name, threshold, faculty_only, country, limit)
    if csv_path:
        to_csv(rows, csv_path)
        console.print(f"[green]wrote[/green] {csv_path}")
    if markdown_path:
        to_markdown(rows, markdown_path)
        console.print(f"[green]wrote[/green] {markdown_path}")
    if not csv_path and not markdown_path:
        raise typer.BadParameter("pass --csv or --markdown")


@institutions_app.command("list")
def institutions_list(
    discovered: Annotated[bool, typer.Option("--discovered")] = False,
) -> None:
    """List institutions known to the database."""
    _, _, conn, _ = _context()
    sql = "SELECT id, name, country, discovered FROM institutions"
    if discovered:
        sql += " WHERE discovered = 1"
    for row in conn.execute(sql + " ORDER BY name"):
        console.print(f"{row['id']}\t{row['country'] or '--'}\t{row['name']}")


if __name__ == "__main__":
    app()
