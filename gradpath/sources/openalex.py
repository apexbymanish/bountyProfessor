from __future__ import annotations

from dataclasses import dataclass, field

from gradpath.net.http import HostBlocked, PoliteClient

OPENALEX_BASE = "https://api.openalex.org"


@dataclass(frozen=True)
class ParsedAuthorship:
    author_id: str
    name: str
    orcid: str | None
    position: str
    institution_openalex_id: str | None
    institution_name: str | None
    institution_country: str | None


@dataclass(frozen=True)
class ParsedWork:
    openalex_id: str
    title: str
    abstract: str | None
    year: int | None
    doi: str | None
    venue: str | None
    topics: list[str]
    cited_by: int
    authorships: list[ParsedAuthorship] = field(default_factory=list)


@dataclass(frozen=True)
class TopicHit:
    topic_id: str
    label: str
    works_count: int


def _short_id(url: str | None) -> str | None:
    """OpenAlex ids arrive as URLs; the trailing segment is the usable id."""
    return url.rsplit("/", 1)[-1] if url else None


def with_mailto(client: PoliteClient, params: dict) -> dict:
    """Merge `mailto` into `params` for OpenAlex's polite pool.

    This lives here, not on PoliteClient: PoliteClient is generic (it also
    serves Crossref, ORCID and arbitrary university sites), and attaching a
    `mailto` to those requests would be meaningless at best and would hand
    the operator's address to hosts that never asked for it. `setdefault`
    keeps this idempotent rather than duplicating the key if one is somehow
    already present.
    """
    merged = dict(params)
    merged.setdefault("mailto", client.contact_email)
    return merged


def reconstruct_abstract(index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stores abstracts as an inverted index; rebuild the running text."""
    if not index:
        return None
    positioned: list[tuple[int, str]] = [
        (position, word) for word, positions in index.items() for position in positions
    ]
    if not positioned:
        return None
    positioned.sort(key=lambda pair: pair[0])
    return " ".join(word for _, word in positioned)


def parse_work(raw: dict) -> ParsedWork:
    location = raw.get("primary_location") or {}
    source = location.get("source") or {}
    doi = raw.get("doi")
    authorships = []
    for entry in raw.get("authorships") or []:
        author = entry.get("author") or {}
        institutions = entry.get("institutions") or []
        institution = institutions[0] if institutions else {}
        authorships.append(
            ParsedAuthorship(
                author_id=_short_id(author.get("id")) or "",
                name=author.get("display_name") or "",
                orcid=_short_id(author.get("orcid")),
                position=entry.get("author_position") or "middle",
                institution_openalex_id=_short_id(institution.get("id")),
                institution_name=institution.get("display_name"),
                institution_country=institution.get("country_code"),
            )
        )
    return ParsedWork(
        openalex_id=_short_id(raw.get("id")) or "",
        title=raw.get("title") or "",
        abstract=reconstruct_abstract(raw.get("abstract_inverted_index")),
        year=raw.get("publication_year"),
        doi=doi.replace("https://doi.org/", "") if doi else None,
        venue=source.get("display_name"),
        topics=[_short_id(t.get("id")) or "" for t in (raw.get("topics") or [])],
        cited_by=raw.get("cited_by_count") or 0,
        authorships=authorships,
    )


@dataclass(frozen=True)
class DoiResolution:
    """What a batch of seed DOIs actually produced. Nothing is dropped silently."""
    texts: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    title_only: list[str] = field(default_factory=list)


def normalise_doi(doi: str) -> str:
    """Strip the URL forms people paste, leaving the bare `10.x/y` identifier."""
    cleaned = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):]
    return cleaned


def fetch_work_by_doi(client: PoliteClient, doi: str) -> ParsedWork | None:
    """Look one DOI up in OpenAlex. Returns None when it cannot be resolved.

    Never raises: an unknown DOI is a normal outcome the caller has to report
    to the user by name, not an error that should abort a `match` run.
    """
    identifier = normalise_doi(doi)
    if not identifier:
        return None
    try:
        payload = client.get_json(
            f"{OPENALEX_BASE}/works/doi:{identifier}", with_mailto(client, {})
        )
    except (HostBlocked, ValueError):
        return None
    if not isinstance(payload, dict) or not payload.get("id"):
        return None
    return parse_work(payload)


def resolve_doi_texts(client: PoliteClient, dois: list[str]) -> DoiResolution:
    """Resolve seed DOIs to the text that describes them, keeping the misses.

    An abstract describes a research area far better than a self-written
    paragraph, which is why these are weighted double downstream. Roughly 40%
    of OpenAlex records carry no abstract, so the title is used instead and
    that DOI is named as title-only -- a weaker signal the user should know
    about. A DOI that resolves to nothing is returned in `unresolved` so the
    caller can print it: a seed paper the user believes is steering the
    ranking, silently dropped, is precisely the failure this project refuses.
    """
    resolution = DoiResolution([], [], [])
    for doi in dois:
        work = fetch_work_by_doi(client, doi)
        text = (work.abstract or work.title or "").strip() if work else ""
        if not text:
            resolution.unresolved.append(doi)
            continue
        if not (work and work.abstract):
            resolution.title_only.append(doi)
        resolution.texts.append(text)
    return resolution


def search_topics(client: PoliteClient, query: str) -> list[TopicHit]:
    payload = client.get_json(
        f"{OPENALEX_BASE}/topics", with_mailto(client, {"search": query, "per_page": 25})
    )
    return [
        TopicHit(
            topic_id=_short_id(item.get("id")) or "",
            label=item.get("display_name") or "",
            works_count=item.get("works_count") or 0,
        )
        for item in payload.get("results", [])
    ]
