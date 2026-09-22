from __future__ import annotations

from dataclasses import dataclass, field

from gradpath.net.http import PoliteClient

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


def search_topics(client: PoliteClient, query: str) -> list[TopicHit]:
    payload = client.get_json(f"{OPENALEX_BASE}/topics", {"search": query, "per_page": 25})
    return [
        TopicHit(
            topic_id=_short_id(item.get("id")) or "",
            label=item.get("display_name") or "",
            works_count=item.get("works_count") or 0,
        )
        for item in payload.get("results", [])
    ]
