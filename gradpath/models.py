from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldDef:
    slug: str
    label: str
    topics: list[str]
    aliases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Settings:
    embedding_model: str
    embedding_batch_size: int
    rate_limit_per_host: float
    default_since_year: int
    show_min_score: float
    rerank_model: str
    rerank_default_budget_usd: float
    cache_dir: str


@dataclass(frozen=True)
class Profile:
    name: str
    contact_email: str
    interests: str
    keywords: list[str]
    seed_papers: list[str]
    my_papers: list[str]
    fields: list[str]
    countries: list[str]


@dataclass(frozen=True)
class FacultyRecord:
    """One row parsed from an institution's faculty directory."""
    name: str
    email: str | None
    title: str | None
    homepage: str | None
    dept: str | None


@dataclass(frozen=True)
class InstitutionSeed:
    slug: str
    name: str
    country: str | None
    site: str | None
    adapter: str | None
