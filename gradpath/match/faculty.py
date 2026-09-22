"""Faculty-likelihood scoring. Pure functions only.

Most OpenAlex authors at a university are PhD students and postdocs, and
OpenAlex does not label who is faculty. This module estimates it — and
deliberately never removes anyone. A heuristic that silently hides the right
professor is the worst failure this tool could have; showing a ranked PhD
student costs the reader five seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PROFESSOR_TITLE_TERMS = (
    "professor", "faculty", "lecturer", "reader",
    "principal investigator", "group leader", "chair",
)

SPAN_FULL_CREDIT_YEARS = 12.0
VOLUME_FULL_CREDIT_WORKS = 40.0

HIGH_THRESHOLD = 0.7
MEDIUM_THRESHOLD = 0.4


@dataclass(frozen=True)
class PublicationHistory:
    first_year: int
    last_year: int
    works_count: int
    last_author_count: int
    first_author_count: int
    directory_confirmed: bool = False
    orcid_title: str | None = None
    alphabetical_field: bool = False


@dataclass(frozen=True)
class FacultyAssessment:
    score: float
    confidence: str          # high | medium | low
    signals: dict = field(default_factory=dict)


def _has_professor_title(title: str | None) -> bool:
    if not title:
        return False
    lowered = title.lower()
    return any(term in lowered for term in PROFESSOR_TITLE_TERMS)


def score_faculty(history: PublicationHistory) -> FacultyAssessment:
    signals: dict = {}

    # Authoritative signals win outright. They are facts; the rest are guesses.
    if history.directory_confirmed:
        signals["directory_confirmed"] = True
        return FacultyAssessment(1.0, "high", signals)
    if _has_professor_title(history.orcid_title):
        signals["orcid_title"] = history.orcid_title
        return FacultyAssessment(1.0, "high", signals)

    span = max(history.last_year - history.first_year, 0)
    span_score = min(span / SPAN_FULL_CREDIT_YEARS, 1.0)
    volume_score = min(history.works_count / VOLUME_FULL_CREDIT_WORKS, 1.0)
    signals["career_span"] = span
    signals["works_count"] = history.works_count

    if history.alphabetical_field:
        # Author order carries no seniority meaning in these fields, so using it
        # would systematically mislabel mathematicians and economists.
        signals["alphabetical_field"] = True
        score = span_score * 0.7 + volume_score * 0.3
    else:
        total = history.works_count or 1
        last_ratio = history.last_author_count / total
        first_ratio = history.first_author_count / total
        position_score = max(0.0, last_ratio - first_ratio)
        signals["last_author_ratio"] = round(last_ratio, 3)
        signals["first_author_ratio"] = round(first_ratio, 3)
        score = span_score * 0.4 + position_score * 0.4 + volume_score * 0.2

    if score >= HIGH_THRESHOLD:
        confidence = "high"
    elif score >= MEDIUM_THRESHOLD:
        confidence = "medium"
    else:
        confidence = "low"
    return FacultyAssessment(round(score, 4), confidence, signals)
