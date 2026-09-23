from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass

from rich.table import Table

FIELDNAMES = [
    "person_id", "rank", "name", "institution", "country", "fit", "faculty_score",
    "faculty_confidence", "email", "email_confidence", "email_source", "sparse",
    "reason",
]


@dataclass(frozen=True)
class ResultRow:
    person_id: int
    rank: int
    name: str
    institution: str | None
    country: str | None
    fit: float
    faculty_score: float | None
    faculty_confidence: str | None
    email: str | None
    email_confidence: str | None
    email_source: str | None
    sparse: bool
    reason: str | None

    def as_dict(self) -> dict:
        return asdict(self)


def query_results(
    conn: sqlite3.Connection,
    profile_id: str,
    min_score: float = 0.0,
    faculty_only: bool = False,
    country: str | None = None,
    limit: int | None = None,
) -> list[ResultRow]:
    """Read the ranking. `limit` is display-only and never affects what is stored."""
    clauses = ["m.profile_id = ?", "COALESCE(m.stage2_score, m.stage1_score) >= ?"]
    params: list = [profile_id, min_score]
    if faculty_only:
        clauses.append("p.faculty_confidence IN ('high','medium')")
    if country:
        clauses.append("i.country = ?")
        params.append(country.upper())

    sql = f"""
        SELECT m.person_id, p.name, i.name AS institution, i.country,
               COALESCE(m.stage2_score, m.stage1_score) AS fit,
               p.faculty_score, p.faculty_confidence,
               p.email, p.email_confidence, p.email_source, m.sparse, m.reason
        FROM matches m
        JOIN people p ON p.id = m.person_id
        LEFT JOIN institutions i ON i.id = p.institution_id
        WHERE {' AND '.join(clauses)}
        ORDER BY fit DESC, p.faculty_score DESC
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    return [
        ResultRow(
            person_id=row["person_id"],
            rank=index,
            name=row["name"],
            institution=row["institution"],
            country=row["country"],
            fit=round(row["fit"], 4),
            faculty_score=row["faculty_score"],
            faculty_confidence=row["faculty_confidence"],
            email=row["email"],
            email_confidence=row["email_confidence"],
            email_source=row["email_source"],
            sparse=bool(row["sparse"]),
            reason=row["reason"],
        )
        for index, row in enumerate(conn.execute(sql, params).fetchall(), start=1)
    ]


def render_table(rows: list[ResultRow]) -> Table:
    """Fit and faculty stay in separate columns — they are never blended.

    `src` is where the address came from — crossref, orcid, adapter or
    crawler. Those differ enormously in trustworthiness (a Crossref address
    the author themselves published, versus one scraped off a page), and an
    address with no visible provenance invites equal trust in all of them.
    """
    table = Table(title="gradpath results")
    for header in ("#", "name", "institution", "cc", "fit", "faculty", "email", "src"):
        table.add_column(header)
    for row in rows:
        marker = " *sparse" if row.sparse else ""
        table.add_row(
            str(row.rank),
            row.name + marker,
            row.institution or "-",
            row.country or "-",
            f"{row.fit:.3f}",
            f"{row.faculty_confidence or '-'}",
            row.email or "(not found)",
            row.email_source or "-",
        )
    return table
