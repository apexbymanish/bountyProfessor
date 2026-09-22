from __future__ import annotations

import csv
from pathlib import Path

from gradpath.report.table import FIELDNAMES, ResultRow


def to_csv(rows: list[ResultRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def to_markdown(rows: list[ResultRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(FIELDNAMES) + " |"
    divider = "|" + "|".join(["---"] * len(FIELDNAMES)) + "|"
    lines = [header, divider]
    for row in rows:
        values = row.as_dict()
        lines.append("| " + " | ".join(str(values[f] if values[f] is not None else "")
                                       for f in FIELDNAMES) + " |")
    path.write_text("\n".join(lines) + "\n")
