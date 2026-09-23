from __future__ import annotations

from pathlib import Path

import yaml

from gradpath.models import FieldDef, InstitutionSeed, Profile, Settings
from gradpath.util import slugify

MAX_RATE_LIMIT = 1.0

PROFILE_TEMPLATE = """\
# gradpath profile — this file is gitignored, keep it that way.
name: default
contact_email: ""          # REQUIRED: used in the polite User-Agent
fields: []                 # e.g. [efficient-ml]
countries: []              # e.g. [KR, JP, DE]

interests: >
  Describe in a paragraph what you want to work on.

keywords: []
seed_papers: []            # DOIs of papers you admire — the strongest signal
my_papers: []              # DOIs of your own papers — weighted the same as seed_papers
"""

# `init` writes these when the file is absent, so a fresh --root is a workspace
# every other command can run in. They are the defaults that ship in the repo;
# an existing file is never touched, so a workspace someone has tuned survives
# a re-run of `init` unchanged.
SETTINGS_TEMPLATE = """\
embedding_model: sentence-transformers/all-MiniLM-L6-v2
embedding_batch_size: 256
rate_limit_per_host: 1.0   # may be lowered, never raised — this tool stays polite
default_since_year: 2021
show_min_score: 0.65
rerank_model: claude-opus-5
rerank_default_budget_usd: 2.00
cache_dir: .cache
"""

FIELDS_TEMPLATE = """\
# Research fields, mapped to OpenAlex topic ids.
# Run `gradpath fields search "<your area>"` to find the ids for a new one.
efficient-ml:
  label: Efficient machine learning
  topics: [T10028, T11689]
  aliases: [edge-ml, on-device-ml]
"""

INSTITUTIONS_TEMPLATE = """\
# Curated institutions and their faculty-directory adapters. Ranks are never
# written here -- build a tier with `gradpath institutions top` or
# `gradpath institutions import` instead.
kaist:
  name: Korea Advanced Institute of Science and Technology
  country: KR
  site: https://www.kaist.ac.kr
  adapter: kaist

gist:
  name: Gwangju Institute of Science and Technology
  country: KR
  site: https://www.gist.ac.kr
  adapter: gist

snu:
  name: Seoul National University
  country: KR
  site: https://www.snu.ac.kr
  adapter: snu
"""


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return yaml.safe_load(path.read_text()) or {}


def load_settings(path: Path) -> Settings:
    raw = _read_yaml(path)
    rate = float(raw.get("rate_limit_per_host", MAX_RATE_LIMIT))
    if rate > MAX_RATE_LIMIT:
        raise ValueError(
            f"rate_limit_per_host may be lowered but not raised above {MAX_RATE_LIMIT}"
        )
    return Settings(
        embedding_model=raw["embedding_model"],
        embedding_batch_size=int(raw["embedding_batch_size"]),
        rate_limit_per_host=rate,
        default_since_year=int(raw["default_since_year"]),
        show_min_score=float(raw["show_min_score"]),
        rerank_model=raw["rerank_model"],
        rerank_default_budget_usd=float(raw["rerank_default_budget_usd"]),
        cache_dir=raw.get("cache_dir", ".cache"),
    )


def load_profile(path: Path) -> Profile:
    raw = _read_yaml(path)
    contact = (raw.get("contact_email") or "").strip()
    if not contact:
        raise ValueError(
            "contact_email is required in profile.yaml — it identifies you to the "
            "APIs and sites this tool contacts"
        )
    interests = (raw.get("interests") or "").strip()
    keywords = raw.get("keywords") or []
    seed = raw.get("seed_papers") or []
    mine = raw.get("my_papers") or []
    if not (interests or keywords or seed or mine):
        raise ValueError(
            "profile needs at least one of: interests, keywords, seed_papers, my_papers"
        )
    return Profile(
        name=raw.get("name", "default"),
        contact_email=contact,
        interests=interests,
        keywords=list(keywords),
        seed_papers=list(seed),
        my_papers=list(mine),
        fields=list(raw.get("fields") or []),
        countries=[c.upper() for c in (raw.get("countries") or [])],
    )


def scaffold_profile(path: Path) -> None:
    """Generate a profile template, raising FileExistsError if the file already exists."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PROFILE_TEMPLATE)


def scaffold_file(path: Path, content: str) -> bool:
    """Write `content` to `path` only if nothing is there. Returns True if written.

    Never overwrites: a workspace file the user has edited is theirs, and
    `init` is expected to be safe to re-run.
    """
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


def load_fields(path: Path) -> dict[str, FieldDef]:
    raw = _read_yaml(path)
    return {
        slugify(slug): FieldDef(
            slug=slugify(slug),
            label=body["label"],
            topics=list(body["topics"]),
            aliases=list(body.get("aliases") or []),
        )
        for slug, body in raw.items()
    }


def resolve_field(name: str, fields: dict[str, FieldDef]) -> FieldDef:
    key = name.strip().lower()
    if key in fields:
        return fields[key]
    for definition in fields.values():
        if key in {a.lower() for a in definition.aliases}:
            return definition
    raise KeyError(
        f"unknown field {name!r} — run `gradpath fields search` to find its OpenAlex topics"
    )


def load_institutions(path: Path) -> dict[str, InstitutionSeed]:
    raw = _read_yaml(path)
    return {
        slugify(slug): InstitutionSeed(
            slug=slugify(slug),
            name=body["name"],
            country=body.get("country"),
            site=body.get("site"),
            adapter=body.get("adapter"),
        )
        for slug, body in raw.items()
    }
