import pytest

from gradpath.config import (
    load_fields, load_profile, load_settings, resolve_field, scaffold_profile,
)

SETTINGS = """
embedding_model: sentence-transformers/all-MiniLM-L6-v2
embedding_batch_size: 256
rate_limit_per_host: 1.0
default_since_year: 2021
show_min_score: 0.65
rerank_model: claude-opus-5
rerank_default_budget_usd: 2.00
cache_dir: .cache
"""

PROFILE = """
name: default
contact_email: me@example.com
fields: [efficient-ml]
countries: [KR, JP]
interests: on-device inference
keywords: [quantization]
seed_papers: ['10.1145/3458864.3467882']
my_papers: []
"""

FIELDS = """
efficient-ml:
  label: Efficient machine learning
  topics: [T10028, T11689]
  aliases: [edge-ml, on-device-ml]
"""


def test_loads_settings(tmp_path):
    p = tmp_path / "settings.yaml"
    p.write_text(SETTINGS)
    s = load_settings(p)
    assert s.embedding_batch_size == 256
    assert s.show_min_score == 0.65


def test_rate_limit_may_not_be_raised(tmp_path):
    p = tmp_path / "settings.yaml"
    p.write_text(SETTINGS.replace("rate_limit_per_host: 1.0", "rate_limit_per_host: 20.0"))
    with pytest.raises(ValueError, match="rate_limit_per_host"):
        load_settings(p)


def test_profile_requires_contact_email(tmp_path):
    p = tmp_path / "profile.yaml"
    p.write_text(PROFILE.replace("contact_email: me@example.com", "contact_email: ''"))
    with pytest.raises(ValueError, match="contact_email"):
        load_profile(p)


def test_profile_requires_at_least_one_interest_input(tmp_path):
    p = tmp_path / "profile.yaml"
    p.write_text(
        "name: d\ncontact_email: me@example.com\ninterests: ''\n"
        "keywords: []\nseed_papers: []\nmy_papers: []\n"
    )
    with pytest.raises(ValueError, match="interests"):
        load_profile(p)


def test_scaffold_then_load_roundtrips(tmp_path):
    p = tmp_path / "profile.yaml"
    scaffold_profile(p)
    assert p.exists()
    assert "contact_email" in p.read_text()


def test_resolve_field_by_alias(tmp_path):
    p = tmp_path / "fields.yaml"
    p.write_text(FIELDS)
    fields = load_fields(p)
    assert resolve_field("edge-ml", fields).topics == ["T10028", "T11689"]
    assert resolve_field("efficient-ml", fields).label == "Efficient machine learning"


def test_resolve_unknown_field_raises(tmp_path):
    p = tmp_path / "fields.yaml"
    p.write_text(FIELDS)
    with pytest.raises(KeyError, match="nonsense"):
        resolve_field("nonsense", load_fields(p))
