from urllib.parse import urlparse

import pytest

from gradpath.config import (
    load_fields,
    load_institutions,
    load_profile,
    load_settings,
    resolve_field,
    scaffold_profile,
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

    # Fill in required fields and load it
    profile_content = """name: default
contact_email: test@example.com
fields: []
countries: []
interests: test research area
keywords: []
seed_papers: []
my_papers: []
"""
    p.write_text(profile_content)

    profile = load_profile(p)
    assert profile.contact_email == "test@example.com"
    assert profile.interests == "test research area"


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


def test_resolve_field_with_mixed_case_slug(tmp_path):
    mixed_case_fields = """
Efficient-ML:
  label: Efficient machine learning
  topics: [T10028, T11689]
  aliases: [edge-ml, on-device-ml]
"""
    p = tmp_path / "fields.yaml"
    p.write_text(mixed_case_fields)
    fields = load_fields(p)
    # The mixed-case key is normalized to lowercase
    assert resolve_field("Efficient-ML", fields).label == "Efficient machine learning"
    assert resolve_field("efficient-ml", fields).label == "Efficient machine learning"


def test_load_institutions_with_adapters(tmp_path):
    p = tmp_path / "institutions.yaml"
    p.write_text("""
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
""")
    institutions = load_institutions(p)
    assert "kaist" in institutions
    assert institutions["kaist"].adapter == "kaist"
    assert institutions["gist"].adapter == "gist"


def test_load_institutions_sites_are_absolute_urls(tmp_path):
    p = tmp_path / "institutions.yaml"
    p.write_text("""
kaist:
  name: Korea Advanced Institute of Science and Technology
  country: KR
  site: https://www.kaist.ac.kr
  adapter: kaist
""")
    institutions = load_institutions(p)
    site = institutions["kaist"].site
    parsed = urlparse(site)
    assert parsed.scheme != ""
    assert parsed.netloc != ""
    assert site.startswith("https://")


def test_load_institutions_optional_fields_become_none(tmp_path):
    p = tmp_path / "institutions.yaml"
    p.write_text("""
minimal:
  name: Minimal Institution
""")
    institutions = load_institutions(p)
    assert "minimal" in institutions
    assert institutions["minimal"].country is None
    assert institutions["minimal"].site is None
    assert institutions["minimal"].adapter is None
