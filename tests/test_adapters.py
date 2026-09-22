import pathlib

import pytest

from gradpath.sources.adapters import all_adapters, get_adapter
from gradpath.sources.adapters.base import InstitutionAdapter

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("slug", ["kaist", "gist", "snu"])
def test_adapter_is_registered(slug):
    assert get_adapter(slug) is not None


@pytest.mark.parametrize("adapter_cls", all_adapters(), ids=lambda c: c.slug)
def test_adapter_satisfies_contract(adapter_cls):
    """One contract suite over every adapter — this is what makes community PRs safe."""
    assert issubclass(adapter_cls, InstitutionAdapter)
    adapter = adapter_cls()
    assert adapter_cls.slug
    assert adapter_cls.domains
    urls = list(adapter.faculty_urls())
    assert urls and all(u.startswith("https://") for u in urls)
    assert adapter.parse_faculty("<html></html>", urls[0]) == []


def test_kaist_adapter_parses_names_titles_and_emails():
    adapter = get_adapter("kaist")()
    html = (FIXTURES / "kaist_faculty.html").read_text()
    records = adapter.parse_faculty(html, "https://cs.kaist.ac.kr/people")
    assert [r.name for r in records] == ["Sunmi Park", "Jaewon Kim"]
    assert records[0].email == "park@kaist.ac.kr"
    assert records[0].title == "Associate Professor"


def test_kaist_adapter_decodes_obfuscated_address():
    adapter = get_adapter("kaist")()
    html = (FIXTURES / "kaist_faculty.html").read_text()
    records = adapter.parse_faculty(html, "https://cs.kaist.ac.kr/people")
    assert records[1].email == "kim@kaist.ac.kr"


def test_kaist_adapter_resolves_relative_homepage_to_absolute():
    adapter = get_adapter("kaist")()
    html = (FIXTURES / "kaist_faculty.html").read_text()
    records = adapter.parse_faculty(html, "https://cs.kaist.ac.kr/people")
    assert records[0].homepage == "https://cs.kaist.ac.kr/people/park"


def test_gist_adapter_parses_names_titles_and_emails():
    adapter = get_adapter("gist")()
    html = (FIXTURES / "gist_faculty.html").read_text()
    records = adapter.parse_faculty(html, "https://cse.gist.ac.kr/people")
    assert [r.name for r in records] == ["Jihye Lee", "Minho Choi"]
    assert records[0].email == "lee@gist.ac.kr"
    assert records[0].title == "Assistant Professor"


def test_gist_adapter_decodes_obfuscated_address():
    adapter = get_adapter("gist")()
    html = (FIXTURES / "gist_faculty.html").read_text()
    records = adapter.parse_faculty(html, "https://cse.gist.ac.kr/people")
    assert records[1].email == "choi@gist.ac.kr"


def test_gist_adapter_resolves_relative_homepage_to_absolute():
    adapter = get_adapter("gist")()
    html = (FIXTURES / "gist_faculty.html").read_text()
    records = adapter.parse_faculty(html, "https://cse.gist.ac.kr/people")
    assert records[0].homepage == "https://cse.gist.ac.kr/faculty/lee"


def test_snu_adapter_parses_names_titles_and_emails():
    adapter = get_adapter("snu")()
    html = (FIXTURES / "snu_faculty.html").read_text()
    records = adapter.parse_faculty(html, "https://cse.snu.ac.kr/people")
    assert [r.name for r in records] == ["Soojin Han", "Taeyang Yoon"]
    assert records[0].email == "han@snu.ac.kr"
    assert records[0].title == "Associate Professor"


def test_snu_adapter_decodes_obfuscated_address():
    adapter = get_adapter("snu")()
    html = (FIXTURES / "snu_faculty.html").read_text()
    records = adapter.parse_faculty(html, "https://cse.snu.ac.kr/people")
    assert records[1].email == "yoon@snu.ac.kr"


def test_snu_adapter_resolves_relative_homepage_to_absolute():
    adapter = get_adapter("snu")()
    html = (FIXTURES / "snu_faculty.html").read_text()
    records = adapter.parse_faculty(html, "https://cse.snu.ac.kr/people")
    assert records[0].homepage == "https://cse.snu.ac.kr/professor/han"
