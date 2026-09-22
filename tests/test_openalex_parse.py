import httpx
import respx

from gradpath.sources.openalex import parse_work, reconstruct_abstract, search_topics

WORK = {
    "id": "https://openalex.org/W123",
    "title": "Quantized inference on mobile devices",
    "publication_year": 2024,
    "doi": "https://doi.org/10.1145/1234",
    "cited_by_count": 17,
    "primary_location": {"source": {"display_name": "NeurIPS"}},
    "topics": [{"id": "https://openalex.org/T10028"}],
    "abstract_inverted_index": {"We": [0], "quantize": [1], "models": [2]},
    "authorships": [
        {
            "author_position": "first",
            "author": {"id": "https://openalex.org/A1", "display_name": "J. Kim",
                       "orcid": "https://orcid.org/0000-0001-0000-0001"},
            "institutions": [{"id": "https://openalex.org/I1",
                              "display_name": "KAIST", "country_code": "KR"}],
        },
        {
            "author_position": "last",
            "author": {"id": "https://openalex.org/A2", "display_name": "S. Park",
                       "orcid": None},
            "institutions": [],
        },
    ],
}


def test_reconstruct_abstract_orders_words_by_position():
    assert reconstruct_abstract({"b": [1], "a": [0], "c": [2]}) == "a b c"


def test_reconstruct_abstract_handles_repeated_words():
    assert reconstruct_abstract({"the": [0, 2], "cat": [1]}) == "the cat the"


def test_reconstruct_abstract_returns_none_when_absent():
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None


def test_parse_work_extracts_core_fields():
    w = parse_work(WORK)
    assert w.openalex_id == "W123"
    assert w.abstract == "We quantize models"
    assert w.year == 2024
    assert w.doi == "10.1145/1234"
    assert w.venue == "NeurIPS"
    assert w.topics == ["T10028"]
    assert w.cited_by == 17


def test_parse_work_extracts_authorships_with_position_and_institution():
    w = parse_work(WORK)
    assert [a.author_id for a in w.authorships] == ["A1", "A2"]
    first = w.authorships[0]
    assert first.position == "first"
    assert first.institution_openalex_id == "I1"
    assert first.institution_country == "KR"
    assert first.orcid == "0000-0001-0000-0001"
    assert w.authorships[1].institution_openalex_id is None


@respx.mock
def test_search_topics_returns_id_and_label(tmp_path):
    from gradpath.models import Settings
    from gradpath.net.http import PoliteClient

    settings = Settings("m", 8, 1000.0, 2021, 0.6, "claude-opus-5", 1.0, ".cache")
    client = PoliteClient(settings, "me@example.com", tmp_path / "c")
    respx.get("https://api.openalex.org/topics").mock(
        return_value=httpx.Response(200, json={"results": [
            {"id": "https://openalex.org/T10028",
             "display_name": "Efficient ML", "works_count": 4211},
        ]})
    )
    hits = search_topics(client, "efficient machine learning")
    assert hits[0].topic_id == "T10028"
    assert hits[0].label == "Efficient ML"
    assert hits[0].works_count == 4211
