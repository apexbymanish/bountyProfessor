import httpx
import pytest
import respx

from gradpath.models import Settings
from gradpath.net.http import PoliteClient
from gradpath.sources.crossref import crossref_email
from gradpath.sources.orcid import orcid_record

SETTINGS = Settings("m", 8, 1000.0, 2021, 0.6, "claude-opus-5", 1.0, ".cache")


@pytest.fixture
def client(tmp_path):
    return PoliteClient(SETTINGS, "me@example.com", tmp_path / "c")


@respx.mock
def test_crossref_returns_matching_author_email(client):
    respx.get("https://api.crossref.org/works/10.1145/1234").mock(
        return_value=httpx.Response(200, json={"message": {"author": [
            {"given": "Sunmi", "family": "Park", "email": "sunmi@kaist.ac.kr"},
            {"given": "Other", "family": "Person"},
        ]}})
    )
    assert crossref_email(client, "10.1145/1234", "Sunmi Park") == "sunmi@kaist.ac.kr"


@respx.mock
def test_crossref_returns_none_when_name_does_not_match(client):
    respx.get("https://api.crossref.org/works/10.1145/1234").mock(
        return_value=httpx.Response(200, json={"message": {"author": [
            {"given": "Someone", "family": "Else", "email": "else@x.ac.kr"},
        ]}})
    )
    assert crossref_email(client, "10.1145/1234", "Sunmi Park") is None


@respx.mock
def test_crossref_returns_none_on_missing_record(client):
    respx.get("https://api.crossref.org/works/10.1145/9999").mock(
        return_value=httpx.Response(404, json={})
    )
    assert crossref_email(client, "10.1145/9999", "Anyone") is None


@respx.mock
def test_orcid_extracts_email_and_employment_title(client):
    respx.get("https://pub.orcid.org/v3.0/0000-0001-0000-0001/record").mock(
        return_value=httpx.Response(200, json={
            "person": {"emails": {"email": [{"email": "p@kaist.ac.kr"}]}},
            "activities-summary": {"employments": {"affiliation-group": [
                {"summaries": [{"employment-summary": {"role-title": "Associate Professor"}}]}
            ]}},
        })
    )
    record = orcid_record(client, "0000-0001-0000-0001")
    assert record.email == "p@kaist.ac.kr"
    assert record.title == "Associate Professor"


@respx.mock
def test_orcid_handles_record_with_no_public_email(client):
    respx.get("https://pub.orcid.org/v3.0/0000-0002/record").mock(
        return_value=httpx.Response(200, json={
            "person": {"emails": {"email": []}},
            "activities-summary": {"employments": {"affiliation-group": []}},
        })
    )
    record = orcid_record(client, "0000-0002")
    assert record.email is None
    assert record.title is None
