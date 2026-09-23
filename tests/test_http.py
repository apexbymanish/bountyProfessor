from unittest.mock import Mock

import httpx
import pytest
import respx

from gradpath.models import Settings
from gradpath.net.http import HostBlocked, PoliteClient

SETTINGS = Settings(
    embedding_model="m", embedding_batch_size=8, rate_limit_per_host=1000.0,
    default_since_year=2021, show_min_score=0.6, rerank_model="claude-opus-5",
    rerank_default_budget_usd=1.0, cache_dir=".cache",
)
ROBOTS_ALLOW = "User-agent: *\nAllow: /\n"
ROBOTS_DENY = "User-agent: *\nDisallow: /private\n"


@pytest.fixture
def client(tmp_path):
    # rate limit relaxed in tests so the suite does not sleep
    return PoliteClient(SETTINGS, "me@example.com", tmp_path / "cache")


@respx.mock
def test_user_agent_carries_contact_email(client):
    route = respx.get("https://api.example.com/x").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client.get_json("https://api.example.com/x")
    assert "me@example.com" in route.calls[0].request.headers["user-agent"]


@respx.mock
def test_second_identical_request_is_served_from_cache(client):
    route = respx.get("https://api.example.com/y").mock(
        return_value=httpx.Response(200, json={"n": 1})
    )
    assert client.get_json("https://api.example.com/y") == {"n": 1}
    assert client.get_json("https://api.example.com/y") == {"n": 1}
    assert route.call_count == 1


@respx.mock
def test_robots_disallow_returns_none(client):
    respx.get("https://site.example.com/robots.txt").mock(
        return_value=httpx.Response(200, text=ROBOTS_DENY)
    )
    assert client.get_text("https://site.example.com/private/staff") is None


@respx.mock
def test_robots_allow_permits_fetch(client):
    respx.get("https://site.example.com/robots.txt").mock(
        return_value=httpx.Response(200, text=ROBOTS_ALLOW)
    )
    respx.get("https://site.example.com/staff").mock(
        return_value=httpx.Response(200, text="<html>hi</html>")
    )
    assert client.get_text("https://site.example.com/staff") == "<html>hi</html>"


@respx.mock
def test_missing_robots_is_treated_as_allowed(client):
    respx.get("https://site2.example.com/robots.txt").mock(
        return_value=httpx.Response(404)
    )
    respx.get("https://site2.example.com/staff").mock(
        return_value=httpx.Response(200, text="ok")
    )
    assert client.get_text("https://site2.example.com/staff") == "ok"


@respx.mock
def test_retries_on_server_error_then_succeeds(client, monkeypatch):
    # No real wall-clock cost: the backoff sleep is stubbed out so this test
    # exercises the retry path (503 then 200) without pausing the suite.
    monkeypatch.setattr("gradpath.net.http.time.sleep", lambda _seconds: None)
    respx.get("https://api.example.com/z").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json={"ok": 1})]
    )
    assert client.get_json("https://api.example.com/z") == {"ok": 1}


@respx.mock
def test_robots_fetch_is_paced_through_rate_limiter(client):
    respx.get("https://site3.example.com/robots.txt").mock(
        return_value=httpx.Response(200, text=ROBOTS_ALLOW)
    )
    client._limiter.wait = Mock(wraps=client._limiter.wait)
    client.allowed("https://site3.example.com/staff")
    client._limiter.wait.assert_called_once_with("site3.example.com")


@respx.mock
def test_robots_fetch_failure_degrades_to_allowed(client):
    # A network-level failure fetching robots.txt must never disable content
    # fetching for the host -- it degrades to "allowed", exactly like an
    # absent (404) robots.txt does.
    respx.get("https://site4.example.com/robots.txt").mock(
        side_effect=httpx.ConnectError("boom")
    )
    respx.get("https://site4.example.com/staff").mock(
        return_value=httpx.Response(200, text="ok")
    )
    assert client.get_text("https://site4.example.com/staff") == "ok"


@respx.mock
def test_retry_after_http_date_falls_back_to_default_backoff(client, monkeypatch):
    # Retry-After may legally be an HTTP-date (RFC 7231), not just a number
    # of seconds. That must degrade to the default backoff rather than
    # raising ValueError out of the retry path.
    monkeypatch.setattr("gradpath.net.http.time.sleep", lambda _seconds: None)
    respx.get("https://api.example.com/w429").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
            httpx.Response(200, json={"ok": 1}),
        ]
    )
    assert client.get_json("https://api.example.com/w429") == {"ok": 1}


@respx.mock
def test_non_json_body_returns_empty_dict_instead_of_raising(client):
    # Crossref answers an unknown DOI with a plain-text 404 body, not JSON.
    # get_json must degrade to {} -- the same shape callers already treat as
    # "no data" -- rather than letting json.JSONDecodeError escape.
    respx.get("https://api.crossref.org/works/10.9999/not-a-real-doi").mock(
        return_value=httpx.Response(404, text="Resource not found.")
    )
    assert client.get_json("https://api.crossref.org/works/10.9999/not-a-real-doi") == {}


@respx.mock
def test_non_json_body_is_not_cached(client):
    # Caching an unparseable body would turn one transient upstream 404 into a
    # permanent empty result for that URL until the cache is cleared by hand.
    route = respx.get("https://api.crossref.org/works/10.9999/not-a-real-doi").mock(
        return_value=httpx.Response(404, text="Resource not found.")
    )
    client.get_json("https://api.crossref.org/works/10.9999/not-a-real-doi")
    client.get_json("https://api.crossref.org/works/10.9999/not-a-real-doi")
    assert route.call_count == 2


@respx.mock
def test_circuit_breaks_host_after_repeated_failures(client, monkeypatch):
    monkeypatch.setattr("gradpath.net.http.time.sleep", lambda _seconds: None)
    route = respx.get("https://api.example.com/broken").mock(
        return_value=httpx.Response(503)
    )
    # Each call exhausts MAX_ATTEMPTS retries and counts as one failure
    # against the host; CIRCUIT_BREAK_FAILURES=5 such calls trip the breaker.
    for _ in range(5):
        with pytest.raises(HostBlocked):
            client.get_json("https://api.example.com/broken")
    calls_before_trip = route.call_count
    with pytest.raises(HostBlocked):
        client.get_json("https://api.example.com/broken")
    # The tripped breaker must short-circuit before making any request.
    assert route.call_count == calls_before_trip
