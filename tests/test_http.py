from unittest.mock import Mock

import httpx
import pytest
import respx

from gradpath.models import Settings
from gradpath.net.http import PoliteClient

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
