"""The single-network-chokepoint constraint is load-bearing, so it is tested."""
import pathlib

FORBIDDEN = ("import httpx", "import requests", "from httpx", "from requests",
             "import urllib.request")
ROOT = pathlib.Path(__file__).resolve().parents[1] / "gradpath"


def test_only_net_module_imports_an_http_client():
    offenders = []
    for path in ROOT.rglob("*.py"):
        if path.parent.name == "net":
            continue
        text = path.read_text()
        if any(token in text for token in FORBIDDEN):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], f"HTTP client imported outside net/: {offenders}"
