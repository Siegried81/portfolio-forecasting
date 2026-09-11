"""Unit tests for src/http_utils.py's safe_get_json. Fully mocked."""
import requests

import src.http_utils as http_utils
from src.http_utils import safe_get_json


class _FakeResponse:
    def __init__(self, json_data: dict, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._json_data


def test_safe_get_json_returns_the_parsed_body_on_success(monkeypatch):
    monkeypatch.setattr(http_utils.requests, "get", lambda *a, **k: _FakeResponse({"key": "value"}))
    assert safe_get_json("http://example.com") == {"key": "value"}


def test_safe_get_json_passes_params_through(monkeypatch):
    captured = {}

    def _fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse({})

    monkeypatch.setattr(http_utils.requests, "get", _fake_get)
    safe_get_json("http://example.com", params={"q": "test"})
    assert captured["url"] == "http://example.com"
    assert captured["params"] == {"q": "test"}


def test_safe_get_json_returns_none_on_request_exception(monkeypatch):
    def _raise(*a, **k):
        raise requests.RequestException("network down")
    monkeypatch.setattr(http_utils.requests, "get", _raise)
    assert safe_get_json("http://example.com") is None


def test_safe_get_json_returns_none_on_http_error_status(monkeypatch):
    monkeypatch.setattr(http_utils.requests, "get", lambda *a, **k: _FakeResponse({}, status_code=500))
    assert safe_get_json("http://example.com") is None


def test_safe_get_json_returns_none_on_invalid_json(monkeypatch):
    class _BadJsonResponse:
        def raise_for_status(self):
            pass

        def json(self):
            raise ValueError("not valid json")

    monkeypatch.setattr(http_utils.requests, "get", lambda *a, **k: _BadJsonResponse())
    assert safe_get_json("http://example.com") is None


def test_safe_get_json_never_raises_even_with_a_context_label(monkeypatch):
    def _raise(*a, **k):
        raise requests.RequestException("down")
    monkeypatch.setattr(http_utils.requests, "get", _raise)
    assert safe_get_json("http://example.com", context="DGS10") is None