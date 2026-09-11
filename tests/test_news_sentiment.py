"""Unit tests for the news-sentiment cascade in src/news_data.py — priority on fallback behavior and the circuit breaker."""
import dataclasses

import pytest
import requests
import streamlit as st

import src.news_data as news_data
from src.news_data import (
    _trip_finnhub_sentiment_breaker_if_permanent,
    compute_local_sentiment,
    fetch_finbert_sentiment,
    fetch_finnhub_sentiment,
    get_ticker_sentiment,
)


@pytest.fixture(autouse=True)
def _reset_finnhub_sentiment_state(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setattr(news_data, "_finnhub_sentiment_plan_restricted", False)
    fake_settings = dataclasses.replace(
        news_data.LLM_SETTINGS, finnhub_api_key="fake-test-key", huggingface_api_key="fake-hf-key",
    )
    monkeypatch.setattr(news_data, "LLM_SETTINGS", fake_settings)
    yield
    st.cache_data.clear()


class _FakeResponse:
    def __init__(self, status_code: int, json_data: dict | None = None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} error")
            error.response = self  # type: ignore[assignment]
            raise error

    def json(self):
        return self._json_data


# ---------------------------------------------------------------------------
# compute_local_sentiment (VADER)
# ---------------------------------------------------------------------------

def test_compute_local_sentiment_none_when_no_articles():
    assert compute_local_sentiment([]) is None


def test_compute_local_sentiment_positive_for_clearly_positive_headlines():
    articles = [
        {"title": "Company smashes earnings expectations, stock soars", "description": "Record profits and a raised outlook."},
        {"title": "Analysts upgrade rating after outstanding quarter", "description": "Strong growth across every segment."},
    ]
    result = compute_local_sentiment(articles)
    assert result is not None
    assert result["provider"].startswith("VADER")
    assert result["score"] > 0.1
    assert result["n_articles"] == 2


def test_compute_local_sentiment_negative_for_clearly_negative_headlines():
    articles = [
        {"title": "Company misses estimates, shares plunge", "description": "Disappointing results and a bleak outlook."},
        {"title": "Regulators launch investigation into fraud allegations", "description": "Shares tumble on the news."},
    ]
    result = compute_local_sentiment(articles)
    assert result is not None
    assert result["score"] < -0.1


# ---------------------------------------------------------------------------
# get_ticker_sentiment — FinBERT -> Finnhub -> VADER -> None
# ---------------------------------------------------------------------------

def test_get_ticker_sentiment_prefers_finbert_when_available(monkeypatch):
    fake_finbert_result = {"provider": "FinBERT (financial-domain BERT, via Hugging Face)", "score": 0.4,
                            "bullish_pct": 70, "bearish_pct": 30, "n_articles": 5}
    monkeypatch.setattr(news_data, "fetch_finbert_sentiment", lambda articles: fake_finbert_result)
    monkeypatch.setattr(news_data, "fetch_finnhub_sentiment", lambda ticker: pytest.fail("Finnhub should not be called when FinBERT succeeds"))
    result = get_ticker_sentiment("AAPL", articles=[{"title": "irrelevant, should not be used"}])
    assert result == fake_finbert_result


def test_get_ticker_sentiment_falls_back_to_finnhub_when_finbert_unavailable(monkeypatch):
    monkeypatch.setattr(news_data, "fetch_finbert_sentiment", lambda articles: None)
    fake_finnhub_result = {"provider": "Finnhub (aggregated)", "score": 0.4, "bullish_pct": 70, "bearish_pct": 30, "n_articles": 50}
    monkeypatch.setattr(news_data, "fetch_finnhub_sentiment", lambda ticker: fake_finnhub_result)
    result = get_ticker_sentiment("AAPL", articles=[{"title": "irrelevant, should not be used"}])
    assert result == fake_finnhub_result


def test_get_ticker_sentiment_falls_back_to_vader_when_finbert_and_finnhub_unavailable(monkeypatch):
    monkeypatch.setattr(news_data, "fetch_finbert_sentiment", lambda articles: None)
    monkeypatch.setattr(news_data, "fetch_finnhub_sentiment", lambda ticker: None)
    articles = [{"title": "Great news for shareholders as profits jump", "description": "A very strong quarter."}]
    result = get_ticker_sentiment("AAPL", articles)
    assert result is not None
    assert result["provider"].startswith("VADER")


def test_get_ticker_sentiment_none_when_every_source_has_nothing(monkeypatch):
    monkeypatch.setattr(news_data, "fetch_finbert_sentiment", lambda articles: None)
    monkeypatch.setattr(news_data, "fetch_finnhub_sentiment", lambda ticker: None)
    result = get_ticker_sentiment("AAPL", articles=[])
    assert result is None


# ---------------------------------------------------------------------------
# _trip_finnhub_sentiment_breaker_if_permanent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", [401, 403])
def test_trip_breaker_on_permanent_status_codes(monkeypatch, status):
    monkeypatch.setattr(news_data, "_finnhub_sentiment_plan_restricted", False)
    error = requests.HTTPError(f"{status} error")
    error.response = _FakeResponse(status)  # type: ignore[assignment]
    _trip_finnhub_sentiment_breaker_if_permanent(error)
    assert news_data._finnhub_sentiment_plan_restricted is True


@pytest.mark.parametrize("status", [429, 500, 503])
def test_does_not_trip_breaker_on_transient_status_codes(monkeypatch, status):
    monkeypatch.setattr(news_data, "_finnhub_sentiment_plan_restricted", False)
    error = requests.HTTPError(f"{status} error")
    error.response = _FakeResponse(status)  # type: ignore[assignment]
    _trip_finnhub_sentiment_breaker_if_permanent(error)
    assert news_data._finnhub_sentiment_plan_restricted is False


def test_does_not_trip_breaker_when_response_is_missing():
    error = requests.HTTPError("no response object")
    _trip_finnhub_sentiment_breaker_if_permanent(error)
    assert news_data._finnhub_sentiment_plan_restricted is False


# ---------------------------------------------------------------------------
# fetch_finnhub_sentiment — circuit breaker end to end
# ---------------------------------------------------------------------------

def test_fetch_finnhub_sentiment_trips_breaker_on_403_and_skips_later_tickers(monkeypatch):
    call_count = {"n": 0}

    def _fake_get(*a, **k):
        call_count["n"] += 1
        return _FakeResponse(403)

    monkeypatch.setattr(news_data.requests, "get", _fake_get)

    first = fetch_finnhub_sentiment("AAPL")
    assert first is None
    assert call_count["n"] == 1

    second = fetch_finnhub_sentiment("MSFT")
    assert second is None
    assert call_count["n"] == 1


def test_fetch_finnhub_sentiment_does_not_trip_breaker_on_rate_limit(monkeypatch):
    call_count = {"n": 0}

    def _fake_get(*a, **k):
        call_count["n"] += 1
        return _FakeResponse(429)

    monkeypatch.setattr(news_data.requests, "get", _fake_get)

    fetch_finnhub_sentiment("AAPL")
    fetch_finnhub_sentiment("MSFT")
    assert call_count["n"] == 2


def test_fetch_finnhub_sentiment_skips_network_call_without_a_key(monkeypatch):
    fake_settings = dataclasses.replace(news_data.LLM_SETTINGS, finnhub_api_key=None)
    monkeypatch.setattr(news_data, "LLM_SETTINGS", fake_settings)

    def _fail(*a, **k):
        pytest.fail("requests.get should not be called without a Finnhub key")

    monkeypatch.setattr(news_data.requests, "get", _fail)
    assert fetch_finnhub_sentiment("AAPL") is None


# ---------------------------------------------------------------------------
# fetch_finbert_sentiment
# ---------------------------------------------------------------------------

def _article(title="Company beats earnings", description="Strong quarter"):
    return {"title": title, "description": description}


class _FinbertResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._payload


def _finbert_classes(positive=0.1, negative=0.1, neutral=0.8):
    return [
        {"label": "positive", "score": positive},
        {"label": "negative", "score": negative},
        {"label": "neutral", "score": neutral},
    ]


def test_fetch_finbert_sentiment_returns_none_without_a_key(monkeypatch):
    fake_settings = dataclasses.replace(news_data.LLM_SETTINGS, huggingface_api_key=None)
    monkeypatch.setattr(news_data, "LLM_SETTINGS", fake_settings)

    def _fail(*a, **k):
        pytest.fail("requests.post should not be called without a Hugging Face key")

    monkeypatch.setattr(news_data.requests, "post", _fail)
    assert fetch_finbert_sentiment([_article()]) is None


def test_fetch_finbert_sentiment_returns_none_with_no_articles():
    assert fetch_finbert_sentiment([]) is None


def test_fetch_finbert_sentiment_computes_score_from_positive_negative_gap(monkeypatch):
    monkeypatch.setattr(
        news_data.requests, "post",
        lambda *a, **k: _FinbertResponse(_finbert_classes(positive=0.7, negative=0.1, neutral=0.2)),
    )
    result = fetch_finbert_sentiment([_article()])
    assert result is not None
    assert result["provider"].startswith("FinBERT")
    assert result["score"] == pytest.approx(0.6, abs=1e-9)
    assert result["n_articles"] == 1

def test_fetch_finbert_sentiment_unwraps_nested_list_response_shape(monkeypatch):
    # router.huggingface.co wraps a single-input result in an extra list
    # layer — [[{...}, ...]] instead of the older flat [{...}, ...].
    nested_payload = [_finbert_classes(positive=0.7, negative=0.1, neutral=0.2)]
    monkeypatch.setattr(
        news_data.requests, "post",
        lambda *a, **k: _FinbertResponse(nested_payload),
    )
    result = fetch_finbert_sentiment([_article()])
    assert result is not None
    assert result["score"] == pytest.approx(0.6, abs=1e-9)

def test_fetch_finbert_sentiment_averages_across_multiple_articles(monkeypatch):
    calls = {"n": 0}

    def _fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FinbertResponse(_finbert_classes(positive=0.9, negative=0.0, neutral=0.1))
        return _FinbertResponse(_finbert_classes(positive=0.0, negative=0.9, neutral=0.1))

    monkeypatch.setattr(news_data.requests, "post", _fake_post)
    result = fetch_finbert_sentiment([_article("Great quarter"), _article("Terrible quarter")])
    assert result["n_articles"] == 2
    assert result["score"] == pytest.approx(0.0, abs=1e-9)


def test_fetch_finbert_sentiment_respects_max_articles_cap(monkeypatch):
    calls = {"n": 0}

    def _fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        return _FinbertResponse(_finbert_classes())

    monkeypatch.setattr(news_data.requests, "post", _fake_post)
    many_articles = [_article(f"Headline {i}") for i in range(20)]
    fetch_finbert_sentiment(many_articles)
    assert calls["n"] == news_data.FINBERT_MAX_ARTICLES


def test_fetch_finbert_sentiment_skips_an_article_on_request_failure_but_keeps_going(monkeypatch):
    calls = {"n": 0}

    def _fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.RequestException("network blip")
        return _FinbertResponse(_finbert_classes(positive=0.5, negative=0.1, neutral=0.4))

    monkeypatch.setattr(news_data.requests, "post", _fake_post)
    result = fetch_finbert_sentiment([_article("first"), _article("second")])
    assert result is not None
    assert result["n_articles"] == 1


def test_fetch_finbert_sentiment_returns_none_when_every_article_fails(monkeypatch):
    monkeypatch.setattr(
        news_data.requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.RequestException("down")),
    )
    assert fetch_finbert_sentiment([_article()]) is None


def test_fetch_finbert_sentiment_handles_model_loading_cold_start_gracefully(monkeypatch):
    monkeypatch.setattr(
        news_data.requests, "post",
        lambda *a, **k: _FinbertResponse({"error": "Model ProsusAI/finbert is currently loading"}),
    )
    assert fetch_finbert_sentiment([_article()]) is None