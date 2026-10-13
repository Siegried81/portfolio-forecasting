"""
Guardrail tests: an API key carried in a request URL's query string must never
reach the logs. `requests`' HTTPError message embeds the full URL, so every
fetcher that puts a key in the query string (NewsAPI `apiKey=`, Finnhub
`token=`) must pass its exception through `redact_secrets` before logging.
Fully offline — requests.get is faked, keys are dummy strings.
"""
import dataclasses
import logging

import pytest
import requests
import streamlit as st

from src import news_data
from src.config import FINANCIAL_ADVICE_DISCLAIMER
from src.http_utils import redact_secrets

FAKE_KEY = "dummy-secret-value-123"


@pytest.mark.parametrize("param", ["apiKey", "apikey", "api_key", "token", "access_token", "key"])
def test_redact_secrets_masks_every_credential_param_name(param):
    text = f"401 Client Error for url: https://x.test/v1?q=AAPL&{param}={FAKE_KEY}&page=1"
    redacted = redact_secrets(text)
    assert FAKE_KEY not in redacted
    assert f"{param}=***REDACTED***" in redacted
    assert "q=AAPL" in redacted and "page=1" in redacted  # non-secret params untouched


def test_redact_secrets_leaves_words_merely_ending_in_key_alone():
    assert redact_secrets("monkey=banana") == "monkey=banana"


@pytest.fixture
def _fake_keys(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setattr(news_data, "_finnhub_sentiment_plan_restricted", False)
    monkeypatch.setattr(
        news_data, "LLM_SETTINGS",
        dataclasses.replace(news_data.LLM_SETTINGS, newsapi_key=FAKE_KEY, finnhub_api_key=FAKE_KEY),
    )
    yield
    st.cache_data.clear()


def _raise_http_error_with_key_in_url(status_code):
    """A fake requests.get whose response raises an HTTPError embedding the
    request URL (key included), exactly as real `requests` does."""
    def _fake_get(url, params=None, **kwargs):
        query = "&".join(f"{k}={v}" for k, v in (params or {}).items())
        response = requests.Response()
        response.status_code = status_code

        def _raise():
            raise requests.HTTPError(f"{status_code} Client Error for url: {url}?{query}", response=response)
        response.raise_for_status = _raise
        return response
    return _fake_get


@pytest.mark.usefixtures("_fake_keys")
@pytest.mark.parametrize("status_code", [401, 403, 429, 500])
@pytest.mark.parametrize("call", [
    lambda: news_data.fetch_ticker_headlines("AAPL"),
    lambda: news_data.fetch_finnhub_news("AAPL"),
    lambda: news_data.fetch_finnhub_sentiment("AAPL"),
], ids=["newsapi", "finnhub_news", "finnhub_sentiment"])
def test_api_key_in_http_error_url_never_reaches_the_log(monkeypatch, caplog, call, status_code):
    monkeypatch.setattr(news_data.requests, "get", _raise_http_error_with_key_in_url(status_code))
    with caplog.at_level(logging.WARNING, logger="src.news_data"):
        call()
    assert caplog.records, "the failure should still be logged (just redacted)"
    assert FAKE_KEY not in caplog.text
    assert "***REDACTED***" in caplog.text


def test_financial_advice_disclaimer_is_shown_in_the_app():
    # The constant must exist, say what SECURITY.md promises, and be used by app.py.
    assert "not financial advice" in FINANCIAL_ADVICE_DISCLAIMER.lower()
    with open("app.py", encoding="utf-8") as f:
        assert "FINANCIAL_ADVICE_DISCLAIMER" in f.read()
