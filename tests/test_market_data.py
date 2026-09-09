"""
Unit tests for src/market_data.py.

Fully mocked — no real network calls, no dependency on a live TWELVEDATA_API_KEY.
Priorities, per the code review:
1. A multi-ticker Twelve Data response
   that drops a bad/delisted symbol (instead of keeping it as an error-tagged
   key) must not corrupt the tickers that DID come back.
2. The Yahoo circuit breaker actually skips yfinance/direct-Yahoo once tripped.
3. _redact_api_key never leaks a key into a message that could reach st.error().
"""
import datetime as dt
import dataclasses

import pandas as pd
import pytest
import streamlit as st

import src.market_data as market_data
from src.market_data import (
    MarketDataError,
    _configure_yfinance_cache,
    _download_alpha_vantage,
    _download_tiingo,
    _download_twelvedata,
    _redact_api_key,
    fetch_adjusted_close,
)


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Isolate every test from the module-level circuit breaker and Streamlit's
    process-wide cache — both are shared state that would otherwise leak
    between tests depending on execution order."""
    st.cache_data.clear()
    monkeypatch.setattr(market_data, "_yahoo_down_until", 0.0)
    # LLM_SETTINGS is a frozen dataclass (config.py, deliberately immutable) —
    # can't monkeypatch a single field on it directly (raises
    # FrozenInstanceError). Replace the whole object with a copy instead,
    # scoped to market_data's own imported name so other modules are unaffected.
    fake_settings = dataclasses.replace(
        market_data.LLM_SETTINGS, twelvedata_api_key="fake-test-key",
        alpha_vantage_api_key="fake-av-key", tiingo_api_key="fake-tiingo-key",
    )
    monkeypatch.setattr(market_data, "LLM_SETTINGS", fake_settings)
    yield
    st.cache_data.clear()


class _FakeResponse:
    def __init__(self, json_data: dict, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise __import__("requests").HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


# ---------------------------------------------------------------------------
# _redact_api_key
# ---------------------------------------------------------------------------

def test_redact_api_key_strips_the_value():
    text = "GET https://api.twelvedata.com/time_series?symbol=AAPL&apikey=sk-super-secret-123"
    redacted = _redact_api_key(text)
    assert "sk-super-secret-123" not in redacted
    assert "apikey=***REDACTED***" in redacted


# ---------------------------------------------------------------------------
# _download_twelvedata — the partial-batch parsing behaviour
# ---------------------------------------------------------------------------

def test_download_twelvedata_partial_batch_does_not_lose_good_tickers(monkeypatch):
    """
    Simulates Twelve Data DROPPING one requested ticker entirely from a
    multi-symbol response (instead of keeping it as an error-tagged key) —
    the exact scenario that silently corrupted the whole batch before the fix.
    """
    good_ticker_payload = {
        "meta": {"symbol": "AAPL"},
        "values": [
            {"datetime": "2024-01-02", "close": "100.0"},
            {"datetime": "2024-01-03", "close": "101.5"},
        ],
    }
    # Only "AAPL" comes back — "BADTICKER" is silently absent, not even as an
    # error-tagged key, which is the real-world case that broke the old logic.
    response_payload = {"AAPL": good_ticker_payload}

    monkeypatch.setattr(
        market_data.requests, "get",
        lambda *a, **k: _FakeResponse(response_payload),
    )

    result = _download_twelvedata(["AAPL", "BADTICKER"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))

    # AAPL must still be parsed correctly — this is what broke before the fix.
    assert "AAPL" in result.columns
    assert list(result["AAPL"].values) == [100.0, 101.5]
    # BADTICKER legitimately has no data and should simply be absent, not raise.
    assert "BADTICKER" not in result.columns


def test_download_twelvedata_single_ticker_shape_still_works(monkeypatch):
    """Single-ticker requests return the flat {"meta":..., "values":[...]} shape
    directly (no outer ticker key) — must still be handled after the fix."""
    payload = {
        "meta": {"symbol": "AAPL"},
        "values": [{"datetime": "2024-01-02", "close": "100.0"}],
    }
    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: _FakeResponse(payload))

    result = _download_twelvedata(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    assert list(result.columns) == ["AAPL"]
    assert result["AAPL"].iloc[0] == 100.0


def test_download_twelvedata_raises_when_nothing_comes_back(monkeypatch):
    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: _FakeResponse({"AAPL": {"values": []}}))
    with pytest.raises(MarketDataError):
        _download_twelvedata(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))


# ---------------------------------------------------------------------------
# _download_tiingo — the third, most generous tier
# ---------------------------------------------------------------------------

def _tiingo_rows(dates_closes: list[tuple[str, float]]) -> list[dict]:
    return [{"date": d, "adjClose": c} for d, c in dates_closes]


def test_download_tiingo_parses_a_single_ticker(monkeypatch):
    monkeypatch.setattr(
        market_data.requests, "get",
        lambda *a, **k: _FakeResponse(_tiingo_rows([("2024-01-02", 100.0), ("2024-01-03", 101.5)])),
    )
    result = _download_tiingo(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    assert list(result["AAPL"].values) == [100.0, 101.5]


def test_download_tiingo_sends_bearer_style_token_header(monkeypatch):
    captured = {}

    def _fake_get(url, params=None, headers=None, timeout=None):
        captured["headers"] = headers
        return _FakeResponse(_tiingo_rows([("2024-01-02", 100.0)]))

    monkeypatch.setattr(market_data.requests, "get", _fake_get)
    _download_tiingo(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    assert captured["headers"]["Authorization"] == "Token fake-tiingo-key"


def test_download_tiingo_raises_without_a_configured_key(monkeypatch):
    fake_settings = dataclasses.replace(market_data.LLM_SETTINGS, tiingo_api_key=None)
    monkeypatch.setattr(market_data, "LLM_SETTINGS", fake_settings)
    with pytest.raises(MarketDataError):
        _download_tiingo(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))


def test_download_tiingo_raises_when_nothing_comes_back(monkeypatch):
    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: _FakeResponse([]))
    with pytest.raises(MarketDataError):
        _download_tiingo(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))


def test_fetch_adjusted_close_tries_tiingo_before_twelvedata(monkeypatch):
    monkeypatch.setattr(market_data, "_download_yfinance", lambda *a, **k: (_ for _ in ()).throw(MarketDataError("down")))
    monkeypatch.setattr(market_data, "_download_yahoo_direct", lambda *a, **k: (_ for _ in ()).throw(MarketDataError("down")))
    monkeypatch.setattr(market_data, "_download_twelvedata", lambda *a, **k: pytest.fail("Twelve Data should not be called when Tiingo succeeds"))
    fake_tiingo_df = pd.DataFrame({"AAPL": [100.0]}, index=pd.to_datetime(["2024-01-02"]))
    monkeypatch.setattr(market_data, "_download_tiingo", lambda *a, **k: fake_tiingo_df.copy())

    result = fetch_adjusted_close(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    assert result.attrs["source"] == "tiingo (yahoo unavailable)"


# ---------------------------------------------------------------------------
# _download_alpha_vantage — the fifth, last-resort tier
# ---------------------------------------------------------------------------

def _alpha_vantage_payload(daily_closes: dict[str, str]) -> dict:
    return {"Time Series (Daily)": {date: {"4. close": close} for date, close in daily_closes.items()}}


def test_download_alpha_vantage_parses_a_single_ticker(monkeypatch):
    payload = _alpha_vantage_payload({"2024-01-02": "100.0", "2024-01-03": "101.5"})
    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: _FakeResponse(payload))

    result = _download_alpha_vantage(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))

    assert "AAPL" in result.columns
    assert list(result["AAPL"].values) == [100.0, 101.5]


def test_download_alpha_vantage_makes_one_call_per_ticker_and_paces_between_them(monkeypatch):
    calls = []

    def _fake_get(url, params=None, timeout=None):
        calls.append(params["symbol"])
        return _FakeResponse(_alpha_vantage_payload({"2024-01-02": "10.0"}))

    monkeypatch.setattr(market_data.requests, "get", _fake_get)
    monkeypatch.setattr(market_data.time, "sleep", lambda seconds: None)  # skip the real rate-limit pacing in tests

    result = _download_alpha_vantage(["AAPL", "MSFT"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))

    assert calls == ["AAPL", "MSFT"]
    assert set(result.columns) == {"AAPL", "MSFT"}


def test_download_alpha_vantage_filters_to_the_requested_date_range(monkeypatch):
    payload = _alpha_vantage_payload({
        "2023-12-31": "99.0",  # before the requested range -> excluded
        "2024-01-02": "100.0",  # inside the range -> kept
        "2024-02-01": "110.0",  # after the range -> excluded
    })
    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: _FakeResponse(payload))

    result = _download_alpha_vantage(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))

    assert list(result["AAPL"].values) == [100.0]


def test_download_alpha_vantage_skips_a_ticker_reporting_an_error_message(monkeypatch):
    def _fake_get(url, params=None, timeout=None):
        if params["symbol"] == "BADTICKER":
            return _FakeResponse({"Error Message": "Invalid API call"})
        return _FakeResponse(_alpha_vantage_payload({"2024-01-02": "100.0"}))

    monkeypatch.setattr(market_data.requests, "get", _fake_get)
    monkeypatch.setattr(market_data.time, "sleep", lambda seconds: None)

    result = _download_alpha_vantage(["AAPL", "BADTICKER"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))

    assert "AAPL" in result.columns
    assert "BADTICKER" not in result.columns


def test_download_alpha_vantage_skips_a_ticker_reporting_a_rate_limit_note(monkeypatch):
    # Alpha Vantage's free-tier rate-limit message comes back as a "Note" or
    # "Information" field in an otherwise-200 response, not an HTTP error.
    monkeypatch.setattr(
        market_data.requests, "get",
        lambda *a, **k: _FakeResponse({"Information": "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day."}),
    )
    with pytest.raises(MarketDataError):
        _download_alpha_vantage(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))


def test_download_alpha_vantage_raises_without_a_configured_key(monkeypatch):
    fake_settings = dataclasses.replace(market_data.LLM_SETTINGS, alpha_vantage_api_key=None)
    monkeypatch.setattr(market_data, "LLM_SETTINGS", fake_settings)
    with pytest.raises(MarketDataError):
        _download_alpha_vantage(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))


def test_download_alpha_vantage_raises_when_nothing_comes_back(monkeypatch):
    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: _FakeResponse({}))
    with pytest.raises(MarketDataError):
        _download_alpha_vantage(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))


def test_fetch_adjusted_close_falls_back_to_alpha_vantage_when_twelvedata_also_fails(monkeypatch):
    # Full five-tier cascade: yfinance, direct Yahoo, and Tiingo all fail,
    # Twelve Data ALSO fails -> must reach Alpha Vantage, not raise.
    monkeypatch.setattr(market_data, "_download_yfinance", lambda *a, **k: (_ for _ in ()).throw(MarketDataError("down")))
    monkeypatch.setattr(market_data, "_download_yahoo_direct", lambda *a, **k: (_ for _ in ()).throw(MarketDataError("down")))
    monkeypatch.setattr(market_data, "_download_tiingo", lambda *a, **k: (_ for _ in ()).throw(MarketDataError("down")))
    monkeypatch.setattr(market_data, "_download_twelvedata", lambda *a, **k: (_ for _ in ()).throw(MarketDataError("down")))
    fake_av_df = pd.DataFrame({"AAPL": [100.0, 101.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    monkeypatch.setattr(market_data, "_download_alpha_vantage", lambda *a, **k: fake_av_df.copy())

    result = fetch_adjusted_close(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))

    assert "AAPL" in result.columns
    assert result.attrs["source"] == "alpha vantage (yahoo unavailable)"


# ---------------------------------------------------------------------------
# Circuit breaker — once both Yahoo paths fail, skip straight to Twelve Data
# ---------------------------------------------------------------------------

def test_circuit_breaker_skips_yahoo_after_a_prior_failure(monkeypatch):
    yfinance_calls = {"count": 0}
    direct_calls = {"count": 0}

    def _failing_yfinance(*a, **k):
        yfinance_calls["count"] += 1
        raise MarketDataError("Yahoo down (simulated)")

    def _failing_direct(*a, **k):
        direct_calls["count"] += 1
        raise MarketDataError("Yahoo direct also down (simulated)")

    fake_twelvedata_df = pd.DataFrame(
        {"X1": [100.0, 101.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    monkeypatch.setattr(market_data, "_download_yfinance", _failing_yfinance)
    monkeypatch.setattr(market_data, "_download_yahoo_direct", _failing_direct)
    monkeypatch.setattr(market_data, "_download_tiingo", lambda *a, **k: (_ for _ in ()).throw(MarketDataError("Tiingo down (simulated)")))
    monkeypatch.setattr(market_data, "_download_twelvedata", lambda *a, **k: fake_twelvedata_df.copy())

    # Call #1: both Yahoo paths fail -> trips the breaker, falls back to Twelve Data.
    fetch_adjusted_close(["X1"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    assert yfinance_calls["count"] == 1
    assert direct_calls["count"] == 1
    assert market_data._yahoo_down_until > 0

    # Call #2, DIFFERENT tickers (so Streamlit's cache can't just return the
    # first call's result) but WITHIN the breaker window: must skip Yahoo
    # entirely and go straight to Twelve Data.
    fake_twelvedata_df2 = pd.DataFrame(
        {"X2": [50.0, 51.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    monkeypatch.setattr(market_data, "_download_twelvedata", lambda *a, **k: fake_twelvedata_df2.copy())
    result = fetch_adjusted_close(["X2"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))

    assert yfinance_calls["count"] == 1  # unchanged — Yahoo was never retried
    assert direct_calls["count"] == 1
    assert "X2" in result.columns
    assert result.attrs["source"] == "twelvedata (yahoo recently unavailable)"


# ---------------------------------------------------------------------------
# _configure_yfinance_cache — WSL/DrvFs sqlite "unable to open database file" fix
# ---------------------------------------------------------------------------

def test_configure_yfinance_cache_calls_set_tz_cache_location(monkeypatch):
    calls = []
    monkeypatch.setattr(market_data.yf, "set_tz_cache_location", lambda path: calls.append(path))
    _configure_yfinance_cache()
    assert len(calls) == 1
    assert "py-yfinance-cache" in calls[0]


def test_configure_yfinance_cache_never_raises_if_yfinance_api_is_missing(monkeypatch):
    # This touches a yfinance INTERNAL API — must degrade to "use yfinance's
    # own default" rather than crash app import if a future yfinance release
    # renames/removes set_tz_cache_location.
    def _raise(path):
        raise AttributeError("set_tz_cache_location no longer exists")

    monkeypatch.setattr(market_data.yf, "set_tz_cache_location", _raise)
    _configure_yfinance_cache()  # must not raise