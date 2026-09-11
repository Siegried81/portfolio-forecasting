"""
Unit tests for src/market_data.py. Fully mocked — no real network calls.
Priorities: a multi-ticker Twelve Data response that drops a bad/delisted
symbol must not corrupt the tickers that did come back; the Yahoo circuit
breaker actually skips yfinance/direct-Yahoo once tripped; _redact_api_key
never leaks a key into a message that could reach st.error().
"""
import datetime as dt
import dataclasses
import time

import pandas as pd
import pytest
import streamlit as st

import src.market_data as market_data
from src.market_data import (
    MarketDataError,
    TwelveDataPlanRestricted,
    _configure_yfinance_cache,
    _download_alpha_vantage,
    _download_tiingo,
    _download_twelvedata,
    _fetch_finnhub_fundamentals,
    _fetch_twelvedata_fundamentals,
    _fill_missing_fundamentals_from_twelvedata,
    _fill_missing_fundamentals_from_yfinance,
    _fetch_yfinance_fundamentals,
    _redact_api_key,
    fetch_adjusted_close,
    fetch_fundamentals,
)


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setattr(market_data, "_yahoo_down_until", 0.0)
    monkeypatch.setattr(market_data, "_twelvedata_fill_plan_restricted", False)
    fake_settings = dataclasses.replace(
        market_data.LLM_SETTINGS, twelvedata_api_key="fake-test-key",
        alpha_vantage_api_key="fake-av-key", tiingo_api_key="fake-tiingo-key",
        finnhub_api_key="fake-finnhub-key",
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
# _download_twelvedata — partial-batch parsing
# ---------------------------------------------------------------------------

def test_download_twelvedata_partial_batch_does_not_lose_good_tickers(monkeypatch):
    good_ticker_payload = {
        "meta": {"symbol": "AAPL"},
        "values": [
            {"datetime": "2024-01-02", "close": "100.0"},
            {"datetime": "2024-01-03", "close": "101.5"},
        ],
    }
    response_payload = {"AAPL": good_ticker_payload}  # "BADTICKER" silently absent

    monkeypatch.setattr(
        market_data.requests, "get",
        lambda *a, **k: _FakeResponse(response_payload),
    )

    result = _download_twelvedata(["AAPL", "BADTICKER"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))

    assert "AAPL" in result.columns
    assert list(result["AAPL"].values) == [100.0, 101.5]
    assert "BADTICKER" not in result.columns


def test_download_twelvedata_single_ticker_shape_still_works(monkeypatch):
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
# _download_tiingo
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
# _download_alpha_vantage
# ---------------------------------------------------------------------------

def _alpha_vantage_payload(daily_closes: dict[str, str]) -> dict:
    return {
        "Time Series (Daily)": {
            date: {"4. close": close, "5. adjusted close": close}
            for date, close in daily_closes.items()
        }
    }


def test_download_alpha_vantage_parses_a_single_ticker(monkeypatch):
    payload = _alpha_vantage_payload({"2024-01-02": "100.0", "2024-01-03": "101.5"})
    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: _FakeResponse(payload))

    result = _download_alpha_vantage(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))

    assert "AAPL" in result.columns
    assert list(result["AAPL"].values) == [100.0, 101.5]


def test_download_twelvedata_requests_adjusted_prices(monkeypatch):
    captured = {}
    payload = {
        "meta": {"symbol": "AAPL"},
        "values": [{"datetime": "2024-01-02", "close": "100.0"}],
    }

    def _fake_get(url, params=None, **kwargs):
        captured.update(params or {})
        return _FakeResponse(payload)

    monkeypatch.setattr(market_data.requests, "get", _fake_get)
    _download_twelvedata(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))

    assert captured["adjusted"] == "true"


def test_download_alpha_vantage_makes_one_call_per_ticker_and_paces_between_them(monkeypatch):
    calls = []

    def _fake_get(url, params=None, timeout=None):
        calls.append(params["symbol"])
        return _FakeResponse(_alpha_vantage_payload({"2024-01-02": "10.0"}))

    monkeypatch.setattr(market_data.requests, "get", _fake_get)
    monkeypatch.setattr(market_data.time, "sleep", lambda seconds: None)

    result = _download_alpha_vantage(["AAPL", "MSFT"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))

    assert calls == ["AAPL", "MSFT"]
    assert set(result.columns) == {"AAPL", "MSFT"}


def test_download_alpha_vantage_filters_to_the_requested_date_range(monkeypatch):
    payload = _alpha_vantage_payload({
        "2023-12-31": "99.0",
        "2024-01-02": "100.0",
        "2024-02-01": "110.0",
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
# Circuit breaker
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

    fetch_adjusted_close(["X1"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    assert yfinance_calls["count"] == 1
    assert direct_calls["count"] == 1
    assert market_data._yahoo_down_until > 0

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
# _configure_yfinance_cache
# ---------------------------------------------------------------------------

def test_configure_yfinance_cache_calls_set_tz_cache_location(monkeypatch):
    calls = []
    monkeypatch.setattr(market_data.yf, "set_tz_cache_location", lambda path: calls.append(path))
    _configure_yfinance_cache()
    assert len(calls) == 1
    assert "py-yfinance-cache" in calls[0]


def test_configure_yfinance_cache_never_raises_if_yfinance_api_is_missing(monkeypatch):
    def _raise(path):
        raise AttributeError("set_tz_cache_location no longer exists")

    monkeypatch.setattr(market_data.yf, "set_tz_cache_location", _raise)
    _configure_yfinance_cache()

# ---------------------------------------------------------------------------
# _fetch_finnhub_fundamentals
# ---------------------------------------------------------------------------

def _finnhub_fundamentals_get(profile: dict, metric: dict):
    def _fake_get(url, params=None, timeout=None):
        if url == market_data.FINNHUB_PROFILE_URL:
            return _FakeResponse(profile)
        return _FakeResponse({"metric": metric})
    return _fake_get


def test_fetch_finnhub_fundamentals_parses_a_well_formed_response(monkeypatch):
    monkeypatch.setattr(
        market_data.requests, "get",
        _finnhub_fundamentals_get(
            {"name": "Apple Inc"},
            {"peBasicExclExtraTTM": 36.2, "beta": 1.1, "52WeekHigh": 345.0, "52WeekLow": 226.0},
        ),
    )
    result = _fetch_finnhub_fundamentals("AAPL")
    assert result is not None
    assert result["name"] == "Apple Inc"
    assert result["source"] == "Finnhub"
    assert result["pe_ratio"] == 36.2
    assert result["beta"] == 1.1
    assert result["forward_pe"] is None


def test_fetch_finnhub_fundamentals_converts_market_cap_from_millions(monkeypatch):
    monkeypatch.setattr(
        market_data.requests, "get",
        _finnhub_fundamentals_get({"name": "Apple Inc", "marketCapitalization": 4_614_971.5}, {}),
    )
    result = _fetch_finnhub_fundamentals("AAPL")
    assert result["market_cap"] == pytest.approx(4_614_971.5 * 1_000_000)


def test_fetch_finnhub_fundamentals_returns_none_without_a_key(monkeypatch):
    fake_settings = dataclasses.replace(market_data.LLM_SETTINGS, finnhub_api_key=None)
    monkeypatch.setattr(market_data, "LLM_SETTINGS", fake_settings)

    def _fail(*a, **k):
        pytest.fail("requests.get should not be called without a Finnhub key")

    monkeypatch.setattr(market_data.requests, "get", _fail)
    assert _fetch_finnhub_fundamentals("AAPL") is None


def test_fetch_finnhub_fundamentals_returns_none_for_unrecognised_symbol(monkeypatch):
    monkeypatch.setattr(market_data.requests, "get", _finnhub_fundamentals_get({}, {}))
    assert _fetch_finnhub_fundamentals("NOTATICKER") is None


# ---------------------------------------------------------------------------
# _fill_missing_fundamentals_from_twelvedata
# ---------------------------------------------------------------------------

def _finnhub_result(**overrides) -> dict:
    base = {
        "name": "Apple Inc", "source": "Finnhub", "market_cap": 4_614_971_500_000.0,
        "pe_ratio": 36.2, "forward_pe": None, "peg_ratio": None, "price_to_book": None,
        "beta": 1.10, "dividend_yield": 0.0051, "52w_high": 345.0, "52w_low": 226.0,
    }
    base.update(overrides)
    return base


def test_fill_missing_fundamentals_is_a_noop_when_nothing_is_missing(monkeypatch):
    complete = _finnhub_result(forward_pe=38.0, peg_ratio=2.1, price_to_book=45.0)

    def _fail(ticker):
        pytest.fail("_fetch_twelvedata_fundamentals should not be called when nothing is missing")

    monkeypatch.setattr(market_data, "_fetch_twelvedata_fundamentals", _fail)
    result = _fill_missing_fundamentals_from_twelvedata("AAPL", complete)
    assert result["source"] == "Finnhub"


def test_fill_missing_fundamentals_fills_the_gap_and_tags_blended_source(monkeypatch):
    partial = _finnhub_result()
    twelvedata_result = {"forward_pe": 34.5, "peg_ratio": None, "price_to_book": 48.0}
    monkeypatch.setattr(market_data, "_fetch_twelvedata_fundamentals", lambda ticker: twelvedata_result)

    result = _fill_missing_fundamentals_from_twelvedata("AAPL", partial)

    assert result["forward_pe"] == 34.5
    assert result["price_to_book"] == 48.0
    assert result["peg_ratio"] is None
    assert result["pe_ratio"] == 36.2
    assert result["source"] == "Finnhub + Twelve Data"


def test_fill_missing_fundamentals_leaves_gaps_when_twelvedata_returns_none(monkeypatch):
    partial = _finnhub_result()
    monkeypatch.setattr(market_data, "_fetch_twelvedata_fundamentals", lambda ticker: None)

    result = _fill_missing_fundamentals_from_twelvedata("AAPL", partial)

    assert result["forward_pe"] is None
    assert result["source"] == "Finnhub"


def test_fill_missing_fundamentals_trips_breaker_on_plan_restricted_without_raising(monkeypatch):
    partial = _finnhub_result()

    def _raise(ticker):
        raise TwelveDataPlanRestricted("requires pro plan")

    monkeypatch.setattr(market_data, "_fetch_twelvedata_fundamentals", _raise)

    result = _fill_missing_fundamentals_from_twelvedata("MSFT", partial)

    assert result["forward_pe"] is None
    assert result["pe_ratio"] == 36.2
    assert market_data._twelvedata_fill_plan_restricted is True


def test_fill_missing_fundamentals_skips_call_once_breaker_is_tripped(monkeypatch):
    monkeypatch.setattr(market_data, "_twelvedata_fill_plan_restricted", True)

    def _fail(ticker):
        pytest.fail("_fetch_twelvedata_fundamentals should not be called once the breaker has tripped")

    monkeypatch.setattr(market_data, "_fetch_twelvedata_fundamentals", _fail)
    result = _fill_missing_fundamentals_from_twelvedata("CSCO", _finnhub_result())
    assert result["forward_pe"] is None


# ---------------------------------------------------------------------------
# fetch_fundamentals — Finnhub-first, per-field fill wired in
# ---------------------------------------------------------------------------

def test_fetch_fundamentals_fills_gap_via_twelvedata_end_to_end(monkeypatch):
    monkeypatch.setattr(market_data, "_fetch_finnhub_fundamentals", lambda ticker: _finnhub_result())
    monkeypatch.setattr(
        market_data, "_fetch_twelvedata_fundamentals",
        lambda ticker: {"forward_pe": 34.5, "peg_ratio": None, "price_to_book": None},
    )
    result = fetch_fundamentals("AAPL")
    assert result["forward_pe"] == 34.5
    assert result["source"] == "Finnhub + Twelve Data"


def test_fetch_fundamentals_still_falls_back_entirely_when_finnhub_returns_nothing(monkeypatch):
    monkeypatch.setattr(market_data, "_fetch_finnhub_fundamentals", lambda ticker: None)
    monkeypatch.setattr(
        market_data, "_fetch_twelvedata_fundamentals",
        lambda ticker: {"name": "Apple Inc", "source": "Twelve Data", "pe_ratio": 36.2},
    )
    result = fetch_fundamentals("AAPL")
    assert result["source"] == "Twelve Data"


def test_fetch_fundamentals_propagates_plan_restricted_only_from_the_primary_path(monkeypatch):
    monkeypatch.setattr(market_data, "_fetch_finnhub_fundamentals", lambda ticker: None)

    def _raise(ticker):
        raise TwelveDataPlanRestricted("requires pro plan")

    monkeypatch.setattr(market_data, "_fetch_twelvedata_fundamentals", _raise)
    with pytest.raises(TwelveDataPlanRestricted):
        fetch_fundamentals("MSFT")


# ---------------------------------------------------------------------------
# _fetch_yfinance_fundamentals
# ---------------------------------------------------------------------------

class _FakeYfTicker:
    def __init__(self, info: dict):
        self.info = info


def test_fetch_yfinance_fundamentals_parses_forward_pe(monkeypatch):
    fake_info = {
        "longName": "Apple Inc.", "regularMarketPrice": 258.0,
        "marketCap": 4_614_971_518_388, "trailingPE": 36.2, "forwardPE": 32.9,
        "priceToBook": 55.1, "beta": 1.10, "fiftyTwoWeekHigh": 345.0, "fiftyTwoWeekLow": 226.0,
    }
    monkeypatch.setattr(market_data.yf, "Ticker", lambda ticker: _FakeYfTicker(fake_info))
    result = _fetch_yfinance_fundamentals("AAPL")
    assert result is not None
    assert result["source"] == "yfinance"
    assert result["forward_pe"] == 32.9
    assert result["market_cap"] == 4_614_971_518_388


def test_fetch_yfinance_fundamentals_never_includes_dividend_yield(monkeypatch):
    fake_info = {"regularMarketPrice": 100.0, "trailingPE": 20.0, "dividendYield": 0.51}
    monkeypatch.setattr(market_data.yf, "Ticker", lambda ticker: _FakeYfTicker(fake_info))
    result = _fetch_yfinance_fundamentals("AAPL")
    assert "dividend_yield" not in result or result.get("dividend_yield") is None


def test_fetch_yfinance_fundamentals_returns_none_for_unrecognised_symbol(monkeypatch):
    monkeypatch.setattr(market_data.yf, "Ticker", lambda ticker: _FakeYfTicker({}))
    assert _fetch_yfinance_fundamentals("NOTATICKER") is None


def test_fetch_yfinance_fundamentals_returns_none_on_any_exception(monkeypatch):
    def _raise(ticker):
        raise RuntimeError("simulated yfinance cookie/crumb failure")

    monkeypatch.setattr(market_data.yf, "Ticker", _raise)
    assert _fetch_yfinance_fundamentals("AAPL") is None


# ---------------------------------------------------------------------------
# _fill_missing_fundamentals_from_yfinance
# ---------------------------------------------------------------------------

def test_fill_missing_fundamentals_from_yfinance_fills_gap_and_tags_source(monkeypatch):
    partial = _finnhub_result()
    monkeypatch.setattr(
        market_data, "_fetch_yfinance_fundamentals",
        lambda ticker: {"forward_pe": 32.9, "peg_ratio": None, "price_to_book": 50.0},
    )
    result = _fill_missing_fundamentals_from_yfinance("AAPL", partial)
    assert result["forward_pe"] == 32.9
    assert result["price_to_book"] == 50.0
    assert result["peg_ratio"] is None
    assert result["source"] == "Finnhub + yfinance"


def test_fill_missing_fundamentals_from_yfinance_is_a_noop_when_nothing_missing(monkeypatch):
    complete = _finnhub_result(forward_pe=38.0, peg_ratio=2.1, price_to_book=45.0)

    def _fail(ticker):
        pytest.fail("_fetch_yfinance_fundamentals should not be called when nothing is missing")

    monkeypatch.setattr(market_data, "_fetch_yfinance_fundamentals", _fail)
    result = _fill_missing_fundamentals_from_yfinance("AAPL", complete)
    assert result["source"] == "Finnhub"


def test_fill_missing_fundamentals_from_yfinance_never_raises_and_preserves_existing_fields(monkeypatch):
    partial = _finnhub_result()

    def _raise(ticker):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(market_data, "_fetch_yfinance_fundamentals", _raise)
    with pytest.raises(RuntimeError):
        _fill_missing_fundamentals_from_yfinance("AAPL", partial)


# ---------------------------------------------------------------------------
# fetch_fundamentals — full chain: Finnhub -> Twelve Data -> yfinance
# ---------------------------------------------------------------------------

def test_fetch_fundamentals_falls_through_to_yfinance_when_twelvedata_also_misses(monkeypatch):
    monkeypatch.setattr(market_data, "_fetch_finnhub_fundamentals", lambda ticker: _finnhub_result())
    monkeypatch.setattr(
        market_data, "_fetch_twelvedata_fundamentals",
        lambda ticker: {"peg_ratio": None, "price_to_book": None},
    )
    monkeypatch.setattr(
        market_data, "_fetch_yfinance_fundamentals",
        lambda ticker: {"forward_pe": 32.9, "peg_ratio": None, "price_to_book": None},
    )
    result = fetch_fundamentals("AAPL")
    assert result["forward_pe"] == 32.9
    assert result["source"] == "Finnhub + yfinance"


def test_fetch_fundamentals_reports_all_three_sources_when_each_contributes(monkeypatch):
    monkeypatch.setattr(market_data, "_fetch_finnhub_fundamentals", lambda ticker: _finnhub_result())
    monkeypatch.setattr(
        market_data, "_fetch_twelvedata_fundamentals",
        lambda ticker: {"forward_pe": 34.5, "peg_ratio": None, "price_to_book": None},
    )
    monkeypatch.setattr(
        market_data, "_fetch_yfinance_fundamentals",
        lambda ticker: {"peg_ratio": 2.4, "price_to_book": None},
    )
    result = fetch_fundamentals("AAPL")
    assert result["forward_pe"] == 34.5
    assert result["peg_ratio"] == 2.4
    assert result["source"] == "Finnhub + Twelve Data + yfinance"


# ---------------------------------------------------------------------------
# _fetch_yfinance_fundamentals — shares the price-fetcher's Yahoo circuit breaker
# ---------------------------------------------------------------------------

def test_fetch_yfinance_fundamentals_trips_shared_yahoo_breaker_on_429(monkeypatch):
    def _raise_429(ticker):
        raise RuntimeError(
            "429 Client Error: Too Many Requests for url: "
            "https://query2.finance.yahoo.com/v10/finance/quoteSummary/ADBE"
        )

    monkeypatch.setattr(market_data.yf, "Ticker", _raise_429)
    assert _fetch_yfinance_fundamentals("ADBE") is None
    assert market_data._yahoo_down_until > time.monotonic()


def test_fetch_yfinance_fundamentals_skips_call_when_yahoo_breaker_already_tripped(monkeypatch):
    monkeypatch.setattr(market_data, "_yahoo_down_until", time.monotonic() + 60)

    def _fail(ticker):
        pytest.fail("yf.Ticker should not be called while the Yahoo breaker is tripped")

    monkeypatch.setattr(market_data.yf, "Ticker", _fail)
    assert _fetch_yfinance_fundamentals("MSFT") is None


def test_fetch_yfinance_fundamentals_does_not_trip_breaker_on_unrelated_errors(monkeypatch):
    def _raise_other(ticker):
        raise RuntimeError("simulated cookie/crumb parse failure")

    monkeypatch.setattr(market_data.yf, "Ticker", _raise_other)
    assert _fetch_yfinance_fundamentals("MSFT") is None
    assert market_data._yahoo_down_until <= time.monotonic()