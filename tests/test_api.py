"""Unit tests for the FastAPI layer (api/main.py).

Fully mocked network: the market-data fetch is replaced with a small synthetic
prices DataFrame and the LLM `chat` call is replaced with a canned response, so
nothing here touches the real Yahoo/Groq/etc. backends. The real forecasting
and optimization code runs on the synthetic prices on purpose. It's pure
numeric code with no network, and running it is what proves the endpoints use
the same engine as the app.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import src.ai_features as ai_features
from src.llm_client import LLMUnavailableError

client = TestClient(api_main.app)


def _synthetic_prices(tickers: list[str], rows: int = 300) -> pd.DataFrame:
    """A deterministic daily price history: a gentle upward trend plus a fixed
    per-ticker offset, enough rows for the trend models and covariance to fit.

    Seeded so every run sees the same numbers. A flaky optimizer result would
    otherwise look like a real regression.
    """
    rng = np.random.default_rng(42)
    index = pd.bdate_range(end=dt.date.today(), periods=rows)
    data = {}
    for i, ticker in enumerate(tickers):
        base = 100.0 + 10.0 * i
        drift = np.linspace(0.0, 15.0, rows)
        noise = rng.normal(0.0, 1.0, rows).cumsum() * 0.1
        data[ticker] = base + drift + noise
    return pd.DataFrame(data, index=index)


@pytest.fixture
def mock_prices(monkeypatch):
    """Replace the market-data fetch with synthetic prices for the requested
    tickers, so no endpoint test makes a real HTTP call."""
    def _fake_fetch(tickers, start, end):
        return _synthetic_prices(list(tickers))

    monkeypatch.setattr(api_main, "fetch_adjusted_close", _fake_fetch)


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------

def test_health_returns_ok():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /forecast
# ---------------------------------------------------------------------------

def test_forecast_returns_rounded_forecast_per_ticker(mock_prices):
    response = client.post("/forecast", json={"tickers": ["AAPL", "MSFT"], "horizon": 5})
    assert response.status_code == 200
    forecast = response.json()["forecast"]
    assert set(forecast.keys()) == {"AAPL", "MSFT"}
    # 2-decimal rounding: no value carries more than two fractional digits.
    for series in forecast.values():
        for value in series.values():
            assert round(value, 2) == value


def test_forecast_rejects_empty_tickers():
    response = client.post("/forecast", json={"tickers": [], "horizon": 5})
    assert response.status_code == 422


def test_forecast_rejects_out_of_range_horizon():
    response = client.post("/forecast", json={"tickers": ["AAPL"], "horizon": 0})
    assert response.status_code == 422
    response = client.post("/forecast", json={"tickers": ["AAPL"], "horizon": 253})
    assert response.status_code == 422


def test_forecast_unknown_model_is_422(mock_prices):
    response = client.post(
        "/forecast",
        json={"tickers": ["AAPL"], "horizon": 5, "model_name": "not a real model"},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /optimize
# ---------------------------------------------------------------------------

def test_optimize_returns_weights_summing_to_one(mock_prices):
    response = client.post("/optimize", json={"tickers": ["AAPL", "MSFT", "TSLA"]})
    assert response.status_code == 200
    weights = response.json()["weights"]
    assert set(weights.keys()) == {"AAPL", "MSFT", "TSLA"}
    # Long-only max-Sharpe weights are a fully-invested allocation.
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-2)
    # 4-decimal rounding.
    for value in weights.values():
        assert round(value, 4) == value


def test_optimize_requires_at_least_two_tickers():
    response = client.post("/optimize", json={"tickers": ["AAPL"]})
    assert response.status_code == 422


def test_optimize_rejects_bad_max_weight():
    response = client.post(
        "/optimize", json={"tickers": ["AAPL", "MSFT"], "max_weight_per_asset": 0.0}
    )
    assert response.status_code == 422
    response = client.post(
        "/optimize", json={"tickers": ["AAPL", "MSFT"], "max_weight_per_asset": 1.5}
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------

def test_ask_returns_answer_and_backend(monkeypatch):
    """Happy path: the LLM is mocked at `chat`, so no network call is made and
    the real answer_portfolio_question still assembles the prompt around it."""
    def _fake_chat(messages, temperature, max_tokens):
        return ("Your Sharpe ratio is 0.67.", "test-backend")

    monkeypatch.setattr(ai_features, "chat", _fake_chat)
    response = client.post(
        "/ask",
        json={"question": "What is my portfolio's return?", "results_context": "Sharpe: 0.67"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Your Sharpe ratio is 0.67."
    assert body["answered_by"] == "test-backend"


def test_ask_maps_llm_unavailable_to_503(monkeypatch):
    def _raise(messages, temperature, max_tokens):
        raise LLMUnavailableError("All backends failed")

    monkeypatch.setattr(ai_features, "chat", _raise)
    response = client.post(
        "/ask", json={"question": "What is my return?", "results_context": "Sharpe: 0.67"}
    )
    assert response.status_code == 503


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": "", "results_context": "x"})
    assert response.status_code == 422


def test_ask_rejects_too_long_question():
    response = client.post(
        "/ask", json={"question": "x" * 1001, "results_context": "x"}
    )
    assert response.status_code == 422
