"""FastAPI JSON API exposing the portfolio-forecasting engine over HTTP.

Each endpoint wraps a function app.py already calls. The API adds no analytics
of its own: it fetches prices, hands them to src/, and serialises the result.
Change the forecasting or optimization code and both front-ends pick it up.

Launch:
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8004

Default price window
--------------------
The request models take tickers but no dates, so the service picks a window:
`end` is today and `start` is ~5 years earlier (5 * 365 days). Five years of
daily closes is enough history for the covariance estimator and the trend
models the app ships with, while staying small enough to fetch quickly. A
caller who needs a different window should extend the request models rather
than have the endpoint guess.
"""
from __future__ import annotations

import datetime as dt

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.ai_features import answer_portfolio_question
from src.forecasting import forecast_all_assets
from src.llm_client import LLMUnavailableError
from src.market_data import fetch_adjusted_close
from src.optimization import (
    historical_mu_cov,
    optimize_max_sharpe,
    resolve_weight_bounds,
)

# Number of calendar days of price history fetched by default (~5 years). See
# the module docstring for why this window was chosen.
DEFAULT_HISTORY_DAYS = 5 * 365

DEFAULT_FORECAST_MODEL = "ETS (Holt linear trend)"

app = FastAPI(title="Portfolio Forecasting API")


def _default_price_window() -> tuple[dt.date, dt.date]:
    """Return (start, end) for the default fetch: today back ~5 years.

    Kept in one place so /forecast and /optimize fetch over the same window
    and a reader sees exactly what dates the endpoints use.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=DEFAULT_HISTORY_DAYS)
    return start, end


# ---------------------------------------------------------------------------
# Request models. Pydantic enforces the input contract; a violation becomes a
# 422 automatically, which matches the "invalid input -> 422" rule below.
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    tickers: list[str] = Field(min_length=1)
    horizon: int = Field(ge=1, le=252)
    model_name: str = DEFAULT_FORECAST_MODEL


class OptimizeRequest(BaseModel):
    tickers: list[str] = Field(min_length=2)
    max_weight_per_asset: float = Field(default=1.0, gt=0.0, le=1.0)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    results_context: str


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe. Returns as long as the process is up."""
    return {"status": "ok"}


@app.post("/forecast")
def forecast(request: ForecastRequest) -> dict:
    """Forecast future prices for each ticker over `horizon` trading days.

    Fetches the default price window, runs the chosen model per asset, and
    returns the forecast DataFrame rounded to 2 decimals as nested dicts.
    """
    start, end = _default_price_window()
    try:
        prices = fetch_adjusted_close(request.tickers, start, end)
        forecasts = forecast_all_assets(prices, request.horizon, request.model_name)
    except Exception as exc:
        # Unknown model name, empty/bad price data, fetch failure: all caller-
        # or data-driven, so 422 (unprocessable) rather than a 500.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"forecast": forecasts.round(2).to_dict()}


@app.post("/optimize")
def optimize(request: OptimizeRequest) -> dict:
    """Max-Sharpe weights for the tickers, long-only, capped per asset.

    Fetches the default window, estimates mu/cov from the history, applies the
    per-asset cap (long-only: short-selling is not exposed by this endpoint),
    and returns the optimal weights rounded to 4 decimals.
    """
    start, end = _default_price_window()
    try:
        prices = fetch_adjusted_close(request.tickers, start, end)
        mu, cov = historical_mu_cov(prices)
        bounds = resolve_weight_bounds(request.max_weight_per_asset, False)
        weights = optimize_max_sharpe(mu, cov, weight_bounds=bounds)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"weights": weights.round(4).to_dict()}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    """Grounded Q&A over a results context the caller already computed.

    Delegates to answer_portfolio_question with an empty chat history (the API
    is stateless; a caller that wants a conversation passes the prior turns in
    the context string). Returns the answer and which model produced it.
    """
    try:
        answer, model = answer_portfolio_question(
            request.question, request.results_context, chat_history=[]
        )
    except LLMUnavailableError as exc:
        # Every LLM backend failed. That's an upstream outage, not a bad
        # request, so 503.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"answer": answer, "answered_by": model}
