"""
Forecasted covariance via GARCH(1,1) — the direct answer to a gap this app's
own docs used to flag plainly: in `optimize_max_sharpe`'s three-portfolio
comparison, only mu (expected return) ever came from the forecasting layer;
the covariance matrix stayed historical even for the "Forecast-based"
portfolio (see `optimization.py`'s module docstring). This module forecasts
the VOLATILITY (diagonal) half of that gap.

What this forecasts, precisely, and what it deliberately still doesn't:
- Per-asset volatility: a genuine forecast, via GARCH(1,1) fit independently
  per asset — GARCH models exactly the thing historical (unconditional)
  variance can't: that volatility clusters (a calm period tends to stay calm,
  a turbulent one tends to stay turbulent), so the forecast reacts to recent
  conditions rather than averaging over the whole history.
- Correlation between assets: still historical (Ledoit-Wolf shrinkage on the
  same return window), NOT forecasted. A genuine correlation forecast is a
  DCC-GARCH (Dynamic Conditional Correlation) model — jointly estimating how
  every pair's correlation evolves over time. No well-maintained Python
  package implements DCC-GARCH robustly today (the `arch` package used here
  for the univariate fits deliberately does not), and hand-rolling a
  multivariate GARCH estimator is a research project, not a proportionate
  addition to this app. "Forecasted volatilities, historical correlation,
  combined via D @ Corr @ D" is a genuine, commonly used simplification in
  practice (not invented for this codebase) — documented here honestly as a
  partial answer, the same spirit as every other documented trade-off in this
  project (TF-IDF vs. neural embeddings in rag.py, the single-factor Hurst
  estimator in timeseries_diagnostics.py).

Per-asset graceful degradation: GARCH(1,1) needs enough history to reliably
estimate 3 parameters and can fail to converge on short or unusual series
(exactly the same shape of problem `forecasting.py`'s ARIMA grid search
already handles for price forecasts). Each asset's GARCH fit is attempted
independently; a failure or too-short history for ONE asset falls back to
that asset's own historical volatility rather than raising and losing the
whole covariance matrix — the same fails-soft philosophy used throughout
this codebase (Yahoo circuit breaker, Groq->Ollama, ARIMA->naive).
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from pypfopt import risk_models

from src.config import MIN_HISTORY_POINTS_FOR_GARCH, TRADING_DAYS_PER_YEAR

warnings.filterwarnings("ignore", module="arch")  # same convergence-warning noise
# pattern already suppressed for statsmodels in forecasting.py — a non-converged
# GARCH fit is already handled explicitly below (falls back to historical
# variance), so the warning itself would just be noise on top of that.


def garch_forecast_variance(returns: pd.Series, horizon_periods: int) -> float | None:
    """
    Forecast the AVERAGE variance over the next `horizon_periods`, from a
    GARCH(1,1) fit on one asset's returns. Not annualised here — the caller
    applies `periods_per_year` scaling (same division of responsibility as
    `historical_mu_cov`, which also takes annualisation as an explicit
    argument rather than assuming a frequency).

    Returns the average of the period-by-period forecasted variance across
    the horizon (not just the 1-step-ahead value) — a GARCH forecast decays
    toward the model's long-run variance as the horizon extends, so a 30-day
    average is a meaningfully different, more representative number than the
    next single day's forecast alone would be.

    Returns None (not NaN — the caller distinguishes "couldn't fit" from a
    real, if degenerate, numeric result) if there's too little history or the
    optimizer fails to converge, so the caller can fall back to that asset's
    historical variance rather than propagate a bad fit into the covariance
    matrix.
    """
    clean = returns.dropna()
    if len(clean) < MIN_HISTORY_POINTS_FOR_GARCH:
        return None

    # arch's optimizer is documented to be numerically happier with returns
    # scaled to roughly O(1)-O(10) magnitude rather than raw fractional
    # returns (O(0.01)) — scaling by 100 (i.e. working in percentage-point
    # returns) is the package's own documented convention, not a magic number
    # invented here. Scaled back down (÷ 100**2, since variance scales
    # quadratically) before returning.
    from arch import arch_model  # imported lazily — optional dependency, only needed for this cov_method

    scaled = clean.values * 100
    try:
        model = arch_model(scaled, vol="GARCH", p=1, q=1, mean="Zero", dist="normal")
        fit = model.fit(disp="off")
        if fit.convergence_flag != 0:
            return None  # optimizer did not converge — same "skip, don't trust it" as a raised exception
        forecast = fit.forecast(horizon=horizon_periods, reindex=False)
        # forecast.variance is one row (the origin), horizon_periods columns —
        # the average expected variance over the whole horizon, not just step 1.
        avg_scaled_variance = float(forecast.variance.values[0].mean())
    except Exception:
        return None  # non-convergent fit — same "skip, don't crash" pattern as forecasting.py's ARIMA grid

    if not np.isfinite(avg_scaled_variance) or avg_scaled_variance <= 0:
        return None  # degenerate result (e.g. NaN/zero/negative) — don't propagate into a covariance matrix

    return avg_scaled_variance / (100 ** 2)


def garch_forecast_cov(
    prices: pd.DataFrame,
    horizon_periods: int,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Forecasted covariance matrix: GARCH(1,1)-forecasted volatility per asset
    (see `garch_forecast_variance`) combined with the HISTORICAL correlation
    matrix (Ledoit-Wolf shrinkage, for the same noise-reduction reason it's
    used everywhere else in this app) via `Cov = D @ Corr @ D`, where `D` is
    the diagonal matrix of forecasted annualised volatilities. See this
    module's own docstring for exactly what is and isn't a genuine forecast
    here.

    `periods_per_year` MUST match `prices`' actual frequency — same
    annualisation contract as `optimization.historical_mu_cov` and
    `forecasting.forecast_mu`; passing the wrong value silently over/under-
    states annualised volatility rather than raising.

    Returns (covariance_df, diagnostics). `diagnostics["n_assets_via_garch"]`
    /`diagnostics["n_assets_via_fallback"]` report how many assets actually
    got a GARCH fit versus fell back to historical variance — surfaced rather
    than silent, same spirit as `factor_models.pca_factor_cov`'s own
    diagnostics dict, so a caller/UI can show the user when this "forecasted"
    covariance is mostly historical in practice (e.g. a wide universe where
    several short-history or newly-listed tickers can't support a GARCH fit).
    """
    returns = prices.pct_change().dropna(how="all")
    tickers = list(prices.columns)

    forecasted_variances: dict[str, float] = {}
    fallback_tickers: list[str] = []
    for ticker in tickers:
        variance = garch_forecast_variance(returns[ticker], horizon_periods)
        if variance is None:
            # Fall back to this asset's own historical (unconditional)
            # variance over the same window — annualised the same way the
            # GARCH path is below, so the two are on a comparable scale.
            variance = float(returns[ticker].dropna().var(ddof=1))
            fallback_tickers.append(ticker)
        forecasted_variances[ticker] = variance * periods_per_year

    forecasted_vols = pd.Series(forecasted_variances).pow(0.5).reindex(tickers)

    # Historical correlation, extracted from the SAME shrinkage covariance
    # estimator used elsewhere in this app (not a second, inconsistent
    # correlation estimate) — Ledoit-Wolf on the shrinkage covariance, then
    # normalised to a correlation matrix via D^-1 @ Cov @ D^-1.
    historical_cov = risk_models.CovarianceShrinkage(prices, frequency=periods_per_year).ledoit_wolf()
    historical_vols = np.sqrt(np.diag(historical_cov.values))
    inv_d = np.diag(1.0 / historical_vols)
    correlation = inv_d @ historical_cov.values @ inv_d

    d_forecast = np.diag(forecasted_vols.values)
    forecasted_cov = d_forecast @ correlation @ d_forecast
    cov_df = pd.DataFrame(forecasted_cov, index=tickers, columns=tickers)
    # Same symmetry safeguard as factor_models.pca_factor_cov — floating-point
    # rounding through the two matrix products above can drift it slightly.
    cov_df = (cov_df + cov_df.T) / 2.0

    diagnostics = {
        "n_assets_via_garch": len(tickers) - len(fallback_tickers),
        "n_assets_via_fallback": len(fallback_tickers),
        "fallback_tickers": fallback_tickers,
    }
    return cov_df, diagnostics