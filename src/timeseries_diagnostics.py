"""
Time-series diagnostics: stationarity, trend persistence, and rolling
risk-adjusted return — kept in their own module rather than
folded into metrics.py because they answer a different question. metrics.py
summarises PERFORMANCE (a single number per return series); this module
diagnoses the series' own STRUCTURE (is it stationary? trending or
mean-reverting? how does risk-adjusted return evolve over time?) — the kind
of question an economist/quant asks before trusting a forecast, not after.

Two of the three functions here directly validate assumptions made elsewhere
in this codebase rather than adding a disconnected number: the ADF test
checks forecasting.py's own d=1 first-differencing assumption instead of just
asserting it, and the Hurst exponent gives a concrete, per-asset answer to
the efficient-market-hypothesis caveat already stated throughout the app
("short-horizon price forecasts carry weak predictive power") — a Hurst near
0.5 is the quantitative confirmation of exactly that caveat, per asset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from typing import Any

from src.config import DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR

MIN_OBSERVATIONS_FOR_ADF = 20  # statsmodels itself warns below ~20 the test is unreliable
MIN_OBSERVATIONS_FOR_HURST = 40  # needs max_lag*2 at minimum; a bit of slack above that


def adf_stationarity_test(returns: pd.Series) -> dict[str, Any]:
    """
    Augmented Dickey-Fuller test, run on RETURNS (not prices) — the null
    hypothesis is that the series has a unit root (is non-stationary).
    Directly validates forecasting.py's own ARIMA `d=1` choice: ARIMA fixes
    first-differencing because price LEVELS are non-stationary by
    construction, and this test checks whether the differenced series (i.e.
    returns) actually IS stationary, rather than the codebase just asserting
    it.

    Returns {"adf_statistic": float, "p_value": float, "is_stationary": bool}.
    `is_stationary` is `p_value < 0.05` (reject the unit-root null at the
    standard 5% level). Returns an all-None dict (not zeros — zeros would
    read as a real, very significant p-value) if there's too little history
    or the test itself fails to converge.
    """
    clean = returns.dropna()
    if len(clean) < MIN_OBSERVATIONS_FOR_ADF:
        return {"adf_statistic": None, "p_value": None, "is_stationary": None}
    try:
        statistic, p_value, *_ = adfuller(clean, autolag="AIC")
    except Exception:
        return {"adf_statistic": None, "p_value": None, "is_stationary": None}
    # bool(...) matters here: p_value is numpy.float64, so p_value < 0.05 is a
    # numpy.bool_, not a Python bool — `numpy.bool_(True) is True` is False
    # (an identity check, not a value check), which silently breaks any
    # caller doing `if result["is_stationary"] is True`.
    return {"adf_statistic": float(statistic), "p_value": float(p_value), "is_stationary": bool(p_value < 0.05)}


def hurst_exponent(prices: pd.Series, max_lag: int = 20) -> float:
    """
    Hurst exponent via the variance-of-lagged-differences method: a standard,
    widely-used SIMPLIFIED estimator — log-log regression of the STANDARD
    DEVIATION of lagged price differences against the lag, slope = H directly
    (not full rescaled-range (R/S) analysis, which is more robust but a
    meaningfully bigger implementation). Documented as a simplification
    deliberately, same spirit as other documented trade-offs in this codebase
    (e.g. TF-IDF vs neural embeddings in rag.py).

    Scaling, precisely (this is the part worth getting right): for fractional
    Brownian motion, Var[X(t+lag) - X(t)] scales as lag^(2H) BY DEFINITION —
    so std(diff), which is what `tau` below actually is, scales as lag^H, not
    lag^(2H). Regressing log(std(diff)) against log(lag) therefore recovers H
    directly from the slope. Multiplying that slope by 2 (as an earlier
    version of this function did) silently doubles the result — confirmed
    numerically: a pure random walk (textbook H=0.5) computed as ~0.98 with
    the erroneous ×2, ~0.495 without it. The ×2 factor only belongs in this
    derivation if `tau` were the VARIANCE of lagged differences instead of
    the standard deviation — it is not, here.

    How to read it: H > 0.5 = trending/momentum (a move tends to be followed
    by a move in the same direction); H < 0.5 = mean-reverting (a move tends
    to reverse); H ≈ 0.5 = a random walk — the quantitative, per-asset version
    of the efficient-market-hypothesis caveat already stated throughout this
    app's forecasting UI. Computed on PRICES (not returns) since it measures
    the persistence of the level series itself, the standard convention.

    Requires at least `max_lag * 2` price observations; returns NaN below
    that or if the regression is degenerate (e.g. a perfectly flat series).
    """
    clean = prices.dropna()
    if len(clean) < max_lag * 2:
        return float("nan")

    lags = range(2, max_lag)
    values = clean.values
    tau = [np.std(values[lag:] - values[:-lag]) for lag in lags]
    if any(t <= 0 for t in tau):
        return float("nan")  # degenerate (flat) series — no meaningful trend/reversion signal

    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(poly[0])


def rolling_sharpe(
    returns: pd.Series,
    window: int,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Rolling-window Sharpe ratio — a single aggregate Sharpe over the whole
    sample can hide a regime change (calm-then-crisis, or the reverse). This
    shows how risk-adjusted performance actually evolved period to period,
    which is the direct answer to "decrypt the trend" rather than one
    end-of-sample snapshot. Same annualisation convention as
    `metrics.sharpe_ratio` (period risk-free rate subtracted before
    annualising with `√periods_per_year`), just computed on a rolling window
    instead of the full series. First `window - 1` points are NaN, as usual
    for any rolling statistic.
    """
    period_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns - period_rf
    rolling_mean = excess.rolling(window).mean()
    rolling_std = excess.rolling(window).std(ddof=1)
    return (rolling_mean / rolling_std) * np.sqrt(periods_per_year)