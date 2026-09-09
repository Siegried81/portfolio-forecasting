"""
Unit tests for src/volatility_forecasting.py.

Priority, same spirit as test_forecasting.py's ARIMA tests: the FALLBACK path
matters more than the happy path, since a silently-skipped fallback here
would be invisible (the app would just look like it's using GARCH when it's
actually degraded). Also covers the correctness properties every covariance
matrix in this codebase is expected to have (symmetric, PSD — same checks
test_factor_models.py already applies to the PCA covariance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import MIN_HISTORY_POINTS_FOR_GARCH
from src.volatility_forecasting import garch_forecast_cov, garch_forecast_variance


def _price_panel(n_periods: int, tickers: list[str], seed: int = 0, correlated: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-01", periods=n_periods)
    if correlated:
        common = rng.normal(0, 0.01, n_periods)
        idio = rng.normal(0, 0.005, (n_periods, len(tickers)))
        returns = common[:, None] * 0.8 + idio
    else:
        returns = rng.normal(0.0003, 0.015, (n_periods, len(tickers)))
    prices = 100 * np.cumprod(1 + returns, axis=0)
    return pd.DataFrame(prices, columns=tickers, index=idx)


# ---------------------------------------------------------------------------
# garch_forecast_variance — the per-asset building block
# ---------------------------------------------------------------------------

def test_garch_forecast_variance_returns_none_on_short_history():
    short_returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, MIN_HISTORY_POINTS_FOR_GARCH - 10))
    assert garch_forecast_variance(short_returns, horizon_periods=30) is None


def test_garch_forecast_variance_returns_a_positive_number_on_sufficient_history():
    returns = pd.Series(np.random.default_rng(1).normal(0.0003, 0.015, 300))
    variance = garch_forecast_variance(returns, horizon_periods=30)
    assert variance is not None
    assert variance > 0


def test_garch_forecast_variance_none_on_non_convergent_series():
    # A perfectly constant series (zero variance throughout) is the same kind
    # of degenerate input that makes ARIMA fail to converge in
    # test_forecasting.py — GARCH has nothing to fit here either.
    constant_returns = pd.Series([0.0] * (MIN_HISTORY_POINTS_FOR_GARCH + 20))
    assert garch_forecast_variance(constant_returns, horizon_periods=30) is None


# ---------------------------------------------------------------------------
# garch_forecast_cov — full covariance matrix
# ---------------------------------------------------------------------------

def test_garch_forecast_cov_returns_symmetric_psd_matrix():
    prices = _price_panel(300, ["AAA", "BBB", "CCC"], seed=2)
    cov, _ = garch_forecast_cov(prices, horizon_periods=30, periods_per_year=252)
    values = cov.values
    assert np.allclose(values, values.T, atol=1e-8)
    eigenvalues = np.linalg.eigvalsh(values)
    assert (eigenvalues >= -1e-6).all()


def test_garch_forecast_cov_diagnostics_report_all_assets_via_garch_with_enough_history():
    prices = _price_panel(300, ["AAA", "BBB"], seed=3)
    _, diagnostics = garch_forecast_cov(prices, horizon_periods=30, periods_per_year=252)
    assert diagnostics["n_assets_via_garch"] == 2
    assert diagnostics["n_assets_via_fallback"] == 0
    assert diagnostics["fallback_tickers"] == []


def test_garch_forecast_cov_falls_back_for_a_short_history_asset_without_crashing():
    prices = _price_panel(300, ["AAA", "SHORT"], seed=4)
    # Simulate SHORT being newly listed: no price history before the last 50 rows.
    prices.loc[prices.index[:-50], "SHORT"] = np.nan

    cov, diagnostics = garch_forecast_cov(prices, horizon_periods=30, periods_per_year=252)

    assert diagnostics["n_assets_via_garch"] == 1
    assert diagnostics["n_assets_via_fallback"] == 1
    assert diagnostics["fallback_tickers"] == ["SHORT"]
    # Must still produce a complete, valid (non-NaN) covariance matrix.
    assert not cov.isna().any().any()


def test_garch_forecast_cov_preserves_historical_correlation_structure():
    # Correlated synthetic data (shared common factor) -> the forecasted
    # covariance's correlation should clearly reflect that, since correlation
    # is deliberately still historical (see the module's own docstring).
    prices = _price_panel(400, ["AAA", "BBB", "CCC"], seed=5, correlated=True)
    cov, _ = garch_forecast_cov(prices, horizon_periods=30, periods_per_year=252)
    vols = np.sqrt(np.diag(cov.values))
    correlation = cov.values / np.outer(vols, vols)
    off_diagonal = correlation[np.triu_indices_from(correlation, k=1)]
    assert (off_diagonal > 0.3).all()  # clearly correlated, not near-zero


def test_garch_forecast_cov_columns_match_input_tickers_in_order():
    prices = _price_panel(300, ["ZZZ", "AAA", "MMM"], seed=6)
    cov, _ = garch_forecast_cov(prices, horizon_periods=30, periods_per_year=252)
    assert list(cov.columns) == ["ZZZ", "AAA", "MMM"]
    assert list(cov.index) == ["ZZZ", "AAA", "MMM"]


def test_garch_forecast_cov_scales_with_periods_per_year():
    # Same underlying data, different annualisation factor -> daily-scaled
    # covariance should be exactly 252/52 times the weekly-scaled one, same
    # invariant test_factor_models.py applies to the PCA covariance.
    # Correlated data deliberately (not the default uncorrelated fixture):
    # with weak/no true correlation, Ledoit-Wolf shrinkage can legitimately
    # shrink an off-diagonal all the way to exactly 0, which would make the
    # ratio below a spurious 0/0 — a test-fixture issue, not a bug to guard
    # against here.
    prices = _price_panel(300, ["AAA", "BBB"], seed=7, correlated=True)
    cov_daily, _ = garch_forecast_cov(prices, horizon_periods=30, periods_per_year=252)
    cov_weekly, _ = garch_forecast_cov(prices, horizon_periods=30, periods_per_year=52)
    ratio = cov_daily.values / cov_weekly.values
    assert np.allclose(ratio, 252 / 52, atol=1e-6)