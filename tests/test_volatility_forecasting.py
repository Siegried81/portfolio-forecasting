"""Unit tests for src/volatility_forecasting.py."""
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
# garch_forecast_variance
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
    constant_returns = pd.Series([0.0] * (MIN_HISTORY_POINTS_FOR_GARCH + 20))
    assert garch_forecast_variance(constant_returns, horizon_periods=30) is None


# ---------------------------------------------------------------------------
# garch_forecast_cov
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
    prices.loc[prices.index[:-50], "SHORT"] = np.nan

    cov, diagnostics = garch_forecast_cov(prices, horizon_periods=30, periods_per_year=252)

    assert diagnostics["n_assets_via_garch"] == 1
    assert diagnostics["n_assets_via_fallback"] == 1
    assert diagnostics["fallback_tickers"] == ["SHORT"]
    assert not cov.isna().any().any()


def test_garch_forecast_cov_preserves_historical_correlation_structure():
    prices = _price_panel(400, ["AAA", "BBB", "CCC"], seed=5, correlated=True)
    cov, _ = garch_forecast_cov(prices, horizon_periods=30, periods_per_year=252)
    vols = np.sqrt(np.diag(cov.values))
    correlation = cov.values / np.outer(vols, vols)
    off_diagonal = correlation[np.triu_indices_from(correlation, k=1)]
    assert (off_diagonal > 0.3).all()


def test_garch_forecast_cov_columns_match_input_tickers_in_order():
    prices = _price_panel(300, ["ZZZ", "AAA", "MMM"], seed=6)
    cov, _ = garch_forecast_cov(prices, horizon_periods=30, periods_per_year=252)
    assert list(cov.columns) == ["ZZZ", "AAA", "MMM"]
    assert list(cov.index) == ["ZZZ", "AAA", "MMM"]


def test_garch_forecast_cov_scales_with_periods_per_year():
    prices = _price_panel(300, ["AAA", "BBB"], seed=7, correlated=True)
    cov_daily, _ = garch_forecast_cov(prices, horizon_periods=30, periods_per_year=252)
    cov_weekly, _ = garch_forecast_cov(prices, horizon_periods=30, periods_per_year=52)
    ratio = cov_daily.values / cov_weekly.values
    assert np.allclose(ratio, 252 / 52, atol=1e-6)