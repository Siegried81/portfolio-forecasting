"""Unit tests for src/timeseries_diagnostics.py."""
import numpy as np
import pandas as pd
import pytest

from src.timeseries_diagnostics import adf_stationarity_test, hurst_exponent, rolling_sharpe


# ---------------------------------------------------------------------------
# adf_stationarity_test
# ---------------------------------------------------------------------------

def test_adf_detects_stationary_series():
    rng = np.random.default_rng(0)
    stationary_returns = pd.Series(rng.normal(0, 0.01, 300))
    result = adf_stationarity_test(stationary_returns)
    assert result["is_stationary"] is True
    assert result["p_value"] < 0.05


def test_adf_detects_non_stationary_random_walk():
    rng = np.random.default_rng(1)
    random_walk = pd.Series(np.cumsum(rng.normal(0, 1, 300)))
    result = adf_stationarity_test(random_walk)
    assert result["is_stationary"] is False


def test_adf_none_with_too_little_history():
    short_series = pd.Series([0.01, -0.01, 0.02])
    result = adf_stationarity_test(short_series)
    assert result["p_value"] is None
    assert result["is_stationary"] is None


# ---------------------------------------------------------------------------
# hurst_exponent
# ---------------------------------------------------------------------------

def test_hurst_detects_trending_series():
    # Positive-autocorrelation increments (momentum), not a deterministic
    # linear trend — this estimator measures self-similarity of increments.
    rng = np.random.default_rng(2)
    n = 500
    increments = np.zeros(n)
    noise = rng.normal(0, 1, n)
    for t in range(1, n):
        increments[t] = 0.4 * increments[t - 1] + noise[t]
    trending_prices = pd.Series(np.cumsum(increments) + 100)
    h = hurst_exponent(trending_prices)
    assert h > 0.5


def test_hurst_detects_mean_reverting_series():
    rng = np.random.default_rng(3)
    n = 300
    prices = np.zeros(n)
    prices[0] = 100.0
    theta, mu = 0.3, 100.0
    for t in range(1, n):
        prices[t] = prices[t - 1] + theta * (mu - prices[t - 1]) + rng.normal(0, 0.5)
    h = hurst_exponent(pd.Series(prices))
    assert h < 0.5


def test_hurst_nan_with_too_little_history():
    short_prices = pd.Series([100.0, 101.0, 99.0])
    assert np.isnan(hurst_exponent(short_prices, max_lag=20))


def test_hurst_nan_for_flat_series():
    flat_prices = pd.Series([100.0] * 100)
    assert np.isnan(hurst_exponent(flat_prices))


def test_hurst_close_to_half_for_a_pure_random_walk():
    # For fBm, std(diff) scales as lag^H (not lag^(2H)) — H=0.5 for a
    # textbook random walk. Averaged over several seeds.
    hs = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        random_walk = pd.Series(np.cumsum(rng.normal(0, 1, 500)) + 1000)
        hs.append(hurst_exponent(random_walk, max_lag=20))
    assert np.mean(hs) == pytest.approx(0.5, abs=0.1)


# ---------------------------------------------------------------------------
# rolling_sharpe
# ---------------------------------------------------------------------------

def test_rolling_sharpe_has_expected_nan_prefix_and_length():
    returns = pd.Series(np.random.default_rng(4).normal(0.0005, 0.01, 100))
    window = 20
    rs = rolling_sharpe(returns, window=window, risk_free_rate=0.0, periods_per_year=252)
    assert len(rs) == len(returns)
    assert rs.iloc[: window - 1].isna().all()
    assert not rs.iloc[window:].isna().all()


def test_rolling_sharpe_flags_a_regime_change():
    calm = pd.Series(np.random.default_rng(5).normal(0.001, 0.003, 100))
    stressed = pd.Series(np.random.default_rng(6).normal(-0.002, 0.03, 100))
    combined = pd.concat([calm, stressed], ignore_index=True)
    rs = rolling_sharpe(combined, window=20, risk_free_rate=0.0, periods_per_year=252)
    calm_avg = rs.iloc[20:100].mean()
    stressed_avg = rs.iloc[120:200].mean()
    assert calm_avg > stressed_avg