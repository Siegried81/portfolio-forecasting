"""
Unit tests for src/timeseries_diagnostics.py.

Priority: does the ADF test correctly distinguish a stationary series from a
random walk (the exact assumption forecasting.py's ARIMA d=1 choice rests
on), and does the Hurst exponent correctly separate a trending series from a
mean-reverting one — the two things this module exists to check.
"""
import numpy as np
import pandas as pd
import pytest

from src.timeseries_diagnostics import adf_stationarity_test, hurst_exponent, rolling_sharpe


# ---------------------------------------------------------------------------
# adf_stationarity_test
# ---------------------------------------------------------------------------

def test_adf_detects_stationary_series():
    # White noise is the textbook stationary series — no unit root.
    rng = np.random.default_rng(0)
    stationary_returns = pd.Series(rng.normal(0, 0.01, 300))
    result = adf_stationarity_test(stationary_returns)
    assert result["is_stationary"] is True
    assert result["p_value"] < 0.05


def test_adf_detects_non_stationary_random_walk():
    # A cumulative sum of white noise is the textbook random walk — has a
    # unit root, should NOT reject the null (p >= 0.05 typically).
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
    # A deterministic linear trend + i.i.d. noise is NOT the right test case
    # here (verified numerically): this variance-of-lagged-differences
    # estimator measures self-similarity of INCREMENTS (the fBm model), and a
    # pure linear trend's lagged differences don't grow with lag the way a
    # persistent stochastic process's do. Build genuine persistence instead:
    # increments with positive autocorrelation (momentum), the actual
    # definition of a trending/persistent series this estimator targets.
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
    # An Ornstein-Uhlenbeck-style mean-reverting process should score H below 0.5.
    rng = np.random.default_rng(3)
    n = 300
    prices = np.zeros(n)
    prices[0] = 100.0
    theta, mu = 0.3, 100.0  # strong pull back to the mean
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
    # REGRESSION test for a magnitude bug (not just sign): a previous version
    # of this function multiplied the log-log regression slope by 2.0 as if
    # `tau` were the VARIANCE of lagged differences, when it's actually the
    # STANDARD DEVIATION — for fBm, std(diff) already scales as lag^H, so the
    # slope IS H directly and doubling it silently inflated every result
    # (confirmed: a pure random walk computed as H~0.98 with the bug, ~0.5
    # without it). The trending/mean-reverting tests above only check the
    # SIGN relative to 0.5, which a doubled value can still pass — this test
    # checks the actual magnitude against the one case with a known textbook
    # answer (H=0.5 exactly, by definition, for Brownian motion). Averaged
    # over several seeds since any single random walk is noisy.
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
    # Calm first half, volatile/negative second half -> rolling Sharpe should
    # be visibly higher in the first half than the second.
    calm = pd.Series(np.random.default_rng(5).normal(0.001, 0.003, 100))
    stressed = pd.Series(np.random.default_rng(6).normal(-0.002, 0.03, 100))
    combined = pd.concat([calm, stressed], ignore_index=True)
    rs = rolling_sharpe(combined, window=20, risk_free_rate=0.0, periods_per_year=252)
    calm_avg = rs.iloc[20:100].mean()
    stressed_avg = rs.iloc[120:200].mean()
    assert calm_avg > stressed_avg