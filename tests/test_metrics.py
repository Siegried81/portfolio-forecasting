
"""Unit tests for src/metrics.py, using small hand-checkable synthetic series."""
import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    annualised_return,
    annualised_volatility,
    calmar_ratio,
    conditional_value_at_risk,
    information_ratio,
    jensens_alpha,
    max_drawdown,
    omega_ratio,
    portfolio_returns,
    return_kurtosis,
    return_skewness,
    sharpe_ratio,
    sharpe_ratio_standard_error,
    sortino_ratio,
    summarise_performance,
    treynor_ratio,
    ulcer_index,
    value_at_risk,
)


def test_portfolio_returns_drops_rows_with_partial_missing_assets():
    asset_returns = pd.DataFrame(
        {"A": [0.01, float("nan"), 0.03], "B": [0.02, 0.01, 0.04]}
    )
    weights = pd.Series({"A": 0.5, "B": 0.5})

    result = portfolio_returns(asset_returns, weights)

    assert len(result) == 2


def test_annualised_return_matches_hand_calculation():
    returns = pd.Series([0.10, 0.10])
    result = annualised_return(returns, periods_per_year=2)
    assert result == pytest.approx(0.21, abs=1e-9)


def test_annualised_return_zero_for_flat_series():
    returns = pd.Series([0.0, 0.0, 0.0])
    assert annualised_return(returns, periods_per_year=252) == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_on_known_path():
    returns = pd.Series([0.20, -0.25, 0.1667])
    dd = max_drawdown(returns)
    assert dd == pytest.approx(-0.25, abs=1e-3)


def test_max_drawdown_is_zero_for_monotonic_gains():
    returns = pd.Series([0.01, 0.02, 0.015])
    assert max_drawdown(returns) == pytest.approx(0.0, abs=1e-9)


def test_sortino_ignores_upside_volatility():
    calm_upside = pd.Series([0.05, -0.01, 0.05, -0.01])
    volatile_upside = pd.Series([0.20, -0.01, 0.20, -0.01])

    sharpe_calm = sharpe_ratio(calm_upside, risk_free_rate=0.0, periods_per_year=4)
    sharpe_volatile = sharpe_ratio(volatile_upside, risk_free_rate=0.0, periods_per_year=4)
    sortino_calm = sortino_ratio(calm_upside, risk_free_rate=0.0, periods_per_year=4)
    sortino_volatile = sortino_ratio(volatile_upside, risk_free_rate=0.0, periods_per_year=4)

    ratio_calm = sortino_calm / sharpe_calm
    ratio_volatile = sortino_volatile / sharpe_volatile
    assert ratio_volatile > ratio_calm


def test_value_at_risk_and_cvar_ordering():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.02, 500))
    var_95 = value_at_risk(returns, confidence=0.95)
    cvar_95 = conditional_value_at_risk(returns, confidence=0.95)
    assert cvar_95 <= var_95


def test_annualised_volatility_scales_with_sqrt_time():
    returns = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01])
    vol_daily = annualised_volatility(returns, periods_per_year=252)
    vol_monthly = annualised_volatility(returns, periods_per_year=12)
    assert vol_daily > vol_monthly


def test_calmar_ratio_manual_calculation():
    returns = pd.Series([0.20, -0.25, 0.1667])
    calmar = calmar_ratio(returns, periods_per_year=3)
    expected = annualised_return(returns, periods_per_year=3) / 0.25
    assert calmar == pytest.approx(expected, rel=1e-6)


def test_calmar_ratio_nan_when_no_drawdown():
    returns = pd.Series([0.01, 0.02, 0.015])
    assert np.isnan(calmar_ratio(returns))


def test_omega_ratio_above_one_for_upside_skewed_returns():
    skewed = pd.Series([0.03, 0.03, 0.03, -0.01, -0.01])
    omega = omega_ratio(skewed, threshold=0.0)
    assert omega > 1.0


def test_omega_ratio_below_one_for_downside_skewed_returns():
    skewed = pd.Series([0.01, 0.01, -0.03, -0.03, -0.03])
    omega = omega_ratio(skewed, threshold=0.0)
    assert omega < 1.0


def test_information_ratio_nan_when_tracking_error_zero():
    benchmark = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01])
    portfolio = benchmark.copy()
    assert np.isnan(information_ratio(portfolio, benchmark, periods_per_year=252))


def test_information_ratio_positive_when_portfolio_outperforms():
    np.random.seed(11)
    benchmark = pd.Series(np.random.normal(0.0004, 0.01, 300))
    portfolio = benchmark + 0.0003
    ir = information_ratio(portfolio, benchmark, periods_per_year=252)
    assert ir > 0


def test_treynor_ratio_manual_calculation():
    treynor = treynor_ratio(pd.Series([0.15]), beta=1.5, risk_free_rate=0.04, periods_per_year=1)
    assert treynor == pytest.approx((0.15 - 0.04) / 1.5, rel=1e-9)


def test_treynor_ratio_nan_for_zero_beta():
    assert np.isnan(treynor_ratio(pd.Series([0.10]), beta=0.0, periods_per_year=1))


def test_summarise_performance_includes_benchmark_metrics_only_when_provided():
    np.random.seed(12)
    returns = pd.Series(np.random.normal(0.0005, 0.012, 200))
    benchmark = pd.Series(np.random.normal(0.0003, 0.010, 200))

    without_benchmark = summarise_performance(returns, periods_per_year=252)
    assert "calmar_ratio" in without_benchmark and "omega_ratio" in without_benchmark
    assert "information_ratio" not in without_benchmark and "beta" not in without_benchmark

    with_benchmark = summarise_performance(returns, periods_per_year=252, benchmark_returns=benchmark)
    assert "information_ratio" in with_benchmark and "treynor_ratio" in with_benchmark and "beta" in with_benchmark
    assert "jensens_alpha" in with_benchmark


def test_jensens_alpha_zero_when_portfolio_exactly_matches_capm_prediction():
    benchmark = pd.Series([0.10])
    beta = 1.5
    rf = 0.0
    portfolio_return = rf + beta * (0.10 - rf)
    portfolio = pd.Series([portfolio_return])
    alpha = jensens_alpha(portfolio, benchmark, beta=beta, risk_free_rate=rf, periods_per_year=1)
    assert alpha == pytest.approx(0.0, abs=1e-9)


def test_jensens_alpha_positive_when_outperforming_capm_prediction():
    benchmark = pd.Series([0.10])
    beta = 1.0
    rf = 0.0
    portfolio = pd.Series([0.14])
    alpha = jensens_alpha(portfolio, benchmark, beta=beta, risk_free_rate=rf, periods_per_year=1)
    assert alpha == pytest.approx(0.04, abs=1e-9)


def test_jensens_alpha_nan_when_beta_is_nan():
    benchmark = pd.Series([0.10, 0.05])
    portfolio = pd.Series([0.08, 0.03])
    assert np.isnan(jensens_alpha(portfolio, benchmark, beta=float("nan"), periods_per_year=1))


# ---------------------------------------------------------------------------
# ulcer_index
# ---------------------------------------------------------------------------

def test_ulcer_index_zero_for_monotonic_gains():
    returns = pd.Series([0.01, 0.02, 0.015])
    assert ulcer_index(returns) == pytest.approx(0.0, abs=1e-9)


def test_ulcer_index_distinguishes_duration_from_max_drawdown():
    long_underwater = pd.Series([0.05, -0.20, 0.0, 0.0, 0.15])
    quick_recovery = pd.Series([0.05, -0.20, 0.15, 0.0, 0.0])
    ui_long = ulcer_index(long_underwater)
    ui_quick = ulcer_index(quick_recovery)
    assert max_drawdown(long_underwater) == pytest.approx(max_drawdown(quick_recovery), abs=1e-6)
    assert ui_long > ui_quick


# ---------------------------------------------------------------------------
# return_skewness / return_kurtosis
# ---------------------------------------------------------------------------

def test_return_skewness_negative_for_left_tailed_series():
    returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, -0.15])
    assert return_skewness(returns) < 0


def test_return_skewness_positive_for_right_tailed_series():
    returns = pd.Series([-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, 0.15])
    assert return_skewness(returns) > 0


def test_return_kurtosis_higher_for_fatter_tailed_series():
    normal_ish = pd.Series([0.00, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.00])
    fat_tailed = pd.Series([0.00, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.25])
    assert return_kurtosis(fat_tailed) > return_kurtosis(normal_ish)


# ---------------------------------------------------------------------------
# sharpe_ratio_standard_error
# ---------------------------------------------------------------------------

def test_sharpe_se_matches_lo_2002_formula_at_unit_frequency():
    returns = pd.Series([0.10, -0.05, 0.08, -0.02, 0.03])
    sr = sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=1)
    se = sharpe_ratio_standard_error(returns, risk_free_rate=0.0, periods_per_year=1)
    expected = np.sqrt((1.0 + 0.5 * sr ** 2) / len(returns))
    assert se == pytest.approx(expected, rel=1e-9)


def test_sharpe_se_scales_with_sqrt_periods_per_year():
    returns = pd.Series([0.02, -0.01, 0.015, -0.005, 0.01, -0.02, 0.03])
    se_unit = sharpe_ratio_standard_error(returns, risk_free_rate=0.0, periods_per_year=1)
    se_annual = sharpe_ratio_standard_error(returns, risk_free_rate=0.0, periods_per_year=252)
    assert se_annual == pytest.approx(se_unit * np.sqrt(252), rel=1e-9)


def test_sharpe_se_shrinks_with_more_observations():
    np.random.seed(7)
    short_series = pd.Series(np.random.normal(0.0005, 0.01, 30))
    long_series = pd.concat([short_series] * 10, ignore_index=True)
    se_short = sharpe_ratio_standard_error(short_series, risk_free_rate=0.0, periods_per_year=252)
    se_long = sharpe_ratio_standard_error(long_series, risk_free_rate=0.0, periods_per_year=252)
    assert se_long < se_short


def test_sharpe_se_nan_for_fewer_than_two_observations():
    assert np.isnan(sharpe_ratio_standard_error(pd.Series([0.01]), periods_per_year=252))


def test_summarise_performance_includes_sharpe_se():
    np.random.seed(21)
    returns = pd.Series(np.random.normal(0.0005, 0.012, 100))
    summary = summarise_performance(returns, periods_per_year=252)
    assert "sharpe_se" in summary
    assert summary["sharpe_se"] > 0


def test_summarise_performance_always_includes_ulcer_skew_kurtosis():
    np.random.seed(21)
    returns = pd.Series(np.random.normal(0.0005, 0.012, 100))
    summary = summarise_performance(returns, periods_per_year=252)
    assert "ulcer_index" in summary
    assert "skewness" in summary
    assert "kurtosis" in summary
    assert summary["ulcer_index"] >= 0