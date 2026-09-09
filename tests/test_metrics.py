"""
Unit tests for the pure finance calculations in src/metrics.py.

These are worth having precisely BECAUSE this is a finance app: a silently wrong
Sharpe/drawdown formula is much more damaging (and much harder to spot by eye)
than a UI bug. Tests use small hand-checkable synthetic series rather than real
market data, so expected values can be verified by hand, not just "does it run".
"""
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


def test_annualised_return_matches_hand_calculation():
    # 2 periods of +10% each, compounded: 1.1 * 1.1 - 1 = 0.21
    returns = pd.Series([0.10, 0.10])
    result = annualised_return(returns, periods_per_year=2)
    assert result == pytest.approx(0.21, abs=1e-9)


def test_annualised_return_zero_for_flat_series():
    returns = pd.Series([0.0, 0.0, 0.0])
    assert annualised_return(returns, periods_per_year=252) == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_on_known_path():
    # Wealth path: 1.0 -> 1.2 -> 0.9 -> 1.05  => trough 0.9 vs peak 1.2 => -25%
    returns = pd.Series([0.20, -0.25, 0.1667])
    dd = max_drawdown(returns)
    assert dd == pytest.approx(-0.25, abs=1e-3)


def test_max_drawdown_is_zero_for_monotonic_gains():
    returns = pd.Series([0.01, 0.02, 0.015])
    assert max_drawdown(returns) == pytest.approx(0.0, abs=1e-9)


def test_sortino_ignores_upside_volatility():
    # Same mean/vol profile as Sharpe would see, but Sortino should differ because
    # only the negative deviations count towards the denominator.
    calm_upside = pd.Series([0.05, -0.01, 0.05, -0.01])  # big upside swings, small downside
    volatile_upside = pd.Series([0.20, -0.01, 0.20, -0.01])  # even bigger upside swings, same downside

    sharpe_calm = sharpe_ratio(calm_upside, risk_free_rate=0.0, periods_per_year=4)
    sharpe_volatile = sharpe_ratio(volatile_upside, risk_free_rate=0.0, periods_per_year=4)
    sortino_calm = sortino_ratio(calm_upside, risk_free_rate=0.0, periods_per_year=4)
    sortino_volatile = sortino_ratio(volatile_upside, risk_free_rate=0.0, periods_per_year=4)

    # Both series share the EXACT SAME downside observations (-0.01, -0.01), only
    # the upside differs. Sharpe's denominator (total std dev) grows with the bigger
    # upside swings, while Sortino's denominator (downside deviation only) does not.
    # So Sortino should reward the bigger upside relatively MORE than Sharpe does -
    # i.e. the Sortino/Sharpe ratio should be higher for the more upside-volatile series.
    ratio_calm = sortino_calm / sharpe_calm
    ratio_volatile = sortino_volatile / sharpe_volatile
    assert ratio_volatile > ratio_calm


def test_value_at_risk_and_cvar_ordering():
    # CVaR (average of the tail) must always be <= VaR (the tail threshold itself)
    # for a left-tail loss convention, since the tail average includes worse outcomes.
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.02, 500))
    var_95 = value_at_risk(returns, confidence=0.95)
    cvar_95 = conditional_value_at_risk(returns, confidence=0.95)
    assert cvar_95 <= var_95


def test_annualised_volatility_scales_with_sqrt_time():
    returns = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01])
    vol_daily = annualised_volatility(returns, periods_per_year=252)
    vol_monthly = annualised_volatility(returns, periods_per_year=12)
    # Same raw series, higher periods_per_year -> higher annualised vol (sqrt scaling)
    assert vol_daily > vol_monthly


def test_calmar_ratio_manual_calculation():
    # Wealth path 1.0 -> 1.2 -> 0.9 -> 1.05 => max drawdown -25% (same series as
    # the max_drawdown test above). Annualised return over 3 periods/year=3 is
    # the compounded growth itself: 1.05 - 1 = 0.05 -> Calmar = 0.05 / 0.25 = 0.2
    returns = pd.Series([0.20, -0.25, 0.1667])
    calmar = calmar_ratio(returns, periods_per_year=3)
    expected = annualised_return(returns, periods_per_year=3) / 0.25
    assert calmar == pytest.approx(expected, rel=1e-6)


def test_calmar_ratio_nan_when_no_drawdown():
    returns = pd.Series([0.01, 0.02, 0.015])  # monotonic gains -> max_drawdown = 0
    assert np.isnan(calmar_ratio(returns))


def test_omega_ratio_above_one_for_upside_skewed_returns():
    # Big consistent gains, small consistent losses -> Omega should clearly exceed 1
    skewed = pd.Series([0.03, 0.03, 0.03, -0.01, -0.01])
    omega = omega_ratio(skewed, threshold=0.0)
    assert omega > 1.0


def test_omega_ratio_below_one_for_downside_skewed_returns():
    skewed = pd.Series([0.01, 0.01, -0.03, -0.03, -0.03])
    omega = omega_ratio(skewed, threshold=0.0)
    assert omega < 1.0


def test_information_ratio_nan_when_tracking_error_zero():
    # Portfolio identical to its benchmark -> zero tracking error -> undefined IR
    benchmark = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01])
    portfolio = benchmark.copy()
    assert np.isnan(information_ratio(portfolio, benchmark, periods_per_year=252))


def test_information_ratio_positive_when_portfolio_outperforms():
    np.random.seed(11)
    benchmark = pd.Series(np.random.normal(0.0004, 0.01, 300))
    portfolio = benchmark + 0.0003  # small but real, noisy-free outperformance each period
    ir = information_ratio(portfolio, benchmark, periods_per_year=252)
    assert ir > 0


def test_treynor_ratio_manual_calculation():
    # Single-period annual return 15%, risk-free 4%, beta 1.5 -> (0.15-0.04)/1.5
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
    # Construct a portfolio whose return is EXACTLY rf + beta*(Rm - rf) at every
    # period (beta=1.5, rf=0 for a clean single-period hand check) -> alpha must be 0.
    benchmark = pd.Series([0.10])  # single period, 10% benchmark return
    beta = 1.5
    rf = 0.0
    portfolio_return = rf + beta * (0.10 - rf)  # = 0.15
    portfolio = pd.Series([portfolio_return])
    alpha = jensens_alpha(portfolio, benchmark, beta=beta, risk_free_rate=rf, periods_per_year=1)
    assert alpha == pytest.approx(0.0, abs=1e-9)


def test_jensens_alpha_positive_when_outperforming_capm_prediction():
    benchmark = pd.Series([0.10])
    beta = 1.0
    rf = 0.0
    # CAPM would predict exactly the benchmark return (beta=1) -> anything above it is alpha
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
    # Wealth curve never dips below a prior peak -> every drawdown point is 0.
    returns = pd.Series([0.01, 0.02, 0.015])
    assert ulcer_index(returns) == pytest.approx(0.0, abs=1e-9)


def test_ulcer_index_distinguishes_duration_from_max_drawdown():
    # Both series first rise (establishing an explicit peak WITHIN the series —
    # max_drawdown()'s running peak comes from cummax() over the series itself,
    # so a drop on period 0 has no prior peak to be measured against and would
    # score a false zero). Both then drop 20% from that peak, share the exact
    # same Max Drawdown — but the first stays underwater for 3 periods while
    # the second recovers immediately.
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
    # Many small gains, one large loss -> classic negative-skew equity shape.
    returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, -0.15])
    assert return_skewness(returns) < 0


def test_return_skewness_positive_for_right_tailed_series():
    returns = pd.Series([-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, 0.15])
    assert return_skewness(returns) > 0


def test_return_kurtosis_higher_for_fatter_tailed_series():
    # Same mean and similar spread, but one series has an extreme outlier
    # (fat tail) the other doesn't -> higher excess kurtosis for the outlier series.
    normal_ish = pd.Series([0.00, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.00])
    fat_tailed = pd.Series([0.00, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.25])
    assert return_kurtosis(fat_tailed) > return_kurtosis(normal_ish)


# ---------------------------------------------------------------------------
# sharpe_ratio_standard_error
# ---------------------------------------------------------------------------

def test_sharpe_se_matches_lo_2002_formula_at_unit_frequency():
    # At periods_per_year=1, "per-period" and "annualised" Sharpe coincide, so
    # the formula SE = sqrt((1 + 0.5*SR^2) / n) can be checked directly against
    # sharpe_ratio()'s own output rather than a separately hand-computed number.
    returns = pd.Series([0.10, -0.05, 0.08, -0.02, 0.03])
    sr = sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=1)
    se = sharpe_ratio_standard_error(returns, risk_free_rate=0.0, periods_per_year=1)
    expected = np.sqrt((1.0 + 0.5 * sr ** 2) / len(returns))
    assert se == pytest.approx(expected, rel=1e-9)


def test_sharpe_se_scales_with_sqrt_periods_per_year():
    # Same per-period excess returns (risk_free_rate=0 makes the period-rf
    # conversion a no-op regardless of periods_per_year) -> annualising SE by
    # sqrt(m) must match the same scaling sharpe_ratio() itself uses.
    returns = pd.Series([0.02, -0.01, 0.015, -0.005, 0.01, -0.02, 0.03])
    se_unit = sharpe_ratio_standard_error(returns, risk_free_rate=0.0, periods_per_year=1)
    se_annual = sharpe_ratio_standard_error(returns, risk_free_rate=0.0, periods_per_year=252)
    assert se_annual == pytest.approx(se_unit * np.sqrt(252), rel=1e-9)


def test_sharpe_se_shrinks_with_more_observations():
    # More observations of the same underlying process -> lower estimation
    # uncertainty on the Sharpe ratio itself, holding the process fixed.
    np.random.seed(7)
    short_series = pd.Series(np.random.normal(0.0005, 0.01, 30))
    long_series = pd.concat([short_series] * 10, ignore_index=True)  # same SR, 10x the observations
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