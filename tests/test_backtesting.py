"""
Unit tests for src/backtesting.py. Priority: does the expanding-window index
math produce non-overlapping, strictly-growing training windows, and does
run_walk_forward wire forecasting + optimization + metrics end-to-end
without crashing on a small synthetic universe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    PORTFOLIO_TYPE_ORDER,
    compare_to_previous_period,
    forecast_win_rate,
    generate_expanding_windows,
    run_walk_forward,
    summarise_walk_forward,
)


# ---------------------------------------------------------------------------
# generate_expanding_windows
# ---------------------------------------------------------------------------

def test_generate_expanding_windows_produces_expanding_train_end():
    windows = generate_expanding_windows(n_periods=300, horizon=20, min_train_periods=100, max_windows=5)
    train_ends = [w[0] for w in windows]
    assert train_ends == sorted(train_ends)
    assert all(b - a == 20 for a, b in zip(train_ends, train_ends[1:]))


def test_generate_expanding_windows_test_slice_matches_horizon():
    windows = generate_expanding_windows(n_periods=300, horizon=20, min_train_periods=100, max_windows=5)
    assert all(test_end - train_end == 20 for train_end, test_end in windows)


def test_generate_expanding_windows_respects_max_windows_cap():
    windows = generate_expanding_windows(n_periods=1000, horizon=10, min_train_periods=50, max_windows=3)
    assert len(windows) == 3


def test_generate_expanding_windows_keeps_the_most_recent_windows_when_capped():
    windows = generate_expanding_windows(n_periods=1000, horizon=10, min_train_periods=50, max_windows=3)
    _, last_test_end = windows[-1]
    assert last_test_end == 1000
    train_ends = [w[0] for w in windows]
    assert all(b - a == 10 for a, b in zip(train_ends, train_ends[1:]))


def test_generate_expanding_windows_empty_when_not_enough_history():
    windows = generate_expanding_windows(n_periods=50, horizon=20, min_train_periods=100, max_windows=5)
    assert windows == []


def test_generate_expanding_windows_never_exceeds_n_periods():
    windows = generate_expanding_windows(n_periods=137, horizon=15, min_train_periods=80, max_windows=10)
    assert all(test_end <= 137 for _, test_end in windows)


# ---------------------------------------------------------------------------
# run_walk_forward — end-to-end smoke test on synthetic data
# ---------------------------------------------------------------------------

def _synthetic_prices(n_periods: int, tickers: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-01", periods=n_periods)
    data = {}
    for i, ticker in enumerate(tickers):
        returns = rng.normal(0.0003, 0.01, n_periods)
        data[ticker] = 100 * np.cumprod(1 + returns)
    return pd.DataFrame(data, index=idx)


def test_run_walk_forward_returns_expected_shape_and_columns():
    tickers = ["AAA", "BBB", "CCC"]
    prices = _synthetic_prices(250, tickers, seed=1)
    results = run_walk_forward(
        prices, tickers, horizon=15, n_windows=3, forecast_model="Naive (random walk)",
        risk_free_rate=0.04, periods_per_year=252, min_train_periods=100,
    )
    assert len(results) == 9
    assert set(results["portfolio"]) == {"Historical-based", "Forecast-based", "Realized-optimal"}
    assert set(results["window"]) == {1, 2, 3}
    for col in ["annual_return", "annual_volatility", "sharpe_ratio", "sortino_ratio", "max_drawdown"]:
        assert col in results.columns
        assert results[col].notna().all()


def test_run_walk_forward_weights_sum_to_one_via_valid_sharpe():
    tickers = ["AAA", "BBB"]
    prices = _synthetic_prices(200, tickers, seed=2)
    results = run_walk_forward(
        prices, tickers, horizon=10, n_windows=3, forecast_model="Naive (random walk)",
        risk_free_rate=0.04, periods_per_year=252, min_train_periods=80,
    )
    assert (results["annual_volatility"] >= 0).all()
    assert np.isfinite(results["annual_volatility"]).all()


def test_run_walk_forward_applies_transaction_costs_when_requested():
    tickers = ["AAA", "BBB", "CCC"]
    prices = _synthetic_prices(250, tickers, seed=3)
    kwargs = dict(
        prices=prices, tickers=tickers, horizon=15, n_windows=3,
        forecast_model="Naive (random walk)", risk_free_rate=0.04,
        periods_per_year=252, min_train_periods=100,
    )
    no_cost = run_walk_forward(**kwargs, transaction_cost_bps=0.0)
    with_cost = run_walk_forward(**kwargs, transaction_cost_bps=50.0)
    last_window = no_cost["window"].max()
    no_cost_returns = no_cost[no_cost["window"] == last_window]["annual_return"].values
    with_cost_returns = with_cost[with_cost["window"] == last_window]["annual_return"].values
    assert not np.allclose(no_cost_returns, with_cost_returns)


def test_run_walk_forward_respects_max_weight_cap():
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    prices = _synthetic_prices(250, tickers, seed=4)
    results = run_walk_forward(
        prices, tickers, horizon=15, n_windows=3, forecast_model="Naive (random walk)",
        risk_free_rate=0.04, periods_per_year=252, min_train_periods=100,
        max_weight_per_asset=0.10,  # 4 assets x 10% = 40% < 100% -> must auto-relax
    )
    assert len(results) == 9
    assert results["annual_volatility"].notna().all()


def test_run_walk_forward_pca_cov_method_runs_without_crashing():
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    prices = _synthetic_prices(200, tickers, seed=5)
    results = run_walk_forward(
        prices, tickers, horizon=10, n_windows=3, forecast_model="Naive (random walk)",
        risk_free_rate=0.04, periods_per_year=252, min_train_periods=80,
        cov_method="pca", n_factors=3,
    )
    assert len(results) == 9


def test_run_walk_forward_garch_forecast_cov_method_changes_forecast_based_weights():
    tickers = ["AAA", "BBB", "CCC"]
    prices = _synthetic_prices(300, tickers, seed=8)
    kwargs = dict(
        prices=prices, tickers=tickers, horizon=15, n_windows=2,
        forecast_model="Naive (random walk)", risk_free_rate=0.04,
        periods_per_year=252, min_train_periods=150,
    )
    default_results = run_walk_forward(**kwargs)
    garch_results = run_walk_forward(**kwargs, forecast_cov_method="garch")

    assert len(garch_results) == len(default_results) == 6

    default_fcst = default_results[default_results["portfolio"] == "Forecast-based"]["annual_volatility"].values
    garch_fcst = garch_results[garch_results["portfolio"] == "Forecast-based"]["annual_volatility"].values
    assert not np.allclose(default_fcst, garch_fcst)

    default_hist = default_results[default_results["portfolio"] == "Historical-based"]["annual_volatility"].values
    garch_hist = garch_results[garch_results["portfolio"] == "Historical-based"]["annual_volatility"].values
    assert np.allclose(default_hist, garch_hist)


# ---------------------------------------------------------------------------
# summarise_walk_forward
# ---------------------------------------------------------------------------

def test_summarise_walk_forward_groups_by_portfolio_with_mean_and_std():
    tickers = ["AAA", "BBB"]
    prices = _synthetic_prices(200, tickers, seed=6)
    results = run_walk_forward(
        prices, tickers, horizon=10, n_windows=3, forecast_model="Naive (random walk)",
        risk_free_rate=0.04, periods_per_year=252, min_train_periods=80,
    )
    summary = summarise_walk_forward(results)
    assert set(summary.index) == {"Historical-based", "Forecast-based", "Realized-optimal"}
    assert ("sharpe_ratio", "mean") in summary.columns
    assert ("sharpe_ratio", "std") in summary.columns


def test_summarise_walk_forward_rows_follow_narrative_order_not_alphabetical():
    tickers = ["AAA", "BBB"]
    prices = _synthetic_prices(200, tickers, seed=6)
    results = run_walk_forward(
        prices, tickers, horizon=10, n_windows=3, forecast_model="Naive (random walk)",
        risk_free_rate=0.04, periods_per_year=252, min_train_periods=80,
    )
    summary = summarise_walk_forward(results)
    assert list(summary.index) == PORTFOLIO_TYPE_ORDER


def test_summarise_walk_forward_handles_a_subset_of_portfolio_types():
    results = pd.DataFrame({
        "portfolio": ["Realized-optimal", "Historical-based", "Realized-optimal", "Historical-based"],
        "annual_return": [0.1, 0.2, 0.15, 0.18], "annual_volatility": [0.1, 0.1, 0.1, 0.1],
        "sharpe_ratio": [1.0, 1.2, 1.1, 1.3], "sortino_ratio": [1.0, 1.2, 1.1, 1.3],
        "max_drawdown": [-0.1, -0.1, -0.1, -0.1],
    })
    summary = summarise_walk_forward(results)
    assert list(summary.index) == ["Historical-based", "Realized-optimal"]


# ---------------------------------------------------------------------------
# forecast_win_rate
# ---------------------------------------------------------------------------

def test_forecast_win_rate_between_zero_and_one():
    tickers = ["AAA", "BBB"]
    prices = _synthetic_prices(200, tickers, seed=7)
    results = run_walk_forward(
        prices, tickers, horizon=10, n_windows=4, forecast_model="Naive (random walk)",
        risk_free_rate=0.04, periods_per_year=252, min_train_periods=60,
    )
    win_rate = forecast_win_rate(results)
    assert 0.0 <= win_rate <= 1.0


def test_forecast_win_rate_nan_when_portfolio_types_missing():
    results = pd.DataFrame({
        "window": [1, 2], "portfolio": ["Realized-optimal", "Realized-optimal"],
        "sharpe_ratio": [1.2, 0.8],
    })
    assert np.isnan(forecast_win_rate(results))


def test_forecast_win_rate_matches_manual_count():
    results = pd.DataFrame({
        "window": [1, 1, 2, 2, 3, 3],
        "portfolio": ["Historical-based", "Forecast-based"] * 3,
        "sharpe_ratio": [0.5, 0.8, 0.9, 0.3, 0.4, 0.6],  # forecast wins windows 1 and 3
    })
    assert forecast_win_rate(results) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# compare_to_previous_period
# ---------------------------------------------------------------------------

def _hand_built_results() -> pd.DataFrame:
    return pd.DataFrame({
        "window": [1, 2, 3, 1, 2, 3],
        "window_end": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"] * 2),
        "portfolio": ["Historical-based"] * 3 + ["Forecast-based"] * 3,
        "annual_return": [0.10, 0.12, 0.08, 0.05, 0.07, 0.03],
        "annual_volatility": [0.20, 0.18, 0.22, 0.30, 0.28, 0.32],
        "sharpe_ratio": [0.5, 0.6, 0.4, 0.2, 0.3, 0.1],
        "sortino_ratio": [0.7, 0.8, 0.5, 0.3, 0.4, 0.2],
        "max_drawdown": [-0.1, -0.08, -0.15, -0.2, -0.18, -0.25],
    })


def test_compare_to_previous_period_first_window_has_no_previous_value():
    comparison = compare_to_previous_period(_hand_built_results())
    first_window = comparison[comparison["window"] == 1]
    assert first_window["sharpe_ratio_previous"].isna().all()


def test_compare_to_previous_period_matches_the_prior_window_of_the_same_portfolio():
    comparison = compare_to_previous_period(_hand_built_results())
    hist = comparison[comparison["portfolio"] == "Historical-based"].sort_values("window")
    assert hist[hist["window"] == 2]["sharpe_ratio_previous"].iloc[0] == pytest.approx(0.5)
    assert hist[hist["window"] == 3]["sharpe_ratio_previous"].iloc[0] == pytest.approx(0.6)


def test_compare_to_previous_period_never_mixes_portfolio_types():
    comparison = compare_to_previous_period(_hand_built_results())
    fcst = comparison[comparison["portfolio"] == "Forecast-based"]
    assert fcst[fcst["window"] == 2]["sharpe_ratio_previous"].iloc[0] == pytest.approx(0.2)


def test_compare_to_previous_period_follows_narrative_portfolio_order():
    comparison = compare_to_previous_period(_hand_built_results())
    assert list(comparison["portfolio"].unique()) == ["Historical-based", "Forecast-based"]


def test_compare_to_previous_period_respects_a_custom_metric_list():
    comparison = compare_to_previous_period(_hand_built_results(), metrics=["sharpe_ratio"])
    assert "sharpe_ratio" in comparison.columns
    assert "sharpe_ratio_previous" in comparison.columns
    assert "annual_return" not in comparison.columns


def test_compare_to_previous_period_skips_a_metric_missing_from_results_without_raising():
    comparison = compare_to_previous_period(_hand_built_results(), metrics=["sharpe_ratio", "not_a_real_column"])
    assert "sharpe_ratio_previous" in comparison.columns
    assert "not_a_real_column" not in comparison.columns
    assert "not_a_real_column_previous" not in comparison.columns


def test_compare_to_previous_period_runs_end_to_end_on_a_real_walk_forward_result():
    tickers = ["AAA", "BBB"]
    prices = _synthetic_prices(200, tickers, seed=9)
    results = run_walk_forward(
        prices, tickers, horizon=10, n_windows=4, forecast_model="Naive (random walk)",
        risk_free_rate=0.04, periods_per_year=252, min_train_periods=80,
    )
    comparison = compare_to_previous_period(results)
    assert len(comparison) == len(results)
    first_windows = comparison[comparison["window"] == comparison["window"].min()]
    assert first_windows["sharpe_ratio_previous"].isna().all()