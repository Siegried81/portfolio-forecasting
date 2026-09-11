"""
Walk-forward (multi-window) backtesting: repeats the historical/forecast/
realized-optimal comparison across several expanding windows instead of
one, so the result is a distribution rather than a single lucky split.
"""
from __future__ import annotations

import pandas as pd
from typing import Any

from src.forecasting import forecast_all_assets
from src.metrics import apply_transaction_cost, compute_returns, compute_turnover, portfolio_returns, summarise_performance
from src.config import COV_METHOD_GARCH, DEFAULT_COV_METHOD, DEFAULT_PCA_FACTORS
from src.optimization import forecast_mu, historical_mu_cov, optimize_max_sharpe, resolve_weight_bounds

PORTFOLIO_TYPE_ORDER: list[str] = ["Historical-based", "Forecast-based", "Realized-optimal"]


def generate_expanding_windows(
    n_periods: int, horizon: int, min_train_periods: int, max_windows: int,
) -> list[tuple[int, int]]:
    """
    Build (train_end, test_end) index pairs for expanding-window walk-forward.
    Each window sees strictly more training data than the last. If more
    windows are possible than `max_windows`, the most recent ones are kept.
    """
    all_windows: list[tuple[int, int]] = []
    train_end = min_train_periods
    while train_end + horizon <= n_periods:
        all_windows.append((train_end, train_end + horizon))
        train_end += horizon
    if len(all_windows) > max_windows:
        return all_windows[-max_windows:]
    return all_windows


def run_walk_forward(
    prices: pd.DataFrame,
    tickers: list[str],
    horizon: int,
    n_windows: int,
    forecast_model: str,
    risk_free_rate: float,
    periods_per_year: int,
    min_train_periods: int,
    max_weight_per_asset: float = 1.0,
    allow_short_selling: bool = False,
    transaction_cost_bps: float = 0.0,
    cov_method: str = DEFAULT_COV_METHOD,
    n_factors: int = DEFAULT_PCA_FACTORS,
    forecast_cov_method: str | None = None,
    frequency: str = "daily",
) -> pd.DataFrame:
    """
    Run the historical/forecast/realized-optimal comparison across several
    expanding windows. Returns a long-format DataFrame, one row per
    (window, portfolio type). `forecast_cov_method=COV_METHOD_GARCH` makes
    the Forecast-based portfolio use garch_forecast_cov instead of the
    historical covariance.
    """
    windows = generate_expanding_windows(len(prices), horizon, min_train_periods, n_windows)
    records: list[dict[str, Any]] = []
    previous_weights: dict[str, pd.Series] = {}

    for window_idx, (train_end, test_end) in enumerate(windows, start=1):
        train_prices = prices.iloc[:train_end]
        # train_end - 1 so the test window includes the last training price:
        # pct_change() then yields exactly `horizon` returns, not horizon - 1.
        test_prices = prices.iloc[train_end - 1:test_end]
        test_returns = compute_returns(test_prices[tickers])

        weight_bounds = resolve_weight_bounds(max_weight_per_asset, allow_short_selling)

        mu_hist, cov_hist = historical_mu_cov(
            train_prices[tickers], periods_per_year, cov_method=cov_method, n_factors=n_factors,
        )
        w_hist = optimize_max_sharpe(mu_hist, cov_hist, risk_free_rate, weight_bounds)

        forecasted = forecast_all_assets(train_prices[tickers], horizon, forecast_model, frequency)
        mu_fcst = forecast_mu(train_prices.iloc[-1][tickers], forecasted, horizon, periods_per_year)
        if forecast_cov_method == COV_METHOD_GARCH:
            from src.volatility_forecasting import garch_forecast_cov
            cov_fcst, _diagnostics = garch_forecast_cov(train_prices[tickers], horizon, periods_per_year)
        else:
            cov_fcst = cov_hist
        w_fcst = optimize_max_sharpe(mu_fcst, cov_fcst, risk_free_rate, weight_bounds)

        mu_real, cov_real = historical_mu_cov(
            test_prices[tickers], periods_per_year, cov_method=cov_method, n_factors=n_factors,
        )
        w_real = optimize_max_sharpe(mu_real, cov_real, risk_free_rate, weight_bounds)

        for label, weights in zip(PORTFOLIO_TYPE_ORDER, [w_hist, w_fcst, w_real]):
            port_returns = portfolio_returns(test_returns, weights)
            if transaction_cost_bps > 0:
                turnover = compute_turnover(weights, previous_weights.get(label))
                port_returns = apply_transaction_cost(port_returns, turnover, transaction_cost_bps)
            previous_weights[label] = weights
            perf = summarise_performance(port_returns, risk_free_rate, periods_per_year)
            records.append({"window": window_idx, "window_end": test_prices.index[-1], "portfolio": label, **perf})

    return pd.DataFrame(records)


def summarise_walk_forward(results: pd.DataFrame) -> pd.DataFrame:
    """Mean/median/std of each metric per portfolio type across all windows."""
    summary = results.groupby("portfolio")[
        ["annual_return", "annual_volatility", "sharpe_ratio", "sortino_ratio", "max_drawdown"]
    ].agg(["mean", "std"])
    order = [p for p in PORTFOLIO_TYPE_ORDER if p in summary.index]
    return summary.loc[order]


def forecast_win_rate(results: pd.DataFrame) -> float:
    """Fraction of windows where Forecast-based Sharpe beat Historical-based Sharpe."""
    pivot = results.pivot(index="window", columns="portfolio", values="sharpe_ratio")
    if "Forecast-based" not in pivot.columns or "Historical-based" not in pivot.columns:
        return float("nan")
    return float((pivot["Forecast-based"] > pivot["Historical-based"]).mean())


DEFAULT_PERIOD_COMPARISON_METRICS: list[str] = [
    "annual_return", "annual_volatility", "sharpe_ratio", "sortino_ratio", "max_drawdown",
]


def compare_to_previous_period(
    results: pd.DataFrame, metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Reshape the long-format output into a "this period vs. the previous
    one" table, per portfolio type independently. Window 1 has NaN
    "_previous" columns.
    """
    if metrics is None:
        metrics = DEFAULT_PERIOD_COMPARISON_METRICS

    sorted_results = results.sort_values(["portfolio", "window"]).reset_index(drop=True)
    columns = ["window", "window_end", "portfolio"] + [m for m in metrics if m in sorted_results.columns]
    out = sorted_results[columns].copy()
    for metric in metrics:
        if metric not in sorted_results.columns:
            continue
        out[f"{metric}_previous"] = sorted_results.groupby("portfolio")[metric].shift(1)

    order = [p for p in PORTFOLIO_TYPE_ORDER if p in out["portfolio"].unique()]
    out["portfolio"] = pd.Categorical(out["portfolio"], categories=order, ordered=True)
    return out.sort_values(["portfolio", "window"]).reset_index(drop=True)