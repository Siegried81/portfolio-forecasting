"""
Walk-forward (multi-window) backtesting.

Rationale (why this exists on top of the single train/test split in app.py's
"Forecast & Compare" tab): one held-out window tells you whether forecasting
helped THAT ONE TIME — which can easily be luck. Walk-forward repeats the exact
same historical/forecast/realized comparison across several EXPANDING windows
(window k's training data = everything up to that point; test data = the next
`horizon` periods), producing a distribution of outcomes instead of one number.
This is standard practice in any credible backtest and is what separates
"I built an optimizer" from "I validated that it actually adds value" — the
second is what a quant/portfolio-management interviewer is really probing for.

Kept in its own module (not bolted onto app.py) because it's pure orchestration
over the SAME pure functions used by the single-window comparison — metrics.py,
optimization.py, forecasting.py — so the two comparisons can never silently
compute things differently.
"""
from __future__ import annotations

import pandas as pd
from typing import Any

from src.forecasting import forecast_all_assets
from src.metrics import apply_transaction_cost, compute_returns, compute_turnover, portfolio_returns, summarise_performance
from src.config import COV_METHOD_GARCH, DEFAULT_COV_METHOD, DEFAULT_PCA_FACTORS
from src.optimization import forecast_mu, historical_mu_cov, optimize_max_sharpe, resolve_weight_bounds

# Canonical narrative order for the three portfolio types, used everywhere
# this app displays them side by side (this module's own records, the
# walk-forward summary table, and app.py's box plot) — NOT alphabetical
# order. 
PORTFOLIO_TYPE_ORDER: list[str] = ["Historical-based", "Forecast-based", "Realized-optimal"]


def generate_expanding_windows(
    n_periods: int, horizon: int, min_train_periods: int, max_windows: int,
) -> list[tuple[int, int]]:
    """
    Build (train_end, test_end) index pairs for expanding-window walk-forward.
    Window 1 trains on [0, min_train_periods), tests on the next `horizon`
    periods; window 2 trains on [0, min_train_periods + horizon), and so on —
    each window sees strictly more training data than the last, which mirrors
    how a real strategy would actually be refit over time (using everything
    known so far), rather than an artificially fixed-size rolling window.

    When MORE windows are mathematically possible than `max_windows` allows
    (a long price history combined with a small window count), the MOST
    RECENT windows are kept, not the earliest ones. Anchoring from the start
    of history instead would silently test only the OLDEST slice of a long
    date range and never reach anywhere near the present — a 5-year price
    history with 6 requested windows would validate performance from year
    one only, never showing how the strategy did recently, which is the
    question walk-forward validation exists to answer.
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
) -> pd.DataFrame:
    """
    Run the historical / forecast / realized-optimal comparison across several
    expanding windows. Returns a tidy long-format DataFrame — one row per
    (window, portfolio type) — ready for both a summary table and a box plot.

    Columns: window, window_end (date), portfolio, annual_return,
    annual_volatility, sharpe_ratio, sortino_ratio, max_drawdown, var_95, cvar_95.

    Transaction costs: `transaction_cost_bps` charges turnover × rate at EVERY
    window boundary for every portfolio type, tracked independently (each
    portfolio type rebalances against its OWN previous window's weights, not a
    shared reference) — this is also where "rebalancing frequency" shows up in
    practice: a shorter `horizon` means more windows over the same history, i.e.
    more frequent rebalancing, i.e. more cumulative cost drag realised here.

    `cov_method`/`n_factors` are passed straight through to
    every `historical_mu_cov` call below (both the historical-training-window
    fit and the realized/hindsight fit) — so a walk-forward run over a wide
    universe uses the SAME covariance estimator throughout, not Ledoit-Wolf in
    one place and PCA in another. Defaults preserve the original
    Ledoit-Wolf-everywhere behaviour for any existing caller that doesn't pass
    these.

    `forecast_cov_method` is a SEPARATE switch, deliberately: it only affects
    the Forecast-based portfolio's covariance, not Historical/Realized-optimal
    (those keep using `cov_method` as always). Pass `COV_METHOD_GARCH` to make
    the Forecast-based portfolio use `volatility_forecasting.garch_forecast_cov`
    instead of the historical covariance — genuinely forecasting BOTH halves
    (mu and Sigma) of that portfolio's inputs, closing the gap this module's
    docstring (and README's "Key design decisions") used to flag plainly:
    "only mu ever came from the forecast, Sigma stayed historical." Defaults to
    `None`, which preserves the original behaviour (cov_hist reused as-is) for
    every existing caller that doesn't pass this.
    """
    windows = generate_expanding_windows(len(prices), horizon, min_train_periods, n_windows)
    records: list[dict[str, Any]] = []
    previous_weights: dict[str, pd.Series] = {}

    for window_idx, (train_end, test_end) in enumerate(windows, start=1):
        train_prices = prices.iloc[:train_end]
        test_prices = prices.iloc[train_end:test_end]
        test_returns = compute_returns(test_prices[tickers])

        weight_bounds = resolve_weight_bounds(max_weight_per_asset, allow_short_selling)

        # 1. Historical-based — fit on this window's training slice only
        mu_hist, cov_hist = historical_mu_cov(
            train_prices[tickers], periods_per_year, cov_method=cov_method, n_factors=n_factors,
        )
        w_hist = optimize_max_sharpe(mu_hist, cov_hist, risk_free_rate, weight_bounds)

        # 2. Forecast-based — forecast over this window's test horizon.
        # Covariance: historical by default (cov_hist, same as Historical-based
        # above) unless forecast_cov_method requests a genuine forecast instead.
        forecasted = forecast_all_assets(train_prices[tickers], horizon, forecast_model)
        mu_fcst = forecast_mu(train_prices.iloc[-1][tickers], forecasted, horizon, periods_per_year)
        if forecast_cov_method == COV_METHOD_GARCH:
            from src.volatility_forecasting import garch_forecast_cov
            cov_fcst, _diagnostics = garch_forecast_cov(train_prices[tickers], horizon, periods_per_year)
        else:
            cov_fcst = cov_hist
        w_fcst = optimize_max_sharpe(mu_fcst, cov_fcst, risk_free_rate, weight_bounds)

        # 3. Realized-optimal — hindsight fit on this window's actual test-period returns
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
    """Mean/median/std of each metric per portfolio type across all windows —
    the distribution summary that answers "does forecasting help ON AVERAGE,
    and how consistently?", not just "did it help once".
    """
    summary = results.groupby("portfolio")[
        ["annual_return", "annual_volatility", "sharpe_ratio", "sortino_ratio", "max_drawdown"]
    ].agg(["mean", "std"])
    order = [p for p in PORTFOLIO_TYPE_ORDER if p in summary.index]
    return summary.loc[order]


def forecast_win_rate(results: pd.DataFrame) -> float:
    """Fraction of windows where Forecast-based Sharpe beat Historical-based Sharpe —
    a simple, interview-friendly headline number to pair with the full distribution."""
    pivot = results.pivot(index="window", columns="portfolio", values="sharpe_ratio")
    if "Forecast-based" not in pivot.columns or "Historical-based" not in pivot.columns:
        return float("nan")
    return float((pivot["Forecast-based"] > pivot["Historical-based"]).mean())


# Metrics compared period-over-period by default in `compare_to_previous_period` —
# the headline performance/risk numbers a reader scans window to window, not
# every column `run_walk_forward` produces (var_95/cvar_95 are available in
# `results` for anyone who wants them, just not duplicated into this view by
# default).
DEFAULT_PERIOD_COMPARISON_METRICS: list[str] = [
    "annual_return", "annual_volatility", "sharpe_ratio", "sortino_ratio", "max_drawdown",
]


def compare_to_previous_period(
    results: pd.DataFrame, metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Reshape `run_walk_forward`'s long-format output (one row per window,
    per portfolio type) into a "this period vs. the previous one" table:
    for each metric, one column holding the CURRENT window's value and one
    holding the SAME portfolio type's value from the immediately PRECEDING
    window — side by side in the same row, so a reader can see whether a
    number improved or worsened without cross-referencing two separate rows
    themselves.

    Comparison is done PER PORTFOLIO TYPE independently (Historical-based's
    window 3 is compared to Historical-based's window 2, never to another
    portfolio type's window 2) — the three portfolio types are only ever
    meaningfully compared against their OWN trajectory over time, the same
    way the walk-forward validation itself treats them as three independent
    series across windows.

    Window 1 has nothing before it: its "_previous" columns are NaN, not a
    fabricated zero or a wrap-around to the last window — there genuinely is
    no prior period for the first window to be compared against.

    Returns a tidy DataFrame with columns: window, window_end, portfolio,
    then `{metric}` / `{metric}_previous` pairs for each metric in `metrics`
    (defaults to `DEFAULT_PERIOD_COMPARISON_METRICS`), sorted by portfolio
    (in `PORTFOLIO_TYPE_ORDER`) then window — ready to hand straight to
    `st.dataframe` for the UI's period-over-period tab.
    """
    if metrics is None:
        metrics = DEFAULT_PERIOD_COMPARISON_METRICS

    sorted_results = results.sort_values(["portfolio", "window"]).reset_index(drop=True)
    columns = ["window", "window_end", "portfolio"] + [m for m in metrics if m in sorted_results.columns]
    out = sorted_results[columns].copy()
    for metric in metrics:
        if metric not in sorted_results.columns:
            continue  # a caller-requested metric this results DataFrame doesn't have — skip, don't KeyError
        # shift(1) WITHIN each portfolio group: window N's "_previous" is
        # window N-1's value for that SAME portfolio, never another
        # portfolio's row or a value from a non-adjacent window.
        out[f"{metric}_previous"] = sorted_results.groupby("portfolio")[metric].shift(1)

    order = [p for p in PORTFOLIO_TYPE_ORDER if p in out["portfolio"].unique()]
    out["portfolio"] = pd.Categorical(out["portfolio"], categories=order, ordered=True)
    return out.sort_values(["portfolio", "window"]).reset_index(drop=True)