"""
Mean-variance portfolio optimization via PyPortfolioOpt (chosen over Riskfolio-Lib:
lighter footprint, actively maintained, and covers exactly what the brief needs —
max-Sharpe / min-vol optimal weights + an efficient frontier — without the extra
install weight of Riskfolio-Lib's convex-optimization stack).

This module is deliberately agnostic to WHERE mu (expected returns) and the
covariance matrix come from — the same `optimize_max_sharpe` / `efficient_frontier_points`
functions are used for all three portfolios the brief asks for:
  1. Historical-based  : mu/cov estimated from realised historical returns
  2. Forecast-based    : mu from the forecasting module, cov still historical
                          (covariance forecasting is a research problem in itself;
                          using the historical covariance is standard practice
                          even in forecast-driven allocation)
  3. Realized-optimal  : mu/cov estimated from the ACTUAL future returns (hindsight
                          optimum) — the benchmark the other two are judged against
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.exceptions import OptimizationError

from src.config import (
    COV_METHOD_PCA,
    DEFAULT_COV_METHOD,
    DEFAULT_PCA_FACTORS,
    DEFAULT_RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
)


def historical_mu_cov(
    prices: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    cov_method: str = DEFAULT_COV_METHOD,
    n_factors: int = DEFAULT_PCA_FACTORS,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Annualised mean historical return (mu) and covariance from a price
    history. `cov_method` selects between two genuinely different covariance
    ESTIMATORS — mu is always the same historical mean either way:

    - `"ledoit_wolf"` (default): shrinkage covariance, the long-standing
      approach here. With few assets and a short history the raw sample
      covariance is noisy and can produce unstable, extreme optimal weights
      — shrinkage is the standard practitioner fix (Ledoit & Wolf, 2004).
    - `"pca"`: a statistical factor-model covariance (see
      `factor_models.pca_factor_cov`) — the tractable alternative once the
      universe gets wide relative to the history available (see that
      module's docstring and README's "Why not all 500 S&P constituents").

    Single dispatch point, kept here rather than scattered across call
    sites, so every caller (frontier, single-window comparison, walk-forward)
    picks up the same estimator from one config choice — this module's own
    stated design goal of staying agnostic to WHERE mu/cov come from.

    `periods_per_year` MUST match the actual periodicity of `prices` (252 for
    daily, 52 for weekly, 12 for monthly — see config.FREQUENCY_TO_PERIODS_PER_YEAR).
    Passing the wrong value here doesn't error — it just silently annualises
    as if every row were one trading day, which massively overstates
    "expected return" on weekly/monthly data (a ~5-year weekly series has ~260
    rows; annualising with frequency=252 treats that as barely one year of daily
    data, compounding the horizon by roughly 5x). Always pass the caller's actual
    `periods_per_year`, never rely on the default.
    """
    mu = expected_returns.mean_historical_return(prices, frequency=periods_per_year)
    if cov_method == COV_METHOD_PCA:
        from src.factor_models import pca_factor_cov
        returns = prices.pct_change().dropna(how="all")
        cov, _diagnostics = pca_factor_cov(returns, n_factors=n_factors, periods_per_year=periods_per_year)
        # Column/row order must match mu's index (PyPortfolioOpt assumes
        # alignment by position, not by label, in several internal steps).
        cov = cov.reindex(index=mu.index, columns=mu.index)
    else:
        cov = risk_models.CovarianceShrinkage(prices, frequency=periods_per_year).ledoit_wolf()
    return mu, cov


def forecast_mu(
    current_prices: pd.Series,
    forecasted_prices: pd.DataFrame,
    horizon_periods: int,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Convert forecasted end-of-horizon prices into an ANNUALISED expected return,
    so it's on the same scale as `historical_mu_cov`'s mu and can be swapped into
    the same optimizer unchanged.

    `horizon_periods` is a count of periods AT THE SELECTED FREQUENCY (e.g. 30
    weekly steps if the sidebar frequency is "weekly", not 30 calendar days) —
    same `periods_per_year` mismatch risk as `historical_mu_cov` above applies here.
    """
    last_forecast = forecasted_prices.iloc[-1]
    horizon_return = last_forecast / current_prices[forecasted_prices.columns] - 1.0
    years = horizon_periods / periods_per_year
    # Compound the horizon return out to an annual rate; guard against negative
    # base (a forecast crashing below zero) which would make the exponent undefined.
    annualised = (1.0 + horizon_return).clip(lower=1e-6) ** (1.0 / years) - 1.0
    return annualised


def resolve_weight_bounds(max_weight_per_asset: float, allow_short_selling: bool) -> tuple[float, float]:
    """
    Turn the two sidebar controls (max weight, short-selling toggle) into the
    `weight_bounds` tuple every optimizer call needs. Long-only by default
    (0, cap); enabling short-selling makes it symmetric (-cap, cap) — the
    portfolio still stays fully invested (weights sum to 1), this only lets
    individual positions go negative, it does not add gross leverage.
    """
    lower = -max_weight_per_asset if allow_short_selling else 0.0
    return (lower, max_weight_per_asset)


def optimize_max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
) -> pd.Series:
    """Weights maximising the Sharpe ratio (long-only by default — weight_bounds
    can be opened to allow shorting if the UI ever exposes that).

    Guards against an infeasible upper bound: with N assets, weights summing to
    1.0 is impossible if the cap is below 1/N (e.g. a 35% max-weight constraint
    with only 2 assets selected — no valid allocation can sum to 100%). Relaxes
    the cap to 1/N in that case rather than letting PyPortfolioOpt raise.

    Note this doesn't fully solve short-selling's own feasibility limits: with a
    small universe and a tight per-asset cap, there may not be enough long
    exposure available elsewhere to "fund" a meaningful short (e.g. 3 assets
    capped at 35% each can push a losing asset down to at most +30%, never
    negative — two other assets maxed at 35% only sum to 70%, forcing the third
    to make up the remaining 30%). That outcome is mathematically correct given
    the constraints, not a bug — more assets or a higher cap creates the room a
    short position needs.
    """
    lower, upper = weight_bounds
    n_assets = len(mu)
    if n_assets > 0 and upper < 1.0 / n_assets:
        upper = 1.0 / n_assets
        weight_bounds = (lower, upper)

    if lower < 0:
        # max_sharpe()'s internal convex reformulation (the Cornuejols-Tütüncü
        # auxiliary-variable trick) is documented to raise a spurious "infeasible"
        # OptimizationError once the lower bound goes negative — confirmed
        # directly against this exact mu/cov/bounds combination, and matches a
        # long-standing open PyPortfolioOpt issue (github.com/robertmartin8/
        # PyPortfolioOpt/issues/436). max_quadratic_utility() uses a different,
        # unaffected formulation, so a small scan over risk_aversion values finds
        # the best-Sharpe point on the frontier instead.
        return _max_sharpe_via_utility_scan(mu, cov, risk_free_rate, weight_bounds)

    ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
    try:
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    except (OptimizationError, ValueError):
        # Degenerate case — e.g. every asset's expected return is below the
        # risk-free rate (a bear-market date range, or a bad forecast going
        # negative). PyPortfolioOpt raises a plain ValueError here, not its own
        # OptimizationError, so both must be caught. Fall back to min-volatility,
        # which is always solvable, rather than crashing the whole app.
        ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
        ef.min_volatility()
    weights = ef.clean_weights()
    return pd.Series(weights)


_UTILITY_SCAN_RISK_AVERSIONS = [0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 5, 10, 20, 50]


def _max_sharpe_via_utility_scan(
    mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float, weight_bounds: tuple[float, float],
) -> pd.Series:
    """Approximate max-Sharpe by scanning `max_quadratic_utility`'s risk_aversion
    parameter and keeping whichever point achieves the best REALIZED Sharpe —
    a standard, documented workaround for cases where `max_sharpe()` itself is
    unreliable (see the caller's docstring). Falls back to min-volatility (always
    solvable, including with negative bounds) if every scan point fails."""
    best_sharpe, best_weights = -np.inf, None
    for risk_aversion in _UTILITY_SCAN_RISK_AVERSIONS:
        try:
            ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
            ef.max_quadratic_utility(risk_aversion=risk_aversion)
            _, _, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        except (OptimizationError, ValueError):
            continue
        if sharpe > best_sharpe:
            best_sharpe, best_weights = sharpe, ef.clean_weights()

    if best_weights is None:
        ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
        ef.min_volatility()
        best_weights = ef.clean_weights()
    return pd.Series(best_weights)


def efficient_frontier_points(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 25,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    """
    Sample the efficient frontier by sweeping target returns between the
    min-volatility portfolio's return and the max achievable return, solving a
    min-volatility QP at each target. Returns a tidy DataFrame ready for plotting:
    columns = ["volatility", "return"].

    Degenerate case, handled explicitly rather than left to crash: with a
    single asset (or, more generally, whenever every feasible portfolio has
    the SAME expected return — e.g. every remaining asset carries identical
    mu after upstream filtering), there is mathematically only one point on
    the "frontier": weights are forced to put 100% in the only asset
    available, so return and volatility can't be traded off against each
    other at all. Sweeping target returns in that case asks the solver for
    expected returns the single feasible portfolio cannot produce, which
    PyPortfolioOpt correctly reports as `OptimizationError: Solver status:
    infeasible` — not a bug in the solver, but a sweep that was never
    meaningful to run here. Returning the single min-volatility point
    directly sidesteps the sweep entirely for this case.

    A SECOND, separate infeasibility guard: an upper weight bound below
    1/N (e.g. a 35% per-asset cap with only 1-2 assets selected) makes even
    the very first `min_volatility()` call below infeasible, since no
    combination of weights capped at `upper` can sum to 100% — the same
    "cap < 1/N" case `optimize_max_sharpe` already relaxes automatically.
    This function relaxes the SAME way, before either optimizer call below,
    so a tight cap can never reach the solver at all as an infeasible bound.
    """
    lower, upper = weight_bounds
    n_assets = len(mu)
    if n_assets > 0 and upper < 1.0 / n_assets:
        weight_bounds = (lower, 1.0 / n_assets)

    ef_minvol = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
    ef_minvol.min_volatility()
    min_vol_return, min_vol_vol, _ = ef_minvol.portfolio_performance()

    max_return = mu.max()
    if len(mu) < 2 or max_return <= min_vol_return:
        return pd.DataFrame([{"return": min_vol_return, "volatility": min_vol_vol}])

    target_returns = np.linspace(min_vol_return, max_return * 0.999, n_points)

    points = []
    for target in target_returns:
        try:
            ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
            ef.efficient_return(target_return=target)
            ret, vol, _ = ef.portfolio_performance()
            points.append({"return": ret, "volatility": vol})
        except (OptimizationError, ValueError):
            continue  # infeasible target on this grid point — skip, keep the rest

    return pd.DataFrame(points)


def portfolio_performance(
    mu: pd.Series,
    cov: pd.DataFrame,
    weights: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> dict[str, float]:
    """Expected annual return / volatility / Sharpe for a given weight vector,
    using the SAME mu/cov the weights were optimised on (so this reports the
    optimizer's own expectation, distinct from realised performance metrics.py computes)."""
    w = weights.reindex(mu.index).fillna(0.0).values
    ret = float(np.dot(w, mu))
    vol = float(np.sqrt(np.dot(w, np.dot(cov, w))))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else float("nan")
    return {"expected_return": ret, "expected_volatility": vol, "expected_sharpe": sharpe}


def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    """
    The correlation matrix (Overview tab) is visual, not a
    number — this is the number. Choueifaty & Coignard's Diversification Ratio
    (2008): the weighted average of each asset's OWN volatility, divided by
    the ACTUAL portfolio volatility. DR > 1 whenever correlations are below 1
    everywhere (the normal case) — the gap between the two is precisely the
    risk reduction diversification bought. DR = 1 only in the degenerate case
    of a single-asset portfolio or perfectly correlated assets (correlation
    matrix all 1.0), where diversification buys nothing. Higher is better;
    there's no universal "good" threshold — compare it across the three
    portfolios (historical/forecast/realized) the way Calmar is compared, not
    against a memorised number.
    """
    w = weights.reindex(cov.index).fillna(0.0).values
    individual_vols = np.sqrt(np.diag(cov.values))
    weighted_avg_vol = float(np.dot(np.abs(w), individual_vols))
    portfolio_vol = float(np.sqrt(np.dot(w, np.dot(cov.values, w))))
    if portfolio_vol == 0:
        return float("nan")
    return weighted_avg_vol / portfolio_vol


def concentration_hhi(weights: pd.Series) -> float:
    """
    Herfindahl-Hirschman Index on portfolio weights: sum(w_i^2). Ranges from
    1/N (perfectly equal-weighted across N assets) to 1.0 (fully concentrated
    in one asset). Distinct in kind from diversification_ratio above — HHI
    measures ALLOCATION concentration itself (a quick sanity check that the
    max-weight cap is actually doing its job), while the diversification
    ratio measures the RISK benefit actually realised from that allocation.
    Two portfolios can have the same HHI with very different diversification
    ratios if their assets are differently correlated — that's not a
    contradiction, it's the reason both numbers earn their place.
    """
    return float((weights ** 2).sum())