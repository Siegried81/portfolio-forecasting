"""
Unit tests for src/optimization.py.

Deliberately NOT testing the happy path only (that's what app.py's manual runs
already cover). These target the fragile branches called out in the code review:
- resolve_weight_bounds: long-only vs short-selling toggle
- optimize_max_sharpe: the negative-bounds workaround (_max_sharpe_via_utility_scan),
  the infeasible-cap auto-relax (cap < 1/N), and the degenerate all-mu-below-rf
  fallback to min-volatility (PyPortfolioOpt raises a plain ValueError here, not
  its own OptimizationError — see optimize_max_sharpe's docstring)
- efficient_frontier_points: returns a usable, non-empty frontier
"""
import numpy as np
import pandas as pd
import pytest

from src.optimization import (
    efficient_frontier_points,
    optimize_max_sharpe,
    portfolio_performance,
    resolve_weight_bounds,
)


def _synthetic_mu_cov(n_assets: int, seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    """Small, hand-controllable mu/cov — not real market data, just enough
    structure (positive variances, plausible correlations) for the optimizer
    to have a real problem to solve."""
    rng = np.random.default_rng(seed)
    tickers = [f"A{i}" for i in range(n_assets)]
    mu = pd.Series(rng.uniform(0.03, 0.15, n_assets), index=tickers)
    # Build a valid (positive semi-definite) covariance via a random loading matrix
    factor = rng.normal(0, 0.15, (n_assets, n_assets))
    cov = pd.DataFrame(factor @ factor.T / n_assets, index=tickers, columns=tickers)
    return mu, cov


# ---------------------------------------------------------------------------
# resolve_weight_bounds
# ---------------------------------------------------------------------------

def test_resolve_weight_bounds_long_only():
    assert resolve_weight_bounds(0.35, allow_short_selling=False) == (0.0, 0.35)


def test_resolve_weight_bounds_short_selling_is_symmetric():
    assert resolve_weight_bounds(0.35, allow_short_selling=True) == (-0.35, 0.35)


# ---------------------------------------------------------------------------
# optimize_max_sharpe — negative bounds (short-selling workaround)
# ---------------------------------------------------------------------------

def test_optimize_max_sharpe_short_selling_does_not_crash_and_sums_to_one():
    mu, cov = _synthetic_mu_cov(5, seed=1)
    weights = optimize_max_sharpe(mu, cov, risk_free_rate=0.04, weight_bounds=(-0.35, 0.35))
    assert weights.sum() == pytest.approx(1.0, abs=1e-3)
    # symmetric bounds respected on both sides
    assert weights.min() >= -0.35 - 1e-6
    assert weights.max() <= 0.35 + 1e-6


def test_optimize_max_sharpe_short_selling_can_go_negative():
    # With 5 assets, a 35% cap gives enough long exposure elsewhere to fund a
    # short (see optimize_max_sharpe's own docstring on feasibility) — a
    # deliberately unfavourable asset (very low mu, high covariance with the
    # rest) should end up negative rather than clipped at 0.
    mu, cov = _synthetic_mu_cov(5, seed=1)
    mu.iloc[0] = -0.20  # make the first asset clearly unattractive
    weights = optimize_max_sharpe(mu, cov, risk_free_rate=0.04, weight_bounds=(-0.35, 0.35))
    assert weights.iloc[0] < 0


# ---------------------------------------------------------------------------
# optimize_max_sharpe — infeasible cap auto-relax (cap < 1/N)
# ---------------------------------------------------------------------------

def test_optimize_max_sharpe_relaxes_infeasible_cap():
    # 4 assets, 10% cap -> 4 * 0.10 = 0.40 < 1.0, mathematically impossible to
    # sum to 100%. Should auto-relax to 1/N = 0.25 rather than raising.
    mu, cov = _synthetic_mu_cov(4, seed=2)
    weights = optimize_max_sharpe(mu, cov, risk_free_rate=0.04, weight_bounds=(0.0, 0.10))
    assert weights.sum() == pytest.approx(1.0, abs=1e-3)
    assert weights.max() <= 0.25 + 1e-6


# ---------------------------------------------------------------------------
# optimize_max_sharpe — degenerate case: every mu below the risk-free rate
# ---------------------------------------------------------------------------

def test_optimize_max_sharpe_falls_back_to_min_vol_when_all_returns_below_rf():
    mu, cov = _synthetic_mu_cov(4, seed=3)
    mu[:] = -0.05  # every asset expected to lose money — bear-market scenario
    # Must not raise (this is exactly the ValueError PyPortfolioOpt raises here,
    # per optimize_max_sharpe's docstring) and must still return valid weights.
    weights = optimize_max_sharpe(mu, cov, risk_free_rate=0.04, weight_bounds=(0.0, 1.0))
    assert weights.sum() == pytest.approx(1.0, abs=1e-3)
    assert (weights >= -1e-9).all()


# ---------------------------------------------------------------------------
# efficient_frontier_points
# ---------------------------------------------------------------------------

def test_efficient_frontier_points_returns_usable_frontier():
    mu, cov = _synthetic_mu_cov(5, seed=4)
    frontier = efficient_frontier_points(mu, cov, n_points=10, weight_bounds=(0.0, 1.0))
    assert not frontier.empty
    assert {"return", "volatility"}.issubset(frontier.columns)
    assert (frontier["volatility"] >= 0).all()


def test_efficient_frontier_points_single_asset_returns_one_point_without_crashing():
    # REGRESSION coverage: a single selected ticker forces weight=1 on that
    # one asset — there is no return/risk trade-off to sweep, so the target-
    # return sweep used for n>=2 assets is infeasible by construction and
    # must not reach PyPortfolioOpt at all for this case.
    mu = pd.Series({"AAPL": 0.12})
    cov = pd.DataFrame({"AAPL": [0.04]}, index=["AAPL"])
    frontier = efficient_frontier_points(mu, cov, n_points=25, weight_bounds=(0.0, 1.0))
    assert len(frontier) == 1
    assert frontier.iloc[0]["return"] == pytest.approx(0.12, abs=1e-6)
    assert frontier.iloc[0]["volatility"] == pytest.approx(0.2, abs=1e-6)  # sqrt(0.04)


def test_efficient_frontier_points_single_asset_with_default_cap_does_not_crash():
    # REGRESSION coverage for the live-reported crash: a single asset with
    # the app's own DEFAULT 35% per-asset cap makes even min_volatility()
    # infeasible on its own (100% in one asset > 35% cap) -- min_volatility
    # itself must never reach the solver with an infeasible upper bound.
    mu = pd.Series({"AAPL": 0.12})
    cov = pd.DataFrame({"AAPL": [0.04]}, index=["AAPL"])
    frontier = efficient_frontier_points(mu, cov, n_points=25, weight_bounds=(0.0, 0.35))
    assert len(frontier) == 1
    assert frontier.iloc[0]["return"] == pytest.approx(0.12, abs=1e-6)


def test_efficient_frontier_points_identical_expected_returns_returns_one_point():
    # A less obvious version of the same degeneracy: more than one asset, but
    # every one shares the SAME expected return (e.g. after some upstream
    # filtering) — still no return/risk trade-off to sweep, even though
    # len(mu) >= 2.
    mu = pd.Series({"AAA": 0.10, "BBB": 0.10, "CCC": 0.10})
    factor = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.25], [0.2, 0.25, 1.0]]) * 0.04
    cov = pd.DataFrame(factor, index=mu.index, columns=mu.index)
    frontier = efficient_frontier_points(mu, cov, n_points=25, weight_bounds=(0.0, 1.0))
    assert len(frontier) == 1


# ---------------------------------------------------------------------------
# portfolio_performance — sanity check against a hand-computable case
# ---------------------------------------------------------------------------

def test_portfolio_performance_matches_manual_calculation_for_equal_weights():
    mu = pd.Series({"A": 0.10, "B": 0.20})
    cov = pd.DataFrame({"A": [0.04, 0.0], "B": [0.0, 0.09]}, index=["A", "B"])  # no correlation
    weights = pd.Series({"A": 0.5, "B": 0.5})
    perf = portfolio_performance(mu, cov, weights, risk_free_rate=0.04)
    assert perf["expected_return"] == pytest.approx(0.15, abs=1e-9)
    # var = 0.25*0.04 + 0.25*0.09 = 0.0325 -> vol = sqrt(0.0325)
    assert perf["expected_volatility"] == pytest.approx(0.0325 ** 0.5, abs=1e-9)