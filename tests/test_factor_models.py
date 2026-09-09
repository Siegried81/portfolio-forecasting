"""
Unit tests for src/factor_models.py.

Priority: does the PCA estimate genuinely recover known factor structure on
synthetic data (not just "does it run"), and do the guarantees the docstring
promises actually hold — symmetric, positive-semi-definite, correct
clamping, informative failure on too-little data.
"""
import numpy as np
import pandas as pd
import pytest

from src.factor_models import pca_factor_cov


def _synthetic_factor_returns(n_obs: int, n_assets: int, n_true_factors: int, seed: int = 0) -> pd.DataFrame:
    """
    Build returns with a KNOWN factor structure: a handful of common factors
    driving co-movement, plus asset-specific idiosyncratic noise — so a
    correct PCA estimate should recover something close to the true
    covariance, not just produce a plausible-looking matrix.
    """
    rng = np.random.default_rng(seed)
    factor_returns = rng.normal(0, 0.01, (n_obs, n_true_factors))
    loadings = rng.uniform(0.3, 1.2, (n_assets, n_true_factors))
    idiosyncratic = rng.normal(0, 0.003, (n_obs, n_assets))
    returns = factor_returns @ loadings.T + idiosyncratic
    columns = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(returns, columns=columns)


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------

def test_pca_factor_cov_returns_symmetric_psd_matrix():
    returns = _synthetic_factor_returns(n_obs=300, n_assets=20, n_true_factors=3, seed=1)
    cov, _ = pca_factor_cov(returns, n_factors=5, periods_per_year=252)
    values = cov.values
    assert np.allclose(values, values.T, atol=1e-8)  # symmetric
    eigenvalues = np.linalg.eigvalsh(values)
    assert (eigenvalues >= -1e-6).all()  # positive semi-definite (small numerical slack)


def test_pca_factor_cov_diagnostics_report_actual_factor_count():
    returns = _synthetic_factor_returns(n_obs=300, n_assets=20, n_true_factors=3, seed=2)
    _, diagnostics = pca_factor_cov(returns, n_factors=5, periods_per_year=252)
    assert diagnostics["n_factors_used"] == 5
    assert len(diagnostics["explained_variance_ratio"]) == 5
    assert diagnostics["cumulative_explained"] == pytest.approx(
        sum(diagnostics["explained_variance_ratio"]), abs=1e-9
    )


def test_pca_factor_cov_explains_most_variance_when_true_structure_matches_n_factors():
    # 3 true underlying factors, ask for exactly 3 -> should explain the
    # large majority of variance (idiosyncratic noise is deliberately small
    # relative to the factor-driven co-movement in the synthetic data).
    returns = _synthetic_factor_returns(n_obs=400, n_assets=25, n_true_factors=3, seed=3)
    _, diagnostics = pca_factor_cov(returns, n_factors=3, periods_per_year=252)
    assert diagnostics["cumulative_explained"] > 0.8


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------

def test_pca_factor_cov_clamps_n_factors_to_available_assets():
    # 5 assets -> at most 4 factors are mathematically possible, regardless
    # of how many were requested.
    returns = _synthetic_factor_returns(n_obs=200, n_assets=5, n_true_factors=2, seed=4)
    _, diagnostics = pca_factor_cov(returns, n_factors=20, periods_per_year=252)
    assert diagnostics["n_factors_used"] <= 4


def test_pca_factor_cov_respects_min_factors_floor():
    returns = _synthetic_factor_returns(n_obs=200, n_assets=20, n_true_factors=3, seed=5)
    _, diagnostics = pca_factor_cov(returns, n_factors=0, periods_per_year=252)
    assert diagnostics["n_factors_used"] >= 2  # MIN_PCA_FACTORS


# ---------------------------------------------------------------------------
# Failure case
# ---------------------------------------------------------------------------

def test_pca_factor_cov_raises_with_too_little_data():
    tiny = pd.DataFrame({"A": [0.01]})  # 1 asset, 1 observation
    with pytest.raises(ValueError):
        pca_factor_cov(tiny, n_factors=5)


# ---------------------------------------------------------------------------
# Annualisation consistency
# ---------------------------------------------------------------------------

def test_pca_factor_cov_scales_with_periods_per_year():
    returns = _synthetic_factor_returns(n_obs=300, n_assets=15, n_true_factors=3, seed=6)
    cov_daily, _ = pca_factor_cov(returns, n_factors=3, periods_per_year=252)
    cov_weekly, _ = pca_factor_cov(returns, n_factors=3, periods_per_year=52)
    # Same underlying data, different annualisation factor -> daily-scaled
    # covariance should be exactly 252/52 times the weekly-scaled one.
    ratio = cov_daily.values / cov_weekly.values
    assert np.allclose(ratio, 252 / 52, atol=1e-6)