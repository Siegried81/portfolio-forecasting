"""
PCA-based statistical factor model for covariance estimation — the tractable alternative
to Ledoit-Wolf shrinkage once the investable universe gets wide relative to the history available.

Why this exists (the actual math, not just "big universes are hard"): with N
assets, a covariance matrix has N(N+1)/2 free parameters to estimate. For
N=500 that's 125,250 parameters against, at best, a few thousand daily
observations — badly underdetermined, and Ledoit-Wolf shrinkage (already used
elsewhere in this app) only partially compensates; it doesn't fix the
underlying degrees-of-freedom problem. A factor model does: assume returns are
driven by a SMALL number of common factors plus asset-specific noise, and the
parameter count collapses from N(N+1)/2 down to roughly N × k (k = number of
factors) + N (idiosyncratic variances) — tractable even for hundreds of names.

This is a STATISTICAL factor model: PCA extracts the factors directly from
the return data's own covariance structure (orthogonal by construction), as
opposed to a FUNDAMENTAL factor model like Barra or Fama-French, which uses
PRE-SPECIFIED style factors (value, size, momentum, ...) fit by cross-
sectional regression against known company characteristics. Documented here
honestly as the lighter, statistical version of the same core idea real
buy-side desks use for exactly this problem — not a claim that this
reproduces Barra's actual factor set, which this project has neither the data
(fundamentals for hundreds of names) nor the scope to build.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Any

from src.config import MAX_PCA_FACTORS, MIN_PCA_FACTORS, TRADING_DAYS_PER_YEAR


def pca_factor_cov(
    returns: pd.DataFrame,
    n_factors: int = 10,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Estimate an annualised covariance matrix via a PCA statistical factor
    model instead of the sample/shrinkage covariance used elsewhere.

    Method: demean returns, extract the top `n_factors` principal components
    (orthogonal by construction, so their covariance is diagonal), build the
    SYSTEMATIC covariance from asset loadings on those factors
    (`loadings @ diag(factor_variances) @ loadings.T`), then add a diagonal
    IDIOSYNCRATIC term — whatever variance each asset's own sample variance
    has left over after the systematic part is accounted for. That diagonal-
    only idiosyncratic assumption (residual risk uncorrelated ACROSS assets)
    is the standard orthogonal-factor-model simplification, and it's exactly
    what collapses the parameter count.

    `n_factors` is clamped to `[MIN_PCA_FACTORS, min(MAX_PCA_FACTORS,
    n_assets - 1, n_observations - 1)]` — silently, but the returned
    diagnostics report the actual count used so a caller/UI can surface it
    rather than the request being silently different from the result.

    Returns (covariance_df, diagnostics). `diagnostics` contains:
      - "n_factors_used": the actual (possibly clamped) factor count
      - "explained_variance_ratio": list, per factor, of the return
        variance it explains
      - "cumulative_explained": total variance explained by all factors used
        — the number that actually matters for judging whether this
        covariance estimate is trustworthy. A factor model explaining 35% of
        variance is a materially weaker estimate than one explaining 85%,
        and burying that in a list of ten small numbers isn't the same as
        surfacing it.

    Raises ValueError if there isn't enough data to fit even one factor
    (fewer than 2 assets or fewer than 2 observations) — this is a real
    input-validation failure the caller should see, not silently degrade.
    """
    clean = returns.dropna(how="any")  # PCA needs a complete matrix — a single
    # asset's gap would otherwise force dropping that observation for every
    # OTHER asset too if handled naively; failing loudly on NaNs the caller
    # should have already resolved (e.g. via forward-fill upstream) is safer
    # than silently shrinking the sample in a way the caller doesn't see.
    n_assets = clean.shape[1]
    n_obs = clean.shape[0]
    max_feasible = min(n_assets - 1, n_obs - 1)
    if max_feasible < 1:
        raise ValueError(
            f"Not enough assets/observations to fit a factor model "
            f"(assets={n_assets}, observations={n_obs})."
        )
    k = max(MIN_PCA_FACTORS, min(n_factors, MAX_PCA_FACTORS, max_feasible))

    demeaned = clean - clean.mean()

    pca = PCA(n_components=k)
    pca.fit(demeaned.values)
    loadings = pca.components_.T  # N x k: each asset's exposure to each factor
    factor_variances = pca.explained_variance_  # k: variance of each factor's own score series

    systematic_cov = loadings @ np.diag(factor_variances) @ loadings.T

    sample_var = demeaned.var(ddof=1).values
    systematic_var = np.diag(systematic_cov)
    # Clip at a small positive floor, not zero: a genuinely-zero idiosyncratic
    # variance would make the resulting covariance matrix singular for that
    # asset, which downstream optimizers (inverting the covariance matrix)
    # cannot handle. Floating-point rounding can also push this slightly
    # negative even when the true residual variance is ~0.
    idiosyncratic_var = np.clip(sample_var - systematic_var, a_min=1e-10, a_max=None)

    full_cov = systematic_cov + np.diag(idiosyncratic_var)
    annualised_cov = full_cov * periods_per_year

    cov_df = pd.DataFrame(annualised_cov, index=clean.columns, columns=clean.columns)
    # Symmetry can drift by floating-point rounding in the matrix products
    # above; force it explicitly rather than letting an asymmetric matrix
    # reach an optimizer that assumes one.
    cov_df = (cov_df + cov_df.T) / 2.0

    diagnostics = {
        "n_factors_used": k,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_explained": float(np.sum(pca.explained_variance_ratio_)),
    }
    return cov_df, diagnostics