"""
Fama-French factor data and multi-factor exposure analysis.

Why this earns its place: `metrics.py`'s `beta_vs_benchmark`/`jensens_alpha`
already answer "how much of this portfolio's return is explained by ONE
factor — the overall market?" (single-factor CAPM). That's the textbook
starting point, but a real factor-based read of a portfolio asks a wider
question: how much of its return comes from tilting toward small-caps (SMB),
value stocks (HML), or other well-documented return premia, on top of plain
market exposure? Ken French's Data Library is the standard, free, no-key
source for exactly this — the same data academic asset-pricing research has
used since the original Fama-French (1993) paper.

Two entry points:
- `fetch_fama_french_factors()` downloads and parses the raw factor return
  series (free, no API key, no signup — a public ZIP/CSV file).
- `compute_factor_exposures()` regresses a portfolio's own excess returns on
  those factors (OLS via statsmodels, already a hard dependency of this
  project via `forecasting.py`) to get per-factor loadings, an annualised
  alpha, and R² — the multi-factor generalisation of `metrics.jensens_alpha`.

Fails soft on any error (network, malformed file, too little overlapping
history) — same contract as every other fetcher in this codebase
(news_data.py, macro_data.py, academic_search.py): factor exposure is
enrichment for the AI Analyst / an extra tab, never something that can take
the core optimizer/backtest down.

Field-shape caveat, same as `news_data.fetch_gdelt_news`/`fetch_ted_notices`:
the CSV layout parsed here (header/footer junk lines wrapped around a plain
date-indexed table) was implemented from the Data Library's long-standing,
publicly documented file format, not verified against a live download in
this sandboxed environment (network access here is restricted to package
registries). If a downloaded file parses to an empty/wrong-shaped result,
inspect the raw CSV directly before assuming this parser is exhaustive.
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
from typing import Any

import pandas as pd
import requests

from src.cache import cached
from src.config import TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)

# Ken French's Data Library — Dartmouth Tuck School of Business. Free, no
# API key, no signup, no rate limit beyond ordinary courtesy. Daily
# frequency chosen to match this app's own default frequency (config.py's
# FREQUENCY_TO_PERIODS_PER_YEAR); the Data Library also publishes weekly and
# monthly files at the same base URL with different filenames, not wired up
# here since this app's own data defaults to daily.
FAMA_FRENCH_3_FACTOR_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)
FAMA_FRENCH_5_FACTOR_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)

# The Data Library's own factor names, kept exactly as published (not
# relabelled to this app's usual snake_case) so a reader cross-referencing
# the original academic papers or the Data Library's own documentation sees
# the same symbols. RF (the risk-free rate) is handled separately — it's a
# regression INPUT (used to compute excess returns), not itself a factor
# loading a portfolio is exposed to.
THREE_FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML"]
FIVE_FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

MIN_OBSERVATIONS_FOR_FACTOR_REGRESSION = 30  # same spirit as
# timeseries_diagnostics.py's MIN_OBSERVATIONS_FOR_ADF — below this an OLS
# fit is technically computable but not a result worth trusting or showing.

_DATE_LINE_RE = re.compile(r"^\s*(\d{8}|\d{6}),")  # a data row starts with an
# 8-digit (daily/monthly-as-YYYYMMDD) or 6-digit (YYYYMM) date, comma-separated
# — the one structural signal that reliably distinguishes a real data row from
# the Data Library's surrounding title/copyright/footer text, which doesn't
# follow this shape.


def _extract_data_block(csv_text: str) -> pd.DataFrame | None:
    """
    Ken French's published CSVs wrap the actual data table in title and
    copyright/footer text that isn't itself tabular — this pulls out just the
    contiguous run of genuine data rows (see `_DATE_LINE_RE`) and the header
    line immediately above them, and parses that block alone. Returns None
    (never raises) if no data-shaped block is found at all.
    """
    lines = csv_text.splitlines()
    data_line_indices = [i for i, line in enumerate(lines) if _DATE_LINE_RE.match(line)]
    if not data_line_indices:
        return None

    start = data_line_indices[0]
    # The contiguous run ends at the first gap in data_line_indices (the
    # Data Library sometimes appends a second table, e.g. annual factors,
    # after a blank/footer line — only the FIRST contiguous block is used).
    end = start
    for idx in data_line_indices:
        if idx != end:
            break
        end = idx + 1

    header_line = lines[start - 1] if start > 0 else ""
    block = "\n".join([header_line] + lines[start:end])
    try:
        df = pd.read_csv(io.StringIO(block), index_col=0)
    except Exception as exc:
        logger.warning("Fama-French data block failed to parse as CSV: %s", exc)
        return None

    df.columns = [c.strip() for c in df.columns]
    df.index.name = "date"
    return df


@cached(ttl_seconds=7 * 24 * 3600)  # French updates this monthly at most — a
# week's cache is generous headroom, not a risk of serving stale-by-design data.
def fetch_fama_french_factors(model: str = "3-factor") -> pd.DataFrame | None:
    """
    Download and parse daily Fama-French factor returns. `model` is
    `"3-factor"` (Mkt-RF, SMB, HML) or `"5-factor"` (adds RMW, CMA — the
    profitability and investment factors from Fama & French, 2015).

    Returns a DataFrame indexed by date (as decimal fractions, e.g. 0.0012
    for 0.12% — the Data Library itself publishes percentage POINTS, divided
    by 100 here to match every other return series in this codebase) with
    columns for each factor plus `RF` (the risk-free rate, needed by
    `compute_factor_exposures` to convert portfolio returns to EXCESS
    returns before regressing). Returns None (never raises) on any failure —
    network error, unexpected file layout, or an unrecognised `model`.
    """
    url = {"3-factor": FAMA_FRENCH_3_FACTOR_URL, "5-factor": FAMA_FRENCH_5_FACTOR_URL}.get(model)
    if url is None:
        logger.warning("Unrecognised Fama-French model %r (expected '3-factor' or '5-factor')", model)
        return None

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                logger.warning("Fama-French ZIP for %s contained no CSV member", model)
                return None
            csv_text = archive.read(csv_names[0]).decode("utf-8", errors="replace")
    except (requests.RequestException, zipfile.BadZipFile, OSError) as exc:
        logger.warning("Fama-French fetch failed for %s: %s", model, exc)
        return None

    df = _extract_data_block(csv_text)
    if df is None or df.empty:
        return None

    try:
        df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    except ValueError:
        try:
            df.index = pd.to_datetime(df.index.astype(str), format="%Y%m")
        except ValueError as exc:
            logger.warning("Fama-French date index for %s did not match YYYYMMDD or YYYYMM: %s", model, exc)
            return None

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0
    return df.dropna(how="all")


def compute_factor_exposures(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, Any] | None:
    """
    Regress a portfolio's excess returns on Fama-French factor returns via
    OLS (statsmodels — already a hard dependency via `forecasting.py`,
    reused rather than adding `sklearn.linear_model` for the same job).

    `factors` is whatever `fetch_fama_french_factors` returned — this
    function auto-detects 3-factor vs 5-factor from which columns are
    present (`THREE_FACTOR_COLUMNS`/`FIVE_FACTOR_COLUMNS`), so callers don't
    pass the model choice twice.

    Aligns `portfolio_returns` and `factors` on their shared dates (an inner
    join — dates either series doesn't have are simply excluded, not
    forward-filled or otherwise fabricated) before fitting.

    Returns None (never raises) if there are fewer than
    `MIN_OBSERVATIONS_FOR_FACTOR_REGRESSION` overlapping dates, or the
    regression itself fails for any reason (e.g. degenerate/collinear
    factor data). Otherwise returns:
      - "alpha_annualised": the regression intercept, annualised the same
        way `metrics.annualised_return` compounds a per-period figure — the
        return this portfolio earned that its factor exposures DON'T
        explain (the multi-factor generalisation of Jensen's alpha).
      - "loadings": {factor_name: coefficient} — how many units of excess
        portfolio return move with one unit of that factor's own return.
      - "r_squared": how much of the portfolio's excess-return variance the
        factor set explains — the honest complement to the loadings
        themselves (high loadings from a low-R² fit are a noisy fit, not a
        confident exposure read).
      - "n_obs": the actual overlapping sample size the fit used, so a
        caller/UI can flag a thin-history result rather than presenting it
        with the same confidence as a multi-year fit.
    """
    factor_columns = [c for c in FIVE_FACTOR_COLUMNS if c in factors.columns]
    if len(factor_columns) < len(THREE_FACTOR_COLUMNS) or "RF" not in factors.columns:
        logger.warning("Fama-French factors DataFrame is missing expected columns; got %s", list(factors.columns))
        return None

    aligned = pd.concat([portfolio_returns.rename("portfolio"), factors], axis=1, join="inner").dropna()
    if len(aligned) < MIN_OBSERVATIONS_FOR_FACTOR_REGRESSION:
        return None

    excess_returns = aligned["portfolio"] - aligned["RF"]
    X = aligned[factor_columns]

    try:
        import statsmodels.api as sm

        X_with_const = sm.add_constant(X)
        fit = sm.OLS(excess_returns, X_with_const).fit()
    except Exception as exc:
        logger.warning("Fama-French factor regression failed: %s", exc)
        return None

    alpha_per_period = float(fit.params.get("const", 0.0))
    return {
        "alpha_annualised": (1.0 + alpha_per_period) ** periods_per_year - 1.0,
        "loadings": {factor: float(fit.params[factor]) for factor in factor_columns},
        "r_squared": float(fit.rsquared),
        "n_obs": int(len(aligned)),
    }