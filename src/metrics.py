"""
Core financial time-series metrics: returns, risk, and risk-adjusted performance.

Pure functions only (no Streamlit, no I/O) - this is intentional: it's the module
most likely to be unit-tested (see tests/test_metrics.py) and reused outside the
app (a notebook, a batch job), so it must not depend on the UI layer.

Conventions used throughout:
- Returns are simple (arithmetic) period returns, i.e. p(t)/p(t-1) - 1, NOT log
  returns. Simple returns are additive across assets (needed for portfolio-level
  aggregation via weights) but NOT across time; log returns are the reverse. Since
  every metric here is portfolio-level first, time-aggregated second, simple
  returns are the correct choice and we compound them explicitly where needed.
- "Annualised" always means scaled by `periods_per_year`, passed in explicitly
  rather than hardcoded, because the app supports daily/weekly/monthly frequencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple period returns from a price DataFrame. First row is dropped (undefined)."""
    return prices.pct_change().dropna(how="all")


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Weighted-sum portfolio return series.

    Aligns columns explicitly on `weights.index` so a mismatched ticker order
    (a classic silent bug when weights come from a different function) fails loud
    via a KeyError rather than silently multiplying the wrong asset by the wrong weight.
    """
    aligned = asset_returns[weights.index].dropna(how="any")
    return aligned.mul(weights, axis=1).sum(axis=1)


def annualised_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Geometric (compounded) annualised return - correct for multi-period comparisons,
    unlike a naive mean(returns) * periods_per_year which overstates volatile series."""
    compounded_growth = (1.0 + returns).prod()
    n_periods = returns.shape[0]
    if n_periods == 0 or compounded_growth <= 0:
        return float("nan")
    return float(compounded_growth ** (periods_per_year / n_periods) - 1.0)


def annualised_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualised Sharpe ratio: excess return per unit of TOTAL volatility (upside + downside).
    `risk_free_rate` is annual - converted to a per-period rate before subtracting,
    a detail that's easy to get wrong (subtracting the annual rate from period returns
    directly would massively understate excess return).
    """
    period_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_returns = returns - period_rf
    vol = excess_returns.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float((excess_returns.mean() / vol) * np.sqrt(periods_per_year))


def sharpe_ratio_standard_error(
    returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Approximate standard error of the estimated Sharpe ratio (Lo, 2002, "The
    Statistics of Sharpe Ratios" — asymptotic variance under IID returns):
    `SE(SR_period) = sqrt((1 + 0.5*SR_period^2) / n)`, computed at the
    PER-PERIOD Sharpe ratio and period count `n`, then scaled to the same
    annualised units as `sharpe_ratio()` above by `sqrt(periods_per_year)` —
    the same scaling the point estimate itself uses (`SR_annual =
    SR_period * sqrt(periods_per_year)`), so the two numbers stay directly
    comparable (e.g. `sharpe_ratio ± 1.96 * sharpe_ratio_standard_error` for
    a rough 95% CI).

    Exists specifically to put a real number behind the "a Sharpe of 5.89 on
    a 30-period window isn't reliable" caveat stated throughout this app's UI
    — a large SE relative to the point estimate is the statistical reason
    that caveat holds, not just an appeal to intuition about small samples.
    IID-normal-returns is a simplification (real returns show autocorrelation
    and fat tails), same documented-trade-off spirit as the Hurst exponent
    estimator in `timeseries_diagnostics.py` — a rough, honest order-of-
    magnitude figure, not a precise confidence bound.
    """
    n = returns.shape[0]
    if n < 2:
        return float("nan")
    period_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_returns = returns - period_rf
    vol = excess_returns.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    sr_period = excess_returns.mean() / vol
    se_period = np.sqrt((1.0 + 0.5 * sr_period ** 2) / n)
    return float(se_period * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualised Sortino ratio: like Sharpe, but penalises only DOWNSIDE deviation
    (returns below the target/MAR, here the risk-free rate). This is the metric
    that separates candidates who understand risk-adjusted return from those who
    just compute Sharpe and stop - a symmetric-vol strategy and a "small steady
    gains, rare big losses" strategy can share a Sharpe ratio but have very
    different Sortino ratios.
    """
    period_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_returns = returns - period_rf
    downside = excess_returns[excess_returns < 0]
    downside_deviation = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 0.0
    if downside_deviation == 0 or np.isnan(downside_deviation):
        return float("nan")
    return float((excess_returns.mean() / downside_deviation) * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """
    Largest peak-to-trough decline of the cumulative return curve, as a negative
    fraction (e.g. -0.35 = -35%). Computed on the cumulative wealth index rather
    than on prices directly so it works identically for a single asset or an
    already-weighted portfolio return series.
    """
    wealth_index = (1.0 + returns).cumprod()
    running_max = wealth_index.cummax()
    drawdown = wealth_index / running_max - 1.0
    return float(drawdown.min())


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical (non-parametric) VaR at the given confidence level, as a negative
    fraction of the portfolio. E.g. 95% VaR of -0.03 means: on the worst 5% of
    periods historically, the loss exceeded 3%. Historical (not Gaussian) VaR is
    used deliberately - equity returns are fat-tailed, and a normal-distribution
    VaR would understate real tail risk.
    """
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def conditional_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """CVaR / Expected Shortfall: the AVERAGE loss in the tail beyond VaR - a more
    informative risk measure than VaR alone, since VaR says nothing about how bad
    the tail beyond the threshold actually is."""
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def beta_vs_benchmark(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """CAPM beta: covariance(asset, benchmark) / variance(benchmark)."""
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    cov_matrix = aligned.cov()
    benchmark_var = cov_matrix.iloc[1, 1]
    if benchmark_var == 0:
        return float("nan")
    return float(cov_matrix.iloc[0, 1] / benchmark_var)


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Annualised return divided by the ABSOLUTE worst peak-to-trough loss. Where
    Sharpe/Sortino penalise volatility, Calmar penalises the single worst outcome
    an investor actually lived through — the metric a risk committee asks for
    when volatility "looks fine on average" but the drawdown was still brutal.
    Common convention: compute on a 3-year (36-month) window in institutional
    reporting, but the underlying formula is horizon-agnostic; the horizon
    context here is whatever `returns` covers.
    """
    ann_return = annualised_return(returns, periods_per_year)
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return ann_return / abs(mdd)


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualised active return (portfolio minus benchmark) divided by tracking
    error (the volatility OF that difference). This is the metric that answers
    "is this portfolio's edge over the benchmark consistent, or one lucky
    period?" — Sharpe answers a related but different question (risk-adjusted
    return in isolation, not relative to a mandate's benchmark), which is why
    active-mandate performance reviews lead with IR, not Sharpe.
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = active_returns.std(ddof=1) * np.sqrt(periods_per_year)
    if tracking_error == 0 or np.isnan(tracking_error):
        return float("nan")
    # Arithmetic (not geometric) annualisation of the mean active return is the
    # standard IR convention — it isolates the average per-period edge, whereas
    # compounding it would conflate skill with the path-dependence Sharpe already
    # captures elsewhere.
    active_return_annualised = active_returns.mean() * periods_per_year
    return float(active_return_annualised / tracking_error)


def treynor_ratio(
    returns: pd.Series,
    beta: float,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Excess return per unit of SYSTEMATIC (market) risk (beta), rather than per
    unit of TOTAL risk (Sharpe's denominator). Two portfolios can share a Sharpe
    ratio while one carries far more market exposure than the other — Treynor is
    the metric that surfaces that difference, and it's the natural complement to
    Sharpe/Sortino for anyone evaluating how much of the return came from simply
    being exposed to the market vs. genuine diversification/timing skill.
    """
    if beta == 0 or np.isnan(beta):
        return float("nan")
    ann_return = annualised_return(returns, periods_per_year)
    return (ann_return - risk_free_rate) / beta


def jensens_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    beta: float,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Jensen's Alpha (CAPM alpha): the annualised return actually earned ABOVE what
    CAPM predicts given the portfolio's market exposure (beta) alone. This is the
    standard textbook complement to Beta and Treynor — without it, Beta/Treynor
    tell you HOW MUCH market risk was taken, but nothing answers "did that
    risk-taking actually pay off beyond what beta alone would predict?" A
    positive alpha means the portfolio outperformed its own CAPM benchmark (some
    genuine edge, or luck — a single window can't tell those apart, same caveat
    as everywhere else in this app); negative means it underperformed what its
    market exposure alone would predict, even if raw Sharpe/Sortino look
    acceptable in isolation.

    Formula: alpha = R_p − [rf + beta × (R_m − rf)], every return annualised
    first so alpha is expressed on the same annual scale as every other return
    figure in this app.
    """
    ann_return = annualised_return(returns, periods_per_year)
    ann_benchmark_return = annualised_return(benchmark_returns, periods_per_year)
    if np.isnan(beta) or np.isnan(ann_return) or np.isnan(ann_benchmark_return):
        return float("nan")
    expected_return = risk_free_rate + beta * (ann_benchmark_return - risk_free_rate)
    return ann_return - expected_return


def ulcer_index(returns: pd.Series) -> float:
    """
    Ulcer Index (Peter Martin, 1987) — because Max Drawdown
    alone only captures the SINGLE worst point, not how long the portfolio
    stayed underwater. Computed as the root-mean-square of the percentage
    drawdown at EVERY point in the series (in percentage points, Martin's
    original convention — e.g. 12.3, not 0.123), not just the worst one: a
    portfolio stuck at -20% for a year scores far worse here than one that
    dips to -20% for a single week and recovers, even though both share the
    identical Max Drawdown. Lower is better; 0 means the wealth curve never
    dipped below any prior peak. No universal "good" threshold — compare it
    across the three portfolios the same way Calmar is compared, not against
    a memorised number.
    """
    wealth_index = (1.0 + returns).cumprod()
    running_max = wealth_index.cummax()
    drawdown_pct = (wealth_index / running_max - 1.0) * 100.0
    return float(np.sqrt((drawdown_pct ** 2).mean()))


def return_skewness(returns: pd.Series) -> float:
    """
    Skewness of the return distribution (third standardised moment) — added
    alongside kurtosis to make explicit WHY Omega sometimes
    diverges from Sharpe/Sortino, instead of leaving that as an implicit
    "trust the number" (Omega's docstring already flagged this gap). Zero =
    symmetric; negative = a longer/fatter LEFT tail (occasional large losses
    — the shape most equity return series actually have, and exactly the
    asymmetry Sortino/Omega are built to catch that Sharpe's symmetric
    variance can't); positive = a longer/fatter right tail (occasional large
    gains). Uses pandas' own skew() (sample skewness, bias-corrected) — needs
    at least 3 observations, NaN below that.
    """
    return float(returns.skew())


def return_kurtosis(returns: pd.Series) -> float:
    """
    EXCESS kurtosis of the return distribution — added alongside
    skewness, same rationale. Pandas' convention (matches the Fisher
    definition used throughout finance): 0 = same tail thickness as a normal
    distribution. Positive = fatter tails than normal (more extreme outcomes
    than a Gaussian model would predict — the standard finding for equity
    returns, and the reason VaR/CVaR above are computed non-parametrically
    from the empirical distribution rather than assumed normal in the first
    place). Negative = thinner tails than normal (uncommon for real return
    series). Needs at least 4 observations; NaN below that.
    """
    return float(returns.kurt())


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Probability-weighted ratio of total gains to total losses above/below a
    threshold return (0% by default). Unlike Sharpe/Sortino, which reduce the
    whole return DISTRIBUTION to a mean and a (semi-)variance, Omega uses the
    full empirical distribution — it will differ from Sharpe/Sortino precisely
    when returns are skewed or fat-tailed, which is exactly when a
    mean-variance summary is most likely to mislead. A useful second opinion
    alongside Sharpe/Sortino, not a replacement for either.
    """
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def summarise_performance(
    returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, float]:
    """
    One-stop summary dict used by both the UI tables and the LLM commentary
    prompt. `benchmark_returns` is optional: when provided, benchmark-relative
    metrics (beta, information ratio, Treynor) are included; without it, the
    dict still contains every metric that only needs the return series itself.
    """
    summary = {
        "period_return": float((1.0 + returns).prod() - 1.0),  # raw, NOT annualised — the actual
        # compounded return over the sample as given. Kept alongside annual_return specifically so
        # a short window's annualised figure can be sanity-checked against what actually happened:
        # +12% over 30 weeks annualises to a triple-digit number that never occurred and never will
        # again — reporting only the annualised figure without this context is misleading, not
        # just "aggressive rounding".
        "n_periods": int(returns.shape[0]),
        "annual_return": annualised_return(returns, periods_per_year),
        "annual_volatility": annualised_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sharpe_se": sharpe_ratio_standard_error(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "omega_ratio": omega_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "ulcer_index": ulcer_index(returns),
        "skewness": return_skewness(returns),
        "kurtosis": return_kurtosis(returns),
        "var_95": value_at_risk(returns, 0.95),
        "cvar_95": conditional_value_at_risk(returns, 0.95),
    }
    if benchmark_returns is not None:
        beta = beta_vs_benchmark(returns, benchmark_returns)
        summary["beta"] = beta
        summary["information_ratio"] = information_ratio(returns, benchmark_returns, periods_per_year)
        summary["treynor_ratio"] = treynor_ratio(returns, beta, risk_free_rate, periods_per_year)
        summary["jensens_alpha"] = jensens_alpha(returns, benchmark_returns, beta, risk_free_rate, periods_per_year)
    return summary


def compute_turnover(new_weights: pd.Series, old_weights: pd.Series | None) -> float:
    """
    Sum of absolute weight changes when rebalancing from `old_weights` to
    `new_weights` — the standard turnover measure a transaction-cost estimate is
    built on (cost ≈ turnover × cost rate). `old_weights=None` means starting
    from all-cash (e.g. the very first rebalance in a backtest), in which case
    turnover is simply the sum of the new weights' magnitudes (everything has to
    be bought from scratch).
    """
    if old_weights is None:
        return float(new_weights.abs().sum())
    idx = new_weights.index.union(old_weights.index)
    diff = new_weights.reindex(idx, fill_value=0.0) - old_weights.reindex(idx, fill_value=0.0)
    return float(diff.abs().sum())


def apply_transaction_cost(returns: pd.Series, turnover: float, cost_bps: float) -> pd.Series:
    """
    Charge a one-time transaction cost (turnover × cost rate) against a return
    series, applied at the moment of rebalancing — i.e. against the FIRST period
    of the series, since that's when the trades to reach the new weights
    actually happen. Multiplicative (not a flat subtraction): the cost eats into
    whatever wealth exists at that point, which compounds correctly with the
    rest of the series rather than becoming a slightly-wrong flat percentage
    shift.

    `cost_bps` is in basis points (1 bp = 0.01%) — typical retail/ETF trading
    costs are in the 1-20 bps range per trade; institutional desks can be lower
    for liquid large-caps, higher for illiquid names. Returns an unmodified copy
    if the series is empty or turnover/cost is zero.
    """
    if len(returns) == 0 or turnover == 0 or cost_bps == 0:
        return returns.copy()
    cost = turnover * cost_bps / 10_000
    adjusted = returns.copy()
    adjusted.iloc[0] = (1.0 + adjusted.iloc[0]) * (1.0 - cost) - 1.0
    return adjusted