"""Capacity-planning benchmark for src.backtesting.run_walk_forward.

Not gated in CI (see .github/workflows/ci.yml, which only runs this in --quick mode
as a non-blocking smoke test). Run manually before a real deployment to answer
"how long does one walk-forward run take, and how does that scale with universe size?"

Uses synthetic price data (same generator as tests/test_backtesting.py) so this has
no network dependency and no market-data rate limits to worry about — it isolates
compute cost, not I/O cost.

Calls src.backtesting.run_walk_forward directly (see that module for the full
parameter docstring) — risk-free rate, periods-per-year, and similar finance
inputs below are fixed benchmark constants, not tuned defaults, so every
scenario in the grid is comparable to every other one.

Usage:
    python scripts/benchmark_walk_forward.py --quick          # fast subset — Naive/ETS only, no ARIMA
    python scripts/benchmark_walk_forward.py                  # full grid — adds ARIMA + a 40-asset PCA-covariance universe
    python scripts/benchmark_walk_forward.py --csv bench.csv  # also write results to CSV, to track over time
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Make `src/` importable when this script is run directly (python scripts/benchmark_walk_forward.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtesting import run_walk_forward  # noqa: E402  (import after sys.path fix, on purpose)
from src.config import (  # noqa: E402
    COV_METHOD_GARCH,
    COV_METHOD_LEDOIT_WOLF,
    COV_METHOD_PCA,
    DEFAULT_RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    WALK_FORWARD_MIN_TRAIN_PERIODS,
)

# Benchmark-only constants — reuse the repo's real defaults (config.py) wherever
# one exists, so this benchmark stays in sync if those defaults ever change.
# Two deliberate departures from the defaults, both to isolate compute cost:
#   - MAX_WEIGHT_PER_ASSET = 1.0 (unconstrained), not config's 0.35 cap — a
#     tighter cap changes optimizer difficulty, which isn't what's being measured.
#   - TRANSACTION_COST_BPS = 0.0, not config's 10.0 default — this benchmarks
#     compute cost, not cost drag.
RISK_FREE_RATE = DEFAULT_RISK_FREE_RATE
PERIODS_PER_YEAR = TRADING_DAYS_PER_YEAR  # synthetic data is generated at daily frequency throughout
MAX_WEIGHT_PER_ASSET = 1.0
ALLOW_SHORT_SELLING = False
TRANSACTION_COST_BPS = 0.0


@dataclass
class Scenario:
    """One point in the benchmark grid — a single (universe size, windows, model, cov) combination."""

    n_assets: int
    n_windows: int
    forecast_model: str  # one of forecasting.FORECAST_MODELS' keys — see build_grid() below
    forecast_cov_method: str | None  # None (historical) | COV_METHOD_GARCH
    cov_method: str  # "ledoit_wolf" | "pca" — PCA only makes sense once n_assets is large


@dataclass
class BenchmarkResult:
    """One row of the output table — a scenario plus what it cost to run."""

    n_assets: int
    n_windows: int
    forecast_model: str
    forecast_cov_method: str
    cov_method: str
    wall_clock_seconds: float
    peak_memory_mb: float


def make_synthetic_prices(n_assets: int, n_periods: int, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic daily adjusted-close prices via geometric Brownian motion.

    Same generator tests/test_backtesting.py uses for its walk-forward smoke test —
    kept in sync deliberately so a benchmark scenario and its corresponding unit test
    are running on directly comparable data, not two different synthetic universes.
    """
    rng = np.random.default_rng(seed)
    # Realistic-ish daily params: ~8% annual drift, ~20% annual vol, translated to daily.
    daily_mu, daily_sigma = 0.08 / 252, 0.20 / np.sqrt(252)
    log_returns = rng.normal(daily_mu, daily_sigma, size=(n_periods, n_assets))
    prices = 100 * np.exp(np.cumsum(log_returns, axis=0))
    dates = pd.bdate_range("2020-01-01", periods=n_periods)
    tickers = [f"SYN{i:03d}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=tickers)


def run_one_scenario(scenario: Scenario, horizon: int = 90) -> BenchmarkResult:
    """Run a single scenario, timing wall-clock and tracking peak Python-heap memory.

    tracemalloc only tracks Python-level allocations (not numpy's underlying C buffers
    directly, though numpy arrays still register through Python's allocator in most
    cases) — good enough for relative comparison across scenarios, not a substitute
    for a proper memory profiler if you need exact RSS numbers.
    """
    # min_train_periods: reuse the app's own walk-forward floor (config.py) rather
    # than an arbitrary multiple of horizon — the first window's training set needs
    # enough history for a stable initial fit, same requirement the real app has.
    min_train_periods = WALK_FORWARD_MIN_TRAIN_PERIODS
    n_periods = min_train_periods + horizon * scenario.n_windows
    prices = make_synthetic_prices(scenario.n_assets, n_periods)
    tickers = list(prices.columns)

    tracemalloc.start()
    start = time.perf_counter()

    run_walk_forward(
        prices,
        tickers=tickers,
        horizon=horizon,
        n_windows=scenario.n_windows,
        forecast_model=scenario.forecast_model,
        risk_free_rate=RISK_FREE_RATE,
        periods_per_year=PERIODS_PER_YEAR,
        min_train_periods=min_train_periods,
        max_weight_per_asset=MAX_WEIGHT_PER_ASSET,
        allow_short_selling=ALLOW_SHORT_SELLING,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        cov_method=scenario.cov_method,
        forecast_cov_method=scenario.forecast_cov_method,
    )

    wall_clock = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        n_assets=scenario.n_assets,
        n_windows=scenario.n_windows,
        forecast_model=scenario.forecast_model,
        forecast_cov_method=scenario.forecast_cov_method or "historical",
        cov_method=scenario.cov_method,
        wall_clock_seconds=round(wall_clock, 2),
        peak_memory_mb=round(peak_bytes / (1024 * 1024), 2),
    )


def build_grid(quick: bool) -> list[Scenario]:
    """Build the scenario grid. --quick skips ARIMA (the slow model) and the 40-asset PCA case."""
    if quick:
        return [
            Scenario(n_assets=5, n_windows=6, forecast_model="Naive (random walk)", forecast_cov_method=None, cov_method=COV_METHOD_LEDOIT_WOLF),
            Scenario(n_assets=5, n_windows=6, forecast_model="ETS (Holt linear trend)", forecast_cov_method=None, cov_method=COV_METHOD_LEDOIT_WOLF),
            Scenario(n_assets=15, n_windows=6, forecast_model="ETS (Holt linear trend)", forecast_cov_method=None, cov_method=COV_METHOD_LEDOIT_WOLF),
        ]

    scenarios: list[Scenario] = []
    for n_assets, cov_method in [(5, COV_METHOD_LEDOIT_WOLF), (15, COV_METHOD_LEDOIT_WOLF), (40, COV_METHOD_PCA)]:
        for forecast_model in ["Naive (random walk)", "ETS (Holt linear trend)", "Theta method", "ARIMA (auto order)"]:
            for forecast_cov_method in [None, COV_METHOD_GARCH]:
                scenarios.append(
                    Scenario(
                        n_assets=n_assets,
                        n_windows=6,
                        forecast_model=forecast_model,
                        forecast_cov_method=forecast_cov_method,
                        cov_method=cov_method,
                    )
                )
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Fast subset — Naive/ETS only, no ARIMA, no 40-asset case.")
    parser.add_argument("--csv", type=Path, default=None, help="Also write results to this CSV path.")
    args = parser.parse_args()

    grid = build_grid(args.quick)
    results: list[BenchmarkResult] = []

    for i, scenario in enumerate(grid, start=1):
        print(f"[{i}/{len(grid)}] {scenario.n_assets} assets, {scenario.forecast_model}, "
              f"cov={scenario.forecast_cov_method or 'historical'} ({scenario.cov_method})...", end=" ", flush=True)
        result = run_one_scenario(scenario)
        results.append(result)
        print(f"{result.wall_clock_seconds}s, {result.peak_memory_mb} MB")

    print("\n=== Summary ===")
    df = pd.DataFrame([asdict(r) for r in results])
    print(df.to_string(index=False))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(r) for r in results)
        print(f"\nResults written to {args.csv}")


if __name__ == "__main__":
    main()