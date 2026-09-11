"""
Capacity-planning benchmark for src.backtesting.run_walk_forward. Not gated
in CI (only run in --quick mode as a non-blocking smoke test). Uses synthetic
price data, no network dependency.

Usage:
    python scripts/benchmark_walk_forward.py --quick
    python scripts/benchmark_walk_forward.py
    python scripts/benchmark_walk_forward.py --csv bench.csv
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtesting import run_walk_forward  # noqa: E402
from src.config import (  # noqa: E402
    COV_METHOD_GARCH,
    COV_METHOD_LEDOIT_WOLF,
    COV_METHOD_PCA,
    DEFAULT_RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    WALK_FORWARD_MIN_TRAIN_PERIODS,
)

# Benchmark-only constants, reusing config.py's defaults where one exists.
# MAX_WEIGHT_PER_ASSET=1.0 (unconstrained) and TRANSACTION_COST_BPS=0.0 are
# deliberate departures — this benchmarks compute cost, not the effect of a
# tighter cap or cost drag.
RISK_FREE_RATE = DEFAULT_RISK_FREE_RATE
PERIODS_PER_YEAR = TRADING_DAYS_PER_YEAR
MAX_WEIGHT_PER_ASSET = 1.0
ALLOW_SHORT_SELLING = False
TRANSACTION_COST_BPS = 0.0


@dataclass
class Scenario:
    n_assets: int
    n_windows: int
    forecast_model: str
    forecast_cov_method: str | None
    cov_method: str


@dataclass
class BenchmarkResult:
    n_assets: int
    n_windows: int
    forecast_model: str
    forecast_cov_method: str
    cov_method: str
    wall_clock_seconds: float
    peak_memory_mb: float


def make_synthetic_prices(n_assets: int, n_periods: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic daily adjusted-close prices via geometric Brownian motion."""
    rng = np.random.default_rng(seed)
    daily_mu, daily_sigma = 0.08 / 252, 0.20 / np.sqrt(252)
    log_returns = rng.normal(daily_mu, daily_sigma, size=(n_periods, n_assets))
    prices = 100 * np.exp(np.cumsum(log_returns, axis=0))
    dates = pd.bdate_range("2020-01-01", periods=n_periods)
    tickers = [f"SYN{i:03d}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=tickers)


def run_one_scenario(scenario: Scenario, horizon: int = 90) -> BenchmarkResult:
    """Time wall-clock and track peak Python-heap memory for one scenario."""
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
    """--quick skips ARIMA and the 40-asset PCA case."""
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