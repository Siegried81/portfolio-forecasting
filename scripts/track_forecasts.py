"""
Opt-in MLflow experiment tracking for the forecasting models.

The walk-forward backtest in src.backtesting already scores the six models
out of sample, but nothing keeps those numbers between runs. This script logs
each run to a local MLflow experiment, so you can compare models, horizons and
tickers later.

It reuses forecast_all_assets, run_walk_forward, summarise_walk_forward and
forecast_win_rate as they are and logs their numbers. It doesn't redefine any
measurement. The store is a local directory (mlruns/ by default); mlflow
creates it on first write, no server needed.

Usage:
    # Track the default universe over real market data (network fetch):
    python scripts/track_forecasts.py --tickers AAPL MSFT SPY --years 3

    # Programmatic / tested entrypoint takes a price frame directly (no network):
    from scripts.track_forecasts import track_model_runs
    track_model_runs(prices, horizon=20, n_windows=3, tracking_uri=...)
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

# MLflow 3.x put the local filesystem tracking backend ("./mlruns") in
# maintenance mode: it raises unless this opt-out is set, steering users to a
# database backend. A local file store is exactly what this script wants, it
# needs no server and no extra service, so we take the documented opt-out.
# setdefault so a value already exported in the shell wins. This script always
# points mlflow at a local directory (see _tracking_uri); it doesn't read
# MLFLOW_TRACKING_URI.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtesting import (  # noqa: E402
    forecast_win_rate,
    run_walk_forward,
    summarise_walk_forward,
)
from src.config import (  # noqa: E402
    DEFAULT_COV_METHOD,
    DEFAULT_RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    WALK_FORWARD_MIN_TRAIN_PERIODS,
)
from src.forecasting import FORECAST_MODELS, forecast_all_assets  # noqa: E402

# Default local file tracking store, next to the repo root. A relative "mlruns"
# would land wherever the process was launched; anchoring it to the repo keeps
# every run in one place regardless of the caller's working directory.
DEFAULT_TRACKING_DIR = Path(__file__).resolve().parent.parent / "mlruns"
DEFAULT_EXPERIMENT_NAME = "portfolio-forecasting"

# The walk-forward summary reports mean and std of these per portfolio type.
# We log the Forecast-based portfolio's means plus the cross-model win rate,
# the numbers a "which model should I trust?" comparison actually turns on.
_SUMMARY_METRICS = [
    "annual_return",
    "annual_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
]


def _tracking_uri(tracking_uri: str | Path | None) -> str:
    """Resolve a tracking target to a file:// URI mlflow accepts on any OS.

    A bare Windows path (``D:\\...\\mlruns``) is ambiguous to mlflow's URI
    parser (the drive letter looks like a scheme), so we always hand it a
    proper file URI built from an absolute path.
    """
    target = Path(tracking_uri) if tracking_uri is not None else DEFAULT_TRACKING_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target.resolve().as_uri()


def _date_range(prices: pd.DataFrame) -> tuple[str, str]:
    """Start/end of the price index as ISO dates, or empty strings if the
    index is not datetime-like (e.g. a plain RangeIndex in a synthetic test)."""
    idx = prices.index
    if len(idx) == 0 or not isinstance(idx, pd.DatetimeIndex):
        return "", ""
    return idx.min().date().isoformat(), idx.max().date().isoformat()


def track_model_runs(
    prices: pd.DataFrame,
    *,
    horizon: int,
    n_windows: int,
    models: list[str] | None = None,
    tracking_uri: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    min_train_periods: int = WALK_FORWARD_MIN_TRAIN_PERIODS,
    cov_method: str = DEFAULT_COV_METHOD,
    frequency: str = "daily",
) -> list[str]:
    """Run the existing walk-forward evaluation for each model and log one
    MLflow run per model to a local file store.

    For every model we log, as params, what defines the run (model_name,
    horizon, n_windows, tickers, date range, n_assets, frequency, cov_method),
    and as metrics the repo's own walk-forward numbers: the Forecast-based
    portfolio's mean annual_return / volatility / Sharpe / Sortino / max
    drawdown across windows, plus forecast_win_rate (the fraction of windows
    the forecast beat the plain historical portfolio on Sharpe). The raw
    per-window results, the summary table and the model's forecast over the
    full history are attached as CSV artifacts, the closest thing to a model
    output here, since these models have no fitted object worth pickling.

    Returns the MLflow run id of each logged run, in model order. Imports
    mlflow lazily so importing this module (and the rest of the test suite)
    does not require mlflow to be installed.
    """
    import mlflow

    model_names = models if models is not None else list(FORECAST_MODELS)
    unknown = [m for m in model_names if m not in FORECAST_MODELS]
    if unknown:
        raise ValueError(f"Unknown forecast model(s): {unknown}. "
                         f"Known models: {list(FORECAST_MODELS)}")

    tickers = list(prices.columns)
    start_date, end_date = _date_range(prices)

    mlflow.set_tracking_uri(_tracking_uri(tracking_uri))
    mlflow.set_experiment(experiment_name)

    run_ids: list[str] = []
    for model_name in model_names:
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_params({
                "model_name": model_name,
                "horizon": horizon,
                "n_windows": n_windows,
                "tickers": ",".join(tickers),
                "n_assets": len(tickers),
                "start_date": start_date,
                "end_date": end_date,
                "frequency": frequency,
                "cov_method": cov_method,
                "min_train_periods": min_train_periods,
            })

            results = run_walk_forward(
                prices,
                tickers=tickers,
                horizon=horizon,
                n_windows=n_windows,
                forecast_model=model_name,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
                min_train_periods=min_train_periods,
                cov_method=cov_method,
                frequency=frequency,
            )

            summary = summarise_walk_forward(results)
            metrics: dict[str, float] = {}
            if "Forecast-based" in summary.index:
                for metric in _SUMMARY_METRICS:
                    value = summary.loc["Forecast-based", (metric, "mean")]
                    if pd.notna(value):
                        metrics[f"forecast_{metric}_mean"] = float(value)
            win_rate = forecast_win_rate(results)
            if pd.notna(win_rate):
                metrics["forecast_win_rate"] = float(win_rate)
            # n_windows_evaluated can differ from the requested n_windows when
            # early windows are skipped for lack of per-ticker history; log the
            # realised count so a metric read against too few windows is visible.
            metrics["n_windows_evaluated"] = float(results["window"].nunique())
            mlflow.log_metrics(metrics)

            _log_artifacts(mlflow, prices, results, summary, model_name, horizon, frequency)
            run_ids.append(run.info.run_id)

    return run_ids


def _log_artifacts(
    mlflow,
    prices: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    model_name: str,
    horizon: int,
    frequency: str,
) -> None:
    """Attach the walk-forward results, the summary, and the model's forecast
    over the full price history as CSV artifacts. Written to a temp dir and
    handed to mlflow.log_artifact, so nothing lands in the repo tree."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        results_csv = tmp_path / "walk_forward_results.csv"
        results.to_csv(results_csv, index=False)
        mlflow.log_artifact(str(results_csv))

        summary_csv = tmp_path / "walk_forward_summary.csv"
        summary.to_csv(summary_csv)
        mlflow.log_artifact(str(summary_csv))

        # The forecast itself over the whole history: the closest thing to a
        # "model output artifact" for estimators we do not pickle. Guarded
        # because a model can raise on a degenerate universe; a failed forecast
        # artifact must not sink the run whose metrics already logged.
        try:
            forecast = forecast_all_assets(prices, horizon, model_name, frequency)
            forecast_csv = tmp_path / "forecast.csv"
            forecast.to_csv(forecast_csv)
            mlflow.log_artifact(str(forecast_csv))
        except Exception as exc:  # pragma: no cover - defensive, universe-dependent
            mlflow.set_tag("forecast_artifact_error", f"{type(exc).__name__}: {exc}")


def _fetch_prices(tickers: list[str], years: int) -> pd.DataFrame:
    """Fetch daily adjusted close for the CLI entrypoint via the repo's own
    market-data layer. Kept out of track_model_runs so the tracking logic is
    testable without any network call."""
    from src.market_data import fetch_adjusted_close

    end = dt.date.today()
    start = end - dt.timedelta(days=round(years * 365.25))
    return fetch_adjusted_close(tickers, start, end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "SPY"],
                        help="Tickers to fetch and forecast.")
    parser.add_argument("--years", type=int, default=3, help="Years of history to fetch.")
    parser.add_argument("--horizon", type=int, default=20, help="Forecast horizon in periods.")
    parser.add_argument("--n-windows", type=int, default=3, help="Walk-forward windows.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of model names to track (default: all).")
    parser.add_argument("--tracking-dir", type=Path, default=None,
                        help="Local MLflow file store directory (default: mlruns/ at the repo root).")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT_NAME,
                        help="MLflow experiment name.")
    args = parser.parse_args()

    prices = _fetch_prices(args.tickers, args.years)
    run_ids = track_model_runs(
        prices,
        horizon=args.horizon,
        n_windows=args.n_windows,
        models=args.models,
        tracking_uri=args.tracking_dir,
        experiment_name=args.experiment,
    )
    print(f"Logged {len(run_ids)} run(s) to experiment '{args.experiment}'.")
    for run_id in run_ids:
        print(f"  {run_id}")


if __name__ == "__main__":
    main()
