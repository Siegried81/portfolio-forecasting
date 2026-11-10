"""
Opt-in MLflow experiment tracking for the forecasting models.

The walk-forward backtest in src.backtesting already scores the six models
out of sample, but nothing keeps those numbers between runs. This script logs
each run to a local MLflow experiment, so you can compare models, horizons and
tickers later.

It reuses forecast_all_assets, run_walk_forward, summarise_walk_forward and
forecast_win_rate as they are and logs their numbers. It doesn't redefine any
measurement. The store is a local SQLite database (mlflow.db at the repo root
by default, artifacts under mlartifacts/ next to it); mlflow creates both on
first write, no server needed. MLFLOW_TRACKING_URI, when set, overrides the
default; --tracking-dir (or tracking_uri=) overrides both.

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
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

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

# Default local tracking store: a SQLite database at the repo root. MLflow 3
# puts the filesystem backend ("./mlruns") in maintenance mode, slated for
# deprecation, and makes SQLite its default backend, so we follow it. A relative
# "mlflow.db" would land wherever the process was launched; anchoring it to the
# repo keeps every run in one place regardless of the caller's working directory.
DEFAULT_TRACKING_DIR = Path(__file__).resolve().parent.parent
DB_FILENAME = "mlflow.db"
ARTIFACTS_DIRNAME = "mlartifacts"
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


def _local_store_dir(tracking_uri: str | Path | None) -> Path | None:
    """The local directory that holds mlflow.db and mlartifacts/, or None when
    the target is a URI the caller chose (MLFLOW_TRACKING_URI or an explicit
    ``scheme://`` string), in which case mlflow owns the layout.

    Precedence: an explicit tracking_uri, then MLFLOW_TRACKING_URI, then the
    repo root.
    """
    if tracking_uri is not None:
        if isinstance(tracking_uri, str) and "://" in tracking_uri:
            return None
        return Path(tracking_uri)
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    return DEFAULT_TRACKING_DIR


def _tracking_uri(tracking_uri: str | Path | None) -> str:
    """Resolve a tracking target to a URI mlflow accepts on any OS.

    A directory becomes ``sqlite:///<dir>/mlflow.db``. The path is absolute and
    written with forward slashes, which SQLAlchemy reads the same way on
    Windows (``sqlite:///D:/.../mlflow.db``) and POSIX
    (``sqlite:////home/.../mlflow.db``). A URI (explicit or from
    MLFLOW_TRACKING_URI) is passed through untouched.
    """
    store_dir = _local_store_dir(tracking_uri)
    if store_dir is None:
        return str(tracking_uri) if tracking_uri is not None else os.environ["MLFLOW_TRACKING_URI"]
    store_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(store_dir.resolve() / DB_FILENAME).as_posix()}"


def _set_experiment(mlflow, experiment_name: str, store_dir: Path | None) -> None:
    """Select the experiment, creating it with its artifacts next to the local
    database when there is one. Without an explicit artifact_location, MLflow's
    SQLite backend stores artifacts under ./mlruns relative to the current
    working directory, which would scatter them depending on where the script
    was launched."""
    if store_dir is not None and mlflow.get_experiment_by_name(experiment_name) is None:
        artifacts = (store_dir.resolve() / ARTIFACTS_DIRNAME).as_uri()
        mlflow.create_experiment(experiment_name, artifact_location=artifacts)
    mlflow.set_experiment(experiment_name)


def prices_fingerprint(prices: pd.DataFrame) -> str:
    """SHA-256 of the price frame (column names, index and values).

    Logged with every run so two runs can be checked to have used the exact
    same data: prices fetched over the network change as the provider revises
    history or the date window moves, and the date range alone doesn't show it.
    """
    digest = hashlib.sha256()
    digest.update("\x1f".join(map(str, prices.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(prices, index=True).to_numpy().tobytes())
    return digest.hexdigest()


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
    MLflow run per model to a local SQLite tracking store.

    For every model we log, as params, what defines the run (model_name,
    horizon, n_windows, tickers, date range, n_assets, frequency, cov_method,
    and prices_sha256, a fingerprint of the exact price data used),
    and as metrics the repo's own walk-forward numbers: the Forecast-based
    portfolio's mean annual_return / volatility / Sharpe / Sortino / max
    drawdown across windows, plus forecast_win_rate (the fraction of windows
    the forecast beat the plain historical portfolio on Sharpe) and
    forecast_mase_mean (the price forecast's MASE, see
    metrics.mean_absolute_scaled_error, averaged over windows). The raw
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

    prices_sha256 = prices_fingerprint(prices)

    mlflow.set_tracking_uri(_tracking_uri(tracking_uri))
    _set_experiment(mlflow, experiment_name, _local_store_dir(tracking_uri))

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
                "prices_sha256": prices_sha256,
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
            # MASE of the price forecast, one value per window: mean over windows.
            if "forecast_mase" in results.columns:
                mase = results.groupby("window")["forecast_mase"].first().mean()
                if pd.notna(mase):
                    metrics["forecast_mase_mean"] = float(mase)
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
                        help="Directory for the local SQLite store (mlflow.db) and its "
                             "mlartifacts/ (default: MLFLOW_TRACKING_URI if set, "
                             "else the repo root).")
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
