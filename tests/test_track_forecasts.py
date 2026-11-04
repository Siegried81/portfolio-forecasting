"""
Tests for scripts/track_forecasts.py, the opt-in MLflow tracking bridge.

No MLflow server: every test points track_model_runs at a file store under
tmp_path, as the script defaults to mlruns/ at the repo root. No network
either: track_model_runs takes a price frame, and the one path that fetches
(main via _fetch_prices) has its fetch mocked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.track_forecasts as tf

mlflow = pytest.importorskip("mlflow")


def _synthetic_prices(n_periods: int, tickers: list[str], seed: int = 0) -> pd.DataFrame:
    """Daily synthetic adjusted-close prices on a real date index (so the
    date-range params are exercised), mirroring tests/test_backtesting.py."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-01", periods=n_periods)
    data = {t: 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n_periods)) for t in tickers}
    return pd.DataFrame(data, index=idx)


def _client(tmp_path) -> "mlflow.tracking.MlflowClient":
    """MlflowClient pointed at the same file store track_model_runs used."""
    return mlflow.tracking.MlflowClient(tracking_uri=tf._tracking_uri(tmp_path))


def test_track_model_runs_logs_run_with_expected_params_and_metrics(tmp_path):
    prices = _synthetic_prices(220, ["AAA", "BBB", "CCC"], seed=1)

    run_ids = tf.track_model_runs(
        prices,
        horizon=15,
        n_windows=3,
        models=["Naive (random walk)"],
        tracking_uri=tmp_path,
        experiment_name="test-exp",
        min_train_periods=100,
    )

    assert len(run_ids) == 1

    client = _client(tmp_path)
    experiment = client.get_experiment_by_name("test-exp")
    assert experiment is not None

    run = client.get_run(run_ids[0])
    # Params that define the run.
    assert run.data.params["model_name"] == "Naive (random walk)"
    assert run.data.params["horizon"] == "15"
    assert run.data.params["n_windows"] == "3"
    assert run.data.params["tickers"] == "AAA,BBB,CCC"
    assert run.data.params["n_assets"] == "3"
    # Date range comes from the price index, not hardcoded.
    assert run.data.params["start_date"] == "2023-01-02"
    assert run.data.params["start_date"] < run.data.params["end_date"]

    # The repo's own walk-forward metrics are logged.
    assert "forecast_sharpe_ratio_mean" in run.data.metrics
    assert "forecast_win_rate" in run.data.metrics
    assert 0.0 <= run.data.metrics["forecast_win_rate"] <= 1.0
    assert run.data.metrics["n_windows_evaluated"] == 3.0
    assert np.isfinite(run.data.metrics["forecast_sharpe_ratio_mean"])


def test_track_model_runs_logs_forecast_and_walk_forward_artifacts(tmp_path):
    prices = _synthetic_prices(220, ["AAA", "BBB"], seed=2)
    run_ids = tf.track_model_runs(
        prices, horizon=15, n_windows=3, models=["Naive (random walk)"],
        tracking_uri=tmp_path, experiment_name="test-artifacts", min_train_periods=100,
    )
    client = _client(tmp_path)
    names = {a.path for a in client.list_artifacts(run_ids[0])}
    assert {"walk_forward_results.csv", "walk_forward_summary.csv", "forecast.csv"} <= names


def test_track_model_runs_one_run_per_model(tmp_path):
    prices = _synthetic_prices(220, ["AAA", "BBB"], seed=3)
    models = ["Naive (random walk)", "ETS (Holt linear trend)"]
    run_ids = tf.track_model_runs(
        prices, horizon=15, n_windows=3, models=models,
        tracking_uri=tmp_path, experiment_name="test-multi", min_train_periods=100,
    )
    assert len(run_ids) == len(set(run_ids)) == 2

    client = _client(tmp_path)
    logged_models = {client.get_run(rid).data.params["model_name"] for rid in run_ids}
    assert logged_models == set(models)


def test_date_range_returns_empty_strings_for_non_datetime_index():
    """A RangeIndex has no dates to report; the params must be empty, not the
    epoch that pd.Timestamp(0) would give."""
    prices = pd.DataFrame({"AAA": [1.0, 2.0, 3.0], "BBB": [2.0, 3.0, 4.0]})
    assert isinstance(prices.index, pd.RangeIndex)
    assert tf._date_range(prices) == ("", "")


def test_track_model_runs_rejects_unknown_model(tmp_path):
    prices = _synthetic_prices(150, ["AAA", "BBB"], seed=4)
    with pytest.raises(ValueError, match="Unknown forecast model"):
        tf.track_model_runs(
            prices, horizon=10, n_windows=3, models=["not-a-model"],
            tracking_uri=tmp_path, min_train_periods=100,
        )


def test_main_entrypoint_uses_mocked_fetch_no_network(tmp_path, monkeypatch):
    """The CLI path fetches market data; mock that fetch so no HTTP happens and
    verify it still logs a run to the local store."""
    prices = _synthetic_prices(220, ["AAA", "BBB"], seed=5)
    fetch_calls: list[tuple] = []

    def fake_fetch(tickers, start, end):
        fetch_calls.append((tuple(tickers), start, end))
        return prices

    # Patch the symbol inside src.market_data, which _fetch_prices imports.
    monkeypatch.setattr("src.market_data.fetch_adjusted_close", fake_fetch)
    monkeypatch.setattr(
        "sys.argv",
        ["track_forecasts.py", "--tickers", "AAA", "BBB", "--horizon", "15",
         "--n-windows", "3", "--models", "Naive (random walk)",
         "--tracking-dir", str(tmp_path), "--experiment", "test-cli"],
    )

    tf.main()

    assert len(fetch_calls) == 1
    client = _client(tmp_path)
    experiment = client.get_experiment_by_name("test-cli")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.params["model_name"] == "Naive (random walk)"
