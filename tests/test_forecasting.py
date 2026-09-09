"""
Unit tests for src/forecasting.py.

Priority: the FALLBACK branches, not the happy path — a silently-skipped fallback
is invisible in the UI (the app just looks like it's using ARIMA when it's
actually degraded to naive) unless it's covered by a test that forces the
trigger condition directly.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from src.config import MIN_HISTORY_POINTS_FOR_FORECAST, MIN_HISTORY_POINTS_FOR_LSTM
from src.forecasting import (
    FORECAST_MODELS,
    arima_forecast,
    ets_forecast,
    forecast_all_assets,
    lstm_forecast,
    ml_regression_forecast,
    naive_forecast,
    theta_forecast,
)


def _price_series(n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    # Geometric-ish random walk, always positive
    returns = rng.normal(0.0005, 0.01, n)
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices, index=idx, name="PRICE")


# ---------------------------------------------------------------------------
# naive_forecast
# ---------------------------------------------------------------------------

def test_naive_forecast_shape_and_index():
    prices = _price_series(60)
    result = naive_forecast(prices, horizon=10)
    assert len(result["forecast"]) == 10
    assert (result["upper"] >= result["forecast"]).all()
    assert (result["lower"] <= result["forecast"]).all()
    # future index must continue strictly after the last observed date
    assert result["forecast"].index[0] > prices.index[-1]


# ---------------------------------------------------------------------------
# ets_forecast — short-history fallback
# ---------------------------------------------------------------------------

def test_ets_forecast_falls_back_to_naive_on_short_history():
    short_prices = _price_series(MIN_HISTORY_POINTS_FOR_FORECAST - 5)
    result = ets_forecast(short_prices, horizon=10)
    expected = naive_forecast(short_prices, horizon=10)
    pd.testing.assert_series_equal(result["forecast"], expected["forecast"])


def test_ets_forecast_runs_on_sufficient_history():
    prices = _price_series(120)
    result = ets_forecast(prices, horizon=15)
    assert len(result["forecast"]) == 15
    assert not result["forecast"].isna().any()


def test_ets_and_arima_forecast_do_not_raise_supported_index_futurewarning():
    # Fitting directly on `prices`'s real trading-day DatetimeIndex
    # (irregular due to weekends/holidays) triggers `FutureWarning: No
    # supported index is available` from statsmodels — its own message warns
    # a future statsmodels release will RAISE instead of merely warn here.
    # Both functions fit on a plain positional index instead (see
    # forecasting.py's comments), sidestepping the ambiguity at the root
    # rather than suppressing the symptom.
    prices = _price_series(120)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ets_forecast(prices, horizon=10)
        arima_forecast(prices, horizon=10, max_p=1, max_q=1)
    supported_index_warnings = [
        w for w in caught
        if issubclass(w.category, FutureWarning) and "supported index" in str(w.message)
    ]
    assert supported_index_warnings == []


# ---------------------------------------------------------------------------
# theta_forecast — short-history fallback, shape, and the widening-band
# convention shared with every other model here
# ---------------------------------------------------------------------------

def test_theta_forecast_falls_back_to_naive_on_short_history():
    short_prices = _price_series(MIN_HISTORY_POINTS_FOR_FORECAST - 5)
    result = theta_forecast(short_prices, horizon=10)
    expected = naive_forecast(short_prices, horizon=10)
    pd.testing.assert_series_equal(result["forecast"], expected["forecast"])


def test_theta_forecast_runs_on_sufficient_history():
    prices = _price_series(120)
    result = theta_forecast(prices, horizon=15)
    assert len(result["forecast"]) == 15
    assert not result["forecast"].isna().any()
    assert (result["forecast"].index > prices.index[-1]).all()


def test_theta_forecast_confidence_band_widens_with_horizon():
    prices = _price_series(120, seed=2)
    result = theta_forecast(prices, horizon=20)
    band_width = (result["upper"] - result["lower"]).values
    assert band_width[-1] > band_width[0]
    assert (np.diff(band_width) >= -1e-9).all()  # small numerical slack


def test_theta_forecast_lower_upper_bracket_the_point_forecast():
    prices = _price_series(120, seed=3)
    result = theta_forecast(prices, horizon=10)
    assert (result["lower"] <= result["forecast"]).all()
    assert (result["forecast"] <= result["upper"]).all()


def test_theta_forecast_recovers_a_pure_linear_trend_closely():
    # A (near-)noiseless linear trend is the one case with a known right
    # answer: both theta-lines should agree closely with the trend itself,
    # so the averaged point forecast should track the true continuation.
    idx = pd.bdate_range("2024-01-01", periods=150)
    trend_prices = pd.Series(100 + 0.5 * np.arange(150), index=idx)
    result = theta_forecast(trend_prices, horizon=10)
    true_continuation = 100 + 0.5 * np.arange(150, 160)
    np.testing.assert_allclose(result["forecast"].values, true_continuation, rtol=0.05)


def test_theta_forecast_falls_back_to_naive_when_ses_fails(monkeypatch):
    import src.forecasting as forecasting_module

    prices = _price_series(120, seed=4)

    def _broken_ses(*args, **kwargs):
        raise RuntimeError("simulated SES failure")

    monkeypatch.setattr(forecasting_module, "SimpleExpSmoothing", _broken_ses)
    result = theta_forecast(prices, horizon=10)
    expected = naive_forecast(prices, horizon=10)
    pd.testing.assert_series_equal(result["forecast"], expected["forecast"])


# ---------------------------------------------------------------------------
# arima_forecast — short-history fallback + non-convergence fallback
# ---------------------------------------------------------------------------

def test_arima_forecast_falls_back_to_naive_on_short_history():
    short_prices = _price_series(MIN_HISTORY_POINTS_FOR_FORECAST - 5)
    result = arima_forecast(short_prices, horizon=10)
    expected = naive_forecast(short_prices, horizon=10)
    pd.testing.assert_series_equal(result["forecast"], expected["forecast"])


def test_arima_forecast_falls_back_to_naive_when_no_order_converges():
    # A perfectly constant series makes every ARIMA(p,1,q) order degenerate
    # (zero variance after differencing) — this must fall back to naive
    # rather than raising, since forecast_all_assets has no other safety net.
    idx = pd.bdate_range("2024-01-01", periods=MIN_HISTORY_POINTS_FOR_FORECAST + 10)
    constant_prices = pd.Series([100.0] * len(idx), index=idx)
    result = arima_forecast(constant_prices, horizon=10, max_p=1, max_q=1)
    assert len(result["forecast"]) == 10
    assert not result["forecast"].isna().any()


# ---------------------------------------------------------------------------
# forecast_all_assets — shape contract every optimizer call relies on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", list(FORECAST_MODELS.keys()))
def test_forecast_all_assets_returns_one_column_per_ticker(model_name):
    prices = pd.DataFrame({
        "AAA": _price_series(80, seed=1).values,
        "BBB": _price_series(80, seed=2).values,
    }, index=_price_series(80, seed=1).index)
    forecast = forecast_all_assets(prices, horizon=10, model_name=model_name)
    assert list(forecast.columns) == ["AAA", "BBB"]
    assert len(forecast) == 10
    assert not forecast.isna().any().any()


# ---------------------------------------------------------------------------
# ml_regression_forecast — short-history/insufficient-data fallback, shape,
# the widening-band convention, and the model_type backend switch
# ---------------------------------------------------------------------------

def test_ml_regression_forecast_falls_back_to_naive_on_short_history():
    short_prices = _price_series(MIN_HISTORY_POINTS_FOR_FORECAST - 5)
    result = ml_regression_forecast(short_prices, horizon=10)
    expected = naive_forecast(short_prices, horizon=10)
    pd.testing.assert_series_equal(result["forecast"], expected["forecast"])


def test_ml_regression_forecast_falls_back_to_naive_with_too_few_windows_after_lookback():
    # Long enough to clear MIN_HISTORY_POINTS_FOR_FORECAST, but a large
    # `lookback` relative to that history leaves too few (n_rows < 20)
    # training windows — must degrade to naive, not train on a handful of rows.
    prices = _price_series(MIN_HISTORY_POINTS_FOR_FORECAST + 5)
    result = ml_regression_forecast(prices, horizon=10, lookback=MIN_HISTORY_POINTS_FOR_FORECAST - 10)
    expected = naive_forecast(prices, horizon=10)
    pd.testing.assert_series_equal(result["forecast"], expected["forecast"])


def test_ml_regression_forecast_runs_on_sufficient_history():
    prices = _price_series(150, seed=1)
    result = ml_regression_forecast(prices, horizon=15, model_type="gradient_boosting")
    assert len(result["forecast"]) == 15
    assert not result["forecast"].isna().any()
    assert (result["forecast"].index > prices.index[-1]).all()


def test_ml_regression_forecast_confidence_band_widens_with_horizon():
    prices = _price_series(150, seed=2)
    result = ml_regression_forecast(prices, horizon=20, model_type="gradient_boosting")
    band_width = (result["upper"] - result["lower"]).values
    assert band_width[-1] > band_width[0]
    assert (np.diff(band_width) >= -1e-9).all()  # small numerical slack


def test_ml_regression_forecast_lower_upper_bracket_the_point_forecast():
    prices = _price_series(150, seed=3)
    result = ml_regression_forecast(prices, horizon=10, model_type="gradient_boosting")
    assert (result["lower"] <= result["forecast"]).all()
    assert (result["forecast"] <= result["upper"]).all()


def test_ml_regression_forecast_random_forest_backend_runs_without_crashing():
    prices = _price_series(150, seed=4)
    result = ml_regression_forecast(prices, horizon=10, model_type="random_forest")
    assert len(result["forecast"]) == 10
    assert not result["forecast"].isna().any()


def test_ml_regression_forecast_auto_falls_back_to_gradient_boosting_without_xgboost(monkeypatch):
    # Simulate XGBoost not being installed: "auto" must silently pick
    # scikit-learn's GradientBoostingRegressor rather than raising.
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("simulated: xgboost not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    prices = _price_series(150, seed=5)
    result = ml_regression_forecast(prices, horizon=10, model_type="auto")
    assert len(result["forecast"]) == 10
    assert not result["forecast"].isna().any()


def test_ml_regression_forecast_falls_back_to_naive_when_fitting_raises(monkeypatch):
    import src.forecasting as forecasting_module

    prices = _price_series(150, seed=6)

    def _broken_regressor(*args, **kwargs):
        raise RuntimeError("simulated fit failure")

    monkeypatch.setattr(forecasting_module, "_get_gradient_boosted_regressor", _broken_regressor)
    result = ml_regression_forecast(prices, horizon=10)
    expected = naive_forecast(prices, horizon=10)
    pd.testing.assert_series_equal(result["forecast"], expected["forecast"])


# ---------------------------------------------------------------------------
# lstm_forecast — the fallback path, the "widening tube" property the fan
# chart depends on, and determinism
# ---------------------------------------------------------------------------

def test_lstm_forecast_falls_back_to_naive_on_short_history():
    short_prices = _price_series(MIN_HISTORY_POINTS_FOR_LSTM - 10)
    result = lstm_forecast(short_prices, horizon=10)
    expected = naive_forecast(short_prices, horizon=10)
    pd.testing.assert_series_equal(result["forecast"], expected["forecast"])


def test_lstm_forecast_runs_on_sufficient_history():
    prices = _price_series(150)
    result = lstm_forecast(prices, horizon=15, epochs=20)  # fewer epochs — this test only
    # needs a valid, non-degenerate fit, not a well-converged one; keeps the suite fast.
    assert len(result["forecast"]) == 15
    assert not result["forecast"].isna().any()
    assert (result["forecast"].index > prices.index[-1]).all()


def test_lstm_forecast_confidence_band_widens_with_horizon():
    # The whole point of the fan chart this was built for: forecasts further
    # out are less certain, and the band must visibly reflect that, not stay
    # a constant width or (worse) narrow.
    prices = _price_series(150, seed=3)
    result = lstm_forecast(prices, horizon=20, epochs=20)
    band_width = (result["upper"] - result["lower"]).values
    assert band_width[-1] > band_width[0]
    # Monotonic widening (sqrt(t) scaling) — not just wider at the end, but
    # actually increasing step over step, matching every other model here.
    assert (np.diff(band_width) >= -1e-9).all()  # small numerical slack


def test_lstm_forecast_lower_upper_bracket_the_point_forecast():
    prices = _price_series(150, seed=4)
    result = lstm_forecast(prices, horizon=10, epochs=20)
    assert (result["lower"] <= result["forecast"]).all()
    assert (result["forecast"] <= result["upper"]).all()


def test_lstm_forecast_is_deterministic_given_the_same_input():
    # torch.manual_seed(0) inside the function -- same series in, same
    # forecast out, every time. Matters for reproducible walk-forward runs.
    prices = _price_series(150, seed=5)
    result_a = lstm_forecast(prices, horizon=10, epochs=15)
    result_b = lstm_forecast(prices, horizon=10, epochs=15)
    pd.testing.assert_series_equal(result_a["forecast"], result_b["forecast"])


def test_lstm_forecast_falls_back_to_naive_when_training_raises(monkeypatch):
    import torch

    prices = _price_series(150, seed=6)

    def _broken_lstm(*args, **kwargs):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(torch.nn, "LSTM", _broken_lstm)
    result = lstm_forecast(prices, horizon=10)
    expected = naive_forecast(prices, horizon=10)
    pd.testing.assert_series_equal(result["forecast"], expected["forecast"])