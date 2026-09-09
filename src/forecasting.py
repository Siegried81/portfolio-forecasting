"""
Price forecasting models — statsmodels-based (NOT Kats: Kats has been effectively
unmaintained since 2021 and conflicts with modern pandas/numpy, which would burn
hours of a 2-day deadline on dependency resolution instead of on the actual
analysis). statsmodels is the industry-standard, well-maintained alternative and
is what most quant/finance teams actually reach for.

Six models, increasing sophistication, all returning the SAME shape so the
optimization layer doesn't need to know which one produced a forecast:

- `naive_forecast`   : random-walk-with-drift baseline. Always include this - if a
                        fancier model can't beat it, the fancier model isn't adding value.
- `ets_forecast`      : Exponential Smoothing (Holt's linear trend). Robust, few
                        assumptions, good default for noisy financial series.
- `theta_forecast`    : the Theta method (Assimakopoulos & Nikolopoulos, 2000) — the
                        best-performing simple method in the M3 forecasting competition,
                        and still a standard benchmark every fancier model is expected to
                        beat before it's trusted. Splits the series into a long-term linear
                        trend and a locally-smoothed short-term component, forecasts each
                        separately, then averages them. Cheap (closed-form + one SES fit,
                        no grid search), which makes it a useful middle point between ETS
                        and ARIMA on the speed/sophistication spectrum.
- `arima_forecast`    : ARIMA with a small grid search over (p,d,q) by AIC. Can
                        capture autocorrelation structure the other two miss, at
                        the cost of being slower and more prone to overfitting on
                        short series.
- `ml_regression_forecast` : gradient-boosted (or bagged) tree regression on
                        lagged returns — XGBoost if installed, else scikit-learn's
                        GradientBoostingRegressor or RandomForestRegressor. The
                        first genuinely ML-based (tabular, not deep-learning)
                        member of this set; trains in milliseconds per asset.
- `lstm_forecast`     : a small, from-scratch-trained recurrent network per asset — the
                        heaviest, deep-learning member of this set. See its own
                        docstring for the honest scope/speed trade-off against the
                        other five.

IMPORTANT, and stated explicitly in the UI/README: stock price forecasting from
price history alone has very weak predictive power (near random-walk territory
per the efficient market hypothesis). These models exist to demonstrate the
forecast -> optimize -> compare methodology the brief asks for, not because
short-horizon price forecasts should be trusted for real allocation decisions.
"""
from __future__ import annotations

import concurrent.futures
import warnings
from typing import cast, Any


import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

from src.config import MIN_HISTORY_POINTS_FOR_FORECAST, MIN_HISTORY_POINTS_FOR_LSTM

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning)

ForecastResult = dict[str, pd.Series]  # {"forecast": Series, "lower": Series, "upper": Series}


def _future_index(history: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """Business-day index continuing directly after the last observed date."""
    return pd.bdate_range(start=history.index[-1], periods=horizon + 1, freq="B")[1:]


def naive_forecast(prices: pd.Series, horizon: int) -> ForecastResult:
    """Random walk with drift: tomorrow's price = last price + average historical daily change."""
    daily_changes = prices.diff().dropna()
    drift = daily_changes.mean()
    last_price = prices.iloc[-1]
    steps = np.arange(1, horizon + 1)
    point_forecast = last_price + drift * steps

    # Widening confidence band ~ sqrt(t) x historical daily std dev (standard random-walk property)
    std = daily_changes.std(ddof=1)
    band = std * np.sqrt(steps)
    idx = _future_index(prices, horizon)
    return {
        "forecast": pd.Series(point_forecast, index=idx, name="naive"),
        "lower": pd.Series(point_forecast - 1.96 * band, index=idx),
        "upper": pd.Series(point_forecast + 1.96 * band, index=idx),
    }


def ets_forecast(prices: pd.Series, horizon: int) -> ForecastResult:
    """Holt's linear-trend Exponential Smoothing."""
    if len(prices) < MIN_HISTORY_POINTS_FOR_FORECAST:
        return naive_forecast(prices, horizon)
    # Fitting directly on `prices` (a real trading-day
    # DatetimeIndex, irregular due to weekends/holidays) makes statsmodels
    # raise "FutureWarning: No supported index is available" — it can't infer
    # a fixed frequency from actual market dates, and its own message warns a
    # future release will RAISE instead of warn here. Harmless today because
    # this app never calls the FITTED model's own date-aware forecast() —
    # `_future_index()` below reconstructs the real output dates
    # independently from `prices` itself, untouched. Fitting on a plain
    # positional index instead sidesteps the ambiguity at the root, rather
    # than just suppressing the symptom (and rather than risking a future
    # statsmodels upgrade turning this into a hard crash).
    fit_values = prices.reset_index(drop=True)
    model = ExponentialSmoothing(fit_values, trend="add", damped_trend=True, initialization_method="estimated")
    fitted = model.fit(optimized=True)
    point_forecast = fitted.forecast(horizon)

    resid_std = fitted.resid.std(ddof=1)
    steps = np.arange(1, horizon + 1)
    band = resid_std * np.sqrt(steps)
    idx = _future_index(prices, horizon)  # prices (real dates) — unaffected by the reset above
    return {
        "forecast": pd.Series(point_forecast.values, index=idx, name="ets"),
        "lower": pd.Series(point_forecast.values - 1.96 * band, index=idx),
        "upper": pd.Series(point_forecast.values + 1.96 * band, index=idx),
    }


def theta_forecast(prices: pd.Series, horizon: int) -> ForecastResult:
    """
    The Theta method (Assimakopoulos & Nikolopoulos, 2000): decompose the
    series into a long-term linear trend and a short-term residual
    component, forecast each with the tool suited to it, then recombine.

    Implemented via the algebraically equivalent, more direct form of the
    original two-"theta-line" decomposition: fit an OLS linear trend, take
    SES (Simple Exponential Smoothing) of the DETRENDED residuals — which
    have no linear trend left by construction, exactly what SES needs to
    behave well — and add the two forecasts back together. Applying SES to
    the raw series directly (skipping the detrend step) would systematically
    under-forecast any real trend, since SES has no trend term of its own;
    detrending first is what lets this method actually track a genuine trend
    while still reacting to short-term deviations from it.

    Deliberately NOT the generalised multi-theta-line variant (Hyndman &
    Billah's later extension, or the automatic-weighting "optimised Theta"
    of Fiorucci et al., 2016) — this equal-weight, two-component version has
    no extra hyperparameters to tune and was still the strongest overall
    performer in the original M3 competition despite that simplicity.
    """
    if len(prices) < MIN_HISTORY_POINTS_FOR_FORECAST:
        return naive_forecast(prices, horizon)

    values = prices.to_numpy()
    n = len(values)
    t = np.arange(n)

    # Long-term component: OLS linear trend, extrapolated directly.
    slope, intercept = np.polyfit(t, values, 1)
    trend = intercept + slope * t
    future_t = np.arange(n, n + horizon)
    trend_forecast = intercept + slope * future_t

    # Short-term component: SES on the detrended residuals (mean ~0, no
    # linear trend left by construction — exactly the input SES handles
    # well). A degenerate residual series (e.g. a perfectly linear input,
    # all-zero residuals) can make SES fail to converge on some inputs —
    # fall back to naive rather than crash the whole forecast.
    residuals = values - trend
    try:
        ses_fit = SimpleExpSmoothing(residuals, initialization_method="estimated").fit()
        residual_forecast = ses_fit.forecast(horizon)
    except Exception:
        return naive_forecast(prices, horizon)

    point_forecast = trend_forecast + residual_forecast

    # Confidence band: same sqrt(t)-widening convention as every other model
    # here, scaled by the trend line's own residual std dev.
    resid_std = float(np.std(residuals, ddof=1))
    steps = np.arange(1, horizon + 1)
    band = resid_std * np.sqrt(steps)

    idx = _future_index(prices, horizon)
    return {
        "forecast": pd.Series(point_forecast, index=idx, name="theta"),
        "lower": pd.Series(point_forecast - 1.96 * band, index=idx),
        "upper": pd.Series(point_forecast + 1.96 * band, index=idx),
    }


def arima_forecast(prices: pd.Series, horizon: int, max_p: int = 3, max_q: int = 3) -> ForecastResult:
    """
    ARIMA(p,1,q) with a small AIC grid search. d=1 is fixed (first-differencing)
    since price levels are non-stationary by construction — searching d as well
    would mostly just re-discover d=1 at extra compute cost for no real benefit here.
    """
    if len(prices) < MIN_HISTORY_POINTS_FOR_FORECAST:
        return naive_forecast(prices, horizon)

    # Same index-ambiguity fix as ets_forecast above — see that function's
    # comment for the full rationale. `prices` itself stays untouched below,
    # for `_future_index()`'s real output dates.
    fit_values = prices.reset_index(drop=True)

    best_aic, best_fit = np.inf, None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                fit = ARIMA(fit_values, order=(p, 1, q)).fit()
                if fit.aic < best_aic:
                    best_aic, best_fit = fit.aic, fit
            except Exception:
                continue  # non-convergent order — skip, don't crash the whole forecast

    if best_fit is None:
        return naive_forecast(prices, horizon)

    result = best_fit.get_forecast(steps=horizon)
    idx = _future_index(prices, horizon)
    conf_int = result.conf_int(alpha=0.05)
    return {
        "forecast": pd.Series(result.predicted_mean.values, index=idx, name="arima"),
        "lower": pd.Series(conf_int.iloc[:, 0].values, index=idx),
        "upper": pd.Series(conf_int.iloc[:, 1].values, index=idx),
    }


def _get_gradient_boosted_regressor(model_type: str, n_estimators: int) -> Any:    
    """
    Pick the tree-ensemble regressor backend for `ml_regression_forecast`.

    - `"xgboost"`   : gradient-boosted trees via the XGBoost library — usually
                       the strongest of the three on tabular features like the
                       lagged returns used here, at the cost of an extra
                       dependency.
    - `"gradient_boosting"` : scikit-learn's own GradientBoostingRegressor —
                       the always-available fallback, since scikit-learn is
                       already a hard dependency of this project (rag.py's
                       TF-IDF, factor_models.py's PCA), unlike XGBoost.
    - `"random_forest"`     : scikit-learn's RandomForestRegressor — bagged
                       rather than boosted trees; less prone to overfitting a
                       short, noisy financial series than boosting can be,
                       at some cost in raw accuracy on cleaner signals.
    - `"auto"` (default)    : XGBoost if it's installed, else
                       GradientBoostingRegressor — same "optional heavier
                       dependency, always-available fallback" shape as
                       PyTorch's LSTM relative to this module's other models,
                       just one tier lighter: this model is itself already
                       the lightweight fallback if a caller only wants
                       scikit-learn installed.
    """
    if model_type == "auto":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=n_estimators, max_depth=3, learning_rate=0.05, verbosity=0)
        except ImportError:
            model_type = "gradient_boosting"

    if model_type == "xgboost":
        from xgboost import XGBRegressor  # raises ImportError if explicitly requested but missing
        return XGBRegressor(n_estimators=n_estimators, max_depth=3, learning_rate=0.05, verbosity=0)
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=n_estimators, max_depth=5, random_state=0)
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=n_estimators, max_depth=3, learning_rate=0.05, random_state=0)


def _return_features(window: np.ndarray) -> list[float]:
    """One feature row from a window of `lookback` consecutive daily returns:
    the lags themselves plus the window's own mean/std — a small,
    interpretable feature set derived purely from price history, consistent
    with every other model here (no external data feeds into a forecast)."""
    return list(window) + [float(window.mean()), float(window.std(ddof=0) if len(window) > 1 else 0.0)]


def ml_regression_forecast(
    prices: pd.Series, horizon: int, lookback: int = 10, n_estimators: int = 200, model_type: str = "auto",
) -> ForecastResult:
    """
    Gradient-boosted (or bagged, via `model_type="random_forest"`) tree
    regression on lagged returns — a lightweight, tabular-data ML alternative
    to `lstm_forecast`'s sequence model, trained in milliseconds rather than
    seconds per asset.

    Feature set: `lookback` lagged one-day returns plus their own rolling
    mean/std, predicting the NEXT day's return (not price directly — returns
    are closer to stationary, the same reasoning `arima_forecast` fixes d=1
    for). Forecasts one step at a time, recursively feeding each prediction
    back in as the newest lag before predicting the next — the standard way
    to extend a single-step regressor to a multi-step horizon without
    training `horizon` separate models. Recursive forecasts compound their
    own error at longer horizons, same honest caveat as `lstm_forecast`'s.

    See `_get_gradient_boosted_regressor` for the `model_type` options
    ("auto" | "xgboost" | "gradient_boosting" | "random_forest").

    Falls back to `naive_forecast` on short history, too little training
    data after windowing, or if fitting/predicting raises for any reason —
    same fails-soft contract as every other model in this module.
    """
    if len(prices) < MIN_HISTORY_POINTS_FOR_FORECAST:
        return naive_forecast(prices, horizon)

    returns = prices.pct_change().dropna().to_numpy()
    n_rows = len(returns) - lookback
    if n_rows < 20:  # not enough windows to train a meaningful model
        return naive_forecast(prices, horizon)

    X = np.array([_return_features(returns[i:i + lookback]) for i in range(n_rows)])
    y = returns[lookback:]

    try:
        model = _get_gradient_boosted_regressor(model_type, n_estimators)
        model.fit(X, y)
        in_sample_pred = model.predict(X)
    except Exception:
        return naive_forecast(prices, horizon)

    # Recursive multi-step forecast: predict one return, append it to the
    # rolling window, predict the next.
    window = list(returns[-lookback:])
    predicted_returns = []
    try:
        for _ in range(horizon):
            feature_row = np.array([_return_features(np.array(window[-lookback:]))])
            next_return = float(model.predict(feature_row)[0])
            predicted_returns.append(next_return)
            window.append(next_return)
    except Exception:
        return naive_forecast(prices, horizon)

    last_price = float(prices.iloc[-1])
    point_forecast = last_price * np.cumprod(1.0 + np.array(predicted_returns))

    # Confidence band: same sqrt(t)-widening convention as every other model
    # here, scaled by the model's own IN-SAMPLE residual std dev (a rough,
    # optimistic proxy — a true out-of-sample residual would need a proper
    # walk-forward fit, which is exactly what backtesting.py's walk-forward
    # validation checks separately) and by the price level, since the model
    # forecasts RETURNS, not price levels directly.
    resid_std = float(np.std(y - in_sample_pred, ddof=1))
    steps = np.arange(1, horizon + 1)
    band = point_forecast * resid_std * np.sqrt(steps)

    idx = _future_index(prices, horizon)
    return {
        "forecast": pd.Series(point_forecast, index=idx, name="ml_regression"),
        "lower": pd.Series(point_forecast - 1.96 * band, index=idx),
        "upper": pd.Series(point_forecast + 1.96 * band, index=idx),
    }


def lstm_forecast(
    prices: pd.Series, horizon: int, lookback: int = 20, epochs: int = 50, hidden_size: int = 32,
) -> ForecastResult:
    """
    A small univariate LSTM (Long Short-Term Memory recurrent neural network),
    trained from scratch on THIS asset's own historical returns at forecast
    time — there's no pretrained "universal" price-forecasting LSTM to load,
    so (like GARCH's per-asset volatility fit in volatility_forecasting.py)
    this trains a fresh, tiny model per call rather than reusing weights
    across assets or calls.

    Why returns, not raw prices: prices are non-stationary (see ARIMA's own
    d=1 differencing above) and sit on wildly different scales across assets
    (a $50 stock vs. a $2000 one) — training directly on price levels would
    make the network's job harder for no benefit. Training on returns keeps
    the input roughly stationary and scale-comparable, then the forecasted
    returns are compounded back into a price path.

    Architecture, deliberately small: one LSTM layer (`hidden_size` units) +
    a linear output layer, trained with Adam/MSE for `epochs` passes over
    sliding windows of length `lookback`. This is NOT a competitive
    forecasting architecture (no attention, no multi-layer stacking, no
    hyperparameter search) — it exists to demonstrate the technique
    alongside naive/ETS/ARIMA, same "methodology over model sophistication"
    framing as the rest of this module's docstring. Multi-step forecasting
    is done RECURSIVELY: predict one step, feed that prediction back in as
    the newest point in the window, repeat `horizon` times — the standard
    approach for an RNN that was only ever trained to predict one step ahead.

    Confidence band: same convention as every other model here
    (`std(training residuals) * sqrt(steps)`, see naive_forecast) rather than
    a network-specific uncertainty estimate (e.g. MC dropout) — keeps the
    band directly comparable across every model in the UI's fan chart,
    at the cost of not capturing the network's own genuine predictive
    uncertainty as precisely as a purpose-built method would.

    SPEED CAVEAT, stated honestly (same spirit as ARIMA's grid search and
    GARCH's per-asset fit elsewhere in this app): training a fresh network
    per asset, per walk-forward window, is meaningfully slower than
    naive/ETS and slower than ARIMA too — see scripts/benchmark_walk_forward.py
    for actual numbers before selecting this for a wide universe or many
    walk-forward windows.

    Falls back to `naive_forecast` if there's too little history for a
    stable fit (`MIN_HISTORY_POINTS_FOR_LSTM` — higher than the other
    models' threshold, since a `lookback`-sized sliding window needs enough
    of them to actually learn from) or if training produces a non-finite
    loss or raises (rare, but the same "don't propagate a broken fit"
    philosophy as ARIMA's grid search and GARCH's convergence check).
    """
    if len(prices) < MIN_HISTORY_POINTS_FOR_LSTM:
        return naive_forecast(prices, horizon)

    import torch  # imported lazily — optional, heavy dependency, only needed for this one model
    from torch import nn

    returns = prices.pct_change().dropna().values.astype("float32")
    if len(returns) < lookback + 10:  # not enough sliding windows to train on meaningfully
        return naive_forecast(prices, horizon)

    # Scale returns to roughly unit variance — neural nets train more reliably
    # on inputs of a sane magnitude than on raw daily returns (typically ~0.01).
    scale = float(returns.std()) or 1.0
    scaled = returns / scale

    X_list = [scaled[i:i + lookback] for i in range(len(scaled) - lookback)]
    y_list = [scaled[i + lookback] for i in range(len(scaled) - lookback)]
    X = torch.tensor(np.array(X_list)).unsqueeze(-1)  # (n_samples, lookback, 1)
    y = torch.tensor(np.array(y_list)).unsqueeze(-1)  # (n_samples, 1)

    class _TinyLSTM(nn.Module):  # local — this app never reuses the architecture outside this function
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            # torch's own type stubs return Any here — same reasoning as the explicit
            # float(...)/str(...) wraps elsewhere in this codebase (metrics.py,
            # llm_client.py) for library calls whose stubs don't narrow past Any.
            return cast("torch.Tensor", self.head(out[:, -1, :]))  # last timestep -> one-step-ahead prediction

    try:
        torch.manual_seed(0)  # deterministic training — the same asset/window always trains to
        # the same fit, matching this app's general preference for reproducible results over
        # run-to-run noise. Model construction is inside this try block too, not just the
        # training loop below — a broken/incompatible torch install should degrade the same
        # way a training-time failure does, not crash the whole forecast.
        model = _TinyLSTM(hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            prediction = model(X)
            loss = loss_fn(prediction, y)
            if not torch.isfinite(loss):
                return naive_forecast(prices, horizon)  # diverged — don't trust this fit
            loss.backward()
            optimizer.step()
    except Exception:
        return naive_forecast(prices, horizon)  # any training failure — same "skip, don't crash" as ARIMA

    model.eval()
    window = scaled.tolist()
    forecast_returns_scaled = []
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor([window[-lookback:]], dtype=torch.float32).unsqueeze(-1)
            next_scaled = model(x).item()
            forecast_returns_scaled.append(next_scaled)
            window.append(next_scaled)  # recursive: this step's prediction feeds the next step's input

        train_resid = (model(X) - y).numpy().flatten() * scale

    forecast_returns = np.array(forecast_returns_scaled) * scale

    # Compound the forecasted returns into a price path, starting from the
    # last observed price — same convention naive_forecast's drift uses.
    last_price = prices.iloc[-1]
    point_forecast = last_price * np.cumprod(1 + forecast_returns)

    resid_std = float(train_resid.std(ddof=1)) if len(train_resid) > 1 else 0.0
    steps = np.arange(1, horizon + 1)
    # Scale the return-space residual std by the forecasted PRICE level at each
    # step, since a fixed fractional return uncertainty translates to a growing
    # absolute price uncertainty as the price path compounds — same sqrt(t)
    # convention as every other model here (see naive_forecast), just applied
    # in price space instead of directly to price differences.
    band = point_forecast * resid_std * np.sqrt(steps)

    idx = _future_index(prices, horizon)
    return {
        "forecast": pd.Series(point_forecast, index=idx, name="lstm"),
        "lower": pd.Series(point_forecast - 1.96 * band, index=idx),
        "upper": pd.Series(point_forecast + 1.96 * band, index=idx),
    }


FORECAST_MODELS = {
    "Naive (random walk)": naive_forecast,
    "ETS (Holt linear trend)": ets_forecast,
    "Theta method": theta_forecast,
    "ARIMA (auto order)": arima_forecast,
    "ML regression (gradient boosting)": ml_regression_forecast,
    "LSTM (recurrent neural net)": lstm_forecast,
}


def forecast_all_assets(
    prices: pd.DataFrame,
    horizon: int,
    model_name: str = "ETS (Holt linear trend)",
) -> pd.DataFrame:
    """
    Run the chosen model independently per asset (no cross-asset correlation in
    the forecast itself — that's handled downstream by the covariance matrix used
    in optimization) and return a single DataFrame of forecasted prices.

    PARALLELIZED across tickers with a thread pool — this was the
    single biggest speed bottleneck flagged in the README ("ARIMA across many
    windows/assets is noticeably slower"), and it compounds badly because this
    exact function is also called once PER walk-forward window. Threads, not
    processes: ARIMA/ETS fitting is dominated by numpy/scipy linear algebra,
    which releases the GIL during BLAS calls, so threads give a real wall-clock
    speedup here without the pickling and Streamlit-context fragility that
    spinning subprocesses from inside a Streamlit callback would introduce.
    `max_workers` is capped at 8 to avoid oversubscribing a small container
    (e.g. Render's free tier) when a large universe is selected.
    """
    model_fn = FORECAST_MODELS[model_name]
    tickers = list(prices.columns)

    def _forecast_one(ticker: str) -> pd.Series:
        series = prices[ticker].dropna()
        return model_fn(series, horizon)["forecast"]

    max_workers = min(8, len(tickers)) or 1
    forecasts: dict[str, pd.Series] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map preserves input order, so the resulting dict — and the
        # DataFrame built from it — always has columns in the same order as
        # `prices.columns`, regardless of which thread finishes first.
        for ticker, forecast in zip(tickers, executor.map(_forecast_one, tickers)):
            forecasts[ticker] = forecast
    return pd.DataFrame(forecasts)