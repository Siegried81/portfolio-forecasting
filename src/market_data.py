"""
Market data acquisition via yfinance (free, no API key required), with automatic
retry and a Twelve Data fallback for when Yahoo Finance itself is unreachable.

Why the fallback exists: as of late 2026, Yahoo Finance's anti-bot layer (the
cookie/crumb handshake yfinance depends on) has become noticeably unreliable —
this is a widely reported, ongoing issue across the yfinance community, not
specific to any one network or account. A finance tool that goes dark whenever
Yahoo has a bad day is a real reliability gap, so a second source is used as a
fallback rather than just failing.

Twelve Data (not Stooq) was chosen for the fallback after live testing: Stooq's
CSV export now runs a JavaScript proof-of-work anti-bot challenge (confirmed by
inspecting its raw response — a `crypto.subtle.digest` SHA-256 puzzle that a
plain HTTP client cannot solve), so a `requests.get()`-based fallback against it
is a dead end, not a fix. Twelve Data is a genuine key-based REST API rather
than a scraped endpoint, which sidesteps that entire arms race. Free tier: 800
requests/day, 8/minute — and it accepts multiple tickers in ONE call, which
matters for staying inside that budget with a 5+ ticker portfolio.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
import tempfile
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from src.cache import cached
from src.config import LLM_SETTINGS, VIX_TICKER

logger = logging.getLogger(__name__)


def _configure_yfinance_cache() -> None:
    """
    Redirect yfinance's internal SQLite cache (cookies + per-ticker timezone
    lookups) to the system temp directory, instead of yfinance's own default
    location.

    Why: yfinance's default cache path can end up somewhere SQLite can't
    reliably lock — most commonly reported on WSL, when the shell's `$HOME`
    (or the process's working directory, which some cache-path resolution
    logic falls back to) resolves onto a Windows-mounted drive (`/mnt/c`,
    `/mnt/d`, DrvFs/9p) rather than the WSL distro's own native filesystem.
    DrvFs doesn't support the POSIX file locks SQLite needs, so yfinance then
    raises `OperationalError: unable to open database file` on every ticker
    that hits the cache — confirmed live: `'^VIX'` failing this way while the
    main equity tickers still succeeded via a different fetch path (the
    circuit-breaker/Twelve-Data fallback doesn't touch this cache at all).
    The system temp directory is always on the native filesystem and always
    writable regardless of where this repo itself happens to be checked out,
    sidestepping the whole class of problem rather than depending on the
    user's `$HOME` being configured a particular way.

    Best-effort: wrapped in try/except because this calls a yfinance internal
    API (`set_tz_cache_location`) that isn't guaranteed stable across
    versions — if it's ever renamed/removed upstream, this must degrade to
    "use yfinance's own default location" rather than crash the app at
    import time over a caching optimisation.
    """
    try:
        cache_dir = os.path.join(tempfile.gettempdir(), "py-yfinance-cache")
        os.makedirs(cache_dir, exist_ok=True)
        yf.set_tz_cache_location(cache_dir)
    except Exception as exc:
        logger.warning("Could not redirect yfinance's cache location, using its own default: %s", exc)


_configure_yfinance_cache()  # run once at import time, before any yf.download() call below


def _redact_api_key(text: str) -> str:
    """
    Strip any `apikey=...` query param value out of a string before it's ever
    logged or, critically, raised as an exception message that Streamlit's
    `st.error()` displays verbatim on screen. `requests`' HTTPError includes the
    full request URL (query string and all) in its default __str__ — without
    this, a failed Twelve Data call leaks the API key straight into the UI.
    """
    return re.sub(r"apikey=[^&\s]+", "apikey=***REDACTED***", text)

TWELVEDATA_URL = "https://api.twelvedata.com/time_series"
TWELVEDATA_MAX_ATTEMPTS = 3
TWELVEDATA_BACKOFF_SECONDS = 20.0  # doubles each retry: 20s, 40s — 429 free-tier limits reset per-minute

TIINGO_URL = "https://api.tiingo.com/tiingo/daily"
# 500 req/HOUR on the free tier — far more generous than Twelve Data (800/day)
# or Alpha Vantage (25/day), and Tiingo's own response already includes
# `adjClose` directly (no separate adjustment step needed, unlike a raw
# "close" field). Positioned as the THIRD tier (before Twelve Data), ahead
# of the two stingier fallbacks, precisely because of that generous quota.
TIINGO_MAX_ATTEMPTS = 3
TIINGO_BACKOFF_SECONDS = 5.0

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
# The free tier's actual published limit is 25 requests/DAY (not just a
# per-minute rate) — the stingiest constraint of this app's four
# market-data sources by a wide margin, which is exactly why this sits as
# the last-resort, fifth tier rather than an earlier one. One ticker per
# call (no batch endpoint, unlike Twelve Data), so even a handful of
# tickers can burn a meaningful fraction of a day's quota in one fetch.
ALPHA_VANTAGE_SECONDS_BETWEEN_CALLS = 13.0  # keeps this app's own calls under
# the vendor's separate per-minute throttle within whatever's left of the
# 25/day budget — does not, and cannot, raise the daily ceiling itself.
YFINANCE_MAX_ATTEMPTS = 3
YFINANCE_BACKOFF_SECONDS = 2.0  # doubles each retry: 2s, 4s

# Circuit breaker: once BOTH Yahoo paths (yfinance library + direct API) fail in
# the same call, skip re-attempting Yahoo entirely for this long — avoids paying
# the full multi-second retry cost again on every Streamlit rerun (every widget
# interaction) while Yahoo is known to be down for this process. Module-level
# (not per-request), on purpose: the whole point is state that survives across
# separate `fetch_adjusted_close` calls within the same running process.
YAHOO_CIRCUIT_BREAKER_SECONDS = 180.0  # 3 minutes — long enough to skip a burst of
# rapid interactions, short enough to retry Yahoo again well within one debugging session
_yahoo_down_until: float = 0.0


class MarketDataError(RuntimeError):
    """Raised when price data cannot be retrieved from ANY source."""


class TwelveDataPlanRestricted(RuntimeError):
    """Raised when Twelve Data returns a 403 indicating an endpoint (typically
    /statistics for non-demo tickers) requires a paid plan — distinct from a
    transient failure, so callers can surface the real reason to the user."""


def _download_yfinance(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    One yfinance attempt with retry-with-backoff for transient failures (mainly
    429 rate-limiting on Yahoo's cookie/crumb endpoint). Retrying helps with a
    momentary block; it does nothing for a sustained one, which is what the
    Twelve Data fallback in `fetch_adjusted_close` is for.
    """
    last_error: Exception | None = None
    for attempt in range(1, YFINANCE_MAX_ATTEMPTS + 1):
        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end + dt.timedelta(days=1),  # yfinance's `end` is exclusive
                auto_adjust=True,  # returns already-adjusted "Close" (splits + dividends)
                progress=False,
                group_by="ticker",
            )
            if not raw.empty:
                return raw
            last_error = MarketDataError("Empty response")
        except Exception as exc:  # yfinance raises a mix of requests/JSON/custom errors
            last_error = exc
            logger.warning("yfinance attempt %d/%d failed: %s", attempt, YFINANCE_MAX_ATTEMPTS, exc)

        if attempt < YFINANCE_MAX_ATTEMPTS:
            time.sleep(YFINANCE_BACKOFF_SECONDS * attempt)  # 2s, then 4s

    raise MarketDataError(f"Yahoo Finance unreachable after {YFINANCE_MAX_ATTEMPTS} attempts: {last_error}")


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def _download_yahoo_direct(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Direct call to Yahoo Finance's own REST API (`v8/finance/chart`) — bypassing
    the `yfinance` library entirely. Added per the brief's literal wording ("the
    Yahoo Finance API through the yfinance package"): this is what "the Yahoo
    Finance API" actually is underneath the library.

    Honest expectation, stated plainly rather than left implicit: this endpoint
    sits behind the SAME cookie/crumb anti-bot layer that already blocks the
    `yfinance` library above (see `_download_yfinance`'s docstring) — a raw
    unauthenticated request here is very likely to fail identically. It's kept
    as a single, fast, no-retry attempt specifically because it's cheap insurance
    (Yahoo's anti-bot posture does fluctuate) with no cost when it doesn't pay
    off, not because it's expected to reliably succeed where the library fails.
    """
    period1 = int(dt.datetime.combine(start, dt.time.min).timestamp())
    period2 = int(dt.datetime.combine(end + dt.timedelta(days=1), dt.time.min).timestamp())
    headers = {"User-Agent": "Mozilla/5.0 (compatible; portfolio-forecasting/1.0)"}

    columns = {}
    for ticker in tickers:
        try:
            response = requests.get(
                YAHOO_CHART_URL.format(ticker=ticker),
                params={"period1": period1, "period2": period2, "interval": "1d", "events": "div,splits"},
                headers=headers, timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Direct Yahoo API failed for %s: %s", ticker, exc)
            continue

        result = (payload.get("chart") or {}).get("result")
        if not result:
            continue
        result = result[0]
        timestamps = result.get("timestamp")
        closes = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
        if not timestamps or not closes:
            continue

        series = pd.Series(closes, index=pd.to_datetime(timestamps, unit="s").normalize(), name=ticker)
        columns[ticker] = series.dropna()

    if not columns:
        raise MarketDataError("Direct Yahoo Finance API also returned no data for any ticker.")
    return pd.concat(columns, axis=1)


def _download_twelvedata(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Fallback source: Twelve Data's time_series endpoint, one batched call for
    every ticker (comma-separated `symbol` param) — deliberately NOT one call
    per ticker, to conserve the free tier's 800/day, 8/min budget.

    Response shape differs by ticker count: a single symbol returns
    {"meta": ..., "values": [...]} directly; multiple symbols return a dict
    KEYED by symbol, each holding that same shape — both are handled here.
    """
    if not LLM_SETTINGS.twelvedata_api_key:
        raise MarketDataError("No TWELVEDATA_API_KEY configured — cannot use the fallback source.")

    params = {
        "symbol": ",".join(tickers),
        "interval": "1day",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "apikey": LLM_SETTINGS.twelvedata_api_key,
        "order": "ASC",
    }
    try:
        response = requests.get(TWELVEDATA_URL, params=params, timeout=15)
        payload = None
        for attempt in range(1, TWELVEDATA_MAX_ATTEMPTS + 1):
            if response.status_code == 429:
                if attempt == TWELVEDATA_MAX_ATTEMPTS:
                    raise MarketDataError(
                        "Twelve Data rate limit hit (free tier: 8 req/min, 800/day). "
                        "Wait ~60s and retry — this is usually transient, not a quota exhaustion."
                    )
                logger.warning("Twelve Data 429, retrying in %.0fs (attempt %d/%d)", TWELVEDATA_BACKOFF_SECONDS * attempt, attempt, TWELVEDATA_MAX_ATTEMPTS)
                time.sleep(TWELVEDATA_BACKOFF_SECONDS * attempt)
                response = requests.get(TWELVEDATA_URL, params=params, timeout=15)
                continue
            response.raise_for_status()
            payload = response.json()
            break
    except (requests.RequestException, ValueError) as exc:
        raise MarketDataError(f"Twelve Data request failed: {_redact_api_key(str(exc))}") from exc

    if payload is None:  # unreachable in practice (the loop above always raises or
        # assigns), but makes the invariant explicit for both mypy and future edits
        raise MarketDataError("Twelve Data request failed: no response received.")

    if payload.get("status") == "error" or "code" in payload and "message" in payload and "values" not in payload:
        raise MarketDataError(f"Twelve Data error: {payload.get('message', payload)}")

    # Normalise both response shapes into {ticker: {"values": [...]}}.
    # Twelve Data can drop a bad/delisted symbol from a multi-ticker batch
    # response entirely, instead of keeping it as an error-tagged key — a
    # check like `all(t in payload for t in tickers)` would then fail for
    # the WHOLE batch and mis-parse every ticker, not just the missing one.
    # Detect the shape directly instead: a single-ticker response has
    # "values"/"meta" at the TOP level; a multi-ticker response is a
    # dict keyed by symbol, each holding that shape one level down.
    per_ticker: dict[str, dict[str, Any]] = (
        {tickers[0]: payload} if ("values" in payload or "meta" in payload) else payload
    )

    columns = {}
    for ticker in tickers:
        entry = per_ticker.get(ticker)
        values = entry.get("values") if entry else None
        if not values:
            logger.warning("Twelve Data returned no values for %s", ticker)
            continue
        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["close"] = pd.to_numeric(df["close"])
        columns[ticker] = df.set_index("datetime")["close"].sort_index()

    if not columns:
        raise MarketDataError("Twelve Data fallback also returned no data for any ticker.")
    return pd.concat(columns, axis=1)


def _download_alpha_vantage(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Fourth-tier fallback: Alpha Vantage's `TIME_SERIES_DAILY` endpoint. ONE
    call per ticker (Alpha Vantage has no batched multi-symbol endpoint,
    unlike Twelve Data above) — paced by `ALPHA_VANTAGE_SECONDS_BETWEEN_CALLS`
    between calls to respect the vendor's per-minute throttle within the
    free tier's real constraint: 25 requests per DAY, the tightest budget
    of this app's three market-data sources by far — even a handful of
    tickers can burn a meaningful fraction of a day's quota in one fetch. That trade-off is exactly why this sits as the
    LAST resort (after yfinance, direct Yahoo, AND Twelve Data), not an
    earlier one: a rarely-needed, slow-but-working fourth option is worth
    more here than a faster one that adds nothing when the first three
    already cover the common case.

    `outputsize="full"` (the entire available history, not just the last
    100 days) is requested unconditionally — the alternative would be
    picking outputsize based on how far `start` reaches back, which adds
    complexity for a tier that's already the least-used path in this
    cascade.
    """
    if not LLM_SETTINGS.alpha_vantage_api_key:
        raise MarketDataError("No ALPHA_VANTAGE_API_KEY configured — cannot use this fallback source.")

    columns: dict[str, pd.Series] = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(ALPHA_VANTAGE_SECONDS_BETWEEN_CALLS)
        params = {
            "function": "TIME_SERIES_DAILY", "symbol": ticker,
            "outputsize": "full", "apikey": LLM_SETTINGS.alpha_vantage_api_key,
        }
        try:
            response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Alpha Vantage request failed for %s: %s", ticker, _redact_api_key(str(exc)))
            continue

        # Alpha Vantage reports errors/rate-limit notices as plain-text
        # fields in an otherwise-200 response, not an HTTP error status —
        # "Error Message" (bad symbol) and "Note"/"Information" (rate limit
        # or a bad/demo key) are its own documented conventions for this.
        if "Error Message" in payload or "Note" in payload or "Information" in payload:
            logger.warning(
                "Alpha Vantage returned no data for %s: %s", ticker,
                payload.get("Error Message") or payload.get("Note") or payload.get("Information"),
            )
            continue

        series = payload.get("Time Series (Daily)")
        if not series:
            continue
        df = pd.DataFrame.from_dict(series, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        if df.empty:
            continue
        columns[ticker] = pd.to_numeric(df["4. close"]).sort_index()

    if not columns:
        raise MarketDataError("Alpha Vantage fallback also returned no data for any ticker.")
    return pd.concat(columns, axis=1)


# Stays on @st.cache_data (not @cached from src/cache.py): returns a
# DataFrame, which cached()'s JSON-based Redis path deliberately doesn't
# support — see that module's docstring.
def _download_tiingo(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Third-tier fallback: Tiingo's `/tiingo/daily/{ticker}/prices` endpoint.
    ONE call per ticker (no batch endpoint), but the free tier's 500
    req/hour ceiling makes that a non-issue for any realistic ticker count
    — a meaningfully looser budget than Twelve Data or Alpha Vantage.
    Uses `adjClose` directly (Tiingo pre-computes the dividend/split
    adjustment; other tiers here derive it differently or use a raw close).
    """
    if not LLM_SETTINGS.tiingo_api_key:
        raise MarketDataError("No TIINGO_API_KEY configured — cannot use this fallback source.")

    headers = {"Authorization": f"Token {LLM_SETTINGS.tiingo_api_key}"}
    columns: dict[str, pd.Series] = {}
    for ticker in tickers:
        params = {"startDate": start.isoformat(), "endDate": end.isoformat(), "format": "json"}
        try:
            response = requests.get(f"{TIINGO_URL}/{ticker}/prices", params=params, headers=headers, timeout=15)
            for attempt in range(1, TIINGO_MAX_ATTEMPTS + 1):
                if response.status_code == 429 and attempt < TIINGO_MAX_ATTEMPTS:
                    time.sleep(TIINGO_BACKOFF_SECONDS * attempt)
                    response = requests.get(f"{TIINGO_URL}/{ticker}/prices", params=params, headers=headers, timeout=15)
                    continue
                break
            response.raise_for_status()
            rows = response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Tiingo request failed for %s: %s", ticker, _redact_api_key(str(exc)))
            continue
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if "adjClose" not in df.columns or "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        columns[ticker] = pd.to_numeric(df["adjClose"]).set_axis(df["date"]).sort_index()

    if not columns:
        raise MarketDataError("Tiingo fallback also returned no data for any ticker.")
    return pd.concat(columns, axis=1)


def _download_post_yahoo_fallback_chain(tickers: list[str], start: dt.date, end: dt.date) -> tuple[pd.DataFrame, str]:
    """
    Shared tail of the fallback cascade, used by BOTH `fetch_adjusted_close`
    branches that reach past Yahoo (the circuit-breaker-active path and the
    both-Yahoo-paths-just-failed path) — defined once here so the
    Tiingo → Twelve Data → Alpha Vantage order can't drift between the two
    call sites. Tiingo tried FIRST among these three: its free tier (500
    req/hour) is meaningfully looser than Twelve Data's (800/day) or Alpha
    Vantage's (25/day), so it's the best bet when Yahoo itself is down.
    Returns (prices, tier_label) so the caller can build an accurate
    `source` string without duplicating the try/except.
    """
    try:
        return _download_tiingo(tickers, start, end), "tiingo"
    except MarketDataError as tiingo_error:
        logger.warning("Tiingo failed, trying Twelve Data: %s", tiingo_error)
    try:
        return _download_twelvedata(tickers, start, end), "twelvedata"
    except MarketDataError as twelvedata_error:
        logger.warning("Twelve Data failed, trying Alpha Vantage: %s", twelvedata_error)
        return _download_alpha_vantage(tickers, start, end), "alpha vantage"


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_adjusted_close(
    tickers: list[str],
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices for one or more tickers. Five-step
    fallback chain: (1) the `yfinance` library, with retry-with-backoff — this
    is "the Yahoo Finance API" per the brief; (2) a direct, unauthenticated call
    to Yahoo's own REST endpoint, bypassing the library — same underlying source,
    different code path, in case the library's cookie/crumb handling specifically
    (not Yahoo itself) is what's failing; (3) Tiingo, a genuinely different
    provider with a generous free tier (500 req/hour), tried first once both
    Yahoo-based attempts are exhausted; (4) Twelve Data, tried if Tiingo fails;
    (5) Alpha Vantage, tried only if Twelve Data ALSO fails — its free tier's real
    ceiling (25 requests per DAY, one call per ticker, no batch endpoint)
    makes it the most constrained of the four, which is exactly why it sits last.

    Circuit breaker: Streamlit reruns the WHOLE script on every widget
    interaction — adding one ticker, moving a slider — which means a naive
    implementation re-attempts the full Yahoo retry sequence (up to 6 failed
    calls with backoff sleeps) on every single interaction while Yahoo is down,
    even though the previous attempt (seconds ago) already proved it's down.
    Once both Yahoo paths fail, `_yahoo_down_until` records "don't bother
    retrying Yahoo before this time" — subsequent calls within that window skip
    straight to Twelve Data. This is a genuine responsiveness fix, not just log
    noise reduction: observed in practice, a user rapidly toggling tickers while
    Yahoo was down was hitting the full multi-second retry chain on every click.

    Returns a DataFrame indexed by date, one column per ticker, forward-filled for
    isolated missing sessions (holidays that differ slightly across exchanges/ETFs)
    but NOT filled at the edges - leading/trailing NaNs are dropped so every column
    only spans dates where it actually traded.
    """
    if not tickers:
        raise MarketDataError("No tickers provided.")

    global _yahoo_down_until
    skip_yahoo = time.monotonic() < _yahoo_down_until

    source = "yfinance"
    if skip_yahoo:
        logger.info("Yahoo circuit breaker active (down recently) — skipping straight to Twelve Data.")
        prices, tier = _download_post_yahoo_fallback_chain(tickers, start, end)
        source = f"{tier} (yahoo recently unavailable)"
    else:
        try:
            raw = _download_yfinance(tickers, start, end)
            if isinstance(raw.columns, pd.MultiIndex):
                prices = pd.concat(
                    {t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)},
                    axis=1,
                )
            else:
                prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
        except MarketDataError as yfinance_error:
            logger.warning("yfinance library failed, trying direct Yahoo API: %s", yfinance_error)
            try:
                prices = _download_yahoo_direct(tickers, start, end)
                source = "yahoo direct API"
            except MarketDataError as yahoo_direct_error:
                logger.warning("Direct Yahoo API also failed, falling back to Twelve Data: %s", yahoo_direct_error)
                _yahoo_down_until = time.monotonic() + YAHOO_CIRCUIT_BREAKER_SECONDS
                prices, tier = _download_post_yahoo_fallback_chain(tickers, start, end)
                source = f"{tier} (yahoo unavailable)"

    missing = set(tickers) - set(prices.columns)
    if missing:
        raise MarketDataError(f"No data for: {', '.join(sorted(missing))} (source: {source}). Check the ticker symbols.")

    prices = prices.ffill().dropna(how="all")
    if prices.empty:
        raise MarketDataError(f"Downloaded data is empty after cleaning (source: {source}) — widen the date range.")

    prices.attrs["source"] = source
    return prices


def resample_prices(prices: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Resample daily adjusted-close prices to weekly/monthly, taking the LAST
    observation of each period (standard convention for price series — unlike
    returns, prices should never be averaged or summed across a period).
    """
    freq_map = {"daily": None, "weekly": "W-FRI", "monthly": "ME"}
    rule = freq_map.get(frequency)
    if rule is None:
        return prices
    source = prices.attrs.get("source")  # .resample() drops .attrs — carry it through
    resampled = prices.resample(rule).last().dropna(how="all")
    if source:
        resampled.attrs["source"] = source
    return resampled


# Stays on @st.cache_data — returns a Series, same reason as fetch_adjusted_close above.
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_vix_snapshot(lookback_days: int = 90) -> pd.Series | None:
    """
    Recent CBOE VIX levels (not an "adjusted close" — it's an index, not a
    tradeable asset, but VIX_TICKER still resolves to a valid `Close` series on
    every source below). Returns None on failure rather than raising, since the
    VIX panel is contextual risk information, not core to the app's maths.

    Reuses `fetch_adjusted_close`'s full 3-step fallback chain (yfinance ->
    direct Yahoo API -> Twelve Data) instead of a separate yfinance-only path —
    the VIX panel showing "n/a" whenever Yahoo alone is blocked (which is often,
    per the rest of this module's docstrings) was a real gap, not a deliberate
    trade-off worth keeping.
    """
    start = dt.date.today() - dt.timedelta(days=lookback_days)
    end = dt.date.today()
    try:
        prices = fetch_adjusted_close([VIX_TICKER], start, end)
    except MarketDataError:
        return None

    if VIX_TICKER not in prices.columns:
        return None
    return prices[VIX_TICKER].dropna()


TWELVEDATA_STATISTICS_URL = "https://api.twelvedata.com/statistics"


def _fetch_twelvedata_fundamentals(ticker: str) -> dict[str, Any] | None:
    """
    Twelve Data's /statistics endpoint — kept as a FALLBACK only. Confirmed via
    live testing that the free tier restricts this endpoint to
    their public demo symbol (AAPL); every other ticker 403s regardless of
    retry/pacing. Finnhub (tried first in `fetch_fundamentals` below) covers
    every ticker on its free tier, so this mostly matters if FINNHUB_API_KEY
    isn't configured, or specifically for AAPL where Twelve Data happens to work.
    """
    if not LLM_SETTINGS.twelvedata_api_key:
        return None

    for attempt in range(1, TWELVEDATA_MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                TWELVEDATA_STATISTICS_URL,
                params={"symbol": ticker, "apikey": LLM_SETTINGS.twelvedata_api_key},
                timeout=10,
            )
        except requests.RequestException as exc:
            logger.warning("Twelve Data fundamentals request failed for %s: %s", ticker, _redact_api_key(str(exc)))
            return None

        if response.status_code == 429:
            if attempt == TWELVEDATA_MAX_ATTEMPTS:
                logger.warning("Twelve Data fundamentals rate-limited for %s after %d attempts", ticker, attempt)
                return None
            time.sleep(TWELVEDATA_BACKOFF_SECONDS * attempt)
            continue

        try:
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Twelve Data fundamentals request failed for %s: %s", ticker, _redact_api_key(str(exc)))
            return None
        break
    else:
        return None

    if payload.get("status") == "error" or "code" in payload and "message" in payload:
        message = payload.get("message", str(payload))
        if payload.get("code") == 403:
            raise TwelveDataPlanRestricted(message)
        logger.warning("Twelve Data fundamentals unavailable for %s: %s", ticker, message)
        return None

    stats = payload.get("statistics", {})
    valuations = stats.get("valuations_metrics", {}) or {}
    stock_stats = stats.get("stock_statistics", {}) or {}
    dividends = stats.get("dividends_and_splits", {}) or {}

    result = {
        "name": (payload.get("meta") or {}).get("name"),
        "source": "Twelve Data",
        "market_cap": valuations.get("market_capitalization"),
        "pe_ratio": valuations.get("trailing_pe"),
        "forward_pe": valuations.get("forward_pe"),
        "peg_ratio": valuations.get("peg_ratio"),
        "price_to_book": valuations.get("price_to_book_mrq"),
        "beta": stock_stats.get("beta"),
        "dividend_yield": dividends.get("forward_annual_dividend_yield"),
        "52w_high": stock_stats.get("52_week_high"),
        "52w_low": stock_stats.get("52_week_low"),
    }
    return result if any(v is not None for k, v in result.items() if k not in ("name", "source")) else None


FINNHUB_PROFILE_URL = "https://finnhub.io/api/v1/stock/profile2"
FINNHUB_METRIC_URL = "https://finnhub.io/api/v1/stock/metric"


def _fetch_finnhub_fundamentals(ticker: str) -> dict[str, Any] | None:
    """
    Fundamentals via Finnhub's `/stock/profile2` (company profile) and
    `/stock/metric` (valuation/risk ratios) — PRIMARY source, tried before
    Twelve Data. Unlike Twelve Data's /statistics, these two endpoints are
    confirmed free-tier for arbitrary tickers (not restricted to a demo
    symbol) per Finnhub's own free-tier feature list and multiple independent
    working code examples — verified via web search, not a live key in this
    environment, so if field names have drifted, `curl` both endpoints
    directly before assuming this parser is stale.

    Quirk worth knowing if debugging: Finnhub's `marketCapitalization` is
    denominated in MILLIONS of the reporting currency, not raw units — this
    function multiplies by 1e6 so the UI's `$X,XXX,XXX,XXX` formatting stays
    consistent with the Twelve Data fallback's raw-unit convention.
    """
    if not LLM_SETTINGS.finnhub_api_key:
        return None

    try:
        profile_resp = requests.get(
            FINNHUB_PROFILE_URL, params={"symbol": ticker, "token": LLM_SETTINGS.finnhub_api_key}, timeout=10,
        )
        metric_resp = requests.get(
            FINNHUB_METRIC_URL, params={"symbol": ticker, "metric": "all", "token": LLM_SETTINGS.finnhub_api_key}, timeout=10,
        )
        profile_resp.raise_for_status()
        metric_resp.raise_for_status()
        profile = profile_resp.json()
        metric = (metric_resp.json() or {}).get("metric", {}) or {}
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Finnhub fundamentals request failed for %s: %s", ticker, exc)
        return None

    if not profile and not metric:
        return None  # Finnhub returns {} (not an error field) for an unrecognised symbol

    market_cap_millions = profile.get("marketCapitalization")
    # Computed once into a local rather than inline in the dict literal below:
    # calling metric.get(...) twice in one ternary (once for the truthiness
    # check, once for the value) looks identical to mypy but isn't provably the
    # SAME call result, so it couldn't narrow the "else None" branch away from
    # the divide — a local variable makes the None-check and the division
    # operate on the exact same narrowed value.
    dividend_yield_raw = metric.get("dividendYieldIndicatedAnnual") or metric.get("currentDividendYieldTTM")
    result = {
        "name": profile.get("name"),
        "source": "Finnhub",
        "market_cap": market_cap_millions * 1_000_000 if market_cap_millions else None,
        "pe_ratio": metric.get("peBasicExclExtraTTM") or metric.get("peTTM"),
        "forward_pe": metric.get("peForward"),  # often absent on free tier — stays None, shows as "—"
        "peg_ratio": metric.get("pegRatio"),
        "price_to_book": metric.get("pbQuarterly") or metric.get("pbAnnual"),
        "beta": metric.get("beta"),
        "dividend_yield": dividend_yield_raw / 100 if dividend_yield_raw else None,
        # Finnhub returns yield as a percentage number (e.g. 0.44 for 0.44%), our UI expects
        # a decimal fraction (0.0044) since it formats with :.2% — divide by 100 to match.
        "52w_high": metric.get("52WeekHigh"),
        "52w_low": metric.get("52WeekLow"),
    }
    return result if any(v is not None for k, v in result.items() if k not in ("name", "source")) else None


@cached(ttl_seconds=6 * 3600)
def fetch_fundamentals(ticker: str) -> dict[str, Any] | None:
    """
    Per-ticker fundamentals (P/E, market cap, dividend yield, beta, 52w range).
    Tries Finnhub first (free tier covers arbitrary tickers), falls back to
    Twelve Data only if Finnhub isn't configured or returns nothing — Twelve
    Data's free tier is confirmed restricted to their demo symbol (AAPL) for
    this data, so it's a fallback, not the primary path.

    Returns None (not an exception) if both sources fail or neither key is
    configured — fundamentals are an enrichment, not something that should
    ever block the rest of the app. May still raise TwelveDataPlanRestricted
    from the Twelve Data fallback — callers already handle that distinctly.
    """
    result = _fetch_finnhub_fundamentals(ticker)
    if result is not None:
        return result
    return _fetch_twelvedata_fundamentals(ticker)