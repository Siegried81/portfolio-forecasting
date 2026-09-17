"""
Portfolio Forecasting & Optimization — Streamlit app.

app.py:

Layout: a sidebar to configure the universe/dates/model, and four tabs:
  1. Overview          — prices, returns, correlation of the selected universe
  2. Efficient Frontier — historical mean-variance optimization
  3. Forecast & Compare — the brief's core ask: historical vs forecast-based vs
                           realized-optimal portfolios, side by side
  4. AI Analyst         — LLM commentary, news digest, and a grounded Q&A chatbot

Design choice: this file is the ORCHESTRATION layer only — it wires UI widgets to
the pure functions in src/. No financial formula and no LLM-prompting logic lives
here, so the calculation logic stays independently testable (see tests/) and the
UI stays skimmable.
"""
from __future__ import annotations

import datetime as dt
import time
from urllib.parse import quote as _url_quote

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
import streamlit as st

from src.academic_search import METHODOLOGY_SEARCH_QUERIES, detect_methodology_terms, search_academic_papers
from src.ai_features import answer_portfolio_question, build_results_context, generate_commentary, generate_news_digest
from src.rag import build_chunks, load_chunks, save_chunks
from src.backtesting import (
    PORTFOLIO_TYPE_ORDER,
    compare_to_previous_period,
    forecast_win_rate,
    generate_expanding_windows,
    run_walk_forward,
)
from src.config import (
    ALL_KNOWN_TICKERS,
    BENCHMARK_TICKER,
    COV_METHOD_LEDOIT_WOLF,
    COV_METHOD_PCA,
    DEFAULT_EQUITY_TICKERS,
    DEFAULT_FORECAST_HORIZON_DAYS,
    DEFAULT_MAX_WEIGHT_PER_ASSET,
    DEFAULT_PCA_FACTORS,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_TRANSACTION_COST_BPS,
    DEFAULT_WALK_FORWARD_WINDOWS,
    FREQUENCY_TO_PERIODS_PER_YEAR,
    LLM_SETTINGS,
    MAX_CHAT_HISTORY_MESSAGES,
    MAX_PCA_FACTORS,
    MAX_WALK_FORWARD_WINDOWS,
    MEGA_CAP_TICKERS,
    MIN_PCA_FACTORS,
    MIN_WALK_FORWARD_WINDOWS,
    QUICK_DATE_RANGES,
    SP500_SECTOR_UNIVERSE,
    WALK_FORWARD_MIN_TRAIN_PERIODS,
)
from src.forecasting import FORECAST_MODELS, forecast_all_assets
from src.llm_client import LLMUnavailableError
from src.factor_data import MIN_OBSERVATIONS_FOR_FACTOR_REGRESSION, compute_factor_exposures, fetch_fama_french_factors
from src.macro_data import fetch_current_risk_free_rate, fetch_macro_snapshot
from src.market_data import (
    MarketDataError,
    TwelveDataPlanRestricted,
    fetch_adjusted_close,
    fetch_fundamentals,
    fetch_vix_snapshot,
    resample_prices,
)
from src.metrics import apply_transaction_cost, compute_returns, compute_turnover, portfolio_returns, summarise_performance
from src.timeseries_diagnostics import adf_stationarity_test, hurst_exponent, rolling_sharpe
from src.optimization import (
    concentration_hhi,
    diversification_ratio,
    efficient_frontier_points,
    forecast_mu,
    historical_mu_cov,
    optimize_max_sharpe,
    portfolio_performance,
    resolve_weight_bounds,
)

st.set_page_config(page_title="Portfolio Forecasting & Optimization", layout="wide", page_icon="📈")

# Minimum number of ANNUAL observations before a yearly-frequency covariance
# estimate is treated as trustworthy rather than just technically computable.
# A handful of annual points cannot support a meaningful correlation
# estimate between assets — this is a warning threshold, not a hard block:
# the brief explicitly lists "yearly" as a valid frequency option, so it must
# remain SELECTABLE and functional even on a short date range; the honest
# response to too little data is a visible caveat, not silently disabling
# the option.
MIN_YEARLY_OBSERVATIONS_FOR_RELIABLE_COV = 8

# ----------------------------------------------------------------------------------
# Background — kept here as a real, readable
# multi-line string (encoded to a data-URI at RUNTIME via urllib.parse.quote) instead
# of a pre-encoded one-line blob — if you want to tweak the look, edit the numbers in
# _BACKGROUND_SVG directly below; every opacity/stroke-width/color is a plain, labelled
# value on its own line, not buried in a URL-encoded string.
#
# Reuses the SAME navy (#0E4C92) already used for the efficient-frontier line
# elsewhere in this file — one visual identity, not a decoration bolted on separately.
_BACKGROUND_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1600 900' preserveAspectRatio='xMidYMid slice'>
  <g stroke='#0E4C92' stroke-width='1.5' opacity='0.12'>
    <line x1='0' y1='100' x2='1600' y2='100'/>
    <line x1='0' y1='250' x2='1600' y2='250'/>
    <line x1='0' y1='400' x2='1600' y2='400'/>
    <line x1='0' y1='550' x2='1600' y2='550'/>
    <line x1='0' y1='700' x2='1600' y2='700'/>
    <line x1='0' y1='850' x2='1600' y2='850'/>
    <line x1='200' y1='0' x2='200' y2='900'/>
    <line x1='450' y1='0' x2='450' y2='900'/>
    <line x1='700' y1='0' x2='700' y2='900'/>
    <line x1='950' y1='0' x2='950' y2='900'/>
    <line x1='1200' y1='0' x2='1200' y2='900'/>
    <line x1='1450' y1='0' x2='1450' y2='900'/>
  </g>
  <path d='M0,620 L80,590 L160,640 L240,560 L320,610 L400,480 L480,520 L560,430 L640,470
           L720,380 L800,420 L880,340 L960,390 L1040,300 L1120,350 L1200,260
           L1280,300 L1360,220 L1440,270 L1520,190 L1600,230'
        fill='none' stroke='#0E4C92' stroke-width='2.5' opacity='0.20'/>
  <path d='M0,700 L80,720 L160,690 L240,730 L320,680 L400,710 L480,650 L560,690
           L640,630 L720,660 L800,600 L880,640 L960,590 L1040,610 L1120,560
           L1200,600 L1280,550 L1360,580 L1440,530 L1520,560 L1600,520'
        fill='none' stroke='#0E4C92' stroke-width='2.5' opacity='0.12'/>
</svg>
"""

_background_data_uri = "data:image/svg+xml," + _url_quote(
    " ".join(line.strip() for line in _BACKGROUND_SVG.splitlines())
)

st.markdown(
    f"""
    <style>
    /* Page + sidebar background — BOTH containers, not just the main area:
       the sidebar has its own opaque background in Streamlit by default and
       was silently hiding the pattern entirely before this fix. */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
        background-color: #E9EDF5;
        background-image: url("{_background_data_uri}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
        background-color: rgba(0, 0, 0, 0);
    }}
    /* Input fields — text inputs, text areas, chat input, number inputs, and
       select/multiselect boxes now get an explicit white fill + visible
       border, so they read as fields instead of blending into the new
       background color above. */
    .stTextInput input, .stTextArea textarea, .stChatInput textarea, .stNumberInput input {{
        background-color: #FFFFFF !important;
        border: 1px solid #C7CEDA !important;
        border-radius: 6px !important;
    }}
    [data-baseweb="select"] > div {{
        background-color: #FFFFFF !important;
        border: 1px solid #C7CEDA !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _format_ratio(x: float) -> str:
    """
    Format a ratio/metric value for display, distinguishing three genuinely
    different cases that a single blanket "—" used to collapse into one:
    NaN (undefined — e.g. zero volatility), +/-inf (Omega ratio with zero
    losing periods in the sample), and an ordinary finite number. Showing
    infinity as a bare "—" reads as a minus sign or a missing value, not as
    "there were no losses at all" — a materially different, more informative
    fact worth its own symbol.
    """
    if pd.isna(x):
        return "—"          # genuinely missing/undefined
    if np.isinf(x):
        return "∞"          # no losing periods at all — not a minus sign
    return f"{x:.2f}"


# ----------------------------------------------------------------------------------
# Sidebar — all user inputs live here, nowhere else, so the rest of the app can
# just read st.session_state / the returned tuple without re-deriving inputs.
# ----------------------------------------------------------------------------------
def render_sidebar() -> dict:
    st.sidebar.header("Portfolio configuration")

    def _apply_universe_preset() -> None:
        # .get(), not [...]: this callback and _apply_sector_selection below
        # each write to the OTHER widget's session_state key (this one resets
        # "sector_select", that one resets "universe_preset_select") — on
        # some Streamlit rerun timings that cross-write can race with this
        # callback firing before its own widget's key is (re)registered,
        # raising a KeyError ("did you forget to initialize it?") instead of
        # applying the preset. Bailing out to a no-op when the key genuinely
        # isn't there yet is safe: worst case this one rerun doesn't apply a
        # preset change, it doesn't corrupt any state.
        preset = st.session_state.get("universe_preset_select")
        if preset is None:
            return
        if preset == "By default (5)":
            st.session_state["assets_select"] = DEFAULT_EQUITY_TICKERS.copy()
        elif preset == "Mega Caps (15)":
            st.session_state["assets_select"] = MEGA_CAP_TICKERS.copy()
        # "Custom (edit manually)" leaves the current selection untouched — an
        # explicit escape hatch from the two presets above, distinct from
        # picking a sector (which is always available below, not gated behind
        # this option).
        if preset != "Custom (edit manually)":
            # Symmetric with _apply_sector_selection resetting THIS dropdown
            # to "Custom" below: picking an actual preset here REPLACES the
            # asset list too, so a sector chosen earlier ("Technology") would
            # otherwise keep showing as selected while the assets underneath
            # it no longer match that sector at all — the exact mismatch this
            # reset prevents, in the opposite direction.
            st.session_state["sector_select"] = "— choose a sector —"

    st.sidebar.selectbox(
        "Universe preset", options=["By default (5)", "Mega Caps (15)", "Custom (edit manually)"],
        key="universe_preset_select", on_change=_apply_universe_preset,
        help="Presets replace your current asset selection. Pick 'Custom (edit manually)' to "
             "leave your current selection as-is and edit the Assets pills below by hand — the "
             "GICS sector picker below is available regardless of which preset is chosen here.",
    )

    def _apply_sector_selection() -> None:
        # Same .get()-with-guard reasoning as _apply_universe_preset above —
        # this callback and that one cross-write each other's session_state key.
        sector = st.session_state.get("sector_select")
        if sector is None:
            return
        if sector in SP500_SECTOR_UNIVERSE:
            st.session_state["assets_select"] = sorted(SP500_SECTOR_UNIVERSE[sector])
            # Reset the "Universe preset" dropdown back to a neutral state:
            # picking a sector REPLACES the asset list, so leaving a preset
            # label like "Mega Caps (15)" displayed here would misleadingly
            # suggest that preset is still active, when it no longer reflects
            # what's actually selected (a sector's own ticker count rarely
            # matches any preset's count). "Custom (edit manually)" is the
            # correct neutral label — it doesn't re-trigger _apply_universe_preset
            # itself (Streamlit only fires on_change for the WIDGET that changed),
            # so this assignment alone is safe and won't overwrite the sector's
            # own selection just made above.
            st.session_state["universe_preset_select"] = "Custom (edit manually)"

    st.sidebar.selectbox(
        "GICS sector", options=["— choose a sector —"] + list(SP500_SECTOR_UNIVERSE.keys()),
        key="sector_select", on_change=_apply_sector_selection,
        help="Replaces the Assets list below with every ticker in the chosen sector — "
             "still editable by hand afterwards. Pick again to switch sectors, or edit the "
             "Assets pills below directly to mix sectors. Available regardless of the Universe "
             "preset above — picking a sector here always overrides it.",
    )

    def _apply_custom_tickers() -> None:
        """Sync newly-typed custom tickers into the Assets multiselect's OWN
        session_state ("assets_select") — needed because typing a
        ticker here already feeds the app's calculations correctly, but would never
        appear as a selected pill in the multiselect above, which reads as
        broken even though it wasn't. Only tickers already in
        ALL_KNOWN_TICKERS can become a pill (the multiselect's `options` list
        is fixed, Streamlit can't display a pill for an option outside it) —
        a genuinely unknown symbol still works for computation via the
        text-based merge below, it just won't show as a pill here. Runs
        BEFORE the script reruns (Streamlit's on_change contract), so by the
        time the multiselect widget below actually renders, this update is
        already in session_state.
        """
        custom_text = st.session_state.get("custom_tickers_input", "")
        if not custom_text.strip():
            return
        typed = [t.strip().upper() for t in custom_text.split(",") if t.strip()]
        known_new = [t for t in typed if t in ALL_KNOWN_TICKERS]
        if known_new:
            current = st.session_state.get("assets_select", [])
            st.session_state["assets_select"] = list(dict.fromkeys(current + known_new))

    # Initialised via setdefault (not the widget's own `default=` param) because
    # every OTHER path that changes this selection (preset callback, sector
    # callback, custom-ticker callback above) writes directly into
    # st.session_state["assets_select"] — passing `default=` on the widget
    # itself alongside that is contradictory (session_state always wins) and
    # triggers Streamlit's own check_session_state_rules warning. setdefault
    # only applies on the very first run, before any callback has set anything.
    st.session_state.setdefault("assets_select", DEFAULT_EQUITY_TICKERS.copy())
    tickers = st.sidebar.multiselect(
        "Assets", options=ALL_KNOWN_TICKERS, key="assets_select",
        help="Equities, sector picks, ETFs, commodities and FX proxies — all in one list. "
             "Add custom tickers below if you don't see what you're after.",
    )
    custom = st.sidebar.text_input(
        "Add custom tickers (comma-separated)", value="", key="custom_tickers_input",
        on_change=_apply_custom_tickers,
        help="Recognised tickers (already in this app's universe) are added as pills above "
             "automatically. Unrecognised symbols still work for the calculations below, they "
             "just won't show as a pill — this app has no way to validate an arbitrary symbol "
             "without a network call, so it's trusted as typed.",
    )
    if custom.strip():
        tickers = list(dict.fromkeys(tickers + [t.strip().upper() for t in custom.split(",") if t.strip()]))
    # Always shown, independent of the "Universe preset" dropdown's own label
    # (e.g. "Mega Caps (15)") — that label reflects what the PRESET applied,
    # not the live selection, which can drift the moment a ticker is added
    # or removed by hand afterward. This caption is the one number that's
    # always accurate for what's actually about to be computed.
    st.sidebar.caption(f"**{len(tickers)} ticker{'s' if len(tickers) != 1 else ''} selected**")

    today = dt.date.today()

    def _apply_quick_range() -> None:
        # .get(), not [...]: same defensive pattern as _apply_universe_preset /
        # _apply_sector_selection above — a callback can fire on a rerun before
        # its own widget's key is (re)registered in session_state (e.g. right
        # after a cache clear forces a full script rerun). Bailing out to a
        # no-op is safe: worst case this one rerun doesn't apply the quick-range
        # change, it never corrupts state.
        label = st.session_state.get("quick_range_select")
        if label is None:
            return
        if label != "Custom":
            st.session_state["start_date_input"] = today - dt.timedelta(days=QUICK_DATE_RANGES[label])
            st.session_state["end_date_input"] = today

    st.sidebar.selectbox(
        "Quick range", options=["Custom"] + list(QUICK_DATE_RANGES.keys()),
        index=3,  # defaults to "5 ans" on first load, matching the previous hardcoded default
        key="quick_range_select", on_change=_apply_quick_range,
        help="Picks a preset window. Switch to 'Custom' to set exact dates below.",
    )
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input(
        "Start date", value=today - dt.timedelta(days=QUICK_DATE_RANGES["5 ans"]), max_value=today, key="start_date_input",
    )
    end_date = col2.date_input("End date", value=today, max_value=today, key="end_date_input")

    frequency = st.sidebar.selectbox("Frequency", options=["daily", "weekly", "monthly", "yearly"], index=0)
    if frequency == "yearly":
        approx_years = (end_date - start_date).days / 365.25
        if approx_years < MIN_YEARLY_OBSERVATIONS_FOR_RELIABLE_COV:
            st.sidebar.caption(
                f"⚠️ Only ~{approx_years:.0f} year(s) of history in this date range — a yearly "
                f"covariance estimate needs at least ~{MIN_YEARLY_OBSERVATIONS_FOR_RELIABLE_COV} "
                "annual observations to be trustworthy. Widen the date range, or expect a noisy "
                "efficient frontier and correlation matrix at this frequency."
            )

    fred_rate = fetch_current_risk_free_rate()
    rf_default = fred_rate if fred_rate is not None else DEFAULT_RISK_FREE_RATE
    rf_label = "Risk-free rate (annual)" + (" — live 3M T-bill via FRED" if fred_rate is not None else "")
    risk_free_rate_pct = st.sidebar.slider(rf_label, 0.0, 10.0, rf_default * 100, 0.05, format="%.2f%%")
    risk_free_rate = risk_free_rate_pct / 100.0
    max_weight_pct = st.sidebar.slider(
        "Max weight per asset", 10.0, 100.0, DEFAULT_MAX_WEIGHT_PER_ASSET * 100, 5.0, format="%.0f%%",
        help="Position-size cap applied to every optimized portfolio (including the hindsight "
             "'realized-optimal' benchmark). Standard institutional practice — without it, "
             "optimizers happily put 100% into whichever single name got lucky, which is "
             "mathematically valid but not how any real portfolio is actually run. Automatically "
             "relaxed if it's tighter than 1/(number of assets selected).",
    )
    max_weight_per_asset = max_weight_pct / 100.0
    allow_short_selling = st.sidebar.checkbox(
        "Allow short selling", value=False,
        help="Long-only by default (weights ≥ 0). Enabling this lets the optimizer take negative "
             "(short) positions up to the same magnitude as the max-weight cap — e.g. with a 35% "
             "cap, weights can range -35% to +35% per asset. Portfolio stays fully invested "
             "(weights still sum to 100%); this does NOT add gross leverage beyond that. "
             "Note: with few assets selected, a tight cap can leave no real room to short — e.g. "
             "3 assets at a 35% cap can push a losing asset down to about +30% at most, never "
             "negative, since the other two can only fund 70% of the portfolio between them. "
             "More assets or a higher cap gives shorts more room to actually bind.",
    )
    transaction_cost_bps = st.sidebar.slider(
        "Transaction cost (bps per rebalance)", 0, 50, int(DEFAULT_TRANSACTION_COST_BPS), 1,
        help="Charged as turnover × this rate every time a portfolio rebalances — once for the "
             "initial trade in the comparison above, and at every walk-forward window boundary "
             "below (which is also, in effect, your rebalancing frequency: a shorter forecast "
             "horizon means more windows, i.e. more frequent rebalancing, i.e. more cumulative "
             "cost — try shortening the horizon to see this drag show up). Set to 0 for the "
             "frictionless textbook comparison.",
    )
    horizon_unit = {"daily": "trading days", "weekly": "weeks", "monthly": "months", "yearly": "years"}[frequency]
    forecast_horizon = st.sidebar.slider(
        f"Forecast horizon ({horizon_unit})", min_value=10, max_value=90, value=DEFAULT_FORECAST_HORIZON_DAYS, step=5,
        help="This slice of history at the END of your date range is held out and forecasted — "
             "the actual prices in that slice are then used to compute the 'realized-optimal' benchmark. "
             f"Counted in periods at the selected frequency ({frequency}), not calendar days.",
    )
    forecast_model = st.sidebar.selectbox("Forecast model", options=list(FORECAST_MODELS.keys()), index=1)

    st.sidebar.divider()
    walk_forward_windows = st.sidebar.slider(
        "Walk-forward windows", min_value=MIN_WALK_FORWARD_WINDOWS, max_value=MAX_WALK_FORWARD_WINDOWS,
        value=DEFAULT_WALK_FORWARD_WINDOWS,
        help="Repeats the historical/forecast/realized comparison across this many expanding "
             "windows instead of just one, so the result is a distribution, not a single lucky "
             "(or unlucky) outcome. ARIMA is noticeably slower here than ETS/naive.",
    )

    st.sidebar.divider()
    cov_method_label = st.sidebar.selectbox(
        "Covariance estimator", options=["Ledoit-Wolf shrinkage", "PCA factor model"],
        help="Ledoit-Wolf (default) is a solid fix for a small universe's noisy sample "
             "covariance. Once your universe gets wide (~40+ assets) relative to the history "
             "available, a PCA statistical factor model is the tractable alternative — same "
             "core idea real buy-side desks use (Barra, APT) for exactly this problem. Applied "
             "consistently everywhere a covariance is estimated (frontier, comparison, "
             "walk-forward), not just here.",
    )
    cov_method = COV_METHOD_PCA if cov_method_label == "PCA factor model" else COV_METHOD_LEDOIT_WOLF
    n_factors = DEFAULT_PCA_FACTORS
    if cov_method == COV_METHOD_PCA:
        n_factors = st.sidebar.slider(
            "Number of factors", min_value=MIN_PCA_FACTORS, max_value=MAX_PCA_FACTORS, value=DEFAULT_PCA_FACTORS,
            help="More factors capture more of the universe's variance but bring the parameter "
                 "count back up toward the problem this is meant to solve — the Efficient "
                 "Frontier tab shows cumulative explained variance so you can judge whether this "
                 "count is actually enough for your selected universe.",
        )

    return dict(
        tickers=tickers, start_date=start_date, end_date=end_date, frequency=frequency,
        risk_free_rate=risk_free_rate, forecast_horizon=forecast_horizon, forecast_model=forecast_model,
        walk_forward_windows=walk_forward_windows, max_weight_per_asset=max_weight_per_asset,
        allow_short_selling=allow_short_selling, transaction_cost_bps=transaction_cost_bps,
        cov_method=cov_method, n_factors=n_factors,
    )


# ----------------------------------------------------------------------------------
# Data loading — one shared call, cached inside market_data.py, so every tab
# reuses the same downloaded prices instead of re-fetching per tab.
# ----------------------------------------------------------------------------------
def load_data(config: dict) -> pd.DataFrame | None:
    tickers_with_benchmark = list(dict.fromkeys(config["tickers"] + [BENCHMARK_TICKER]))
    try:
        daily_prices = fetch_adjusted_close(tickers_with_benchmark, config["start_date"], config["end_date"])
    except MarketDataError as exc:
        st.error(f"Could not load market data: {exc}")
        return None
    return resample_prices(daily_prices, config["frequency"])


def render_macro_panel() -> dict:
    """
    Macro & risk context: 3M / 10Y Treasury yields, the term spread (classic
    recession signal when negative), and the VIX level (market "fear gauge").
    Returns a plain dict so it can also be folded into the AI commentary context —
    an analyst's read of portfolio results should account for the macro backdrop,
    not just the numbers in isolation.
    """
    st.subheader("Macro & risk context")
    macro = fetch_macro_snapshot()
    vix_series = fetch_vix_snapshot()
    vix_level = float(vix_series.iloc[-1]) if vix_series is not None and not vix_series.empty else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("3M T-bill (FRED)", f"{macro['three_month_yield']:.2%}" if macro["three_month_yield"] is not None else "n/a")
    col2.metric("10Y Treasury (FRED)", f"{macro['ten_year_yield']:.2%}" if macro["ten_year_yield"] is not None else "n/a")

    spread = macro["term_spread_10y_3m"]
    spread_label = f"{spread:+.2%}" if spread is not None else "n/a"
    col3.metric(
        "10Y-3M spread", spread_label,
        delta="Inverted — recession signal" if (spread is not None and spread < 0) else None,
        delta_color="inverse",
    )

    if vix_level is not None:
        vix_regime = "Calm" if vix_level < 15 else "Normal" if vix_level < 25 else "Elevated"
        col4.metric("VIX (fear gauge)", f"{vix_level:.1f}", delta=vix_regime, delta_color="off")
    else:
        col4.metric("VIX (fear gauge)", "n/a")

    # Second row: the rest of the Fed's dual mandate
    # (inflation + unemployment), the actual policy rate, and a credit-risk
    # signal — the yield curve and VIX alone miss all four of these.
    col5, col6, col7, col8 = st.columns(4)
    cpi = macro["cpi_yoy_inflation"]
    col5.metric("CPI YoY (inflation)", f"{cpi:.1%}" if cpi is not None else "n/a")
    unemployment = macro["unemployment_rate"]
    col6.metric("Unemployment rate", f"{unemployment:.1%}" if unemployment is not None else "n/a")
    fed_funds = macro["fed_funds_rate"]
    col7.metric("Fed funds rate (effective)", f"{fed_funds:.2%}" if fed_funds is not None else "n/a")
    credit_spread = macro["credit_spread_baa10y"]
    col8.metric("Baa credit spread (vs 10Y)", f"{credit_spread:.2%}" if credit_spread is not None else "n/a")

    col9, col10 = st.columns(2)
    gdp_growth = macro["gdp_growth_qoq_annualized"]
    col9.metric(
        "GDP growth (QoQ, annualized)", f"{gdp_growth:+.1%}" if gdp_growth is not None else "n/a",
        help="Real GDP % change, quarterly, seasonally adjusted annual rate (FRED). Updated quarterly, "
             "so this changes far less often than the daily/monthly series above.",
    )
    indpro = macro["industrial_production_yoy"]
    col10.metric(
        "Industrial production (YoY)", f"{indpro:+.1%}" if indpro is not None else "n/a",
        help="Proxy for ISM Manufacturing PMI: ISM's own PMI is a paid, proprietary survey-based "
             "series not available on FRED or any free API. Industrial Production is the closest "
             "free substitute — real output data rather than a survey diffusion index, but it "
             "captures the same underlying signal (manufacturing-sector momentum).",
    )

    sahm = macro["sahm_rule_indicator"]
    if sahm is not None:
        st.metric(
            "Sahm Rule recession indicator", f"{sahm:.2f}",
            delta="≥0.50 — historically marks a recession's start" if sahm >= 0.50 else "Below the 0.50 threshold",
            delta_color="inverse",
        )
        st.caption(
            "Sahm Rule: the 3-month average unemployment rate rising 0.50 points above its low of "
            "the prior 12 months. Every reading ≥0.50 has coincided with the start of a US "
            "recession since 1970, with no false positives to date (Claudia Sahm / FRED)."
        )

    if macro["three_month_yield"] is None or macro["ten_year_yield"] is None:
        st.caption("Set FRED_API_KEY in .env for live Treasury yields.")
    st.caption(
        "The 10Y-3M spread turning negative has preceded every US recession since the 1960s "
        "(with some false positives) — shown here as context, not a trading signal in itself."
    )
    with st.expander("Sources (FRED series)"):
        st.markdown(
            "- [3M T-bill (DGS3MO)](https://fred.stlouisfed.org/series/DGS3MO)\n"
            "- [10Y Treasury (DGS10)](https://fred.stlouisfed.org/series/DGS10)\n"
            "- [CPI, all urban consumers (CPIAUCSL)](https://fred.stlouisfed.org/series/CPIAUCSL)\n"
            "- [Unemployment rate (UNRATE)](https://fred.stlouisfed.org/series/UNRATE)\n"
            "- [Effective Fed funds rate, daily (DFF)](https://fred.stlouisfed.org/series/DFF)\n"
            "- [Sahm Rule recession indicator (SAHMREALTIME)](https://fred.stlouisfed.org/series/SAHMREALTIME)\n"
            "- [Baa corporate bond yield vs 10Y (BAA10Y)](https://fred.stlouisfed.org/series/BAA10Y)\n"
            "- [Real GDP growth (A191RL1Q225SBEA)](https://fred.stlouisfed.org/series/A191RL1Q225SBEA)\n"
            "- [Industrial Production Index — ISM PMI proxy (INDPRO)](https://fred.stlouisfed.org/series/INDPRO)"
        )

    return {"macro": macro, "vix_level": vix_level}


def _chart_palette(n: int) -> list[str]:
    """
    Return `n` visually distinct line colors for a multi-ticker chart.

    Plotly's default qualitative palette (`qualitative.Plotly`) only has 10
    colors — with more than 10 tickers selected, the 11th ticker would
    silently reuse ticker #1's color, making two unrelated lines
    indistinguishable in the legend.

    `qualitative.Alphabet` (26 distinct colors) covers any realistic
    selection. If a user somehow selects more than that, sample a continuous
    colorscale for EXACTLY `n` colors instead of wrapping — colors keep
    changing (even if less visually distinct at that extreme) rather than
    silently repeating and making two unrelated lines indistinguishable.
    """
    palette = plotly.colors.qualitative.Alphabet
    if n <= len(palette):
        return palette[:n]
    return plotly.colors.sample_colorscale("turbo", [i / max(n - 1, 1) for i in range(n)])


def render_overview_tab(prices: pd.DataFrame, tickers: list[str], periods_per_year: int) -> None:
    source = prices.attrs.get("source", "yfinance")
    if "twelvedata" in source:
        st.caption(f"⚠️ Data source: **{source}** — Yahoo Finance was unreachable, using the Twelve Data fallback.")
    elif "yahoo direct" in source:
        st.caption(f"ℹ️ Data source: **{source}** — the yfinance library failed, but a direct Yahoo API call succeeded.")

    st.subheader("Price history (adjusted close)")
    palette = _chart_palette(len(tickers))
    fig = go.Figure()
    for i, ticker in enumerate(tickers):
        color = palette[i]
        # Rebase to 100 at the start so assets with very different price levels
        # (e.g. AAPL ~$200 vs GOOG ~$150) are visually comparable on one chart.
        rebased = prices[ticker] / prices[ticker].iloc[0] * 100
        fig.add_trace(go.Scatter(x=rebased.index, y=rebased.values, name=ticker, mode="lines", line=dict(color=color)))
        # Label the ticker directly at the end of its own line, not just in the
        # legend/hover tooltip — with several overlapping lines, matching a
        # color to a name in the legend is slower than reading it right off
        # the chart where the line actually ends. Font color matches the line
        # explicitly (rather than relying on Plotly's own auto-cycle, which
        # this loop no longer uses now that color is assigned by hand above)
        # so the label is unambiguous even when several lines end close together.
        fig.add_annotation(
            x=rebased.index[-1], y=rebased.values[-1], text=ticker, showarrow=False,
            xanchor="left", xshift=6, font=dict(size=11, color=color),
        )
    fig.update_layout(yaxis_title="Rebased to 100", height=420, legend_title="Ticker", margin=dict(r=60))
    st.plotly_chart(fig, width='stretch')

    returns = compute_returns(prices[tickers])
    # BENCHMARK_TICKER is always fetched in load_data() regardless of the user's
    # selection, specifically so beta/Information Ratio are always computable here
    # even if the user didn't add SPY to their own asset list.
    benchmark_returns = compute_returns(prices[[BENCHMARK_TICKER]])[BENCHMARK_TICKER] if BENCHMARK_TICKER in prices.columns else None

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Per-asset annualised metrics")
        rows = []
        for ticker in tickers:
            asset_benchmark = benchmark_returns if ticker != BENCHMARK_TICKER else None  # a benchmark vs itself is meaningless
            m = summarise_performance(returns[ticker], periods_per_year=periods_per_year, benchmark_returns=asset_benchmark)
            rows.append({
                "Ticker": ticker,
                "Ann. return": f"{m['annual_return']:.1%}",
                "Ann. volatility": f"{m['annual_volatility']:.1%}",
                "Sharpe": f"{m['sharpe_ratio']:.2f}",
                "Sharpe SE": _format_ratio(m['sharpe_se']),
                "Sortino": f"{m['sortino_ratio']:.2f}",
                "Calmar": f"{m['calmar_ratio']:.2f}" if not pd.isna(m['calmar_ratio']) else "—",
                "Omega": _format_ratio(m['omega_ratio']),
                "Beta (vs SPY)": f"{m.get('beta', float('nan')):.2f}" if 'beta' in m and not pd.isna(m['beta']) else "—",
                "Alpha (Jensen, ann.)": f"{m['jensens_alpha']:+.2%}" if 'jensens_alpha' in m and not pd.isna(m['jensens_alpha']) else "—",
                "Max drawdown": f"{m['max_drawdown']:.1%}",
                "Ulcer Index": f"{m['ulcer_index']:.2f}" if not pd.isna(m['ulcer_index']) else "—",
                "Skew": f"{m['skewness']:.2f}" if not pd.isna(m['skewness']) else "—",
                "Kurtosis": f"{m['kurtosis']:.2f}" if not pd.isna(m['kurtosis']) else "—",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Ticker"), width='stretch')

    with col_right:
        st.subheader("Correlation matrix")
        corr = returns.corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1,
            colorscale="RdBu", reversescale=True, text=corr.round(2).values, texttemplate="%{text}",
        ))
        fig_corr.update_layout(height=380)
        st.plotly_chart(fig_corr, width='stretch')
        st.caption(
            "Low/negative correlation between holdings is what actually reduces "
            "portfolio-level risk below the average of the individual assets' risk — "
            "this matrix is the reason diversification works, not just a decorative chart."
        )

    st.divider()
    st.subheader("Time-series diagnostics")
    st.caption(
        "Structure of the series itself, not a performance summary — the kind of check an "
        "economist does before trusting a forecast, not after. The ADF test directly validates "
        "this app's own ARIMA `d=1` assumption (see forecasting.py); the Hurst exponent gives a "
        "concrete, per-asset number for the efficient-market-hypothesis caveat stated throughout "
        "this app's forecasting UI."
    )
    diag_rows = []
    for ticker in tickers:
        adf = adf_stationarity_test(returns[ticker])
        h = hurst_exponent(prices[ticker])
        if h > 0.55:
            regime = "Trending"
        elif h < 0.45:
            regime = "Mean-reverting"
        elif not np.isnan(h):
            regime = "≈ Random walk"
        else:
            regime = "—"
        diag_rows.append({
            "Ticker": ticker,
            "ADF p-value": f"{adf['p_value']:.3f}" if adf["p_value"] is not None else "—",
            "Returns stationary?": (
                "Yes" if adf["is_stationary"] is True else "No" if adf["is_stationary"] is False else "—"
            ),
            "Hurst exponent": f"{h:.3f}" if not np.isnan(h) else "—",
            "Regime": regime,
        })
    st.dataframe(pd.DataFrame(diag_rows).set_index("Ticker"), width='stretch')
    st.caption(
        "ADF: 'Yes' (p < 0.05) means the return series itself shows no unit root — the standard, "
        "expected result for returns (as opposed to price LEVELS, which are non-stationary by "
        "construction). Hurst: >0.55 trending "
        "(momentum), <0.45 mean-reverting, ≈0.5 a random walk — most liquid equities sit close to "
        "0.5, consistent with weak short-horizon predictability."
    )

    # Anchored to a fixed real-world span (~1 quarter), not a raw period count:
    # "60 periods" means wildly different things depending on frequency (~3
    # months daily, ~14 months weekly, ~5 years monthly) — a raw count is
    # comparable-looking but not actually comparable, and the calendar
    # meaning silently depends on a different sidebar control. Converting
    # from `periods_per_year` keeps the WINDOW'S MEANING constant ("roughly
    # the trailing quarter") across daily/weekly/monthly/yearly, which is
    # what a rolling risk-adjusted-return chart is supposed to show — recent
    # regime, not an arbitrary period count.
    target_months = 3
    rolling_window = max(6, min(len(returns) // 4, round(periods_per_year * target_months / 12)))
    window_label = f"{rolling_window}-period (~{round(rolling_window / periods_per_year * 12)}mo)"
    fig_rolling = go.Figure()
    rolling_palette = _chart_palette(len(tickers))
    for i, ticker in enumerate(tickers):
        rs = rolling_sharpe(returns[ticker], window=rolling_window, periods_per_year=periods_per_year)
        fig_rolling.add_trace(go.Scatter(x=rs.index, y=rs.values, name=ticker, mode="lines", line=dict(color=rolling_palette[i])))
    fig_rolling.update_layout(
        title=f"Rolling {window_label} Sharpe ratio",
        yaxis_title="Rolling Sharpe", height=360,
    )
    st.plotly_chart(fig_rolling, width='stretch')
    st.caption(
        "A single end-of-sample Sharpe can hide a regime change (calm-then-crisis, or the "
        "reverse) — this shows how risk-adjusted return actually evolved period to period."
    )

    st.divider()
    st.subheader("Fundamentals")
    st.caption(
        "Via Finnhub first; Twelve Data and then "
        "yfinance each fill in whatever individual fields are still missing on top of that, and Twelve Data is used as the primary source outright if "
        "Finnhub isn't configured or returns nothing — some fields may still be unavailable "
        "depending on the exchange/plan or a temporary Yahoo rate limit. Fetched on demand to conserve API quota."
    )
    if not (LLM_SETTINGS.finnhub_api_key or LLM_SETTINGS.twelvedata_api_key):
        st.info("Set `FINNHUB_API_KEY` (preferred) or `TWELVEDATA_API_KEY` in `.env` to enable this section.")
    elif st.button("Fetch fundamentals for selected assets"):
        # The 8 req/min pacing below only protects Twelve Data's free tier — it's
        # irrelevant (and needlessly slow) when Finnhub is configured, since
        # fetch_fundamentals() tries Finnhub FIRST and Finnhub's own free tier
        # is 60 req/min. Only warn about / apply the slow pacing when Twelve
        # Data is the sole configured source (i.e. actually on the hot path).
        twelvedata_is_primary = not LLM_SETTINGS.finnhub_api_key
        if twelvedata_is_primary and len(tickers) > 8:
            st.caption(
                f"⏳ {len(tickers)} tickers on an 8 req/min free-tier limit — this will take "
                f"~{len(tickers) * 8 // 60 + 1} min. Paced deliberately to avoid hitting the rate limit."
            )
        progress = st.progress(0.0, text="Fetching fundamentals...")
        rows = []
        plan_restricted = False
        for i, ticker in enumerate(tickers):
            if i > 0 and twelvedata_is_primary:
                time.sleep(7.5)  # keeps us under Twelve Data's free-tier 8 req/min, proactively
            if plan_restricted:
                f = None
            else:
                try:
                    f = fetch_fundamentals(ticker)
                except TwelveDataPlanRestricted:
                    plan_restricted = True
                    f = None
            progress.progress((i + 1) / len(tickers), text=f"Fetching fundamentals... ({i + 1}/{len(tickers)})")
            rows.append({
                "Ticker": ticker,
                "Name": (f or {}).get("name") or "—",
                "Market cap": f"${f['market_cap']:,.0f}" if f and f.get("market_cap") else "—",
                "P/E (trailing)": f"{f['pe_ratio']:.1f}" if f and f.get("pe_ratio") else "—",
                "Forward P/E": f"{f['forward_pe']:.1f}" if f and f.get("forward_pe") else "—",
                "Beta": f"{f['beta']:.2f}" if f and f.get("beta") else "—",
                "Div. yield": f"{f['dividend_yield']:.2%}" if f and f.get("dividend_yield") else "—",
                "52w range": (
                    f"${f['52w_low']:.0f} – ${f['52w_high']:.0f}"
                    if f and f.get("52w_low") and f.get("52w_high") else "—"
                ),
                "Source": (f or {}).get("source") or "—",
            })
        progress.empty()
        st.dataframe(pd.DataFrame(rows).set_index("Ticker"), width='stretch')
        if plan_restricted:
            st.warning(
                "Your Twelve Data plan only includes `/statistics` for their public demo symbol "
                "(AAPL) — every other ticker returns a 403 'requires pro/ultra/venture/enterprise "
                "plan' error, confirmed directly against the API. This is a plan limitation, not "
                "a bug: see [twelvedata.com/pricing](https://twelvedata.com/pricing) if you want "
                "fundamentals for more than one demo ticker."
            )
        elif all(r["Market cap"] == "—" and r["P/E (trailing)"] == "—" for r in rows):
            st.warning(
                "No fundamentals fields came back for any ticker — this likely means your "
                "Twelve Data plan doesn't include the `/statistics` endpoint's fields. "
                "Check your plan at twelvedata.com/pricing."
                )


def render_frontier_tab(
    prices: pd.DataFrame, tickers: list[str], risk_free_rate: float, periods_per_year: int, max_weight_per_asset: float,
    allow_short_selling: bool, cov_method: str = COV_METHOD_LEDOIT_WOLF, n_factors: int = DEFAULT_PCA_FACTORS,
) -> pd.Series:
    st.subheader("Mean-variance efficient frontier (historical)")
    mu, cov = historical_mu_cov(prices[tickers], periods_per_year, cov_method=cov_method, n_factors=n_factors)
    if cov_method == COV_METHOD_PCA:
        # Recomputed here (cheap — milliseconds) purely for its diagnostics;
        # historical_mu_cov() itself keeps a clean 2-value (mu, cov) return
        # contract every other call site already relies on.
        from src.factor_models import pca_factor_cov
        _, pca_diagnostics = pca_factor_cov(
            compute_returns(prices[tickers]), n_factors=n_factors, periods_per_year=periods_per_year,
        )
        cumulative = pca_diagnostics["cumulative_explained"]
        used = pca_diagnostics["n_factors_used"]
        st.caption(
            f"PCA factor model: {used} factor(s) explain **{cumulative:.0%}** of this universe's "
            f"variance. " + (
                "Below 50% means most of the risk here is idiosyncratic, not systematic — this "
                "covariance estimate is weaker than it looks; consider more factors or fewer assets."
                if cumulative < 0.5 else
                "A reasonable capture — the systematic risk this covariance is built from represents "
                "most of what actually moves this universe."
            )
        )
    weight_bounds = resolve_weight_bounds(max_weight_per_asset, allow_short_selling)
    weights = optimize_max_sharpe(mu, cov, risk_free_rate=risk_free_rate, weight_bounds=weight_bounds)
    perf = portfolio_performance(mu, cov, weights, risk_free_rate)
    frontier = efficient_frontier_points(mu, cov, weight_bounds=weight_bounds)
    # A single selected ticker (or any universe where every feasible portfolio
    # has the same expected return) makes the "frontier" mathematically a
    # single point — see efficient_frontier_points' own docstring. Flag it
    # explicitly rather than silently showing an empty-looking chart with no
    # frontier line on it.
    if len(frontier) <= 1 and len(tickers) < 2:
        st.info(
            "Only one asset selected — there's no frontier to trade off "
            "return against risk with a single position. Add at least one "
            "more asset to see the efficient frontier curve."
        )

    col_chart, col_weights = st.columns([2, 1])
    with col_chart:
        fig = go.Figure()
        if len(frontier) > 1:
            fig.add_trace(go.Scatter(
                x=frontier["volatility"], y=frontier["return"], mode="lines",
                name="Efficient frontier", line=dict(color="#0E4C92"),
            ))
        # Individual assets, for context — shows why diversifying beats holding any single name
        for ticker in tickers:
            fig.add_trace(go.Scatter(
                x=[cov.loc[ticker, ticker] ** 0.5], y=[mu[ticker]], mode="markers+text",
                text=[ticker], textposition="top center", marker=dict(size=9), name=ticker, showlegend=False,
            ))
        fig.add_trace(go.Scatter(
            x=[perf["expected_volatility"]], y=[perf["expected_return"]], mode="markers",
            marker=dict(size=16, color="red", symbol="star"), name="Max-Sharpe portfolio",
        ))
        fig.update_layout(xaxis_title="Annualised volatility", yaxis_title="Annualised expected return", height=460)

        st.plotly_chart(fig, width='stretch', key=f"frontier_chart_{'-'.join(sorted(tickers))}")

    with col_weights:
        st.metric("Expected annual return", f"{perf['expected_return']:.1%}")
        st.metric("Expected annual volatility", f"{perf['expected_volatility']:.1%}")
        st.metric("Expected Sharpe ratio", f"{perf['expected_sharpe']:.2f}")
        div_ratio = diversification_ratio(weights, cov)
        st.metric(
            "Diversification ratio", f"{div_ratio:.2f}" if not pd.isna(div_ratio) else "n/a",
            help="Weighted-average individual asset volatility ÷ actual portfolio volatility. "
                 ">1 means diversification is genuinely reducing risk below what the assets' own "
                 "volatilities alone would suggest; =1 means none is being captured.",
        )
        hhi = concentration_hhi(weights)
        st.metric(
            "Concentration (HHI)", f"{hhi:.2f}",
            help=f"Sum of squared weights. 1/{len(weights)} ≈ {1 / len(weights):.2f} would be "
                 "perfectly equal-weighted; 1.0 would be a single asset. A quick check that the "
                 "max-weight cap is doing its job.",
        )
        st.write("**Optimal weights (max Sharpe):**")

        significant = weights[weights.abs() > 0.001]
        weights_display = significant.reindex(significant.abs().sort_values(ascending=False).index)
        st.dataframe(weights_display.map(lambda w: f"{w:.1%}").rename("Weight"), width='stretch')
        st.caption(
            f"Sums to {weights.sum():.1%} — should be ~100% (fully invested). Short positions "
            "(if short selling is enabled) show as negative weights above, not omitted."
        )

    st.caption(
        "Note: 'expected' figures above come from the optimizer's own mu/covariance inputs "
        "(historical, Ledoit-Wolf shrinkage) — they are the optimizer's target, not a guarantee. "
        "See the Forecast & Compare tab for how this historical optimum performs OUT of sample."
    )

    with st.expander("Factor exposure (Fama-French)"):
        st.caption(
            "How much of this portfolio's HISTORICAL excess return is explained by well-known "
            "return factors (market, size, value, and — if the 5-factor model is picked — "
            "profitability and investment) rather than a single CAPM beta. Free data from Ken "
            "French's Data Library (Dartmouth); no API key needed."
        )
        factor_model = st.radio(
            "Factor model", ["3-factor", "5-factor"], horizontal=True, key="factor_model_choice",
        )
        with st.spinner("Fetching Fama-French factor data..."):
            factors = fetch_fama_french_factors(factor_model)
        if factors is None:
            st.warning(
                "Couldn't fetch Fama-French factor data right now (network issue, or the Data "
                "Library's file layout changed) — this is enrichment, not required for the rest "
                "of this app to work."
            )
        else:
            portfolio_hist_returns = portfolio_returns(compute_returns(prices[tickers]), weights)
            exposures = compute_factor_exposures(portfolio_hist_returns, factors, periods_per_year)
            if exposures is None:
                st.info(
                    "Not enough overlapping history between this portfolio's returns and the "
                    f"factor data (need {MIN_OBSERVATIONS_FOR_FACTOR_REGRESSION}+ overlapping "
                    "days) — widen the date range in the sidebar."
                )
            else:
                loadings_df = pd.Series(exposures["loadings"]).rename("Loading").to_frame()
                st.dataframe(loadings_df.style.format("{:.2f}"), width='stretch')
                col_alpha, col_r2, col_n = st.columns(3)
                col_alpha.metric("Annualised alpha", f"{exposures['alpha_annualised']:.1%}")
                col_r2.metric("R²", f"{exposures['r_squared']:.2f}")
                col_n.metric("Observations", exposures["n_obs"])
                st.caption(
                    "Alpha here is the return this portfolio earned that its factor exposures "
                    "DON'T explain — the multi-factor generalisation of the single-factor "
                    "Jensen's alpha shown elsewhere in this app. A low R² means these loadings "
                    "are a noisy read, not a confident one."
                )
    return weights


def render_forecast_compare_tab(
    prices: pd.DataFrame, tickers: list[str], config: dict,
) -> tuple[dict, dict, dict] | None:
    """
    The heart of the brief: split history into a training window and a held-out
    window of length `forecast_horizon`. Build THREE portfolios and compare them
    on the SAME held-out window, so it's a fair, apples-to-apples comparison:

      1. Historical-optimal : weights from mu/cov estimated on the TRAINING window only
      2. Forecast-optimal   : weights from mu forecasted FOR the held-out window
                               (cov still historical — see optimization.py docstring)
      3. Realized-optimal   : weights from mu/cov estimated on the held-out window's
                               ACTUAL returns — the hindsight benchmark

    All three portfolios' weights are then applied to the SAME actual realized
    returns of the held-out window, so the bar chart below answers the brief's
    exact question: "did forecasting help, and by how much vs. doing nothing
    fancy, vs. the best that was theoretically achievable?"
    """
    horizon = config["forecast_horizon"]
    if len(prices) <= horizon + 30:
        st.warning(
            f"Not enough history for a {horizon}-day held-out window — widen the date "
            "range or shorten the forecast horizon in the sidebar."
        )
        return None

    train_prices = prices.iloc[:-(horizon)]
    test_prices = prices.iloc[-(horizon + 1):]
    test_returns = compute_returns(test_prices[tickers])
    periods_per_year = FREQUENCY_TO_PERIODS_PER_YEAR[config["frequency"]]
    rf = config["risk_free_rate"]

    with st.spinner(f"Fitting {config['forecast_model']} per asset..."):
        forecasted_prices = forecast_all_assets(
            train_prices[tickers], horizon, config["forecast_model"], config["frequency"],
        )

    st.subheader("Forecast fan chart — confidence band")
    st.caption(
        "The point forecast above feeds the portfolio optimizer, but a single number hides how "
        "much uncertainty compounds over the horizon. The shaded band (95% confidence interval, "
        "widening with `√horizon` — see forecasting.py) shows further-out points are genuinely "
        "less reliable, not just 'the same trend, more of it.'"
    )
    MAX_FAN_CHART_TICKERS = 4  # more than this and overlapping confidence bands become unreadable
    fan_tickers = st.multiselect(
        "Asset(s) to inspect", options=tickers, default=[tickers[0]], key="fan_chart_tickers",
        max_selections=MAX_FAN_CHART_TICKERS,
        help=f"Compare up to {MAX_FAN_CHART_TICKERS} assets' forecasts side by side. With more "
             "than one selected, prices are shown on very different scales unless rebased — "
             "see the toggle below.",
    )
    if not fan_tickers:
        st.info("Pick at least one asset above to see its forecast.")
    else:
        rebase_fan = False
        if len(fan_tickers) > 1:
            rebase_fan = st.checkbox(
                "Rebase to 100 at the start of the shown history", value=True, key="fan_chart_rebase",
                help="With several assets at very different price levels (e.g. $20 vs $400), "
                     "rebasing makes their forecasted TRENDS directly comparable on one chart. "
                     "Turn off to see actual forecasted prices — only readable with assets on "
                     "similar price levels.",
            )
        fig_fan = go.Figure()
        # Translucent per-ticker fill colors, matched by index to the line colors below —
        # Plotly needs an explicit rgba() string for the band's fill (a plain hex color has
        # no alpha channel), so this can't just reuse a hex palette directly.
        fan_fill_colors = [
            "rgba(31,119,180,0.20)", "rgba(255,127,14,0.20)",
            "rgba(44,160,44,0.20)", "rgba(214,39,40,0.20)",
        ]
        fan_line_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, ticker in enumerate(fan_tickers):
            fan_result = FORECAST_MODELS[config["forecast_model"]](
                train_prices[ticker], horizon, frequency=config["frequency"],
            )
            hist_tail = train_prices[ticker].iloc[-60:]  # last ~60 points of history for context, not the whole series

            if rebase_fan:
                # Divide every series (history, forecast, AND both band edges) by the SAME
                # scalar (the first historical point shown) — this preserves the band's
                # PROPORTIONAL width relative to the forecast; scaling only the line and
                # leaving the band on its original absolute scale would make the band look
                # wrong relative to the (now rebased) line it's supposed to surround.
                scale = 100.0 / hist_tail.iloc[0]
                hist_y = hist_tail.values * scale
                forecast_y = fan_result["forecast"].values * scale
                upper_y = fan_result["upper"].values * scale
                lower_y = fan_result["lower"].values * scale
            else:
                hist_y = hist_tail.values
                forecast_y = fan_result["forecast"].values
                upper_y = fan_result["upper"].values
                lower_y = fan_result["lower"].values

            # Single ticker selected: use the FIXED blue=historical / orange=forecast
            # scheme this chart was originally designed around (matches the
            # documented screenshot) — reusing one palette color for both lines
            # would make the whole chart monochrome. With 2+ tickers, fall back
            # to one distinct color PER TICKER (covering both its historical and
            # forecast trace), since that's what actually lets you tell tickers
            # apart when comparing several at once — a fixed orange for every
            # ticker's forecast would make them indistinguishable from each other.
            if len(fan_tickers) == 1:
                hist_color, forecast_color, fill_color = "#1f77b4", "#ff7f0e", "rgba(255,127,14,0.20)"
            else:
                hist_color = forecast_color = fan_line_colors[i % len(fan_line_colors)]
                fill_color = fan_fill_colors[i % len(fan_fill_colors)]
            fig_fan.add_trace(go.Scatter(
                x=hist_tail.index, y=hist_y, name=f"{ticker} — Historical", mode="lines",
                line=dict(color=hist_color),
            ))
            fig_fan.add_trace(go.Scatter(
                x=fan_result["upper"].index, y=upper_y, mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            fig_fan.add_trace(go.Scatter(
                # fill="tonexty" fills the area between THIS trace and the one added right
                # before it (upper) — this order (upper first, then lower with the fill) is
                # what actually draws the widening band, not incidental.
                x=fan_result["lower"].index, y=lower_y, name=f"{ticker} — 95% band", mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor=fill_color, hoverinfo="skip",
            ))
            fig_fan.add_trace(go.Scatter(
                x=fan_result["forecast"].index, y=forecast_y, name=f"{ticker} — Forecast", mode="lines",
                line=dict(color=forecast_color, dash="dash"),
            ))
        fig_fan.update_layout(
            yaxis_title="Rebased to 100" if rebase_fan else "Price",
            height=420, legend_title=None, margin=dict(t=10),
        )
        st.plotly_chart(fig_fan, width='stretch')

    weight_bounds = resolve_weight_bounds(config["max_weight_per_asset"], config["allow_short_selling"])

    # --- 1. Historical-optimal (trained on train_prices only) ---
    mu_hist, cov_hist = historical_mu_cov(
        train_prices[tickers], periods_per_year, cov_method=config["cov_method"], n_factors=config["n_factors"],
    )
    w_historical = optimize_max_sharpe(mu_hist, cov_hist, rf, weight_bounds)

    # --- 2. Forecast-optimal (mu from the forecast, cov still historical) ---
    mu_fcst = forecast_mu(train_prices.iloc[-1][tickers], forecasted_prices, horizon, periods_per_year)
    w_forecast = optimize_max_sharpe(mu_fcst, cov_hist, rf, weight_bounds)

    # --- 3. Realized-optimal (hindsight: fitted on the actual held-out returns) ---
    mu_real, cov_real = historical_mu_cov(
        test_prices[tickers], periods_per_year, cov_method=config["cov_method"], n_factors=config["n_factors"],
    )
    w_realized = optimize_max_sharpe(mu_real, cov_real, rf, weight_bounds)

    # Apply all three weight vectors to the SAME actual realized returns
    portfolios = {"Historical-based": w_historical, "Forecast-based": w_forecast, "Realized-optimal": w_realized}
    test_benchmark_returns = (
        compute_returns(test_prices[[BENCHMARK_TICKER]])[BENCHMARK_TICKER] if BENCHMARK_TICKER in prices.columns else None
    )
    cost_bps = config["transaction_cost_bps"]
    realized_metrics = {}
    for name, w in portfolios.items():
        port_returns = portfolio_returns(test_returns, w)
        if cost_bps > 0:
            turnover = compute_turnover(w, None)  # starting from cash — establishing this position from scratch
            port_returns = apply_transaction_cost(port_returns, turnover, cost_bps)
        realized_metrics[name] = summarise_performance(port_returns, rf, periods_per_year, test_benchmark_returns)

    st.subheader(f"Out-of-sample comparison — last {horizon} periods")
    st.caption(
        "All three portfolios are weighted differently, but evaluated on the exact SAME actual "
        "price moves below — this isolates how much the ALLOCATION choice mattered."
    )

    metrics_df = pd.DataFrame(realized_metrics).T
    display_df = metrics_df.copy()
    for col in ["period_return", "annual_return", "annual_volatility", "max_drawdown", "var_95", "cvar_95"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.1%}")
    if "jensens_alpha" in display_df.columns:
        display_df["jensens_alpha"] = display_df["jensens_alpha"].map(lambda x: f"{x:+.2%}" if not pd.isna(x) else "—")
    for col in ["sharpe_ratio", "sharpe_se", "sortino_ratio", "calmar_ratio", "omega_ratio", "beta", "information_ratio",
                "treynor_ratio", "ulcer_index", "skewness", "kurtosis"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(_format_ratio)
    display_df = display_df.drop(columns=["n_periods"], errors="ignore")
    display_df = display_df.rename(columns={
        "period_return": f"Return ({horizon} periods, not annualised)",
        "annual_return": "Ann. return", "annual_volatility": "Ann. vol.", "sharpe_ratio": "Sharpe",
        "sharpe_se": "Sharpe SE", "sortino_ratio": "Sortino", "calmar_ratio": "Calmar", "omega_ratio": "Omega",
        "max_drawdown": "Max DD", "var_95": "VaR 95%", "cvar_95": "CVaR 95%",
        "beta": "Beta (vs SPY)", "information_ratio": "Info. Ratio", "treynor_ratio": "Treynor",
        "jensens_alpha": "Alpha (Jensen)", "ulcer_index": "Ulcer Index",
        "skewness": "Skew", "kurtosis": "Kurtosis",
    })
    st.dataframe(display_df, width='stretch')
    st.caption(
        "**Read the 'Ann. return' column with the raw window return next to it in mind**: "
        f"this comparison covers only {horizon} periods — annualising a short window's return "
        "amplifies it mathematically (a strong run over a few months can annualise to a triple-"
        "digit number that never actually occurred over a full year). The raw, non-annualised "
        "return shows what actually happened in this specific window."
    )
    st.caption(
        "Calmar = return per unit of worst drawdown lived through. Omega = full-distribution "
        "gain/loss ratio (catches skew Sharpe/Sortino can miss). Information Ratio = consistency "
        "of the edge over SPY. Treynor = return per unit of market (systematic) risk, vs Sharpe's "
        "total-risk denominator. Sharpe SE = approximate standard error of the Sharpe estimate "
        "itself (Lo, 2002) — a rough 95% range is Sharpe ± 1.96×SE; a wide range relative to the "
        "point estimate is the statistical reason a short-window Sharpe shouldn't be over-trusted."
    )

    # text=... + textposition="outside" so the exact value stays readable even
    # when a bar is visually flattened to near-zero next to a much larger one
    # — e.g. Historical/Forecast Sharpe near 0 sitting next to Realized-optimal
    # (a hindsight-perfect portfolio, which routinely dwarfs a real backtest on
    # a short window). The scale gap itself is real, not a chart bug — this
    # just keeps the chart informative regardless of it.
    sharpe_values = [m["sharpe_ratio"] for m in realized_metrics.values()]
    sortino_values = [m["sortino_ratio"] for m in realized_metrics.values()]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(realized_metrics.keys()), y=sharpe_values, name="Sharpe",
        text=[f"{v:.2f}" for v in sharpe_values], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=list(realized_metrics.keys()), y=sortino_values, name="Sortino",
        text=[f"{v:.2f}" for v in sortino_values], textposition="outside",
    ))
    fig.update_layout(barmode="group", height=380, yaxis_title="Ratio")
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Realized-optimal is a HINDSIGHT-perfect portfolio (optimised on the actual realised "
        "returns, after the fact) — it routinely dwarfs a real backtest on a short window like "
        "this one, which is why its bars can look wildly out of scale next to Historical/"
        "Forecast-based. That's expected, not a chart bug: the value labels above each bar show "
        "the exact number even when the bar itself is too small to see."
    )

    gap = realized_metrics["Forecast-based"]["sharpe_ratio"] - realized_metrics["Historical-based"]["sharpe_ratio"]
    ceiling = realized_metrics["Realized-optimal"]["sharpe_ratio"] - realized_metrics["Historical-based"]["sharpe_ratio"]
    st.info(
        f"Forecasting moved realised Sharpe by **{gap:+.2f}** vs. the historical-based portfolio "
        f"(the theoretical ceiling, if the forecast had been perfect, was **{ceiling:+.2f}**). "
        "A small or negative gap here is the expected, honest result for short-horizon price "
        "forecasting — see the caveat below."
    )
    st.caption(
        "⚠️ Per the efficient market hypothesis, short-horizon equity price forecasts carry very "
        "limited genuine predictive power — most of a forecast model's apparent skill on any single "
        "backtest window can be noise. Treat the 'Forecast-based' column as a methodology demonstration, "
        "not a signal to act on."
    )

    st.divider()
    render_walk_forward_section(prices, tickers, config)

    return realized_metrics["Historical-based"], realized_metrics["Forecast-based"], realized_metrics["Realized-optimal"]


def render_walk_forward_section(prices: pd.DataFrame, tickers: list[str], config: dict) -> None:
    """
    Multi-window validation of the single comparison above. Separated into its
    own function (rather than inlined) because it's conceptually a second,
    independent analysis — same three portfolio types, but judged across several
    expanding windows instead of one — and keeping it separate makes that clear
    in the code, not just in the UI.
    """
    st.subheader("Walk-forward validation (multi-window)")
    st.caption(
        "The comparison above uses ONE held-out window — which can be lucky or unlucky. "
        "This repeats it across several expanding windows (each refit uses all data available "
        "up to that point) to check whether any edge holds up, or was a one-off."
    )

    horizon = config["forecast_horizon"]
    n_windows = config["walk_forward_windows"]
    periods_per_year = FREQUENCY_TO_PERIODS_PER_YEAR[config["frequency"]]

    possible_windows = generate_expanding_windows(len(prices), horizon, WALK_FORWARD_MIN_TRAIN_PERIODS, n_windows)
    if len(possible_windows) < MIN_WALK_FORWARD_WINDOWS:
        st.warning(
            f"Not enough history for {MIN_WALK_FORWARD_WINDOWS}+ walk-forward windows at this "
            f"horizon ({len(possible_windows)} possible) — widen the date range or shorten the "
            "forecast horizon in the sidebar."
        )
        return

    if config["forecast_model"] == "ARIMA (auto order)" and len(tickers) * len(possible_windows) > 20:
        st.caption("⏳ ARIMA across many windows/assets can take 20-30s — ETS is much faster if you're iterating.")

    with st.spinner(f"Running {len(possible_windows)} walk-forward windows..."):
        results = run_walk_forward(
            prices, tickers, horizon, n_windows, config["forecast_model"],
            config["risk_free_rate"], periods_per_year, WALK_FORWARD_MIN_TRAIN_PERIODS,
            config["max_weight_per_asset"], config["allow_short_selling"], config["transaction_cost_bps"],
            config["cov_method"], config["n_factors"],
            frequency=config["frequency"],
        )

    # Split into sub-tabs — distribution, summary, period-over-period, and the
    # raw table each answer a different question, and stacking all four
    # vertically would mean a lot of scrolling to reach whichever one a reader
    # actually wanted. Each sub-tab below is self-contained and independently
    # scannable, same pattern already used for the Overview tab's own
    # Macro & Risk / Prices & Analytics split.
    tab_distribution, tab_summary, tab_period, tab_raw = st.tabs(
        ["📊 Distribution", "📋 Summary", "🔄 Period-over-period", "📄 Raw data"]
    )

    with tab_distribution:
        fig = go.Figure()
        for portfolio in PORTFOLIO_TYPE_ORDER:
            subset = results[results["portfolio"] == portfolio]
            fig.add_trace(go.Box(y=subset["sharpe_ratio"], name=portfolio, boxpoints="all"))
        fig.update_layout(yaxis_title=f"Sharpe ratio across {len(possible_windows)} windows", height=380)
        st.plotly_chart(fig, width='stretch')

        win_rate = forecast_win_rate(results)
        if not pd.isna(win_rate):
            st.info(
                f"Forecast-based beat Historical-based on Sharpe in **{win_rate:.0%}** of the "
                f"{len(possible_windows)} windows tested. Close to 50% is the honest expectation for "
                "short-horizon price forecasting (see the caveat above) — a rate consistently well "
                "above 50% across many windows would be the real signal that the forecasting step "
                "is adding value, not just noise from a single lucky split."
            )

    with tab_summary:
        summary = results.groupby("portfolio")[["sharpe_ratio", "sortino_ratio", "annual_return"]].agg(["mean", "std"])
        summary.columns = [f"{a} ({b})" for a, b in summary.columns]
        st.dataframe(summary.style.format("{:.2f}"), width='stretch')
        st.caption(
            "Mean/std of each metric across every window, per portfolio type — does forecasting "
            "help ON AVERAGE, and how consistently? See the Distribution tab for the full spread "
            "behind these two summary numbers."
        )

    with tab_period:
        st.caption(
            "Each portfolio type's own trajectory, one window at a time: the CURRENT window's "
            "value next to the SAME portfolio type's value from the window immediately before "
            "it — so an improving or worsening trend is visible without cross-referencing two "
            "separate rows yourself. The first window for each portfolio type has no prior "
            "period to compare against, shown as a blank cell rather than a fabricated zero."
        )
        comparison = compare_to_previous_period(results)
        display_columns = ["window", "window_end", "portfolio"]
        column_config = {
            # All this app's frequencies (daily/weekly/monthly/yearly) place
            # bars at midnight -- the time component is always 00:00:00 and
            # never informative, so this column is rendered as a plain date.
            "window_end": st.column_config.DateColumn("window_end", format="YYYY-MM-DD"),
        }
        for metric in ["annual_return", "annual_volatility", "sharpe_ratio", "sortino_ratio", "max_drawdown"]:
            if metric not in comparison.columns:
                continue
            display_columns += [f"{metric}_previous", metric]
            column_config[metric] = st.column_config.NumberColumn(f"{metric} (current)", format="%.3f")
            column_config[f"{metric}_previous"] = st.column_config.NumberColumn("← previous", format="%.3f")
        st.dataframe(
            comparison[display_columns], width='stretch', hide_index=True, column_config=column_config,
        )

    with tab_raw:
        st.dataframe(
            results, width='stretch', hide_index=True,
            column_config={"window_end": st.column_config.DateColumn("window_end", format="YYYY-MM-DD")},
        )
        st.caption("Every window's full set of metrics, one row per (window, portfolio type) — the same data the other three tabs summarise, filter, or reshape.")


def render_ai_analyst_tab(
    weights: pd.Series, hist_metrics: dict, fcst_metrics: dict | None, real_metrics: dict | None,
    tickers: list[str], macro_context: dict | None = None,
) -> None:
    st.subheader("AI Portfolio Analyst")

    if not LLM_SETTINGS.groq_api_key:
        st.warning(
            "No GROQ_API_KEY configured — this tab will try a local Ollama fallback "
            "(`ollama serve` running on this machine). Set GROQ_API_KEY in your .env for "
            "the hosted, faster path."
        )

    results_context = build_results_context(weights, hist_metrics, fcst_metrics, real_metrics, macro_context)
    st.session_state.setdefault("results_context", results_context)
    st.session_state["results_context"] = results_context  # always keep the latest run's numbers

    # Streamlit state is ephemeral (nothing survives a refresh or a new session), so a run's
    # numbers are otherwise gone the moment the tab is closed. This reuses the exact same
    # results_context string already built for the LLM above — one source of truth, no LLM
    # or API key required, so it works even with zero keys configured.
    st.download_button(
        "Download run summary (.md)",
        f"# Portfolio Forecasting — Run Summary\n\nGenerated {dt.datetime.now():%Y-%m-%d %H:%M}\n\n{results_context}\n",
        file_name=f"portfolio_run_{dt.datetime.now():%Y%m%d_%H%M}.md",
        mime="text/markdown",
    )

    col_commentary, col_news = st.columns(2)

    with col_commentary:
        st.markdown("**Analyst commentary**")
        if st.button("Generate commentary", key="gen_commentary"):
            with st.spinner("Thinking..."):
                try:
                    text, backend = generate_commentary(results_context)
                    st.session_state["commentary"] = (text, backend)
                except LLMUnavailableError as exc:
                    st.error(f"LLM unavailable: {exc}")
        if "commentary" in st.session_state:
            text, backend = st.session_state["commentary"]
            st.write(text)
            st.caption(f"Generated by: {backend}")

    with col_news:
        st.markdown("**News digest**")
        # RAG's news corpus is session-only by construction (rebuilt from
        # scratch every time "Fetch & summarise news" is clicked) unless a
        # persisted copy exists — see rag.save_chunks/load_chunks for the
        # full opt-in/fails-soft contract. With no REDIS_URL configured this
        # block is a silent no-op; with one configured, a corpus already
        # fetched for this exact ticker set (this session or a previous one)
        # is available to the chatbot immediately, without forcing a fresh
        # fetch first.
        if "news_chunks" not in st.session_state:
            persisted = load_chunks(tickers)
            if persisted:
                st.session_state["news_chunks"] = persisted
                st.caption(
                    f"↺ {len(persisted)} previously-fetched news/filing item(s) for this ticker set "
                    "are already available to the chatbot (persisted via Redis). Click below to "
                    "refresh with current headlines and get a new commentary digest."
                )
        if st.button("Fetch & summarise news", key="gen_news"):
            with st.spinner("Fetching headlines..."):
                # No reliable free "ticker -> company name" API — a small static map
                # covers the brief's example universe; unmapped tickers just use the symbol.
                company_names = {
                    "AAPL": "Apple", "MSFT": "Microsoft", "TSLA": "Tesla",
                    "AMZN": "Amazon", "GOOG": "Google", "SPY": "S&P 500",
                    "QQQ": "Nasdaq", "TLT": "Treasury bonds", "GLD": "Gold", "VNQ": "REIT real estate",
                }
                digest, backend, articles, sentiment_by_ticker = generate_news_digest(tickers, company_names)
                st.session_state["news"] = (digest, backend, articles, sentiment_by_ticker)
                chunks = build_chunks(articles)
                st.session_state["news_chunks"] = chunks
                save_chunks(chunks, tickers)  # no-op without REDIS_URL — see rag.py
        if "news" in st.session_state:
            digest, backend, articles, sentiment_by_ticker = st.session_state["news"]
            st.write(digest)

            st.markdown("**Sentiment by ticker**")
            for ticker, s in sentiment_by_ticker.items():
                if s is None:
                    st.markdown(f"**{ticker}:** sentiment not available (no headlines fetched, or no source configured).")
                    continue
                score = s["score"]
                if score > 0.1:
                    label = "🟢 Bullish"
                elif score < -0.1:
                    label = "🔴 Bearish"
                else:
                    label = "⚪ Neutral"
                st.markdown(f"**{ticker}:** {label} (score {score:+.2f}, n={s['n_articles']}) — source: {s['provider']}")

            if articles:
                provider_counts = pd.Series([a["provider"] for a in articles]).value_counts()
                breakdown = ", ".join(f"{n} {p}" for p, n in provider_counts.items())
                st.caption(f"Generated by: {backend} · {len(articles)} items across {len(provider_counts)} sources ({breakdown})")
                with st.expander("Sources"):
                    for a in articles:
                        # Filing date included explicitly: two Form 4s (or 8-Ks)
                        # by the same insider/company render with an IDENTICAL
                        # title otherwise, making genuinely different filings
                        # look like visual duplicates in this list.
                        date_suffix = f" · {a['published_at']}" if a.get("published_at") else ""
                        st.markdown(f"- [{a['provider']}] [{a['title']}]({a['url']}) — *{a['source']}* ({a['ticker']}){date_suffix}")


def render_chatbot_tab() -> None:
    """
    Grounded Q&A, split into its own tab (rather than tacked onto the bottom of
    AI Analyst) — a chat interface deserves its own space rather than competing
    for attention below two other panels. Reads `results_context`/`news_chunks`
    from session_state, set by render_ai_analyst_tab on this run — if the user
    hasn't visited AI Analyst yet this run, results_context still exists (set
    once per run in render_ai_analyst_tab, called before this tab renders in
    `main()`'s tab order), so the chat is always grounded correctly regardless
    of which tab the user looks at first.
    """
    st.subheader("Ask about your results")
    st.caption("Grounded in the numbers computed above — won't invent figures that aren't there.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("e.g. Why is the Sortino ratio higher than the Sharpe ratio here?")
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            try:
                answer, backend = answer_portfolio_question(
                    question, st.session_state.get("results_context", ""), st.session_state["chat_history"][:-1],
                    st.session_state.get("news_chunks"),
                )
                st.session_state["chat_history"].append({"role": "assistant", "content": f"{answer}\n\n*Generated by: {backend}*"})
            except LLMUnavailableError as exc:
                st.session_state["chat_history"].append({"role": "assistant", "content": f"⚠️ LLM unavailable: {exc}"})
        # Cap history length — keep only the most recent turns (see
        # config.MAX_CHAT_HISTORY_MESSAGES's own comment for why this exists).
        st.session_state["chat_history"] = st.session_state["chat_history"][-MAX_CHAT_HISTORY_MESSAGES:]
        st.rerun()

    with st.expander("🔬 Search academic literature"):
        st.caption(
            "Look up real papers on a quant-finance methodology directly (via Semantic Scholar + "
            "arXiv) — the same sources the chatbot above pulls citations from automatically when "
            "your question names a technique it recognizes (Sharpe, GARCH, Ledoit-Wolf, ...)."
        )
        lit_query = st.text_input(
            "Search query", placeholder="e.g. Ledoit-Wolf shrinkage covariance estimation",
            key="academic_lit_query",
        )
        if st.button("Search papers", key="search_academic_lit") and lit_query.strip():
            detected_terms = detect_methodology_terms(lit_query)
            effective_query = METHODOLOGY_SEARCH_QUERIES[detected_terms[0]] if detected_terms else lit_query.strip()
            if detected_terms:
                st.caption(f"Recognised **{detected_terms[0]}** — searching as: _{effective_query}_")
            with st.spinner("Searching Semantic Scholar..."):
                papers = search_academic_papers(effective_query, limit=5)
            if not papers:
                st.caption("No results — try a broader or differently-worded query.")
            for paper in papers:
                authors = ", ".join(paper["authors"][:3]) + (" et al." if len(paper["authors"]) > 3 else "")
                year = f" ({paper['year']})" if paper.get("year") else ""
                citations = f" · {paper['citation_count']} citations" if paper.get("citation_count") else ""
                st.markdown(f"**{paper['title']}**{year}")
                st.caption(f"{authors or 'Unknown authors'}{citations}" + (f" · {paper['venue']}" if paper.get("venue") else ""))
                if paper.get("url"):
                    st.markdown(f"[View paper]({paper['url']})")


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main() -> None:
    st.title("📈 Portfolio Forecasting & Optimization")
    st.caption(
        "Mean-variance optimization, forecast-driven allocation, and an AI analyst — "
        "built as a technical case study."
    )

    config = render_sidebar()
    if not config["tickers"]:
        st.info("Pick at least one asset in the sidebar to get started.")
        return

    prices = load_data(config)
    if prices is None:
        return

    periods_per_year = FREQUENCY_TO_PERIODS_PER_YEAR[config["frequency"]]
    tab_overview, tab_frontier, tab_compare, tab_ai, tab_chat = st.tabs(
        ["Overview", "Efficient Frontier", "Forecast & Compare", "AI Analyst", "Chatbot"]
    )

    with tab_overview:
        sub_macro, sub_prices = st.tabs(["Macro & Risk", "Prices & Analytics"])
        with sub_macro:
            macro_context = render_macro_panel()
        with sub_prices:
            render_overview_tab(prices, config["tickers"], periods_per_year)

    with tab_frontier:
        weights = render_frontier_tab(
            prices, config["tickers"], config["risk_free_rate"], periods_per_year, config["max_weight_per_asset"],
            config["allow_short_selling"], config["cov_method"], config["n_factors"],
        )

    with tab_compare:
        result = render_forecast_compare_tab(prices, config["tickers"], config)
        hist_m, fcst_m, real_m = result if result else (None, None, None)

    with tab_ai:
        if hist_m is None:
            st.info("Run the Forecast & Compare tab first (needs a valid date range/horizon) to unlock AI commentary.")
        else:
            render_ai_analyst_tab(weights, hist_m, fcst_m, real_m, config["tickers"], macro_context)

    with tab_chat:
        if hist_m is None:
            st.info("Run the Forecast & Compare tab first (needs a valid date range/horizon) to unlock the chatbot.")
        else:
            render_chatbot_tab()


if __name__ == "__main__":
    main()