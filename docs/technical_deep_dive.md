# Technical deep dive

KPI formulas, how the pieces fit, the design choices and the known limits. For setup, commands and
screenshots see [../README.md](../README.md).

## The KPIs: formulas and how to read them

Computed in `src/metrics.py` (tests in `tests/test_metrics.py`) unless noted. `r` is the period
return series (daily, weekly, monthly or yearly, per the sidebar), `rf` the risk-free rate, `n` the
periods per year (252, 52, 12 or 1).

Returns, drawdowns, VaR and CVaR are decimal fractions (`-0.343` is a 34.3% loss), shown as
percentages in the UI. Two things are not on that scale: the Ulcer Index is in percentage points
(Martin's convention, `21.85` and not `0.2185`), and the ratios (Sharpe, Sortino, Calmar, Omega,
Beta, IR, Treynor) have no unit.

These tables define formulas, not results. Every number in the app is computed live from the
universe, dates and frequency you pick, and the repo ships no market data, so nothing here is a
measured performance figure. The example values below (a Sharpe of 5.89, a factor model explaining
35% vs 85% of variance) only show how to read a metric. The one measured number quoted in the docs is
the test count.

### 1. Return & risk: the building blocks

| Metric | Formula | How to read it |
|---|---|---|
| Annualised Return | `(∏(1+r))^(n/periods) − 1` | Geometric, not arithmetic mean × n: a ±50% sequence is 0%, not the naive average (classic trap). |
| Annualised Volatility | `std(r) × √n` | Total risk (upside + downside count equally). Not inherently "bad". See Sortino/Omega for asymmetric views. |
| Max Drawdown | `min[W(t)/max(1, W(1..t)) − 1]`, `W(t)=∏(1+r)`. The peak starts at the initial wealth of 1, so a loss in the first period counts. | Worst peak-to-trough loss; correlates most with an investor actually panic-selling. |
| Ulcer Index | RMS of the drawdown at *every* point (Peter Martin, 1987), **in percentage points**: `√(mean( (((W/max(1, cummax(W)))−1)×100)² ))` | Captures *duration* underwater, not just depth. Lower is better; 0 = never dipped below a prior peak. Reads on a 0-100 scale, unlike the fractional Max Drawdown directly above it: `[0, −0.10, 0]` scores `8.16`, not `0.0816`. |
| VaR 95% | 5th percentile of the return distribution (non-parametric) | "Worst 1-in-20 period loss exceeded X%". Says nothing about how much worse those periods got. |
| CVaR 95% | `mean(r \| r ≤ VaR_95)` | Average loss beyond VaR. Always at least as bad; a large VaR/CVaR gap flags a fat, dangerous tail. |

### 2. Risk-adjusted return ratios: reward per unit of risk

| Metric | Formula | How to read it |
|---|---|---|
| Sharpe | `(mean(r−rf_period)/std(r−rf_period)) × √n` | Above 1 generally good, above 2 very good. Penalises upside volatility exactly as much as downside. |
| Sharpe SE | `√((1+0.5×SR_period²)/n_obs)`, annualised by `×√n` (Lo, 2002) | Standard error of the Sharpe estimate itself: a rough 95% range is `Sharpe ± 1.96×SE`. Puts a real number behind "a Sharpe of 5.89 isn't reliable": measured on **30 daily periods** that Sharpe carries `SE ≈ 3.00`, i.e. a 95% range of roughly `0.02` to `11.76`; an illustration of the formula (it depends only on the Sharpe and the period count, not on any particular asset), not a result from this app. |
| Sortino | Same, denominator = downside deviation only (`√(mean(min(r−rf_period, 0)²))`, averaged over **all** periods, per Sortino & Price: averaging over losing periods only would overstate it) | Sortino ≥ Sharpe is normal for equities: only downside swings count against it. |
| Calmar | `annual_return / \|max_drawdown\|` | Penalises only the single *worst* outcome lived through, not the whole spread: the number a risk committee asks for. |
| Omega | `Σ(r−threshold \| r>threshold) / \|Σ(r−threshold \| r<threshold)\|` | Uses the *entire* empirical distribution, so it diverges from Sharpe/Sortino exactly when returns are skewed/fat-tailed. `∞` (shown `—`) = zero losing periods in the sample. |
| Skewness / Kurtosis | 3rd / 4th standardised moments of `r` | Skew: 0 symmetric, negative = fatter *left* tail (large losses, the typical equity shape). Kurtosis (pandas excess convention): 0 = normal tails, positive = fatter than normal. |

### 3. Benchmark-relative metrics: vs. SPY

Computed whenever a benchmark series is available (SPY is always fetched in the background
regardless of your ticker selection, specifically so these are always computable).

| Metric | Formula | How to read it |
|---|---|---|
| Beta | `Cov(r, r_benchmark) / Var(r_benchmark)` | 1.0 moves with the market; >1.0 amplifies; <1.0 dampens; negative = hedge-like (rare). |
| Information Ratio | `annualised(mean(r−r_benchmark)) / (std(r−r_benchmark)×√n)` | *Consistency* of outperformance vs. SPY, not just its size: what active-mandate reviews lead with, not Sharpe. |
| Treynor | `(annual_return − rf) / beta` | Return per unit of *systematic* (market) risk, vs. Sharpe's total-risk denominator. |
| Jensen's Alpha | `annual_return − [rf + beta×(annual_benchmark_return − rf)]`, both returns annualised over the dates the portfolio and benchmark share | Return above what CAPM predicts given the portfolio's own beta, what a CAPM-literate interviewer asks for the moment Beta is on screen. |

### 4. Optimizer outputs: expected, not realized (Efficient Frontier tab)

These come from the optimizer's own inputs (historical mean/covariance, Ledoit-Wolf shrinkage),
the optimizer's *target*, not a guarantee. Compare against the *realized* metrics in Forecast &
Compare to see the gap between expectation and outcome.

- **Expected return** `w·μ`, **expected volatility** `√(w·Σ·w)`, **expected Sharpe** `(expected_return−rf)/expected_volatility`.
- **Diversification ratio** = weighted-average individual asset volatility ÷ actual
  portfolio volatility, using **absolute** weights (so a short position still contributes its own
  volatility to the numerator rather than cancelling against a long). >1 whenever correlations are
  below 1 (the normal case), the numeric version of the Overview tab's correlation matrix.
  =1 means diversification buys nothing.
- **Concentration / HHI** = `Σ w_i²`. Ranges `1/N` (equal-weighted) to `1.0` (single
  asset): a quick check that the sidebar's max-weight cap is actually doing its job.

### 5. Forecast validation

**Forecast win rate** (walk-forward section, Forecast & Compare tab): across every expanding
window, the fraction where the forecast-based portfolio's *realized* Sharpe beat the
historical-based one's (`backtesting.forecast_win_rate`, a strict `>`: a tie counts as a loss).
Near 50% across many windows is the honest, expected result for short-horizon price forecasting
(consistent with the efficient market hypothesis). A rate consistently well above 50% would be
the real signal of a genuine edge. The denominator is however many windows the run actually
produced: at the defaults that is 5 windows (sidebar range 3-8), so a single window moves the
rate by 20 percentage points and the usual quantisation applies: 0%, 20%, 40%, 60%, 80%, 100%
are the only reachable values. Don't read much into a single window (see "The three-portfolio comparison and the walk-forward check" below).

**Forecast MASE** (`forecast_mase` column of the walk-forward results, `forecast_mase_mean` in
MLflow): the mean absolute scaled error of the price forecast itself, not of a portfolio. It's
the forecast's mean absolute error over the test window divided by the mean absolute error of a
one-step naive forecast on the training data (`mean(|y_t - y_(t-1)|)`, the non-seasonal
definition in [fpp3, section 5.8](https://otexts.com/fpp3/accuracy.html)). Below 1, the forecast
beat that yardstick; above 1, it didn't. A 20-step forecast is up against one-step naive errors,
so values above 1 are normal here. Each asset is scaled by its own history, then the window's
value is the mean over assets (`metrics.mean_absolute_scaled_error`, `backtesting._window_mase`).

**Forecast fan chart** (Forecast & Compare tab, up to four assets at a time via a selector): the
point forecast that feeds the optimizer hides how much uncertainty compounds over the horizon;
the shaded band (95% interval, `±1.96 × band`, widening with `√horizon`) makes that concrete:
further-out points are genuinely less reliable, not just "the same trend, continued."

The bands are **not all computed the same way**, and they are not all on the same scale. Worth
knowing before comparing one model's band against another's:

| Model | What the band is scaled by | Scale |
|---|---|---|
| Naive | std dev of historical price *differences* × `√t` | currency (price units) |
| ETS | std dev of the fit's residuals on price *levels* × `√t` | currency |
| Theta | std dev of the SES fit's one-step residuals × `√t` | currency |
| ARIMA | `statsmodels`' own forecast variance (`get_forecast().conf_int(alpha=0.05)`) | currency: no `√t` factor is applied by this repo; the widening comes from the model's own variance |
| ML regression | **in-sample** residual std dev of predicted *returns* × the forecast price path × `√t` | relative (a fractional error, converted to currency by the price level) |
| LSTM | **training** residual std dev of predicted *returns* × the forecast price path × `√t` | relative, same as ML regression |

The last two are the ones to be careful with: an in-sample residual understates the error a model
makes on data it has never seen, so those two bands are **optimistic**: narrower than a genuine
out-of-sample band. `forecasting.py` exports exactly those two as `IN_SAMPLE_BAND_MODELS` so the
UI can label them where the band is drawn. Walk-forward validation (below) is what measures these
two models honestly, out of sample.

### 6. Macro context (FRED; shapes interpretation, not portfolio-specific)

| Series | What it signals |
|---|---|
| VIX | "Fear gauge": <15 calm, 15–25 normal, >25 elevated stress. |
| 10Y–3M Treasury term spread | Negative has preceded every US recession since the 1960s (some false positives): context, not a trading signal. |
| Risk-free rate (3M T-bill) | `rf` in every ratio above that needs one, overridable by hand in the sidebar. |
| CPI YoY, unemployment rate, Fed funds rate | The rest of the Fed's dual mandate plus the actual policy rate: the yield curve and VIX alone don't cover this. |
| Sahm Rule recession indicator (`SAHMREALTIME`) | ≥0.50 (3-month avg. unemployment 0.50pt above its 12-month low) has coincided with every US recession's start since 1970, no false positives to date. |
| Baa corporate credit spread vs 10Y (`BAA10Y`) | Corporate credit-risk appetite: a different stress channel from the yield curve or VIX. |
| Real GDP growth (quarterly, `A191RL1Q225SBEA`) | Headline "GDP grew at X% annualized". Updates far less often than the rest of the panel. |
| Industrial Production YoY (`INDPRO`) | Proxy for ISM Manufacturing PMI (paid/proprietary, no free API): real output data capturing the same manufacturing-momentum signal. |

Every series above links to its FRED source page from a "Sources" expander under the macro panel.

### 7. Time-series diagnostics (Overview tab, per asset: series structure, not performance)

The kind of check done *before* trusting a forecast, not after.

- **ADF stationarity test** (on returns): validates `forecasting.py`'s own ARIMA `d=1`
  first-differencing choice. "Yes" (p<0.05) is the expected, textbook result for returns (unlike
  price *levels*, non-stationary by construction).
- **Hurst exponent** (on prices, variance-of-lagged-differences estimator; a documented
  simplification, not full rescaled-range analysis): >0.55 trending/momentum, <0.45
  mean-reverting, ≈0.5 random walk. Most liquid large-caps sit close to 0.5.
- **Rolling Sharpe ratio** (window auto-sized to sample length): a single end-of-sample Sharpe can
  hide a regime change (calm-then-crisis or the reverse); the rolling version shows how it actually
  evolved.

### 8. Fundamentals (data points, not ratios)

Per-ticker (Overview tab, fetched on demand): market cap, trailing/forward P/E, beta (the
provider's own calculation; may differ slightly from this app's `beta_vs_benchmark` above due to
lookback/methodology differences), dividend yield, 52-week range. Finnhub tried first, then Twelve Data and
yfinance as gap-fillers (see "Data and LLM fallbacks" below). The table's **Source** column shows which
provider answered for each ticker.

## Repository structure

```
app.py                      Streamlit UI, orchestration only (no finance or LLM logic)
api/main.py                 FastAPI wrapper over the same src/ functions
src/
  academic_search.py        Semantic Scholar + arXiv search, grounds methodology answers in real papers
  ai_features.py            commentary, news digest (parallel per ticker), chatbot prompts
  backtesting.py            walk-forward validation of the 3-portfolio comparison
  cache.py                  optional Redis cache, falls back to @st.cache_data without REDIS_URL
  config.py                 defaults, env vars, constants (single source of truth)
  factor_data.py            Fama-French 3/5-factor data and exposure regression
  factor_models.py          PCA factor model covariance for wide universes
  forecasting.py            naive / ETS / Theta / ARIMA / ML regression / LSTM, parallel across tickers
  http_utils.py             secret redaction and the "GET, expect JSON, fail soft" helper
  llm_client.py             Groq (key rotation) -> OpenRouter -> SambaNova -> Ollama
  macro_data.py             FRED: risk-free rate and macro panel
  market_data.py            yfinance + fallback chain, fundamentals
  metrics.py                pure finance math
  news_data.py              6 news/filing sources and the sentiment cascade
  optimization.py           PyPortfolioOpt wrapper: max Sharpe, efficient frontier
  rag.py                    TF-IDF retrieval over the news corpus, optional Redis persistence
  timeseries_diagnostics.py ADF, Hurst exponent, rolling Sharpe
  volatility_forecasting.py GARCH(1,1) volatility and forecast covariance
tests/                      one test_<module>.py per module, all network mocked
scripts/                    benchmark_walk_forward.py, check_env.py, track_forecasts.py (MLflow)
evals/                      chatbot_groundedness.py (LLM judge, run by hand)
docs/                       decisions.md, mistakes_log.md, this file
```

`metrics.py`, `forecasting.py` and `optimization.py` have no Streamlit or LLM dependency, so they
stay testable and reusable in a notebook. `llm_client.py` is the only module that talks to an LLM
provider, which is what makes the fallback chain a config change.

## Design choices against the brief

| Brief suggests | Used instead | Why |
|---|---|---|
| Kats | statsmodels (ARIMA, ETS, Theta) plus a naive baseline, an ML regression and a small LSTM | Kats' last PyPI release is 0.2.0 from 15 March 2022 ([history](https://pypi.org/project/kats/#history)), older than four years of pandas/numpy releases |
| Riskfolio-Lib | PyPortfolioOpt | Covers max-Sharpe, min-vol and the frontier with the lightest dependencies. If the brief needs Black-Litterman or CVaR-native optimisation, skfolio is the one to migrate to: its walk-forward and covariance estimators overlap with `backtesting.py` and `factor_models.py` |
| GitHub Pages | Streamlit Community Cloud, Render as backup | Pages serves static files only and cannot run a Streamlit server |
| ISM Manufacturing PMI | Industrial Production (FRED `INDPRO`) | The PMI is paid and proprietary. Industrial Production is real output data with the same manufacturing-momentum signal |
| Airflow | none | The app is interactive and synchronous, there is no scheduled batch to orchestrate |
| LangChain / LangGraph | `llm_client.py` | Every LLM use is a single call with injected context. No agent chooses tools, so a framework would only hide a simple fallback loop |

## Data and LLM fallbacks

**Market data, five steps.** `fetch_adjusted_close` tries each step only if the previous one failed:
yfinance (3 attempts with backoff), the direct Yahoo REST API, Tiingo (500 req/hour free), Twelve Data
(800 req/day), Alpha Vantage (25 req/day, one ticker per call, so last). The direct Yahoo call sits
behind the same anti-bot layer as the library; it is cheap insurance, not a reliable fix.

When Yahoo answers only some tickers of a batch (a rate limit leaves `TSLA` as an all-NaN column),
`_fill_missing_tickers` asks the later sources for the missing symbols only. The Overview tab then
captions who served what (`yfinance + tiingo for TSLA`, via `price_source_notice`).

**Circuit breaker.** Once both Yahoo paths fail, Yahoo is skipped for 3 minutes. Streamlit reruns the
script on every widget change, so without it each click would replay the Yahoo retries (3 yfinance
attempts, then one direct call per ticker). It is process-wide on purpose: Yahoo being down is true for
every session, and a per-session breaker would make each new session pay the retries again.

**LLM.** Up to 5 Groq keys (`GROQ_API_KEY`, `_2` to `_5`) are tried in order. The next key is used only
when that key is at fault: a 429 rate limit or a 401 (revoked or mistyped key). Any other error (a
deprecated model, the network) goes straight to the next tier, since every key would fail the same way.
After Groq come OpenRouter and SambaNova (whichever have a key; both speak the OpenAI chat protocol, so
`_call_openai_compatible_provider` serves both), then local Ollama. Ollama alone is no fallback in
production: Render and Streamlit Cloud have no Ollama daemon.

**Risk-free rate.** With `FRED_API_KEY` the sidebar slider starts at the live 3-month T-bill yield
(`DGS3MO`), still editable by hand.

**Fundamentals.** Twelve Data's `/statistics` is restricted to `AAPL` on the free tier (every other
ticker returns a 403). So `fetch_fundamentals` tries Finnhub first, Twelve Data only if Finnhub is
missing or empty, and yfinance (no key) if neither answers. When the primary answers with gaps
(`forward_pe` especially), Twelve Data and then yfinance fill those fields one by one. A value already
present is never overwritten, `dividend_yield` is never taken from yfinance (its unit has changed
between releases), and `field_sources` records the provider of each field. The app lists the filled-in
fields under the table, and the Source column shows the blend (`Finnhub + yfinance`). A Twelve Data 403
is remembered for the process and only reaches the UI if yfinance has nothing either. Finnhub quirks:
market cap comes in millions (multiplied by 1e6) and dividend yield is a percentage number (divided by
100). Field names come from the providers' public docs, not a live key here: if a field looks wrong,
`curl` the endpoint before blaming the parser.

**Sentiment.** FinBERT through the Hugging Face Inference API (needs `HUGGINGFACE_API_KEY`) is tried
first, then Finnhub's aggregated `/news-sentiment` (skipped for the process once a 401 or 403 shows the
plan lacks it), then VADER computed locally on the fetched headlines, which cannot fail. Each reading
is tagged with the tier that produced it, and the UI says "not available" instead of showing a blank,
which would read as neutral.

## Forecasting models

`FORECAST_MODELS` in `forecasting.py` is the single list: the sidebar and the benchmark script both
read from it.

| Model | What it does |
|---|---|
| Naive (random walk) | Last price plus the average historical drift. The baseline every other model must beat |
| ETS (Holt linear trend) | Exponential smoothing with a damped linear trend |
| Theta | SES of the prices plus a drift of half the OLS slope (Hyndman & Billah, 2003), no seasonal step. Matches statsmodels' `ThetaModel(deseasonalize=False)` to 1e-8 (`tests/test_forecasting.py`) |
| ARIMA (auto order) | ARIMA(p,1,q) with an AIC grid search, d fixed at 1 because price levels are non-stationary. Slower, and prone to overfit short series |
| ML regression | Gradient-boosted trees (xgboost if installed, scikit-learn otherwise) on lagged returns plus their rolling mean and std, forecasting one step at a time |
| LSTM | A small recurrent network trained fresh per asset and per call (no universal price model exists to load) |

Naive, ETS, Theta, ARIMA and GARCH are classical statistics. ML regression, the LSTM, the PCA
covariance and FinBERT are machine learning. TF-IDF retrieval and VADER are NLP, not learned models.

## The three-portfolio comparison and the walk-forward check

1. **Historical-based**: max-Sharpe weights from the mean and covariance of the training window only.
2. **Forecast-based**: same optimizer, expected returns from the forecast model, covariance historical
   (or GARCH-forecast if chosen in the sidebar).
3. **Realized-optimal**: weights fitted on the held-out window's actual returns, the hindsight benchmark.

All three weight vectors are applied to the same actual returns of the held-out window, so the
comparison isolates the allocation choice. The test window includes the last training price as the
anchor `pct_change()` needs, so the metrics table's `n_periods` and the header's "last {horizon}
periods" agree exactly.

One split can be lucky, so the walk-forward section (`backtesting.py`) repeats the comparison over
expanding windows: each refit uses everything known up to that point and is scored on the next
`horizon`-sized slice. It reports a box plot of Sharpe per portfolio, a mean and sample-std summary
(`summarise_walk_forward` reports a single-window std as 0.0, not NaN; the UI needs 3 to 8 windows, 5 by
default), a period-over-period table, and the forecast win rate. Walk-forward compares strategies on
historical data over sequential windows. It is not A/B testing, which splits live users at random.
ARIMA over many windows and assets is noticeably slower; ETS, Theta or naive are the practical choice
for iterating.

## Universe and covariance

Presets: the 5 default equities (AAPL, MSFT, TSLA, AMZN, GOOG), 15 mega caps, or 11 GICS sectors
(7 to 15 names each, 107 tickers). An ETF sleeve adds other risk drivers: SPY/QQQ (equity indices), TLT
(long Treasuries), GLD/SLV (metals), USO/DBC (energy and commodities), UUP (dollar, as an ETF proxy so it
stays optimisable long-only) and VNQ (REITs).

It stops at 107 names, not 500: a covariance matrix has N(N+1)/2 free parameters (5,778 for 107
assets) against a few thousand daily observations. Ledoit-Wolf shrinkage (the default) is a real fix
for a small universe but does not remove that problem. The PCA factor model does: returns are driven
by a few common factors plus asset-specific noise, so the parameter count falls to about N x k + N. It
is a statistical factor model (factors extracted from the returns, orthogonal by construction), not a
Barra-style fundamental one with pre-specified value or size factors. Use Ledoit-Wolf for 5 to 15
tickers and PCA from about 40. The choice goes through `optimization.historical_mu_cov`, so the
frontier, the comparison and the walk-forward always use the same estimator, and the Efficient Frontier
tab shows the cumulative variance the chosen factors explain (see `factor_models.py`).

## Short selling, costs, RAG and citations

- **Short selling.** "Allow short selling" (off by default) makes the weight bounds symmetric around
  the max-weight cap. Weights still sum to 100%, so there is no gross leverage. `resolve_weight_bounds`
  keeps every optimizer call consistent.
- **Transaction costs.** `turnover x cost rate` (default 10 bps) is charged at every rebalance: once
  for the initial trade in the single comparison, and at each walk-forward window boundary against that
  portfolio's own previous weights. Between rebalances a portfolio is bought and held, so its weights
  drift (`buy_and_hold_returns`) and the next trade is charged from the drifted weights. A shorter
  horizon means more windows, so more rebalancing and more cost drag. Set the cost to 0 for the
  frictionless comparison.
- **RAG for the chatbot.** `rag.py` indexes the news and filings collected in the AI Analyst tab with
  TF-IDF and cosine similarity (a small corpus does not justify a sentence-transformers dependency) and
  injects only the top 4 chunks relevant to the question. Portfolio metrics stay injected in full: a
  small, fixed set of numbers that must be complete. With `REDIS_URL` set, the corpus survives
  sessions and restarts; without it, it is rebuilt each session. A real vector store would only be
  worth it if the corpus grew into hundreds of permanent documents.
- **Citations.** When a question names a known methodology term (Sharpe, GARCH, Ledoit-Wolf,
  walk-forward ...), `academic_search.py` fetches real papers from Semantic Scholar (key optional) and
  arXiv (no key), deduplicated by title, and the prompt tells the model to cite only those. LLMs invent
  plausible-looking citations, which would undo the grounding the rest of the app provides.
- **Prompts.** Every injected data block is wrapped in a named XML tag (`<portfolio_data>`,
  `<retrieved_news>`, `<academic_references>`) so the instructions can point at one, and the system
  prompt's main rule (do not report an inflated annualised figure from a short window at face value)
  comes with a GOOD and a BAD example.

## Known limitations

- Short-horizon price forecasting has weak predictive power. A win rate near 50% is the expected
  result, not a bug, so read the Forecast-based portfolio as a methodology demo, never a signal. The
  UI says so in several places.
- Correlations stay historical (Ledoit-Wolf) even with the GARCH covariance: a true correlation
  forecast needs DCC-GARCH, which no well-maintained Python package implements. See
  `volatility_forecasting.py` for what is and is not forecast.
- `fetch_sec_filings` searches by company name, then keeps only hits whose `ciks` contain the ticker's
  real CIK (`_fetch_cik_for_ticker`, from `company_tickers.json`) and dedupes by accession number. With
  no CIK (an ETF, a non-US ticker such as MC.PA, a failed lookup) it returns no filings rather than
  another company's.
- Survivorship bias: tickers are picked today, so companies that failed or were delisted are absent and
  every backtest looks better than it would have in real time.
- Free-tier data: the free sources drop delisted tickers and offer adjusted close only, and several
  parsers (GDELT, TED, SEC EDGAR, Ken French) were written from public documentation, not checked
  against live responses in this environment.

## Next steps

1. Cloud deployment (Azure Container Apps or AKS, Key Vault for secrets, Application Insights,
   a staging slot).
2. Paid market data (Polygon, IEX, Bloomberg or Refinitiv): no rate limits, survivorship-bias-free
   history, proper corporate actions.
3. Stronger forecasting: a per-asset hyperparameter search, an ensemble weighted by each model's
   walk-forward record, and a test of whether a time-series foundation model beats naive out of sample.
4. A managed LLM tier with an SLA, and a regression suite for prompt changes.
5. Turn `benchmark_walk_forward.py` into a load-testing target (concurrent walk-forward runs, p95).
6. LangGraph only if the AI Analyst becomes a real multi-step agent.
