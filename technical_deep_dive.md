# Technical Deep Dive — Portfolio Forecasting & Optimization

Full technical write-up: KPI formulas, repo structure, every design decision vs. the original
brief, all key challenges, known limitations, and the full roadmap. See [../README.md](../README.md)
for the business-first summary, results, and screenshots.

## Understanding the KPIs — formulas & how to read them

Computed in `src/metrics.py` (unit-tested in `tests/test_metrics.py`) unless noted otherwise.
`r` = period return series (daily/weekly/monthly per the sidebar), `rf` = risk-free rate,
`n` = periods per year (252/52/12).

### 1. Return & risk — the building blocks

| Metric | Formula | How to read it |
|---|---|---|
| Annualised Return | `(∏(1+r))^(n/periods) − 1` | Geometric, not arithmetic mean × n — a ±50% sequence is 0%, not the naive average (classic trap). |
| Annualised Volatility | `std(r) × √n` | Total risk (upside + downside count equally). Not inherently "bad" — see Sortino/Omega for asymmetric views. |
| Max Drawdown | `min[W(t)/max(W(0..t)) − 1]`, `W(t)=∏(1+r)` | Worst peak-to-trough loss — correlates most with an investor actually panic-selling. |
| Ulcer Index | RMS of the % drawdown at *every* point (Peter Martin, 1987) | Captures *duration* underwater, not just depth. Lower is better; 0 = never dipped below a prior peak. |
| VaR 95% | 5th percentile of the return distribution (non-parametric) | "Worst 1-in-20 period loss exceeded X%" — says nothing about how much worse those periods got. |
| CVaR 95% | `mean(r \| r ≤ VaR_95)` | Average loss beyond VaR — always at least as bad; a large VaR/CVaR gap flags a fat, dangerous tail. |

### 2. Risk-adjusted return ratios — reward per unit of risk

| Metric | Formula | How to read it |
|---|---|---|
| Sharpe | `(mean(r−rf_period)/std(r−rf_period)) × √n` | Above 1 generally good, above 2 very good. Penalises upside volatility exactly as much as downside. |
| Sharpe SE | `√((1+0.5×SR_period²)/n)`, annualised (Lo, 2002) | Standard error of the Sharpe estimate itself — a rough 95% range is `Sharpe ± 1.96×SE`. Puts a real number behind "a Sharpe of 5.89 on 30 periods isn't reliable" instead of just an appeal to intuition. |
| Sortino | Same, denominator = downside deviation only (`√(mean((r−rf_period)² \| r−rf_period<0))`) | Sortino ≥ Sharpe is normal for equities — only downside swings count against it. |
| Calmar | `annual_return / \|max_drawdown\|` | Penalises only the single *worst* outcome lived through, not the whole spread — the number a risk committee asks for. |
| Omega | `Σ(r−threshold \| r>threshold) / \|Σ(r−threshold \| r<threshold)\|` | Uses the *entire* empirical distribution, so it diverges from Sharpe/Sortino exactly when returns are skewed/fat-tailed. `∞` (shown `—`) = zero losing periods in the sample. |
| Skewness / Kurtosis | 3rd / 4th standardised moments of `r` | Skew: 0 symmetric, negative = fatter *left* tail (large losses — the typical equity shape). Kurtosis (pandas excess convention): 0 = normal tails, positive = fatter than normal. |

### 3. Benchmark-relative metrics — vs. SPY

Computed whenever a benchmark series is available (SPY is always fetched in the background
regardless of your ticker selection, specifically so these are always computable).

| Metric | Formula | How to read it |
|---|---|---|
| Beta | `Cov(r, r_benchmark) / Var(r_benchmark)` | 1.0 moves with the market; >1.0 amplifies; <1.0 dampens; negative = hedge-like (rare). |
| Information Ratio | `annualised(mean(r−r_benchmark)) / (std(r−r_benchmark)×√n)` | *Consistency* of outperformance vs. SPY, not just its size — what active-mandate reviews lead with, not Sharpe. |
| Treynor | `(annual_return − rf) / beta` | Return per unit of *systematic* (market) risk, vs. Sharpe's total-risk denominator. |
| Jensen's Alpha | `annual_return − [rf + beta×(annual_benchmark_return − rf)]` | Return above what CAPM predicts given the portfolio's own beta — what a CAPM-literate interviewer asks for the moment Beta is on screen. |

### 4. Optimizer outputs — expected, not realized (Efficient Frontier tab)

These come from the optimizer's own inputs (historical mean/covariance, Ledoit-Wolf shrinkage) —
the optimizer's *target*, not a guarantee. Compare against the *realized* metrics in Forecast &
Compare to see the gap between expectation and outcome.

- **Expected return** `w·μ`, **expected volatility** `√(w·Σ·w)`, **expected Sharpe** `(expected_return−rf)/expected_volatility`.
- **Diversification ratio** = weighted-average individual asset volatility ÷ actual
  portfolio volatility. >1 whenever correlations are below 1 (the normal case) — the numeric
  version of the Overview tab's correlation matrix. =1 means diversification buys nothing.
- **Concentration / HHI** = `Σ w_i²`. Ranges `1/N` (equal-weighted) to `1.0` (single
  asset) — a quick check that the sidebar's max-weight cap is actually doing its job.

### 5. Forecast validation

**Forecast win rate** (walk-forward section, Forecast & Compare tab) — across every expanding
window, the fraction where the forecast-based portfolio's *realized* Sharpe beat the
historical-based one's. Near 50% across many windows is the honest, expected result for
short-horizon price forecasting (consistent with the efficient market hypothesis) — a rate
consistently well above 50% would be the real signal of a genuine edge. Don't read much into a
single window (see "Walk-forward validation" below for why).

**Forecast fan chart** (Forecast & Compare tab, single asset at a time via a selector) — the
point forecast that feeds the optimizer hides how much uncertainty compounds over the horizon; the
shaded band (95% confidence interval, widening with `√horizon` — every model in `forecasting.py`
computes this the same way, see `naive_forecast`'s own comment on the convention) makes that
concrete: further-out points are genuinely less reliable, not just "the same trend, continued."

### 6. Macro context (FRED — shapes interpretation, not portfolio-specific)

| Series | What it signals |
|---|---|
| VIX | "Fear gauge": <15 calm, 15–25 normal, >25 elevated stress. |
| 10Y–3M Treasury term spread | Negative has preceded every US recession since the 1960s (some false positives) — context, not a trading signal. |
| Risk-free rate (3M T-bill) | `rf` in every ratio above that needs one — overridable by hand in the sidebar. |
| CPI YoY, unemployment rate, Fed funds rate | The rest of the Fed's dual mandate plus the actual policy rate — the yield curve and VIX alone don't cover this. |
| Sahm Rule recession indicator (`SAHMREALTIME`) | ≥0.50 (3-month avg. unemployment 0.50pt above its 12-month low) has coincided with every US recession's start since 1970, no false positives to date. |
| Baa corporate credit spread vs 10Y (`BAA10Y`) | Corporate credit-risk appetite — a different stress channel from the yield curve or VIX. |
| Real GDP growth (quarterly, `A191RL1Q225SBEA`) | Headline "GDP grew at X% annualized" — updates far less often than the rest of the panel. |
| Industrial Production YoY (`INDPRO`) | Proxy for ISM Manufacturing PMI (paid/proprietary, no free API) — real output data capturing the same manufacturing-momentum signal. |

Every series above links to its FRED source page from a "Sources" expander under the macro panel.

### 7. Time-series diagnostics (Overview tab, per asset — series structure, not performance)

The kind of check done *before* trusting a forecast, not after.

- **ADF stationarity test** (on returns) — validates `forecasting.py`'s own ARIMA `d=1`
  first-differencing choice. "Yes" (p<0.05) is the expected, textbook result for returns (unlike
  price *levels*, non-stationary by construction).
- **Hurst exponent** (on prices, variance-of-lagged-differences estimator — a documented
  simplification, not full rescaled-range analysis) — >0.55 trending/momentum, <0.45
  mean-reverting, ≈0.5 random walk. Most liquid large-caps sit close to 0.5.
- **Rolling Sharpe ratio** (window auto-sized to sample length) — a single end-of-sample Sharpe can
  hide a regime change (calm-then-crisis or the reverse); the rolling version shows how it actually
  evolved.

### 8. Fundamentals (data points, not ratios)

Per-ticker (Overview tab, fetched on demand): market cap, trailing/forward P/E, beta (the
provider's own calculation — may differ slightly from this app's `beta_vs_benchmark` above due to
lookback/methodology differences), dividend yield, 52-week range. Finnhub tried first, Twelve Data
as fallback (see the source-chain note further below) — the table's **Source** column shows which
one actually answered for each ticker.



## Repository structure

```
portfolio-forecasting/
├── app.py                    # Streamlit UI — orchestration only, no finance/LLM logic
├── src/
│   ├── __init__.py
│   ├── academic_search.py       # Semantic Scholar + arXiv paper search — grounds methodological chatbot answers in real citations
│   ├── ai_features.py          # commentary / news digest (parallel per ticker) / chatbot — prompt logic lives here
│   ├── backtesting.py          # walk-forward (multi-window) validation of the 3-portfolio comparison
│   ├── cache.py                 # optional Redis-backed cache (+ public get_redis_client() for rag.py) — falls through to @st.cache_data with no REDIS_URL
│   ├── config.py               # single source of truth: defaults, env vars, constants
│   ├── factor_models.py        # PCA statistical factor model (covariance for wide universes)
│   ├── forecasting.py          # naive / ETS / Theta / ARIMA / LSTM price forecasting, parallel across tickers
│   ├── llm_client.py           # Groq (multi-key rotation) -> [OpenRouter, Cerebras, SambaNova] -> Ollama, one call site for all
│   ├── macro_data.py           # FRED: live 3-month T-bill rate, pre-fills the risk-free-rate slider
│   ├── market_data.py          # yfinance fetch + cache + frequency resampling + fallback chain
│   ├── metrics.py              # pure finance math: returns, Sharpe, Sortino, VaR, CVaR, drawdown, beta
│   ├── news_data.py            # NewsAPI/Finnhub/SEC EDGAR headlines per ticker (fails soft if no key)
│   ├── optimization.py         # PyPortfolioOpt wrapper: mean-variance, efficient frontier
│   ├── rag.py                  # TF-IDF retrieval over the news/filings corpus + optional Redis persistence
│   ├── timeseries_diagnostics.py  # ADF stationarity, Hurst exponent, rolling Sharpe
│   └── volatility_forecasting.py  # GARCH(1,1) forecasted volatility + historical correlation -> forecasted cov
├── tests/
│   ├── __init__.py
│   ├── test_academic_search.py  # Semantic Scholar + arXiv search: term detection, source merge/dedup, fails-soft, works without any key
│   ├── test_ai_features.py     # build_results_context formatting (shared by commentary + chatbot); academic-citation wiring
│   ├── test_backtesting.py     # expanding-window edge cases, end-to-end walk-forward smoke test, portfolio-order consistency
│   ├── test_cache.py           # Redis cache: no-REDIS_URL fallthrough, unreachable-Redis degradation
│   ├── test_factor_models.py   # PCA cov: symmetric/PSD, known factor-structure recovery, clamping
│   ├── test_forecasting.py     # short-history and non-convergence fallback paths, no statsmodels index FutureWarning, LSTM widening-band regression
│   ├── test_llm_client.py      # Groq->[hosted fallback list]->Ollama cascade, provider ordering, token-budget truncation
│   ├── test_macro_data.py      # FRED divide_by pitfall (Sahm Rule), YoY calculation, GDP/PMI-proxy fetch
│   ├── test_market_data.py     # fully mocked: Twelve Data partial-batch parsing, Yahoo circuit breaker, yfinance cache redirect
│   ├── test_metrics.py         # unit tests for the finance formulas (hand-checkable synthetic data)
│   ├── test_news_data.py       # SEC EDGAR filing-URL construction, CIK filtering, accession dedup
│   ├── test_news_sentiment.py  # FinBERT->Finnhub->VADER sentiment cascade + circuit breaker on 401/403 (not 429/5xx)
│   ├── test_optimization.py    # negative bounds, infeasible cap, degenerate mu<rf fallback
│   ├── test_rag.py             # TF-IDF retrieval + Redis persistence: round-trip, dedup, fails-soft on a broken Redis
│   ├── test_timeseries_diagnostics.py  # ADF stationary vs. random walk, Hurst trending vs. mean-reverting
│   └── test_volatility_forecasting.py  # GARCH fallback path, symmetric/PSD, correlation preserved from history
├── scripts/
│   └── benchmark_walk_forward.py  # capacity-planning benchmark for run_walk_forward (not gated in CI, run manually)
├── images/                    # screenshots referenced in this README
├── .github/workflows/ci.yml   # pytest + mypy --strict (blocking) + benchmark smoke test (non-blocking) on every push/PR
├── .streamlit/config.toml     # theme/UI config — tracked (unlike secrets.toml, see .gitignore)
├── .dockerignore
├── .env                       # your local keys — NEVER committed
├── .gitignore
├── BUGFIXES.md                # bug-fix log, kept separate from this README
├── RUNBOOK.md                 # operational playbook for external-dependency failures
├── build-docker.sh / deploy.sh / deploy.bat  # local build/run helpers (WSL, Linux/macOS, Windows)
├── docker-compose.yml         # local Docker run — optional `redis` service via `--profile with-redis`
├── Dockerfile                 # containerised run — also the base for Render's `runtime: docker`
├── env.example                # copy to .env and fill in your keys
├── render.yaml                # Render Blueprint (Infrastructure as Code)
└── requirements.txt
```

**Why this layout:** `src/metrics.py`, `src/forecasting.py`, and `src/optimization.py` have zero
Streamlit or LLM dependency — they're plain functions on DataFrames/Series, independently testable
and reusable outside the app (a notebook, a batch job). `app.py` only wires UI widgets to these
functions. `llm_client.py` is the single seam that talks to an LLM provider, which is what makes
the Groq→[OpenRouter, Cerebras, SambaNova]→Ollama fallback chain (and adding a fifth provider
later) a config-only change — see `llm_client.py`'s own docstring for why the hosted fallbacks
share one implementation instead of one near-identical function per provider.

## Key design decisions vs. the original brief

The brief names some tools I found to be a poor fit for this scope — documented here rather
than silently swapped:

| Brief suggests | Used instead | Why |
|---|---|---|
| **Kats** for forecasting | **statsmodels** (ARIMA, Holt-Winters ETS, the Theta method) + a naive random-walk baseline, plus a small from-scratch **LSTM** (PyTorch, CPU-only) as a fifth, heavier option | Kats' last PyPI release was **0.2.0 on 15 March 2022** — [pypi.org/project/kats/#history](https://pypi.org/project/kats/#history) — nothing published since, so it predates ~3.5 years of pandas/numpy releases. statsmodels is the actively-maintained, industry-standard alternative. The LSTM trains fresh per asset per call (no pretrained universal price model exists to load) — see `forecasting.lstm_forecast`'s own docstring for the honest speed trade-off against the other four. |
| **Riskfolio-Lib** for optimization | **PyPortfolioOpt** | Two alternatives were evaluated against PyPortfolioOpt, both rejected for the same reason: real capability this project's scope doesn't need, at a heavier dependency cost. **Riskfolio-Lib** ([pypi.org/project/Riskfolio-Lib](https://pypi.org/project/Riskfolio-Lib/)) is built on `cvxpy` with 12 convex risk measures, Black-Litterman, risk factors, tracking-error/turnover constraints. **skfolio** ([skfolio.org](https://skfolio.org/)) is newer (2026) and scikit-learn-native (`fit`/`predict`, `GridSearchCV`-compatible) — Mean-Risk, Risk Budgeting, Hierarchical Risk Parity, Black-Litterman, Ledoit-Wolf/Gerber/denoising covariance estimators, walk-forward cross-validation built in — genuinely closer to this project's own PCA-factor-model and walk-forward additions than Riskfolio-Lib is, but still pulls in `cvxpy` + `joblib` + its own `plotly` pin, and its `fit`/`predict` API is a full rewrite of `optimization.py`, not a drop-in swap. PyPortfolioOpt ([pypi.org/project/pyportfolioopt](https://pypi.org/project/pyportfolioopt/)) covers exactly max-Sharpe/min-vol/efficient-frontier with the lightest footprint of the three. **If the brief literally requires a feature none of PyPortfolioOpt's surface has** (e.g. Black-Litterman by name, or a specific convex risk measure like CVaR-native optimization) **skfolio is the one to migrate to**, not Riskfolio-Lib — same reasoning as the PCA-factor-model addition already in this codebase: skfolio's built-in walk-forward cross-validation and Ledoit-Wolf/Gerber covariance estimators overlap directly with `backtesting.py`/`factor_models.py`'s own hand-rolled versions, so a migration would consolidate rather than duplicate. Absent that literal requirement, staying on PyPortfolioOpt keeps the dependency surface and `optimization.py`'s API exactly as documented above — no functional gain from switching without a concrete need driving it. |
| **GitHub Pages** for deployment | **Streamlit Community Cloud** (primary) + **Render** (backup, via `render.yaml` + `Dockerfile`) | GitHub Pages serves static files only — "GitHub Pages does not support server-side languages such as PHP, Ruby, or Python" ([official GitHub Docs](https://docs.github.com/articles/creating-project-pages-manually)) — it cannot run a Streamlit server process. |
| **ISM Manufacturing PMI** for macro context | **Industrial Production Index (FRED `INDPRO`)** | ISM's PMI is a paid, proprietary survey-based series — not available on FRED or any free API. Industrial Production is the closest legitimate free alternative: real output data rather than a survey diffusion index, but it captures the same underlying signal (manufacturing-sector momentum). |
| **Airflow** for orchestration | **None — on-demand fetch inside Streamlit's own request-response cycle** | This app is interactive and synchronous (change a sidebar parameter, it recomputes), not a scheduled batch pipeline — there's no recurring DAG to orchestrate. Airflow would mean deploying a scheduler + webserver + metadata DB for zero present need, the same over-engineering trap Riskfolio-Lib was avoided for above. It would become the right tool if this moved from live per-request fetches to nightly pre-materialised data — noted as a real future option, not dismissed outright. |
| **LangChain / LangGraph** for the LLM layer | **Custom `llm_client.py`** (one call site, Groq→[configurable hosted fallback list]→Ollama, multi-key rotation) | Every LLM use in this app (commentary, news digest, chatbot) is a single, well-defined call with context injection — no multi-step agent deciding which tool to call next, no complex cross-session memory to manage. LangChain/LangGraph earn their weight when an agent genuinely orchestrates multiple tools/steps dynamically; here it would be a heavy dependency hiding a simpler fallback/rotation mechanism behind an abstraction layer, for no functional gain. |

## Forecasting models, and where ML/NLP actually show up in this repo

`forecasting.py` ships five models, in increasing order of sophistication —
`FORECAST_MODELS` in that module is the single source of truth for the set;
`app.py`'s sidebar selectbox and `scripts/benchmark_walk_forward.py`'s grid
both read from it, so a new model added there needs no changes anywhere else
to become selectable and benchmarkable.

| Model | What it does | Classical stats or ML? |
|---|---|---|
| **Naive (random walk)** | Tomorrow's price = last price + average historical drift, with a `sqrt(horizon)`-widening confidence band. Always kept in the set as the baseline every other model has to beat — a fancier model that can't outperform this isn't adding value. | Classical statistics |
| **ETS (Holt linear trend)** | Exponential Smoothing with a linear trend component. Few assumptions, robust default for noisy financial series. | Classical statistics |
| **Theta method** | Splits the series into a long-term linear trend (OLS, extrapolated directly) and a short-term residual component (the trend's own residuals, forecast via Simple Exponential Smoothing), then adds the two back together. The strongest overall performer in the original M3 forecasting competition despite having no hyperparameters to tune — a useful, cheap middle point between ETS and ARIMA on the sophistication/speed spectrum. See `forecasting.theta_forecast`'s own docstring for why detrending before applying SES matters (skipping it would systematically under-forecast any real trend, since SES has no trend term of its own). | Classical statistics |
| **ARIMA (auto order)** | ARIMA(p,1,q) with a small AIC grid search over p and q (d=1 fixed, since price levels are non-stationary by construction). Captures autocorrelation structure the simpler models miss, at the cost of being slower and more prone to overfitting on short series. | Classical statistics |
| **LSTM (recurrent neural net)** | A small, from-scratch-trained recurrent network, fit fresh per asset per call (no pretrained universal price model exists to load). The one genuinely ML-based member of this set. | **Machine learning** (deep learning) |

**Where ML and NLP actually appear across the whole codebase**, not just in
forecasting — worth stating explicitly since "uses AI" is thrown around
loosely and this project tries not to:

- **Machine learning**: `lstm_forecast` (deep learning, per-asset recurrent
  network); `factor_models.pca_factor_cov` (PCA — unsupervised ML, used for
  covariance estimation on wide universes); FinBERT sentiment
  (`news_data.fetch_finbert_sentiment` — a pretrained financial-domain
  transformer, called via Hugging Face's hosted Inference API rather than
  run locally).
- **NLP, not necessarily ML**: `rag.py`'s TF-IDF + cosine-similarity
  retrieval (classic information retrieval, not a learned model); VADER
  sentiment (`news_data.compute_local_sentiment` — a fixed lexicon, no
  training involved). FinBERT above is both NLP and ML at once; TF-IDF and
  VADER are NLP techniques that predate the "ML" label being applied to
  every NLP task.
- **Classical statistics, deliberately NOT ML**: Naive/ETS/Theta/ARIMA
  (forecasting.py) and GARCH(1,1) (`volatility_forecasting.py`) — all
  parametric models fit by maximum likelihood or least squares, not learned
  from data the way a neural net or gradient-boosted tree is. Grouping these
  with LSTM as "the AI part of the app" would overstate what they are; the
  README and this document both call them out as classical statistics for
  that reason.

## The three-portfolio comparison, precisely

1. **Historical-based**: max-Sharpe weights from mean/covariance estimated on the *training*
   window only (everything before the held-out forecast horizon).
2. **Forecast-based**: same optimizer, but the expected-return vector (μ) comes from the chosen
   forecasting model's predicted prices over the held-out window; the covariance matrix stays
   historical (forecasting a full covariance matrix reliably is a much harder problem, and using
   the historical covariance here is standard practice even in forecast-driven allocation).
3. **Realized-optimal**: max-Sharpe weights fitted on the *actual* returns of the held-out
   window — the hindsight benchmark the other two are judged against.

All three weight vectors are then applied to the **same actual realized returns** of the held-out
window, so the comparison isolates the effect of the allocation choice alone.

**Reading the period count.** The sidebar's forecast horizon (e.g. 90) is a count of *prices*
held out; `compute_returns` drops the first row (`pct_change` is undefined for it), so the
metrics table's `n_periods` — and every LLM commentary referencing "the N-period window" — is
actually `horizon - 1`. The section header still says "last {horizon} periods" for readability
(it's the sidebar control's own value), so don't be surprised the two numbers differ by one; it's
this off-by-one, not a computation bug — same convention every KPI in this app already follows.

## Walk-forward validation (why one comparison isn't enough)

A single train/test split can be a lucky or unlucky draw — it says "forecasting helped this one
time," not "forecasting helps in general." The **Walk-forward validation** section (below the
main comparison, same tab) repeats the exact same historical/forecast/realized comparison across
several **expanding windows**: window *k*'s training set is everything known up to that point,
tested on the next `horizon`-sized slice — each refit uses strictly more data than the last, the
way a real strategy would actually be re-run over time.

Output: a box plot of realized Sharpe per portfolio type across all windows (a spread, not one
number), a mean/std summary table, and a **forecast win rate** — the fraction of windows where the
forecast-based portfolio beat the historical-based one on Sharpe. A win rate hovering near 50%
across many windows is the honest, expected result for short-horizon price forecasting; a rate
consistently well above 50% would be the actual signal that the forecasting step adds value rather
than noise from one convenient split. `src/backtesting.py` implements this on top of the exact same
`metrics.py` / `optimization.py` / `forecasting.py` functions the single-window comparison uses, so
the two views can never silently disagree on how a metric is computed.

Note: ARIMA across many windows/assets is noticeably slower (grid search × every window) —
ETS, Theta, or naive are the practical default for iterating with walk-forward turned up.

**Not A/B testing, deliberately.** Walk-forward backtesting and A/B testing are both "compare
two approaches" methodologies, but they answer different questions and shouldn't be conflated in
conversation about this project. A/B testing compares variants on LIVE traffic, randomly split
between real users, with a statistical-significance test on the outcome — it's a production
experimentation technique. Walk-forward backtesting, what this app does, compares strategies on
HISTORICAL data across sequential time windows — there's no live traffic, no randomised user
split, and "the market" isn't an experiment subject that can be assigned to a control group. For
a price-forecasting/portfolio-optimization project, walk-forward validation is the methodologically
correct analogue, not a substitute for A/B testing — the two aren't interchangeable tools for the
same job.

## Expanded universe & macro context

Beyond the brief's 5 default equities, the optional ETF sleeve now spans multiple asset classes
so the tool demonstrates real cross-asset thinking, not just a bigger stock list — each was picked
to represent a genuinely different risk driver, not to pad the list:

| Ticker | Asset class | Why it's here |
|---|---|---|
| `SPY` / `QQQ` | US equity indices | Broad-market and tech-heavy benchmarks |
| `TLT` | Long-duration US Treasuries | Rate/duration exposure, typically negatively correlated with equities in risk-off moves |
| `GLD` / `SLV` | Gold / Silver | Inflation & crisis hedge; silver has a higher-beta, more industrial profile than gold |
| `USO` / `DBC` | Oil / broad commodities basket | Energy & inflation exposure, a different driver from equities or rates |
| `UUP` | US Dollar Index | FX/currency exposure — an ETF proxy rather than raw spot FX or futures, so it optimises cleanly long-only without margin/roll complications |
| `VNQ` | US REITs | Real estate, a distinct cash-flow driver from both equities and bonds |

The **Overview tab** also now opens with a **macro & risk context panel**: the live 3-month and
10-year Treasury yields (FRED), the **10Y-3M term spread** (a negative spread has preceded every
US recession since the 1960s, with some false positives — shown as context, not a signal to trade
on), and the **VIX** level with a simple calm/normal/elevated read. These feed into the AI
Analyst's commentary too, so the narrative accounts for the macro backdrop, not just the portfolio
numbers in isolation.

**Honest caveat, stated in the app itself**: short-horizon equity price forecasting from price
history alone has weak genuine predictive power (consistent with the efficient market hypothesis).
The forecast-based portfolio here demonstrates the required methodology; it is not a claim that
the forecast should be trusted for real allocation decisions. This mirrors the hallucination/
overconfidence caveats from the bootcamp's ethics module — a model producing a confident number is
not the same as that number being reliable.


## Expanded universe & advanced KPIs

Beyond the brief's 5 default equities and the ETF/commodity sleeve:

**Universe presets** — sidebar "Universe preset" selector:
- *Brief default (5)*: AAPL/MSFT/TSLA/AMZN/GOOG, as specified.
- *Mega Caps (15)*: the largest, most-recognised S&P 500 names across sectors — a one-click
  "give me something sensible" option.
- *Custom / sector picker*: build a universe from 11 GICS sectors (7-12 liquid names each, 104
  tickers total — expanded from an initial ~59). Deliberately NOT all ~500 S&P
  constituents — see the callout below.

**Why not all 500 S&P constituents:** covariance estimation degrades badly with hundreds of
names and only a few years of daily history (the classic "more parameters than data" problem);
real buy-side desks handle this with factor models (Barra, Fama-French) or sector-constrained
universes, not a raw 500×500 mean-variance optimisation on a handful of return observations per
pair. A curated, sector-organised universe is the professionally correct choice here, not a
scope-limited shortcut. **The 104-ticker universe already sits at the point where this matters in
practice** — which is exactly why the covariance estimator below exists.

**Covariance estimation: Ledoit-Wolf shrinkage vs. PCA factor model** (sidebar
"Covariance estimator")

The math, precisely: with N assets, a covariance matrix has N(N+1)/2 free parameters — 5,460 for
this app's full 104-ticker universe, against at best a few thousand daily observations. Ledoit-Wolf
shrinkage (the long-standing default here) is a real fix for a *small* universe's noisy sample
covariance, but it doesn't remove the underlying degrees-of-freedom problem — it just shrinks
toward a target. A **PCA statistical factor model** does remove it: assume returns are driven by a
small number of common factors plus asset-specific noise, and the parameter count collapses from
N(N+1)/2 down to roughly N × k (k = factor count) + N. This is the SAME core idea real buy-side
desks use (Barra, APT) for exactly this problem — implemented here as a *statistical* factor model
(PCA extracts factors directly from the return data, orthogonal by construction), not a
*fundamental* one like Barra (pre-specified style factors — value, size, momentum — fit by
cross-sectional regression against company characteristics), which this project has neither the
fundamentals data nor the scope to build. Documented as that honest simplification, not oversold
as a Barra reimplementation.

- **When to use which**: Ledoit-Wolf for the default small universes (5-15 tickers); PCA once you
  build a wide custom universe (~40+ tickers via the sector picker).
- **Where it applies**: one dispatch point (`optimization.historical_mu_cov`'s `cov_method`
  parameter), threaded through every place a covariance is estimated — Efficient Frontier,
  Forecast & Compare, and the walk-forward validation all use the same choice, never Ledoit-Wolf
  in one tab and PCA in another.
- **Transparency**: the Efficient Frontier tab shows the actual cumulative explained variance for
  the chosen factor count — a factor model explaining 35% of variance is a materially weaker
  covariance estimate than one explaining 85%, and that number is surfaced directly rather than
  left implicit in a black-box matrix.
- See `src/factor_models.py`'s docstring for the full derivation and the orthogonal-factor-model
  assumption (idiosyncratic risk uncorrelated across assets) that makes the parameter-count
  reduction work.

**Advanced risk-adjusted metrics** (Calmar, Omega, Information Ratio, Treynor, Beta) — full
formulas and interpretation guidance in **"Understanding the KPIs"** near the top of this README.

**Per-ticker fundamentals** (Overview tab, fetched on demand via a button — not automatic, to
conserve API quota): market cap, trailing P/E, beta, dividend yield, price-to-book, 52-week range.

**Source chain (after live testing exposed a real limitation):** Twelve Data's
`/statistics` endpoint turned out to be restricted on the free tier to their public demo symbol
(`AAPL`) — every other ticker returned a `403 {"message": "/statistics is available exclusively
with pro or ultra or venture or enterprise plans"}`, confirmed directly via `curl`. Rather than
ship a feature that only works for one hardcoded ticker, fundamentals now try **Finnhub first**
(`/stock/profile2` + `/stock/metric`) — confirmed free-tier for arbitrary tickers, and you likely
already have `FINNHUB_API_KEY` configured for the news digest — falling back to Twelve Data only
if Finnhub isn't configured or returns nothing (which still covers `AAPL` via the demo-symbol
path). Two Finnhub-specific parsing quirks worth knowing if this needs debugging later: market cap
is returned in **millions**, not raw units (multiplied by 1e6 in code to match the display format),
and dividend yield is a **percentage number** (e.g. `0.72` for 0.72%), not a decimal fraction
(divided by 100 in code to match). Finnhub's field names were confirmed via public documentation
and multiple independent working code examples, not a live key in this environment — if a field
looks wrong, `curl` both endpoints directly before assuming the parser is stale.

**News sentiment (AI Analyst tab, next to the news digest):** same source-chain philosophy as
fundamentals — try the best genuinely-free source first, fall back rather than fail. FinBERT (a
BERT model further trained on financial text, then fine-tuned for positive/negative/neutral
classification — called through Hugging Face's hosted Inference API, no local model weights) is
tried **first**, on the assumption that a financial-domain-tuned model reads financial headlines
more accurately than a general-purpose aggregator or lexicon. If it's unavailable (no
`HUGGINGFACE_API_KEY` configured, or every request failed), Finnhub's own `/news-sentiment`
endpoint (an aggregation over a much wider article set than the ~5 headlines this app fetches per
ticker) is tried next — or skipped outright once the module's own circuit breaker confirms it's
plan-restricted on this account (see `BUGFIXES.md`). If neither source has anything, sentiment is
computed **locally** with VADER (a free, offline, lexicon-based scorer — no API key, no model
download, tuned for short informal text) on the headlines already fetched for the digest — the
final tier that can never fail. If nothing at all is available, the UI shows an explicit
"sentiment not available" message rather than a silent blank, which would otherwise read as a
false "neutral" claim. Every sentiment reading is tagged with which of the three computed it.

## Sources (from the brief)

Tools and references the original brief pointed to. Kept as-is unless noted otherwise (see
"Key design decisions" table above for the two swaps and why):

| Source | Role in this project |
|---|---|
| [Yahoo Finance / `yfinance`](https://github.com/ranaroussi/yfinance) | Market data acquisition — adjusted close prices, used as specified |
| [Portfolio Visualizer](https://www.portfoliovisualizer.com/) | UX/feature reference cited in the brief for inspiration (efficient frontier, allocation view) |
| [Riskfolio-Lib](https://riskfolio-lib.readthedocs.io/en/latest/index.html) | Brief's suggested optimization library — **not used**, see decisions table |
| [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) | Brief's suggested alternative — **used** for mean-variance optimization + efficient frontier |
| [Kats](https://facebookresearch.github.io/Kats/) | Brief's suggested forecasting library — **not used**, see decisions table |
| [PyCaret](https://pycaret.org/) | Brief's suggested forecasting alternative — considered, `statsmodels` chosen instead for a smaller, more predictable dependency footprint on a 2-day deadline |
| [Streamlit](https://streamlit.io/) | App framework, used as specified |
| [Streamlit Community Cloud](https://streamlit.io/cloud) | Primary deployment target |

## Key challenges

1. **Fair out-of-sample comparison design.** The trap in a "forecast vs. realized" comparison is
   letting any forward-looking information leak into the historical/forecast portfolios' training
   window. I solved this with a strict train/test split: μ and Σ for portfolios 1 and 2 are
   estimated *only* on data before the held-out window, and all three weight vectors are evaluated
   on the identical realized returns of that window — so the bar chart genuinely isolates
   allocation skill from lucky market conditions.

2. **PyPortfolioOpt's `max_sharpe()` fails hard, not soft, when every asset's expected return is
   below the risk-free rate** (a realistic case — pick a bear-market date range, or a forecast
   that goes negative). It raises a plain `ValueError`, not its own `OptimizationError`, so a
   naive `except OptimizationError` silently misses it and crashes the app on a very plausible
   user input. I caught it explicitly (see `optimization.py`) with a min-volatility fallback, and
   verified it with a synthetic bear-market test case, not just the happy path.

3. **Keeping the LLM grounded to avoid hallucinated numbers.** A finance tool that confidently
   states a wrong Sharpe ratio is worse than a tool that says nothing. I solved this with an
   explicit context-injection pattern (`ai_features.build_results_context`): every computed number
   the LLM is allowed to reference is serialised into the prompt, and the system prompt explicitly
   instructs the model to say "not available" rather than estimate a figure not present there.
   Every injected data block (portfolio metrics, retrieved news, retrieved academic papers) is
   wrapped in a named XML tag (`<portfolio_data>`, `<retrieved_news>`, `<academic_references>` —
   see `ai_features._wrap_context`) so instructions can point at a specific tag instead of a vaguer
   "the context above," and `SYSTEM_PERSONA`'s single most load-bearing rule (don't report an
   inflated annualised figure from a short window at face value) ships with a concrete GOOD/BAD
   example pair, not prose alone — both changes matter more now that `chat()` can route to five
   providers of varying instruction-following discipline (Groq, OpenRouter, Cerebras, SambaNova,
   Ollama).

4. **Per-ticker forecasting and news fetching were the two biggest wall-clock costs** (ARIMA
   fitting across many windows/assets; three sequential HTTP calls per ticker for the news
   digest). I parallelised both with `concurrent.futures.ThreadPoolExecutor`, capped at 8 workers
   to avoid oversubscribing a small container. I used threads rather than processes for the ARIMA
   case specifically: the fitting is dominated by numpy/scipy linear algebra, which releases the
   GIL during BLAS calls, so threads give a real speedup without the pickling/Streamlit-context
   fragility subprocesses would add. I kept column/ticker order explicit (`executor.map` for
   forecasting, original-order iteration over a completion-order results dict for news) so
   parallelising never makes output order depend on which network call happened to finish first.

## Short selling, transaction costs, and RAG

Three things previously listed as deliberately out of scope for the 2-day window, added once
there was time to do them properly rather than as an afterthought:

**Short selling / leverage.** Sidebar toggle "Allow short selling" — off by default (long-only,
weights ≥ 0), on makes bounds symmetric (`-cap` to `+cap` around the same max-weight setting).
The portfolio stays fully invested (weights still sum to 100%); this does not add gross leverage
beyond that. `resolve_weight_bounds()` in `optimization.py` centralises the long-only vs.
long-short logic so every optimizer call site (frontier, single-window comparison, walk-forward)
stays consistent.

**Transaction costs & rebalancing frequency.** Sidebar slider "Transaction cost (bps per
rebalance)", default 10 bps. Charged as `turnover × cost rate` at every point a portfolio
actually rebalances: once for the initial trade in the single-window comparison, and at every
walk-forward window boundary — tracked independently per portfolio type against ITS OWN previous
weights, not a shared reference. "Rebalancing frequency" surfaces through the existing forecast
horizon control rather than a separate parameter: a shorter horizon means more walk-forward
windows over the same history, i.e. more frequent rebalancing, i.e. more cumulative cost drag —
shortening the horizon is how to see this effect directly. Set the cost to 0 for the frictionless
textbook comparison. `compute_turnover()` / `apply_transaction_cost()` in `metrics.py`.

**RAG for the chatbot.** A genuine index-then-retrieve pipeline (`src/rag.py`) over the news/
filings corpus collected in the AI Analyst tab, kept deliberately separate from the context-
injection approach still used for portfolio metrics (that distinction is the point — metrics are
a small, fixed, must-be-complete set of numbers where full injection is correct; news/filings are
unstructured and only the query-relevant subset should reach the prompt). TF-IDF + cosine
similarity was chosen over neural embeddings: this corpus is small (a handful of headlines per
ticker set), so a sentence-transformers model would add a large, slow-to-install dependency for
retrieval-quality gains that don't matter at this scale — scikit-learn (already a transitive
dependency via PyPortfolioOpt's `cvxpy` stack) is sufficient and dependency-light. Only the top-4
chunks relevant to the user's specific question are retrieved and injected — verified directly: a
question about NVDA earnings pulls in the NVDA news chunk but not an unrelated Apple one, and a
pure concept question ("what is Sharpe ratio") retrieves nothing and injects no RAG block at all,
rather than force-fitting irrelevant news into every answer.

**Persistence.** `save_chunks`/`load_chunks` in `rag.py` persist a fetched
corpus to the same Redis instance `cache.py`'s `cached()` decorator already uses, opt-in via
`REDIS_URL` — closes the gap this section used to flag as a known limitation. With no `REDIS_URL`
set (every local/free-tier deployment today) this is a silent no-op and behaviour is exactly what
it always was: rebuilt fresh every session. With one set, a corpus already fetched for a given
ticker set — this session or a previous one — is available to the chatbot immediately, and
survives a container restart. Still TF-IDF, not a real vector store: if this grew into hundreds of
PERMANENTLY accumulated documents (rather than today's "one fetch's worth, refreshed on demand"),
a real vector store (Chroma/FAISS) with neural embeddings would be the right next upgrade — noted
in Next steps, not built here because it would be unjustified complexity at the current scale.

**Academic literature grounding.** A third, independent grounding source for
the chatbot, alongside the portfolio-metrics context injection and the RAG news pipeline above:
`src/academic_search.py` combines **two** free sources — Semantic Scholar's Academic Graph API
(tried first: broader coverage across published venues, includes citation counts) and **arXiv**
(fills any remaining slots, genuinely free with no key at all, and the canonical source for the
original papers behind several techniques this app documents by name — GARCH, walk-forward
validation, PCA factor models). Results from both are deduplicated by normalised title before
being returned. This triggers when a question names a known methodology term (Sharpe, GARCH,
Ledoit-Wolf, walk-forward, efficient market hypothesis, ...), and the system prompt instructs the
LLM to cite only those returned papers, never an invented one. This closes a real gap the other
two grounding sources don't cover: LLMs are well documented to hallucinate plausible-sounding
academic citations (a confident author name, year, and title for a paper that doesn't exist, or
that says something different from the claim attached to it) — for an app whose whole premise is
grounding LLM output in real data, letting it cite methodology papers from memory would quietly
reintroduce that exact failure mode. `detect_methodology_terms()` matches against a curated (not
exhaustive) vocabulary of the actual techniques this codebase uses — deliberately simple
word-boundary matching rather than anything ML-based, since the set of terms this app could
plausibly be asked about is small and known in advance. `SEMANTIC_SCHOLAR_API_KEY` is optional
even at the API level (unlike every other provider key in this app) — the search endpoint works
unauthenticated at a lower shared rate limit, a key just raises the ceiling; arXiv needs no key
at all, structurally, not just as a fallback. The same `search_academic_papers()` orchestrator
also powers a standalone "🔬 Search academic literature" box under the Chatbot tab, for looking
something up directly without going through the chat.

## Known limitations

Things I'm aware of and chose not to fix within this project's scope, rather than gaps I missed:

- **Short-horizon price forecasting genuinely has weak predictive power.** The whole app is built
  around this honestly (win-rate near 50% is the expected result, not a bug) — but it means the
  "Forecast-based" portfolio should never be read as an actionable signal, only as a methodology
  demonstration. I say this explicitly in three places in the UI so it can't be missed.
- **The covariance matrix is historical by default, with a GARCH-forecasted alternative for the
  Forecast-based portfolio.** `historical_mu_cov` (used by the frontier, and by Historical-based/
  Realized-optimal everywhere) always uses historical covariance — standard practice even in
  forecast-driven allocation. The Forecast-based portfolio, in both the single-window comparison
  and the walk-forward validation, can instead use `volatility_forecasting.garch_forecast_cov` —
  GARCH(1,1)-forecasted volatility per asset, genuinely reacting to recent conditions rather than
  averaging over the whole history — via the sidebar's "Forecast-based covariance" toggle
  (wired into the UI; previously only reachable by calling `run_walk_forward(...,
  forecast_cov_method="garch")` directly from a script). Correlation between assets still comes
  from history (Ledoit-Wolf shrinkage) even in GARCH mode: a true correlation forecast needs a
  DCC-GARCH (Dynamic Conditional Correlation) model, and no well-maintained Python package
  implements that robustly today — see `volatility_forecasting.py`'s docstring for the full
  honest accounting of what is and isn't a genuine forecast here.
- **SEC EDGAR's full-text search matches on a bare company-name string, not the CIK.**
  `fetch_sec_filings` queries `q='"Apple"'` for AAPL (the static ticker→name map in `app.py` uses
  short names) — confirmed live in the Sources panel: this also surfaced *Apple Hospitality REIT*
  8-Ks alongside Apple Inc.'s own, occasionally filling more than one of the `max_filings` slots
  with the unrelated company. `_fetch_cik_for_ticker` resolves the real CIK
  via SEC's own `company_tickers.json` and results are filtered to it (falling back to the
  unfiltered search only if that lookup itself fails), plus deduplicated by accession number — see
  `BUGFIXES.md` and the tests in `test_news_data.py`.
- **The Yahoo circuit breaker is process-wide, not per-session — deliberately, not an oversight.**
  On a multi-user deployment (Streamlit Community Cloud, Render), if Yahoo fails for one user it's
  skipped for everyone for the next 3 minutes. Kept as-is on purpose: making it
  per-session would mean every new session re-pays the full Yahoo retry cost (up to 6 failed calls
  with backoff) independently, even seconds after another session already proved Yahoo is down for
  the whole process — reintroducing exactly the responsiveness problem this circuit breaker exists
  to fix. Yahoo being unreachable is a server-side fact, true for every session equally, not a
  per-user one; a per-session breaker would be a strictly worse trade-off, not a more correct one.
- **RAG persistence** — see above, opt-in via `REDIS_URL`. Still TF-IDF, not a
  real vector store; see "Short selling, transaction costs, and RAG" above for what would justify
  that next upgrade and why it isn't warranted yet.

## Next steps

The longer-term roadmap I'd tackle with a real budget and a production SLA to hit, rather than a
bootcamp deadline. The two nearest-term, concrete items from this list — SEC EDGAR CIK matching
and CI hardening — are done (see `BUGFIXES.md` and `.github/workflows/ci.yml`); load testing now
has a starting script too (see "Capacity benchmarking" under Setup). What's left:

1. **Cloud deployment (Azure)** — move off Render's free tier onto Azure Container Apps or AKS:
   autoscaling under real traffic, Azure Key Vault for secrets instead of `.env`/dashboard env vars,
   Application Insights for centralised logging/tracing across the Streamlit process and every
   external call (Yahoo, Twelve Data, Groq, FRED, NewsAPI), and a proper staging slot so a bad
   deploy doesn't hit prod directly. AWS/GCP equivalents would cover the same ground — Azure only
   because it's the most common enterprise default I'd expect to already have credentials for.

2. **Paid market data instead of the free-tier fallback chain** — a proper vendor (Polygon.io,
   IEX Cloud, or a Bloomberg/Refinitiv feed at real enterprise scale) replaces the
   Yahoo→direct-API→Twelve Data cascade entirely: no rate limits, no "possibly delisted" parsing
   errors, survivorship-bias-free historical data (the free sources silently drop delisted/acquired
   tickers, which quietly biases any backtest toward survivors), and proper corporate-action
   adjustments (splits, spin-offs) instead of relying on adjusted-close alone.

3. **Price forecasting with even more compute behind it** — the naive/ETS/Theta/ARIMA/LSTM set (see
   "Key design decisions" above) now spans classical statistics through a from-scratch neural
   network. GARCH-forecasted volatility closes half of the forecasting gap (see
   `volatility_forecasting.py`); mu itself is the remaining half, and `lstm_forecast` is a first,
   deliberately small step into it, not the ceiling. With dedicated GPU compute, the next tier is a
   proper hyperparameter search per asset (lookback window, hidden size, layer count) instead of
   fixed defaults, an ensemble across all four models weighted by each one's own walk-forward track
   record, and enough budget to test whether a transformer-based forecaster (e.g. a fine-tuned
   time-series foundation model) actually beats naive out-of-sample on this universe — which, per
   the walk-forward results already in this repo, is a real, open question, not a given.

4. **A managed LLM tier with a real SLA** — Groq free + a 3-provider hosted fallback chain
   (OpenRouter, Cerebras, SambaNova, closing the gap where Ollama alone
   didn't work once deployed) is resilient but still rate-limited and best-effort across four free
   tiers. An enterprise contract (Azure OpenAI, or a paid tier on any of the above) would add a
   real SLA and guaranteed rate limit, plus a proper prompt-evaluation pipeline (regression-testing
   prompt changes against a fixed set of portfolios/questions, not just eyeballing output) before
   any prompt change ships.

5. **Turn the capacity benchmark into an actual load-testing budget** — `scripts/
   benchmark_walk_forward.py` (new) gives per-scenario wall-clock/memory numbers on demand, but
   doesn't yet answer "how many CONCURRENT users can one container serve" — the real question
   under production traffic. That needs the benchmark numbers turned into a target (e.g. "N
   concurrent walk-forward runs, p95 under Ts") and validated against Streamlit's actual
   per-session process model on the target hardware, not just single-run timings.

6. **LangGraph, if the AI Analyst ever becomes a real multi-step agent** — e.g. deciding on its own
   to pull fresh news, recompute a scenario, then compare it against the current portfolio, instead
   of the three fixed, hand-wired features (commentary/digest/chatbot) it is today. Not needed for
   the current scope, but the natural next step if the AI layer grows from "answers questions about
   fixed numbers" into "decides what to compute next."