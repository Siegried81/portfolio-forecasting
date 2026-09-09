# Evaluation checklist — brief vs. actual state

Working document, not a deliverable — decide whether to commit it or keep it as personal prep
notes for the presentation. Legend: ✅ done & verified · ⚠️ done but needs a check before D-day ·
❌ not done · **NTH** = beyond the brief (Nice To Have), not required to pass.

## 1. Data requirements (brief, "Data" section)

| Requirement | Status |
|---|---|
| yfinance / Yahoo Finance API | ✅ `src/market_data.py`, with a documented 4-step fallback chain (yfinance → Yahoo direct → Twelve Data → Alpha Vantage) |
| US equities (AAPL/MSFT/TSLA/AMZN/GOOG as a starting point) | ✅ `DEFAULT_EQUITY_TICKERS`, brief's exact 5 as the default preset |
| Adjusted closing prices | ✅ `fetch_adjusted_close` |
| User-defined frequency (daily/monthly/yearly) | ✅ daily/weekly/monthly selectbox — brief said "yearly" as an option, not implemented, but daily/weekly/monthly covers the practical range and yearly would leave too few points for covariance estimation anyway |
| User-defined date range, enough history for returns + forecast window | ✅ sidebar date pickers + quick-range presets, with an explicit warning if the range is too short for the chosen horizon |
| **NTH**: ETFs / indices beyond equities | ✅ SPY/QQQ/TLT/GLD/SLV/USO/DBC/UUP/VNQ + 104-ticker sector universe |

## 2. Mission objectives

| Objective | Status |
|---|---|
| Time-series manipulation (returns, volatility, correlations) | ✅ Overview tab |
| Time-series forecasting to predict future prices | ✅ Naive / ETS / Theta / ARIMA / ML regression (XGBoost or scikit-learn) / LSTM — 6 models (`src/forecasting.py`) |
| Finance vocabulary (Sharpe, Sortino, efficient frontier) | ✅ + extensively documented in README with formulas |
| Mean-variance optimization | ✅ PyPortfolioOpt (`src/optimization.py`) |
| Deployed interactive Streamlit app | ⚠️ **see §5 — this is the one real risk right now** |

## 3. The Mission — input/output contract

| Ask | Status |
|---|---|
| Input: stocks, date range, frequency, user-selected | ✅ |
| Output: optimal weights via mean-variance | ✅ Efficient Frontier tab |
| Output: efficient frontier visualization | ✅ Plotly chart, frontier line + individual assets + max-Sharpe star |
| Output: return, volatility, Sharpe, Sortino | ✅ + Calmar, Omega, VaR/CVaR, Beta, Information Ratio, Treynor, Jensen's Alpha, Ulcer Index, skew/kurtosis (**NTH**, all beyond the brief's 4 named metrics) |
| Forecast future prices over a new range | ✅ |
| Construct optimal portfolio from forecasted data | ✅ "Forecast-based" portfolio |
| Compute true realized-optimal from actual future prices | ✅ "Realized-optimal" (hindsight) portfolio |
| Compare all three (historical / forecast / realized) | ✅ side-by-side table + bar chart, exact brief wording |

## 4. Roadmap & tools

| Item | Status |
|---|---|
| Riskfolio-Lib **or** PyPortfolioOpt | ✅ PyPortfolioOpt — brief explicitly allows either, choice justified in README's decisions table |
| Kats **or** PyCaret for forecasting | ⚠️ **Neither** — statsmodels/scikit-learn used instead (now 6 models: Naive/ETS/Theta/ARIMA/ML regression/LSTM, well beyond the brief's ask). Justified at length in README (Kats unmaintained since 2022; PyCaret considered, heavier dependency footprint). Defensible, but a strict grader could mark this row "no" on a literal reading since neither named option was used — be ready to explain this trade-off verbally in the demo |
| Streamlit app: inputs (stocks/range/frequency), outputs (allocation/frontier/metrics) | ✅ |
| Optional tabs: historical vs forecast vs realized | ✅ (built as one tab with a comparison table, not 3 separate tabs — same information) |
| Deploy (Streamlit Community Cloud / Render / similar) | ⚠️ see §5 |
| **Live demo prepared (5 min)** | ❌ **Not found in the repo** — no slides/script file. This is a real, separate deliverable from the brief's "Deliverables" list — make sure it exists somewhere (even outside git), it's not something the codebase can show for you |

## 5. ⚠️ Deployment — re-check before submission (large diff since the last live check)

**This section needs a fresh live check** — a large batch of changes landed since it was last
verified (4-tier market data, 6 forecast models, 6 news sources, Fama-French factor exposure,
period-over-period matrix, Form 4 insider trades, `LICENSE`, `check_env.py`), so treat the rows
below as the LAST known status, not current:

| Target | What I actually observed |
|---|---|
| Streamlit Community Cloud (`portfolio-forecasting-sieg.streamlit.app`) | Redirects to a **login wall** (`share.streamlit.io/-/auth/...`) instead of the app. Reads as the app being restricted to invited viewers rather than public. **A grader clicking this link today would be blocked.** Fix: Streamlit Cloud → app settings → sharing → set to public. |
| Render (`portfolio-forecasting.onrender.com`) | Returned **503 Service Unavailable** twice. Could just be free-tier cold-start (spins down after inactivity, ~30-60s to wake) — reload it yourself and watch, but don't assume it's fine untested. |
| GitHub Actions CI | ✅ Both pushed commits (`99b3af0`, `c17d727`) are green. Note: CI has **not yet run** against the 2 bugfixes + 3 new test files still sitting uncommitted locally as of this check — push them to get a real signal before the deadline. |

**Action before submission**: open both URLs yourself in a private/incognito window (logged out of Streamlit) — that's what a grader's browser looks like.

## 6. Repo cleanliness ("no unnecessary files")

| Item | Status |
|---|---|
| Old redundant docs (DEPLOYMENT.md, QUICK_START.md, STREAMLIT_CLOUD.md) | ✅ already removed in a prior commit |
| `explanations.txt` (personal notes) | ✅ removed from the repo (confirmed gone) |
| `python-version` (wrong filename for pyenv) | ✅ removed from the repo (confirmed gone) |
| `images/` screenshots | ✅ now referenced from README's new Screenshots section — no longer dead weight |
| `.venv/`, `.pytest_cache/`, `__pycache__/` | ✅ correctly gitignored, not tracked |
| `HARNESS.md` | Deliberately held back by you for a last-moment push — not forgotten, just sequenced |

## 7. Python typing

✅ `from __future__ import annotations` + typed signatures across all 13 `src/` modules and `app.py`.
`mypy` runs in CI but is non-blocking (documented, honest known-limitation) — acceptable for the
timeline; a full `--strict` pass is listed as a "Next step," not a gap you missed.

## 8. NTH — genuinely beyond the brief (your differentiators)

- Walk-forward validation (multi-window, not just one train/test split) + forecast win-rate metric + period-over-period comparison matrix
- PCA statistical factor model as an alternative covariance estimator for wide universes (40+ tickers)
- Fama-French multi-factor exposure analysis (free, no key — Ken French Data Library)
- Macro/FRED panel: yields, term spread, VIX, CPI, unemployment, Fed funds, Sahm Rule, credit spread, GDP, industrial production
- Short selling / leverage toggle, transaction costs & turnover modeling
- RAG chatbot over news/filings (TF-IDF retrieval, grounded Q&A)
- Six-source news digest (NewsAPI, Finnhub, SEC EDGAR 8-K + Form 4 insider trades, GDELT, Google News, TED EU procurement) with per-source attribution
- News sentiment (FinBERT financial-domain model → Finnhub API → VADER local fallback, three-tier cascade)
- LLM commentary with Groq multi-key rotation + 3 hosted fallbacks (OpenRouter/Cerebras/SambaNova) + local Ollama fallback
- Four-tier market-data fallback (yfinance → Yahoo direct → Twelve Data → Alpha Vantage)
- 104-ticker sector-organized universe beyond the brief's 5 defaults
- `check_env.py` — one-command report of which optional API keys are configured
- `LICENSE` (MIT) — makes the repo's reuse terms explicit for a public showcase
- `HARNESS.md`: a documented AI-collaboration methodology with a real recurring-errors log — a genuine differentiator for a GenAI Developer track evaluation specifically
