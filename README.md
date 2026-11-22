# Portfolio Forecasting & Optimization

A Streamlit app I built as a case study for BeCode's AI & Data Science bootcamp (GenAI Developer
track). It builds and compares three portfolios, **historical-based**, **forecast-based** and
**realized-optimal (hindsight)**, with mean-variance optimization. An AI analyst layer sits on top
(LLM commentary, news digest, grounded Q&A chatbot).

## What it does

You pick stocks or ETFs, a date range and a frequency. The app computes returns, volatility,
correlation and the max-Sharpe efficient-frontier portfolio. It then holds out the last N periods and
forecasts each asset's price over that window with one of six models (naive random walk, ETS, Theta,
ARIMA, gradient-boosted ML regression, LSTM). It builds an optimal portfolio from the forecasted
returns and compares its realized out-of-sample performance with the historical-based portfolio and
with the hindsight-optimal one, built from the actual returns of the same window. A walk-forward
section repeats the comparison over several expanding windows.

The AI Analyst tab (Groq, then OpenRouter, SambaNova and local Ollama as fallbacks) comments on the
results and answers questions grounded in the computed numbers, with a news digest per ticker from
NewsAPI, Finnhub, SEC EDGAR, GDELT, Google News and TED (EU public procurement). Screenshots:
[SCREENSHOTS.md](SCREENSHOTS.md).

**Results.** There are no fixed result figures: every number is computed live from the universe, dates
and frequency you choose, and the repo ships no market data. Short-horizon price forecasting has weak
predictive power, so the forecast-based portfolio is a methodology demo, not a trading signal. A
forecast win rate near 50% in the walk-forward section is the expected outcome. Formulas, design
choices and limits: [docs/technical_deep_dive.md](docs/technical_deep_dive.md).

## Live demo

Both targets are unverified, treat them as likely down. Last time they were checked, Streamlit
Community Cloud redirected to a login wall and Render returned 503 (its free tier sleeps after
inactivity, so give it 30 to 60 seconds). Running it yourself is the reliable way:

- Streamlit Community Cloud: `https://portfolio-forecasting-siegried81.streamlit.app/`
- Render: `https://portfolio-forecasting.onrender.com`

## Setup

```bash
git clone https://github.com/Siegried81/portfolio-forecasting && cd portfolio-forecasting
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # then fill in your keys
streamlit run app.py
```

The app runs with no key at all; you only lose the AI Analyst content and the live risk-free rate.
`python scripts/check_env.py` lists which keys are set, what each unlocks and where to get the missing
ones. Up to 5 Groq keys can be set (`GROQ_API_KEY`, `GROQ_API_KEY_2` .. `_5`). The next one is tried
only on a 429 or a 401.

| Key | Free tier | Without it |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | the AI Analyst falls back to OpenRouter, SambaNova, then local Ollama |
| `OPENROUTER_API_KEY`, `SAMBANOVA_API_KEY` | [openrouter.ai](https://openrouter.ai), [cloud.sambanova.ai](https://cloud.sambanova.ai) | those hosted fallback tiers are skipped |
| `NEWSAPI_KEY` | [newsapi.org](https://newsapi.org/register), 100 req/day | one of six news sources is missing |
| `FINNHUB_API_KEY` | [finnhub.io](https://finnhub.io/register), 60 req/min | no Finnhub news and sentiment tier; fundamentals come from Twelve Data or yfinance |
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | risk-free rate defaults to 4%, macro panel shows n/a |
| `TIINGO_API_KEY`, `TWELVEDATA_API_KEY`, `ALPHA_VANTAGE_API_KEY` | [tiingo.com](https://www.tiingo.com/account/api/token) (500 req/hour), [twelvedata.com](https://twelvedata.com) (800 req/day), [alphavantage.co](https://www.alphavantage.co/support/#api-key) (25 req/day) | market-data fallbacks after Yahoo, tried in that order |
| `HUGGINGFACE_API_KEY` | [huggingface.co](https://huggingface.co/settings/tokens) | no FinBERT sentiment, the cascade starts at Finnhub then VADER |
| `SEMANTIC_SCHOLAR_API_KEY` | [semanticscholar.org](https://www.semanticscholar.org/product/api) | optional: the search works unauthenticated at a lower rate limit |
| `REDIS_URL` | any Redis | cache and news corpus stay per process |
| Ollama | `ollama serve` + `ollama pull llama3.1` | last LLM tier, only reachable when self-hosting |

Market data (`yfinance`), SEC EDGAR, GDELT, Google News, TED and arXiv need no key.

## Tests

```bash
pytest tests/ -v
```

541 tests across 23 files, all passing on Python 3.14 against the pinned `requirements.txt`, in about a
minute. Every network call is mocked, so the count does not depend on any key. `mypy --strict src/` is
clean.

`torch` is only needed for the LSTM and only runs on CPU here, but its default PyPI wheel pulls several
GB of CUDA packages. On a test machine or CI, install the CPU build first and the `requirements.txt`
entry is then satisfied:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
```

CI (`.github/workflows/ci.yml`, Python 3.14 like the Dockerfile) runs `pytest` and `mypy --strict`
(both blocking) on every push and PR, plus the `--quick` benchmark below as a non-blocking smoke test.

## Other entry points

**Capacity benchmark.** Wall-clock time and peak Python-heap memory of `run_walk_forward`, on synthetic
prices (no network), across universe sizes, window counts, forecast models and the GARCH toggle. For
capacity planning, not a pass/fail check.

```bash
python scripts/benchmark_walk_forward.py --quick      # Naive/ETS only
python scripts/benchmark_walk_forward.py              # full grid, incl. a 40-asset PCA universe
python scripts/benchmark_walk_forward.py --csv bench.csv
```

**JSON API.** `python -m uvicorn api.main:app --port 8004` exposes `/forecast`, `/optimize`, `/ask`
and `/api/health`. Each calls the same `src/` function as the app, over the last 5 years of daily
prices, and adds no analytics of its own (`tests/test_api.py` mocks the fetch and the LLM).

**MLflow tracking.**

```bash
python scripts/track_forecasts.py --tickers AAPL MSFT SPY --years 3
mlflow ui --backend-store-uri sqlite:///mlflow.db      # from the repo root
```

One run per model, with the walk-forward means of the forecast-based portfolio (annual return,
volatility, Sharpe, Sortino, max drawdown), the win rate, `forecast_mase_mean`, the result CSVs and
`prices_sha256` (a fingerprint of the exact prices used). Runs go to `mlflow.db` and `mlartifacts/` at
the repo root (git-ignored). `MLFLOW_TRACKING_URI` overrides the location and `--tracking-dir DIR`
overrides both. To move runs from an old `./mlruns` file store (keep `mlruns/`, the migrated runs point
at its artifacts): `mlflow migrate-filestore --source . --target sqlite:///mlflow.db`.

## Troubleshooting

**`pandas` fails to build from source.** You are probably on a Python newer than the wheels you pin.
`requirements.txt` uses minimum versions for this reason (pandas has working 3.14 wheels from 3.0.5;
3.0.0 to 3.0.4 segfault on 3.14). Check `python --version`, run `pip install --upgrade pandas numpy
scipy statsmodels streamlit`, and if that still fails use Python 3.12 (`pyenv` or `uv python install
3.12`) and recreate the venv.

**Yahoo returns no data or `crumb = 'Edge: Too Many Requests'`.** Yahoo's anti-bot handshake is a known
`yfinance` problem. The app falls back in order to the direct Yahoo API, Tiingo, Twelve Data and Alpha
Vantage. If Yahoo answers only some tickers (typically a rate limit on `TSLA`), only the missing ones
are fetched again from the next sources. The Overview tab captions the source whenever prices did not
come straight from `yfinance` (e.g. `yfinance + tiingo for TSLA`). A wall of `Failed to get ticker`
lines in the terminal does not mean the app is broken: yfinance logs every attempt the fallback
chain then handles. Only the red "Could not load market data" banner means every source failed. If so,
clear the cookie cache with `rm -rf ~/.cache/py-yfinance` and retry in a few minutes.

**WSL: `OperationalError('unable to open database file')`.** yfinance's SQLite cache can land on a
Windows-mounted drive where SQLite cannot lock files. `market_data.py` redirects it to the system temp
directory at import. If that does not apply, `rm -rf ~/.cache/py-yfinance`. The harmless
`gio: http://localhost:8501: Operation not supported` line goes away with `streamlit run app.py
--server.headless true`.

## Docker and deployment

```bash
docker build -t portfolio-forecasting .
docker run -p 8501:8501 --env-file .env portfolio-forecasting
```

The published image (GitHub Container Registry, and Docker Hub when its secrets are set) is built and
smoke-tested by `.github/workflows/publish.yml`; its Docker Hub page text is `DOCKERHUB.md`. Secrets
are never baked in: pass them with `--env-file`.

**Streamlit Community Cloud** (primary): push to GitHub, then [share.streamlit.io](https://share.streamlit.io),
New app, `app.py` as the entrypoint, and add the keys under Settings, Secrets (TOML, same names as
`.env.example`). A new app is restricted to invited viewers by default: set Settings, Sharing to
public and check it in a logged-out window.

**Render** (backup): New, Blueprint, point at this repo. `render.yaml` builds the `Dockerfile`; set
the keys in the Render dashboard (they are `sync: false`, never committed).

**Branch protection** is a repo setting, not a file, so a direct push to `main` can bypass CI until it
is applied once by an admin:

```bash
gh api repos/{owner}/{repo}/branches/main/protection -X PUT \
  -f required_status_checks[strict]=true \
  -f 'required_status_checks[contexts][]=test' \
  -f enforce_admins=true \
  -f required_pull_request_reviews=null \
  -f restrictions=null
```

## More

- [docs/technical_deep_dive.md](docs/technical_deep_dive.md): KPI formulas, architecture, design choices, limitations, roadmap
- [docs/decisions.md](docs/decisions.md): changes to what a number means
- [docs/mistakes_log.md](docs/mistakes_log.md): past mistakes and the rule each one left
- [SECURITY.md](SECURITY.md), [LICENSE](LICENSE) (MIT)
