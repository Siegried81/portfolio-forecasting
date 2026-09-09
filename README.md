# Portfolio Forecasting & Optimization

I built this as a technical case study for BeCode's AI & Data Science bootcamp (GenAI Developer
track): an interactive Streamlit app that builds and compares three portfolios —
**historical-based**, **forecast-based**, and **realized-optimal (hindsight)** — using
mean-variance optimization, plus an AI analyst layer I added on top (LLM commentary, news digest,
grounded Q&A chatbot).

## Live demo
- Streamlit Community Cloud: [portfolio-forecasting-sieg.streamlit.app](https://portfolio-forecasting-sieg.streamlit.app/)
- Render: [portfolio-forecasting.onrender.com](https://portfolio-forecasting.onrender.com)

## What it does, in one paragraph

You pick a universe of stocks/ETFs, a date range, and a frequency. I compute historical returns,
volatility, correlation, and the max-Sharpe efficient-frontier portfolio. Then I hold out the last
N periods, forecast each asset's price over that window (ARIMA / Exponential Smoothing / naive
random walk — see "Why not Kats" below), build an optimal portfolio from the *forecasted* returns,
and compare its **actual, realized** out-of-sample performance against (a) the historical-based
portfolio and (b) the hindsight-optimal portfolio built from the *actual* returns of that same
window. An AI Analyst tab (Groq, falling back through OpenRouter, Cerebras, and SambaNova, then
local Ollama) narrates the results and answers questions about them, grounded in the computed
numbers — plus a news digest per ticker cross-referencing NewsAPI, Finnhub, SEC EDGAR,
GDELT, Google News, and TED (EU public procurement)
filings.

## Contents

- [Screenshots](#screenshots)
- [Full technical write-up](#full-technical-write-up) → [technical_deep_dive.md](technical_deep_dive.md)
- [Setup](#setup)
- [Deployment](#deployment)
- [License](#license)

## Screenshots

See **[SCREENSHOTS.md](SCREENSHOTS.md)** for the full gallery — split out of this README so the
main doc stays readable rather than being dominated by images.




## Full technical write-up

This README stays business-first on purpose — results, screenshots, how to run it. Everything
else (KPI formulas, full repo structure, every design decision vs. the original brief, the
walk-forward methodology in detail, known limitations, and the full roadmap) lives in
**[docs/technical_deep_dive.md](docs/technical_deep_dive.md)**, so neither doc has to duplicate
the other.

## Setup

```bash
git clone https://github.com/Siegried81/portfolio-forecasting && cd portfolio-forecasting
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # then fill in your keys, see below
streamlit run app.py
```

### API keys (all free tiers)

| Key | Where to get it | Required? |
|---|---|---|
| `GROQ_API_KEY` (+ optional `_2` .. `_5`) | [console.groq.com](https://console.groq.com) | Optional — AI Analyst tab falls back through the hosted providers below, then local Ollama, without it |
| `NEWSAPI_KEY` | [newsapi.org](https://newsapi.org/register) (100 req/day free) | Optional — one of three news/filing sources; digest still works with any subset configured |
| `FINNHUB_API_KEY` | [finnhub.io](https://finnhub.io/register) (free, 60 req/**min**) | Optional — dedicated financial-news API, cross-referenced alongside NewsAPI and SEC EDGAR; also the primary source for per-ticker fundamentals |
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | Optional — risk-free rate slider and macro panel fall back to a fixed 4% default / n/a without it |
| `TWELVEDATA_API_KEY` | [twelvedata.com](https://twelvedata.com) (free, 800 req/day) | Optional — fallback market data source if Yahoo Finance is unreachable, and fallback fundamentals source (mainly covers `AAPL` on the free tier — see caveat below) |
| `TIINGO_API_KEY` | [tiingo.com](https://www.tiingo.com/account/api/token) (free, 500 req/hour) | Optional — third market-data tier, tried FIRST among the three fallbacks (most generous free tier of the three) |
| `ALPHA_VANTAGE_API_KEY` | [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key) (free, 25 req/day) | Optional — fourth, last-resort market-data tier, only tried if Yahoo, Tiingo AND Twelve Data all fail |
| `SEMANTIC_SCHOLAR_API_KEY` | [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api) | Optional — genuinely optional even without one: the search endpoint works unauthenticated at a lower shared rate limit; a key just raises it |
| `HUGGINGFACE_API_KEY` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (free) | Optional — but required for the FinBERT sentiment tier specifically; without it that tier is skipped and the cascade falls to Finnhub then VADER |
| `OPENROUTER_API_KEY` | [openrouter.ai](https://openrouter.ai) (free-tier models available) | Optional — 1st hosted LLM fallback tier, tried if every Groq key fails |
| `CEREBRAS_API_KEY` | [cloud.cerebras.ai](https://cloud.cerebras.ai) (free tier available) | Optional — 2nd hosted LLM fallback tier, independent account/infrastructure from OpenRouter |
| `SAMBANOVA_API_KEY` | [cloud.sambanova.ai](https://cloud.sambanova.ai) (free tier available) | Optional — 3rd hosted LLM fallback tier, tried last among the hosted providers |
| Ollama (local fallback) | `ollama serve` + `ollama pull llama3.1` — [ollama.com](https://ollama.com) | Optional — final tier, only reachable when self-hosting locally; silently unreachable in a deployed container |

**News digest cross-referencing.** The AI Analyst's news digest pulls from six genuinely
different source types rather than one: NewsAPI (general media), Finnhub (dedicated financial
news, far more generous free-tier quota), SEC EDGAR full-text search (free, no key —
primary-source 8-K "material event" filings, not journalism about the company), GDELT (free, no
key — a worldwide news index reaching well beyond the US/English-language press), Google News
RSS (free, no key — a fast cross-check search feed), and TED (free, no key — the EU's official
public-procurement notice database; NOT "TED Talks", see `technical_deep_dive.md`). Each headline
in the UI's "Sources" expander is tagged with which provider it came from; the LLM prompt
explicitly treats SEC filings and TED notices as more authoritative primary sources than media
coverage of the same event, and notes when multiple sources corroborate the same story. Works
with any subset of the two key-requiring sources (NewsAPI, Finnhub) configured — the other four
need no key at all — missing a key just means fewer sources, not a broken feature.

Market data (`yfinance`) needs no key. The app runs fully without *any* key — you just lose the
AI Analyst tab's content and the live risk-free rate, not the core optimization/forecasting/
comparison functionality.

**Groq key rotation.** Free-tier Groq accounts hit daily/per-minute rate limits fast, especially
during a demo. Up to 5 keys can be set (`GROQ_API_KEY`, `GROQ_API_KEY_2` .. `GROQ_API_KEY_5`) —
`llm_client.py` tries them in order and advances to the next one **only on a 429 rate-limit
error**. A non-rate-limit error (bad key, deprecated model) fails immediately to the hosted
fallback chain instead of burning time cycling through keys that all share the same problem.
Same pattern already proven on the Innovation Radar project's `llm_client.py`.

**Why hosted fallbacks sit BETWEEN Groq and Ollama.** Ollama alone left a real gap: it only runs on whatever
machine has it installed, so it's a genuine fallback in local dev but silently unreachable once
this app is deployed (Render/Streamlit Community Cloud have no Ollama daemon in the container) —
a Groq outage in production had no working fallback at all before this. OpenRouter, Cerebras, and
SambaNova are all hosted (work in prod, not just on a dev laptop) and OpenAI-compatible (same
request/response shape this app already speaks to Groq with — plain `requests`, no extra SDK per
provider; see `llm_client.py`'s `_call_openai_compatible_provider`, the one shared implementation
all three use). Cerebras and SambaNova are, like Groq, dedicated fast-inference hardware
providers — genuine redundancy against each other (independent accounts/infrastructure), not a
random third pick. Configuring any subset of the three works: `_fallback_providers()` only tries
whichever ones actually have a key set, in the fixed order OpenRouter → Cerebras → SambaNova,
skipping the rest. Ollama stays as the final tier: free and unlimited, but only useful to
whoever is running this locally.

**Live risk-free rate (FRED).** Rather than a hardcoded guess, the sidebar's risk-free rate slider
pre-fills with the actual current 3-month T-bill yield (FRED series `DGS3MO`) when `FRED_API_KEY`
is set — still fully overridable by hand. `Alpha Vantage` and `Finnhub` were considered too, but
both mostly duplicate what `yfinance` (prices) and `NewsAPI` (headlines) already cover; FRED adds
a genuinely new, finance-relevant data point (a real macro rate) instead of a redundant one.

### Run the tests

```bash
pytest tests/ -v
```

320 tests across 18 files, all passing — `mypy --strict` is also clean on every file in `src/`.

### Check your environment

```bash
python scripts/check_env.py
```

Every API key this app uses is optional (see the table above) — the app runs with zero keys
configured, just with fewer features. This script is a two-second report of which keys are set
in your `.env`, what each one unlocks, and a signup link for anything missing — nothing here
blocks or gates anything, it's purely informational.

### Capacity benchmarking

```bash
python scripts/benchmark_walk_forward.py --quick          # fast subset — Naive/ETS only, no ARIMA
python scripts/benchmark_walk_forward.py                  # full grid — adds ARIMA + a 40-asset PCA-covariance universe
python scripts/benchmark_walk_forward.py --csv bench.csv  # also write results to CSV, to track over time
```

Wall-clock time and peak (Python-heap) memory for `backtesting.run_walk_forward` — the single
most compute-heavy path in the app — across a grid of universe sizes, window counts, forecast
models, and the GARCH forecast-covariance toggle. Synthetic price data (same generator
`tests/test_backtesting.py` uses), so no network call and no market-data dependency. Not part of
`pytest`/CI gating on purpose: this is for capacity planning before a real deployment (how long
does one run take, how does that scale with universe size), not a pass/fail correctness check —
CI only runs it in `--quick` mode as a non-blocking smoke test that the script itself still works.

### Troubleshooting: `pandas` fails to build from source

If `pip install -r requirements.txt` fails while compiling `pandas` (a wall of Cython/C++
compiler output ending in a `meson`/`ninja` error), you're very likely on **Python 3.14 or
newer** on a system where pip has no prebuilt wheel to fall back on for an older `pandas` pin.
`requirements.txt` uses minimum-version pins (`pandas>=3.0.5`, not `==2.2.2`) specifically to
avoid this — pandas only shipped working Python 3.14 wheels starting at 3.0.5 (3.0.0-3.0.4 have
a confirmed segfault regression on 3.14). If you still hit a build error:

```bash
python --version              # confirm which Python the venv is actually using
pip install --upgrade pandas numpy scipy statsmodels streamlit   # force the latest wheels
```

If that still fails, your Python is newer than every dependency has wheels for yet — the
reliable fix is a slightly older interpreter (3.12 is the safest bet) via `pyenv`,
`uv python install 3.12`, or your distro's package manager, then recreating the venv with it.

### Troubleshooting: Yahoo Finance returns no data / `crumb = 'Edge: Too Many Requests'`

Yahoo Finance's anti-bot cookie/crumb handshake (which `yfinance` depends on) is a widely reported issue across the `yfinance`
community, not specific to this app or your network. The app handles this with a five-step
chain, each step only running if the previous one actually failed:

1. **`yfinance` library** — with retry-with-backoff (3 attempts). This is "the Yahoo Finance API"
   as named in the brief.
2. **Direct Yahoo Finance REST API** — bypasses the `yfinance` library entirely, in case its
   cookie/crumb handling specifically (not Yahoo itself) is the point of failure. Same underlying
   source, different code path. Honest expectation: this sits behind the same anti-bot layer, so
   it's cheap insurance rather than a reliable fix — included because it's literally what the
   brief specifies, not because it's expected to outperform the library.
3. **Tiingo** — a genuinely different provider, tried FIRST among the three fallbacks once both
   Yahoo-based attempts are exhausted: its free tier (500 req/hour) is meaningfully more generous
   than the other two.
4. **Twelve Data** (free key, 800 req/day) — tried if Tiingo also fails.
5. **Alpha Vantage** — a fourth, LAST-RESORT provider, only reached if both Tiingo and Twelve Data
   fail. Its free tier (25 req/DAY, one ticker per call — no batch endpoint) is the stingiest of
   the four market-data sources this app knows about, which is exactly why it sits last.

The Overview tab shows a caption indicating which source actually served the data. Set
`TWELVEDATA_API_KEY` (and optionally `TIINGO_API_KEY`/`ALPHA_VANTAGE_API_KEY`) in `.env` to enable
steps 3-5 — without them, the app surfaces Yahoo's error once both Yahoo-based steps fail.

**A wall of `Failed to get ticker '...' reason: Expecting value...` in the terminal does NOT mean
the app is broken.** yfinance logs every failed attempt loudly, including the ones this app's own
fallback chain expects and handles — if you see ARIMA/statsmodels fit warnings (or any forecasting
output) appear AFTER that wall of text, prices loaded successfully via a later step in the chain
(most likely Twelve Data); check the Overview tab's data-source caption to confirm which one. Only
a red `st.error("Could not load market data: ...")` banner in the browser itself means every
fallback failed and nothing loaded.

If you still see no data after all five:
```bash
rm -rf ~/.cache/py-yfinance   # clears a possibly-stale cached cookie
```
then retry, or wait a few minutes — Yahoo's blocks are usually transient, not permanent.

**WSL-specific: `OperationalError('unable to open database file')`.** yfinance keeps its own local
SQLite cache (cookies + per-ticker timezone lookups); on WSL, if that cache path resolves onto a
Windows-mounted drive (`/mnt/c`, `/mnt/d`, DrvFs/9p) rather than the WSL distro's native
filesystem, SQLite can't lock the file there and every ticker touching the cache fails this way
(e.g. `'^VIX'`, while the main tickers still succeed via the Twelve Data fallback, a
different code path that never touches this cache). `market_data.py`
redirects yfinance's cache to the system temp directory at import time. If you're on a setup
where that redirect doesn't apply, `rm -rf ~/.cache/py-yfinance` (or
wherever `yf.get_yf_data_dir()` reports) is the manual equivalent.

**`gio: http://localhost:8501: Operation not supported` on startup is harmless noise, not an app
error.** Streamlit tries to auto-open your default browser via WSL's `gio` URI-opener, which isn't
configured to reach a Windows browser from inside WSL by default — the server itself starts fine
regardless; just open the Local URL manually. Run `streamlit run app.py --server.headless true` to
stop Streamlit from attempting this at all.

### Docker (local)

```bash
docker build -t portfolio-forecasting .
docker run -p 8501:8501 --env-file .env portfolio-forecasting
```

## Deployment

**Streamlit Community Cloud** (primary): push to GitHub → [share.streamlit.io](https://share.streamlit.io)
→ New app → point at this repo, `app.py` as the entrypoint → add `GROQ_API_KEY` / `NEWSAPI_KEY`
under app Settings → Secrets (TOML format, same keys as `.env.example`).

**Render** (backup): push to GitHub → Render dashboard → New → Blueprint → point at this repo.
`render.yaml` provisions the service from the `Dockerfile` automatically; set `GROQ_API_KEY` and
`NEWSAPI_KEY` in the Render dashboard's environment variables (they're marked `sync: false` in
`render.yaml` so they're never committed).

**Before pushing:** run `git diff --cached` to check nothing sensitive slipped into a config
file, and keep a local backup of the repo before pulling/pushing on a shared/team remote.

**CI & branch protection.** `.github/workflows/ci.yml` runs `pytest` and `mypy --strict` (both
blocking) on every push/PR, plus a `--quick` capacity-benchmark smoke test (non-blocking). This
is a workflow file, not a repo setting — it can't stop a direct push to `main` from bypassing it
on its own. Branch protection is a GitHub repository setting, not something committable in this
repo, so it has to be applied once, by hand, by whoever administers the GitHub repo:

```bash
# via the gh CLI, once, from repo admin:
gh api repos/{owner}/{repo}/branches/main/protection -X PUT \
  -f required_status_checks[strict]=true \
  -f 'required_status_checks[contexts][]=test' \
  -f enforce_admins=true \
  -f required_pull_request_reviews=null \
  -f restrictions=null
```

or via the UI: repo → **Settings → Branches → Add branch protection rule** → branch name pattern
`main` → check "Require status checks to pass before merging" → select the `test` check from
`ci.yml` above.
