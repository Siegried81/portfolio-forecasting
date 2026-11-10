# Test harness: the checks around this code

This file is the part of the project that doesn't show up in the diff. It's not about the
portfolio optimizer itself. It's about the checks I built around the code so a change is
right, not just plausible: a precise spec, a small set of files that hold the context, tests
that fail loudly, and a log of the mistakes I've made so I don't make them twice.

## The parts, and where each one lives in this repo

**1. A precise spec for each change**
Every change in this project started from a specific ask I could check: fix this bug, add
this metric with this formula, wire it into this exact table. Never "make the app better."
The "Understanding the KPIs" section of `README.md` is itself a spec. Each metric's formula is
written down before any code computes it, and apart from that code, so there's something
precise to test against.

**2. One place for the context**
`config.py` holds every default, env var and constant. It's the first file to open to answer
"what's the default risk-free rate?" or "which env var sets the Groq model?". Keeping that in
one small file, not spread across the codebase, keeps every change easy to check.

**3. A real feedback loop**
`pytest` (530 tests across 22 files, run by `.github/workflows/ci.yml` on every push) and
`mypy --strict` (blocking, and the whole codebase passes) turn "looks right" into "is right."
Almost every fix in this project's history was caught because a test failed (or, for one
entry in the log below, mypy itself). Not because I read the diff closely enough.

**4. Written guardrails: what NOT to do**
They're in code comments and in the README's "Key design decisions" table, not left implicit.
Don't use Kats (I chose against it for compatibility with modern pandas/numpy, not because
it's abandoned: its repo still gets commits). Don't assume Twelve Data's free tier covers
`/statistics` for any ticker. Don't divide every FRED series by 100 (`SAHMREALTIME` isn't a
percentage of 100, see the log). Don't let one bad Groq key stop the other four. Don't send
an LLM prompt's full context with no limit (see the log). Each one is a mistake I actually
made once and then turned into a standing rule.

**5. Tests first**
For every new metric (Jensen's Alpha, Ulcer Index, skewness/kurtosis, the Hurst exponent,
the ADF stationarity test), I wrote down the exact formula and the expected result on a case
I could check by hand before writing the code.

**6. A recurring-errors log that I actually reread**
This is the part most teams skip. Below is the real log from this project, not a made-up
example. It's here so the next change doesn't rediscover the same failure from scratch.

## Recurring-errors log (this repo, real incidents)

| # | What broke | Root cause | Fix, now a standing rule |
|---|---|---|---|
| 1 | Twelve Data multi-ticker batch silently returned zero data | Shape-detection logic assumed `all(t in payload for t in tickers)` — broke the moment one bad ticker was silently dropped from the response instead of kept as an error-tagged key | Detect response shape from `"values"`/`"meta"` presence, never from whether every requested ticker is present |
| 2 | Groq key rotation abandoned all 5 keys on one bad key | Any non-`RateLimitError` exception re-raised immediately, treating an individually-revoked key the same as a systemic failure | Rotate on `RateLimitError` **and** `AuthenticationError`/`NotFoundError` (key-specific); fail fast only on errors that affect every key equally |
| 3 | Every macro/market-data test raised `FrozenInstanceError` | `LLMSettings` is a deliberately immutable `@dataclass(frozen=True)` — `monkeypatch.setattr(obj, "field", value)` can't mutate a field on it | Never patch a field directly; `dataclasses.replace()` the whole object and monkeypatch the *name* in the importing module |
| 4 | Sahm Rule recession indicator would have silently read as permanently "no recession" | Reused the generic FRED "latest value, divide by 100" fetcher — but `SAHMREALTIME` is already expressed in the units its own 0.50 threshold uses | Every FRED series' unit convention gets verified explicitly (`divide_by` parameter), never assumed from the pattern of the series fetched just before it |
| 5 | Hurst-exponent test asserted the wrong thing and would have shipped a false confidence in the metric | Tested "trending" with a deterministic linear trend + noise — mathematically the wrong test case for a variance-of-lagged-differences (fBm-style) estimator, which measures increment self-similarity, not a naive slope | Verify statistical/financial formulas **numerically** against a hand-built case before trusting the test, not just checking the code compiles |
| 6 | Ulcer Index test asserted `0.0 > 0.0` — passed for the wrong reason it didn't even run | Synthetic return series dropped 20% on the very first period; `max_drawdown()`'s running peak comes from `cummax()` over the series itself, so period 0 has no prior peak to be measured against | Any drawdown-based test needs an explicit up-move establishing a peak *within* the series before the drop being tested |
| 7 | Local Python version (tested against 3.14-specific pandas/numpy fixes) doesn't match the Dockerfile's `python:3.12-slim` | Never actually run end-to-end after the fact — flagged, not verified | Resolved: full suite run against Python 3.12.3 with the pinned `requirements.txt`, all tests pass — the version-floor pins already covered 3.12-3.14, they just hadn't been exercised on 3.12 |
| 8 | Sidebar's risk-free-rate and max-weight sliders displayed "0%"/"1%" instead of "4%"/"35%" | Assumed `st.slider(format="%.0f%%")` multiplies the underlying fraction by 100 for display — it doesn't, the format string only controls rendering of the raw value | Run percent-like sliders in percentage-point units (0-100) and convert to a fraction immediately after, never rely on a `%` format string to rescale |
| 9 | `truncate_to_token_budget` could crash with an uncaught `HTTPError` instead of degrading | `_count_tokens` had a try/except fallback for a blocked `tiktoken` encoding download, but the truncation branch called `tiktoken.get_encoding()` a second time, unguarded | Any function documented as "fails soft" needs every one of its own internal calls to the same flaky dependency guarded, not just the first one — caught by a test that mocks the encoding call to fail |
| 10 | The Theta forecast's drift was twice too steep. A first version kept half the slope on a straight line. I read that as a bug and "fixed" it to the full slope. | Half the slope is the right answer: Hyndman & Billah (2003) show Theta is SES plus a drift of half the OLS slope. My hand-worked "correct" case was wrong, so the test locked in the error. | Check a statistical method against a reference implementation, not only against my own idea of the answer. `theta_forecast` now matches statsmodels' `ThetaModel` to 1e-8 and a test checks it. |
| 11 | `efficient_frontier_points` crashed with `OptimizationError: Solver status: infeasible` the moment a single ticker was selected | The function always swept a RANGE of target returns to sample the frontier — with one asset, weight is forced to 100% and there is exactly one achievable return, so every other target in the sweep is mathematically infeasible by construction, not a solver problem | Any function that sweeps a range of targets over an optimizer needs an explicit degenerate-input check (here: `len(mu) < 2` or no spread between min/max achievable return) before the sweep, not just a try/except around each point |
| 12 | Every ticker's news sentiment read "Neutral (score +0.00)" after migrating FinBERT off the deprecated `api-inference.huggingface.co` domain | The new `router.huggingface.co/hf-inference` endpoint wraps a single-input result in an extra list layer (`[[{"label": ..., "score": ...}, ...]]`) instead of the old flat shape (`[{"label": ..., "score": ...}, ...]`) — `isinstance(item, dict)` silently filtered out every element (each one a list, not a dict), leaving positive=negative=0.0 with no exception anywhere | Any response-shape assumption from a migrated/renamed API endpoint gets re-verified against a live capture, not assumed unchanged just because the request shape and status code still work — `fetch_finbert_sentiment` now unwraps one extra list layer when present, with a regression test covering both the old flat shape and the new nested one |
| 13 | `mypy --strict` failed on `src/forecasting.py` the first time it was actually run against this exact codebase | `_get_gradient_boosted_regressor` returns one of two unrelated external classes (XGBoost's or scikit-learn's regressor) with no return-type annotation at all | Any helper returning a third-party type this module doesn't want to import at type-check time gets an explicit `-> Any` rather than an unannotated signature — silent under `--ignore-missing-imports` alone, but not under `--strict` |
| 14 | Chatbot crashed the entire LLM fallback cascade (Groq → OpenRouter → Cerebras → SambaNova → Ollama, all failing at once) | `answer_portfolio_question`'s system prompt (persona + portfolio data + retrieved news + academic references) was never token-bounded — only the news-digest block was; a long-enough context (9098 tokens) exceeded Groq's 8000 TPM limit, and the resulting 413 wasn't a Groq-specific outage, so every fallback failed identically | Apply `truncate_to_token_budget` to the full assembled `system_content`, not just individual data blocks, before every `chat()` call |

## What this gets me, concretely

- **For this bootcamp project**: every fix above shipped with a regression test (or, for #13,
  a passing `mypy --strict` run), so none of these fourteen mistakes can silently come back.
- **For an interview**: "my tests pass" is easy to say. "Here's my log of real mistakes with
  their root causes, and here's the test that catches each one again" is harder to fake.
- **Going forward**: this file is versioned with the code so it stays current. A harness
  nobody maintains is just a changelog.