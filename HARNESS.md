# Harness Engineering — how I actually work with an AI coding assistant

I'm writing this down because it's the part of this project that doesn't show up in the
diff: not "I prompted Claude to build a portfolio optimizer," but the environment of
constraints, context, and feedback I built around that collaboration to make the output
reliable. The concept is called **Harness Engineering** — popularized by Birgitta
Böckeler (Thoughtworks) and referenced since by both OpenAI and Anthropic in their own
agentic-coding guidance: an AI coding agent's reliability comes less from the model
itself than from the "harness" around it — the task specification, the context it's
given, and the feedback loop that catches what it gets wrong.

I'm treating this file as both documentation and evidence — it's the concrete answer to
"how do you actually use AI in your engineering work," not an abstract claim.

## The components, and how each one shows up in this repo

**1. Precise task specification** — little room for interpretation.
Every change request in this project started from a specific, falsifiable ask (fix this
bug, add this metric with this formula, wire it into this exact table), not "make the
app better." `README.md`'s "Understanding the KPIs" section is itself a spec: every
metric's formula is written down before — and independently of — any code that computes
it, so an AI assistant (or a teammate) has something precise to implement against, not a
vague description to guess at.

**2. Curated context**
`config.py` is deliberately the single source of truth for every default, env var, and
constant — the first (and often only) file an assistant needs to see to answer "what's
the current risk-free rate default" or "which env var controls the Groq model." Keeping
that context small and canonical, instead of scattering constants across the codebase,
is what makes "give the assistant the right context" actually tractable.

**3. A real feedback loop**
`pytest` (367 tests across 19 files, `.github/workflows/ci.yml` running it on every push) plus
`mypy --strict` (blocking — the full suite passes clean) are the loop that turns "looks
right" into "is right." Nearly every fix in this project's history was caught *because*
a test — or, for one entry in this project's own log below, mypy itself — failed, not
because a human read the diff carefully enough to notice.

**4. Explicit guardrails — what NOT to do**
Documented directly in code comments and the README's "Key design decisions" table,
not left implicit: don't use Kats (chosen against for compatibility with modern pandas/numpy, not because it's
abandoned — its repo shows ongoing commits), don't assume Twelve Data's free tier covers
`/statistics` for arbitrary tickers, don't divide every FRED series by 100 (`SAHMREALTIME`
isn't a percentage-of-100 figure — see the recurring-errors log), don't let a single Groq
key failure abandon the other four, don't send an LLM prompt's full context unbounded (see
the recurring-errors log). Each is a mistake that was actually made once during development and then turned into a permanent constraint.

**5. Test-driven prompting**
For every new metric added this session (Jensen's Alpha, Ulcer Index, skewness/kurtosis,
the Hurst exponent, the ADF stationarity test), the request specified the exact formula
and the exact expected behavior on a hand-checkable case *before* the implementation —
the same discipline as writing the test first, just expressed as the prompt itself.

**6. A recurring-errors log, actually reinjected into context**
This is the part most teams skip. Below is the real log from this project — not a
hypothetical — kept here specifically so the next session (mine or an assistant's)
doesn't rediscover the same failure mode from scratch.

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
| 10 | A first implementation of the Theta forecasting method silently under-forecast any real trend — a pure linear-trend test case came back roughly half the true slope | Applied SES (Simple Exponential Smoothing, which has no trend term) directly to an amplified "theta line" derived from the raw series, instead of detrending first — SES has no way to extrapolate a slope it was never shown | Verify a new forecasting/statistical formula against a case with a KNOWN correct answer (here: a noiseless linear trend, where the right forecast is exactly computable by hand) before trusting it, not just checking it runs and produces plausible-looking numbers |
| 11 | `efficient_frontier_points` crashed with `OptimizationError: Solver status: infeasible` the moment a single ticker was selected | The function always swept a RANGE of target returns to sample the frontier — with one asset, weight is forced to 100% and there is exactly one achievable return, so every other target in the sweep is mathematically infeasible by construction, not a solver problem | Any function that sweeps a range of targets over an optimizer needs an explicit degenerate-input check (here: `len(mu) < 2` or no spread between min/max achievable return) before the sweep, not just a try/except around each point |
| 12 | Every ticker's news sentiment read "Neutral (score +0.00)" after migrating FinBERT off the deprecated `api-inference.huggingface.co` domain | The new `router.huggingface.co/hf-inference` endpoint wraps a single-input result in an extra list layer (`[[{"label": ..., "score": ...}, ...]]`) instead of the old flat shape (`[{"label": ..., "score": ...}, ...]`) — `isinstance(item, dict)` silently filtered out every element (each one a list, not a dict), leaving positive=negative=0.0 with no exception anywhere | Any response-shape assumption from a migrated/renamed API endpoint gets re-verified against a live capture, not assumed unchanged just because the request shape and status code still work — `fetch_finbert_sentiment` now unwraps one extra list layer when present, with a regression test covering both the old flat shape and the new nested one |
| 13 | `mypy --strict` failed on `src/forecasting.py` the first time it was actually run against this exact codebase | `_get_gradient_boosted_regressor` returns one of two unrelated external classes (XGBoost's or scikit-learn's regressor) with no return-type annotation at all | Any helper returning a third-party type this module doesn't want to import at type-check time gets an explicit `-> Any` rather than an unannotated signature — silent under `--ignore-missing-imports` alone, but not under `--strict` |
| 14 | Chatbot crashed the entire LLM fallback cascade (Groq → OpenRouter → Cerebras → SambaNova → Ollama, all failing at once) | `answer_portfolio_question`'s system prompt (persona + portfolio data + retrieved news + academic references) was never token-bounded — only the news-digest block was; a long-enough context (9098 tokens) exceeded Groq's 8000 TPM limit, and the resulting 413 wasn't a Groq-specific outage, so every fallback failed identically | Apply `truncate_to_token_budget` to the full assembled `system_content`, not just individual data blocks, before every `chat()` call |

## What this gets me, concretely

- **For this bootcamp project**: every fix above shipped with a regression test (or, for #13,
  a passing `mypy --strict` run), so none of these thirteen mistakes can silently come back.
- **For an interview**: "I use AI to write code" is table stakes. "Here's my
  recurring-errors log with the actual root causes, and here's why my test suite is
  built to catch each one again" is a materially different, harder-to-fake claim — and
  it's the actual GenAI Developer skill (structuring a reliable human+AI system), not
  just prompting.
- **Going forward**: this file is versioned alongside the code specifically so it stays
  current — a harness that isn't maintained is just a changelog.