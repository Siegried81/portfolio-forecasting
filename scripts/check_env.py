"""
check_env.py — a quick, human-readable report of which optional API keys
are configured in this environment, and what each one turns on/off.

Why this exists: every external key this app uses is OPTIONAL by design
(see config.py and every fetcher's own "fails soft" docstring) — the app
runs with zero keys configured, just with fewer features. That's the right
production behaviour, but it makes it easy to forget you never actually set
FRED_API_KEY and wonder for ten minutes why the macro panel looks empty.
This script is the two-second answer to "what did I forget to configure?"
before starting the app or opening a PR — nothing here is a hard gate; it
always exits 0 and never blocks anything, it just tells you what's set.

Usage:
    python scripts/check_env.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Make `src/` importable when this script is run directly (python scripts/check_env.py),
# same pattern as scripts/benchmark_walk_forward.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402  (import after sys.path fix, on purpose)

load_dotenv()


@dataclass(frozen=True)
class EnvCheck:
    """One env var this app reads, and what setting/not setting it changes."""
    var_name: str
    unlocks: str
    signup_url: str
    required: bool = False  # True only for something the app cannot do ANYTHING
    # useful without — as of this writing, nothing in this app meets that bar
    # (even the AI Analyst tab has a local Ollama fallback that needs no key
    # at all), so every entry below is currently False. Kept as a field
    # rather than removed so a future genuinely-required key has somewhere
    # to go without restructuring this script.


CHECKS: list[EnvCheck] = [
    EnvCheck("GROQ_API_KEY", "AI Analyst / chatbot / news digest (primary, fastest LLM backend)",
             "https://console.groq.com/keys"),
    EnvCheck("OPENROUTER_API_KEY", "AI Analyst — hosted LLM fallback #1 if Groq is down/rate-limited",
             "https://openrouter.ai/keys"),
    EnvCheck("CEREBRAS_API_KEY", "AI Analyst — hosted LLM fallback #2",
             "https://cloud.cerebras.ai/"),
    EnvCheck("SAMBANOVA_API_KEY", "AI Analyst — hosted LLM fallback #3",
             "https://cloud.sambanova.ai/apis"),
    EnvCheck("NEWSAPI_KEY", "News digest — general media coverage",
             "https://newsapi.org/register"),
    EnvCheck("FINNHUB_API_KEY", "News digest (dedicated financial news) + fundamentals + sentiment",
             "https://finnhub.io/register"),
    EnvCheck("FRED_API_KEY", "Live risk-free rate + macro & risk panel (Treasury yields, CPI, Sahm Rule, ...)",
             "https://fred.stlouisfed.org/docs/api/api_key.html"),
    EnvCheck("TWELVEDATA_API_KEY", "Market-data fallback if Yahoo Finance is unreachable + fundamentals",
             "https://twelvedata.com/pricing"),
    EnvCheck("TIINGO_API_KEY", "Market-data fallback, tier 3 — tried first among the fallbacks (most generous free tier)",
             "https://www.tiingo.com/account/api/token"),
    EnvCheck("ALPHA_VANTAGE_API_KEY", "Market-data fallback, tier 4 — only tried if Yahoo, Tiingo AND Twelve Data all fail",
             "https://www.alphavantage.co/support/#api-key"),
    EnvCheck("SEMANTIC_SCHOLAR_API_KEY", "Academic citations — works unauthenticated too, a key just raises the rate limit",
             "https://www.semanticscholar.org/product/api#api-key-form"),
    EnvCheck("HUGGINGFACE_API_KEY", "FinBERT sentiment (middle tier of the sentiment cascade, before VADER)",
             "https://huggingface.co/settings/tokens"),
    EnvCheck("REDIS_URL", "Persistent cache across restarts/replicas (production next-step, not needed locally)",
             "n/a — self-hosted or a managed Redis instance, e.g. Render's or Upstash's free tier"),
]

# Every fetcher below needs NO key at all, ever — listed here purely so the
# report is a complete picture of this app's data sources, not just the ones
# that happen to need configuration.
NO_KEY_NEEDED = [
    "yfinance (primary market data)", "SEC EDGAR (regulatory filings)",
    "GDELT (worldwide news index)", "Google News RSS (news search feed)",
    "TED (EU public-procurement notices)", "arXiv (academic preprints)",
]


def main() -> None:
    configured = [c for c in CHECKS if os.getenv(c.var_name)]
    missing = [c for c in CHECKS if not os.getenv(c.var_name)]

    print("=== Configured ===")
    if not configured:
        print("  (none)")
    for c in configured:
        print(f"  ✅ {c.var_name} — {c.unlocks}")

    print("\n=== Not configured (feature runs in its documented fallback/degraded mode) ===")
    for c in missing:
        print(f"  ⬜ {c.var_name} — {c.unlocks}")
        print(f"     Get one at: {c.signup_url}")

    print("\n=== Never need a key ===")
    for name in NO_KEY_NEEDED:
        print(f"  🔓 {name}")

    required_missing = [c for c in missing if c.required]
    if required_missing:
        # Unreachable today (see EnvCheck.required's own comment) — kept as
        # a real branch, not dead code, so a future required key is reported
        # loudly instead of silently falling into the optional list above.
        print("\n⚠️  Missing required configuration:")
        for c in required_missing:
            print(f"  {c.var_name} — {c.unlocks}")

    print(f"\n{len(configured)}/{len(CHECKS)} optional keys configured. "
          "This is informational only — every key above is optional; the app runs without any of them.")


if __name__ == "__main__":
    main()
