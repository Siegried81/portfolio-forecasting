"""
AI-powered features layered on top of the finance engine: a narrative commentary
on portfolio results, a news digest per ticker, and a grounded Q&A chatbot.

Grounding strategy (explicit, and worth stating for anyone reviewing this against
the ethics/RAG material): this is CONTEXT INJECTION, not full RAG. The computed
metrics/weights/news are serialised into the system prompt so the model answers
from the actual numbers on screen rather than from memorised training data — this
removes most hallucination risk for THIS narrow use case (a fixed, small set of
facts) without the complexity of a vector store, which would be overkill for a
handful of numbers that already fit in a prompt.
"""
from __future__ import annotations

import concurrent.futures
from typing import Any

import pandas as pd

from src.llm_client import chat, truncate_to_token_budget
from src.news_data import (
    fetch_finnhub_news,
    fetch_gdelt_news,
    fetch_google_news_rss,
    fetch_sec_filings,
    fetch_sec_insider_trades,
    fetch_ted_notices,
    fetch_ticker_headlines,
    get_ticker_sentiment,
)

SYSTEM_PERSONA = (
    "You are a CFA-level portfolio analyst writing for an informed but non-technical "
    "reader. Be precise, quantitative, and honest about uncertainty. Never invent "
    "numbers that are not given to you in the context — if something isn't in the "
    "context, say so explicitly rather than guessing.\n\n"
    "FORMATTING: never use LaTeX or markdown math syntax (\\;, \\text{}, \\[...\\], "
    "$...$) — this app renders plain text, not math markup, so LaTeX shows up as "
    "raw, broken-looking symbols. Write any formula in plain words and standard "
    "keyboard symbols instead, e.g. 'Sortino ratio = (mean excess return) / "
    "(downside deviation), annualised by multiplying by the square root of the "
    "number of periods per year'.\n\n"
    "CRITICAL: sanity-check annualised figures before reporting them at face value. "
    "An annualised return is calculated by compounding a short window's actual return "
    "out to a full year — over a short window (a handful of weeks or months), this "
    "can turn an ordinary result into a triple-digit annualised number that never "
    "actually occurred and is not a repeatable rate. If annualised return exceeds "
    "roughly 50-80% or Sharpe exceeds roughly 3, and the context shows the underlying "
    "window is short (well under a year, i.e. few periods), you MUST say so explicitly "
    "— e.g. 'this annualises a strong N-period run into an inflated headline figure; "
    "the raw period return of X% is the more honest read of what actually happened.' "
    "Never describe volatility as 'modest' or a portfolio as low-risk purely because "
    "its annualised return is very high — a sky-high annualised return is itself a "
    "signal to look at the raw window return, not evidence of a great risk-adjusted "
    "outcome.\n\n"
    "Example of applying this rule correctly. If the context shows:\n"
    "  Sharpe: 6.29, computed over a 90-period (not annualised) window, raw period return 35.2%\n"
    "GOOD: \"This portfolio's annualised Sharpe of 6.29 is inflated by compounding a strong "
    "90-period run out to a full year — the raw 35.2% period return is the more honest read "
    "of what actually happened; a Sharpe this high over such a short window is a red flag "
    "about the window, not evidence of a repeatable edge.\"\n"
    "BAD (never write this): \"This portfolio shows an outstanding annualised Sharpe of 6.29, "
    "reflecting excellent risk-adjusted performance.\" — this states the inflated number as if "
    "it were a reliable, repeatable result, with no caveat about the short window that produced it."
)


def _wrap_context(tag: str, content: str) -> str:
    """
    Wrap a block of injected DATA (computed metrics, retrieved news, retrieved
    papers) in an XML tag, so the model can clearly tell it apart from the
    surrounding INSTRUCTIONS (plain prose in the rest of the system prompt).
    Recommended practice for a prompt this long, and especially useful now
    that `chat()` can route to five different providers of varying
    instruction-following discipline (Groq, OpenRouter, Cerebras, SambaNova,
    Ollama) — not just Claude, which tends to track unmarked prose structure
    more reliably than smaller/less-tuned models do. Every call site below
    can then refer to the tag by name in its instructions ("only cite numbers
    from <portfolio_data>") instead of a vaguer "the context above".
    """
    return f"<{tag}>\n{content}\n</{tag}>"


def _format_metrics_block(label: str, metrics: dict[str, float]) -> str:
    lines = [f"{label}:"]
    if "period_return" in metrics and "n_periods" in metrics:
        lines.append(f"  Raw return over the {metrics['n_periods']}-period window (NOT annualised): {metrics['period_return']:.2%}")
    lines += [
        f"  Annual return (compounded from the window above — see caveat if window is short): {metrics['annual_return']:.2%}",
        f"  Annual volatility: {metrics['annual_volatility']:.2%}",
        f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}",
        f"  Sortino ratio: {metrics['sortino_ratio']:.2f}",
        f"  Max drawdown: {metrics['max_drawdown']:.2%}",
        f"  95% VaR (period): {metrics['var_95']:.2%}",
        f"  95% CVaR (period): {metrics['cvar_95']:.2%}",
    ]
    return "\n".join(lines) + "\n"


def build_results_context(
    weights: pd.Series,
    historical_metrics: dict[str, float],
    forecast_metrics: dict[str, float] | None,
    realized_metrics: dict[str, float] | None,
    macro_context: dict[str, Any] | None = None,
) -> str:
    """Serialise the app's computed results into a compact text block reused as
    grounding context for BOTH the commentary generator and the chatbot — one
    source of truth, so the two features can never disagree with each other."""
    lines = ["PORTFOLIO WEIGHTS (optimal, max-Sharpe):"]
    for ticker, w in weights.items():
        # abs(), not w > 0.001: a plain positive-only filter would silently drop
        # SHORT positions (negative weights) when short selling is enabled,
        # which would make the LLM describe a fully-invested short as an
        # unexplained gap in the weights instead of an actual short position.
        if abs(w) > 0.001:
            lines.append(f"  {ticker}: {w:.1%}")

    lines.append("")
    lines.append(_format_metrics_block("HISTORICAL-BASED PORTFOLIO (realised, in-sample)", historical_metrics))
    if forecast_metrics:
        lines.append(_format_metrics_block("FORECAST-BASED PORTFOLIO (realised out-of-sample)", forecast_metrics))
    if realized_metrics:
        lines.append(_format_metrics_block("REALIZED-OPTIMAL PORTFOLIO (hindsight benchmark)", realized_metrics))

    if macro_context:
        macro = macro_context.get("macro", {})
        vix = macro_context.get("vix_level")
        macro_lines = ["MACRO & RISK BACKDROP (current, not historical to the portfolio period):"]
        if macro.get("three_month_yield") is not None:
            macro_lines.append(f"  3-month T-bill yield: {macro['three_month_yield']:.2%}")
        if macro.get("ten_year_yield") is not None:
            macro_lines.append(f"  10-year Treasury yield: {macro['ten_year_yield']:.2%}")
        if macro.get("term_spread_10y_3m") is not None:
            spread = macro["term_spread_10y_3m"]
            note = " (INVERTED — a historical recession signal)" if spread < 0 else ""
            macro_lines.append(f"  10Y-3M term spread: {spread:+.2%}{note}")
        if macro.get("cpi_yoy_inflation") is not None:
            macro_lines.append(f"  CPI inflation (year-over-year): {macro['cpi_yoy_inflation']:.2%}")
        if macro.get("unemployment_rate") is not None:
            macro_lines.append(f"  Unemployment rate: {macro['unemployment_rate']:.2%}")
        if macro.get("fed_funds_rate") is not None:
            macro_lines.append(f"  Fed funds rate (effective, daily): {macro['fed_funds_rate']:.2%}")
        if macro.get("sahm_rule_indicator") is not None:
            sahm = macro["sahm_rule_indicator"]
            note = " (>=0.50 — has historically marked the start of every US recession since 1970)" if sahm >= 0.50 else ""
            macro_lines.append(f"  Sahm Rule recession indicator: {sahm:.2f}{note}")
        if macro.get("credit_spread_baa10y") is not None:
            macro_lines.append(f"  Baa corporate credit spread (vs 10Y Treasury): {macro['credit_spread_baa10y']:.2%}")
        if vix is not None:
            regime = "calm" if vix < 15 else "normal" if vix < 25 else "elevated"
            macro_lines.append(f"  VIX (market fear gauge): {vix:.1f} ({regime})")
        if len(macro_lines) > 1:  # only include if we actually got at least one live figure
            lines.append("")
            lines.append("\n".join(macro_lines))

    return "\n".join(lines)


def generate_commentary(results_context: str) -> tuple[str, str]:
    """Narrative, analyst-style commentary on the computed portfolio results.
    Returns (text, backend_used)."""
    messages = [
        {"role": "system", "content": SYSTEM_PERSONA},
        {
            "role": "user",
            "content": (
                "Write a concise (150-200 words) portfolio commentary based ONLY on the "
                "data in <portfolio_data> below. Cover EXACTLY these four points, in this "
                "order, and do not add a fifth section or a separate conclusion/take-away "
                "beyond them: "
                "(1) what drove the historical allocation, (2) how the "
                "forecast-based portfolio compares to the realized-optimal benchmark and "
                "what that gap implies about forecast reliability, (3) one concrete risk "
                "to flag (drawdown, concentration, or volatility), and (4) if a macro "
                "backdrop is provided, one line on how it colours the picture (e.g. an "
                "inverted yield curve or an elevated VIX warrants more caution than the "
                "portfolio numbers alone would suggest). Do not give investment advice or "
                "recommendations to buy/sell.\n\n"
                "Format: flowing prose in 4 short paragraphs (one per point above) — no "
                "bullet points, no numbered list, no markdown headers, no bold labels.\n\n"
                + _wrap_context("portfolio_data", results_context)
            ),
        },
    ]
    return chat(messages, temperature=0.4, max_tokens=4000)


def _fetch_all_sources_for_ticker(ticker: str, company: str | None) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None]:
    """One ticker's worth of the six-source fetch, plus sentiment — factored
    out so it can run inside a thread pool. Each individual fetcher already
    fails soft (returns []/None), so this never raises. Sentiment is computed
    HERE (not in a separate pass afterwards) so it rides the same thread pool
    round-trip instead of adding a second sequential wait per ticker."""
    articles = (
        fetch_ticker_headlines(ticker, company)
        + fetch_finnhub_news(ticker)
        + (fetch_sec_filings(company, ticker) if company else [])
        + (fetch_sec_insider_trades(company, ticker) if company else [])
        + fetch_gdelt_news(ticker, company)
        + fetch_google_news_rss(ticker, company)
        + fetch_ted_notices(ticker, company)
    )
    for a in articles:
        a["ticker"] = ticker
    sentiment = get_ticker_sentiment(ticker, articles)
    return ticker, articles, sentiment


def generate_news_digest(
    tickers: list[str], company_names: dict[str, str],
) -> tuple[str, str, list[dict[str, Any]], dict[str, dict[str, Any] | None]]:
    """
    Fetch recent headlines/filings per ticker from SIX sources — NewsAPI
    (general media), Finnhub (dedicated financial news), SEC EDGAR (primary
    regulatory filings), GDELT (broad worldwide news index), Google News RSS
    (fast cross-check search feed), TED (EU public-procurement notices) —
    and ask the LLM for a short digest that cross-references them rather
    than relying on a single provider's coverage.

    PARALLELIZED across tickers with a thread pool: this is pure I/O wait,
    not CPU work, so a thread pool gives a near-linear speedup with no GIL
    concern — a 5-ticker digest takes roughly as long as the single slowest
    ticker's calls, not the sum of all of them. `max_workers` capped at 8 for
    the same reason as forecast_all_assets: don't oversubscribe a small
    container.

    Returns (digest_text, backend_used, raw_articles, sentiment_by_ticker).
    raw_articles are kept so the UI can render clickable source links (the LLM
    output alone should never be the only trace of a claim; always show the
    reader where it came from). `sentiment_by_ticker` maps each ticker to a
    dict from `news_data.get_ticker_sentiment` (Finnhub aggregated, or VADER
    computed locally, tagged by `provider` either way) or None if nothing was
    available at all — callers must show an explicit "not available" message
    on None, not a silent blank.
    """
    max_workers = min(8, len(tickers)) or 1
    results: dict[str, tuple[list[dict[str, Any]], dict[str, Any] | None]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_all_sources_for_ticker, ticker, company_names.get(ticker)): ticker
            for ticker in tickers
        }
        for future in concurrent.futures.as_completed(futures):
            ticker, articles, sentiment = future.result()
            results[ticker] = (articles, sentiment)

    all_articles: list[dict[str, Any]] = []
    news_block_parts = []
    sentiment_by_ticker: dict[str, dict[str, Any] | None] = {}
    for ticker in tickers:  # iterate in the ORIGINAL order, not completion order,
        articles, sentiment = results.get(ticker, ([], None))  # so output reads the same every run
        sentiment_by_ticker[ticker] = sentiment
        all_articles.extend(articles)
        if articles:
            headlines = "\n".join(f"  - [{a['provider']}] {a['title']} ({a['source']})" for a in articles)
            news_block_parts.append(f"{ticker}:\n{headlines}")

    if not news_block_parts:
        return (
            "No recent news/filings available from any source (NewsAPI, Finnhub, SEC EDGAR) — "
            "check that at least one of NEWSAPI_KEY / FINNHUB_API_KEY is configured, or the "
            "free tier's quota may be exhausted.",
            "n/a",
            [],
            sentiment_by_ticker,  # still populated per-ticker (all None here, since no articles)
        )

    news_block = truncate_to_token_budget("\n\n".join(news_block_parts))
    messages = [
        {"role": "system", "content": SYSTEM_PERSONA},
        {
            "role": "user",
            "content": (
                "Summarise the recent news/filings in <news_headlines> below in 3-5 bullet "
                "points, grouped by ticker if relevant. Each item is tagged with its source "
                "in brackets "
                "([NewsAPI]/[Finnhub] = media coverage, [SEC EDGAR] = a primary regulatory "
                "8-K filing). Treat SEC EDGAR items as more authoritative than media coverage "
                "of the same event, and note explicitly when multiple sources corroborate the "
                "same story (a stronger signal than a single outlet). Focus on anything that "
                "could plausibly move the stock (earnings, guidance, litigation, product "
                "launches, macro exposure). Stay factual - do not speculate beyond what's "
                "stated.\n\n" + _wrap_context("news_headlines", news_block)
            ),
        },
    ]
    text, backend = chat(messages, temperature=0.3, max_tokens=4000)
    return text, backend, all_articles, sentiment_by_ticker


def answer_portfolio_question(
    question: str, results_context: str, chat_history: list[dict[str, Any]], news_chunks: list[Any] | None = None,
) -> tuple[str, str]:
    """
    Grounded Q&A: answers are constrained to the computed results context plus
    ordinary finance knowledge for explaining concepts (e.g. "what is Sortino
    ratio") — but the system prompt explicitly forbids fabricating numbers not
    present in the context, which is the main hallucination risk for a finance
    tool (a confidently wrong Sharpe ratio is worse than no answer).

    `news_chunks` (from src.rag.build_chunks, built once after fetching the news
    digest) enables genuine RAG for questions ABOUT news/filings rather than the
    portfolio numbers: only the top-k chunks relevant to THIS specific question
    are retrieved and injected, not the entire news corpus on every turn. This
    is a real index-then-retrieve pipeline, not a rebrand of the metrics
    context-injection above — see src/rag.py's docstring for why the two use
    different approaches (small fixed structured data vs. a growable
    unstructured corpus).

    A third, independent grounding source: when the question
    names a known methodology term (Sharpe, GARCH, Ledoit-Wolf, walk-forward,
    ...), real papers are fetched live from Semantic Scholar and the LLM is
    instructed to cite only those, never an invented one — see
    src/academic_search.py's module docstring for why (LLMs are well documented
    to hallucinate plausible-sounding academic citations).

    Every injected data block (portfolio metrics, retrieved news, retrieved
    papers) is wrapped in a named XML tag via `_wrap_context` — `<portfolio_data>`,
    `<retrieved_news>`, `<academic_references>` — so the instructions can point
    at a specific tag ("cite ONLY <academic_references>") instead of a vaguer
    "the context above". This matters more now that `chat()` can route to five
    different providers of varying instruction-following discipline (Groq,
    OpenRouter, Cerebras, SambaNova, Ollama), not just Claude.
    """
    retrieved_block = ""
    if news_chunks:
        from src.rag import format_retrieved_chunks, retrieve
        relevant = retrieve(question, news_chunks, top_k=4)
        retrieved_block = format_retrieved_chunks(relevant)

    # Academic grounding: only triggered when the question actually names a
    # known methodology term (Sharpe, GARCH, Ledoit-Wolf, ...) — never on
    # every question. Real papers, fetched live, so the LLM cites something
    # that actually exists instead of a plausible-sounding invented one. See
    # src/academic_search.py's module docstring for the full rationale.
    academic_block = ""
    from src.academic_search import detect_methodology_terms, format_papers_for_prompt, search_academic_papers
    matched_terms = detect_methodology_terms(question)
    if matched_terms:
        # One search per DISTINCT query string (several terms can share the
        # same underlying query, e.g. "ledoit-wolf" and "ledoit wolf") —
        # dedupe before fetching so a question naming both doesn't double-count.
        from src.academic_search import METHODOLOGY_SEARCH_QUERIES
        queries = list(dict.fromkeys(METHODOLOGY_SEARCH_QUERIES[t] for t in matched_terms))
        papers = search_academic_papers(queries[0], limit=3)
        academic_block = format_papers_for_prompt(papers)

    system_content = (
        SYSTEM_PERSONA
        + "\n\nYou may explain general finance concepts (e.g. what Sharpe ratio "
        "means) from your own knowledge. But any SPECIFIC NUMBER about THIS "
        "portfolio must come from <portfolio_data> below - if asked about a number "
        "not present there, say it isn't available rather than estimating it.\n\n"
        "This also applies to questions about a REAL ticker's general market "
        "behaviour (e.g. 'why did GOOG go up', 'how much would an earnings beat "
        "move the stock') even when they're not strictly about THIS portfolio: "
        "you may name the qualitative catalysts that typically matter (earnings, "
        "guidance, product launches, buybacks), but do NOT attach an invented "
        "quantified price-impact figure to any of them (e.g. 'a 2-5% jump', '~1% "
        "effect') unless that number is explicitly present in <portfolio_data> or "
        "<retrieved_news> (if provided below). A specific-sounding percentage you made up reads as "
        "authoritative but is exactly the kind of confidently wrong number this "
        "app exists to avoid — describe the direction and mechanism, never a "
        "magnitude you cannot support with real data.\n\n"
        "CRITICAL — if the user pastes a bare sequence of numbers (e.g. '16.7% "
        "28.1% 0.55 0.56...') without saying which metric each one is, DO NOT "
        "guess an order or say a number 'is likely to represent' some metric. "
        "That is hallucination by another name — matching numbers to labels by "
        "assumed position is exactly the kind of unstated guess this app exists "
        "to avoid. Instead: (1) check whether the pasted numbers match a row in "
        "<portfolio_data> below for the ticker/portfolio they named — if they match "
        "exactly, confirm the mapping explicitly ('these correspond to X, Y, Z ' "
        "from the table below') rather than silently assuming it; (2) if they "
        "don't clearly match, or there are more numbers than you can confidently "
        "map, ask the user to state which number is which metric before explaining "
        "any of them."
        "\n\n" + _wrap_context("portfolio_data", results_context)
    )
    if retrieved_block:
        system_content += (
            "\n\nIf the question is about news/filings, use ONLY the items in "
            "<retrieved_news> below — if nothing relevant was retrieved, say so rather than "
            "inventing news.\n\n" + _wrap_context("retrieved_news", retrieved_block)
        )
    if academic_block:
        system_content += (
            "\n\nThe question touches on a specific methodology this app uses. Real papers "
            "were retrieved into <academic_references> below — cite ONLY those by title/"
            "author/year if you reference literature, and never invent a citation (author, "
            "title, or year) that isn't in that tag.\n\n"
            + _wrap_context("academic_references", academic_block)
        )
    system_content = truncate_to_token_budget(system_content)
    messages = [
        {"role": "system", "content": system_content},
        *chat_history,
        {"role": "user", "content": question},
    ]
    return chat(messages, temperature=0.2, max_tokens=4000)