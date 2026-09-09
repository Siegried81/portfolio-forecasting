"""
Academic literature grounding via the Semantic Scholar Academic Graph API
(free, api key optional at semanticscholar.org/product/api) — real paper
citations for METHODOLOGICAL questions the chatbot gets asked ("why ARIMA
and not an LSTM", "what is Ledoit-Wolf shrinkage"), instead of letting the
LLM answer purely from its own training-data memory of "papers that sound
right."

Why this earns its place: LLMs are well documented to hallucinate academic
citations — plausible-sounding author names, years, and titles for papers
that don't exist, or that say something different from the claim attached to
them. For a technical case-study app whose whole selling point is grounding
LLM output in real data (see ai_features.py's `answer_portfolio_question`
docstring on hallucination), letting the AI Analyst cite methodology papers
from memory would quietly reintroduce the exact failure mode the rest of the
app is built to avoid. Fetching real papers via API and grounding the answer
in their actual titles/authors/years closes that gap for the one remaining
class of question (methodology, not portfolio numbers) that wasn't already
covered by `build_results_context`'s (numbers) or `rag.py`'s (news) grounding.

Two entry points:
- `detect_methodology_terms()` + `search_academic_papers()` are wired into
  `ai_features.answer_portfolio_question` — auto-triggered only when the
  user's question actually contains a known quant-finance methodology term,
  never on every question (a query about "what does my portfolio return"
  has no business searching academic literature).
- `search_academic_papers()` alone also powers a manual "Search academic
  literature" box in the Chatbot tab (app.py), for a user who wants to look
  something up directly rather than asking the chatbot about it.

Fails soft on any error (missing key, network, malformed response) — same
contract as every other fetcher in this codebase (news_data.py, macro_data.py):
this is enrichment, never something that can take a chatbot answer down.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

import requests

from src.cache import cached
from src.config import LLM_SETTINGS

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_FIELDS = "title,authors,year,abstract,url,citationCount,venue"

# Curated, not exhaustive: every methodology term this app's own docstrings
# already name (metrics.py, optimization.py, forecasting.py, backtesting.py,
# factor_models.py, volatility_forecasting.py, timeseries_diagnostics.py) —
# matched case-insensitively as whole words/phrases against a user's
# question. Maps each term to a slightly EXPANDED search query rather than
# the bare term itself: "ARIMA" alone returns results across every field
# that has ever used ARIMA (epidemiology, hydrology, ...), not just finance;
# appending the actual context this app uses each technique in (portfolio
# optimization, volatility, time series) biases results toward the papers
# actually relevant to what this app does with the technique.
METHODOLOGY_SEARCH_QUERIES: dict[str, str] = {
    "ledoit-wolf": "Ledoit Wolf shrinkage covariance estimation portfolio",
    "ledoit wolf": "Ledoit Wolf shrinkage covariance estimation portfolio",
    "shrinkage": "covariance shrinkage estimator portfolio optimization",
    "efficient frontier": "Markowitz mean-variance portfolio efficient frontier",
    "markowitz": "Markowitz mean-variance portfolio optimization",
    "mean-variance": "mean-variance portfolio optimization",
    "sharpe ratio": "Sharpe ratio standard error statistical significance",
    "sortino": "Sortino ratio downside risk portfolio performance",
    "garch": "GARCH volatility forecasting clustering",
    "arima": "ARIMA time series forecasting financial",
    "random walk": "random walk hypothesis efficient market",
    "efficient market": "efficient market hypothesis price predictability",
    "hurst exponent": "Hurst exponent long memory financial time series",
    "value at risk": "Value at Risk VaR financial risk measurement",
    "cvar": "Conditional Value at Risk expected shortfall",
    "expected shortfall": "Conditional Value at Risk expected shortfall",
    "walk-forward": "walk-forward validation backtesting time series",
    "walk forward": "walk-forward validation backtesting time series",
    "look-ahead bias": "look-ahead bias backtesting financial",
    "principal component analysis": "PCA factor model asset covariance",
    "pca factor model": "principal component factor model covariance estimation",
    "factor model": "statistical factor model asset returns covariance",
    "black-litterman": "Black-Litterman portfolio allocation model",
    "capm": "capital asset pricing model CAPM",
    "jensen's alpha": "Jensen's alpha portfolio performance measurement",
    "treynor ratio": "Treynor ratio systematic risk portfolio",
    "calmar ratio": "Calmar ratio drawdown risk-adjusted return",
}


def detect_methodology_terms(question: str) -> list[str]:
    """
    Return every known methodology term (see `METHODOLOGY_SEARCH_QUERIES`
    keys) that appears as a whole word/phrase in `question`, case-
    insensitively. [] if none match — the caller (`answer_portfolio_question`)
    treats that as "no academic grounding needed for this question," not an
    error. Deliberately simple substring/word-boundary matching rather than
    anything ML-based: the set of terms this app could plausibly be asked
    about is small and known in advance (it's this app's own methodology),
    so a fixed vocabulary check is more predictable and testable than a
    fuzzy classifier would be here.
    """
    found = []
    lowered = question.lower()
    for term in METHODOLOGY_SEARCH_QUERIES:
        if re.search(r"\b" + re.escape(term) + r"\b", lowered):
            found.append(term)
    return found


@cached(ttl_seconds=7 * 24 * 3600)  # published papers don't change — cache
# aggressively (a week) rather than re-fetching the same query every chat turn
def _search_semantic_scholar(query: str, limit: int) -> list[dict[str, Any]]:
    """
    Search Semantic Scholar for papers matching `query`. Returns up to
    `limit` results as [{"title", "authors" (list[str]), "year", "url",
    "citation_count", "venue"}, ...] — [] on any failure (missing/invalid
    response, network error, rate limit) or if nothing matched, never raises.

    `SEMANTIC_SCHOLAR_API_KEY` is OPTIONAL, unlike this app's other provider
    keys: Semantic Scholar's search endpoint works unauthenticated (a lower,
    shared rate-limit pool: ~100 requests/5min per their docs), a configured
    key just raises that ceiling. The header is only added when a key is
    actually set, rather than the whole feature being gated on one.

    Private (leading underscore): the public entry point is
    `search_academic_papers`, which combines this with `search_arxiv_papers`
    below. Kept as its own function (not inlined) so its own parsing logic
    has direct, isolated test coverage.
    """
    headers = {"x-api-key": LLM_SETTINGS.semantic_scholar_api_key} if LLM_SETTINGS.semantic_scholar_api_key else {}
    params = {"query": query, "limit": limit, "fields": SEMANTIC_SCHOLAR_FIELDS}

    try:
        response = requests.get(SEMANTIC_SCHOLAR_SEARCH_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Semantic Scholar search failed for query %r: %s", query, exc)
        return []

    papers = payload.get("data") or []
    results = []
    for paper in papers[:limit]:
        title = paper.get("title")
        if not title:
            continue  # a malformed/empty entry — skip rather than show a blank citation
        authors = [a.get("name", "") for a in (paper.get("authors") or []) if a.get("name")]
        results.append({
            "title": title,
            "authors": authors,
            "year": paper.get("year"),
            "url": paper.get("url") or "",
            "citation_count": paper.get("citationCount"),
            "venue": paper.get("venue") or "",
        })
    return results


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_ATOM_NS = "{http://www.w3.org/2005/Atom}"  # arXiv's API returns an Atom XML feed, not
# JSON, unlike every other source in this app — see search_arxiv_papers' docstring


@cached(ttl_seconds=7 * 24 * 3600)
def search_arxiv_papers(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """
    Search arXiv for preprints matching `query`. Genuinely free with NO key
    required at all — even lower-friction than Semantic Scholar, and the
    canonical source for the original papers behind several techniques this
    app documents by name (GARCH, walk-forward validation, PCA factor
    models, ...). Complements Semantic Scholar rather than replacing it:
    arXiv is preprints only (no peer review, no citation counts), Semantic
    Scholar aggregates across published venues too.

    Returns up to `limit` results in the SAME shape as `_search_semantic_scholar`
    ([{"title", "authors", "year", "url", "citation_count", "venue"}, ...]) so
    both sources can be merged interchangeably by `search_academic_papers` —
    `citation_count` is always `None` here (arXiv doesn't track it) and
    `venue` is always `"arXiv preprint"`, not a specific journal/conference.

    arXiv's API replies with an Atom XML feed (`<feed><entry>...</entry></feed>`),
    not JSON — parsed with the standard-library `xml.etree.ElementTree`
    rather than adding a dependency for it. [] on any failure (network,
    malformed XML) or if nothing matched, never raises.
    """
    params = {"search_query": f"all:{query}", "start": 0, "max_results": limit}
    try:
        response = requests.get(ARXIV_API_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("arXiv search failed for query %r: %s", query, exc)
        return []

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        logger.warning("arXiv returned unparsable XML for query %r: %s", query, exc)
        return []

    results = []
    for entry in root.findall(f"{ARXIV_ATOM_NS}entry")[:limit]:
        title_el = entry.find(f"{ARXIV_ATOM_NS}title")
        title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
        if not title:
            continue  # a malformed/empty entry — skip rather than show a blank citation

        authors = [
            (name_el.text or "").strip()
            for name_el in entry.findall(f"{ARXIV_ATOM_NS}author/{ARXIV_ATOM_NS}name")
        ]
        authors = [a for a in authors if a]

        year = None
        published_el = entry.find(f"{ARXIV_ATOM_NS}published")
        if published_el is not None and published_el.text:
            try:
                year = int(published_el.text[:4])  # Atom date format: "YYYY-MM-DDTHH:MM:SSZ"
            except ValueError:
                year = None

        id_el = entry.find(f"{ARXIV_ATOM_NS}id")
        url = (id_el.text or "").strip() if id_el is not None else ""

        results.append({
            "title": title, "authors": authors, "year": year, "url": url,
            "citation_count": None, "venue": "arXiv preprint",
        })
    return results


def search_academic_papers(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """
    Public entry point: search academic literature for `query`, combining
    TWO free sources — Semantic Scholar (tried first: broader coverage
    across published venues, includes citation counts) and arXiv (fills any
    remaining slots up to `limit`, genuinely free with no key at all,
    canonical source for many of the specific techniques this app cites).
    Deduplicated by normalised (lowercased, stripped) title, since the same
    paper is often indexed on both — a Semantic Scholar hit for a paper
    already found there is never re-added from arXiv.

    This is the function `ai_features.answer_portfolio_question` and the
    Chatbot tab's manual literature-search box both call — neither needed
    to change when arXiv was added here, since the combined result has the
    exact same shape and signature as before.
    """
    semantic_scholar_results = _search_semantic_scholar(query, limit)
    remaining = limit - len(semantic_scholar_results)
    arxiv_results = search_arxiv_papers(query, remaining) if remaining > 0 else []

    combined = list(semantic_scholar_results)
    seen_titles = {r["title"].strip().lower() for r in combined}
    for paper in arxiv_results:
        normalised = paper["title"].strip().lower()
        if normalised not in seen_titles:
            combined.append(paper)
            seen_titles.add(normalised)
    return combined[:limit]


def format_papers_for_prompt(papers: list[dict[str, Any]]) -> str:
    """
    Render papers into a text block for prompt injection, in the same spirit
    as `rag.format_retrieved_chunks` — tagged clearly as ACADEMIC REFERENCES
    so the LLM (and the system prompt wired around this in ai_features.py)
    treats these as real, citable sources rather than inventing its own.
    Returns "" for an empty list, matching format_retrieved_chunks's contract
    (an empty prompt block is simply omitted by the caller, not injected as
    an empty section header).
    """
    if not papers:
        return ""
    lines = []
    for p in papers:
        author_str = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
        year_str = f" ({p['year']})" if p.get("year") else ""
        lines.append(f"  - {p['title']}{year_str} — {author_str or 'unknown authors'}. {p['url']}")
    return "ACADEMIC REFERENCES (real papers, cite these — do not invent others):\n" + "\n".join(lines)