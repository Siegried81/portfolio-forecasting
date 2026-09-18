"""
News & filings acquisition from six complementary sources — deliberately not
just one, so results can be cross-referenced rather than relying on a single
provider's coverage and rate limits:

1. NewsAPI (general news aggregator, 100 req/day free) — broad media coverage.
2. Finnhub (dedicated financial-news API, 60 req/MIN free) — company-news
   endpoint purpose-built for tickers, much more generous quota than NewsAPI.
3. SEC EDGAR full-text search (free, no key) — 8-K filings ("material event"
   disclosures), a genuine PRIMARY source rather than journalism about a
   company. Distinct in kind from the other two, not just a third news feed.
4. GDELT (free, no key) — the Global Database of Events, Language, and Tone's
   DOC 2.0 API: a real-time index of online news coverage across tens of
   thousands of outlets worldwide, including many non-US/non-English sources
   NewsAPI and Finnhub don't reach. Genuinely complementary in COVERAGE
   (breadth of outlets), not just a fourth copy of the same US financial
   press.
5. Google News RSS (free, no key) — a lightweight, always-current search feed
   over Google's own news index; useful as a fast, redundant check against
   the dedicated APIs above, especially for a ticker/company Finnhub or
   NewsAPI's own index happens to cover thinly.
6. TED — Tenders Electronic Daily (free, no key), the EU's official public-
   procurement notice database. NOT "TED Talks" — the EU's tender-publication
   platform (ted.europa.eu). A structurally different kind of signal from the
   five sources above: an upcoming or awarded public-sector contract
   mentioning a company is a forward-looking business signal (new revenue,
   market entry, regulatory exposure), not commentary about the company —
   the same "primary source, not journalism" distinction SEC EDGAR earns
   among the news aggregators.

Kept deliberately dumb (no sentiment scoring here): the LLM does the qualitative
read in `ai_features.py`. Every fetch function fails SOFTLY - a missing/expired
key or a down provider must never crash the app, since this is enrichment, not
core to the portfolio maths. `generate_news_digest` in ai_features.py merges
whatever came back from however many of the six sources succeeded.
"""
from __future__ import annotations

import datetime as dt
import logging
import xml.etree.ElementTree as ET
from typing import Any

import requests

from src.cache import cached
from src.config import LLM_SETTINGS

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"
FINNHUB_URL = "https://finnhub.io/api/v1/company-news"
SEC_FULLTEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
TED_API_URL = "https://api.ted.europa.eu/v3/notices/search"

# Free NewsAPI dev tier only returns articles from the last ~30 days.
NEWS_LOOKBACK_DAYS = 14

# SEC requires a descriptive User-Agent identifying the requester (their policy,
# not optional) — a generic/missing one gets throttled or blocked outright.
SEC_USER_AGENT = "portfolio-forecasting-bootcamp-project research@example.com"


@cached(ttl_seconds=1800)
def fetch_ticker_headlines(ticker: str, company_name: str | None = None, max_articles: int = 5) -> list[dict[str, Any]]:
    """
    NewsAPI: general-media headlines mentioning a ticker (or its company name,
    which gives much better recall than the raw ticker symbol — "AAPL" barely
    appears in prose, "Apple" does). Returns [] (never raises) on any failure.
    """
    if not LLM_SETTINGS.newsapi_key:
        return []

    query = company_name or ticker
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=NEWS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    params: dict[str, Any] = {
        "q": query, "from": since, "sortBy": "relevancy", "language": "en",
        "pageSize": max_articles, "apiKey": LLM_SETTINGS.newsapi_key,
    }
    try:
        response = requests.get(NEWSAPI_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("NewsAPI fetch failed for %s: %s", ticker, exc)
        return []

    articles = payload.get("articles", [])
    return [
        {
            "title": a.get("title", ""),
            "description": a.get("description", "") or "",
            "source": (a.get("source") or {}).get("name", "unknown"),
            "published_at": a.get("publishedAt", ""),
            "url": a.get("url", ""),
            "provider": "NewsAPI",
        }
        for a in articles
        if a.get("title") and a.get("title") != "[Removed]"
    ]


@cached(ttl_seconds=1800)
def fetch_finnhub_news(ticker: str, max_articles: int = 5) -> list[dict[str, Any]]:
    """
    Finnhub: dedicated company-news endpoint (not a general search — it's
    purpose-built per-ticker), 60 req/min free tier. Returns [] on any failure,
    including a missing FINNHUB_API_KEY.
    """
    if not LLM_SETTINGS.finnhub_api_key:
        return []

    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=NEWS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    params = {"symbol": ticker, "from": since, "to": today, "token": LLM_SETTINGS.finnhub_api_key}

    try:
        response = requests.get(FINNHUB_URL, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Finnhub fetch failed for %s: %s", ticker, exc)
        return []

    if not isinstance(articles, list):  # Finnhub returns {} with an error message on bad symbol/key
        return []

    return [
        {
            "title": a.get("headline", ""),
            "description": a.get("summary", "") or "",
            "source": a.get("source", "unknown"),
            "published_at": dt.datetime.fromtimestamp(a["datetime"]).isoformat() if a.get("datetime") else "",
            "url": a.get("url", ""),
            "provider": "Finnhub",
        }
        for a in articles[:max_articles]
        if a.get("headline")
    ]


@cached(ttl_seconds=1800)
def fetch_gdelt_news(ticker: str, company_name: str | None = None, max_articles: int = 5) -> list[dict[str, Any]]:
    """
    GDELT DOC 2.0 API: real-time full-text search across GDELT's index of
    online news coverage worldwide. No key required at all — genuinely free,
    same tier as SEC EDGAR and arXiv (`academic_search.py`) in that respect,
    unlike NewsAPI/Finnhub above which both need a configured key.

    Searched by company name (falling back to the ticker itself if no name
    is available) for the same recall reason as `fetch_ticker_headlines` —
    "AAPL" barely appears in ordinary prose, "Apple" does. `sourcelang:english`
    keeps results consumable by the downstream LLM/sentiment pipeline, which
    only handles English text; dropping that filter would surface GDELT's
    real multilingual breadth at the cost of silently feeding non-English
    text into FinBERT/VADER, which isn't calibrated for other languages.

    Returns [] (never raises) on any failure, including a malformed/empty
    response — the DOC API's JSON field shape was implemented from public
    documentation, not verified against a live query in this sandboxed
    environment (network access here is restricted to package registries);
    if results look wrong, inspect the raw JSON directly before assuming the
    parser is exhaustive.
    """
    query = f'"{company_name or ticker}" sourcelang:english'
    params: dict[str, Any] = {
        "query": query, "mode": "artlist", "format": "json",
        "maxrecords": max_articles, "sort": "DateDesc",
        "timespan": f"{NEWS_LOOKBACK_DAYS}d",
    }
    try:
        response = requests.get(GDELT_DOC_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("GDELT fetch failed for %s: %s", ticker, exc)
        return []

    articles = (payload or {}).get("articles") or []
    results = []
    for a in articles[:max_articles]:
        title = a.get("title", "")
        if not title:
            continue  # a malformed/empty entry — skip rather than show a blank headline
        # GDELT's "seendate" is a compact "20260908T120000Z" timestamp, not ISO —
        # reformatted to the same ISO shape every other provider here uses.
        published_at = ""
        raw_date = a.get("seendate", "")
        if raw_date:
            try:
                published_at = dt.datetime.strptime(raw_date, "%Y%m%dT%H%M%SZ").isoformat()
            except ValueError:
                published_at = ""
        results.append({
            "title": title,
            "description": "",  # DOC API's artlist mode returns headlines only, no summary text
            "source": a.get("domain", "unknown"),
            "published_at": published_at,
            "url": a.get("url", ""),
            "provider": "GDELT",
        })
    return results


@cached(ttl_seconds=1800)
def fetch_google_news_rss(ticker: str, company_name: str | None = None, max_articles: int = 5) -> list[dict[str, Any]]:
    """
    Google News' public RSS search feed. No key required. Plain RSS 2.0 XML
    (`<rss><channel><item>...</item></channel></rss>`), parsed with the
    standard-library `xml.etree.ElementTree` — same dependency-avoidance
    reasoning as `academic_search.search_arxiv_papers`'s Atom parsing, just a
    different feed dialect (flat `<item>` elements, no XML namespace).

    A fast, always-current cross-check against the dedicated news APIs above
    — useful when NewsAPI's/Finnhub's own index happens to cover a ticker
    thinly, since Google's crawl breadth differs from either.

    Returns [] (never raises) on any failure, including malformed XML or an
    empty feed. Field-shape caveat, same spirit as GDELT above: implemented
    from the feed's publicly observable structure, not verified against a
    live query in this sandboxed environment.
    """
    query = company_name or ticker
    params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    try:
        response = requests.get(GOOGLE_NEWS_RSS_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Google News RSS fetch failed for %s: %s", ticker, exc)
        return []

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        logger.warning("Google News RSS returned unparsable XML for %s: %s", ticker, exc)
        return []

    results = []
    for item in root.findall("./channel/item")[:max_articles]:
        title = (item.findtext("title") or "").strip()
        if not title:
            continue  # a malformed/empty entry — skip rather than show a blank headline
        source_el = item.find("source")
        results.append({
            "title": title,
            "description": (item.findtext("description") or "").strip(),
            "source": (source_el.text or "unknown").strip() if source_el is not None else "unknown",
            "published_at": (item.findtext("pubDate") or "").strip(),
            "url": (item.findtext("link") or "").strip(),
            "provider": "Google News",
        })
    return results


def _ted_field_text(value: Any) -> str:
    """
    TED's notice fields commonly come back either as a plain string or as a
    dict keyed by language code (e.g. {"eng": "Cloud services framework
    agreement"}) for multilingual fields like the notice title. Normalises
    either shape to a single display string, preferring English when a
    language-keyed dict is present, falling back to whichever value comes
    first otherwise. Returns "" for anything else (missing field, unexpected
    shape) rather than raising — this app's TED field-shape handling is
    implemented from the API's public documentation, not verified against a
    live query in this sandboxed environment (network access here is
    restricted to package registries), so staying defensive here matters
    more than for a shape that's already been confirmed live.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("eng") or next(iter(value.values()), ""))
    return ""


@cached(ttl_seconds=3600)
def fetch_ted_notices(ticker: str, company_name: str | None = None, max_notices: int = 5) -> list[dict[str, Any]]:
    """
    TED (Tenders Electronic Daily) v3 notices search — the EU's official
    public-procurement publication platform. NOT "TED Talks"; see this
    module's own docstring for why this is a structurally different kind of
    signal from the news/filings sources above (a forward-looking business
    signal, not commentary about the company).

    No API key required for notice search. A POST request with a free-text
    query (`FT~"..."` full-text search syntax, TED's own documented
    convention) against the company name — falling back to the ticker itself
    when no name is available, same recall reasoning as every other
    name-based search in this module.

    Returns [] (never raises) on any failure, including a malformed/empty
    response — same field-shape caveat as `fetch_gdelt_news` above: this
    parser was implemented from TED's public API documentation, not verified
    against a live query in this sandboxed environment. `description` and
    `url` are best-effort here (a notice's own detail page isn't reliably
    derivable from the search response alone) — if a filed URL looks wrong,
    inspect the raw JSON directly before assuming this parser is exhaustive.
    """
    query = f'FT~"{company_name or ticker}"'
    body: dict[str, Any] = {
        "query": query,
        "fields": [
            "publication-number", "notice-title", "buyer-name",
            "buyer-country", "publication-date",
        ],
        "page": 1,
        "limit": max_notices,
    }
    try:
        response = requests.post(TED_API_URL, json=body, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("TED fetch failed for %s: %s", ticker, exc)
        return []

    notices = (payload or {}).get("notices") or []
    results = []
    for notice in notices[:max_notices]:
        title = _ted_field_text(notice.get("notice-title"))
        if not title:
            continue  # a malformed/empty entry — skip rather than show a blank headline
        buyer = _ted_field_text(notice.get("buyer-name"))
        country = _ted_field_text(notice.get("buyer-country"))
        publication_number = notice.get("publication-number", "")
        results.append({
            "title": title,
            "description": f"Buyer: {buyer} ({country})" if buyer else "",
            "source": "TED (EU public procurement)",
            "published_at": notice.get("publication-date", ""),
            "url": (
                f"https://ted.europa.eu/en/notice/-/detail/{publication_number}"
                if publication_number else "https://ted.europa.eu/"
            ),
            "provider": "TED",
        })
    return results


@cached(ttl_seconds=24 * 3600)  # ticker->CIK mappings change rarely (only on a listing/delisting
# event) — a day-long TTL is generous, not stale-risking, and keeps this off the hot path.
def _fetch_cik_for_ticker(ticker: str) -> str | None:
    """
    Resolve TICKER -> CIK via SEC's own public `company_tickers.json` (no key,
    same User-Agent requirement as full-text search below, no extra rate
    limit). Returns the CIK as a plain (non-zero-padded) numeric string, or
    None if the ticker isn't in SEC's list (not a US-listed filer, e.g. an
    ETF or FX proxy) or the fetch itself fails — callers must fall back to
    unfiltered behaviour on None, never raise, same fails-soft contract as
    every other fetcher in this module.

    A bare company-name text search (`q='"{company_name}"'`) also matches
    UNRELATED companies whose name happens to contain the same word (e.g.
    searching "Apple" for AAPL also surfaces Apple Hospitality REIT's own
    8-Ks). This lookup gives `fetch_sec_filings` the real CIK to filter the
    search results down to the actual company, rather than trusting the
    free-text match alone.
    """
    headers = {"User-Agent": SEC_USER_AGENT}
    try:
        response = requests.get(SEC_COMPANY_TICKERS_URL, headers=headers, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("SEC company_tickers.json fetch failed for %s: %s", ticker, exc)
        return None

    # Payload shape: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, "1": {...}, ...}
    # — an object keyed by an arbitrary index, not a list; iterate .values().
    target = ticker.upper()
    for entry in (payload or {}).values():
        if str(entry.get("ticker", "")).upper() == target:
            cik_str = entry.get("cik_str")
            return str(cik_str) if cik_str is not None else None
    return None  # not in SEC's list — a non-US-filer ticker (ETF, FX proxy, etc.), not an error


def _hit_ciks(hit: dict[str, Any]) -> set[str]:
    """Normalise one search hit's `_source.ciks` (a list of possibly zero-padded
    strings, e.g. "0000320193") to plain numeric strings, so they compare
    equal to `_fetch_cik_for_ticker`'s own plain-string return value."""
    raw = (hit.get("_source", {}) or {}).get("ciks") or []
    normalised = set()
    for c in raw:
        try:
            normalised.add(str(int(c)))
        except (TypeError, ValueError):
            continue  # a malformed entry — skip it rather than crash the whole filter
    return normalised


@cached(ttl_seconds=3600)
def fetch_sec_filings(company_name: str, ticker: str, max_filings: int = 3, form_type: str = "8-K") -> list[dict[str, Any]]:
    """
    SEC EDGAR full-text search: recent filings of `form_type` mentioning the
    company. Defaults to 8-K ("material event" disclosures — earnings,
    executive changes, M&A, major agreements). No API key required, but SEC
    requires a descriptive User-Agent (enforced — a missing/generic one gets
    throttled). This is a PRIMARY regulatory source, genuinely different in
    kind from the news aggregators above, not a third copy of the same
    headlines.

    `fetch_sec_insider_trades` (below) is a thin wrapper over this same
    function with `form_type="4"` — Form 4 (insider buy/sell disclosures)
    shares the identical search/CIK-filtering/dedup mechanics as 8-Ks, just a
    different EDGAR form code and result title, so it earns a wrapper rather
    than a second near-duplicate function.

    The underlying full-text search matches on the bare
    `company_name` string, which can also match unrelated companies sharing a
    word in their name (e.g. "Apple" matching Apple Hospitality REIT as well
    as Apple Inc.). Results are filtered down to hits whose own `ciks`
    field contains `ticker`'s actual CIK (via `_fetch_cik_for_ticker`) before
    being returned — and deduplicated by accession number, since the same
    filing can otherwise appear more than once within `max_filings`. If the
    CIK lookup itself fails (network issue), filtering is skipped entirely —
    this must never return FEWER results than an unfiltered search would,
    only more accurate ones when the lookup succeeds.

    Returns [] on any failure. Field-shape caveat, same spirit as the Twelve
    Data fundamentals parser: EDGAR's full-text search response shape was
    implemented from public documentation, not verified against a live query in
    this environment (network access here is restricted to package registries)
    — if results look wrong, inspect the raw JSON directly before assuming the
    parser is exhaustive.
    """
    params = {"q": f'"{company_name}"', "forms": form_type, "dateRange": "custom",
              "startdt": (dt.date.today() - dt.timedelta(days=30)).isoformat(),
              "enddt": dt.date.today().isoformat()}
    headers = {"User-Agent": SEC_USER_AGENT}

    try:
        response = requests.get(SEC_FULLTEXT_SEARCH_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("SEC EDGAR fetch failed for %s (form %s): %s", ticker, form_type, exc)
        return []

    hits = (payload.get("hits") or {}).get("hits") or []

    target_cik = _fetch_cik_for_ticker(ticker)
    if target_cik is not None:
        hits = [h for h in hits if target_cik in _hit_ciks(h)]

    results: list[dict[str, Any]] = []
    seen_accessions: set[str] = set()
    for hit in hits:
        if len(results) >= max_filings:
            break
        source = hit.get("_source", {})
        cik = str(source.get("ciks", [""])[0]).lstrip("0") if source.get("ciks") else ""
        accession = (hit.get("_id", "") or "").split(":")[0]
        if accession:
            if accession in seen_accessions:
                continue  # the same filing can appear more than once in raw search hits
            seen_accessions.add(accession)
        # Direct link to the specific filing (not just the company's generic
        # filing list) when we have both pieces — standard SEC EDGAR index
        # URL format: /Archives/edgar/data/{cik}/{accession-no-dashes}/{accession}-index.htm.
        # Best-effort (see this function's field-shape caveat above) — falls
        # back to the generic company/search page when either piece is
        # missing or the format doesn't match what's expected.
        if cik and accession:
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{accession}-index.htm"
        elif cik:
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
        else:
            url = "https://www.sec.gov/edgar/search/"
        results.append({
            "title": f"{form_type} filing: {source.get('display_names', [company_name])[0]}",
            "description": f"Filed {source.get('file_date', 'unknown date')}",
            "source": "SEC EDGAR",
            "published_at": source.get("file_date", ""),
            "url": url,
            "provider": "SEC EDGAR",
        })
    return results


def fetch_sec_insider_trades(company_name: str, ticker: str, max_filings: int = 5) -> list[dict[str, Any]]:
    """
    Recent Form 4 filings (insider buy/sell disclosures — an officer,
    director, or 10%+ shareholder trading their own company's stock, legally
    required to be disclosed within 2 business days) for `ticker`. A thin
    wrapper over `fetch_sec_filings(form_type="4")` — see that function's own
    docstring for the full mechanics (CIK filtering, dedup, fails-soft
    contract, field-shape caveat), all identical here.

    Genuinely different signal from an 8-K or ordinary news coverage:
    cluster insider buying (multiple executives independently purchasing) is
    a well-documented, if noisy, bullish signal in the academic literature;
    heavy clustered selling is the inverse. This function surfaces the raw
    filings only — no attempt is made here to classify buy vs. sell or
    aggregate a "net insider sentiment" score, since Form 4's actual
    transaction-code/share-count fields aren't reliably parseable from the
    full-text search response alone (that would need the filing's own XML,
    a meaningfully bigger fetch this app doesn't currently make) — showing
    the filing and letting the LLM/reader open it is more honest than a
    confident-looking but unverified buy/sell classification.
    """
    return fetch_sec_filings(company_name, ticker, max_filings=max_filings, form_type="4")


# ---------------------------------------------------------------------------
# News sentiment — Cascade: FinBERT (financial-domain model, PRIMARY) ->
# Finnhub's own aggregated sentiment (wider corpus than what this app fetches
# itself) -> VADER computed locally from the headlines already fetched above
# (FALLBACK, free, offline, no key) -> None if there's truly nothing to score.
# Callers must show an explicit "not available" message on None rather than
# silently omitting the section — a blank sentiment section reads as
# "neutral", which is a claim, not an absence of one.
# ---------------------------------------------------------------------------

FINNHUB_SENTIMENT_URL = "https://finnhub.io/api/v1/news-sentiment"

# Circuit breaker: a 403 on this endpoint means the current Finnhub plan
# doesn't include it AT ALL, not a transient rate limit that resolves
# itself. Retrying identically on every subsequent ticker just wastes a
# network round-trip and repeats a warning for something already known.
# Once seen, skip straight to None (VADER fallback) for the rest of this
# process. Module-level (not per-ticker), same pattern as market_data.py's
# Yahoo circuit breaker: a plan restriction is an account-wide fact, not a
# per-ticker one.
_finnhub_sentiment_plan_restricted = False


def _trip_finnhub_sentiment_breaker_if_permanent(exc: Exception) -> None:
    """
    Inspect an HTTPError from the sentiment endpoint and trip the module-level
    circuit breaker only for status codes that mean "this account can never
    use this endpoint" (403 plan-restricted, 401 bad/revoked key) — a 429
    rate limit or a 5xx server error is transient and must NOT trip it, since
    those genuinely can succeed on a later, unrelated ticker. Factored out
    from `fetch_finnhub_sentiment` so this status-code decision has its own
    direct test coverage, separate from the network call around it.
    """
    global _finnhub_sentiment_plan_restricted
    status = exc.response.status_code if isinstance(exc, requests.HTTPError) and exc.response is not None else None
    if status in (401, 403):
        logger.warning(
            "Finnhub sentiment endpoint returned %s (plan-restricted or bad key) — "
            "skipping it for the rest of this session, falling back to VADER: %s", status, exc,
        )
        _finnhub_sentiment_plan_restricted = True
    else:
        logger.warning("Finnhub sentiment fetch failed: %s", exc)


@cached(ttl_seconds=1800)
def fetch_finnhub_sentiment(ticker: str) -> dict[str, Any] | None:
    """
    Finnhub's own aggregated news-sentiment endpoint: bullish/bearish % across
    a wider article set than the ~5 headlines this app fetches per ticker, plus
    a weekly article-volume ("buzz") figure. Tried FIRST because it's a genuine
    independent aggregation, not a second opinion computed from the same small
    sample already on screen.

    Note: this endpoint is plan-restricted on some Finnhub free-tier accounts
    (every ticker 403s). Fails soft (returns None) on ANY error, including a
    403, so `get_ticker_sentiment` below always has the local VADER fallback
    to fall back to rather than surfacing a raw API error to the user. A
    confirmed 403 ALSO trips the module-level circuit breaker above, so every
    ticker after the first skips the network call entirely instead of
    repeating one already known to fail — a 401 (bad/revoked key) is treated
    the same way, since neither resolves without the user changing their
    Finnhub plan or key, unlike a 429 rate limit, which is left to retry
    normally (the account isn't broken, just temporarily throttled).
    """
    if not LLM_SETTINGS.finnhub_api_key or _finnhub_sentiment_plan_restricted:
        return None
    try:
        response = requests.get(
            FINNHUB_SENTIMENT_URL, params={"symbol": ticker, "token": LLM_SETTINGS.finnhub_api_key}, timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.HTTPError as exc:
        _trip_finnhub_sentiment_breaker_if_permanent(exc)
        return None
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Finnhub sentiment fetch failed for %s: %s", ticker, exc)
        return None

    sentiment = payload.get("sentiment") or {}
    bullish = sentiment.get("bullishPercent")
    bearish = sentiment.get("bearishPercent")
    if bullish is None or bearish is None:
        return None  # Finnhub returns an empty/zeroed shape for tickers it has no coverage for

    return {
        "provider": "Finnhub (aggregated)",
        "bullish_pct": bullish * 100,
        "bearish_pct": bearish * 100,
        # Normalised to [-1, 1] so it's directly comparable to VADER's compound
        # score below — callers apply the SAME +/-0.1 threshold either way.
        "score": bullish - bearish,
        "n_articles": (payload.get("buzz") or {}).get("articlesInLastWeek"),
    }


_vader_analyzer: Any = None  # lazy singleton — building the lexicon has a small fixed cost


def _get_vader_analyzer() -> Any:
    # Returns Any (not SentimentIntensityAnalyzer) deliberately: the import
    # stays lazy/local to avoid the vaderSentiment dependency at module load
    # time, and the package ships no type stubs either way.
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def compute_local_sentiment(articles: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Fallback sentiment scored locally from the headlines/descriptions already
    fetched above, when neither FinBERT nor Finnhub's own endpoint is
    available. VADER (Valence Aware Dictionary and sEntiment Reasoner) was
    chosen for the same reason TF-IDF was chosen over neural embeddings in
    rag.py: this is a small, cheap, deterministic scoring pass on a handful
    of short texts — not worth burning LLM budget or adding latency for.
    It's free, fully offline (a bundled lexicon, no model download, no API
    key), and specifically tuned for short, informal text, which headlines
    are closer to than to long-form prose.

    Returns None if there's nothing to score (no articles at all) — that case
    is distinct from "computed a neutral score" and callers must not conflate
    the two.
    """
    if not articles:
        return None
    analyzer = _get_vader_analyzer()
    scores = [
        analyzer.polarity_scores(f"{a.get('title', '')}. {a.get('description', '')}")["compound"]
        for a in articles
    ]
    avg_score = sum(scores) / len(scores)
    bullish_count = sum(1 for s in scores if s > 0.05)   # VADER's own documented neutral band
    bearish_count = sum(1 for s in scores if s < -0.05)
    return {
        "provider": "VADER (computed locally from fetched headlines)",
        "bullish_pct": 100 * bullish_count / len(scores),
        "bearish_pct": 100 * bearish_count / len(scores),
        "score": avg_score,
        "n_articles": len(scores),
    }


# HuggingFace deprecated the classic `api-inference.huggingface.co` domain in
# favour of a unified router — `router.huggingface.co/hf-inference/models/...`
# is the current endpoint for this same serverless Inference API, per HF's own
# migration notice. Request shape is unchanged; the RESPONSE shape is NOT —
# see fetch_finbert_sentiment's docstring for the nested-list wrinkle this
# migration introduced.
FINBERT_INFERENCE_URL = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
FINBERT_MAX_ARTICLES = 5  # one HTTP request per article (see fetch_finbert_sentiment's
# docstring for why) — capped so one ticker's sentiment doesn't cost 10+ round-trips


@cached(ttl_seconds=1800)
def fetch_finbert_sentiment(articles: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Financial-domain sentiment via FinBERT (ProsusAI/finbert — BERT further
    trained on financial text, then fine-tuned for positive/negative/neutral
    classification), called through Hugging Face's hosted Inference API —
    plain `requests`, no local model weights, no `torch`/`transformers`
    dependency, matching this app's existing "thin HTTP fetcher" pattern for
    every other external NLP/data source. Tried FIRST in
    `get_ticker_sentiment`'s cascade — a financial-domain-tuned model should
    read financial headlines more accurately than either Finnhub's generic
    aggregation or VADER's general-purpose lexicon — falling back to Finnhub
    (broader corpus, but no domain tuning) and finally VADER (fully offline,
    the safety net that can never fail) if this tier is unavailable — no
    key configured, or every request failed.

    Scores each of the first `FINBERT_MAX_ARTICLES` articles' title+description
    one at a time (not batched), since HF-hosted models don't consistently
    share a single-vs-batch response shape across endpoints — scoring one
    text per call keeps the request shape unambiguous.

    Response-shape handling: the older `api-inference.huggingface.co`
    endpoint returned a FLAT list for one input string —
    `[{"label": "positive", "score": 0.7}, {"label": "negative", ...}, ...]`.
    The current `router.huggingface.co/hf-inference` endpoint wraps that same
    list in one extra layer — `[[{"label": "positive", "score": 0.7}, ...]]`
    — a list containing ONE list of class scores, not a list of class-score
    dicts directly. Both shapes are handled: if the top-level list's first
    element is itself a list, that inner list is used as the class-score
    list instead.

    Returns None (never raises) if: no `HUGGINGFACE_API_KEY` configured, no
    articles to score, every request fails (network, rate limit, or the
    well-known HF serverless "model is currently loading" cold-start — a 503
    with a JSON body, not a network error, so it's caught by the same
    generic handler rather than requiring special-cased retry logic here).
    """
    if not LLM_SETTINGS.huggingface_api_key or not articles:
        return None

    headers = {"Authorization": f"Bearer {LLM_SETTINGS.huggingface_api_key}"}
    scores: list[float] = []  # positive_score - negative_score per article, in [-1, 1]

    for article in articles[:FINBERT_MAX_ARTICLES]:
        text = f"{article.get('title', '')}. {article.get('description', '')}".strip()
        if not text or text == ".":
            continue
        try:
            response = requests.post(FINBERT_INFERENCE_URL, headers=headers, json={"inputs": text}, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("FinBERT request failed, skipping this article: %s", exc)
            continue

        if not isinstance(payload, list):
            # Either an {"error": "...currently loading..."} cold-start body, or a
            # shape genuinely different from what's documented — either way, skip
            # this article rather than guessing at a malformed structure.
            logger.warning("FinBERT returned an unexpected payload shape, skipping this article: %r", payload)
            continue

        # router.huggingface.co wraps a single-input result in an extra list
        # layer — see this function's own docstring. The flat shape (older
        # endpoint) still works unchanged: payload[0] is then a dict, not a
        # list, so no unwrapping happens and class_scores is built the same
        # way either way.
        if payload and isinstance(payload[0], list):
            payload = payload[0]

        class_scores = {item.get("label"): item.get("score", 0.0) for item in payload if isinstance(item, dict)}
        positive = class_scores.get("positive", 0.0)
        negative = class_scores.get("negative", 0.0)
        scores.append(positive - negative)

    if not scores:
        return None

    avg_score = sum(scores) / len(scores)
    bullish_count = sum(1 for s in scores if s > 0.1)
    bearish_count = sum(1 for s in scores if s < -0.1)
    return {
        "provider": "FinBERT (financial-domain BERT, via Hugging Face)",
        "bullish_pct": 100 * bullish_count / len(scores),
        "bearish_pct": 100 * bearish_count / len(scores),
        "score": avg_score,
        "n_articles": len(scores),
    }


def get_ticker_sentiment(ticker: str, articles: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Single entry point for sentiment on one ticker — a 3-tier cascade:
    FinBERT (financial-domain model computed on `articles`, already fetched
    by the caller — tried FIRST as the primary method, since a model
    specifically tuned on financial text should read financial headlines
    more accurately than a general-purpose aggregator or lexicon) ->
    Finnhub's aggregated score (broader corpus than the ~5 headlines this
    app fetches per ticker, used as a fallback if FinBERT itself isn't
    available — no key configured, or every request failed) -> VADER (fully
    offline general-purpose lexicon, the final fallback that can never
    fail). Callers show an explicit "sentiment not available" message on
    None, never a silent blank.
    """
    sentiment = fetch_finbert_sentiment(articles)
    if sentiment is not None:
        return sentiment
    sentiment = fetch_finnhub_sentiment(ticker)
    if sentiment is not None:
        return sentiment
    return compute_local_sentiment(articles)