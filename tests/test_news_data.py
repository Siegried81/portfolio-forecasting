"""
Unit tests for src/news_data.py's fetch_sec_filings — previously had zero test
coverage. Priority: the URL construction logic, since a real bug was found
there (accession number extracted from the response but never used, so every
filing linked to the generic company browse page instead of the specific
8-K found). Fully mocked — no real network call, no SEC EDGAR rate limits hit.

Also covers CIK filtering: a bare company-name text search
("Apple") can also match unrelated companies sharing the word (confirmed
live: Apple Hospitality REIT alongside Apple Inc.) — fetch_sec_filings
resolves the ticker's real CIK via _fetch_cik_for_ticker and filters hits
down to it, deduplicating by accession number along the way.
"""
import pytest
import streamlit as st

import src.news_data as news_data
from src.news_data import (
    _fetch_cik_for_ticker,
    fetch_gdelt_news,
    fetch_google_news_rss,
    fetch_sec_filings,
    fetch_sec_insider_trades,
    fetch_ted_notices,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """fetch_sec_filings is decorated with @cached() (src/cache.py), which
    falls through to @st.cache_data with no REDIS_URL set — without this,
    identical (company_name, ticker) args across tests would return the
    first test's cached result instead of re-invoking the mocked requests.get."""
    st.cache_data.clear()
    yield
    st.cache_data.clear()


class _FakeResponse:
    def __init__(self, json_data: dict):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def _edgar_payload(hits: list[dict]) -> dict:
    return {"hits": {"hits": hits}}


def _hit(cik: str | None, accession_id: str | None, display_name: str = "Apple Inc.", file_date: str = "2026-08-15"):
    source = {"display_names": [display_name], "file_date": file_date}
    if cik is not None:
        source["ciks"] = [cik]
    hit = {"_source": source}
    if accession_id is not None:
        hit["_id"] = accession_id
    return hit


def _sec_get(search_payload: dict, ticker_map: dict | None = None):
    """Fake requests.get for EDGAR: SEC's ticker map (AAPL -> 320193 unless
    given) for the CIK lookup, `search_payload` for the full-text search."""
    ticker_map = ticker_map if ticker_map is not None else {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

    def _fake_get(url, **kwargs):
        if url == news_data.SEC_COMPANY_TICKERS_URL:
            return _FakeResponse(ticker_map)
        return _FakeResponse(search_payload)
    return _fake_get


def test_fetch_sec_filings_links_directly_to_the_specific_filing(monkeypatch):
    # `accession` must be used to build a direct link to the specific filing,
    # not just fall back to the generic per-company browse page regardless
    # of which filing was actually found.
    payload = _edgar_payload([_hit(cik="0000320193", accession_id="0000320193-26-000106:aapl-20260815.htm")])
    monkeypatch.setattr(news_data.requests, "get", _sec_get(payload))

    results = fetch_sec_filings("Apple Inc.", "AAPL")

    assert len(results) == 1
    assert results[0]["url"] == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019326000106/0000320193-26-000106-index.htm"
    )


def test_fetch_sec_filings_falls_back_to_generic_company_page_without_accession(monkeypatch):
    # _id missing/empty -> no accession to build a direct link from; cik alone
    # should still produce the per-company browse page, not crash or link nowhere.
    payload = _edgar_payload([_hit(cik="0000320193", accession_id=None)])
    monkeypatch.setattr(news_data.requests, "get", _sec_get(payload))

    results = fetch_sec_filings("Apple Inc.", "AAPL")

    assert results[0]["url"] == "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=320193"


def test_fetch_sec_filings_drops_a_hit_whose_filer_cannot_be_verified(monkeypatch):
    # A hit without `ciks` cannot be matched to the ticker's CIK, so it may be
    # another company's filing: dropped rather than attributed to AAPL.
    payload = _edgar_payload([_hit(cik=None, accession_id="0000320193-26-000106:aapl-20260815.htm")])
    monkeypatch.setattr(news_data.requests, "get", _sec_get(payload))

    assert fetch_sec_filings("Apple Inc.", "AAPL") == []


def test_fetch_sec_filings_returns_empty_list_on_request_failure(monkeypatch):
    import requests

    def _raise(*a, **k):
        raise requests.RequestException("network down")

    monkeypatch.setattr(news_data.requests, "get", _raise)
    assert fetch_sec_filings("Apple Inc.", "AAPL") == []


def test_fetch_sec_filings_respects_max_filings(monkeypatch):
    payload = _edgar_payload([_hit(cik="0000320193", accession_id=f"000032019326-00{i:04d}:x.htm") for i in range(5)])
    monkeypatch.setattr(news_data.requests, "get", _sec_get(payload))

    results = fetch_sec_filings("Apple Inc.", "AAPL", max_filings=2)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# _fetch_cik_for_ticker — the ticker -> CIK lookup the filter above relies on
# ---------------------------------------------------------------------------

def _ticker_map_payload(entries: list[tuple[int, str, str]]) -> dict:
    """Build a fake company_tickers.json payload: {"0": {cik_str, ticker, title}, ...}."""
    return {str(i): {"cik_str": cik, "ticker": ticker, "title": title} for i, (cik, ticker, title) in enumerate(entries)}


def test_fetch_cik_for_ticker_returns_cik_on_match(monkeypatch):
    payload = _ticker_map_payload([(320193, "AAPL", "Apple Inc."), (789019, "MSFT", "Microsoft Corp")])
    monkeypatch.setattr(news_data.requests, "get", lambda *a, **k: _FakeResponse(payload))
    assert _fetch_cik_for_ticker("AAPL") == "320193"


def test_fetch_cik_for_ticker_case_insensitive(monkeypatch):
    payload = _ticker_map_payload([(320193, "AAPL", "Apple Inc.")])
    monkeypatch.setattr(news_data.requests, "get", lambda *a, **k: _FakeResponse(payload))
    assert _fetch_cik_for_ticker("aapl") == "320193"


def test_fetch_cik_for_ticker_none_when_ticker_not_in_sec_list(monkeypatch):
    # e.g. an ETF or FX proxy ticker that isn't a US SEC filer at all — not an error.
    payload = _ticker_map_payload([(320193, "AAPL", "Apple Inc.")])
    monkeypatch.setattr(news_data.requests, "get", lambda *a, **k: _FakeResponse(payload))
    assert _fetch_cik_for_ticker("SPY") is None


def test_fetch_cik_for_ticker_none_on_request_failure(monkeypatch):
    import requests as requests_module

    def _raise(*a, **k):
        raise requests_module.RequestException("network down")

    monkeypatch.setattr(news_data.requests, "get", _raise)
    assert _fetch_cik_for_ticker("AAPL") is None


# ---------------------------------------------------------------------------
# fetch_sec_filings — CIK filtering (the Apple / Apple Hospitality REIT bug)
# ---------------------------------------------------------------------------

def test_fetch_sec_filings_filters_out_an_unrelated_company_matching_by_name_only(monkeypatch):
    # A text search for "Apple" also matches *Apple Hospitality REIT* (a
    # completely different, unrelated company) alongside Apple Inc.'s own
    # filings. Only the hit whose CIK matches AAPL's real CIK should survive.
    search_payload = _edgar_payload([
        _hit(cik="0001418121", accession_id="0001418121-26-000050:aple.htm", display_name="Apple Hospitality REIT, Inc."),
        _hit(cik="0000320193", accession_id="0000320193-26-000106:aapl.htm", display_name="Apple Inc."),
    ])
    cik_lookup_payload = _ticker_map_payload([(320193, "AAPL", "Apple Inc.")])

    def _fake_get(url, **kwargs):
        if url == news_data.SEC_COMPANY_TICKERS_URL:
            return _FakeResponse(cik_lookup_payload)
        return _FakeResponse(search_payload)

    monkeypatch.setattr(news_data.requests, "get", _fake_get)

    results = fetch_sec_filings("Apple", "AAPL")

    assert len(results) == 1
    assert "Apple Inc." in results[0]["title"]


def test_fetch_sec_filings_returns_nothing_when_the_cik_lookup_fails(monkeypatch):
    # Without a verified CIK the name search cannot tell this company's
    # filings from another one's, so nothing is returned (and nothing searched).
    import requests as requests_module
    searched = []

    def _fake_get(url, **kwargs):
        if url == news_data.SEC_COMPANY_TICKERS_URL:
            raise requests_module.RequestException("network down")
        searched.append(url)
        return _FakeResponse(_edgar_payload([_hit(cik="0000320193", accession_id="0000320193-26-000106:aapl.htm")]))

    monkeypatch.setattr(news_data.requests, "get", _fake_get)
    assert fetch_sec_filings("Apple Inc.", "AAPL") == []
    assert searched == []


def test_fetch_sec_filings_returns_nothing_for_a_ticker_sec_does_not_list(monkeypatch):
    # MC.PA (LVMH) files with the AMF, not the SEC: a name search would only
    # surface other companies' filings that mention "LVMH".
    other_company = _edgar_payload([_hit(cik="0000012345", accession_id="0000012345-26-000001:x.htm", display_name="Some US Retailer")])
    monkeypatch.setattr(news_data.requests, "get", _sec_get(other_company))
    assert fetch_sec_filings("LVMH", "MC.PA") == []


def test_fetch_sec_filings_dedupes_duplicate_accession_numbers(monkeypatch):
    # The same filing can appear more than once among raw EDGAR search hits
    # (seen live) — must collapse to a single result, not be returned twice.
    dup_hit = _hit(cik="0000320193", accession_id="0000320193-26-000106:aapl.htm")
    other_hit = _hit(cik="0000320193", accession_id="0000320193-26-000107:aapl2.htm")
    payload = _edgar_payload([dup_hit, dup_hit, other_hit])
    monkeypatch.setattr(news_data.requests, "get", _sec_get(payload))

    results = fetch_sec_filings("Apple Inc.", "AAPL", max_filings=5)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# fetch_sec_insider_trades — Form 4, a thin wrapper over fetch_sec_filings
# ---------------------------------------------------------------------------

def test_fetch_sec_insider_trades_requests_form_4(monkeypatch):
    captured = {}

    route = _sec_get(_edgar_payload([_hit(cik="0000320193", accession_id="0000320193-26-000108:form4.htm")]))

    def _fake_get(url, params=None, headers=None, timeout=None):
        if url == news_data.SEC_FULLTEXT_SEARCH_URL:
            captured["params"] = params
        return route(url)

    monkeypatch.setattr(news_data.requests, "get", _fake_get)
    fetch_sec_insider_trades("Apple Inc.", "AAPL")
    assert captured["params"]["forms"] == "4"


def test_fetch_sec_insider_trades_returns_form_4_titled_results(monkeypatch):
    payload = _edgar_payload([_hit(cik="0000320193", accession_id="0000320193-26-000108:form4.htm")])
    monkeypatch.setattr(news_data.requests, "get", _sec_get(payload))

    results = fetch_sec_insider_trades("Apple Inc.", "AAPL")

    assert len(results) == 1
    assert results[0]["title"].startswith("4 filing:")
    assert results[0]["provider"] == "SEC EDGAR"


def test_fetch_sec_insider_trades_applies_the_same_cik_filter_as_8k(monkeypatch):
    # Same false-positive risk as 8-K filings (a bare name search matching an
    # unrelated company) -> Form 4 must go through the identical CIK filter.
    search_payload = _edgar_payload([
        _hit(cik="0001418121", accession_id="0001418121-26-000051:aple4.htm", display_name="Apple Hospitality REIT, Inc."),
        _hit(cik="0000320193", accession_id="0000320193-26-000108:aapl4.htm", display_name="Apple Inc."),
    ])
    cik_lookup_payload = _ticker_map_payload([(320193, "AAPL", "Apple Inc.")])

    def _fake_get(url, **kwargs):
        if url == news_data.SEC_COMPANY_TICKERS_URL:
            return _FakeResponse(cik_lookup_payload)
        return _FakeResponse(search_payload)

    monkeypatch.setattr(news_data.requests, "get", _fake_get)

    results = fetch_sec_insider_trades("Apple", "AAPL")

    assert len(results) == 1
    assert "Apple Inc." in results[0]["title"]


def test_fetch_sec_insider_trades_returns_empty_list_on_request_failure(monkeypatch):
    import requests as requests_module

    def _raise(*a, **k):
        raise requests_module.RequestException("network down")

    monkeypatch.setattr(news_data.requests, "get", _raise)
    assert fetch_sec_insider_trades("Apple Inc.", "AAPL") == []


# ---------------------------------------------------------------------------
# fetch_gdelt_news — free, no key, JSON DOC 2.0 API
# ---------------------------------------------------------------------------

def _gdelt_payload(articles: list[dict]) -> dict:
    return {"articles": articles}


def _gdelt_article(title="Apple announces new product", domain="reuters.com",
                    url="http://x", seendate="20260908T120000Z"):
    return {"title": title, "domain": domain, "url": url, "seendate": seendate}


def test_fetch_gdelt_news_returns_parsed_results(monkeypatch):
    payload = _gdelt_payload([_gdelt_article()])
    monkeypatch.setattr(news_data.requests, "get", lambda *a, **k: _FakeResponse(payload))

    results = fetch_gdelt_news("AAPL", "Apple")

    assert len(results) == 1
    assert results[0]["title"] == "Apple announces new product"
    assert results[0]["source"] == "reuters.com"
    assert results[0]["provider"] == "GDELT"
    assert results[0]["published_at"] == "2026-09-08T12:00:00+00:00"


def test_fetch_gdelt_news_respects_max_articles(monkeypatch):
    payload = _gdelt_payload([_gdelt_article(title=f"Headline {i}") for i in range(5)])
    monkeypatch.setattr(news_data.requests, "get", lambda *a, **k: _FakeResponse(payload))
    results = fetch_gdelt_news("AAPL", "Apple", max_articles=2)
    assert len(results) == 2


def test_fetch_gdelt_news_skips_entries_without_a_title(monkeypatch):
    payload = _gdelt_payload([_gdelt_article(title=""), _gdelt_article(title="Real headline")])
    monkeypatch.setattr(news_data.requests, "get", lambda *a, **k: _FakeResponse(payload))
    results = fetch_gdelt_news("AAPL", "Apple")
    assert len(results) == 1
    assert results[0]["title"] == "Real headline"


def test_fetch_gdelt_news_handles_malformed_date_gracefully(monkeypatch):
    payload = _gdelt_payload([_gdelt_article(seendate="not-a-date")])
    monkeypatch.setattr(news_data.requests, "get", lambda *a, **k: _FakeResponse(payload))
    results = fetch_gdelt_news("AAPL", "Apple")
    assert results[0]["published_at"] == ""  # degrades to empty string, never raises


def test_fetch_gdelt_news_returns_empty_list_on_request_failure(monkeypatch):
    import requests as requests_module

    def _raise(*a, **k):
        raise requests_module.RequestException("network down")

    monkeypatch.setattr(news_data.requests, "get", _raise)
    assert fetch_gdelt_news("AAPL", "Apple") == []


def test_fetch_gdelt_news_returns_empty_list_on_empty_payload(monkeypatch):
    monkeypatch.setattr(news_data.requests, "get", lambda *a, **k: _FakeResponse({}))
    assert fetch_gdelt_news("AAPL", "Apple") == []


def test_fetch_gdelt_news_never_sends_an_api_key(monkeypatch):
    # No key concept at all for GDELT — confirm the call succeeds with
    # nothing resembling an API-key parameter.
    captured = {}

    def _fake_get(url, params=None, timeout=None):
        captured["params"] = params
        return _FakeResponse(_gdelt_payload([_gdelt_article()]))

    monkeypatch.setattr(news_data.requests, "get", _fake_get)
    fetch_gdelt_news("AAPL", "Apple")
    assert not any("key" in str(k).lower() for k in captured["params"])


# ---------------------------------------------------------------------------
# fetch_google_news_rss — free, no key, plain RSS 2.0 XML
# ---------------------------------------------------------------------------

class _FakeRssResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests as requests_module
            raise requests_module.HTTPError(f"{self.status_code} error")


def _rss_feed(items: list[dict]) -> str:
    items_xml = ""
    for it in items:
        source_xml = f'<source url="http://s">{it.get("source", "")}</source>' if "source" in it else ""
        items_xml += (
            "<item>"
            f"<title>{it.get('title', '')}</title>"
            f"<link>{it.get('link', 'http://x')}</link>"
            f"<pubDate>{it.get('pubDate', 'Tue, 08 Sep 2026 12:00:00 GMT')}</pubDate>"
            f"<description>{it.get('description', '')}</description>"
            f"{source_xml}"
            "</item>"
        )
    return f"<?xml version='1.0'?><rss><channel>{items_xml}</channel></rss>"


def test_fetch_google_news_rss_returns_parsed_results(monkeypatch):
    feed = _rss_feed([{"title": "Apple beats estimates", "source": "Reuters"}])
    monkeypatch.setattr(news_data.requests, "get", lambda *a, **k: _FakeRssResponse(feed))

    results = fetch_google_news_rss("AAPL", "Apple")

    assert len(results) == 1
    assert results[0]["title"] == "Apple beats estimates"
    assert results[0]["source"] == "Reuters"
    assert results[0]["provider"] == "Google News"


def test_fetch_google_news_rss_respects_max_articles(monkeypatch):
    feed = _rss_feed([{"title": f"Headline {i}"} for i in range(5)])
    monkeypatch.setattr(news_data.requests, "get", lambda *a, **k: _FakeRssResponse(feed))
    results = fetch_google_news_rss("AAPL", "Apple", max_articles=2)
    assert len(results) == 2


def test_fetch_google_news_rss_skips_entries_without_a_title(monkeypatch):
    feed = _rss_feed([{"title": ""}, {"title": "Real headline"}])
    monkeypatch.setattr(news_data.requests, "get", lambda *a, **k: _FakeRssResponse(feed))
    results = fetch_google_news_rss("AAPL", "Apple")
    assert len(results) == 1
    assert results[0]["title"] == "Real headline"


def test_fetch_google_news_rss_defaults_source_to_unknown_without_a_source_tag(monkeypatch):
    feed = _rss_feed([{"title": "No source tag here"}])
    monkeypatch.setattr(news_data.requests, "get", lambda *a, **k: _FakeRssResponse(feed))
    results = fetch_google_news_rss("AAPL", "Apple")
    assert results[0]["source"] == "unknown"


def test_fetch_google_news_rss_returns_empty_list_on_request_failure(monkeypatch):
    import requests as requests_module

    def _raise(*a, **k):
        raise requests_module.RequestException("network down")

    monkeypatch.setattr(news_data.requests, "get", _raise)
    assert fetch_google_news_rss("AAPL", "Apple") == []


def test_fetch_google_news_rss_returns_empty_list_on_malformed_xml(monkeypatch):
    monkeypatch.setattr(news_data.requests, "get", lambda *a, **k: _FakeRssResponse("not valid xml{{{"))
    assert fetch_google_news_rss("AAPL", "Apple") == []


def test_fetch_google_news_rss_never_sends_an_api_key(monkeypatch):
    captured = {}

    def _fake_get(url, params=None, timeout=None):
        captured["params"] = params
        return _FakeRssResponse(_rss_feed([{"title": "X"}]))

    monkeypatch.setattr(news_data.requests, "get", _fake_get)
    fetch_google_news_rss("AAPL", "Apple")
    assert not any("key" in str(k).lower() for k in captured["params"])


# ---------------------------------------------------------------------------
# fetch_ted_notices — free, no key, POST-based EU procurement search
# ---------------------------------------------------------------------------

def _ted_payload(notices: list[dict]) -> dict:
    return {"notices": notices}


def _ted_notice(title="Cloud services framework agreement", buyer="City of Amsterdam",
                 country="NL", publication_number="123456-2026", pub_date="2026-09-01"):
    return {
        "notice-title": title,
        "buyer-name": buyer,
        "buyer-country": country,
        "publication-number": publication_number,
        "publication-date": pub_date,
    }


def test_fetch_ted_notices_returns_parsed_results(monkeypatch):
    payload = _ted_payload([_ted_notice()])
    monkeypatch.setattr(news_data.requests, "post", lambda *a, **k: _FakeResponse(payload))

    results = fetch_ted_notices("AAPL", "Apple")

    assert len(results) == 1
    assert results[0]["title"] == "Cloud services framework agreement"
    assert "City of Amsterdam" in results[0]["description"]
    assert results[0]["provider"] == "TED"
    assert results[0]["url"] == "https://ted.europa.eu/en/notice/-/detail/123456-2026"


def test_fetch_ted_notices_handles_language_keyed_title_field(monkeypatch):
    # TED's multilingual fields can come back as {"eng": "...", "fra": "..."}
    # rather than a plain string — must resolve to the English value.
    notice = _ted_notice()
    notice["notice-title"] = {"eng": "Cloud services framework agreement", "fra": "Accord-cadre de services cloud"}
    payload = _ted_payload([notice])
    monkeypatch.setattr(news_data.requests, "post", lambda *a, **k: _FakeResponse(payload))

    results = fetch_ted_notices("AAPL", "Apple")
    assert results[0]["title"] == "Cloud services framework agreement"


def test_fetch_ted_notices_respects_max_notices(monkeypatch):
    payload = _ted_payload([_ted_notice(title=f"Notice {i}") for i in range(5)])
    monkeypatch.setattr(news_data.requests, "post", lambda *a, **k: _FakeResponse(payload))
    results = fetch_ted_notices("AAPL", "Apple", max_notices=2)
    assert len(results) == 2


def test_fetch_ted_notices_skips_entries_without_a_title(monkeypatch):
    payload = _ted_payload([_ted_notice(title=""), _ted_notice(title="Real notice")])
    monkeypatch.setattr(news_data.requests, "post", lambda *a, **k: _FakeResponse(payload))
    results = fetch_ted_notices("AAPL", "Apple")
    assert len(results) == 1
    assert results[0]["title"] == "Real notice"


def test_fetch_ted_notices_falls_back_to_generic_url_without_a_publication_number(monkeypatch):
    notice = _ted_notice(publication_number="")
    payload = _ted_payload([notice])
    monkeypatch.setattr(news_data.requests, "post", lambda *a, **k: _FakeResponse(payload))
    results = fetch_ted_notices("AAPL", "Apple")
    assert results[0]["url"] == "https://ted.europa.eu/"


def test_fetch_ted_notices_returns_empty_list_on_request_failure(monkeypatch):
    import requests as requests_module

    def _raise(*a, **k):
        raise requests_module.RequestException("network down")

    monkeypatch.setattr(news_data.requests, "post", _raise)
    assert fetch_ted_notices("AAPL", "Apple") == []


def test_fetch_ted_notices_returns_empty_list_on_empty_payload(monkeypatch):
    monkeypatch.setattr(news_data.requests, "post", lambda *a, **k: _FakeResponse({}))
    assert fetch_ted_notices("AAPL", "Apple") == []


def test_fetch_ted_notices_never_sends_an_api_key(monkeypatch):
    captured = {}

    def _fake_post(url, json=None, timeout=None):
        captured["json"] = json
        return _FakeResponse(_ted_payload([_ted_notice()]))

    monkeypatch.setattr(news_data.requests, "post", _fake_post)
    fetch_ted_notices("AAPL", "Apple")
    assert not any("key" in str(k).lower() for k in captured["json"])


def test_fetch_ted_notices_renders_list_valued_fields_as_plain_text(monkeypatch):
    # Multi-valued TED fields come back as lists, directly or inside the
    # language-keyed dict — they must not render as a Python list repr.
    notice = _ted_notice(buyer={"deu": ["Stadt München"]}, country=["DEU"])
    monkeypatch.setattr(news_data.requests, "post", lambda *a, **k: _FakeResponse(_ted_payload([notice])))

    results = fetch_ted_notices("SAP", "SAP")

    assert results[0]["description"] == "Buyer: Stadt München (DEU)"


def test_fetch_ted_notices_omits_empty_country_parentheses(monkeypatch):
    notice = _ted_notice(country=None)
    monkeypatch.setattr(news_data.requests, "post", lambda *a, **k: _FakeResponse(_ted_payload([notice])))
    assert fetch_ted_notices("AAPL", "Apple")[0]["description"] == "Buyer: City of Amsterdam"


# ---------------------------------------------------------------------------
# Timestamps — every provider's published_at carries an explicit UTC offset
# ---------------------------------------------------------------------------

def test_fetch_finnhub_news_converts_epoch_in_utc_not_local_time(monkeypatch):
    import dataclasses
    import datetime as dt

    fake_settings = dataclasses.replace(news_data.LLM_SETTINGS, finnhub_api_key="fake-key")
    monkeypatch.setattr(news_data, "LLM_SETTINGS", fake_settings)
    epoch = int(dt.datetime(2026, 9, 8, 12, 0, tzinfo=dt.timezone.utc).timestamp())
    payload = [{"headline": "Apple beats estimates", "datetime": epoch, "source": "Reuters", "url": "http://x"}]
    monkeypatch.setattr(news_data.requests, "get", lambda *a, **k: _FakeResponse(payload))

    results = news_data.fetch_finnhub_news("AAPL")

    assert results[0]["published_at"] == "2026-09-08T12:00:00+00:00"


def test_fetch_google_news_rss_converts_pubdate_to_iso_utc(monkeypatch):
    feed = _rss_feed([{"title": "Apple beats estimates", "pubDate": "Tue, 08 Sep 2026 14:00:00 +0200"}])
    monkeypatch.setattr(news_data.requests, "get", lambda *a, **k: _FakeRssResponse(feed))
    assert fetch_google_news_rss("AAPL", "Apple")[0]["published_at"] == "2026-09-08T12:00:00+00:00"


def test_fetch_google_news_rss_keeps_an_unparsable_pubdate_unchanged(monkeypatch):
    feed = _rss_feed([{"title": "Apple beats estimates", "pubDate": "sometime last week"}])
    monkeypatch.setattr(news_data.requests, "get", lambda *a, **k: _FakeRssResponse(feed))
    assert fetch_google_news_rss("AAPL", "Apple")[0]["published_at"] == "sometime last week"


def test_fetch_sec_filings_does_not_crash_on_empty_display_names(monkeypatch):
    hit = _hit(cik="0000320193", accession_id="0000320193-26-000106:aapl-20260815.htm")
    hit["_source"]["display_names"] = []
    monkeypatch.setattr(news_data.requests, "get", _sec_get(_edgar_payload([hit])))

    results = fetch_sec_filings("Apple Inc.", "AAPL")

    assert results[0]["title"] == "8-K filing: Apple Inc."