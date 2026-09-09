"""
Unit tests for src/academic_search.py. Had no test coverage before this pass
(new module). Priority, same as every other fetcher in this codebase: the
FAILS-SOFT paths (missing key, network error, malformed response) matter as
much as the happy path, since this is enrichment (real paper citations for
methodological chatbot questions) that must never be able to take a chatbot
answer down. Fully mocked — no real network call, no Semantic Scholar rate
limit hit.
"""
import dataclasses

import pytest
import requests
import streamlit as st

import src.academic_search as academic_search
from src.academic_search import (
    METHODOLOGY_SEARCH_QUERIES,
    detect_methodology_terms,
    format_papers_for_prompt,
    search_academic_papers,
    search_arxiv_papers,
)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Isolate every test from Streamlit's process-wide @cached fallthrough
    cache — same pattern as test_news_data.py's own fixture, for the same
    reason (identical args across tests would otherwise return a stale
    cached result instead of re-invoking the mocked requests.get)."""
    st.cache_data.clear()
    fake_settings = dataclasses.replace(academic_search.LLM_SETTINGS, semantic_scholar_api_key=None)
    monkeypatch.setattr(academic_search, "LLM_SETTINGS", fake_settings)
    yield
    st.cache_data.clear()


class _FakeResponse:
    def __init__(self, json_data: dict):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def _paper(title="A Paper", authors=("Jane Doe",), year=2020, url="http://x", citations=10, venue="Journal of X"):
    return {
        "title": title,
        "authors": [{"name": a} for a in authors],
        "year": year,
        "url": url,
        "citationCount": citations,
        "venue": venue,
    }


# ---------------------------------------------------------------------------
# detect_methodology_terms
# ---------------------------------------------------------------------------

def test_detect_methodology_terms_matches_a_known_term():
    assert "garch" in detect_methodology_terms("Why did you use GARCH for volatility?")


def test_detect_methodology_terms_case_insensitive():
    assert "sharpe ratio" in detect_methodology_terms("what is the SHARPE RATIO exactly")


def test_detect_methodology_terms_empty_for_unrelated_question():
    assert detect_methodology_terms("what is my portfolio's current return?") == []


def test_detect_methodology_terms_matches_whole_words_only():
    # "arima" must not match inside an unrelated longer word.
    assert detect_methodology_terms("pharmacist recommendation") == []


def test_detect_methodology_terms_can_return_multiple_matches():
    matches = detect_methodology_terms("compare GARCH and ARIMA for this")
    assert "garch" in matches
    assert "arima" in matches


def test_every_methodology_term_has_a_search_query():
    # Sanity check on the data itself: no term should map to an empty query.
    for term, query in METHODOLOGY_SEARCH_QUERIES.items():
        assert query.strip(), f"{term!r} maps to an empty search query"


# ---------------------------------------------------------------------------
# _search_semantic_scholar (renamed from search_academic_papers when arXiv
# was added as a second source — same logic, own isolated tests)
# ---------------------------------------------------------------------------

def test_search_semantic_scholar_returns_parsed_results(monkeypatch):
    payload = {"data": [_paper(title="GARCH Models", authors=("Bollerslev",), year=1986)]}
    monkeypatch.setattr(academic_search.requests, "get", lambda *a, **k: _FakeResponse(payload))

    results = academic_search._search_semantic_scholar("GARCH volatility forecasting", limit=3)

    assert len(results) == 1
    assert results[0]["title"] == "GARCH Models"
    assert results[0]["authors"] == ["Bollerslev"]
    assert results[0]["year"] == 1986


def test_search_semantic_scholar_respects_limit(monkeypatch):
    payload = {"data": [_paper(title=f"Paper {i}") for i in range(5)]}
    monkeypatch.setattr(academic_search.requests, "get", lambda *a, **k: _FakeResponse(payload))

    results = academic_search._search_semantic_scholar("some query", limit=2)
    assert len(results) == 2


def test_search_semantic_scholar_skips_entries_without_a_title(monkeypatch):
    payload = {"data": [_paper(title=""), _paper(title="Real Paper")]}
    monkeypatch.setattr(academic_search.requests, "get", lambda *a, **k: _FakeResponse(payload))

    results = academic_search._search_semantic_scholar("query", limit=3)
    assert len(results) == 1
    assert results[0]["title"] == "Real Paper"


def test_search_semantic_scholar_returns_empty_list_on_request_failure(monkeypatch):
    import requests as requests_module

    def _raise(*a, **k):
        raise requests_module.RequestException("network down")

    monkeypatch.setattr(academic_search.requests, "get", _raise)
    assert academic_search._search_semantic_scholar("query", limit=3) == []


def test_search_semantic_scholar_returns_empty_list_when_no_results(monkeypatch):
    monkeypatch.setattr(academic_search.requests, "get", lambda *a, **k: _FakeResponse({"data": []}))
    assert academic_search._search_semantic_scholar("nonsense query xyz", limit=3) == []


def test_search_semantic_scholar_works_without_an_api_key(monkeypatch):
    # UNLIKE every other provider in this app, Semantic Scholar's search
    # works fully unauthenticated (a lower shared rate limit, not a hard
    # requirement) — confirm the call succeeds with no key configured and no
    # auth header is even sent.
    captured = {}

    def _fake_get(url, params=None, headers=None, timeout=None):
        captured["headers"] = headers
        return _FakeResponse({"data": [_paper()]})

    monkeypatch.setattr(academic_search.requests, "get", _fake_get)
    results = academic_search._search_semantic_scholar("query", limit=3)
    assert len(results) == 1
    assert "x-api-key" not in captured["headers"]


def test_search_semantic_scholar_sends_api_key_header_when_configured(monkeypatch):
    fake_settings = dataclasses.replace(academic_search.LLM_SETTINGS, semantic_scholar_api_key="fake-test-key")
    monkeypatch.setattr(academic_search, "LLM_SETTINGS", fake_settings)
    captured = {}

    def _fake_get(url, params=None, headers=None, timeout=None):
        captured["headers"] = headers
        return _FakeResponse({"data": []})

    monkeypatch.setattr(academic_search.requests, "get", _fake_get)
    academic_search._search_semantic_scholar("query", limit=3)
    assert captured["headers"]["x-api-key"] == "fake-test-key"


# ---------------------------------------------------------------------------
# search_arxiv_papers — genuinely free, no key, Atom XML instead of JSON
# ---------------------------------------------------------------------------

def _arxiv_atom_feed(entries: list[dict]) -> str:
    """Build a minimal, valid arXiv-shaped Atom XML feed for `entries` — each
    a dict with optional title/authors (list[str])/year/id."""
    entries_xml = ""
    for e in entries:
        authors_xml = "".join(f"<author><name>{a}</name></author>" for a in e.get("authors", []))
        entries_xml += (
            f"<entry>"
            f"<id>{e.get('id', 'http://arxiv.org/abs/0000.00000')}</id>"
            f"<title>{e.get('title', '')}</title>"
            f"<published>{e.get('year', 2020)}-01-01T00:00:00Z</published>"
            f"{authors_xml}"
            f"</entry>"
        )
    return f'<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">{entries_xml}</feed>'


class _FakeArxivResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def test_search_arxiv_papers_returns_parsed_results(monkeypatch):
    feed = _arxiv_atom_feed([{"title": "A Long Short-Term Memory Model", "authors": ["Sepp Hochreiter"], "year": 1997}])
    monkeypatch.setattr(academic_search.requests, "get", lambda *a, **k: _FakeArxivResponse(feed))

    results = search_arxiv_papers("LSTM time series")

    assert len(results) == 1
    assert results[0]["title"] == "A Long Short-Term Memory Model"
    assert results[0]["authors"] == ["Sepp Hochreiter"]
    assert results[0]["year"] == 1997
    assert results[0]["citation_count"] is None
    assert results[0]["venue"] == "arXiv preprint"


def test_search_arxiv_papers_respects_limit(monkeypatch):
    feed = _arxiv_atom_feed([{"title": f"Paper {i}"} for i in range(5)])
    monkeypatch.setattr(academic_search.requests, "get", lambda *a, **k: _FakeArxivResponse(feed))
    results = search_arxiv_papers("query", limit=2)
    assert len(results) == 2


def test_search_arxiv_papers_skips_entries_without_a_title(monkeypatch):
    feed = _arxiv_atom_feed([{"title": ""}, {"title": "Real Paper"}])
    monkeypatch.setattr(academic_search.requests, "get", lambda *a, **k: _FakeArxivResponse(feed))
    results = search_arxiv_papers("query")
    assert len(results) == 1
    assert results[0]["title"] == "Real Paper"


def test_search_arxiv_papers_returns_empty_list_on_request_failure(monkeypatch):
    def _raise(*a, **k):
        raise requests.RequestException("network down")
    monkeypatch.setattr(academic_search.requests, "get", _raise)
    assert search_arxiv_papers("query") == []


def test_search_arxiv_papers_returns_empty_list_on_malformed_xml(monkeypatch):
    monkeypatch.setattr(academic_search.requests, "get", lambda *a, **k: _FakeArxivResponse("not valid xml{{{"))
    assert search_arxiv_papers("query") == []


def test_search_arxiv_papers_never_sends_any_auth_header(monkeypatch):
    # No API key concept at all for arXiv — confirm no key-lookup crash and
    # no auth header, structurally different from every other provider.
    captured = {}

    def _fake_get(url, params=None, timeout=None):
        captured["params"] = params
        return _FakeArxivResponse(_arxiv_atom_feed([{"title": "X"}]))

    monkeypatch.setattr(academic_search.requests, "get", _fake_get)
    search_arxiv_papers("query")
    assert "headers" not in captured  # no headers kwarg passed at all


# ---------------------------------------------------------------------------
# search_academic_papers — the public orchestrator (Semantic Scholar + arXiv)
# ---------------------------------------------------------------------------

def test_search_academic_papers_combines_both_sources(monkeypatch):
    def _fake_get(url, **kwargs):
        if url == academic_search.SEMANTIC_SCHOLAR_SEARCH_URL:
            return _FakeResponse({"data": [_paper(title="Semantic Scholar Paper")]})
        return _FakeArxivResponse(_arxiv_atom_feed([{"title": "ArXiv Paper"}]))

    monkeypatch.setattr(academic_search.requests, "get", _fake_get)
    results = search_academic_papers("query", limit=5)
    titles = {r["title"] for r in results}
    assert "Semantic Scholar Paper" in titles
    assert "ArXiv Paper" in titles


def test_search_academic_papers_only_queries_arxiv_for_remaining_slots(monkeypatch):
    # Semantic Scholar alone already fills the requested limit -> arXiv
    # must not even be called.
    calls = {"arxiv": 0}

    def _fake_get(url, **kwargs):
        if url == academic_search.SEMANTIC_SCHOLAR_SEARCH_URL:
            return _FakeResponse({"data": [_paper(title=f"SS Paper {i}") for i in range(3)]})
        calls["arxiv"] += 1
        return _FakeArxivResponse(_arxiv_atom_feed([{"title": "Should not appear"}]))

    monkeypatch.setattr(academic_search.requests, "get", _fake_get)
    results = search_academic_papers("query", limit=3)
    assert len(results) == 3
    assert calls["arxiv"] == 0


def test_search_academic_papers_dedupes_by_normalised_title(monkeypatch):
    # Same paper (case/whitespace-insensitive match) indexed on both sources
    # -> must appear only once, not twice.
    def _fake_get(url, **kwargs):
        if url == academic_search.SEMANTIC_SCHOLAR_SEARCH_URL:
            return _FakeResponse({"data": [_paper(title="GARCH Models  ")]})
        return _FakeArxivResponse(_arxiv_atom_feed([{"title": "garch models"}, {"title": "A Different Paper"}]))

    monkeypatch.setattr(academic_search.requests, "get", _fake_get)
    results = search_academic_papers("query", limit=5)
    titles = [r["title"] for r in results]
    assert len(titles) == 2  # "GARCH Models" once (deduped) + "A Different Paper"
    assert "A Different Paper" in titles


def test_search_academic_papers_falls_back_entirely_to_arxiv_when_semantic_scholar_fails(monkeypatch):
    def _fake_get(url, **kwargs):
        if url == academic_search.SEMANTIC_SCHOLAR_SEARCH_URL:
            raise requests.RequestException("down")
        return _FakeArxivResponse(_arxiv_atom_feed([{"title": "ArXiv Only Paper"}]))

    monkeypatch.setattr(academic_search.requests, "get", _fake_get)
    results = search_academic_papers("query", limit=3)
    assert len(results) == 1
    assert results[0]["title"] == "ArXiv Only Paper"


def test_search_academic_papers_empty_list_when_both_sources_fail(monkeypatch):
    monkeypatch.setattr(
        academic_search.requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.RequestException("down")),
    )
    assert search_academic_papers("query", limit=3) == []


# ---------------------------------------------------------------------------
# format_papers_for_prompt
# ---------------------------------------------------------------------------

def test_format_papers_for_prompt_empty_list_returns_empty_string():
    assert format_papers_for_prompt([]) == ""


def test_format_papers_for_prompt_includes_title_authors_year():
    papers = [{"title": "GARCH Models", "authors": ["Bollerslev"], "year": 1986, "url": "http://x",
               "citation_count": 5000, "venue": "Journal of Econometrics"}]
    formatted = format_papers_for_prompt(papers)
    assert "GARCH Models" in formatted
    assert "Bollerslev" in formatted
    assert "1986" in formatted
    assert "http://x" in formatted


def test_format_papers_for_prompt_truncates_authors_with_et_al():
    papers = [{"title": "X", "authors": ["A", "B", "C", "D", "E"], "year": 2020, "url": "", "citation_count": None, "venue": ""}]
    formatted = format_papers_for_prompt(papers)
    assert "et al." in formatted
    assert "A, B, C et al." in formatted  # first 3 authors only, before "et al."
    assert ", D" not in formatted  # the 4th/5th author must not leak into the list