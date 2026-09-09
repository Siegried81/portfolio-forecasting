"""
Unit tests for src/rag.py (TF-IDF retrieval over the news/filings corpus).

Priority: this module had NO test coverage before this pass, despite being
the actual retrieval step behind the chatbot's "answers about news" path in
ai_features.answer_portfolio_question. Focus on the contract callers rely on
(retrieve() never raises, ranks by relevance, drops zero-similarity noise)
rather than TF-IDF internals, which sklearn already tests upstream.

Also covers the 2026-09-08 Redis persistence addition (save_chunks/
load_chunks) — same spirit as test_cache.py: the FALLBACK/no-op paths matter
more than the happy path, since the whole design promise is "never make this
worse than today's session-only behaviour." A fake in-memory Redis client
stands in for a real server, same pattern as test_cache.py's _FakeRedisClient.
"""
import pytest

import src.rag as rag
from src.rag import Chunk, build_chunks, format_retrieved_chunks, load_chunks, retrieve, save_chunks


# ---------------------------------------------------------------------------
# build_chunks
# ---------------------------------------------------------------------------

def test_build_chunks_concatenates_title_and_description():
    articles = [{"title": "AAPL beats estimates", "description": "Strong iPhone sales.",
                 "source": "Reuters", "provider": "NewsAPI", "ticker": "AAPL", "url": "http://x"}]
    chunks = build_chunks(articles)
    assert len(chunks) == 1
    assert chunks[0].text == "AAPL beats estimates. Strong iPhone sales."
    assert chunks[0].ticker == "AAPL"


def test_build_chunks_skips_articles_without_a_title():
    articles = [{"title": "", "description": "no title here"}, {"title": "Real headline", "description": ""}]
    chunks = build_chunks(articles)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Real headline")


def test_build_chunks_empty_input_returns_empty_list():
    assert build_chunks([]) == []


# ---------------------------------------------------------------------------
# retrieve — the actual TF-IDF ranking
# ---------------------------------------------------------------------------

def _chunk(text: str, ticker: str = "AAPL") -> Chunk:
    return Chunk(text=text, source="Reuters", provider="NewsAPI", ticker=ticker, url="http://x")


def test_retrieve_ranks_the_most_relevant_chunk_first():
    chunks = [
        _chunk("Quarterly earnings beat analyst expectations on strong iPhone demand"),
        _chunk("The weather in Cupertino was sunny this week"),
        _chunk("Apple announces new retail store opening in Tokyo"),
    ]
    results = retrieve("earnings iPhone demand", chunks, top_k=2)
    assert len(results) >= 1
    assert results[0].text.startswith("Quarterly earnings")


def test_retrieve_filters_out_zero_similarity_chunks():
    # Query shares no vocabulary at all with either chunk -> nothing should be relevant.
    chunks = [_chunk("Quarterly earnings beat expectations"), _chunk("New product launch event")]
    results = retrieve("xyzxyz nonword qqqqq", chunks, top_k=5)
    assert results == []


def test_retrieve_empty_chunks_returns_empty_list():
    assert retrieve("any query", [], top_k=4) == []


def test_retrieve_empty_query_returns_empty_list():
    chunks = [_chunk("Some headline about earnings")]
    assert retrieve("   ", chunks, top_k=4) == []


def test_retrieve_respects_top_k():
    chunks = [_chunk(f"Earnings report number {i} beats expectations") for i in range(10)]
    results = retrieve("earnings report expectations", chunks, top_k=3)
    assert len(results) <= 3


def test_retrieve_never_raises_on_stopword_only_corpus():
    # Every chunk + the query reduce to nothing but stopwords after cleaning —
    # TfidfVectorizer raises ValueError internally; retrieve() must swallow it.
    chunks = [_chunk("the a an of")]
    assert retrieve("the a an", chunks, top_k=4) == []


# ---------------------------------------------------------------------------
# format_retrieved_chunks
# ---------------------------------------------------------------------------

def test_format_retrieved_chunks_empty_list_returns_empty_string():
    assert format_retrieved_chunks([]) == ""


def test_format_retrieved_chunks_tags_provider_and_source():
    chunks = [_chunk("Earnings beat expectations")]
    formatted = format_retrieved_chunks(chunks)
    assert "[NewsAPI/Reuters, AAPL]" in formatted
    assert "Earnings beat expectations" in formatted


# ---------------------------------------------------------------------------
# save_chunks / load_chunks — Redis persistence (opt-in, fails soft)
# ---------------------------------------------------------------------------

class _FakeRedisClient:
    """In-memory stand-in for a redis.Redis client — same pattern as
    test_cache.py's own _FakeRedisClient, just the get/setex surface these
    two functions actually use."""
    def __init__(self):
        self.store: dict[str, str] = {}

    def get(self, key: str):
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str):
        self.store[key] = value


def test_save_and_load_chunks_round_trips_through_a_configured_redis(monkeypatch):
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(rag, "get_redis_client", lambda: fake_client)
    chunks = [_chunk("Earnings beat expectations", ticker="AAPL")]

    save_chunks(chunks, ["AAPL"])
    loaded = load_chunks(["AAPL"])

    assert loaded == chunks


def test_load_chunks_is_a_noop_returning_empty_list_without_redis(monkeypatch):
    # No REDIS_URL configured — the default, every local/free-tier deployment.
    monkeypatch.setattr(rag, "get_redis_client", lambda: None)
    assert load_chunks(["AAPL"]) == []


def test_save_chunks_is_a_noop_without_redis(monkeypatch):
    # Must not raise, and must not attempt to touch a client that doesn't exist.
    monkeypatch.setattr(rag, "get_redis_client", lambda: None)
    save_chunks([_chunk("Whatever")], ["AAPL"])  # no assertion needed — just must not raise


def test_save_chunks_does_not_persist_an_empty_corpus(monkeypatch):
    # An empty fetch (e.g. every news source temporarily down) overwriting a
    # REAL previously-persisted corpus would be a regression, not a cache
    # update — save_chunks([], ...) must be a no-op, not a corpus-clearing write.
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(rag, "get_redis_client", lambda: fake_client)
    save_chunks([_chunk("Real headline")], ["AAPL"])
    save_chunks([], ["AAPL"])  # must NOT overwrite the real entry above
    assert load_chunks(["AAPL"]) == [_chunk("Real headline")]


def test_load_chunks_returns_empty_list_when_nothing_persisted_yet(monkeypatch):
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(rag, "get_redis_client", lambda: fake_client)
    assert load_chunks(["MSFT"]) == []  # nothing ever saved under this ticker set


def test_persistence_key_ignores_ticker_order_and_case(monkeypatch):
    # A user re-selecting the same universe in a different order (or a
    # different case) should still hit the same persisted corpus.
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(rag, "get_redis_client", lambda: fake_client)
    save_chunks([_chunk("Some news")], ["AAPL", "msft"])
    assert load_chunks(["MSFT", "aapl"]) == [_chunk("Some news")]


def test_load_chunks_returns_empty_list_on_malformed_payload(monkeypatch):
    # A corrupted/truncated Redis value (e.g. a manual edit, a schema change
    # in a future version) must degrade to "nothing persisted", not crash the
    # news tab.
    fake_client = _FakeRedisClient()
    fake_client.store["rag:chunks:AAPL"] = "not valid json{{{"
    monkeypatch.setattr(rag, "get_redis_client", lambda: fake_client)
    assert load_chunks(["AAPL"]) == []


def test_load_chunks_returns_empty_list_when_redis_get_raises(monkeypatch):
    class _BrokenClient:
        def get(self, key):
            raise ConnectionError("connection reset")

    monkeypatch.setattr(rag, "get_redis_client", lambda: _BrokenClient())
    assert load_chunks(["AAPL"]) == []


def test_save_chunks_does_not_raise_when_redis_setex_fails(monkeypatch):
    class _BrokenClient:
        def setex(self, key, ttl, value):
            raise ConnectionError("connection reset")

    monkeypatch.setattr(rag, "get_redis_client", lambda: _BrokenClient())
    save_chunks([_chunk("Whatever")], ["AAPL"])  # must not raise