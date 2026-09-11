"""Unit tests for src/rag.py (TF-IDF retrieval) and its Redis persistence."""
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
# retrieve
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
# save_chunks / load_chunks
# ---------------------------------------------------------------------------

class _FakeRedisClient:
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
    monkeypatch.setattr(rag, "get_redis_client", lambda: None)
    assert load_chunks(["AAPL"]) == []


def test_save_chunks_is_a_noop_without_redis(monkeypatch):
    monkeypatch.setattr(rag, "get_redis_client", lambda: None)
    save_chunks([_chunk("Whatever")], ["AAPL"])


def test_save_chunks_does_not_persist_an_empty_corpus(monkeypatch):
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(rag, "get_redis_client", lambda: fake_client)
    save_chunks([_chunk("Real headline")], ["AAPL"])
    save_chunks([], ["AAPL"])
    assert load_chunks(["AAPL"]) == [_chunk("Real headline")]


def test_load_chunks_returns_empty_list_when_nothing_persisted_yet(monkeypatch):
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(rag, "get_redis_client", lambda: fake_client)
    assert load_chunks(["MSFT"]) == []


def test_persistence_key_ignores_ticker_order_and_case(monkeypatch):
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(rag, "get_redis_client", lambda: fake_client)
    save_chunks([_chunk("Some news")], ["AAPL", "msft"])
    assert load_chunks(["MSFT", "aapl"]) == [_chunk("Some news")]


def test_load_chunks_returns_empty_list_on_malformed_payload(monkeypatch):
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
    save_chunks([_chunk("Whatever")], ["AAPL"])