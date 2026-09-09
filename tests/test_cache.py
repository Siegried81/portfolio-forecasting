"""
Unit tests for src/cache.py.

Priority, same spirit as the rest of this test suite (llm_client's Groq/Ollama
cascade, market_data's Yahoo circuit breaker): the FALLBACK paths matter more
than the happy path, because this decorator's whole design promise is "never
make the app worse than @st.cache_data alone." Redis itself is never actually
required to run these tests — REDIS_URL is unset in the test environment by
default, so `cached()` returns `st.cache_data` directly; the Redis-path tests
below patch `src.cache.REDIS_URL` and `_get_redis_client` instead of needing a
real server.
"""
from __future__ import annotations

import dataclasses

import pytest
import streamlit as st

import src.cache as cache_module
from src.cache import _make_cache_key, cached


# ---------------------------------------------------------------------------
# No REDIS_URL configured — the default, every local/free-tier deployment
# ---------------------------------------------------------------------------

def test_cached_falls_through_to_st_cache_data_with_no_redis_url(monkeypatch):
    monkeypatch.setattr(cache_module, "REDIS_URL", None)
    st.cache_data.clear()
    call_count = {"n": 0}

    @cached(ttl_seconds=60)
    def compute(x: int) -> int:
        call_count["n"] += 1
        return x * 2

    assert compute(5) == 10
    assert compute(5) == 10
    # st.cache_data itself is responsible for not re-calling on the second
    # hit — this just confirms `cached()` actually delegated to it rather
    # than silently becoming a no-op decorator.
    assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# REDIS_URL configured but unreachable — must degrade to direct calls, never raise
# ---------------------------------------------------------------------------

def test_cached_degrades_to_direct_calls_when_redis_unreachable(monkeypatch):
    monkeypatch.setattr(cache_module, "REDIS_URL", "redis://nonexistent-host:6379/0")
    monkeypatch.setattr(cache_module, "_redis_client", None)
    monkeypatch.setattr(cache_module, "_redis_unavailable", False)
    monkeypatch.setattr(cache_module, "_get_redis_client", lambda: None)

    @cached(ttl_seconds=60)
    def compute(x: int) -> int:
        return x + 1

    # Must not raise, and must still return the correct result.
    assert compute(4) == 5


# ---------------------------------------------------------------------------
# REDIS_URL configured and reachable (fake client) — real cache-hit behaviour
# ---------------------------------------------------------------------------

class _FakeRedisClient:
    """In-memory stand-in for a redis.Redis client — enough surface
    (get/setex) to exercise cached()'s real Redis path without a live server."""
    def __init__(self):
        self.store: dict[str, str] = {}

    def get(self, key: str):
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str):
        self.store[key] = value


def test_cached_hits_redis_on_second_call(monkeypatch):
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(cache_module, "REDIS_URL", "redis://fake:6379/0")
    monkeypatch.setattr(cache_module, "_get_redis_client", lambda: fake_client)
    call_count = {"n": 0}

    @cached(ttl_seconds=60)
    def compute(x: int) -> int:
        call_count["n"] += 1
        return x * 10

    assert compute(3) == 30
    assert compute(3) == 30  # second call should hit the fake Redis store, not re-run compute()
    assert call_count["n"] == 1
    assert len(fake_client.store) == 1


def test_cached_treats_different_args_as_different_cache_entries(monkeypatch):
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(cache_module, "REDIS_URL", "redis://fake:6379/0")
    monkeypatch.setattr(cache_module, "_get_redis_client", lambda: fake_client)

    @cached(ttl_seconds=60)
    def compute(x: int) -> int:
        return x * 10

    compute(1)
    compute(2)
    assert len(fake_client.store) == 2  # distinct keys, not a collision


def test_cached_falls_back_to_direct_call_on_non_serialisable_result(monkeypatch):
    # A result that isn't JSON-serialisable (e.g. a stray object) must not
    # crash the call — it just won't be cached for next time. This is the
    # "a broken cache should never take the app down" guarantee from the
    # module docstring, exercised on the write side rather than the read side.
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(cache_module, "REDIS_URL", "redis://fake:6379/0")
    monkeypatch.setattr(cache_module, "_get_redis_client", lambda: fake_client)

    @cached(ttl_seconds=60)
    def compute() -> object:
        return object()  # not JSON-serialisable

    result = compute()  # must not raise
    assert isinstance(result, object)
    assert len(fake_client.store) == 0  # nothing got cached, and that's fine


# ---------------------------------------------------------------------------
# _make_cache_key
# ---------------------------------------------------------------------------

def test_make_cache_key_is_stable_regardless_of_kwarg_order():
    def f(a: int, b: int) -> int:
        return a + b

    key1 = _make_cache_key(f, (), {"a": 1, "b": 2})
    key2 = _make_cache_key(f, (), {"b": 2, "a": 1})
    assert key1 == key2


def test_make_cache_key_differs_for_different_arguments():
    def f(a: int) -> int:
        return a

    assert _make_cache_key(f, (1,), {}) != _make_cache_key(f, (2,), {})