"""Unit tests for src/cache.py. Redis paths patch REDIS_URL/_get_redis_client instead of needing a real server."""
from __future__ import annotations

import dataclasses

import pytest
import streamlit as st

import src.cache as cache_module
from src.cache import _make_cache_key, cached


# ---------------------------------------------------------------------------
# No REDIS_URL configured
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
    assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# REDIS_URL configured but unreachable
# ---------------------------------------------------------------------------

def test_cached_degrades_to_direct_calls_when_redis_unreachable(monkeypatch):
    monkeypatch.setattr(cache_module, "REDIS_URL", "redis://nonexistent-host:6379/0")
    monkeypatch.setattr(cache_module, "_redis_client", None)
    monkeypatch.setattr(cache_module, "_redis_unavailable", False)
    monkeypatch.setattr(cache_module, "_get_redis_client", lambda: None)

    @cached(ttl_seconds=60)
    def compute(x: int) -> int:
        return x + 1

    assert compute(4) == 5


# ---------------------------------------------------------------------------
# REDIS_URL configured and reachable (fake client)
# ---------------------------------------------------------------------------

class _FakeRedisClient:
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
    assert compute(3) == 30
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
    assert len(fake_client.store) == 2


def test_cached_falls_back_to_direct_call_on_non_serialisable_result(monkeypatch):
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(cache_module, "REDIS_URL", "redis://fake:6379/0")
    monkeypatch.setattr(cache_module, "_get_redis_client", lambda: fake_client)

    @cached(ttl_seconds=60)
    def compute() -> object:
        return object()

    result = compute()
    assert isinstance(result, object)
    assert len(fake_client.store) == 0


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

# ---------------------------------------------------------------------------
# Fallback paths still cache in-process (never uncached direct calls)
# ---------------------------------------------------------------------------

def test_cached_uses_in_process_cache_when_redis_unreachable(monkeypatch):
    monkeypatch.setattr(cache_module, "REDIS_URL", "redis://nonexistent-host:6379/0")
    monkeypatch.setattr(cache_module, "_get_redis_client", lambda: None)
    st.cache_data.clear()
    call_count = {"n": 0}

    @cached(ttl_seconds=60)
    def compute_unreachable(x: int) -> int:
        call_count["n"] += 1
        return x + 1

    assert compute_unreachable(4) == 5
    assert compute_unreachable(4) == 5
    assert call_count["n"] == 1


def test_cached_uses_in_process_cache_when_redis_get_fails(monkeypatch):
    class _BrokenGetClient(_FakeRedisClient):
        def get(self, key: str):
            raise ConnectionError("redis connection reset")

    monkeypatch.setattr(cache_module, "REDIS_URL", "redis://fake:6379/0")
    monkeypatch.setattr(cache_module, "_get_redis_client", lambda: _BrokenGetClient())
    st.cache_data.clear()
    call_count = {"n": 0}

    @cached(ttl_seconds=60)
    def compute_broken_get(x: int) -> int:
        call_count["n"] += 1
        return x * 3

    assert compute_broken_get(2) == 6
    assert compute_broken_get(2) == 6
    assert call_count["n"] == 1


def test_cached_caches_non_serialisable_results_in_process(monkeypatch):
    import pandas as pd

    fake_client = _FakeRedisClient()
    monkeypatch.setattr(cache_module, "REDIS_URL", "redis://fake:6379/0")
    monkeypatch.setattr(cache_module, "_get_redis_client", lambda: fake_client)
    st.cache_data.clear()
    call_count = {"n": 0}

    @cached(ttl_seconds=60)
    def compute_frame(n: int) -> pd.DataFrame:
        call_count["n"] += 1
        return pd.DataFrame({"x": range(n)})

    first = compute_frame(3)
    second = compute_frame(3)
    third = compute_frame(3)
    assert first.equals(second) and second.equals(third)
    # First call learns the result can't go to Redis; later calls are served
    # from the in-process cache instead of recomputing.
    assert call_count["n"] == 2
    assert len(fake_client.store) == 0


# ---------------------------------------------------------------------------
# Failed results expire after FAILURE_TTL_SECONDS, not the full TTL
# ---------------------------------------------------------------------------

def _clock(monkeypatch, start: float = 1_000_000.0):
    now = {"t": start}
    monkeypatch.setattr(cache_module.time, "time", lambda: now["t"])
    return now


def test_cached_retries_a_failed_result_after_the_failure_ttl(monkeypatch):
    monkeypatch.setattr(cache_module, "REDIS_URL", None)
    st.cache_data.clear()
    now = _clock(monkeypatch)
    answers = iter([None, {"rate": 0.04}])
    calls = {"n": 0}

    @cached(ttl_seconds=6 * 3600)
    def fetch() -> dict | None:
        calls["n"] += 1
        return next(answers)

    assert fetch() is None
    now["t"] += cache_module.FAILURE_TTL_SECONDS - 1
    assert fetch() is None and calls["n"] == 1  # still inside the failure window: no new call
    now["t"] += 2
    assert fetch() == {"rate": 0.04} and calls["n"] == 2


def test_cached_keeps_a_successful_result_for_the_full_ttl(monkeypatch):
    monkeypatch.setattr(cache_module, "REDIS_URL", None)
    st.cache_data.clear()
    now = _clock(monkeypatch)
    calls = {"n": 0}

    @cached(ttl_seconds=6 * 3600)
    def fetch() -> dict:
        calls["n"] += 1
        return {"rate": 0.04}

    fetch()
    now["t"] += cache_module.FAILURE_TTL_SECONDS * 10
    fetch()
    assert calls["n"] == 1


@pytest.mark.parametrize("result, failed", [
    (None, True), ([], True), ({}, True), ({"a": None, "b": None}, True),
    ({"a": None, "b": 1.0}, False), ([{"title": "x"}], False), (0.0, False),
])
def test_looks_failed(result, failed):
    assert cache_module._looks_failed(result) is failed


def test_cached_stores_a_failed_result_in_redis_with_the_short_ttl(monkeypatch):
    client = _FakeRedisClient()
    ttls = {}
    client.setex = lambda key, ttl, value: ttls.setdefault(key, ttl)
    monkeypatch.setattr(cache_module, "REDIS_URL", "redis://fake:6379/0")
    monkeypatch.setattr(cache_module, "_get_redis_client", lambda: client)

    @cached(ttl_seconds=6 * 3600)
    def fetch(ok: bool) -> list:
        return ["item"] if ok else []

    fetch(True)
    fetch(False)
    assert sorted(ttls.values()) == [cache_module.FAILURE_TTL_SECONDS, 6 * 3600]
