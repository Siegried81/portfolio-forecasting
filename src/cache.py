"""
Optional Redis-backed cache — a production next-step for the gap
`@st.cache_data` cannot close: its cache lives inside ONE process, so it's
lost on every container restart and never shared across multiple app
instances behind a load balancer. Real production deployments (autoscaled
containers, multiple replicas) need a cache that survives both.

Deliberately opt-in and fails soft, matching this app's existing philosophy
(Yahoo circuit breaker, Groq->Ollama fallback): with no `REDIS_URL` set — the
default, and what every local/free-tier deployment already uses — `cached()`
falls straight through to `@st.cache_data`, so nothing about today's app
behaviour changes unless `REDIS_URL` is explicitly configured. If Redis IS
configured but unreachable at call time (a network blip, a restart), calls
fall back to the same `@st.cache_data` in-process cache rather than raising
— a cache is an optimisation, and a broken one should never be able to take
the app down.

Usage: replace `@st.cache_data(show_spinner=False, ttl=1800)` with
`@cached(ttl_seconds=1800)`. Only JSON-serialisable return values (dicts,
lists, floats, None) are stored in Redis; a JSON round trip also turns
tuples into lists and non-string dict keys into strings. A function whose
result json.dumps rejects — e.g. `factor_data.fetch_fama_french_factors`,
which returns a DataFrame — still works and is still cached, but only in
the per-process `@st.cache_data` layer, never shared through Redis.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
from typing import Any, Callable, TypeVar

import streamlit as st

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

REDIS_URL = os.getenv("REDIS_URL")  # e.g. "redis://redis:6379/0" — unset means
# "don't use Redis", not "use it with defaults": an explicit opt-in, since a
# silently-assumed localhost Redis would fail differently (and more
# confusingly) in every environment that doesn't happen to have one running.

_redis_client: Any = None  # lazy singleton, same pattern as news_data.py's VADER analyzer
_redis_unavailable = False  # set once a connection attempt fails, so every
# subsequent call in this process skips straight to the direct-call fallback
# instead of retrying a dead connection on every single cache lookup.


def _get_redis_client() -> Any:
    global _redis_client, _redis_unavailable
    if _redis_client is None and not _redis_unavailable:
        try:
            import redis  # imported lazily — optional dependency, only needed if REDIS_URL is set
            assert REDIS_URL is not None  # guaranteed by cached()'s own early-return, not visible to mypy here
            client = redis.from_url(REDIS_URL, socket_connect_timeout=2, socket_timeout=2)
            client.ping()
            _redis_client = client
        except Exception as exc:
            logger.warning("Redis configured (REDIS_URL set) but unreachable, falling back to direct calls: %s", exc)
            _redis_unavailable = True
    return _redis_client


def get_redis_client() -> Any:
    """
    Public accessor for the SAME lazily-connected Redis singleton `cached()`
    itself uses — for callers that need direct Redis access (not a plain
    function-result cache — e.g. `rag.py` persisting a growable news corpus
    keyed by ticker set, not by a single function call's arguments) but must
    share the same connection/circuit-breaker state rather than duplicating
    it. Returns None if `REDIS_URL` isn't set or Redis isn't reachable — same
    fails-soft contract as `cached()` itself: a caller that gets None should
    degrade to its own no-persistence behaviour, never raise.
    """
    if not REDIS_URL:
        return None
    return _get_redis_client()


def _make_cache_key(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Stable key from function identity + arguments. Args/kwargs are JSON-
    encoded with sorted keys so equal calls always hash identically regardless
    of kwarg order; a hash (not the raw JSON) keeps keys short and avoids any
    issue with special characters in, say, a ticker list."""
    payload = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"cache:{func.__module__}.{func.__qualname__}:{digest}"


def cached(ttl_seconds: int) -> Callable[[F], F]:
    """Redis-backed cache when `REDIS_URL` is set and reachable; transparently
    falls back to `@st.cache_data` otherwise. See module docstring for the
    full rationale and the JSON-serialisability requirement.

    "Otherwise" covers every path where Redis can't serve the call: no
    `REDIS_URL`, Redis unreachable (sticky for the whole process once a
    connection attempt fails), a failed GET, and a result that isn't
    JSON-serialisable. Each of those goes through the same `@st.cache_data`
    wrapper rather than an uncached direct call — otherwise a Redis outage
    would leave every decorated fetcher uncached for the rest of the process
    and re-hit its rate-limited API on every Streamlit rerun."""
    if not REDIS_URL:
        return st.cache_data(show_spinner=False, ttl=ttl_seconds)  # type: ignore[return-value]

    def decorator(func: F) -> F:
        st_cached_func = st.cache_data(show_spinner=False, ttl=ttl_seconds)(func)
        # Set on the first result that json.dumps rejects (e.g. a DataFrame):
        # such a function can never use the Redis path, so later calls go
        # straight to the in-process cache instead of recomputing every time.
        json_unsupported = False

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal json_unsupported
            client = _get_redis_client()
            if client is None or json_unsupported:
                return st_cached_func(*args, **kwargs)

            key = _make_cache_key(func, args, kwargs)
            try:
                cached_value = client.get(key)
                if cached_value is not None:
                    return json.loads(cached_value)
            except Exception as exc:
                logger.warning("Redis GET failed for %s, using the in-process cache: %s", key, exc)
                return st_cached_func(*args, **kwargs)

            result = func(*args, **kwargs)
            try:
                serialised = json.dumps(result)
            except (TypeError, ValueError) as exc:
                json_unsupported = True
                logger.warning("Result of %s isn't JSON-serialisable, using the in-process cache: %s", key, exc)
                return result
            try:
                client.setex(key, ttl_seconds, serialised)
            except Exception as exc:  # a Redis write error is non-fatal: the result is still correct
                logger.warning("Redis SETEX failed for %s: %s", key, exc)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator