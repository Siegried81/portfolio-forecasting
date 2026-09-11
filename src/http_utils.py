"""
Shared "GET a URL, expect a JSON body, fail soft" helper — the one pattern
repeated identically across this codebase's simplest fetchers (a plain GET,
no auth header logic, no response-shape-specific error fields to check
beyond a raised HTTP status). Extracted here so those call sites share
exactly one implementation of that error handling instead of each having
their own copy of the same try/except.

Deliberately NOT applied to every fetcher in this codebase: several
(academic_search.py's Semantic Scholar/arXiv, news_data.py's GDELT/TED/SEC
EDGAR) have their own extra per-provider logic layered on top of the basic
GET-then-parse shape (custom headers, POST instead of GET, XML instead of
JSON, provider-specific error fields checked in the 200 response body) —
forcing those through one shared function would either lose that
provider-specific handling or turn this helper into a large parameter list
trying to cover every case. This helper is for the genuinely IDENTICAL
subset only (macro_data.py's two FRED fetchers today); a future fetcher that
truly matches this exact shape (plain GET, plain JSON, no extra checks) is a
good candidate to route through this too.
"""
from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


def safe_get_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = 10,
    context: str = "",
) -> dict[str, Any] | None:
    """
    GET `url`, return the parsed JSON body, or None on ANY failure (network
    error, non-2xx status, a body that isn't valid JSON) — never raises.
    `context` is an optional short label (e.g. a series ID or ticker)
    included in the warning log line, so a failure is traceable back to
    which call site hit it without needing a full traceback.
    """
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result
    except (requests.RequestException, ValueError) as exc:
        label = f" ({context})" if context else ""
        logger.warning("GET %s%s failed: %s", url, label, exc)
        return None