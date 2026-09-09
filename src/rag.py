"""
Lightweight retrieval-augmented generation (RAG) over the news/filings corpus
collected in the AI Analyst tab — a genuine index-then-retrieve pipeline, kept
deliberately separate from the context-injection approach used for portfolio
metrics elsewhere in `ai_features.py` (that distinction is the point: metrics
are a small, fixed, MUST-be-complete set of numbers where full injection is
correct; news/filings are unstructured, can grow arbitrarily large, and only
the query-relevant subset should reach the prompt — that's what RAG is for).

Why TF-IDF instead of neural embeddings: this corpus is small (a handful of
headlines/filings per ticker set) and the queries are the kind of keyword-
driven questions a user actually types in a chat box ("what's the news on
NVDA earnings"). A sentence-transformers model would add a large, slow-to-
install dependency and a multi-hundred-MB model download for retrieval
quality gains that don't materially matter at this scale. TF-IDF + cosine
similarity is deterministic, fast, dependency-light (scikit-learn is already
pulled in transitively by PyPortfolioOpt's `cvxpy` stack), and sufficient here.

Persistence: opt-in via the SAME Redis instance
`src/cache.py`'s `cached()` decorator already uses — with no `REDIS_URL` set
(every local/free-tier deployment today), `save_chunks`/`load_chunks` are
silent no-ops and behaviour is EXACTLY what it was before this existed: a
corpus rebuilt fresh every session. When `REDIS_URL` IS set, a fetched corpus
survives the session ending, a container restart, or a second user asking
about the same ticker set. Still TF-IDF, not a real vector store: if this
corpus grew into hundreds of PERMANENTLY accumulated documents (rather than
today's "one fetch's worth, refreshed"), a real vector store (Chroma/FAISS)
with neural embeddings would be the right upgrade — noted in the README as
the next step beyond this one, not built here because it would be
unjustified complexity for what's still a "handful of headlines" corpus size.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.cache import get_redis_client
from src.config import RAG_CHUNK_PERSISTENCE_TTL_SECONDS

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """One retrievable unit: a news headline/summary or a filing entry."""
    text: str
    source: str    # display name, e.g. "Reuters", "Bloomberg", "SEC EDGAR"
    provider: str  # which fetcher produced it: "NewsAPI" / "Finnhub" / "SEC EDGAR"
    ticker: str
    url: str


def build_chunks(articles: list[dict[str, Any]]) -> list[Chunk]:
    """
    Turn the raw article/filing dicts already collected by news_data.py (via
    ai_features.generate_news_digest) into retrievable chunks. Title + summary
    are concatenated into one chunk per article — granular enough for a small
    corpus like this without the overhead of further splitting.
    """
    return [
        Chunk(
            text=f"{a.get('title', '')}. {a.get('description', '')}".strip(),
            source=a.get("source", "unknown"),
            provider=a.get("provider", "unknown"),
            ticker=a.get("ticker", ""),
            url=a.get("url", ""),
        )
        for a in articles
        if a.get("title")
    ]


def _persistence_key(tickers: list[str]) -> str:
    """Stable Redis key for one ticker universe's corpus — sorted and
    uppercased so the same set of tickers in a different order or case
    (e.g. user picked them in a different sequence) still hits the same key."""
    return "rag:chunks:" + ",".join(sorted(t.upper() for t in tickers))


def save_chunks(chunks: list[Chunk], tickers: list[str]) -> None:
    """
    Persist a freshly-built corpus to Redis, if configured — see this
    module's docstring for the full opt-in/fails-soft contract. A no-op if
    Redis isn't configured/reachable (`get_redis_client()` returns None) or
    `chunks` is empty (nothing worth persisting, and an empty corpus
    overwriting a real one on a transient empty fetch would be a regression,
    not a cache update).

    Fails soft on any Redis error — persistence is an optimisation, not
    something a broken Redis connection should be able to take the news
    digest down over; the caller's own `chunks` are still usable regardless.
    """
    client = get_redis_client()
    if client is None or not chunks:
        return
    key = _persistence_key(tickers)
    try:
        payload = json.dumps([dataclasses.asdict(c) for c in chunks])
        client.setex(key, RAG_CHUNK_PERSISTENCE_TTL_SECONDS, payload)
    except Exception as exc:  # noqa: BLE001 — any Redis/serialisation error is non-fatal here
        logger.warning("RAG chunk persistence failed for %s: %s", key, exc)


def load_chunks(tickers: list[str]) -> list[Chunk]:
    """
    Load a previously-persisted corpus for this exact ticker set, if Redis is
    configured and has one. Returns [] (never raises) if Redis isn't
    configured, has nothing under this key (first fetch ever, or the TTL
    expired), or the read/parse itself fails — callers should treat [] exactly
    like "no news fetched yet," the same state as before this feature existed.
    """
    client = get_redis_client()
    if client is None:
        return []
    key = _persistence_key(tickers)
    try:
        raw = client.get(key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG chunk load failed for %s: %s", key, exc)
        return []
    if raw is None:
        return []
    try:
        payload = json.loads(raw)
        return [Chunk(**item) for item in payload]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("RAG chunk payload malformed for %s, ignoring: %s", key, exc)
        return []


def retrieve(query: str, chunks: list[Chunk], top_k: int = 4) -> list[Chunk]:
    """
    Return up to `top_k` chunks most relevant to `query` by TF-IDF cosine
    similarity, filtering out zero-similarity matches (a chunk sharing no
    vocabulary with the query is not relevant, regardless of rank). Returns []
    if there's nothing to retrieve from, or if the query/corpus share no
    vocabulary at all — callers should treat that as "no relevant news found"
    and fall back gracefully, never crash on it.
    """
    if not chunks or not query.strip():
        return []

    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(texts + [query])
    except ValueError:
        # e.g. every chunk + the query reduce to nothing but stopwords after
        # cleaning — genuinely nothing to retrieve on, not a bug to raise on.
        return []

    query_vector = matrix[-1]
    doc_vectors = matrix[:-1]
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()

    ranked_indices = sorted(range(len(chunks)), key=lambda i: similarities[i], reverse=True)
    return [chunks[i] for i in ranked_indices[:top_k] if similarities[i] > 0]


def format_retrieved_chunks(chunks: list[Chunk]) -> str:
    """Render retrieved chunks into a text block for prompt injection, tagged
    by provider/source so the LLM (and the README's cross-referencing framing)
    can distinguish primary filings from media coverage."""
    if not chunks:
        return ""
    lines = [f"  - [{c.provider}/{c.source}, {c.ticker}] {c.text}" for c in chunks]
    return "RETRIEVED NEWS/FILINGS CONTEXT (top matches for this question):\n" + "\n".join(lines)