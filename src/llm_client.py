"""
Unified LLM client: Groq (hosted, fast, primary; multi-key rotation) falls
back through an ORDERED LIST of hosted, OpenAI-compatible providers
(OpenRouter, Cerebras, SambaNova — whichever have a key configured, in that
order), then finally to a local Ollama instance if every hosted provider
fails.

Why a list instead of one named fallback: OpenRouter, Cerebras, and SambaNova
all speak the IDENTICAL wire protocol (Bearer auth, POST
{base_url}/chat/completions, {"model", "messages", ...} body,
response["choices"][0]["message"]["content"]) — the only thing that differs
between them is base_url/key/model. Writing a separate `_call_cerebras`,
`_call_sambanova` near-copy of `_call_openrouter` for each new provider would
duplicate the exact same code three times; `_call_openai_compatible_provider`
below is that shared implementation, and `_fallback_providers()` is the one
place that lists which of them are actually configured, in what order. Adding
a fifth provider later is a two-line change to `_fallback_providers()`, not a
new function.

Groq is kept as its own special case (5-key rate-limit rotation is genuinely
different behaviour, not just a different base_url) and so is Ollama (a
different endpoint shape: `/api/chat`, not `/chat/completions`, and a
different response envelope).

Why more than one hosted fallback tier at all: Ollama alone left a real gap —
it only runs on whatever machine has it installed, a genuine fallback in
local dev but silently unreachable once this app is deployed (Render,
Streamlit Community Cloud have no Ollama daemon in the container). The three
hosted fallbacks are also genuine redundancy against EACH OTHER, not just
against Ollama: OpenRouter, Cerebras, and SambaNova are independent
companies/accounts/infrastructure (Cerebras and SambaNova are, like Groq,
dedicated fast-inference hardware providers — the same competitive niche,
not a random unrelated pick), so an outage or rate limit specific to one
provider's account doesn't take out the others.

This is the ONLY place in the codebase that talks to an LLM provider — every
other module calls `chat()` and doesn't know or care which backend answered.
That's what makes the fallback chain possible without duplicating prompt
logic, and it's what "LLM-friendly code" means in practice: one seam to swap
providers, mock in tests, or add another backend later.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import requests
import tiktoken

from src.config import LLM_SETTINGS, MAX_CONTEXT_TOKENS

logger = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


class LLMUnavailableError(RuntimeError):
    """Raised only if EVERY configured backend fails (Groq, every hosted
    fallback provider that has a key set, and Ollama) — lets the UI show one
    clear message instead of a stack trace, without silently pretending
    everything is fine."""


def _count_tokens(text: str) -> int:
    """Approximate token count using OpenAI's cl100k_base encoding — close enough
    across providers for the purpose of staying under a context budget (we don't
    need exact provider-specific tokenisation here, just a safety margin)."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4  # crude fallback if tiktoken's encoding download fails offline


def truncate_to_token_budget(text: str, max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """Hard-truncate free-text context (news articles, etc.) to a token budget
    before it's stuffed into a prompt, so a chatty NewsAPI response never blows
    past a small local model's context window.

    The truncation path uses the SAME char-count fallback as `_count_tokens`
    (~4 chars/token) if `tiktoken.get_encoding()` can't be reached — e.g. an
    offline dev box or a restricted container network policy blocking the
    encoding download. Without this, a blocked download would raise straight
    through as an uncaught HTTPError instead of degrading, breaking this
    module's own documented "fails soft" contract: a network hiccup here
    should never crash a feature that's meant to be enrichment, not core to
    the app.
    """
    if _count_tokens(text) <= max_tokens:
        return text
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)[:max_tokens]
        return encoding.decode(tokens) + "\n[...truncated...]"
    except Exception:
        return text[: max_tokens * 4] + "\n[...truncated...]"


def _call_groq(messages: list[Message], temperature: float, max_tokens: int) -> str:
    if not LLM_SETTINGS.groq_api_keys:
        raise LLMUnavailableError("No GROQ_API_KEY configured.")
    from groq import Groq, RateLimitError  # imported lazily — package is optional if only Ollama is used

    last_error: Exception | None = None
    for i, key in enumerate(LLM_SETTINGS.groq_api_keys):
        try:
            client = Groq(api_key=key)
            # Our `Message` alias (plain {"role", "content"} dicts) is deliberately
            # provider-agnostic so this module stays the one seam every caller goes
            # through regardless of backend. Groq's SDK wants its own stricter
            # per-role TypedDict union; the shapes match at runtime (Groq's API is
            # the standard OpenAI-style chat schema), so adopting its exact type
            # here would leak Groq-specific typing back into that seam.
            response = client.chat.completions.create(
                model=LLM_SETTINGS.groq_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except RateLimitError as exc:
            # This specific key is exhausted (429) — rotate to the next one, if any.
            logger.warning("Groq key #%d rate-limited, trying next key: %s", i + 1, exc)
            last_error = exc
            continue
        except Exception:
            # Any OTHER error (bad key, model deprecated, network) affects every key
            # equally — rotating won't help, so fail fast to the next fallback tier
            # instead of burning time cycling through all 5 keys for nothing.
            raise
    raise LLMUnavailableError(f"All {len(LLM_SETTINGS.groq_api_keys)} Groq keys are rate-limited: {last_error}")


@dataclass(frozen=True)
class _HostedProvider:
    """One hosted, OpenAI-compatible chat-completions backend — everything
    OpenRouter/Cerebras/SambaNova have in common. `label` is what shows up in
    the UI's "Generated by: {label} (fallback)" caption."""
    label: str
    api_key: str
    base_url: str
    model: str


def _fallback_providers() -> list[_HostedProvider]:
    """
    Ordered list of hosted fallback providers tried AFTER Groq, before
    finally falling back to local Ollama. A provider with no API key
    configured is skipped entirely — not even attempted — so setting only
    `GROQ_API_KEY` (the original setup) keeps that exact 2-tier behaviour
    unchanged; each additional key configured (`OPENROUTER_API_KEY`,
    `CEREBRAS_API_KEY`, `SAMBANOVA_API_KEY`) adds one more tier of
    redundancy with zero code changes needed.
    """
    candidates = [
        _HostedProvider("openrouter", LLM_SETTINGS.openrouter_api_key or "", "https://openrouter.ai/api/v1", LLM_SETTINGS.openrouter_model),
        _HostedProvider("cerebras", LLM_SETTINGS.cerebras_api_key or "", "https://api.cerebras.ai/v1", LLM_SETTINGS.cerebras_model),
        _HostedProvider("sambanova", LLM_SETTINGS.sambanova_api_key or "", "https://api.sambanova.ai/v1", LLM_SETTINGS.sambanova_model),
    ]
    return [p for p in candidates if p.api_key]


def _call_openai_compatible_provider(
    provider: _HostedProvider, messages: list[Message], temperature: float, max_tokens: int,
) -> str:
    """
    Shared implementation for every hosted, OpenAI-compatible fallback
    provider (OpenRouter, Cerebras, SambaNova) — see this module's docstring
    for why one shared function instead of one copy per provider. Plain
    `requests`, no extra SDK per provider (same reasoning as `_call_ollama`
    below): this app's `Message` shape already matches the standard
    OpenAI-style chat schema every one of these providers accepts as-is.
    """
    response = requests.post(
        f"{provider.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {provider.api_key}"},
        json={
            "model": provider.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload["choices"][0]["message"]["content"] or "")


def _call_ollama(messages: list[Message], temperature: float, max_tokens: int) -> str:
    response = requests.post(
        f"{LLM_SETTINGS.ollama_host}/api/chat",
        json={
            "model": LLM_SETTINGS.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=60,
    )
    response.raise_for_status()
    return str(response.json().get("message", {}).get("content", ""))


def chat(messages: list[Message], temperature: float = 0.3, max_tokens: int = 600) -> tuple[str, str]:
    """
    Send a chat completion request. Tries Groq first; on ANY failure (missing
    key, network error, rate limit, model deprecation) falls through the
    configured hosted fallback providers in order (see `_fallback_providers`
    — OpenRouter, then Cerebras, then SambaNova, skipping any without a key
    set), then finally to a local Ollama instance if every hosted provider
    fails. Returns (response_text, backend_used) so the UI can be transparent
    about which model actually answered — useful both for debugging and for
    honesty with the end user about provenance.

    Raises LLMUnavailableError only if Groq, every configured hosted
    fallback, AND Ollama all fail.
    """
    errors: dict[str, str] = {}

    try:
        return _call_groq(messages, temperature, max_tokens), "groq"
    except Exception as groq_error:
        errors["groq"] = str(groq_error)
        logger.warning("Groq call failed, trying hosted fallback providers: %s", groq_error)

    for provider in _fallback_providers():
        try:
            result = _call_openai_compatible_provider(provider, messages, temperature, max_tokens)
            return result, f"{provider.label} (fallback)"
        except Exception as exc:
            errors[provider.label] = str(exc)
            logger.warning("%s call failed, trying next fallback: %s", provider.label, exc)

    try:
        return _call_ollama(messages, temperature, max_tokens), "ollama (local fallback)"
    except Exception as ollama_error:
        errors["ollama"] = str(ollama_error)
        raise LLMUnavailableError(f"All LLM backends failed: {errors}") from ollama_error