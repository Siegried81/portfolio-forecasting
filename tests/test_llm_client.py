"""Unit tests for src/llm_client.py — priority on the Groq -> hosted fallbacks -> Ollama cascade."""
import dataclasses

import pytest

import src.llm_client as llm_client
from src.llm_client import LLMUnavailableError, _HostedProvider, chat, truncate_to_token_budget


def _patch_settings(monkeypatch, **overrides):
    fake = dataclasses.replace(llm_client.LLM_SETTINGS, **overrides)
    monkeypatch.setattr(llm_client, "LLM_SETTINGS", fake)


def _no_hosted_fallbacks(monkeypatch):
    _patch_settings(monkeypatch, openrouter_api_key=None, cerebras_api_key=None, sambanova_api_key=None)


# ---------------------------------------------------------------------------
# chat() cascade
# ---------------------------------------------------------------------------

def test_chat_returns_groq_result_and_backend_label_when_groq_succeeds(monkeypatch):
    _no_hosted_fallbacks(monkeypatch)
    monkeypatch.setattr(llm_client, "_call_groq", lambda messages, temperature, max_tokens: "groq answer")
    monkeypatch.setattr(
        llm_client, "_call_openai_compatible_provider",
        lambda *a, **k: pytest.fail("no hosted fallback should be called"),
    )
    monkeypatch.setattr(llm_client, "_call_ollama", lambda *a, **k: pytest.fail("Ollama should not be called"))
    text, backend = chat([{"role": "user", "content": "hi"}])
    assert text == "groq answer"
    assert backend == "groq"


def test_chat_falls_back_to_first_configured_hosted_provider_when_groq_fails(monkeypatch):
    _patch_settings(monkeypatch, openrouter_api_key="fake-key", cerebras_api_key=None, sambanova_api_key=None)
    monkeypatch.setattr(llm_client, "_call_groq", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Groq down")))
    monkeypatch.setattr(
        llm_client, "_call_openai_compatible_provider",
        lambda provider, messages, temperature, max_tokens: f"{provider.label} answer",
    )
    monkeypatch.setattr(llm_client, "_call_ollama", lambda *a, **k: pytest.fail("Ollama should not be called"))
    text, backend = chat([{"role": "user", "content": "hi"}])
    assert text == "openrouter answer"
    assert backend == "openrouter (fallback)"


def test_chat_tries_hosted_fallbacks_in_order_and_stops_at_first_success(monkeypatch):
    _patch_settings(monkeypatch, openrouter_api_key="or-key", cerebras_api_key="cb-key", sambanova_api_key=None)
    monkeypatch.setattr(llm_client, "_call_groq", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Groq down")))

    def _fake_provider_call(provider, messages, temperature, max_tokens):
        if provider.label == "openrouter":
            raise RuntimeError("OpenRouter down")
        return f"{provider.label} answer"

    monkeypatch.setattr(llm_client, "_call_openai_compatible_provider", _fake_provider_call)
    monkeypatch.setattr(llm_client, "_call_ollama", lambda *a, **k: pytest.fail("Ollama should not be called"))

    text, backend = chat([{"role": "user", "content": "hi"}])
    assert text == "cerebras answer"
    assert backend == "cerebras (fallback)"


def test_chat_falls_back_to_ollama_when_groq_and_every_hosted_provider_fail(monkeypatch):
    _patch_settings(monkeypatch, openrouter_api_key="or-key", cerebras_api_key=None, sambanova_api_key=None)
    monkeypatch.setattr(llm_client, "_call_groq", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Groq down")))
    monkeypatch.setattr(
        llm_client, "_call_openai_compatible_provider",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("OpenRouter down")),
    )
    monkeypatch.setattr(llm_client, "_call_ollama", lambda messages, temperature, max_tokens: "ollama answer")
    text, backend = chat([{"role": "user", "content": "hi"}])
    assert text == "ollama answer"
    assert backend == "ollama (local fallback)"


def test_chat_raises_llm_unavailable_when_absolutely_everything_fails(monkeypatch):
    _no_hosted_fallbacks(monkeypatch)
    monkeypatch.setattr(llm_client, "_call_groq", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Groq down")))
    monkeypatch.setattr(llm_client, "_call_ollama", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Ollama down")))
    with pytest.raises(LLMUnavailableError, match="All LLM backends failed"):
        chat([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# _call_groq
# ---------------------------------------------------------------------------

def test_call_groq_raises_llm_unavailable_with_no_keys_configured(monkeypatch):
    _patch_settings(monkeypatch, groq_api_keys=[])
    with pytest.raises(LLMUnavailableError, match="No GROQ_API_KEY"):
        llm_client._call_groq([{"role": "user", "content": "hi"}], temperature=0.3, max_tokens=100)


# ---------------------------------------------------------------------------
# _fallback_providers
# ---------------------------------------------------------------------------

def test_fallback_providers_empty_when_no_keys_configured(monkeypatch):
    _no_hosted_fallbacks(monkeypatch)
    assert llm_client._fallback_providers() == []


def test_fallback_providers_includes_only_configured_ones_in_fixed_order(monkeypatch):
    _patch_settings(monkeypatch, openrouter_api_key=None, cerebras_api_key=None, sambanova_api_key="sn-key")
    labels = [p.label for p in llm_client._fallback_providers()]
    assert labels == ["sambanova"]


def test_fallback_providers_order_is_openrouter_then_cerebras_then_sambanova(monkeypatch):
    _patch_settings(
        monkeypatch, openrouter_api_key="or-key", cerebras_api_key="cb-key", sambanova_api_key="sn-key",
    )
    labels = [p.label for p in llm_client._fallback_providers()]
    assert labels == ["openrouter", "cerebras", "sambanova"]


# ---------------------------------------------------------------------------
# _call_openai_compatible_provider
# ---------------------------------------------------------------------------

def test_call_openai_compatible_provider_sends_bearer_auth_to_the_right_url(monkeypatch):
    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "provider says hi"}}]}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResponse()

    monkeypatch.setattr(llm_client.requests, "post", _fake_post)
    provider = _HostedProvider("cerebras", "fake-cerebras-key", "https://api.cerebras.ai/v1", "llama-3.3-70b")
    result = llm_client._call_openai_compatible_provider(
        provider, [{"role": "user", "content": "hi"}], temperature=0.3, max_tokens=100,
    )

    assert result == "provider says hi"
    assert captured["url"] == "https://api.cerebras.ai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer fake-cerebras-key"
    assert captured["json"]["model"] == "llama-3.3-70b"


def test_call_openai_compatible_provider_raises_on_http_error(monkeypatch):
    import requests as requests_module

    class _FakeResponse:
        def raise_for_status(self):
            raise requests_module.HTTPError("401 Unauthorized")

    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: _FakeResponse())
    provider = _HostedProvider("sambanova", "fake-key", "https://api.sambanova.ai/v1", "Meta-Llama-3.3-70B-Instruct")
    with pytest.raises(requests_module.HTTPError):
        llm_client._call_openai_compatible_provider(
            provider, [{"role": "user", "content": "hi"}], temperature=0.3, max_tokens=100,
        )


# ---------------------------------------------------------------------------
# truncate_to_token_budget / _count_tokens
# ---------------------------------------------------------------------------

def test_truncate_to_token_budget_leaves_short_text_unchanged():
    short_text = "This is a short sentence."
    assert truncate_to_token_budget(short_text, max_tokens=1000) == short_text


def test_truncate_to_token_budget_shortens_long_text_and_marks_it():
    long_text = "word " * 5000
    truncated = truncate_to_token_budget(long_text, max_tokens=50)
    assert len(truncated) < len(long_text)
    assert truncated.endswith("[...truncated...]")


def test_count_tokens_falls_back_to_char_heuristic_when_tiktoken_unavailable(monkeypatch):
    def _broken_get_encoding(name):
        raise RuntimeError("no network access to fetch encoding")
    monkeypatch.setattr(llm_client.tiktoken, "get_encoding", _broken_get_encoding)
    assert llm_client._count_tokens("a" * 40) == 10


def test_truncate_to_token_budget_never_raises_when_tiktoken_unavailable(monkeypatch):
    def _broken_get_encoding(name):
        raise RuntimeError("no network access to fetch encoding")
    monkeypatch.setattr(llm_client.tiktoken, "get_encoding", _broken_get_encoding)
    assert truncate_to_token_budget("short text", max_tokens=1000) == "short text"


def test_truncate_to_token_budget_degrades_to_char_truncation_when_encoding_unavailable(monkeypatch):
    def _broken_get_encoding(name):
        raise RuntimeError("no network access to fetch encoding")
    monkeypatch.setattr(llm_client.tiktoken, "get_encoding", _broken_get_encoding)
    long_text = "word " * 5000
    result = truncate_to_token_budget(long_text, max_tokens=50)
    assert result.endswith("[...truncated...]")
    assert len(result) < len(long_text)