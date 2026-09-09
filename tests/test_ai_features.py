"""
Unit tests for src/ai_features.py. Had NO test coverage before this pass.

Priority: `build_results_context` — the pure text-formatting function that is
the SAME grounding context fed to both the commentary generator and the
chatbot (per its own docstring, "one source of truth"). A formatting bug here
silently propagates into every LLM feature at once, so it's worth testing on
its own rather than only indirectly via a mocked chat() call. The LLM-calling
functions themselves are tested by mocking src.llm_client.chat — never a real
network call.
"""
import pandas as pd
import pytest

import src.ai_features as ai_features
from src.ai_features import build_results_context, generate_commentary


def _metrics(**overrides) -> dict:
    base = {
        "annual_return": 0.12, "annual_volatility": 0.18, "sharpe_ratio": 0.67,
        "sortino_ratio": 0.90, "max_drawdown": -0.22, "var_95": -0.03, "cvar_95": -0.05,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# build_results_context — pure formatting, shared by every LLM feature
# ---------------------------------------------------------------------------

def test_build_results_context_includes_only_nonzero_weights():
    weights = pd.Series({"AAPL": 0.60, "MSFT": 0.0005, "TSLA": 0.40})
    context = build_results_context(weights, _metrics(), None, None)
    assert "AAPL: 60.0%" in context
    assert "TSLA: 40.0%" in context
    assert "MSFT" not in context  # below the 0.1% display threshold


def test_build_results_context_includes_short_positions():
    # BUGFIX regression: `w > 0.001` used to silently drop every SHORT
    # (negative) weight when short selling is enabled, making the LLM
    # describe a short-selling portfolio as if it held fewer assets than
    # it actually does. Must show negative weights, not omit them.
    weights = pd.Series({"AAPL": 0.60, "TSLA": -0.20, "MSFT": 0.60})
    context = build_results_context(weights, _metrics(), None, None)
    assert "TSLA: -20.0%" in context


def test_build_results_context_omits_forecast_and_realized_sections_when_none():
    weights = pd.Series({"AAPL": 1.0})
    context = build_results_context(weights, _metrics(), None, None)
    assert "HISTORICAL-BASED PORTFOLIO" in context
    assert "FORECAST-BASED PORTFOLIO" not in context
    assert "REALIZED-OPTIMAL PORTFOLIO" not in context


def test_build_results_context_includes_all_three_sections_when_provided():
    weights = pd.Series({"AAPL": 1.0})
    context = build_results_context(weights, _metrics(), _metrics(), _metrics())
    assert "HISTORICAL-BASED PORTFOLIO" in context
    assert "FORECAST-BASED PORTFOLIO" in context
    assert "REALIZED-OPTIMAL PORTFOLIO" in context


def test_build_results_context_shows_period_return_only_when_present():
    weights = pd.Series({"AAPL": 1.0})
    with_period = build_results_context(weights, _metrics(period_return=0.05, n_periods=30), None, None)
    assert "Raw return over the 30-period window" in with_period
    without_period = build_results_context(weights, _metrics(), None, None)
    assert "Raw return over the" not in without_period


def test_build_results_context_flags_inverted_yield_curve():
    weights = pd.Series({"AAPL": 1.0})
    macro_context = {"macro": {"term_spread_10y_3m": -0.005}, "vix_level": None}
    context = build_results_context(weights, _metrics(), None, None, macro_context)
    assert "INVERTED" in context


def test_build_results_context_flags_sahm_rule_recession_signal():
    weights = pd.Series({"AAPL": 1.0})
    macro_context = {"macro": {"sahm_rule_indicator": 0.6}, "vix_level": None}
    context = build_results_context(weights, _metrics(), None, None, macro_context)
    assert "0.60" in context and "recession" in context


def test_build_results_context_omits_macro_section_when_nothing_came_back():
    weights = pd.Series({"AAPL": 1.0})
    # Every macro field None -> the block should never be appended (an empty
    # "MACRO & RISK BACKDROP" header with nothing under it would read as a
    # bug, not an absence of data).
    macro_context = {"macro": {}, "vix_level": None}
    context = build_results_context(weights, _metrics(), None, None, macro_context)
    assert "MACRO & RISK BACKDROP" not in context


def test_build_results_context_classifies_vix_regime():
    weights = pd.Series({"AAPL": 1.0})
    calm = build_results_context(weights, _metrics(), None, None, {"macro": {}, "vix_level": 12.0})
    elevated = build_results_context(weights, _metrics(), None, None, {"macro": {}, "vix_level": 30.0})
    assert "(calm)" in calm
    assert "(elevated)" in elevated


# ---------------------------------------------------------------------------
# generate_commentary — mocked chat(), never a real network call
# ---------------------------------------------------------------------------

def test_generate_commentary_passes_context_through_to_chat(monkeypatch):
    captured = {}

    def _fake_chat(messages, temperature, max_tokens):
        captured["messages"] = messages
        captured["temperature"] = temperature
        return "mocked commentary", "groq"

    monkeypatch.setattr(ai_features, "chat", _fake_chat)
    text, backend = generate_commentary("PORTFOLIO WEIGHTS...\nsome context")
    assert text == "mocked commentary"
    assert backend == "groq"
    assert "some context" in captured["messages"][-1]["content"]
    assert captured["messages"][0]["role"] == "system"

# ---------------------------------------------------------------------------
# answer_portfolio_question -- academic-literature grounding (2026-09-08)
# ---------------------------------------------------------------------------

def test_answer_portfolio_question_includes_academic_papers_for_a_methodology_question(monkeypatch):
    from src.ai_features import answer_portfolio_question

    captured = {}

    def _fake_chat(messages, temperature, max_tokens):
        captured["messages"] = messages
        return "mocked answer", "groq"

    def _fake_search(query, limit=3):
        captured["query"] = query
        return [{"title": "GARCH Models", "authors": ["Bollerslev"], "year": 1986, "url": "http://x",
                 "citation_count": 5000, "venue": "Journal of Econometrics"}]

    monkeypatch.setattr(ai_features, "chat", _fake_chat)
    monkeypatch.setattr("src.academic_search.search_academic_papers", _fake_search)

    answer_portfolio_question("Why did you use GARCH for volatility?", "some results context", [])

    system_content = captured["messages"][0]["content"]
    assert "GARCH Models" in system_content
    assert "Bollerslev" in system_content
    assert "ACADEMIC REFERENCES" in system_content


def test_answer_portfolio_question_skips_academic_search_for_unrelated_question(monkeypatch):
    from src.ai_features import answer_portfolio_question

    def _fake_chat(messages, temperature, max_tokens):
        return "mocked answer", "groq"

    def _fail_if_called(query, limit=3):
        pytest.fail("search_academic_papers should not be called for a non-methodology question")

    monkeypatch.setattr(ai_features, "chat", _fake_chat)
    monkeypatch.setattr("src.academic_search.search_academic_papers", _fail_if_called)

    # Must not raise (confirms search_academic_papers is never invoked below)
    answer_portfolio_question("what is my portfolio's current return?", "some results context", [])


# ---------------------------------------------------------------------------
# Prompt-structure improvements (2026-09-08): XML tags around injected data,
# explicit output-format instruction, few-shot example in SYSTEM_PERSONA
# ---------------------------------------------------------------------------

def test_wrap_context_produces_a_named_xml_tag():
    from src.ai_features import _wrap_context
    wrapped = _wrap_context("portfolio_data", "some content here")
    assert wrapped == "<portfolio_data>\nsome content here\n</portfolio_data>"


def test_generate_commentary_wraps_results_context_in_portfolio_data_tag(monkeypatch):
    captured = {}

    def _fake_chat(messages, temperature, max_tokens):
        captured["messages"] = messages
        return "mocked commentary", "groq"

    monkeypatch.setattr(ai_features, "chat", _fake_chat)
    generate_commentary("some context")
    content = captured["messages"][-1]["content"]
    assert "<portfolio_data>" in content
    assert "some context" in content
    assert "</portfolio_data>" in content


def test_generate_commentary_specifies_prose_format_not_bullets(monkeypatch):
    captured = {}

    def _fake_chat(messages, temperature, max_tokens):
        captured["messages"] = messages
        return "mocked commentary", "groq"

    monkeypatch.setattr(ai_features, "chat", _fake_chat)
    generate_commentary("some context")
    content = captured["messages"][-1]["content"]
    assert "no bullet points" in content
    assert "no markdown headers" in content


def test_system_persona_includes_a_good_and_bad_annualisation_example():
    # Regression test for the few-shot addition: the rule must not be prose-only
    # any more — a concrete GOOD/BAD pair reinforces it far more reliably across
    # the 5 different LLM providers this app can now route to.
    assert "GOOD:" in ai_features.SYSTEM_PERSONA
    assert "BAD (never write this):" in ai_features.SYSTEM_PERSONA


def test_answer_portfolio_question_wraps_every_data_block_in_its_own_tag(monkeypatch):
    from src.ai_features import answer_portfolio_question
    from src.rag import Chunk

    captured = {}

    def _fake_chat(messages, temperature, max_tokens):
        captured["content"] = messages[0]["content"]
        return "mocked answer", "groq"

    def _fake_search(query, limit=3):
        return [{"title": "GARCH Models", "authors": ["Bollerslev"], "year": 1986, "url": "http://x",
                 "citation_count": 5000, "venue": "Journal of Econometrics"}]

    monkeypatch.setattr(ai_features, "chat", _fake_chat)
    monkeypatch.setattr("src.academic_search.search_academic_papers", _fake_search)

    news_chunks = [Chunk(text="Some headline about GARCH", source="Reuters", provider="NewsAPI", ticker="AAPL", url="http://x")]
    answer_portfolio_question("Why did you use GARCH for volatility?", "some results", [], news_chunks)

    content = captured["content"]
    assert "<portfolio_data>" in content and "</portfolio_data>" in content
    assert "<retrieved_news>" in content and "</retrieved_news>" in content
    assert "<academic_references>" in content and "</academic_references>" in content