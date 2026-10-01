"""
Groundedness eval for the chatbot — binary PASS/FAIL grading by an LLM
judge. The human-judge agreement check that should precede trusting the
judge at scale is a MANUAL step: this script only prints a reminder of it,
it does not measure agreement itself.

Run manually (not part of CI, makes real LLM calls): python evals/chatbot_groundedness.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai_features import answer_portfolio_question

FIXTURE_CONTEXT = (
    "Historical-based Sharpe: 0.52. Historical-based Sortino: -0.68. "
    "Forecast-based Sharpe: 0.36. Forecast-based Sortino: -2.82. "
    "Realized-optimal Sharpe: 9.10. "
    "Risk-free rate: 3.91%. Comparison window: 29 periods. "
    "Sharpe SE (historical): 2.95. "
    "Long positions: AAPL 25%, AVGO 25%, CSCO 25%, IBM 25%, MSFT 25%, NVDA 25%, "
    "PANW 25%, TXN 3.3%, AMD 5.8%. "
    "Short positions: ADBE -25%, QCOM -25%, NOW -20.4%, CRM -6.8%, ORCL -4.0%, INTC -2.8%."
)

# Each case: a question, and the one fact the answer must contain.
EVAL_CASES = [
    {"question": "What is the historical Sharpe ratio?", "must_contain": "0.52"},
    {"question": "What is the forecast-based Sharpe ratio?", "must_contain": "0.36"},
    {"question": "What is the risk-free rate?", "must_contain": "3.91"},
    {"question": "What time window is this comparison using?", "must_contain": "29"},
    {"question": "What is the historical portfolio's Sortino ratio?", "must_contain": "-0.68"},
    {"question": "What is the forecast-based portfolio's Sortino ratio?", "must_contain": "-2.82"},
    {"question": "What is AMD's weight in the optimal portfolio?", "must_contain": "5.8"},
    {"question": "Is this an equal-weight portfolio?", "must_contain": "No"},
    {"question": "What is the historical portfolio's Sharpe standard error?", "must_contain": "2.95"},
    {"question": "What is the realized-optimal Sharpe ratio?", "must_contain": "9.10"},
]

JUDGE_PROMPT = """You grade whether an assistant's answer is grounded in the context.
Reply with exactly PASS or FAIL on the first line, then one short reason.
Context: {context}
Question: {question}
Required fact: {fact}
Assistant answer: {answer}
PASS only if the answer states the required fact correctly and does not
invent numbers absent from the context."""

def parse_verdict(verdict: str) -> bool:
    """True only if the judge's reply opens with the word PASS, ignoring
    leading markdown decoration (`**PASS**`, `# PASS`). Many models bold their
    verdict despite the prompt; a plain startswith("PASS") graded those as
    FAIL and understated the pass rate. A reply opening with anything else
    (FAIL, prose, empty) still counts as FAIL."""
    match = re.match(r"[\W_]*(PASS|FAIL)\b", verdict.strip(), flags=re.IGNORECASE)
    return bool(match) and match.group(1).upper() == "PASS"


def judge(context: str, question: str, fact: str, answer: str) -> tuple[bool, str]:
    """Ask the LLM judge for a verdict on one answer. Returns (passed,
    backend) so the report shows which model did the grading."""
    from src.llm_client import chat
    prompt = JUDGE_PROMPT.format(context=context, question=question, fact=fact, answer=answer)
    verdict, backend = chat([{"role": "user", "content": prompt}])
    return parse_verdict(verdict), backend

def run_eval() -> None:
    passed = 0
    for case in EVAL_CASES:
        answer, answer_backend = answer_portfolio_question(case["question"], FIXTURE_CONTEXT, [], None)
        ok, judge_backend = judge(FIXTURE_CONTEXT, case["question"], case["must_contain"], answer)
        passed += ok
        # chat() silently falls back across providers, so one run can grade
        # answers from several different models; the backends make that visible.
        print(f"[{'PASS' if ok else 'FAIL'}] {case['question']} (answer: {answer_backend}, judge: {judge_backend})")
    print(f"\n{passed}/{len(EVAL_CASES)} passed ({passed / len(EVAL_CASES):.0%})")
    print(
        "Note: before trusting this judge's verdicts at scale, label ~20 answers by hand "
        "and measure how often the judge agrees with you — below ~80% agreement, rewrite "
        "the judge's rubric before citing its numbers."
    )


if __name__ == "__main__":
    run_eval()