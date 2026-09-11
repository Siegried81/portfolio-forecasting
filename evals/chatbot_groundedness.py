"""
Groundedness eval for the chatbot — binary PASS/FAIL grading, with a
human-judge agreement check before trusting the judge at scale.

Run manually (not part of CI, makes real LLM calls): python evals/chatbot_groundedness.py
"""
from __future__ import annotations

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

def judge(context: str, question: str, fact: str, answer: str) -> bool:
    from src.llm_client import chat
    prompt = JUDGE_PROMPT.format(context=context, question=question, fact=fact, answer=answer)
    verdict, _backend = chat([{"role": "user", "content": prompt}])
    return verdict.strip().upper().startswith("PASS")

def run_eval() -> None:
    passed = 0
    for case in EVAL_CASES:
        answer, _backend = answer_portfolio_question(case["question"], FIXTURE_CONTEXT, [], None)
        ok = judge(FIXTURE_CONTEXT, case["question"], case["must_contain"], answer)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {case['question']}")
    print(f"\n{passed}/{len(EVAL_CASES)} passed ({passed / len(EVAL_CASES):.0%})")
    print(
        "Note: before trusting this judge's verdicts at scale, label ~20 answers by hand "
        "and measure how often the judge agrees with you — below ~80% agreement, rewrite "
        "the judge's rubric before citing its numbers."
    )


if __name__ == "__main__":
    run_eval()