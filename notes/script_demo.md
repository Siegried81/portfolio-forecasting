# Live demo script — 15 minutes (English)

Mirrors `script_demo.md` (French) — same structure, same timings, same numbers. Built off
`demo.pptx` (10 slides). Time it with a stopwatch, not by feel — 15 minutes go faster than
expected once you're also juggling the live app.

---

## 0:00 – 1:00 — Intro (slides 1-2)

*Slide 1 (title)*: introduce yourself — name, BeCode bootcamp, GenAI Developer track, and one
sentence on your background before BeCode if that helps the jury place you.

*Slide 2 (the brief)*: "I was given a second-round-interview-style brief: build a portfolio
optimization app in 5 days that an interviewer could actually use. Three things were asked —
optimize, forecast, compare. I'll show you all three, plus everything I added beyond that,
and how I worked with an AI to ship this in 5 days without cutting corners on rigor."

Don't read the on-screen bullets word for word — the slide is a visual anchor, not a
teleprompter.

## 1:00 – 2:00 — Technical choices (slide 3)

"Four layers: yfinance for data, with fallback if Yahoo goes down; PyPortfolioOpt for
optimization; statsmodels for forecasting — chosen over Kats, which has been unmaintained
since 2021; and an AI layer with Groq as primary, several hosted fallbacks, and a local
Ollama fallback if everything else fails."

One sentence per layer — save the longer justification for questions afterward (see the
safety net at the bottom of this document).

## 2:00 – 2:15 — Transition (slide 4)

"I'll switch to the live app now." Don't comment on the slide further — it's just a visual
anchor while you switch screens.

## 2:15 – 11:00 — LIVE DEMO in the app (8 min 45)

Tighter budget than the old 5-minute version, but wider — ten stops, all timed. Don't
linger on any tab past its budget, even if the jury asks a question — note it and answer
after closing.

1. **Overview → Macro & Risk** (30s) — point at the live risk-free rate via FRED, the VIX,
   the 10Y-3M spread, the Sahm Rule. "None of this is hardcoded — it's all live from FRED,
   including the risk-free rate that pre-fills the sidebar slider."
2. **Overview → Fundamentals** (30s) — click "Fetch fundamentals", show the P/E, market cap,
   beta, dividend yield table. "Finnhub is the primary source, with a genuine field-by-field
   fallback to Twelve Data when one specific number is missing — not all-or-nothing."
3. **Overview → Time-series diagnostics** (30s) — the ADF test and Hurst exponent per
   ticker. "This is the kind of check an economist runs before trusting a forecast, not
   after — and it confirms these series sit close to a random walk, which frames everything
   you're about to see in the next couple of minutes."
4. **Efficient Frontier** (45s) — show the scatter cloud, the frontier line, the max-Sharpe
   star. Open the optimal weights table. Mention the diversification ratio and concentration
   (HHI) in one sentence each.
5. **Efficient Frontier → Factor exposure** (20s) — the Fama-French expander. "This answers
   a different question than the frontier — not just 'how much total risk', but 'how much of
   that risk comes from the market, from size, from value'."
6. **Forecast & Compare** (60s) — THE heart of the demo, never cut this short. Show the
   Historical / Forecast-based / Realized-optimal table side by side. Say explicitly:
   "Realized-optimal is deliberate cheating — it's the optimum built from the actual future
   prices, known only after the fact. It's a benchmark, never a reachable target."
7. **Forecast & Compare → Fan chart** (30s) — briefly switch the model to ARIMA to show why
   the confidence band widens with horizon. "A single point forecast hides how much
   uncertainty compounds — the band doesn't hide it."
8. **Walk-forward validation** (45s) — the multi-window box plot. "One test window can be
   luck. This repeats it across 6 expanding windows — forecast-based beats historical on
   Sharpe in 83% of the windows tested."
9. **AI Analyst** (45s) — click "Generate commentary", show the news digest with clickable
   sources and per-ticker sentiment (FinBERT → Finnhub → VADER). "Every claim the AI makes is
   grounded in a number computed above, or a real cited source — never invented."
10. **Chatbot** (30s, IF time allows) — ask a pre-prepared question live ("why is NVDA
    weighted so heavily?") to show the answer is grounded in the actual computed numbers. If
    there's genuinely time left, also show the manual academic-literature search
    ("Ledoit-Wolf shrinkage") to show real Semantic Scholar/arXiv citations.

**If you're running late at 9:30**, skip straight past steps 9-10 and move to slide 5 — never
cut step 6 (Forecast & Compare) or step 8 (Walk-forward), those are the two centerpieces of
the brief and of your differentiation.

## 11:00 – 12:30 — Results & differentiators (slides 5-7)

*Slide 5*: one number to say out loud — the expected Sharpe ratio, 1.70 — let people read the
rest themselves. Mention in one sentence that the optimal weights lean into AI growth (NVDA,
AVGO, PANW, MSFT) offset by short positions in ADBE and QCOM.

*Slide 6*: reuse the table already shown live, deliver the closing line: "Forecasting adds a
real but modest edge — consistent with the efficient-market hypothesis, not a miracle
signal." That's the kind of line that reassures a jury about your rigor.

*Slide 7*: move quickly, a breather after the demo's pace — "Beyond the brief: walk-forward,
GARCH, PCA, macro panel, RAG chatbot, resilient LLM layer. 300+ tests, strict mypy."

## 12:30 – 13:15 — AI layer & fail-soft (slide 8, new)

"One thing I want to highlight before talking methodology: nothing in this app depends on a
single point of failure. Five market-data providers in cascade, five LLM providers in
cascade, three sentiment tiers — if a provider goes down or an API key expires mid-demo, the
app keeps working, just from a different source. That's a deliberate architecture choice,
not an accident."

## 13:15 – 14:15 — AI methodology (slide 9)

This is your differentiator for the GenAI Developer track — don't rush it, but keep it
tight. "I didn't just ask an AI to write code: I built a real harness — a precise spec,
context centralized in one config file, a real test-driven feedback loop. Here are a couple
of real bugs found by testing live, and how they became permanent rules." Cite ONE example
in detail (the SEC EDGAR CIK bug or the FinBERT sentiment bug after the Hugging Face endpoint
migration are the most compelling — genuine false positives, found by testing live, not by
reading documentation).

## 14:15 – 15:00 — Closing (slide 10)

"To sum up: an app that optimizes, forecasts, and honestly compares the two — including when
the forecast doesn't beat history. Thank you — questions?" Nothing else. Don't repeat the
repo link out loud, it's already on screen.

---

## Safety net

- **If deployment (Render/Streamlit Cloud) is slow to start**: open the app 2-3 minutes
  BEFORE you start talking, in a background tab, to let the free tier's cold start finish.
- **If a technical question comes up mid-demo**: note it mentally, answer after closing —
  never stop the clock to answer it in the middle.
- **Likely question to prepare**: "Why not Riskfolio-Lib / Kats?" → short answer already in
  the README — reread it once before presenting so it comes out without hesitation.
- **Another likely question**: "Why doesn't the forecast beat history more decisively?" →
  short answer: the market sits close to efficient (Hurst near 0.5, already shown in the
  demo at step 3) — a modest edge is the honest result to expect, not a sign the model
  failed.
- **If you need to cut time at the last minute**: cut step 10 (chatbot) first, then step 5
  (factor exposure) — never steps 6 and 8 (Forecast & Compare, Walk-forward), which are the
  core of the brief.
