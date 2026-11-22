# Decisions

What changed the meaning of a number or a behaviour, when and why. The code comments say how a
metric is computed; this file says why it is computed that way. Oldest first.

## 2026-11-08: Jensen's alpha uses the dates the asset and benchmark share

`metrics.jensens_alpha` used to annualise the asset over its own history and the benchmark over its
full history, so a late-listed asset was judged against years it never traded. Beta and the
information ratio already used the shared dates. On a synthetic asset copying SPY from mid-sample,
alpha went from -32.9% a year to 0.0%. Assets with full history are unchanged.

## 2026-11-08: Drawdowns are measured from the starting wealth

`max_drawdown` and `ulcer_index` started the running peak at the first period's close, so a loss in
the first period never counted: returns of -10% then -10% gave -10%, not -19%. The peak now starts at
wealth 1.0, and Calmar moves with it. It matters most on the short held-out window and on each
walk-forward window. Series whose first period is a gain are unchanged.

## 2026-11-20: A rejected Groq key no longer ends the key rotation

`llm_client._call_groq` rotated to the next key only on a 429, so one revoked or mistyped key (401)
sent every request to the fallback providers while the other keys sat unused. A 401 now rotates like a
429. Other errors (a deprecated model, the network) still fail fast, since they hit every key alike.

## 2026-11-22: Fundamentals fall back to yfinance, field by field

With no Finnhub key and Twelve Data limited to AAPL, most tickers came back empty. yfinance now fills
any field still missing after Finnhub and Twelve Data, and is the only source when neither answers. It
never overwrites another provider's value and never fills `dividend_yield` (its unit changed between
releases). Its beta, P/E and 52-week range can differ slightly from Finnhub's, so `field_sources`
records who supplied each field and the app captions the filled-in ones.

## 2026-11-22: CI runs on Python 3.14

CI used 3.12 while the Dockerfile and the local venv use 3.14, so a 3.14-only defect could pass CI and
ship. All pinned dependencies install on 3.14 and the suite and `mypy --strict` pass there.
