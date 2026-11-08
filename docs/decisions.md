# Decisions

Dated record of changes to what a number means. The code comments say how a
metric is computed; this file says when and why that changed.

## 2026-11-08: Jensen's alpha uses the dates the asset and benchmark share

`metrics.jensens_alpha` annualised the asset's return over its own history and
the benchmark's return over the benchmark's full history. For an asset listed
after the start of the selected range, the CAPM prediction then included years
the asset never traded. Beta and the information ratio were already estimated
on the shared dates; alpha now is too.

Measured on a synthetic asset that copies SPY exactly from mid-sample (beta
1.0): alpha was -32.9% a year before, 0.0% after. Assets with full history over
the range get the same number as before.

## 2026-11-08: Drawdowns are measured from the starting wealth

`metrics.max_drawdown` and `metrics.ulcer_index` started the running peak at the
first period's close. A loss in the first period was therefore never counted:
returns of -10% then -10% gave a max drawdown of -10% instead of -19%. The peak
now starts at the initial wealth of 1.0. Calmar uses max drawdown, so it moves
with it.

This matters most on the short held-out window of the Forecast & Compare tab
and on each walk-forward window, where one bad opening day is a large share of
the path. Series whose first period is a gain are unchanged.
