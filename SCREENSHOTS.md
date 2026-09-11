# Screenshots

**Forecast model used:** every screenshot below uses **LSTM (recurrent neural net)**, except the
forecast fan chart, which is shown twice below — once in ML regression, once in LSTM — to compare
how the two models' confidence bands differ.

**Overview — Macro & Risk context**
<img src="images/overview_macro_risks.png" width="1000" alt="Macro and risk context panel: FRED yields, term spread, VIX, Sahm Rule">

**Overview — Prices & Analytics**
<img src="images/prices_analysis.png" width="1000" alt="Rebased price history, per-asset annualised metrics, and correlation matrix">

**Overview — Time-series diagnostics**
<img src="images/timeseries_diagnostics.png" width="1000" alt="ADF stationarity test and Hurst exponent per ticker, rolling Sharpe chart">

**Overview — Fundamentals**
<img src="images/fundamentals.png" width="1000" alt="Per-ticker fundamentals table: market cap, P/E, beta, dividend yield, 52-week range">

**Sidebar — key controls & forecast model selector**
<table><tr>
<td><img src="images/sidebar-controls.png" width="350" alt="Risk-free rate, max weight per asset, short selling, transaction cost, forecast horizon, walk-forward windows, covariance estimator"></td>
<td><img src="images/forecast_model_selector.png" width="350" alt="The six forecast models: Naive, ETS, Theta, ARIMA, ML regression, LSTM"></td>
</tr></table>

**Efficient Frontier — with optimal weights**
<table><tr>
<td><img src="images/efficient_frontier.png" width="900" alt="Mean-variance efficient frontier with diversification ratio and HHI"></td>
<td><img src="images/weights.png" width="100" alt="Optimal portfolio weights table, long and short positions"></td>
</tr></table>

**Efficient Frontier — Fama-French factor exposure**
<img src="images/factor_exposure.png" width="1000" alt="Factor exposure expander: Fama-French loadings, annualised alpha, R-squared">

**Forecast & Compare — the three-portfolio comparison**
<img src="images/forecast_compare.png" width="1000" alt="Historical vs forecast-based vs realized-optimal portfolio comparison table and chart">

**Forecast & Compare — multi-asset confidence band (ML regression)**
<img src="images/forecast_fan_chart_ML.png" width="1000" alt="Forecast fan chart comparing 4 assets (AAPL/AVGO/LLY/WMT) using ML regression (gradient boosting), each with its own 95% confidence band widening with horizon">

Why ML regression here: the recursive multi-step forecasts widen visibly and asymmetrically per
asset, which makes the confidence-band mechanic easy to read on one chart.

**Forecast & Compare — multi-asset confidence band (LSTM)**
<img src="images/forecast_fan_chart_LSTM.png" width="1000" alt="Forecast fan chart comparing the same 4 assets using LSTM (recurrent neural net), showing a mild trend continuation instead">

Same four assets, same horizon, LSTM instead of ML regression: the point forecasts pick up a
slight trend continuation (visible on AAPL/LLY) rather than flattening out — a direct visual
comparison of how each model's forecast behaves on identical data.

**Walk-forward validation — Sharpe distribution across windows**
<img src="images/walk_forward_distrib.png" width="1000" alt="Box plot of Sharpe ratio across 5 walk-forward windows for the three portfolio types, with the forecast-vs-historical win rate">

**Walk-forward validation — period-over-period**
<img src="images/walk_forward_period.png" width="1000" alt="Each portfolio type's own trajectory window by window, current value next to the previous window's value">

**AI Analyst — commentary & news digest**
<img src="images/AI_portfolio_analyst.png" width="1000" alt="LLM-generated portfolio commentary and cross-referenced, per-ticker news digest">

<br>

**AI Analyst — sentiment by ticker and sources**
<table><tr>
<td><img src="images/sentiment_sources.png" width="500" alt="News sentiment by ticker (FinBERT financial-domain model), tagged by source"></td>
<td><img src="images/sources.png" width="500" alt="Every headline/filing behind the digest, tagged by provider (NewsAPI, Finnhub, SEC EDGAR, GDELT, Google News, TED), linking out to the original"></td>
</tr></table>

<br>

**Chatbot — grounded Q&A**
<img src="images/chatbot_1.png" width="1000" alt="Chatbot explaining why MSFT sentiment is bearish, combining the negative model weight, the one filed 8-K, and the current macro backdrop">
<img src="images/chatbot_2.png" width="1000" alt="Chatbot follow-up building a core-satellite allocation framework, context retained across turns">
<img src="images/chatbot_3.png" width="1000" alt="Chatbot explaining why NVDA and AVGO outperformed, grounded in the retrieved news digest">
<img src="images/chatbot_4.png" width="1000" alt="Chatbot explaining the Sortino ratio formula in plain text (no LaTeX) and computing it for the historical/forecast/realized portfolios from the app's own results">
<img src="images/chatbot_5.png" width="1000" alt="Chatbot answering a risk-profile and time-horizon question, flagging its own annualised-return caveat on a short window">

<br>

**Chatbot — academic literature search**
<img src="images/search_academic_literature.png" width="1000" alt="Manual academic-literature search box under the Chatbot tab, showing recognised-term query expansion and real Semantic Scholar/arXiv results for 'walk-forward validation'">
---
