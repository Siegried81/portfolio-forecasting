# Screenshots

Rendered at a fixed width below (via HTML `<img>`, not plain Markdown `![]()`) so every screenshot
lines up consistently regardless of its original crop/resolution — plain Markdown image syntax
renders each one at its native size, which is what made narrower column crops look misaligned
against the rest on GitHub.

**Overview — Macro & Risk context**
<img src="images/overview_macro_risks.png" width="900" alt="Macro and risk context panel: FRED yields, term spread, VIX, Sahm Rule">

**Overview — Prices & Analytics**
<img src="images/prices_analysis.png" width="900" alt="Rebased price history, per-asset annualised metrics, and correlation matrix">

**Overview — Time-series diagnostics**
<img src="images/timeseries_diagnostics.png" width="900" alt="ADF stationarity test and Hurst exponent per ticker, rolling Sharpe chart">

**Overview — Fundamentals**
<img src="images/fundamentals.png" width="900" alt="Per-ticker fundamentals table: market cap, P/E, beta, dividend yield, 52-week range">

**Sidebar — key controls & forecast model selector**
<table><tr>
<td><img src="images/sidebar_controls.png" width="440" alt="Risk-free rate, max weight per asset, short selling, transaction cost, forecast horizon, walk-forward windows, covariance estimator"></td>
<td><img src="images/forecast_model_selector.png" width="440" alt="The six forecast models: Naive, ETS, Theta, ARIMA, ML regression, LSTM"></td>
</tr></table>

**Efficient Frontier — with optimal weights**
<table><tr>
<td><img src="images/weights.png" width="300" alt="Optimal portfolio weights table, long and short positions"></td>
<td><img src="images/efficient_frontier.png" width="580" alt="Mean-variance efficient frontier with diversification ratio and HHI"></td>
</tr></table>

**Efficient Frontier — Fama-French factor exposure**
<img src="images/factor_exposure.png" width="900" alt="Factor exposure expander: 3-factor Fama-French loadings, annualised alpha, R-squared">

**Forecast & Compare — the three-portfolio comparison**
<img src="images/forecast_compare.png" width="900" alt="Historical vs forecast-based vs realized-optimal portfolio comparison table and chart">

**Forecast & Compare — single-asset confidence band**
<img src="images/forecast_fan_chart.png" width="900" alt="Forecast fan chart for a single asset, showing the 95% confidence band widening with horizon">

**Walk-forward validation — period-over-period**
<img src="images/walk_forward_tabs.png" width="900" alt="The four walk-forward sub-tabs (Distribution / Summary / Period-over-period / Raw data)">

**AI Analyst — commentary & news digest**
<img src="images/AI_portfolio_analyst.png" width="900" alt="LLM-generated portfolio commentary and cross-referenced, per-ticker news digest">

**AI Analyst — sentiment by ticker**
<img src="images/sentiment_sources.png" width="650" alt="News sentiment by ticker (FinBERT financial-domain model, falling back to Finnhub aggregated then local VADER), tagged by source">

**AI Analyst — sources**
<img src="images/sources.png" width="900" alt="Every headline/filing behind the digest, tagged by provider (NewsAPI, Finnhub, SEC EDGAR, GDELT, Google News, TED), linking out to the original">

**Chatbot — academic literature search**
<img src="images/search_academic_literature.png" width="900" alt="Manual academic-literature search box, returning real Semantic Scholar and arXiv papers for a quant-finance methodology query">

**Chatbot — grounded Q&A**
<img src="images/chatbot_1.png" width="900" alt="Chatbot answering a question about NVDA/AVGO's recent outperformance, grounded in the news digest">
<img src="images/chatbot_2.png" width="900" alt="Chatbot follow-up building a per-driver comparison table, context retained across turns">
<img src="images/chatbot_3.png" width="900" alt="Chatbot answering a general allocation question while still flagging the back-test's own caveats">