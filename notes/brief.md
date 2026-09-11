# Portfolio Forecasting and Optimization: An Introduction to Financial Engineering

- Repository: `portfolio-forecasting`
- Type of Challenge: `Consolidation`
- Duration: `5 days`
- Deadline: `11/09/2026 17:00`
- Deployment strategy :
  - Github page
- Team challenge : `solo`


## Data

You should use **publicly available financial market data**. The easiest option is the **Yahoo Finance API** through the `yfinance` Python package.

- **Assets**: US equities (e.g., AAPL, MSFT, TSLA, AMZN, GOOG) are good starting points. You may extend to ETFs or indices if desired.
- **Price type**: Adjusted closing prices (to account for dividends and stock splits).
- **Frequency**: User-defined (daily, monthly, yearly).
- **Date range**: User-defined — ensure you handle enough historical data to compute returns and allow for a forecasting window.



## Mission Objectives
This project simulates a **real technical interview case study** where you apply **financial engineering**, **machine learning**, and **data science** skills to create an interactive application for portfolio analysis.

- Understand **time-series manipulations** (returns, volatility, correlations).
- Apply **time-series forecasting** methods to predict future prices.
- Gain familiarity with **finance vocabulary** (Sharpe ratio, Sortino ratio, efficient frontier).
- Tackle an **optimization problem** using mean-variance portfolio theory.
- Deploy an interactive **Streamlit app** for user exploration.



## The Mission

Imagine you are in the **second round of interviews** with a major financial institution. The HR part is done — now it’s time to prove your technical ability.

You have only a few days to deliver a working **portfolio optimization web app**. The requirements:
- Input: user selects **stocks, date range, frequency**.
- Output:
  - The **optimal portfolio weights** via mean-variance optimization.
  - Visualization of the **efficient frontier**.
  - Key portfolio metrics: **return, volatility, Sharpe ratio, Sortino ratio**.

You explore existing tools such as [Portfolio Visualizer](https://www.portfoliovisualizer.com/) and realize you can go further than simple historical optimization. You decide to **stand out** by combining finance and forecasting:

1. Forecast **future stock prices** over a new time range.
2. Construct the **optimal portfolio** using forecasted data.
3. Compute the **true realized optimal portfolio** from actual future prices.
4. Compare all three portfolios:
   - Historical-based optimization.
   - Forecast-based optimization.
   - Actual realized optimal.



## Roadmap

1. **Finance vocabulary**: review portfolio theory, returns, volatility, covariance, Sharpe and Sortino ratios.
2. **Explore Portfolio Visualizer** for user experience inspiration.
3. **Implement optimization** using [Riskfolio-Lib](https://riskfolio-lib.readthedocs.io/en/latest/index.html) (or [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) as an alternative).
4. **Choose forecasting framework**:
   - [Kats](https://facebookresearch.github.io/Kats/) for advanced forecasting, or
   - [PyCaret](https://pycaret.org/) for easy regression/forecasting pipelines.
5. **Develop the web app** in **Streamlit**:
   - User inputs: stock list, time range, frequency.
   - Outputs: optimal portfolio allocation, efficient frontier chart, performance metrics.
   - Optional tabs: historical vs forecast vs realized portfolios.
6. **Deploy app** (Streamlit Community Cloud, Render, or similar).
7. **Prepare a live demo (5 minutes)** for your presentation.



## Tools & Libraries

- **Data acquisition**: `yfinance` (Yahoo Finance API).
- **Data manipulation**: `pandas`, `numpy`.
- **Portfolio optimization**: `Riskfolio-Lib` or `PyPortfolioOpt`.
- **Forecasting**: `Kats` or `PyCaret`.
- **Visualization**: `matplotlib`, `plotly`.
- **Deployment**: `Streamlit`.



## Deliverables

- A **Streamlit web app** where users can test portfolio optimization interactively.
- **Visualizations**: efficient frontier, portfolio allocations, risk-return trade-offs.
- **Metrics**: historical vs forecasted vs realized performance (returns, volatility, Sharpe, Sortino).
- A **short presentation/demo** showing functionality and insights.
- A **clean repository** with documentation, clear README, and no unnecessary files.



## Evaluation Criteria

| Criteria       | Indicator                                                | Yes/No |
| -------------- | -------------------------------------------------------- | ------ |
| **Is complete** | Optimization works, app is clear, presentation understandable | [ ] |
|                | README is polished and informative                       | [ ] |
|                | Forecasting model is trained and produces outputs        | [ ] |
|                | App is deployed with Streamlit                           | [ ] |
|                | MVP meets client needs exactly                           | [ ] |
| **Is good**    | Repo is clean (no extra files)                           | [ ] |
|                | Python typing is used consistently                      | [ ] |
|                | Presentation is clean and professional                   | [ ] |


## Expected Outcome

By the end of this project you will:
- Understand how to **manipulate financial time-series**.
- Be able to **forecast stock prices** using modern ML tools.
- Implement a **mean-variance optimization framework** with clear visualizations.
- Deliver an **interactive app** where end-users can experiment with portfolios.
- Critically evaluate how **forecasting impacts portfolio optimization**, showing both strengths and limitations.
- Impress your interviewers with a **professional, well-rounded solution**.
