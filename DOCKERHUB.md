# Portfolio Forecasting

## Quick reference

- **Source code, Dockerfile, issues:** https://github.com/Siegried81/portfolio-forecasting
- **Maintained by:** Siegried81
- **Also on:** GitHub Container Registry, `ghcr.io/siegried81/<image>`
- **Architecture:** `linux/amd64`

## Supported tags

- `latest`: the current `main` branch
- `X.Y.Z` and `X`: released versions (git tags `vX.Y.Z`), the ones to pin for a stable deployment
- `sha-xxxxxxx`: one specific commit

## What is it?

Streamlit app to backtest, optimise and forecast an equity portfolio: walk-forward backtests, mean-variance and risk-parity optimisation, ARIMA/LSTM/GARCH forecasts, factor exposures, macro context and an LLM analyst that answers from the computed numbers.

Images from this repository:

- `portfolio-forecasting`: Streamlit UI on 8501 (CPU-only PyTorch)

## How to use this image

```bash
docker run -d -p 8501:8501 siegried/portfolio-forecasting
# open http://localhost:8501

# with API keys for the AI analyst, news and fallback data sources
docker run -d -p 8501:8501 --env-file .env siegried/portfolio-forecasting
```

## Environment variables

None are required to start the UI. Secrets are **never** baked into the image: pass
them at run time with `-e NAME=value` or `--env-file .env` (see `.env.example` in the repo).

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | AI analyst (Groq); OpenRouter, Cerebras, SambaNova and Ollama are fallbacks |
| `NEWSAPI_KEY / FINNHUB_API_KEY` | news digest and fundamentals |
| `TWELVEDATA_API_KEY / ALPHA_VANTAGE_API_KEY` | market data fallbacks when Yahoo fails |
| `FRED_API_KEY` | live Treasury yields |
| `REDIS_URL` | optional shared cache, e.g. `redis://redis:6379/0` |
| `PORT` | listening port, set automatically on Render |

## Volumes

- none: the image is self-contained

## License

See the LICENSE file in the source repository, if any. As with every image, it is
your responsibility to check that your use complies with the licenses of the
software it contains.
