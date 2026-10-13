# Portfolio Forecasting - containerised Streamlit app.
# Build:  docker build -t portfolio-forecasting .
# Run:    docker run -p 8501:8501 --env-file .env portfolio-forecasting
FROM python:3.14-slim

# Prevents Python from writing .pyc files and buffers stdout (cleaner container logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Install deps first (separate layer) so code changes don't invalidate the pip cache
COPY requirements.txt .
# CPU-only torch first: on Linux the default PyPI wheel bundles several GB of CUDA
# libraries that Render (no GPU) never uses. requirements.txt's torch>=2.0 is then
# already satisfied, so pip does not pull the CUDA build on top.
# xgboost likewise: on Linux it requires nvidia-nccl (305 MB) for multi-GPU
# training, so the image installs the CPU-only `xgboost-cpu` distribution (same
# `xgboost` import) and drops the `xgboost` line from requirements.txt.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && grep -v '^xgboost' requirements.txt > /tmp/requirements-cpu.txt \
    && pip install --no-cache-dir 'xgboost-cpu>=2.0,<4.0' -r /tmp/requirements-cpu.txt

COPY . .

# Non-root user: if the app is ever compromised, the attacker is not root in the
# container. -m gives it a home, where Streamlit and model caches write; the app
# itself only writes to /tmp (yfinance cache).
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8501

# Basic healthcheck so `docker ps` / orchestrators can see if the app is actually serving
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f\"http://localhost:{os.environ.get('PORT', '8501')}/_stcore/health\")" || exit 1

# Render injects PORT; local docker run / compose fall back to 8501.
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501}"]