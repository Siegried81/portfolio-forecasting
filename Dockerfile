# Portfolio Forecasting - containerised Streamlit app.
# Build:  docker build -t portfolio-forecasting .
# Run:    docker run -p 8501:8501 --env-file .env portfolio-forecasting
FROM python:3.12-slim

# Prevents Python from writing .pyc files and buffers stdout (cleaner container logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Install deps first (separate layer) so code changes don't invalidate the pip cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Basic healthcheck so `docker ps` / orchestrators can see if the app is actually serving
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501"]