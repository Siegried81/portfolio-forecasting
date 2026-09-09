#!/bin/bash
# Build and run the Portfolio Forecasting Docker deployment.
# Run from the repo root (no hardcoded machine-specific path — cd here yourself first).

set -e

PROJECT_NAME="portfolio-forecasting"
IMAGE_NAME="$PROJECT_NAME"
CONTAINER_NAME="$PROJECT_NAME"
PORT="8501"

echo "=== Building Docker image for $PROJECT_NAME ==="
echo "This may take 3-5 minutes due to large dependencies (scipy, cvxpy, statsmodels)..."

# --verbose: stream the full, unbuffered build log (useful the first time you
# build, or when a build fails and you need to see which layer broke).
if [ "$1" == "--verbose" ]; then
    docker build -t "$IMAGE_NAME" . --progress=plain 2>&1 | tail -50
else
    docker build -t "$IMAGE_NAME" .
fi

echo ""
echo "=== Build complete ==="
docker images | grep "$IMAGE_NAME"

echo ""
echo "=== Running container ==="
echo "Container will be accessible at: http://localhost:$PORT"
echo ""

if [ ! -f .env ]; then
    echo "⚠️  WARNING: .env file not found!"
    echo "   Copy .env.example to .env and add your API keys:"
    echo "   - GROQ_API_KEY (for AI Analyst tab)"
    echo "   - NEWSAPI_KEY (for news digest)"
    echo "   - FRED_API_KEY (for live Treasury yields)"
    echo "   - TWELVEDATA_API_KEY (for market data fallback)"
    echo ""
    echo "   The app will run without these keys, but features will be limited."
    echo ""
fi

echo "Starting with docker compose..."
docker compose up -d --pull always

echo ""
echo "✓ Container is starting. Check status with:"
echo "  docker ps | grep $CONTAINER_NAME"
echo ""
echo "✓ View logs with:"
echo "  docker compose logs -f"
echo ""
echo "✓ Stop with:"
echo "  docker compose down"