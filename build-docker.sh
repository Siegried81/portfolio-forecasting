#!/bin/bash
# Build portfolio-forecasting Docker image on WSL

cd /mnt/d/Users/Siegried/Desktop/Becode/portfolio-forecasting

echo "🐳 Building Docker image for portfolio-forecasting..."
echo "This takes 3-5 minutes on first build."
echo ""

docker build -t portfolio-forecasting . --progress=plain 2>&1 | tail -50

echo ""
echo "✅ Build complete!"
echo ""
echo "Checking image..."
docker images | head -2

echo ""
echo "Running container..."
echo "docker run -d --name portfolio-forecasting -p 8501:8501 --env-file .env portfolio-forecasting"
echo ""
echo "Then access: http://localhost:8501"
