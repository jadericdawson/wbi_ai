#!/bin/bash
# Azure Web App startup script with Playwright browser installation
# This script installs Playwright browsers in Azure App Service environment

echo "=== Azure Startup Script ==="

# Install Playwright system dependencies (if not already installed)
# Azure App Service Linux base image may not have all dependencies
if command -v playwright &> /dev/null; then
    echo "Installing Playwright browsers..."

    # Install only Chromium (lighter than full install)
    python -m playwright install chromium --with-deps || {
        echo "WARNING: Failed to install Playwright browsers"
        echo "Playwright scraping will be disabled, falling back to basic scraping"
    }
else
    echo "WARNING: Playwright not found in requirements.txt"
fi

# Set environment variable to indicate Azure environment
export ENVIRONMENT=${ENVIRONMENT:-production}

# Get port from Azure (default 8000)
PORT=${PORT:-8000}

echo "Starting Streamlit on port $PORT..."

# Start Streamlit app
python -m streamlit run app_mcp.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection true
