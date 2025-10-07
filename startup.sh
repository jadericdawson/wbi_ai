#!/bin/bash

# Azure Web App startup script for Streamlit
# Azure sets PORT environment variable - use it or default to 8000
PORT=${PORT:-8000}

# Activate virtual environment (built by Azure Oryx)
if [ -d "/home/site/wwwroot/antenv" ]; then
    source /home/site/wwwroot/antenv/bin/activate
fi

# Run streamlit
python -m streamlit run app_mcp.py \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
