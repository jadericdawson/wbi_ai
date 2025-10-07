#!/bin/bash

# Azure Web App startup script for Streamlit
# Azure sets PORT environment variable - use it or default to 8000
PORT=${PORT:-8000}

python -m streamlit run app_mcp.py \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
