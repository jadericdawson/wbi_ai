#!/bin/bash

# Azure Web App startup script for Streamlit
# Azure sets PORT environment variable - use it or default to 8000
PORT=${PORT:-8000}

# Check if virtual environment has GLIBC compatibility issues and remove it if so
if [ -d "/home/site/wwwroot/antenv" ]; then
    # Test if cryptography module loads (it fails with GLIBC mismatch)
    if ! /home/site/wwwroot/antenv/bin/python -c "import cryptography" 2>/dev/null; then
        echo "Removing incompatible virtual environment..."
        rm -rf /home/site/wwwroot/antenv
        echo "Virtual environment removed. Azure Oryx will rebuild it on next deployment."
        echo "Please redeploy the application."
        exit 1
    fi
    source /home/site/wwwroot/antenv/bin/activate
fi

# Run streamlit
python -m streamlit run app_mcp.py \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
