#!/bin/bash

# Streamlit startup script for Azure App Service
# This overrides the default Oryx-generated startup

echo "========================================="
echo "Starting Streamlit Application"
echo "========================================="

# Activate virtual environment if it exists
if [ -d "antenv" ]; then
    echo "Activating virtual environment..."
    source antenv/bin/activate
fi

# Set Python path
export PYTHONUNBUFFERED=1

# Start Streamlit
echo "Launching Streamlit on port 8000..."
python -m streamlit run app_mcp.py \
    --server.port=8000 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --logger.level=info
