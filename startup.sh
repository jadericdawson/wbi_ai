#!/bin/bash
# Azure Web App startup script
PORT=${PORT:-8000}
python -m streamlit run app_mcp.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
