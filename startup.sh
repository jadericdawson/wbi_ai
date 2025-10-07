#!/bin/bash

# Azure Web App startup script for Streamlit
python -m streamlit run app_mcp.py --server.port 8000 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false
