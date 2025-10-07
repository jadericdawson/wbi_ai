#!/bin/bash
# Pre-build script: Remove any existing incompatible virtual environment
if [ -d "/home/site/wwwroot/antenv" ]; then
    echo "Removing existing virtual environment to force rebuild..."
    rm -rf /home/site/wwwroot/antenv
fi
