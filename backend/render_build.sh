#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Build script started..."

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Generatng placeholder models..."
python download_models.py

echo "Build completed successfully!"
