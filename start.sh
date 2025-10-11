#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Run the pre-start script to patch dependencies
echo "--- Running pre-start script ---"
python backend/prestart.py
echo "--- Pre-start script finished ---"

# Start the main application
echo "--- Starting Uvicorn server ---"
uvicorn backend.app:app --host 0.0.0.0 --port 8000
