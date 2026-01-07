#!/bin/bash
# Helper script to activate virtual environment and run commands
# Usage: ./activate_and_run.sh ingest.py
#        ./activate_and_run.sh chat.py

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Run the provided command
if [ $# -eq 0 ]; then
    echo "Usage: ./activate_and_run.sh <script.py>"
    echo "Example: ./activate_and_run.sh ingest.py"
    echo "Example: ./activate_and_run.sh chat.py"
    exit 1
fi

# Run the Python script
python "$@"

