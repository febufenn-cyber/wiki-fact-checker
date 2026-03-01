#!/usr/bin/env bash
set -euo pipefail

echo "== Wiki Fact-Checker setup =="

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: $PYTHON not found. Install Python 3 and retry." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR..."
  "$PYTHON" -m venv "$VENV_DIR"
else
  echo "Using existing virtual environment in $VENV_DIR"
fi

echo "Activating virtual environment..."
. "$VENV_DIR/bin/activate"

echo "Upgrading pip and installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Downloading spaCy model en_core_web_sm..."
python -m spacy download en_core_web_sm

echo ""
echo "Setup complete. To run the pipeline, activate the venv and run:" 
echo "  source $VENV_DIR/bin/activate"
echo "  python main.py \"World War II\""
echo ""
echo "Outputs will be written under revisions/, comparisons/, stability/, reports/, and output/."
