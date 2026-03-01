"""
Configuration constants.
Sensitive values (API keys) are read from environment variables only —
never hardcoded here.
"""

import os

# --- Extraction engine ---
# Offline, deterministic extraction via spaCy.
# No API key required. Model must be downloaded once:
#   python3 -m spacy download en_core_web_sm
SPACY_MODEL = "en_core_web_sm"

# --- Wikipedia ---
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_REQUEST_TIMEOUT = 15  # seconds

# --- Storage ---
OUTPUT_DIR = "output"
