# Wiki Fact-Checker

Atomic Assertion Analyzer — extract, compare and score Wikipedia claims for epistemic stability.

Overview

- Fetches a Wikipedia article (sectioned), extracts atomic claims using spaCy, compares annual revisions, computes stability scores, and generates a human-readable report.

Quick start

1. Create and activate a Python venv (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies and spaCy model:

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Run the full pipeline for an article (example):

```bash
python main.py "World War II"
```

This command will:

- fetch and store annual revisions under `revisions/{slug}/`
- run cross-revision comparisons under `comparisons/{slug}/`
- compute stability and write `stability/{slug}/stability.json`
- generate a plain-text report in `reports/` and print it to terminal

Notes

- If you prefer a dry run or already have stored revisions, you can run `python main.py "Apollo 11"` to reuse cached data.
- Output JSON assertion files are written to `output/`.

Development

- Code entry points:
  - `main.py` — orchestrates the full pipeline
  - `revisions.py` — fetches and stores annual revisions
  - `compare.py` — compares assertions between revisions
  - `stability.py` — computes stability scoring
  - `reporter.py` / `report.py` — generate human-readable reports

License

- This repository contains research tooling. Add your license as appropriate.
