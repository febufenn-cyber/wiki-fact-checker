"""
Entry point for the Atomic Assertion Analyzer — extraction layer.

Usage:
    python main.py                    # defaults to Apollo 11
    python main.py "Marie Curie"
    python main.py "Battle of Thermopylae"

Output:
    output/{slug}_{revision_id}.json   — all assertions for the article
    Terminal summary with counts, type distribution, avg confidence, samples.

Scope: extraction only. Scoring, embeddings, and coherence analysis are
implemented in later steps.
"""

import dataclasses
import json
import logging
import os
import sys

from config import OUTPUT_DIR
from extractor import extract_assertions
from report import print_report
from reporter import generate_report
from wikipedia import fetch_article

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run(article_title: str) -> None:
    print(f"\nArticle : {article_title!r}")
    print("-" * 50)

    # --- Fetch ---
    try:
        article = fetch_article(article_title)
    except Exception as exc:
        logger.error("Failed to fetch article: %s", exc)
        sys.exit(1)

    revision_id = article["revision_id"]
    sections = article["sections"]
    print(f"Revision ID : {revision_id}")
    print(f"Sections    : {len(sections)}")
    print()

    # --- Extract ---
    all_assertions = []

    for section in sections:
        sec_title = section["title"]
        sec_text = section["text"]
        char_count = len(sec_text)

        if not sec_text.strip():
            print(f"  [skip]  {sec_title!r}  — empty after stripping")
            continue

        print(f"  Extracting  {sec_title!r}  ({char_count} chars) ...", end="", flush=True)

        assertions = extract_assertions(
            section_text=sec_text,
            section_title=sec_title,
            article_title=article_title,
            revision_id=revision_id,
            position_offset=len(all_assertions),
        )
        all_assertions.extend(assertions)
        print(f"  {len(assertions)} assertions")

    # --- Write JSON output (unchanged format) ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slug = article_title.lower().replace(" ", "_")
    filename = f"{slug}_{revision_id}.json"
    output_path = os.path.join(OUTPUT_DIR, filename)

    serialized = [dataclasses.asdict(a) for a in all_assertions]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)

    # --- Terminal report (visual, for the current session) ---
    print()
    print_report(all_assertions, article_title, revision_id, output_path)

    # --- Plain-text report file (saveable, shareable) ---
    try:
        report_path = generate_report(output_path)
        print(f"  Plain-text report: {report_path}")
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)


if __name__ == "__main__":
    title = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Apollo 11"
    run(title)
