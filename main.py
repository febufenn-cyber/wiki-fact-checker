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
from revisions import (
    fetch_revision_history,
    load_revision_index,
    load_revision_content,
)
from compare import chain_compare
from stability import compute_stability

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run(article_title: str) -> None:
    print(f"\nArticle : {article_title!r}")
    print("-" * 50)

    # --- Ensure we have stored annual revisions (used by stability & comparisons)
    try:
        fetch_revision_history(article_title)
    except Exception as exc:
        logger.warning("Could not fetch revision history: %s", exc)

    # Prefer stored revision content (more consistent across pipeline)
    index = load_revision_index(article_title)
    if index:
        newest = index.get("revisions", [])[-1]
        revision_id = str(newest["revid"])
        content = load_revision_content(article_title, int(revision_id))
        if content:
            sections = content.get("sections", [])
            print(f"Revision ID : {revision_id} (from stored revisions)")
            print(f"Sections    : {len(sections)}")
            print()
        else:
            # Fallback to live fetch_article
            try:
                article = fetch_article(article_title)
                revision_id = article["revision_id"]
                sections = article["sections"]
                print(f"Revision ID : {revision_id} (fetched live)")
                print(f"Sections    : {len(sections)}")
                print()
            except Exception as exc:
                logger.error("Failed to load sections for extraction: %s", exc)
                sys.exit(1)
    else:
        # No stored revisions — fall back to live fetch
        try:
            article = fetch_article(article_title)
            revision_id = article["revision_id"]
            sections = article["sections"]
            print(f"Revision ID : {revision_id} (fetched live)")
            print(f"Sections    : {len(sections)}")
            print()
        except Exception as exc:
            logger.error("Failed to fetch article: %s", exc)
            sys.exit(1)

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

    # --- Full pipeline: revisions → comparisons → stability → report ---
    try:
        # Ensure we have an annual revision history stored (resumable)
        print("\nEnsuring revision history is stored (may take a while)...")
        fetch_revision_history(article_title)

        # Build comparisons for all consecutive pairs (skips existing)
        print("Running cross-revision comparisons (chain)...")
        chain_compare(article_title)

        # Compute epistemic stability (writes stability/{slug}/stability.json)
        print("Computing stability scores...")
        compute_stability(article_title)

        # Generate the plain-text stability-aware report and print it
        report_path = generate_report(article_title, print_to_terminal=True)
        print(f"  Plain-text report: {report_path}")
    except Exception as exc:
        logger.error("Pipeline step failed: %s", exc)


if __name__ == "__main__":
    title = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Apollo 11"
    run(title)
