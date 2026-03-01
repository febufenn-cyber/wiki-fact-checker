"""
Step 3 — Revision History Fetching.

Fetches a sampled set of historical revisions for a Wikipedia article and
stores their cleaned section content, ready for assertion extraction in Step 4.

Sampling strategy — one revision per calendar year:
  Wikipedia articles accumulate thousands of individual edits. Fetching every
  edit is infeasible and unnecessary for stability analysis. Annual snapshots
  capture the article's epistemic state at meaningful intervals — long enough
  that genuine content shifts are visible, short enough to trace editorial
  trends over time.

  For each target year, the module fetches the *last* revision of that year
  (the article's state at year-end). This is consistent and reproducible.

Storage layout:
  revisions/{slug}/index.json            — metadata for all sampled revisions
  revisions/{slug}/{revid}_content.json  — cleaned section text for one revision

Content format mirrors fetch_article() output so Step 4 can pass stored
sections directly into extract_assertions() without any re-fetching.

The module is resumable: revisions already present on disk are skipped, so
re-running after a partial failure picks up where it left off.

Usage:
  python3 revisions.py "Apollo 11"              # last 10 years (default)
  python3 revisions.py "Apollo 11" --years 20   # last 20 years
  python3 revisions.py "Apollo 11" --list       # print stored index without fetching
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional

import requests

from config import WIKIPEDIA_API_URL, WIKIPEDIA_REQUEST_TIMEOUT
from wikipedia import SKIP_SECTIONS, strip_wikitext

logger = logging.getLogger(__name__)

REVISIONS_DIR   = "revisions"
DEFAULT_YEARS   = 10
_INTER_REV_DELAY = 0.5   # seconds between full-revision fetches (politeness)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_revision_history(title: str, years_back: int = DEFAULT_YEARS) -> str:
    """
    Fetch and store the sampled revision history for `title`.

    For each of the past `years_back` calendar years, stores:
      - The metadata of the last revision in that year
      - The cleaned section text of that revision (same format as fetch_article)

    Already-stored revisions are skipped, making re-runs safe and fast.

    Args:
        title:      Wikipedia article title (e.g. "Apollo 11").
        years_back: Number of past years to sample. Default: 10.

    Returns:
        Path to the written index.json file.
    """
    slug        = _slug(title)
    article_dir = os.path.join(REVISIONS_DIR, slug)
    os.makedirs(article_dir, exist_ok=True)

    now          = datetime.now(timezone.utc)
    current_year = now.year
    target_years = list(range(current_year - years_back + 1, current_year + 1))

    print(f"\nRevision history : {title!r}")
    print(f"Sampling         : last revision of each year, {target_years[0]}–{target_years[-1]}")
    print()

    sampled: List[dict] = []

    for year in reversed(target_years):   # newest first so progress is visible early
        meta = _fetch_year_revision_meta(title, year)

        if meta is None:
            print(f"  {year}  —  no revision found (article may not have existed)")
            continue

        revid     = int(meta["revid"])
        timestamp = meta["timestamp"]
        size      = meta.get("size", 0)
        user      = meta.get("user", "unknown")
        comment   = (meta.get("comment") or "")[:120]

        content_filename = f"{revid}_content.json"
        content_path     = os.path.join(article_dir, content_filename)

        if os.path.exists(content_path):
            # Already stored — load section count from disk, don't re-fetch
            cached       = _load_json(content_path)
            n_sections   = len(cached.get("sections", []))
            print(f"  {year}  revision {revid}  ({timestamp[:10]})  — already stored  ({n_sections} sections)")
        else:
            print(
                f"  {year}  revision {revid}  ({timestamp[:10]})  {size:>8,} bytes"
                f"  — fetching sections...",
                end="", flush=True,
            )
            sections   = _fetch_sections_for_revision(revid)
            n_sections = len(sections)

            content = {
                "revid":         revid,
                "article_title": title,
                "timestamp":     timestamp,
                "sections":      sections,
            }
            _write_json(content_path, content)
            print(f"  done  ({n_sections} sections)")
            time.sleep(_INTER_REV_DELAY)

        sampled.append({
            "revid":        revid,
            "year":         year,
            "timestamp":    timestamp,
            "user":         user,
            "comment":      comment,
            "size_bytes":   size,
            "content_file": content_filename,
            "n_sections":   n_sections,
        })

    # Write index (sorted oldest → newest for readability)
    sampled_sorted = sorted(sampled, key=lambda r: r["year"])
    index = {
        "article_title":     title,
        "article_slug":      slug,
        "sampling_strategy": "annual_last_revision",
        "years_covered":     [r["year"] for r in sampled_sorted],
        "total_revisions":   len(sampled_sorted),
        "fetched_at":        now.isoformat(),
        "revisions":         sampled_sorted,
    }
    index_path = os.path.join(article_dir, "index.json")
    _write_json(index_path, index)

    print()
    print(f"  Index written : {index_path}")
    print(f"  Revisions     : {len(sampled_sorted)}  ({target_years[0]}–{target_years[-1]})")
    if sampled_sorted:
        oldest = sampled_sorted[0]
        newest = sampled_sorted[-1]
        print(f"  Oldest        : {oldest['timestamp'][:10]}  (revision {oldest['revid']})")
        print(f"  Newest        : {newest['timestamp'][:10]}  (revision {newest['revid']})")
    print()
    return index_path


def load_revision_index(title: str) -> Optional[dict]:
    """
    Load the stored revision index for `title`.
    Returns None if no revision history has been fetched yet.
    """
    path = os.path.join(REVISIONS_DIR, _slug(title), "index.json")
    return _load_json(path) if os.path.isfile(path) else None


def load_revision_content(title: str, revid: int) -> Optional[dict]:
    """
    Load stored section content for a specific revision.
    Returns None if that revision file does not exist.
    """
    path = os.path.join(REVISIONS_DIR, _slug(title), f"{revid}_content.json")
    return _load_json(path) if os.path.isfile(path) else None


def print_index_summary(title: str) -> None:
    """
    Print a human-readable summary of the stored revision index.
    Useful for verifying what's been fetched before running Step 4.
    """
    index = load_revision_index(title)
    if index is None:
        print(f"No revision history stored for {title!r}.")
        print(f"Run:  python3 revisions.py {title!r}")
        return

    print(f"\nStored revision history for {title!r}")
    print(f"Sampling : {index.get('sampling_strategy', '?')}")
    print(f"Fetched  : {index.get('fetched_at', '?')[:19]}")
    print()
    print(f"  {'Year':<6}  {'Revision ID':<12}  {'Date':<12}  {'Sections':>8}  {'Size':>10}  Editor")
    print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*10}  {'─'*20}")
    for r in reversed(index.get("revisions", [])):
        print(
            f"  {r['year']:<6}  {r['revid']:<12}  {r['timestamp'][:10]:<12}"
            f"  {r['n_sections']:>8}  {r['size_bytes']:>10,}  {r.get('user','?')[:30]}"
        )
    print()


# ---------------------------------------------------------------------------
# Revision metadata fetching
# ---------------------------------------------------------------------------

def _fetch_year_revision_meta(title: str, year: int) -> Optional[dict]:
    """
    Fetch metadata for the last revision of `title` in `year`.

    Strategy: ask for the first revision going backwards from Jan 1 of year+1.
    This gives the most recent edit that existed at year-end.

    Returns None if the article had no revisions in that year.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop":   "revisions",
        "rvprop": "ids|timestamp|user|comment|size",
        "rvlimit": 1,
        "rvdir":  "older",
        "rvstart": f"{year + 1}-01-01T00:00:00Z",   # go backwards from here
        "format": "json",
    }
    try:
        data = _api_get(params)
    except Exception as exc:
        logger.error("API error fetching revision for %r year %d: %s", title, year, exc)
        return None

    for page in data.get("query", {}).get("pages", {}).values():
        if "missing" in page:
            return None
        revs = page.get("revisions", [])
        if not revs:
            return None
        rev = revs[0]
        # Confirm this revision actually falls within the target year
        if int(rev["timestamp"][:4]) != year:
            return None   # article didn't exist yet in this year
        return rev

    return None


# ---------------------------------------------------------------------------
# Section content fetching for a specific revision
# ---------------------------------------------------------------------------

def _fetch_sections_for_revision(revid: int) -> List[dict]:
    """
    Fetch and clean all encyclopedic section text for a historical revision.

    Uses oldid= to target the exact revision. Applies the same SKIP_SECTIONS
    filter and strip_wikitext cleaning as the current-article fetcher, so the
    output is ready for extract_assertions() without further processing.

    Returns list of {"title": str, "index": int, "text": str} dicts.
    """
    # Get section list for this revision
    params = {
        "action": "parse",
        "oldid":  revid,
        "prop":   "sections",
        "format": "json",
    }
    try:
        data = _api_get(params)
    except Exception as exc:
        logger.error("API error fetching section list for revid %d: %s", revid, exc)
        return []

    if "error" in data:
        logger.error(
            "API returned error for revid %d section list: %s",
            revid, data["error"].get("info", "?"),
        )
        return []

    sections_meta = data.get("parse", {}).get("sections", [])
    sections: List[dict] = []

    # Lead section (index 0) is not listed in sections_meta
    lead_text = _fetch_section_text(revid, 0)
    if lead_text:
        sections.append({"title": "Lead", "index": 0, "text": lead_text})

    for sec in sections_meta:
        sec_index = int(sec["index"])
        sec_title = sec["line"]

        if sec_title.lower().strip() in SKIP_SECTIONS:
            continue

        text = _fetch_section_text(revid, sec_index)
        if text:
            sections.append({"title": sec_title, "index": sec_index, "text": text})

    return sections


def _fetch_section_text(revid: int, section_index: int) -> str:
    """
    Fetch and clean wikitext for one section of a specific revision.
    Returns an empty string if the section is empty or the API fails.
    """
    params = {
        "action":  "parse",
        "oldid":   revid,
        "prop":    "wikitext",
        "section": section_index,
        "format":  "json",
    }
    try:
        data = _api_get(params)
    except Exception as exc:
        logger.warning(
            "API error for revid %d section %d: %s", revid, section_index, exc
        )
        return ""

    if "error" in data:
        logger.warning(
            "API error for revid %d section %d: %s",
            revid, section_index, data["error"].get("info", "?"),
        )
        return ""

    raw = data.get("parse", {}).get("wikitext", {}).get("*", "")
    return strip_wikitext(raw)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _api_get(params: dict) -> dict:
    """Wikipedia API GET request. Mirrors the function in wikipedia.py."""
    response = requests.get(
        WIKIPEDIA_API_URL,
        params=params,
        timeout=WIKIPEDIA_REQUEST_TIMEOUT,
        headers={"User-Agent": "AtomicAssertionAnalyzer/0.1 (research tool)"},
    )
    response.raise_for_status()
    return response.json()


def _slug(title: str) -> str:
    return title.lower().replace(" ", "_")


def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s: %(message)s",
    )

    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("Usage:")
        print("  python3 revisions.py <article_title>               # last 10 years")
        print("  python3 revisions.py <article_title> --years N     # last N years")
        print("  python3 revisions.py <article_title> --list        # show stored index")
        sys.exit(0)

    years      = DEFAULT_YEARS
    list_only  = False
    title_parts: List[str] = []

    i = 0
    while i < len(args):
        if args[i] == "--years" and i + 1 < len(args):
            try:
                years = int(args[i + 1])
            except ValueError:
                print(f"Invalid value for --years: {args[i + 1]!r}")
                sys.exit(1)
            i += 2
        elif args[i] == "--list":
            list_only = True
            i += 1
        else:
            title_parts.append(args[i])
            i += 1

    article_title = " ".join(title_parts)
    if not article_title:
        print("Please provide an article title.")
        sys.exit(1)

    if list_only:
        print_index_summary(article_title)
    else:
        fetch_revision_history(article_title, years_back=years)
        print("Next step: python3 compare.py (Step 4 — cross-revision assertion comparison)")
