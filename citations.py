"""
Step 6 — Citation Resolution.

Resolves [REF:name] placeholders stored in assertions' citations_in_source
field back to actual reference text, by fetching the raw wikitext of the
article's newest revision and extracting all named <ref> definitions.

Placeholder taxonomy (set by wikipedia.py → _ref_to_placeholder):
  [REF:name]      — named ref (<ref name="name">...</ref>) — RESOLVABLE
  [REF:fragment…] — anonymous ref, first 50 chars of content — text already present
  [REF]           — self-closing with no name or content — UNRESOLVABLE

Resolution strategy:
  One Wikipedia API call fetches full raw wikitext for the newest stored revision.
  All named ref definitions (<ref name="X">...</ref>) are extracted; first definition
  wins (subsequent reuses are self-closing <ref name="X" />). Ref content is cleaned
  with a cite-template–aware formatter so output is human-readable, not wikitext.

Only placeholder names that actually appear in stored revision content are written
to refs_map — unused definitions are counted but not stored.

Storage layout:
  citations/{slug}/refs_map.json

Usage:
  python3 citations.py "Apollo 11"          # resolve citations
  python3 citations.py "Apollo 11" --list   # print stored refs_map
  python3 citations.py "Apollo 11" --force  # re-fetch even if already stored
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import requests

from config import WIKIPEDIA_API_URL, WIKIPEDIA_REQUEST_TIMEOUT
from revisions import load_revision_index

logger = logging.getLogger(__name__)

CITATIONS_DIR = "citations"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_citations(title: str, force: bool = False) -> str:
    """
    Resolve [REF:name] placeholders for all stored revisions of title.

    Fetches the full raw wikitext of the newest stored revision from Wikipedia,
    extracts named ref definitions, cleans them to readable text, then reports
    how many placeholders appearing in stored content were resolved.

    Args:
        title: Wikipedia article title (e.g. "Apollo 11").
        force: Re-fetch even if refs_map.json already exists.

    Returns:
        Path to the written refs_map.json file.
    """
    out_path = _refs_path(title)
    if os.path.isfile(out_path) and not force:
        print(f"\nCitation resolution : {title!r}")
        print(f"  Already resolved — use --force to re-fetch.")
        print(f"  Stored : {out_path}")
        print()
        return out_path

    index = load_revision_index(title)
    if index is None:
        raise FileNotFoundError(
            f"No revision index for {title!r}. Run:  python3 revisions.py {title!r}"
        )

    revisions = index.get("revisions", [])
    if not revisions:
        raise ValueError(f"No revisions stored for {title!r}.")

    newest_rev = revisions[-1]
    revid = newest_rev["revid"]
    year  = newest_rev["year"]

    print(f"\nCitation resolution : {title!r}")
    print(f"  Revision          : {revid}  ({year})")

    # Collect all [REF:name] placeholder names from every stored content file
    print(f"  Scanning stored revisions for placeholders ...", end="", flush=True)
    placeholder_names = _collect_placeholder_names(title)
    print(f"  {len(placeholder_names)} unique names found")

    # Fetch full raw wikitext for the newest revision
    print(f"  Fetching raw wikitext (revid {revid}) ...", end="", flush=True)
    raw = _fetch_raw_wikitext(revid)
    print(f"  {len(raw):,} chars")

    # Extract all named ref definitions from raw wikitext
    all_defs = _extract_ref_definitions(raw)

    # Resolve only the names that appear in stored placeholder text
    refs_map: Dict[str, str] = {}
    unresolved: List[str] = []

    for name in sorted(placeholder_names):
        if name in all_defs:
            refs_map[name] = all_defs[name]
        else:
            unresolved.append(name)

    n_resolved   = len(refs_map)
    n_total      = len(placeholder_names)
    coverage_pct = (n_resolved / n_total * 100) if n_total else 0.0

    record = {
        "article":      title,
        "computed_at":  datetime.now(timezone.utc).isoformat(),
        "revision":     {"revid": revid, "year": year},
        "stats": {
            "named_defs_in_article":        len(all_defs),
            "placeholder_names_in_stored_text": n_total,
            "resolved":                     n_resolved,
            "unresolved":                   len(unresolved),
            "coverage_pct":                 round(coverage_pct, 1),
        },
        "refs_map": refs_map,
    }

    out_path = _write_refs_map(record, title)
    _print_summary(record, unresolved)
    print(f"  Output : {out_path}")
    print()
    return out_path


def load_refs_map(title: str) -> Optional[dict]:
    """Load a stored refs_map record, or None if not yet resolved."""
    path = _refs_path(title)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_refs_summary(title: str) -> None:
    """Print the stored refs_map summary. Used with --list."""
    record = load_refs_map(title)
    if record is None:
        print(f"No citation data for {title!r}.")
        print(f"Run:  python3 citations.py {title!r}")
        return
    _print_summary(record, [])


# ---------------------------------------------------------------------------
# Placeholder collection
# ---------------------------------------------------------------------------

def _collect_placeholder_names(title: str) -> Set[str]:
    """
    Scan every stored *_content.json file for [REF:name] patterns.

    Returns the set of ref names (identifier after the colon) that appear
    in at least one stored revision's section text.

    Only named refs are collected — [REF] (bare, no colon) and anonymous
    fragment refs (containing spaces or very long content) are excluded since
    they cannot be looked up by name.
    """
    slug        = _slug(title)
    article_dir = os.path.join("revisions", slug)
    names: Set[str] = set()

    if not os.path.isdir(article_dir):
        return names

    # Pattern: [REF:name] where name is a short, non-whitespace identifier
    # (named refs are typically 5-40 chars, no internal spaces)
    placeholder_pattern = re.compile(r'\[REF:([^\]\s]{1,80})\]')

    for filename in os.listdir(article_dir):
        if not filename.endswith("_content.json"):
            continue
        path = os.path.join(article_dir, filename)
        try:
            with open(path, encoding="utf-8") as f:
                content = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue

        for section in content.get("sections", []):
            text = section.get("text", "")
            for match in placeholder_pattern.finditer(text):
                names.add(match.group(1))

    return names


# ---------------------------------------------------------------------------
# Wikipedia API — raw wikitext fetch
# ---------------------------------------------------------------------------

def _fetch_raw_wikitext(revid: int) -> str:
    """
    Fetch the full raw wikitext for a specific revision.

    Uses action=query with rvslots=main (current MediaWiki content model).
    Returns an empty string on any error.
    """
    params = {
        "action":   "query",
        "revids":   revid,
        "prop":     "revisions",
        "rvprop":   "content",
        "rvslots":  "main",
        "format":   "json",
    }
    try:
        data = _api_get(params)
    except Exception as exc:
        raise RuntimeError(f"API error fetching wikitext for revid {revid}: {exc}") from exc

    for page in data.get("query", {}).get("pages", {}).values():
        revs = page.get("revisions", [])
        if not revs:
            return ""
        slot = revs[0].get("slots", {}).get("main", {})
        return slot.get("*", "")

    return ""


# ---------------------------------------------------------------------------
# Ref definition extraction
# ---------------------------------------------------------------------------

def _extract_ref_definitions(raw: str) -> Dict[str, str]:
    """
    Extract all named ref definitions from raw wikitext.

    Pattern matches:
      <ref name="identifier">content</ref>
      <ref name='identifier'>content</ref>
      <ref name=identifier>content</ref>   (unquoted)

    First definition wins — subsequent <ref name="X" /> uses are self-closing
    and carry no content. Returns {name: cleaned_content}.
    """
    pattern = re.compile(
        r'<ref\s+name=["\']?([^"\'>/\s]+)["\']?\s*>(.*?)</ref>',
        re.DOTALL | re.IGNORECASE,
    )
    result: Dict[str, str] = {}
    for name, content in pattern.findall(raw):
        if name not in result:
            cleaned = _clean_ref_content(content)
            if cleaned:
                result[name] = cleaned
    return result


# ---------------------------------------------------------------------------
# Ref content cleaning
# ---------------------------------------------------------------------------

def _clean_ref_content(content: str) -> str:
    """
    Convert raw ref content to readable plain text.

    Wikipedia refs are usually {{cite book/web/journal|...}} templates.
    For those, extract key bibliographic fields. For plain text or external
    links, apply basic wikitext stripping.

    Capped at 300 chars to keep output scannable.
    """
    content = content.strip()
    if not content:
        return ""

    # Cite template: extract and format key fields
    if re.search(r'\{\{[Cc]ite', content):
        formatted = _format_cite_template(content)
        if formatted:
            return formatted[:300]

    # Fallback: basic wikitext strip
    # Remove nested templates
    text = _strip_templates(content)
    # Wikilinks
    text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    # External links
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)
    # Remaining HTML / bold / italic
    text = re.sub(r"<[^>]+>|'{2,}", '', text)
    # Whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:300]


def _format_cite_template(content: str) -> str:
    """
    Extract readable bibliographic info from a {{cite ...}} template.

    Parses pipe-delimited fields and assembles:
      Author(s). "Title." Publisher/Journal, Year.  [URL if present]

    Returns empty string if no useful fields can be found.
    """
    # Extract field=value pairs (handles |field = value with spaces)
    fields: Dict[str, str] = {}
    for key, value in re.findall(r'\|\s*(\w[\w\d_-]*)\s*=\s*([^|{}]+)', content):
        key   = key.strip().lower()
        value = value.strip()
        # Strip inner wikilinks and markup from field values
        value = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', value)
        value = re.sub(r'\[\[([^\]]+)\]\]', r'\1', value)
        value = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', value)
        value = re.sub(r"<[^>]+>|'{2,}", '', value)
        value = re.sub(r'\s+', ' ', value).strip()
        if value:
            fields[key] = value

    parts: List[str] = []

    # --- Authors ---
    # Support up to 4 named authors (Wikipedia convention: last1/first1 … last4/first4)
    author_parts: List[str] = []
    for n in ("", "1", "2", "3", "4"):
        last  = fields.get(f"last{n}") or (fields.get("last") if n in ("", "1") else None)
        first = fields.get(f"first{n}") or (fields.get("first") if n in ("", "1") else None)
        if last and first:
            author_parts.append(f"{last}, {first}")
        elif last:
            author_parts.append(last)
        elif not last and not first:
            break   # stop at first missing pair
    # Fallback: author / author1 / authors fields
    if not author_parts:
        for key in ("author", "author1", "authors", "editor", "editor1", "editors"):
            if fields.get(key):
                author_parts.append(fields[key])
                break
    if author_parts:
        parts.append(", ".join(author_parts) + ".")

    # --- Title ---
    title = fields.get("title") or fields.get("chapter")
    if title:
        parts.append(f'"{title}."')

    # --- Venue: publisher / journal / work / website / newspaper ---
    venue_keys = ("publisher", "journal", "work", "website", "newspaper", "magazine", "series")
    venue = next((fields[k] for k in venue_keys if fields.get(k)), None)

    # --- Year / date ---
    year = fields.get("year") or fields.get("date")
    # Keep only the 4-digit year if date is a full date
    if year:
        year_match = re.search(r'\b(1[89]\d{2}|20\d{2})\b', year)
        year = year_match.group(1) if year_match else year

    if venue and year:
        parts.append(f"{venue}, {year}.")
    elif venue:
        parts.append(f"{venue}.")
    elif year:
        parts.append(f"{year}.")

    # --- URL (only if no other content) ---
    if not parts and fields.get("url"):
        parts.append(fields["url"])

    if not parts:
        return ""

    return " ".join(parts)


def _strip_templates(text: str) -> str:
    """Remove {{...}} template markup via character-level scan (handles nesting)."""
    result: List[str] = []
    depth = 0
    i = 0
    while i < len(text):
        if text[i : i + 2] == "{{":
            depth += 1
            i += 2
        elif text[i : i + 2] == "}}":
            depth = max(0, depth - 1)
            i += 2
        else:
            if depth == 0:
                result.append(text[i])
            i += 1
    return "".join(result)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_summary(record: dict, unresolved: List[str]) -> None:
    s    = record["stats"]
    rmap = record.get("refs_map", {})

    print(f"  Named ref defs    : {s['named_defs_in_article']} found in full article wikitext")
    print(f"  Placeholder names : {s['placeholder_names_in_stored_text']} unique names across stored revisions")
    print(f"  Resolved          : {s['resolved']} / {s['placeholder_names_in_stored_text']}  ({s['coverage_pct']:.1f}%)")
    print(f"  Unresolved        : {s['unresolved']}")

    if unresolved:
        print(f"\n  Unresolved names")
        for name in sorted(unresolved):
            print(f"    [{name}]  — no definition found in article wikitext")

    if rmap:
        sample = list(rmap.items())[:5]
        print(f"\n  Sample resolved refs")
        for name, text in sample:
            short = text[:80] + "…" if len(text) > 80 else text
            print(f"    [{name}]")
            print(f"      {short}")

    print()


def _write_refs_map(record: dict, title: str) -> str:
    out_dir = os.path.join(CITATIONS_DIR, _slug(title))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "refs_map.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path


def _refs_path(title: str) -> str:
    return os.path.join(CITATIONS_DIR, _slug(title), "refs_map.json")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _api_get(params: dict) -> dict:
    """Wikipedia API GET request."""
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
        print("  python3 citations.py <article_title>          # resolve citations")
        print("  python3 citations.py <article_title> --list   # show stored refs_map")
        print("  python3 citations.py <article_title> --force  # re-fetch even if stored")
        sys.exit(0)

    list_only   = "--list"  in args
    force       = "--force" in args
    title_parts = [a for a in args if a not in ("--list", "--force")]
    article_title = " ".join(title_parts)

    if not article_title:
        print("Please provide an article title.")
        sys.exit(1)

    if list_only:
        print_refs_summary(article_title)
    else:
        resolve_citations(article_title, force=force)
        print("Next step: python3 reporter.py (generate shareable report)")
