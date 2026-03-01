"""
Wikipedia article fetcher.

Responsibilities:
  1. Fetch the article's section list and revision ID.
  2. Fetch wikitext for each section (lead section + named sections).
  3. Strip wikitext markup, converting <ref> tags to [REF:identifier]
     placeholders so the LLM can attribute citations to specific assertions.

Citation resolution is intentionally deferred (lazy) — the [REF:...] tokens
are passed through to Assertion.citations_in_source and resolved in Step 6.
"""

import html as _html
import logging
import re

import requests

from config import WIKIPEDIA_API_URL, WIKIPEDIA_REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

# Sections that are structural/navigational rather than encyclopedic content.
# Skipped entirely at fetch time — do not attempt to strip or extract from them.
# Exact match on lowercased section title (after stripping whitespace).
SKIP_SECTIONS: frozenset = frozenset({
    "see also",
    "references",
    "external links",
    "notes",
    "further reading",
    "bibliography",
    "multimedia",
    "citations",
    "footnotes",
    "sources",
    "works cited",
    "films and documentaries",
    "books",
    "periodicals",
    "journals",
    "other websites",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_article(title: str) -> dict:
    """
    Fetch a Wikipedia article broken into cleaned sections.

    Returns:
        {
            "title": str,
            "revision_id": str,
            "sections": [{"title": str, "index": int, "text": str}]
        }

    Raises:
        ValueError: if the Wikipedia API returns an error for this title.
        requests.HTTPError: on network-level failures.
    """
    revision_id, sections_meta = _fetch_section_list(title)

    sections = []

    # Lead section (index 0) is not included in sections_meta
    lead_text = _fetch_section_text(title, 0)
    if lead_text:
        sections.append({"title": "Lead", "index": 0, "text": lead_text})
    else:
        logger.warning("Lead section returned empty text for %r", title)

    for sec in sections_meta:
        sec_index = int(sec["index"])
        sec_title = sec["line"]  # plain-text heading

        if sec_title.lower().strip() in SKIP_SECTIONS:
            logger.info("Skipping non-encyclopedic section: %r", sec_title)
            continue

        text = _fetch_section_text(title, sec_index)
        if not text:
            logger.warning(
                "Section %r (index %d) returned empty text after stripping",
                sec_title, sec_index,
            )
            continue
        sections.append({"title": sec_title, "index": sec_index, "text": text})

    logger.info(
        "Fetched %d sections for %r (revision %s)", len(sections), title, revision_id
    )
    return {"title": title, "revision_id": revision_id, "sections": sections}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_section_list(title: str) -> tuple:
    """
    Fetch the article's table of contents and revision ID.

    revid is requested explicitly via prop=sections|revid.
    If absent (cached response or API quirk), falls back to
    action=query&prop=revisions which is always reliable.

    Returns:
        (revision_id: str, sections: list[dict])
    """
    params = {
        "action": "parse",
        "page": title,
        "prop": "sections|revid",   # revid must be requested explicitly
        "format": "json",
    }
    data = _api_get(params)
    if "error" in data:
        raise ValueError(
            f"Wikipedia API error for {title!r}: {data['error']['info']}"
        )

    parse = data["parse"]
    logger.debug("parse keys returned: %s", list(parse.keys()))

    raw_revid = parse.get("revid")
    if raw_revid:
        revision_id = str(raw_revid)
    else:
        logger.warning(
            "revid absent from parse response for %r (keys: %s). "
            "Falling back to action=query.",
            title, list(parse.keys()),
        )
        revision_id = _fetch_revision_id_via_query(title)

    sections = parse["sections"]
    return revision_id, sections


def _fetch_revision_id_via_query(title: str) -> str:
    """
    Fallback: fetch the current revision ID via action=query.
    Returns a placeholder string if the query also fails, so the
    pipeline can continue rather than abort.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "ids",
        "format": "json",
    }
    try:
        data = _api_get(params)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            revisions = page.get("revisions", [])
            if revisions:
                revid = revisions[0].get("revid")
                if revid:
                    logger.info("Revision ID obtained via fallback query: %s", revid)
                    return str(revid)
    except Exception as exc:
        logger.error("Fallback revision ID query failed: %s", exc)

    logger.warning("Could not obtain revision ID for %r; using 'unknown'.", title)
    return "unknown"


def _fetch_section_text(title: str, section_index: int) -> str:
    """
    Fetch and clean the wikitext for a single section.
    Returns an empty string if the section is empty or an API error occurs.
    """
    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "section": section_index,
        "format": "json",
    }
    data = _api_get(params)
    if "error" in data:
        logger.error(
            "API error fetching section %d of %r: %s",
            section_index, title, data["error"]["info"],
        )
        return ""

    raw = data["parse"]["wikitext"]["*"]
    return strip_wikitext(raw)


def _api_get(params: dict) -> dict:
    """Thin wrapper around requests.get for the Wikipedia API."""
    response = requests.get(
        WIKIPEDIA_API_URL,
        params=params,
        timeout=WIKIPEDIA_REQUEST_TIMEOUT,
        headers={"User-Agent": "AtomicAssertionAnalyzer/0.1 (research tool)"},
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Wikitext stripping
# ---------------------------------------------------------------------------

def strip_wikitext(text: str) -> str:
    """
    Convert raw wikitext into clean prose suitable for LLM processing.

    Key design decision:
      <ref> tags are NOT deleted — they are converted to [REF:identifier]
      placeholders so the LLM can attribute citation markers to specific
      assertions. Example: "Apollo 11 launched on July 16 [REF:nasa_ref]..."

    Handles:
      - <ref> / <ref name="..."> / self-closing <ref .../> → [REF:...]
      - {{templates}} (arbitrary nesting)
      - [[wikilinks]] and [[target|display]] links
      - External links [http://... text]
      - File/Image embeds
      - Bold/italic markers (''' / '')
      - Section headers (== ... ==)
      - HTML comments and remaining HTML tags
      - Wiki table markup
    """
    # 0. Decode HTML entities (&nbsp; → space, &amp; → &, &lt; → <, etc.)
    #    Must happen before regex passes so entities don't confuse later patterns.
    text = _html.unescape(text)
    text = text.replace('\u00a0', ' ')   # non-breaking space (U+00A0) → plain space

    # 1. Convert <ref> tags to [REF:identifier] before anything else
    text = re.sub(
        r'<ref(?:\s+name=["\']?([^"\'>/\s]+)["\']?)?\s*(?:/>|>(.*?)</ref>)',
        _ref_to_placeholder,
        text,
        flags=re.DOTALL,
    )

    # 2. HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # 3. File / Image / Media embeds (must precede general wikilink unwrap)
    text = re.sub(
        r'\[\[(?:File|Image|Fichier|Media):[^\]]*\]\]',
        '',
        text,
        flags=re.IGNORECASE,
    )

    # 4. Wikilinks: [[target|display]] → display, [[target]] → target
    text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)

    # 5. External links: [http://... display] → display, bare → removed
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)

    # 6. Templates — character-level scan handles arbitrary nesting
    text = _remove_templates(text)

    # 7. Bold / italic
    text = re.sub(r"'{2,3}", '', text)

    # 8. Section headers
    text = re.sub(r'=+\s*[^=\n]+\s*=+', '', text)

    # 9. Remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # 10. Wiki table markup (lines beginning with |, !, {|, |})
    text = re.sub(r'^\s*[\|!{][^\n]*', '', text, flags=re.MULTILINE)

    # 11. Whitespace normalisation
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()

    return text


def _ref_to_placeholder(match: re.Match) -> str:
    """
    Convert a <ref> regex match to a [REF:identifier] string.
    Named refs use the name; anonymous refs use a fragment of their content.
    """
    name = match.group(1)
    content = match.group(2)
    if name:
        return f' [REF:{name}]'
    if content:
        # Truncate and sanitise so the placeholder is readable
        fragment = content.strip()[:50].replace('\n', ' ').replace(']', ')')
        return f' [REF:{fragment}]'
    return ' [REF]'


def _remove_templates(text: str) -> str:
    """
    Remove all {{...}} template calls from wikitext, handling arbitrary nesting.
    A simple regex loop cannot handle nesting, so we scan character by character.
    """
    result = []
    depth = 0
    i = 0
    while i < len(text):
        if text[i:i + 2] == '{{':
            depth += 1
            i += 2
        elif text[i:i + 2] == '}}':
            if depth > 0:
                depth -= 1
            i += 2
        elif depth == 0:
            result.append(text[i])
            i += 1
        else:
            i += 1
    return ''.join(result)
