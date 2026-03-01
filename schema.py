"""
Assertion schema — single source of truth for the data shape.
No logic lives here. All construction and validation is in extractor.py.

NOTE on assertion_id:
  Currently a SHA-256 hash of normalized claim_text (the hash anchor).
  Embedding-based similarity matching for cross-revision deduplication
  will be added in Step 4, when revision comparison is implemented.
"""

from dataclasses import dataclass
from typing import Optional

VALID_CLAIM_TYPES = {"factual", "interpretive", "statistical"}


@dataclass
class Qualifiers:
    temporal: Optional[str] = None
    spatial: Optional[str] = None
    conditional: Optional[str] = None


@dataclass
class Assertion:
    assertion_id: str           # SHA-256 of normalized claim_text (Step 4: + embedding anchor)
    claim_text: str             # Full natural-language assertion
    claim_type: str             # factual | interpretive | statistical
    subject: Optional[str]      # Grammatical subject (None if implied/passive)
    predicate: str              # Verb or relation
    object: Optional[str]       # Grammatical object or attribute
    qualifiers: Qualifiers      # Temporal, spatial, conditional qualifiers

    source_article: str         # Wikipedia article title
    source_section: str         # Section heading within the article
    source_position: int        # Assertion's ordinal index in the full article output

    extraction_confidence: float  # LLM's confidence in decomposition (0.0–1.0)
    extraction_model: str         # Model used for extraction
    extracted_at: str             # ISO 8601 UTC timestamp of extraction

    wikipedia_revision_id: str    # Wikipedia revision ID at time of fetch
    citations_in_source: list     # [REF:name] placeholders captured from wikitext
                                  # Resolved lazily in Step 6 (scoring layer)

    # Context-dependency flag (set by extractor, not by human)
    # True when the grammatical subject is a pronoun or demonstrative (it, they,
    # this, that, these, those) — meaning the assertion depends on surrounding
    # sentences for its referent.  Kept in the full output but excluded from
    # Creator Safe Starting Points and downstream high-confidence filters.
    context_dependent: bool = False
