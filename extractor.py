"""
Offline, deterministic assertion extractor using spaCy.

Replaces the LLM-based extractor. Schema and output format are unchanged —
only the extraction engine differs. This makes the system fully local:
no API keys, no network calls during extraction, reproducible output.

Approach:
  - Sentence segmentation via spaCy's dependency parser
  - SVO extraction from dependency parse tree
  - Claim type classification via lexical and structural patterns
  - Qualifier extraction from spaCy NER (DATE/TIME/GPE/LOC entities)
  - Confidence scoring from parse quality signals

Tradeoffs vs LLM extraction (for the project record):
  + Fully offline and free
  + Deterministic: same input always yields same output
  + Fast: ~200 sentences/second after model load
  - Lower accuracy on complex/interpretive sentences
  - More null subjects (passive constructions, pronoun references)
  - Cannot infer implied context across sentence boundaries
  - claim_type classification is pattern-based, not semantic

NOT responsible for:
  - Epistemic scoring or labeling (Step 5)
  - Embedding generation (Step 4)
  - Citation resolution (Step 6)
"""

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import spacy

from config import SPACY_MODEL
from schema import VALID_CLAIM_TYPES, Assertion, Qualifiers

logger = logging.getLogger(__name__)

# Load model once at module level — loading takes ~1 second.
try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    raise OSError(
        f"spaCy model {SPACY_MODEL!r} not found.\n"
        f"Run this once to install it:\n"
        f"    python3 -m spacy download {SPACY_MODEL}"
    )

EXTRACTION_MODEL = f"spacy:{SPACY_MODEL}"

# ---------------------------------------------------------------------------
# Pattern sets for claim type classification
# ---------------------------------------------------------------------------

# Patterns that strongly suggest a statistical claim.
# Checked against the sentence text before interpretive patterns.
_STATISTICAL_PATTERNS = [
    r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*'
    r'(?:feet|foot|ft|miles?|mi\b|km\b|kilometers?|metres?|meters?)',
    r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*'
    r'(?:kg\b|kilograms?|pounds?|lbs?\b|tons?|tonnes?)',
    r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*'
    r'(?:days?|hours?|minutes?|seconds?|weeks?|months?|years?)',
    r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:percent|per cent|%)',
    r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion|thousand)',
    r'\b(?:approximately|about|around|roughly|nearly|over|more than|less than)\s+\d+',
    r'\b\d{1,3}(?:,\d{3}){2,}\b',   # large formatted numbers, e.g. 1,234,567
    r'\b\d+\s*(?:°[CF]|degrees?)',   # temperatures
    r'\b\d+(?:\.\d+)?\s*(?:m/s|mph|km/h|knots?)',  # speeds
]

# Patterns that suggest an interpretive claim.
# Evaluated after statistical patterns (statistical takes priority).
_INTERPRETIVE_PATTERNS = [
    r'\b(?:is|are|was|were)\s+(?:often|widely|generally|commonly|universally|'
    r'traditionally|largely|frequently)?\s*'
    r'(?:considered|regarded|seen|viewed|thought|believed|known|recognized|'
    r'described|credited|celebrated|remembered|cited)\b',
    r'\b(?:represents?|marks?|signifies?|symbolizes?|served as|acted as|'
    r'functioned as|stands? as)\b',
    r'\b(?:led to|resulted in|caused|contributed to|paved the way|enabled|'
    r'allowed|prompted|triggered|precipitated)\b',
    r'\b(?:because|therefore|thus|hence|consequently|as a result|'
    r'for this reason|which is why)\b',
    r'\b(?:greatest|most significant|most important|most notable|'
    r'unprecedented|historic|landmark|milestone|seminal|pivotal|defining)\b',
    r'\b(?:is regarded|is seen|is viewed|is considered|is thought)\b',
    r'\b(?:came to be|came to represent|became a symbol)\b',
]

# Sentence prefixes that indicate navigation or meta-text, not assertions.
_SKIP_PREFIXES = (
    'see also', 'for the ', 'for a ', 'for other ', 'for more ',
    'main article:', 'further information:', 'note:', 'notes',
    'references', 'external links', 'see:', 'this article',
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_assertions(
    section_text: str,
    section_title: str,
    article_title: str,
    revision_id: str,
    position_offset: int = 0,
) -> List[Assertion]:
    """
    Extract atomic assertions from a section of cleaned Wikipedia text.

    Args:
        section_text:    Cleaned text from wikipedia.py (wikitext already stripped).
        section_title:   Section heading — stored as source_section metadata.
        article_title:   Wikipedia article title.
        revision_id:     Wikipedia revision ID at time of fetch.
        position_offset: Running assertion count before this section,
                         so source_position is globally ordered across the article.

    Returns:
        List of Assertion objects. Empty on empty input or parse failure.
    """
    if not section_text.strip():
        logger.debug("Skipping empty section: %r", section_title)
        return []

    try:
        doc = nlp(section_text)
    except Exception as exc:
        logger.error("spaCy parse failed for section %r: %s", section_title, exc)
        return []

    # Pre-compute noun chunks per sentence for efficient lookup
    all_chunks = list(doc.noun_chunks)

    assertions: List[Assertion] = []
    seen_ids: set = set()

    for sent in doc.sents:
        sent_chunks = [
            chunk for chunk in all_chunks
            if chunk.start >= sent.start and chunk.end <= sent.end
        ]
        assertion = _process_sentence(
            sent=sent,
            sent_chunks=sent_chunks,
            section_title=section_title,
            article_title=article_title,
            revision_id=revision_id,
            position=position_offset + len(assertions),
        )
        if assertion is None:
            continue
        if assertion.assertion_id in seen_ids:
            logger.debug("Duplicate assertion skipped: %r", assertion.claim_text[:80])
            continue
        seen_ids.add(assertion.assertion_id)
        assertions.append(assertion)

    return assertions


# ---------------------------------------------------------------------------
# Sentence processing
# ---------------------------------------------------------------------------

def _process_sentence(
    sent,
    sent_chunks: list,
    section_title: str,
    article_title: str,
    revision_id: str,
    position: int,
) -> Optional[Assertion]:
    """
    Convert one spaCy sentence into an Assertion, or return None if it
    should be skipped.
    """
    raw_text = sent.text.strip()

    if _should_skip(raw_text):
        return None

    # Extract [REF:...] citation markers before cleaning the text
    citations = re.findall(r'\[REF:[^\]]+\]', raw_text)
    claim_text = re.sub(r'\s*\[REF:[^\]]+\]', '', raw_text).strip()

    if not claim_text or len(claim_text.split()) < 4:
        return None

    subject, predicate, obj, is_passive, is_pronoun_subject = _extract_svo(sent, sent_chunks)

    if predicate is None:
        logger.debug(
            "No root verb found in section %r for sentence: %r",
            section_title, claim_text[:80],
        )
        predicate = "MISSING"

    temporal, spatial, conditional = _extract_qualifiers(sent, claim_text)
    claim_type = _classify_claim_type(claim_text, sent)
    confidence = _compute_confidence(
        subject=subject,
        predicate=predicate,
        obj=obj,
        is_passive=is_passive,
        sent=sent,
        claim_text=claim_text,
    )

    if subject is None:
        logger.debug(
            "Null subject (implied/passive) in section %r: %r",
            section_title, claim_text[:80],
        )
    if is_pronoun_subject:
        logger.debug(
            "Context-dependent subject (pronoun/demonstrative) in section %r: %r",
            section_title, claim_text[:80],
        )

    return Assertion(
        assertion_id=_generate_id(claim_text),
        claim_text=claim_text,
        claim_type=claim_type,
        subject=subject,
        predicate=predicate,
        object=obj,
        qualifiers=Qualifiers(
            temporal=temporal,
            spatial=spatial,
            conditional=conditional,
        ),
        source_article=article_title,
        source_section=section_title,
        source_position=position,
        extraction_confidence=confidence,
        extraction_model=EXTRACTION_MODEL,
        extracted_at=datetime.now(timezone.utc).isoformat(),
        wikipedia_revision_id=revision_id,
        citations_in_source=citations,
        context_dependent=is_pronoun_subject,
    )


# ---------------------------------------------------------------------------
# SVO extraction
# ---------------------------------------------------------------------------

def _extract_svo(sent, sent_chunks: list) -> Tuple[Optional[str], Optional[str], Optional[str], bool, bool]:
    """
    Extract (subject, predicate, object, is_passive, is_pronoun_subject).

    subject:             Grammatical subject (nsubj or nsubjpass). None if absent.
    predicate:           Root verb text. None if sentence has no verbal root.
    object:              Direct object, attribute, or adjectival complement. None if absent.
    is_passive:          True if the subject is nsubjpass (confidence penalty applies).
    is_pronoun_subject:  True if the subject is a pronoun (it, they, he, she, we)
                         or demonstrative (this, that, these, those) — meaning the
                         assertion's referent depends on surrounding context.
                         Stored as Assertion.context_dependent.
    """
    root = sent.root

    # Root must be a verb or auxiliary for us to confidently extract predicate
    predicate: Optional[str] = root.text if root.pos_ in ("VERB", "AUX") else None

    # --- Subject ---
    is_passive = False
    is_pronoun_subject = False
    subject_token = None
    for token in sent:
        if token.dep_ == "nsubj" and token.head == root:
            subject_token = token
            break
        if token.dep_ == "nsubjpass" and token.head == root:
            subject_token = token
            is_passive = True
            break

    subject: Optional[str] = None
    if subject_token is not None:
        subject = _get_phrase(subject_token, sent_chunks)
        # Pronoun: spaCy POS tag (it, they, he, she, we, one ...)
        if subject_token.pos_ == "PRON":
            is_pronoun_subject = True
        # Demonstrative determiners used as noun heads (this, that, these, those)
        elif subject_token.lower_ in ("this", "that", "these", "those", "such"):
            is_pronoun_subject = True

    # --- Object: dobj, attr (copula), acomp (adj complement) ---
    object_token = None
    for token in sent:
        if token.dep_ in ("dobj", "attr", "acomp") and token.head == root:
            object_token = token
            break

    obj: Optional[str] = None
    if object_token is not None:
        obj = _get_phrase(object_token, sent_chunks)

    return subject, predicate, obj, is_passive, is_pronoun_subject


def _get_phrase(token, sent_chunks: list) -> str:
    """
    Return the noun chunk containing this token if one exists,
    otherwise fall back to the token's text.
    Caps output at 80 chars to avoid unwieldy multi-clause spans.
    """
    for chunk in sent_chunks:
        if chunk.root.i == token.i:
            text = chunk.text.strip()
            return text[:80] if len(text) <= 80 else token.text
    return token.text


# ---------------------------------------------------------------------------
# Claim type classification
# ---------------------------------------------------------------------------

def _classify_claim_type(claim_text: str, sent) -> str:
    """
    Assign a claim type from the approved taxonomy:
      factual | interpretive | statistical

    Statistical is checked first (presence of quantitative patterns).
    Interpretive is checked second (hedging, causation, evaluation language).
    Factual is the default.

    This is pattern-based, not semantic — borderline cases will be imprecise.
    Accuracy improves with more specific patterns; the set can be extended.
    """
    for pattern in _STATISTICAL_PATTERNS:
        if re.search(pattern, claim_text, re.IGNORECASE):
            return "statistical"

    for pattern in _INTERPRETIVE_PATTERNS:
        if re.search(pattern, claim_text, re.IGNORECASE):
            return "interpretive"

    return "factual"


# ---------------------------------------------------------------------------
# Qualifier extraction
# ---------------------------------------------------------------------------

def _extract_qualifiers(
    sent, claim_text: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract temporal, spatial, and conditional qualifiers.

    Temporal and spatial are detected via spaCy NER.
    Conditional is detected via regex (pattern-based, not NLP).
    """
    temporal: Optional[str] = None
    spatial: Optional[str] = None

    # Collect NER entities within this sentence
    date_ents = [e for e in sent.ents if e.label_ in ("DATE", "TIME")]
    loc_ents = [e for e in sent.ents if e.label_ in ("GPE", "LOC", "FAC")]

    if date_ents:
        temporal = ", ".join(e.text for e in date_ents)
    if loc_ents:
        spatial = ", ".join(e.text for e in loc_ents)

    # Conditional: simple pattern check
    conditional: Optional[str] = None
    cond_match = re.search(
        r'\b(if|unless|provided that|assuming|given that|in the event that|should)\b',
        claim_text,
        re.IGNORECASE,
    )
    if cond_match:
        # Capture from the conditional marker to end of sentence (capped)
        conditional = claim_text[cond_match.start():cond_match.start() + 120].strip()

    return temporal, spatial, conditional


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _compute_confidence(
    subject: Optional[str],
    predicate: Optional[str],
    obj: Optional[str],
    is_passive: bool,
    sent,
    claim_text: str,
) -> float:
    """
    Heuristic confidence score (0.0–1.0) reflecting extraction quality.

    This is NOT a claim about epistemic stability — it reflects how cleanly
    spaCy decomposed the sentence into the assertion schema.

    Penalties applied:
      - No subject found:        -0.20  (implied or passive without agent)
      - Passive construction:    -0.10  (grammatical subject ≠ semantic agent)
      - Predicate missing:       -0.15  (non-verbal root)
      - No object found:         -0.05  (intransitive is fine; penalise lightly)
      - Long sentence (>35 tok): -0.10  (complex sentences decompose poorly)
      - Multiple root clauses:   -0.10  (compound sentence, one assertion captures only part)
    """
    score = 0.85  # baseline for a well-formed, parseable sentence

    if subject is None:
        score -= 0.20
    if is_passive:
        score -= 0.10
    if predicate is None or predicate == "MISSING":
        score -= 0.15
    if obj is None:
        score -= 0.05

    token_count = len([t for t in sent if not t.is_punct])
    if token_count > 35:
        score -= 0.10

    # Count clausal subjects as a proxy for compound sentence complexity
    clausal_subjects = sum(
        1 for t in sent if t.dep_ in ("csubj", "csubjpass", "advcl")
    )
    if clausal_subjects > 0:
        score -= 0.10

    return round(max(0.30, min(0.95, score)), 2)


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------

def _should_skip(text: str) -> bool:
    """
    Return True if this sentence should not become an assertion.

    Filters:
      - Fewer than 4 space-separated tokens
      - Navigation or meta-text prefixes
      - No alphabetic content
    """
    if len(text.split()) < 4:
        return True

    if not re.search(r'[a-zA-Z]', text):
        return True

    lower = text.lower().strip()
    for prefix in _SKIP_PREFIXES:
        if lower.startswith(prefix):
            return True

    return False


# ---------------------------------------------------------------------------
# Assertion ID
# ---------------------------------------------------------------------------

def _generate_id(claim_text: str) -> str:
    """
    Stable assertion ID: SHA-256 of normalised claim_text.

    Normalisation: lowercase → strip punctuation → collapse whitespace.
    This is the hash anchor. Embedding-based similarity for surviving
    minor rewording across revisions will be added in Step 4.
    """
    normalized = claim_text.lower()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()
