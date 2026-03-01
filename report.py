"""
Human-readable terminal report for extracted assertions.

Audience : researchers and content creators, not programmers.
Purpose  : route attention — show what was extracted, surface what needs scrutiny.

The JSON output file is unchanged. This report is printed on top of it.

Structure:
  1. Overview   — totals, type split, confidence snapshot
  2. Section Map — one line per article section with confidence bar and flag count
  3. Watch List  — specific assertions that deserve a closer look, with plain-English
                   explanations of why they were flagged

IMPORTANT: flags in the Watch List reflect extraction quality (how cleanly the
sentence was parsed), NOT epistemic stability (whether the claim is true or
contested). Epistemic labeling is implemented in a later step.
"""

from collections import Counter, OrderedDict
from typing import List, Optional, Tuple

from schema import Assertion

_W = 68                    # report width in characters
_BAR_WIDTH = 10            # width of confidence bar
_WATCH_LIST_CAP = 25       # max items shown in Watch List before truncating
_SECTION_NAME_WIDTH = 28   # max chars for section name column


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def print_report(
    assertions: List[Assertion],
    article_title: str,
    revision_id: str,
    output_path: str,
) -> None:
    """Print the full human-readable extraction report to stdout."""
    _rule("═")
    print(f"  EXTRACTION REPORT")
    print(f"  {article_title}  ·  revision {revision_id}")
    _rule("═")
    print()

    if not assertions:
        print("  No assertions were extracted.")
        print()
        _footer(output_path, 0)
        return

    _print_overview(assertions)
    print()
    _print_section_map(assertions)
    print()
    _print_watch_list(assertions)
    _footer(output_path, len(assertions))


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def _print_overview(assertions: List[Assertion]) -> None:
    total = len(assertions)
    n_sections = len({a.source_section for a in assertions})
    avg_conf = sum(a.extraction_confidence for a in assertions) / total
    type_counts = Counter(a.claim_type for a in assertions)
    n_flagged = sum(1 for a in assertions if _flag_reasons(a))

    print(f"  {total} claims  ·  {n_sections} sections  ·  avg confidence {avg_conf:.2f}")
    print()

    factual = type_counts.get("factual", 0)
    interpretive = type_counts.get("interpretive", 0)
    statistical = type_counts.get("statistical", 0)
    print(f"  Claim types")
    _type_row("Factual",      factual,      total)
    _type_row("Interpretive", interpretive, total)
    _type_row("Statistical",  statistical,  total)
    print()

    if n_flagged:
        print(f"  ⚠  {n_flagged} claims flagged  —  see Watch List below")
    else:
        print(f"  ✓  All claims passed structural checks")


def _type_row(label: str, count: int, total: int) -> None:
    if total == 0:
        return
    pct = count / total * 100
    bar = "▪" * int(pct / 5)            # one ▪ per 5%
    print(f"    {label:<14}  {count:>4}  ({pct:4.0f}%)  {bar}")


# ---------------------------------------------------------------------------
# Section map
# ---------------------------------------------------------------------------

def _print_section_map(assertions: List[Assertion]) -> None:
    _rule("─")
    print("  SECTIONS")
    _rule("─")
    print()

    # Group by section in order of first appearance
    sections: OrderedDict = OrderedDict()
    for a in assertions:
        sections.setdefault(a.source_section, []).append(a)

    # Column header
    name_col = _SECTION_NAME_WIDTH
    print(f"  {'Section':<{name_col}}  {'Claims':>6}  {'Confidence':<{_BAR_WIDTH + 5}}  Notes")
    print(f"  {'─' * name_col}  {'─' * 6}  {'─' * (_BAR_WIDTH + 5)}  {'─' * 10}")

    for section_name, section_assertions in sections.items():
        count = len(section_assertions)
        avg = sum(a.extraction_confidence for a in section_assertions) / count
        n_flagged = sum(1 for a in section_assertions if _flag_reasons(a))

        bar = _conf_bar(avg)
        display_name = _truncate(section_name, name_col)
        flag_str = f"⚠ {n_flagged} flagged" if n_flagged else ""

        print(
            f"  {display_name:<{name_col}}  {count:>6}  {bar} {avg:.2f}  {flag_str}"
        )

    print()


# ---------------------------------------------------------------------------
# Watch list
# ---------------------------------------------------------------------------

def _print_watch_list(assertions: List[Assertion]) -> None:
    flagged: List[Tuple[Assertion, List[str]]] = [
        (a, _flag_reasons(a)) for a in assertions if _flag_reasons(a)
    ]

    _rule("─")

    if not flagged:
        print("  ✓  WATCH LIST  —  no claims flagged")
        _rule("─")
        print()
        return

    print(f"  ⚠  WATCH LIST  —  {len(flagged)} claims that deserve a closer look")
    _rule("─")
    print()
    print("  These claims had structural issues during extraction. They may represent")
    print("  complex sentences, passive voice, or interpretive judgments without a")
    print("  cited source. Review these before using them in your content.")
    print()
    print("  Note: flags here reflect how the text was parsed — not whether")
    print("  the claim is factually correct. Reliability scoring comes later.")
    print()

    # Sort: most flags first, then by confidence ascending (least confident first)
    flagged.sort(key=lambda x: (-len(x[1]), x[0].extraction_confidence))

    shown = flagged[:_WATCH_LIST_CAP]
    for i, (assertion, reasons) in enumerate(shown, 1):
        _print_watch_entry(i, assertion, reasons)

    if len(flagged) > _WATCH_LIST_CAP:
        remainder = len(flagged) - _WATCH_LIST_CAP
        print(f"  ... and {remainder} more flagged claims in the JSON output.")
        print()

    _rule("─")
    print()


def _print_watch_entry(
    index: int, assertion: Assertion, reasons: List[str]
) -> None:
    label = _type_label(assertion.claim_type)
    conf = assertion.extraction_confidence
    claim = _truncate(assertion.claim_text, 85)

    print(f"  {index:>2}.  [{label}  conf {conf:.2f}]")
    print(f'        "{claim}"')
    print(f"         Section : {assertion.source_section}")
    for reason in reasons:
        print(f"         Flag    : {reason}")
    print()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

def _footer(output_path: str, total: int) -> None:
    _rule("─")
    print(f"  Full output : {output_path}")
    if total:
        print(f"  To inspect  : open the JSON file or re-run with a different article title")
    _rule("─")
    print()


# ---------------------------------------------------------------------------
# Flag logic
# ---------------------------------------------------------------------------

def _flag_reasons(assertion: Assertion) -> List[str]:
    """
    Return a list of human-readable flag reasons for this assertion.
    Empty list means no flags — the assertion passed all checks.

    Flag conditions:
      1. Low extraction confidence   (<0.70)
      2. Subject not identified      (None)
      3. Interpretive, no citations  (opinion without sourcing signal)
      4. Predicate missing           (parse failure)
      5. Context-dependent subject   (pronoun or demonstrative)
    """
    reasons = []

    if assertion.extraction_confidence < 0.70:
        reasons.append(
            "Complex sentence — the breakdown into subject/predicate/object "
            "may be imprecise"
        )

    if assertion.subject is None:
        reasons.append(
            "Who or what this claim is about wasn't clearly identified "
            "(passive voice or pronoun reference)"
        )

    if (
        assertion.claim_type == "interpretive"
        and not assertion.citations_in_source
    ):
        reasons.append(
            "Judgment or opinion — no cited source was found in the surrounding text"
        )

    if assertion.predicate == "MISSING":
        reasons.append(
            "The action or relationship in this sentence wasn't identified by the parser"
        )

    if getattr(assertion, "context_dependent", False):
        reasons.append(
            "Subject is a pronoun or demonstrative (it, they, this, that...) — "
            "meaning depends on surrounding sentences to identify the referent"
        )

    return reasons


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

def _conf_bar(score: float, width: int = _BAR_WIDTH) -> str:
    """Filled/empty block bar representing a confidence score."""
    filled = round(score * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _type_label(claim_type: str) -> str:
    """Fixed-width uppercase label for a claim type."""
    labels = {
        "factual":       "FACTUAL  ",
        "interpretive":  "INTERPRET",
        "statistical":   "STATISTIC",
    }
    return labels.get(claim_type, claim_type.upper()[:9].ljust(9))


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _rule(char: str = "─") -> None:
    print("  " + char * (_W - 2))
