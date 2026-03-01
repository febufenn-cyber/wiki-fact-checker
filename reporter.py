"""
Step 7 — Final Stability-Aware Report.

Reads:
  stability/{slug}/stability.json  — scored assertions (from Step 5)
  citations/{slug}/refs_map.json   — resolved source text (from Step 6, optional)

Writes:
  reports/{slug}_{year}_stability_report.txt

Sections:
  1. ARTICLE SUMMARY             — totals and stability counts
  2. CREATOR SAFE STARTING POINTS — Almost Certain factual claims + inline sources
  3. INVESTIGATE THESE           — Doubtful then Uncertain claims, with plain-English reasons
  4. SECTION RELIABILITY         — stability distribution per section
  5. CITATION GAPS               — claims with no source or an unresolvable reference

Usage:
  python3 reporter.py "Apollo 11"
  python3 reporter.py "Apollo 11" --print   # also print to terminal
"""

import os
import re
import sys
import textwrap
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from citations import load_refs_map
from stability import load_stability

REPORTS_DIR = "reports"

_W     = 72
_HEAVY = "=" * _W
_LIGHT = "-" * _W
_DOTS  = "·" * _W

_CLAIM_WRAP  = 66
_SECTION_W   = 28
_MAX_SAFE    = 25
_MAX_DOUBTFUL  = 15
_MAX_UNCERTAIN = 10
_MAX_GAPS    = 20


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_report(title: str, print_to_terminal: bool = False) -> str:
    """
    Generate the stability-aware plain-text report for an article.

    Loads stability.json and refs_map.json for the given title, then
    assembles a creator-facing report with five sections.

    Args:
        title:             Wikipedia article title (e.g. "Apollo 11").
        print_to_terminal: If True, also print the report to stdout.

    Returns:
        Path to the written report file in reports/.
    """
    stability = load_stability(title)
    if stability is None:
        raise FileNotFoundError(
            f"No stability data for {title!r}.\n"
            f"Run:  python3 stability.py {title!r}"
        )

    refs_record = load_refs_map(title)
    refs_map    = refs_record.get("refs_map", {}) if refs_record else {}

    assertions    = stability["assertions"]   # sorted by stability_score desc
    r             = stability["revision_range"]
    n_pairs       = stability["total_comparison_pairs"]
    article_title = stability["article"]
    year_range    = f"{r['oldest']['year']}–{r['newest']['year']}"
    generated_at  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: List[str] = []

    lines += [
        _HEAVY,
        "EPISTEMIC STABILITY REPORT",
        f"Article   : {article_title}",
        f"Revisions : {year_range}  ({n_pairs} annual editorial reviews)",
        f"Generated : {generated_at}",
        _HEAVY,
        "",
    ]

    _write_article_summary(lines, assertions, stability)
    _write_safe_starting_points(lines, assertions, refs_map, n_pairs)
    _write_investigate(lines, assertions, n_pairs)
    _write_section_reliability(lines, assertions)
    _write_citation_gaps(lines, assertions, refs_map)

    lines += [
        _HEAVY,
        "END OF REPORT",
        _HEAVY,
        "",
    ]

    report_text = "\n".join(lines)

    if print_to_terminal:
        print(report_text)

    report_path = _save(report_text, article_title, r["newest"]["year"])
    return report_path


# ---------------------------------------------------------------------------
# 1. ARTICLE SUMMARY
# ---------------------------------------------------------------------------

def _write_article_summary(
    lines: List[str],
    assertions: List[dict],
    stability: dict,
) -> None:
    _heading(lines, "ARTICLE SUMMARY")

    r       = stability["revision_range"]
    n_pairs = stability["total_comparison_pairs"]
    total   = len(assertions)

    n_certain   = sum(1 for a in assertions if a["stability_label"] == "Almost Certain")
    n_uncertain = sum(1 for a in assertions if a["stability_label"] == "Uncertain")
    n_doubtful  = sum(1 for a in assertions if a["stability_label"] == "Doubtful")

    factual      = sum(1 for a in assertions if a.get("claim_type") == "factual")
    interpretive = sum(1 for a in assertions if a.get("claim_type") == "interpretive")
    statistical  = sum(1 for a in assertions if a.get("claim_type") == "statistical")

    lines.append(
        f"Stability analysis covers {stability['article']} from "
        f"{r['oldest']['year']} to {r['newest']['year']},"
    )
    lines.append(
        f"based on {n_pairs} rounds of annual editorial review."
    )
    lines.append("")
    lines.append(f"  {'':26}  {'Claims':>6}   {'% of Total':>10}   Notes")
    lines.append(f"  {'Almost Certain':<26}  {n_certain:>6}   {n_certain/total:>9.1%}   safe to use")
    lines.append(f"  {'Uncertain':<26}  {n_uncertain:>6}   {n_uncertain/total:>9.1%}   check before using")
    lines.append(f"  {'Doubtful':<26}  {n_doubtful:>6}   {n_doubtful/total:>9.1%}   needs verification")
    lines.append(f"  {_DOTS[:60]}")
    lines.append(f"  {'Total claims analysed':<26}  {total:>6}")
    lines.append("")
    lines.append("Claim types:")
    lines.append(f"  Factual         {factual:>4}   ({factual / total:.0%})")
    lines.append(f"  Interpretive    {interpretive:>4}   ({interpretive / total:.0%})")
    lines.append(f"  Statistical     {statistical:>4}   ({statistical / total:.0%})")
    lines.append("")
    lines.append(
        f"Almost Certain means a claim was present and consistent across all {n_pairs} annual"
    )
    lines.append(
        "editorial reviews. It does not mean the claim has been independently verified."
    )
    lines.append("")


# ---------------------------------------------------------------------------
# 2. CREATOR SAFE STARTING POINTS
# ---------------------------------------------------------------------------

def _write_safe_starting_points(
    lines: List[str],
    assertions: List[dict],
    refs_map: Dict[str, str],
    n_pairs: int,
) -> None:
    _heading(lines, "CREATOR SAFE STARTING POINTS")

    candidates = [
        a for a in assertions
        if a["stability_label"] == "Almost Certain"
        and a.get("claim_type") == "factual"
        and a.get("subject") is not None
        and a.get("predicate", "MISSING") != "MISSING"
        and not a.get("context_dependent", False)
    ]

    # Lead / Introduction first; then remaining by stability_score desc
    # (assertions already sorted stability desc, so relative order is preserved)
    lead   = [a for a in candidates if a.get("source_section", "").lower() in ("lead", "introduction")]
    others = [a for a in candidates if a not in lead]
    ordered = (lead + others)[:_MAX_SAFE]

    if not ordered:
        lines.append(
            "No factual claims met all criteria (Almost Certain, "
            "identified subject, not context-dependent)."
        )
        lines.append("")
        return

    lines.append(
        f"These {len(ordered)} factual claims have been consistently present through "
        f"all {n_pairs} annual"
    )
    lines.append(
        "editorial reviews. Each includes its Wikipedia source so you can "
        "trace and attribute"
    )
    lines.append("the information.")
    lines.append("")
    lines.append(
        "Use these as starting points for your research — not as finished "
        "statements to publish"
    )
    lines.append("without further verification.")
    lines.append(_DOTS)
    lines.append("")

    for i, a in enumerate(ordered, 1):
        sources = _resolve_sources(a, refs_map)
        _write_safe_entry(lines, i, a, sources)

    if len(candidates) > _MAX_SAFE:
        lines.append(
            f"  ... and {len(candidates) - _MAX_SAFE} more Almost Certain factual claims"
        )
        lines.append("  in the full JSON output.")
        lines.append("")

    lines += [
        _DOTS,
        "IMPORTANT: These claims reflect editorial consensus on Wikipedia — not",
        "independent fact-checking. Always cross-check before publishing.",
        "",
    ]


def _write_safe_entry(
    lines: List[str],
    index: int,
    a: dict,
    sources: List[str],
) -> None:
    claim_lines = _wrap(a.get("claim_text", ""), _CLAIM_WRAP, indent="      ")

    lines.append(f" {index:>2}. {claim_lines[0].lstrip()}")
    for cl in claim_lines[1:]:
        lines.append(cl)

    lines.append(f"     Section : {a.get('source_section', '?')}")

    if sources:
        label = "Sources" if len(sources) > 1 else "Source "
        lines.append(f"     {label} : {_clip(sources[0], _CLAIM_WRAP)}")
        for src in sources[1:]:
            lines.append(f"              {_clip(src, _CLAIM_WRAP)}")
    else:
        lines.append("     Source  : (none cited in article)")

    lines.append("")


# ---------------------------------------------------------------------------
# 3. INVESTIGATE THESE
# ---------------------------------------------------------------------------

def _write_investigate(
    lines: List[str],
    assertions: List[dict],
    n_pairs: int,
) -> None:
    doubtful  = [a for a in assertions if a["stability_label"] == "Doubtful"]
    uncertain = [a for a in assertions if a["stability_label"] == "Uncertain"]

    if not doubtful and not uncertain:
        return

    _heading(lines, "INVESTIGATE THESE")

    lines.append("The claims below need extra scrutiny before you use them.")
    lines.append("Doubtful claims were added recently with no editorial track record.")
    lines.append("Uncertain claims were present in earlier versions but were reworded.")
    lines.append("")

    # ---- Doubtful ----
    if doubtful:
        doubtful_sorted = sorted(
            doubtful,
            key=lambda a: (-a.get("first_seen_year", 0), a["stability_score"]),
        )

        # Group by first_seen_year so we can explain each cohort once
        by_year: OrderedDict = OrderedDict()
        for a in doubtful_sorted:
            yr = a.get("first_seen_year", "?")
            by_year.setdefault(yr, []).append(a)

        lines.append(_DOTS)
        lines.append(
            f"DOUBTFUL — Added recently, no editorial track record ({len(doubtful)} claims)"
        )
        lines.append(_DOTS)
        lines.append("")

        shown = 0
        for year, group in by_year.items():
            if shown >= _MAX_DOUBTFUL:
                break
            window = group[0].get("observation_window", "?")
            lines.append(
                f"  First added in {year} — seen in {window} of {n_pairs} reviews:"
            )
            lines.append("")
            for a in group:
                if shown >= _MAX_DOUBTFUL:
                    break
                _write_investigate_entry(lines, a, n_pairs, show_drift=False)
                shown += 1

        if len(doubtful) > _MAX_DOUBTFUL:
            lines.append(
                f"  ... and {len(doubtful) - _MAX_DOUBTFUL} more Doubtful claims "
                f"in the JSON output."
            )
            lines.append("")

    # ---- Uncertain ----
    if uncertain:
        uncertain_sorted = sorted(
            uncertain,
            key=lambda a: (-a.get("pairs_modified", 0), a["stability_score"]),
        )

        lines.append(_DOTS)
        n_unc = len(uncertain)
        lines.append(
            f"UNCERTAIN — Present in earlier versions but changed by editors "
            f"({n_unc} {'claim' if n_unc == 1 else 'claims'})"
        )
        lines.append(_DOTS)
        lines.append("")

        for a in uncertain_sorted[:_MAX_UNCERTAIN]:
            _write_investigate_entry(lines, a, n_pairs, show_drift=True)

        if len(uncertain) > _MAX_UNCERTAIN:
            lines.append(
                f"  ... and {len(uncertain) - _MAX_UNCERTAIN} more Uncertain claims "
                f"in the JSON output."
            )
            lines.append("")


def _write_investigate_entry(
    lines: List[str],
    a: dict,
    n_pairs: int,
    show_drift: bool,
) -> None:
    claim_lines = _wrap(
        f'"{a.get("claim_text", "")}"',
        _CLAIM_WRAP,
        indent="     ",
    )
    lines.append(f"  * {claim_lines[0].lstrip()}")
    for cl in claim_lines[1:]:
        lines.append(cl)

    lines.append(f"    Section      : {a.get('source_section', '?')}")
    lines.append(f"    Reviews seen : {a.get('observation_window', '?')} of {n_pairs}")

    if show_drift:
        n_mod  = a.get("pairs_modified", 0)
        drift  = a.get("avg_text_drift", 0.0)
        if n_mod:
            lines.append(
                f"    Modified     : {n_mod} time{'s' if n_mod != 1 else ''}  "
                f"({_drift_label(drift)})"
            )

    if a.get("first_seen_year"):
        lines.append(f"    First seen   : {a['first_seen_year']}")

    lines.append("")


def _drift_label(drift: float) -> str:
    if drift < 0.05:
        return "minor wording change"
    if drift < 0.20:
        return "moderate revision"
    return "significant rewrite"


# ---------------------------------------------------------------------------
# 4. SECTION RELIABILITY
# ---------------------------------------------------------------------------

def _write_section_reliability(
    lines: List[str],
    assertions: List[dict],
) -> None:
    _heading(lines, "SECTION RELIABILITY")

    lines.append("Stability distribution for each section of the article.")
    lines.append(
        "High Doubtful % means the section was recently expanded and not yet"
    )
    lines.append("fully reviewed by Wikipedia's editorial community.")
    lines.append("")

    # Group and order by first source_position seen (closest to article start)
    sections: OrderedDict = OrderedDict()
    for a in assertions:
        sections.setdefault(a.get("source_section", "Unknown"), []).append(a)

    def _first_pos(sec_list: List[dict]) -> int:
        return min((a.get("source_position", 9999) for a in sec_list), default=9999)

    sections = OrderedDict(
        sorted(sections.items(), key=lambda kv: _first_pos(kv[1]))
    )

    nw = _SECTION_W
    lines.append(
        f"  {'Section':<{nw}}  {'Claims':>6}  "
        f"{'Almost Certain':>14}  {'Uncertain':>9}  {'Doubtful':>8}  Rating"
    )
    lines.append(
        f"  {'-'*nw}  {'-'*6}  {'-'*14}  {'-'*9}  {'-'*8}  {'-'*6}"
    )

    for sec_name, sec_list in sections.items():
        count   = len(sec_list)
        n_ac    = sum(1 for a in sec_list if a["stability_label"] == "Almost Certain")
        n_unc   = sum(1 for a in sec_list if a["stability_label"] == "Uncertain")
        n_dbt   = sum(1 for a in sec_list if a["stability_label"] == "Doubtful")
        ac_pct  = n_ac  / count * 100
        unc_pct = n_unc / count * 100
        dbt_pct = n_dbt / count * 100
        rating   = "Strong" if ac_pct >= 80 else ("Mixed" if ac_pct >= 50 else "Weak")
        display  = _clip(_strip_html(sec_name), nw)

        lines.append(
            f"  {display:<{nw}}  {count:>6}  "
            f"{ac_pct:>13.0f}%  {unc_pct:>8.0f}%  {dbt_pct:>7.0f}%  {rating}"
        )

    lines.append("")
    lines.append("  Rating:  Strong = Almost Certain >= 80%")
    lines.append("           Mixed  = Almost Certain >= 50%")
    lines.append("           Weak   = Almost Certain <  50%")
    lines.append("")


# ---------------------------------------------------------------------------
# 5. CITATION GAPS
# ---------------------------------------------------------------------------

def _write_citation_gaps(
    lines: List[str],
    assertions: List[dict],
    refs_map: Dict[str, str],
) -> None:
    _heading(lines, "CITATION GAPS")

    lines.append("Claims below have no source cited in the article, cite a source")
    lines.append("that could not be resolved, or are interpretive judgments without")
    lines.append("supporting references. Verify these independently.")
    lines.append("")

    # --- Unresolvable named refs ---
    unresolvable: dict = {}   # ref_name → [assertion, ...]
    for a in assertions:
        for ph in a.get("citations_in_source", []):
            m = re.match(r'\[REF:([^\]]+)\]', ph)
            if not m:
                continue
            name = m.group(1)
            if name not in refs_map:
                unresolvable.setdefault(name, []).append(a)

    if unresolvable:
        lines.append(_DOTS)
        lines.append(f"UNRESOLVED REFERENCES ({len(unresolvable)} reference name(s))")
        lines.append(_DOTS)
        lines.append(
            "  These reference names appear in the article but their"
        )
        lines.append("  definitions were not found in the current revision.")
        lines.append(
            "  They may have been removed from the article since these"
        )
        lines.append("  claims were last stored.")
        lines.append("")

        for ref_name, ref_assertions in list(unresolvable.items())[:10]:
            lines.append(f"  [{ref_name}] — cited by {len(ref_assertions)} claim(s):")
            for a in ref_assertions[:3]:
                claim_lines = _wrap(
                    f'"{_clip(a.get("claim_text", ""), 120)}"',
                    _CLAIM_WRAP,
                    indent="       ",
                )
                lines.append(f"    - {claim_lines[0].lstrip()}")
                for cl in claim_lines[1:]:
                    lines.append(cl)
            lines.append("")

    # --- Interpretive claims without any citation ---
    interp_no_cite = [
        a for a in assertions
        if a.get("claim_type") == "interpretive"
        and not a.get("citations_in_source")
    ]

    if interp_no_cite:
        lines.append(_DOTS)
        lines.append(
            f"INTERPRETIVE CLAIMS WITHOUT A SOURCE ({len(interp_no_cite)} claims)"
        )
        lines.append(_DOTS)
        lines.append(
            "  These are judgments or opinions stated without a cited"
        )
        lines.append(
            "  reference in the article. They may reflect editorial"
        )
        lines.append("  consensus but should be treated with extra caution.")
        lines.append("")

        shown = min(len(interp_no_cite), _MAX_GAPS)
        for a in interp_no_cite[:shown]:
            claim_lines = _wrap(
                f'"{_clip(a.get("claim_text", ""), 120)}"',
                _CLAIM_WRAP,
                indent="     ",
            )
            lines.append(f"  - {claim_lines[0].lstrip()}")
            for cl in claim_lines[1:]:
                lines.append(cl)
            lines.append(f"    Section : {a.get('source_section', '?')}")
            lines.append("")

        if len(interp_no_cite) > shown:
            lines.append(
                f"  ... and {len(interp_no_cite) - shown} more in the JSON output."
            )
            lines.append("")

    if not unresolvable and not interp_no_cite:
        lines.append("  No citation gaps detected.")
        lines.append("")


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------

def _resolve_sources(a: dict, refs_map: Dict[str, str]) -> List[str]:
    """
    Resolve citations_in_source placeholder strings to readable text.

    Named refs found in refs_map → resolved text.
    Named refs not in refs_map → "[name] (source definition not found)".
    Fragment refs (anonymous, contain spaces) → their fragment text as-is.
    """
    sources: List[str] = []
    seen: set = set()

    for ph in a.get("citations_in_source", []):
        m = re.match(r'\[REF:([^\]]+)\]', ph)
        if not m:
            continue
        identifier = m.group(1).strip()
        if identifier in seen:
            continue
        seen.add(identifier)

        if identifier in refs_map:
            sources.append(refs_map[identifier])
        elif " " not in identifier and len(identifier) <= 50:
            sources.append(f"[{identifier}] (source definition not found)")
        else:
            # Anonymous fragment ref — identifier IS the (partial) citation text
            sources.append(identifier + ("…" if len(identifier) == 50 else ""))

    return sources


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int, indent: str = "   ") -> List[str]:
    """Wrap text to width. Returns list of lines. Subsequent lines use indent."""
    wrapped = textwrap.fill(text, width=width, subsequent_indent=indent)
    return wrapped.split("\n")


def _clip(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _strip_html(text: str) -> str:
    """Remove HTML tags from section names (e.g. <i>Eagle</i> → Eagle)."""
    return re.sub(r'<[^>]+>', '', text).strip()


def _heading(lines: List[str], title: str) -> None:
    lines.append(_LIGHT)
    lines.append(title)
    lines.append(_LIGHT)
    lines.append("")


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def _save(report_text: str, article_title: str, newest_year: int) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    slug     = article_title.lower().replace(" ", "_")
    filename = f"{slug}_{newest_year}_stability_report.txt"
    path     = os.path.join(REPORTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return path


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print("Usage:")
        print("  python3 reporter.py <article_title>          # write report file")
        print("  python3 reporter.py <article_title> --print  # also print to terminal")
        sys.exit(0)

    print_flag    = "--print" in args
    title_parts   = [a for a in args if a != "--print"]
    article_title = " ".join(title_parts)

    if not article_title:
        print("Please provide an article title.")
        sys.exit(1)

    try:
        report_path = generate_report(article_title, print_to_terminal=print_flag)
        print(f"Report written: {report_path}")
    except FileNotFoundError as exc:
        print(str(exc))
        sys.exit(1)
