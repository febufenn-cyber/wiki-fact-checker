"""
Step 5 — Epistemic Stability Scoring.

Reads all stored comparison JSONs for an article and computes a stability
score for each assertion currently present in the newest revision.

Stability reflects editorial persistence across annual snapshots — NOT truth.
A claim that survives multiple years of Wikipedia editing unchanged is almost
certainly well-established. A claim just added this year has no track record.

Scoring formula:
  score = (pairs_unchanged + avg_text_similarity × pairs_modified) / total_pairs

  total_pairs          = N−1 consecutive comparisons (4 for a 5-year window)
  avg_text_similarity  = mean text_diff_ratio for all modified events in history
  Dividing by total_pairs (not observation_window) penalises recency naturally:
  an assertion seen in only 1 of 4 pairs scores at most 0.25 regardless of
  how stable it was within that window.

Labels:
  Almost Certain  score >= 0.75   — consistent across most or all pairs
  Uncertain       score >= 0.40   — modified, or only partial coverage
  Doubtful        score <  0.40   — recently added, or very short history

Storage layout:
  stability/{slug}/stability.json

Usage:
  python3 stability.py "Apollo 11"         # compute stability
  python3 stability.py "Apollo 11" --list  # print stored summary
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional, Set, Tuple

from compare import load_comparison
from extractor import extract_assertions
from revisions import load_revision_content, load_revision_index
from schema import Assertion

logger = logging.getLogger(__name__)

STABILITY_DIR = "stability"

LABEL_ALMOST_CERTAIN = 0.75
LABEL_UNCERTAIN       = 0.40


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_stability(title: str) -> str:
    """
    Compute epistemic stability for all assertions in title's newest revision.

    Loads stored comparison JSONs, traces each current assertion backward
    through the comparison chain, and writes a labelled stability report.

    Args:
        title: Wikipedia article title (e.g. "Apollo 11").

    Returns:
        Path to the written stability.json file.
    """
    index = load_revision_index(title)
    if index is None:
        raise FileNotFoundError(
            f"No revision index for {title!r}. Run:  python3 revisions.py {title!r}"
        )

    revisions = index.get("revisions", [])  # sorted oldest → newest
    if len(revisions) < 2:
        raise ValueError(
            f"Need at least 2 stored revisions to compute stability for {title!r}. "
            f"Currently have {len(revisions)}."
        )

    oldest_rev = revisions[0]
    newest_rev = revisions[-1]

    # Load comparison JSONs for each consecutive pair
    comparisons: List[dict] = []
    for i in range(len(revisions) - 1):
        old_rev = revisions[i]
        new_rev = revisions[i + 1]
        comp = load_comparison(title, old_rev["revid"], new_rev["revid"])
        if comp is None:
            raise FileNotFoundError(
                f"Comparison {old_rev['revid']} vs {new_rev['revid']} not found. "
                f"Run:  python3 compare.py {title!r} --chain"
            )
        if comp.get("comparison_phase") == "A_hash_only":
            logger.warning(
                "Comparison %d vs %d is Phase A only (fuzzy matching not run). "
                "Stability scores will undercount modifications. "
                "Run compare.py --chain to upgrade to Phase B.",
                old_rev["revid"], new_rev["revid"],
            )
        comparisons.append(comp)

    n_pairs = len(comparisons)

    print(f"\nEpistemic stability : {title!r}")
    print(f"  Revision range    : {oldest_rev['year']} – {newest_rev['year']}")
    print(f"  Comparison pairs  : {n_pairs}")
    print()

    # Pre-build lookup structures for O(1) access during tracing
    comp_lookups = [_build_lookup(c) for c in comparisons]

    # Extract current assertions (newest revision)
    print(
        f"  Extracting assertions from rev {newest_rev['revid']} ...",
        end="", flush=True,
    )
    current_assertions = _get_assertions(title, newest_rev["revid"])
    print(f"  {len(current_assertions)} assertions")

    # Score each assertion
    print("  Scoring ...", end="", flush=True)
    scored: List[dict] = []
    for a in current_assertions:
        score, obs = _trace_assertion(
            a.assertion_id, comp_lookups, n_pairs, oldest_rev["year"]
        )
        scored.append({
            "assertion_id":          a.assertion_id,
            "claim_text":            a.claim_text,
            "claim_type":            a.claim_type,
            "subject":               a.subject,
            "predicate":             a.predicate,
            "source_section":        a.source_section,
            "extraction_confidence": a.extraction_confidence,
            "context_dependent":     a.context_dependent,
            "citations_in_source":   a.citations_in_source,
            "stability_score":       round(score, 3),
            "stability_label":       _score_to_label(score),
            "pairs_unchanged":       obs["unchanged"],
            "pairs_modified":        obs["modified"],
            "observation_window":    obs["window"],
            "avg_text_drift":        obs["avg_drift"],
            "first_seen_year":       obs["first_seen_year"],
        })
    print("  done")

    # Sort: most stable first
    scored.sort(key=lambda x: -x["stability_score"])

    n_certain   = sum(1 for s in scored if s["stability_label"] == "Almost Certain")
    n_uncertain = sum(1 for s in scored if s["stability_label"] == "Uncertain")
    n_doubtful  = sum(1 for s in scored if s["stability_label"] == "Doubtful")
    total       = len(scored)

    record = {
        "article":      title,
        "computed_at":  datetime.now(timezone.utc).isoformat(),
        "revision_range": {
            "oldest": {"revid": oldest_rev["revid"], "year": oldest_rev["year"]},
            "newest": {"revid": newest_rev["revid"], "year": newest_rev["year"]},
        },
        "total_comparison_pairs": n_pairs,
        "scoring": {
            "formula":    "score = (unchanged + avg_similarity * modified) / total_pairs",
            "thresholds": {
                "almost_certain": LABEL_ALMOST_CERTAIN,
                "uncertain":      LABEL_UNCERTAIN,
            },
        },
        "summary": {
            "total_assertions": total,
            "almost_certain":   n_certain,
            "uncertain":        n_uncertain,
            "doubtful":         n_doubtful,
        },
        "assertions": scored,
    }

    out_path = _write_stability(record, title)
    _print_summary(record)
    print(f"  Output : {out_path}")
    print()
    return out_path


def load_stability(title: str) -> Optional[dict]:
    """Load a stored stability report, or None if not yet computed."""
    path = _stability_path(title)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_stability_summary(title: str) -> None:
    """Print the stored stability summary. Used with --list."""
    record = load_stability(title)
    if record is None:
        print(f"No stability data for {title!r}.")
        print(f"Run:  python3 stability.py {title!r}")
        return
    _print_summary(record)


# ---------------------------------------------------------------------------
# Assertion extraction
# ---------------------------------------------------------------------------

def _get_assertions(title: str, revid: int) -> List[Assertion]:
    """
    Load stored revision content and extract assertions.
    Deduplicates by assertion_id across sections (same logic as compare.py).
    """
    content = load_revision_content(title, revid)
    if content is None:
        raise FileNotFoundError(
            f"No stored content for {title!r} revid {revid}. "
            f"Run:  python3 revisions.py {title!r}"
        )

    all_assertions: List[Assertion] = []
    seen_ids: Set[str] = set()
    for section in content.get("sections", []):
        assertions = extract_assertions(
            section_text=section["text"],
            section_title=section["title"],
            article_title=title,
            revision_id=str(revid),
            position_offset=len(all_assertions),
        )
        for a in assertions:
            if a.assertion_id not in seen_ids:
                seen_ids.add(a.assertion_id)
                all_assertions.append(a)

    return all_assertions


# ---------------------------------------------------------------------------
# Comparison lookup structures
# ---------------------------------------------------------------------------

def _build_lookup(comp: dict) -> dict:
    """
    Pre-build O(1) lookup structures from one comparison JSON.

    unchanged_set:      set of assertion_id strings present in both revisions.
    modified_by_new_id: maps new assertion_id → full modified pair dict,
                        so we can follow modification chains backward.
    old_year/new_year:  used to determine first_seen_year during tracing.
    """
    return {
        "unchanged_set":      set(comp["unchanged"]),
        "modified_by_new_id": {
            p["new"]["assertion_id"]: p
            for p in comp.get("modified", [])
        },
        "old_year":  comp["old_revision"]["year"],
        "new_year":  comp["new_revision"]["year"],
    }


# ---------------------------------------------------------------------------
# Stability tracing
# ---------------------------------------------------------------------------

def _trace_assertion(
    current_id: str,
    comp_lookups: List[dict],
    n_pairs: int,
    oldest_year: int,
) -> Tuple[float, dict]:
    """
    Walk backward through comparison pairs to build the assertion's history.

    Walking direction: newest pair first (so we traverse from most recent
    to oldest, following any modification chains along the way).

    For each step:
      1. search_id ∈ unchanged_set  → survived unchanged; pairs_unchanged += 1
      2. search_id ∈ modified_by_new_id  → text changed here;
         record text_diff_ratio; follow chain to old assertion_id
      3. neither  → assertion wasn't in this pair's old revision; stop.

    Scoring:
      score = (pairs_unchanged + avg_text_similarity × pairs_modified) / n_pairs
      Capped at 1.0. Assertions not present in any pair score 0.0 (Doubtful).

    first_seen_year:
      Set to the new_year of the pair where tracing stopped (first appearance).
      If tracing exhausts all pairs, set to oldest_year (present from the start).

    Returns:
        (stability_score, observations_dict)
    """
    pairs_unchanged = 0
    modified_diff_ratios: List[float] = []
    search_id = current_id
    first_seen_year = oldest_year  # default: present since oldest stored revision

    for lookup in reversed(comp_lookups):   # newest → oldest
        if search_id in lookup["unchanged_set"]:
            pairs_unchanged += 1
            # same assertion_id in both revisions — continue tracing backward

        elif search_id in lookup["modified_by_new_id"]:
            pair = lookup["modified_by_new_id"][search_id]
            modified_diff_ratios.append(pair["text_diff_ratio"])
            search_id = pair["old"]["assertion_id"]  # follow the chain

        else:
            # Not in this pair's old revision — assertion first appeared in new_year
            first_seen_year = lookup["new_year"]
            break

    n_modified = len(modified_diff_ratios)
    window     = pairs_unchanged + n_modified
    avg_sim    = (sum(modified_diff_ratios) / n_modified) if n_modified else 1.0
    avg_drift  = round(1.0 - avg_sim, 3)

    score = (pairs_unchanged + avg_sim * n_modified) / n_pairs if n_pairs else 0.0
    score = min(1.0, max(0.0, round(score, 3)))

    return score, {
        "unchanged":      pairs_unchanged,
        "modified":       n_modified,
        "window":         window,
        "avg_drift":      avg_drift,
        "first_seen_year": first_seen_year,
    }


def _score_to_label(score: float) -> str:
    if score >= LABEL_ALMOST_CERTAIN:
        return "Almost Certain"
    if score >= LABEL_UNCERTAIN:
        return "Uncertain"
    return "Doubtful"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_summary(record: dict) -> None:
    r     = record["revision_range"]
    s     = record["summary"]
    total = s["total_assertions"]

    n_certain   = s["almost_certain"]
    n_uncertain = s["uncertain"]
    n_doubtful  = s["doubtful"]

    def pct(n: int) -> str:
        return f"{n / total * 100:.1f}%" if total else "—"

    W    = 54
    line = "─" * W

    print(f"\n  {line}")
    print(f"  EPISTEMIC STABILITY REPORT")
    print(f"  {record['article']}  ·  {r['oldest']['year']}–{r['newest']['year']}")
    print(
        f"  {record['total_comparison_pairs']} comparison pairs"
        f"  ·  {total} current assertions"
    )
    print(f"  {line}")
    print(
        f"  Almost Certain  (score >= {LABEL_ALMOST_CERTAIN:.0%})"
        f" : {n_certain:>4}  ({pct(n_certain)})"
    )
    print(
        f"  Uncertain       (score >= {LABEL_UNCERTAIN:.0%})"
        f" : {n_uncertain:>4}  ({pct(n_uncertain)})"
    )
    print(
        f"  Doubtful        (score <  {LABEL_UNCERTAIN:.0%})"
        f" : {n_doubtful:>4}  ({pct(n_doubtful)})"
    )
    print(f"  {line}")

    assertions = record.get("assertions", [])

    # Almost Certain samples (prefer non-context-dependent factual claims)
    certain_sample = [
        a for a in assertions
        if a["stability_label"] == "Almost Certain"
        and a.get("subject")
        and not a.get("context_dependent", False)
    ][:3]
    if certain_sample:
        print(f"\n  Sample — Almost Certain")
        for a in certain_sample:
            print(f"    [{a['stability_score']:.2f}]  {a['claim_text'][:80]}")

    # Uncertain samples: prefer those with modifications (actual editorial activity)
    uncertain_sample = sorted(
        [a for a in assertions if a["stability_label"] == "Uncertain"],
        key=lambda x: (-x["pairs_modified"], x["stability_score"]),
    )[:3]
    if uncertain_sample:
        print(f"\n  Sample — Uncertain")
        for a in uncertain_sample:
            drift_str = f"  drift={a['avg_text_drift']:.2f}" if a["pairs_modified"] else ""
            window_str = f"  window={a['observation_window']}/{record['total_comparison_pairs']}"
            print(
                f"    [{a['stability_score']:.2f}{drift_str}{window_str}]"
                f"  {a['claim_text'][:75]}"
            )

    # Doubtful samples
    doubtful_sample = sorted(
        [a for a in assertions if a["stability_label"] == "Doubtful"],
        key=lambda x: x["stability_score"],
    )[:3]
    if doubtful_sample:
        print(f"\n  Sample — Doubtful")
        for a in doubtful_sample:
            window_str = f"  window={a['observation_window']}/{record['total_comparison_pairs']}"
            year_str   = f"  first={a['first_seen_year']}"
            print(
                f"    [{a['stability_score']:.2f}{window_str}{year_str}]"
                f"  {a['claim_text'][:70]}"
            )

    print()


def _write_stability(record: dict, title: str) -> str:
    out_dir = os.path.join(STABILITY_DIR, _slug(title))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stability.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path


def _stability_path(title: str) -> str:
    return os.path.join(STABILITY_DIR, _slug(title), "stability.json")


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
        print("  python3 stability.py <article_title>        # compute stability")
        print("  python3 stability.py <article_title> --list # show stored report")
        sys.exit(0)

    list_only   = "--list" in args
    title_parts = [a for a in args if a != "--list"]
    article_title = " ".join(title_parts)

    if not article_title:
        print("Please provide an article title.")
        sys.exit(1)

    if list_only:
        print_stability_summary(article_title)
    else:
        compute_stability(article_title)
        print("Next step: python3 reporter.py (generate shareable report)")
