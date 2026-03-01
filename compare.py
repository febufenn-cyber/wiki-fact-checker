"""
Step 4 — Cross-Revision Assertion Comparison.

Compares atomic assertions between two stored revisions of a Wikipedia article
and categorises each assertion as unchanged, modified, added, or removed.

Matching is done in two phases:

  Phase A (this build) — Exact hash matching.
    Assertions with identical SHA-256 IDs in both revisions are 'unchanged'.
    Remaining assertions are placed in 'unmatched_old' and 'unmatched_new'.
    These are NOT yet labelled as added/removed — Phase B will determine which
    are genuine additions/removals versus rephrased modifications.

  Phase B — Fuzzy matching on unmatched pools.
    Uses subject + predicate structural signals to pair likely modifications.
    Remaining unmatched after fuzzy matching become 'added' and 'removed'.

Storage layout:
  comparisons/{slug}/{old_revid}_vs_{new_revid}.json

Usage:
  python3 compare.py "Apollo 11" 1130551454 1191771826   # one pair
  python3 compare.py "Apollo 11" --chain                 # all consecutive pairs
"""

import dataclasses
import difflib
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

from extractor import extract_assertions
from revisions import load_revision_content, load_revision_index
from schema import Assertion

logger = logging.getLogger(__name__)

COMPARISONS_DIR = "comparisons"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_revisions(title: str, old_revid: int, new_revid: int) -> str:
    """
    Compare assertions between two stored revisions of `title`.

    Phase A only: exact hash matching. Unmatched assertions are stored as
    'unmatched_old' and 'unmatched_new' pending Phase B fuzzy matching.

    Already-computed comparisons are skipped (resumable).

    Args:
        title:     Wikipedia article title (e.g. "Apollo 11").
        old_revid: Revision ID of the earlier revision.
        new_revid: Revision ID of the later revision.

    Returns:
        Path to the written comparison JSON file.
    """
    out_path = _comparison_path(title, old_revid, new_revid)
    if os.path.exists(out_path):
        logger.info("Comparison already stored: %s — skipping.", out_path)
        existing = load_comparison(title, old_revid, new_revid)
        if existing:
            _print_summary(existing)
        return out_path

    index = load_revision_index(title)
    if index is None:
        raise FileNotFoundError(
            f"No revision index for {title!r}. Run:  python3 revisions.py {title!r}"
        )

    meta_by_revid: Dict[int, dict] = {
        r["revid"]: r for r in index.get("revisions", [])
    }

    old_meta = meta_by_revid.get(old_revid)
    new_meta = meta_by_revid.get(new_revid)
    for revid, meta in ((old_revid, old_meta), (new_revid, new_meta)):
        if meta is None:
            raise ValueError(
                f"Revision {revid} not found in index for {title!r}. "
                f"Check stored revisions with:  python3 revisions.py {title!r} --list"
            )

    print(f"\nComparing {title!r}")
    print(
        f"  {old_meta['year']} (rev {old_revid})"
        f"  →  {new_meta['year']} (rev {new_revid})"
    )
    print()

    print(f"  Extracting from rev {old_revid} ...", end="", flush=True)
    old_assertions = _get_assertions(title, old_revid)
    print(f"  {len(old_assertions)} assertions")

    print(f"  Extracting from rev {new_revid} ...", end="", flush=True)
    new_assertions = _get_assertions(title, new_revid)
    print(f"  {len(new_assertions)} assertions")

    print()
    print("  Phase A — exact hash matching ...", end="", flush=True)
    unchanged_ids, unmatched_old, unmatched_new = _match_exact(
        old_assertions, new_assertions
    )
    print(f"  {len(unchanged_ids)} matched")

    print("  Phase B — fuzzy matching on unmatched ...", end="", flush=True)
    modified_pairs, removed, added = _match_fuzzy(unmatched_old, unmatched_new)
    print(f"  {len(modified_pairs)} modifications found")

    record = _build_record(
        title=title,
        old_meta=old_meta,
        new_meta=new_meta,
        old_assertions=old_assertions,
        new_assertions=new_assertions,
        unchanged_ids=unchanged_ids,
        modified_pairs=modified_pairs,
        added=added,
        removed=removed,
    )

    out_path = _write_comparison(record, title, old_revid, new_revid)
    _print_summary(record)
    print(f"  Output : {out_path}")
    print()
    return out_path


def chain_compare(title: str) -> List[str]:
    """
    Compare all consecutive revision pairs in the stored index.

    Runs compare_revisions() on each consecutive pair (oldest→next→...→newest).
    Already-computed comparisons are skipped.

    Returns:
        List of output file paths, one per consecutive pair.
    """
    index = load_revision_index(title)
    if index is None:
        raise FileNotFoundError(
            f"No revision index for {title!r}. Run:  python3 revisions.py {title!r}"
        )

    revisions = index.get("revisions", [])  # already sorted oldest→newest
    if len(revisions) < 2:
        print(
            f"  Only {len(revisions)} revision(s) stored for {title!r}. "
            "Need at least 2 to compare."
        )
        return []

    output_paths: List[str] = []
    for i in range(len(revisions) - 1):
        old_rev = revisions[i]
        new_rev = revisions[i + 1]
        path = compare_revisions(title, old_rev["revid"], new_rev["revid"])
        output_paths.append(path)

    return output_paths


def load_comparison(title: str, old_revid: int, new_revid: int) -> Optional[dict]:
    """Load a stored comparison JSON. Returns None if not yet computed."""
    path = _comparison_path(title, old_revid, new_revid)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Assertion extraction
# ---------------------------------------------------------------------------

def _get_assertions(title: str, revid: int) -> List[Assertion]:
    """
    Load stored revision content and run extract_assertions() on each section.

    Extraction results are NOT cached to disk — re-running compare.py will
    re-extract. This is fast enough (~35 sections per revision, ~200 sents/sec).

    Cross-section deduplication: the same sentence can appear verbatim in
    multiple sections (e.g., "Personnel" and "Backup crew" both quote a crew
    assignment note). extract_assertions() deduplicates within a section, but
    not across sections. We deduplicate by assertion_id here so that identical
    claims from two sections don't inflate the unmatched pools.
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
# Phase A — Exact hash matching
# ---------------------------------------------------------------------------

def _match_exact(
    old_assertions: List[Assertion],
    new_assertions: List[Assertion],
) -> Tuple[Set[str], List[Assertion], List[Assertion]]:
    """
    Phase A: identify assertions present in both revisions by assertion_id
    (SHA-256 of normalised claim_text).

    An exact match means the claim text is character-for-character identical
    after normalisation (lowercase, strip punctuation, collapse whitespace).
    These assertions are stable — the editor did not touch them.

    Returns:
        unchanged_ids:  assertion_id strings present in both old and new.
        unmatched_old:  assertions from old not in unchanged_ids.
                        Could be removed OR rephrased — Phase B determines which.
        unmatched_new:  assertions from new not in unchanged_ids.
                        Could be added OR rephrased — Phase B determines which.
    """
    old_id_set: Set[str] = {a.assertion_id for a in old_assertions}
    new_id_set: Set[str] = {a.assertion_id for a in new_assertions}

    unchanged_ids = old_id_set & new_id_set

    unmatched_old = [a for a in old_assertions if a.assertion_id not in unchanged_ids]
    unmatched_new = [a for a in new_assertions if a.assertion_id not in unchanged_ids]

    return unchanged_ids, unmatched_old, unmatched_new


# ---------------------------------------------------------------------------
# Phase B — Fuzzy matching on unmatched pools
# ---------------------------------------------------------------------------

# Minimum text similarity ratio (difflib SequenceMatcher) for a fuzzy pair to
# be accepted as a 'modified' match rather than independent add/remove.
# Rationale: the "Julian Scheer wrote" case has the same subject+predicate but
# completely different claim text (diff_ratio ~0.20). A threshold of 0.40 rejects
# that pair while accepting genuine rephrasing like "forced to switch" → "bumped"
# (diff_ratio ~0.55) and pronoun expansion "he accepted" → "Anders accepted" (~0.90).
_MIN_DIFF_RATIO = 0.40


def _match_fuzzy(
    unmatched_old: List[Assertion],
    unmatched_new: List[Assertion],
) -> Tuple[List[dict], List[Assertion], List[Assertion]]:
    """
    Phase B: pair likely modifications from the unmatched pools using
    structural signals already present in the Assertion schema.

    Matching signals (applied in priority order):
      subject_predicate  — normalised subject match AND predicate lemma match.
                           Highest confidence: the same entity did the same thing,
                           just phrased differently.
      subject_only       — normalised subject match, predicate lemmas differ.
                           Same entity, but the relationship changed. Lower
                           confidence; still more likely a modification than an
                           independent add+remove.

    Gate: every candidate pair must also have text_diff_ratio >= _MIN_DIFF_RATIO.
    This prevents pairing assertions that share a subject/predicate by coincidence
    but describe entirely different facts (e.g., two sentences about "Julian Scheer
    wrote" covering unrelated topics).

    Assignment is greedy best-first:
      1. Score every (old, new) candidate pair.
      2. Sort by score descending.
      3. Assign the top-scoring pair; remove both from pools.
      4. Repeat until no more candidates remain.
    This ensures a new assertion is never paired with more than one old assertion.

    Assertions with subject=None skip the subject signal and fall through to
    removed/added. This is honest: ~15–30% of assertions have null subjects
    (passive constructions) and cannot be matched structurally without embeddings.

    Returns:
        modified_pairs:  List of dicts (old, new, match_method, match_confidence,
                         section_changed, text_diff_ratio).
        remaining_old:   Unmatched after this pass → 'removed'.
        remaining_new:   Unmatched after this pass → 'added'.
    """
    from extractor import nlp as _nlp  # reuse loaded model; no second load

    # Build score matrix for all candidate pairs
    candidates: List[Tuple[float, str, int, int]] = []
    # (score, match_method, old_idx, new_idx)

    for oi, old_a in enumerate(unmatched_old):
        old_subj = _normalize_subject(old_a.subject)
        if not old_subj:
            continue  # can't match without a normalised subject
        old_lemma = _predicate_lemma(old_a.predicate, _nlp)

        for ni, new_a in enumerate(unmatched_new):
            new_subj = _normalize_subject(new_a.subject)
            if not new_subj:
                continue
            new_lemma = _predicate_lemma(new_a.predicate, _nlp)

            if old_subj != new_subj:
                continue  # subjects must match (normalised exact)

            diff_ratio = difflib.SequenceMatcher(
                None, old_a.claim_text, new_a.claim_text
            ).ratio()

            if diff_ratio < _MIN_DIFF_RATIO:
                continue  # text too dissimilar — coincidental subject match

            if old_lemma == new_lemma:
                score = 0.85 + diff_ratio * 0.10   # up to 0.95
                method = "subject_predicate"
            else:
                score = 0.55 + diff_ratio * 0.10   # up to 0.65
                method = "subject_only"

            candidates.append((score, method, oi, ni))

    # Greedy best-first assignment
    candidates.sort(key=lambda x: -x[0])

    assigned_old: Set[int] = set()
    assigned_new: Set[int] = set()
    modified_pairs: List[dict] = []

    for score, method, oi, ni in candidates:
        if oi in assigned_old or ni in assigned_new:
            continue  # already matched
        assigned_old.add(oi)
        assigned_new.add(ni)

        old_a = unmatched_old[oi]
        new_a = unmatched_new[ni]
        diff_ratio = difflib.SequenceMatcher(
            None, old_a.claim_text, new_a.claim_text
        ).ratio()

        modified_pairs.append({
            "old":              dataclasses.asdict(old_a),
            "new":              dataclasses.asdict(new_a),
            "match_method":     method,
            "match_confidence": round(score, 3),
            "section_changed":  old_a.source_section != new_a.source_section,
            "text_diff_ratio":  round(diff_ratio, 3),
        })

    remaining_old = [a for i, a in enumerate(unmatched_old) if i not in assigned_old]
    remaining_new = [a for i, a in enumerate(unmatched_new) if i not in assigned_new]

    return modified_pairs, remaining_old, remaining_new


def _normalize_subject(subject: Optional[str]) -> str:
    """
    Normalise a subject string for fuzzy comparison.

    Transformations:
      - lowercase
      - strip leading determiners: the, a, an
      - strip trailing punctuation
      - collapse whitespace

    Returns empty string for None subjects (signals: do not attempt to match).
    """
    if subject is None:
        return ""
    s = subject.lower().strip()
    for det in ("the ", "a ", "an "):
        if s.startswith(det):
            s = s[len(det):]
    return s.strip(".,;:'\"")


def _predicate_lemma(predicate: str, nlp) -> str:
    """
    Return the lemma of the first token of a predicate string.

    Uses the spaCy nlp instance passed in from the caller (shared, not reloaded).
    Falls back to lowercased predicate if parsing fails.
    """
    if predicate == "MISSING":
        return "MISSING"
    try:
        doc = nlp(predicate)
        return doc[0].lemma_.lower() if doc else predicate.lower()
    except Exception:
        return predicate.lower()


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def _build_record(
    title: str,
    old_meta: dict,
    new_meta: dict,
    old_assertions: List[Assertion],
    new_assertions: List[Assertion],
    unchanged_ids: Set[str],
    modified_pairs: List[dict],
    added: List[Assertion],
    removed: List[Assertion],
) -> dict:
    """
    Assemble the final comparison record after both phases.

    match_methods in summary counts how many matches came from each signal,
    so Step 5 (stability scoring) can weight hash matches differently from
    fuzzy matches when computing epistemic stability.
    """
    total_old = len(old_assertions)
    total_new = len(new_assertions)

    methods: Dict[str, int] = {
        "hash_exact":         len(unchanged_ids),
        "subject_predicate":  sum(1 for p in modified_pairs if p["match_method"] == "subject_predicate"),
        "subject_only":       sum(1 for p in modified_pairs if p["match_method"] == "subject_only"),
    }

    return {
        "article":            title,
        "comparison_phase":   "B_fuzzy_matched",
        "old_revision": {
            "revid":     old_meta["revid"],
            "year":      old_meta["year"],
            "timestamp": old_meta["timestamp"],
        },
        "new_revision": {
            "revid":     new_meta["revid"],
            "year":      new_meta["year"],
            "timestamp": new_meta["timestamp"],
        },
        "summary": {
            "total_old":   total_old,
            "total_new":   total_new,
            "unchanged":   len(unchanged_ids),
            "modified":    len(modified_pairs),
            "added":       len(added),
            "removed":     len(removed),
            "match_methods": methods,
        },
        # unchanged: IDs only — assertions are character-identical
        "unchanged": sorted(unchanged_ids),
        # modified: both old and new assertion, plus match metadata
        "modified":  modified_pairs,
        # added/removed: full assertion dicts
        "added":     [dataclasses.asdict(a) for a in added],
        "removed":   [dataclasses.asdict(a) for a in removed],
    }


def _write_comparison(record: dict, title: str, old_revid: int, new_revid: int) -> str:
    out_dir = os.path.join(COMPARISONS_DIR, _slug(title))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{old_revid}_vs_{new_revid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path


def _print_summary(record: dict) -> None:
    phase    = record.get("comparison_phase", "?")
    old_r    = record["old_revision"]
    new_r    = record["new_revision"]
    s        = record["summary"]
    total_old = s["total_old"]
    total_new = s["total_new"]

    W    = 54
    line = "─" * W

    print(f"  {line}")
    print(
        f"  {old_r['year']} (rev {old_r['revid']})"
        f"  →  {new_r['year']} (rev {new_r['revid']})"
    )
    print(f"  Old assertions : {total_old}")
    print(f"  New assertions : {total_new}")
    print(f"  {line}")

    if phase == "A_hash_only":
        matched   = s["hash_matched"]
        pct       = s["hash_matched_pct"]
        u_old     = s["unmatched_old"]
        u_new     = s["unmatched_new"]
        print(f"  Hash-matched (unchanged) : {matched:>4}  ({pct:.1f}% of old)")
        print(f"  Unmatched in old rev     : {u_old:>4}  (pending Phase B)")
        print(f"  Unmatched in new rev     : {u_new:>4}  (pending Phase B)")
        print(f"  {line}")
        print(f"  Phase A (hash-only) complete.")
        print(f"  Phase B will classify unmatched as modified / added / removed.")

    else:
        # Phase B format
        unchanged = s.get("unchanged", 0)
        modified  = s.get("modified", 0)
        added     = s.get("added", 0)
        removed   = s.get("removed", 0)
        methods   = s.get("match_methods", {})

        def _pct(n: int) -> str:
            return f"{n / total_old * 100:.1f}%" if total_old else "—"

        print(f"  Unchanged (hash-exact)   : {unchanged:>4}  ({_pct(unchanged)})")
        print(f"  Modified  (fuzzy-match)  : {modified:>4}  ({_pct(modified)})")
        print(f"  Added                    : {added:>4}")
        print(f"  Removed                  : {removed:>4}")
        print(f"  {line}")
        if methods:
            print(
                f"  Match methods: "
                f"hash={methods.get('hash_exact', 0)}, "
                f"subj+pred={methods.get('subject_predicate', 0)}, "
                f"subj_only={methods.get('subject_only', 0)}"
            )

    print()


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _comparison_path(title: str, old_revid: int, new_revid: int) -> str:
    return os.path.join(
        COMPARISONS_DIR, _slug(title), f"{old_revid}_vs_{new_revid}.json"
    )


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
        print("  python3 compare.py <title> <old_revid> <new_revid>")
        print("  python3 compare.py <title> --chain")
        sys.exit(0)

    chain_mode = "--chain" in args
    remaining  = [a for a in args if a != "--chain"]

    if chain_mode:
        if not remaining:
            print("Please provide an article title.")
            sys.exit(1)
        article_title = " ".join(remaining)
        paths = chain_compare(article_title)
        if paths:
            print(f"Chain complete. {len(paths)} comparison(s) written.")
            for p in paths:
                print(f"  {p}")
        print("\nNext: python3 stability.py (Step 5 — epistemic stability scoring)")

    else:
        if len(remaining) < 3:
            print("Error: need article title and two revision IDs.")
            print("Usage: python3 compare.py <title> <old_revid> <new_revid>")
            print("       python3 compare.py <title> --chain")
            sys.exit(1)

        # Last two args are revids; everything before is the article title
        old_revid_str = remaining[-2]
        new_revid_str = remaining[-1]
        title_parts   = remaining[:-2]

        try:
            old_revid = int(old_revid_str)
            new_revid = int(new_revid_str)
        except ValueError:
            print(
                f"Invalid revision IDs: {old_revid_str!r} {new_revid_str!r}. "
                "Both must be integers."
            )
            sys.exit(1)

        article_title = " ".join(title_parts)
        if not article_title:
            print("Please provide an article title.")
            sys.exit(1)

        compare_revisions(article_title, old_revid, new_revid)
        print("\nNext: --chain to compare all pairs, or Phase B for fuzzy matching.")
