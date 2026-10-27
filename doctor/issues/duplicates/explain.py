"""
Dataset Doctor: Duplicates Explanation Generator.
Translates diagnostic evidence for exact duplicates, near duplicates,
conflicting labels, and pre-balanced datasets into clear, plain English.
Follows DESIGN.md rules: short sentences, evidence-first, zero jargon.
"""
from typing import Any, Dict, Optional


def explain_duplicates_issue(evidence: Dict[str, Any], kind: str = "exact") -> str:
    """
    Generates a concise, plain-English summary of what was found for duplicate checks.
    """
    total_rows = evidence.get("total_rows", 0)

    if kind == "exact":
        surplus = evidence.get("surplus_copies", 0)
        pct = evidence.get("duplicate_pct", 0.0)
        involved = evidence.get("total_involved_rows", 0)
        return (
            f"{surplus} redundant duplicate rows found across {involved} records ({pct:.1f}% of dataset). "
            f"Exact duplicate rows provide no new learning signal and cause test leakage if split across partitions."
        )

    if kind == "conflicting":
        count = evidence.get("conflicting_rows_count", 0)
        target_col = evidence.get("target_column", "target")
        return (
            f"{count} rows have identical features but contradictory target values in '{target_col}'. "
            f"Conflicting labels confuse ML algorithms and should be verified or dropped before training."
        )

    if kind == "near":
        count = evidence.get("near_duplicate_count", 0)
        types_seen = evidence.get("variation_types", ["case/whitespace"])
        return (
            f"{count} near-duplicate rows detected differing only in {', '.join(types_seen)}. "
            f"Standardizing text casing and trimming whitespace can unify these records."
        )

    if kind == "id":
        col = evidence.get("column", "id")
        dup_cnt = evidence.get("duplicate_id_count", 0)
        return (
            f"Identifier column '{col}' contains {dup_cnt} duplicate values. "
            f"Unique IDs must never repeat across distinct observations."
        )

    if kind == "pre_balanced":
        ratio = evidence.get("target_ratio_str", "50.0% / 50.0%")
        repeated = evidence.get("repeated_minority_rows", 0)
        return (
            f"Pre-balanced file warning: class distribution is exactly {ratio} with {repeated} repeated minority feature vectors. "
            f"This suggests the dataset was synthetically oversampled before saving, risking severe test leakage."
        )

    return "Duplicate data detected."


def format_duplicates_badge(evidence: Dict[str, Any]) -> str:
    """Returns a short snippet suitable for a badge or summary cell in the GUI."""
    kind = evidence.get("kind", "exact")
    if kind == "exact":
        return f"{evidence.get('surplus_copies', 0)} exact copies"
    if kind == "conflicting":
        return f"{evidence.get('conflicting_rows_count', 0)} conflicting"
    if kind == "near":
        return f"{evidence.get('near_duplicate_count', 0)} near duplicates"
    if kind == "id":
        return f"{evidence.get('duplicate_id_count', 0)} duplicate IDs"
    if kind == "pre_balanced":
        return f"Pre-balanced ({evidence.get('target_ratio_str', '50/50')})"
    return "Duplicate issue"
