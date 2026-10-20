"""
Dataset Doctor: Null / Missing Value Explanation Generator.
Translates diagnostic evidence into plain, unambiguous text for the GUI and CLI.
Follows DESIGN.md rules: short sentences, evidence-first, no jargon.
"""
from typing import Any, Dict, List, Optional
from doctor.base import Issue


def explain_null_issue(evidence: Dict[str, Any], column: Optional[str] = None) -> str:
    """
    Generates a concise, plain-English summary of what was found for a column.
    Example: '120 of 1000 rows missing (12.0%). Missing is NOT linked to target (p = 0.61).'
    """
    total_rows = evidence.get("total_rows", 0)
    missing_count = evidence.get("total_missing_count", 0)
    missing_pct = evidence.get("total_missing_pct", 0.0)
    is_target = evidence.get("is_target", False)

    parts = []

    # 1. Main count & percent
    if total_rows > 0:
        parts.append(f"{missing_count} of {total_rows} rows missing ({missing_pct:.1f}%).")
    else:
        parts.append(f"{missing_count} missing cells detected.")

    # 2. Hidden tokens breakdown if present
    hidden = evidence.get("hidden_token_counts", {})
    if hidden:
        token_strs = [f"'{t}': {c}" for t, c in hidden.items()]
        parts.append(f"Hidden tokens: {', '.join(token_strs)}.")

    # 3. Sentinel breakdown if present
    sentinels = evidence.get("sentinel_counts", {})
    if sentinels:
        sent_strs = [f"'{s}': {c}" for s, c in sentinels.items()]
        parts.append(f"Possible sentinels: {', '.join(sent_strs)}.")

    # 4. Target link pattern
    target_pattern = evidence.get("target_pattern")
    if target_pattern:
        p_val = target_pattern.get("p_value", 1.0)
        if target_pattern.get("related", False):
            parts.append(f"Missingness IS statistically linked to target (p = {p_val:.3f}).")
        else:
            parts.append(f"Missing is NOT linked to target (p = {p_val:.2f}).")

    # 5. Target column warning
    if is_target:
        parts.append("Target column missingness must be dropped before training.")

    return " ".join(parts)


def format_evidence_badge(evidence: Dict[str, Any]) -> str:
    """Returns a short snippet suitable for a table cell in the GUI."""
    nan_cnt = evidence.get("nan_count", 0)
    hidden_cnt = sum(evidence.get("hidden_token_counts", {}).values())
    sentinel_cnt = sum(evidence.get("sentinel_counts", {}).values())

    components = []
    if nan_cnt > 0:
        components.append(f"{nan_cnt} NaN")
    if hidden_cnt > 0:
        components.append(f"{hidden_cnt} hidden")
    if sentinel_cnt > 0:
        components.append(f"{sentinel_cnt} sentinel")

    return " + ".join(components) if components else "0 missing"
