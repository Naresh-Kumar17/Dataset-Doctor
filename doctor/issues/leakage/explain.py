"""
Dataset Doctor: Data Leakage Plain-English Explanations & Badge Formatters (Phase 11).
"""
from typing import Any, Optional
from doctor.base import Issue


def explain_leakage_issue(issue: Issue) -> str:
    """
    Generates a clear plain-English explanation of why the detected feature is a data leak.
    """
    ev = issue.evidence or {}
    col = issue.column or "column"
    target = ev.get("target_column", "target")
    subtype = ev.get("subtype", "leakage")
    corr = ev.get("correlation")

    corr_text = f" (correlation: {corr:.3f})" if corr is not None else ""

    if subtype == "target_copy":
        return (
            f"**Critical Target Leak in '{col}':**\n"
            f"This column is an exact duplicate of target '{target}'. "
            f"Any machine learning algorithm will assign 100% feature importance to it, "
            f"scoring flawless accuracy in testing that completely collapses to random guessing in deployment.\n\n"
            f"**Remediation:** Drop '{col}' from the feature set before model training."
        )
    elif subtype == "target_inverted":
        return (
            f"**Inverted Target Leak in '{col}':**\n"
            f"This column is an exact inverted clone (1 - target) of '{target}'. "
            f"Models will trivially memorize the flipped labels, giving the dangerous illusion of predictive power.\n\n"
            f"**Remediation:** Drop '{col}' from the feature set immediately."
        )
    elif subtype in ("target_scaled", "perfect_predictor"):
        return (
            f"**Near-Perfect Predictor / Post-Event Leak in '{col}'{corr_text}:**\n"
            f"Features that correlate above 0.95 with the target outcome are almost always populated "
            f"AFTER the target event occurs (e.g. payout amount after a claim is approved, or hospital discharge notes). "
            f"In production, this field is not yet known at decision time.\n\n"
            f"**Remediation:** Drop '{col}' unless you have mathematical proof that it is strictly known before the event."
        )
    elif subtype == "index_leak":
        return (
            f"**Row Sequence / Index Leak in '{col}'{corr_text}:**\n"
            f"This feature reflects dataset sorting order that leaks the target outcome. "
            f"A tree model will simply split on row position rather than learning genuine causal patterns.\n\n"
            f"**Remediation:** Drop '{col}' or shuffle the dataset order before partitioning."
        )
    elif subtype == "time_order_risk":
        return (
            f"**Temporal Ordering Risk in '{col}':**\n"
            f"This column represents a monotonically increasing time series. "
            f"Applying random k-fold or random train/test splits lets future records predict past events, "
            f"violating causality and producing misleadingly high validation scores.\n\n"
            f"**Remediation:** Partition the data using a strict chronological time-split (earliest rows for training, latest for testing)."
        )
    else:
        return (
            f"**Suspicious Feature Correlation in '{col}'{corr_text}:**\n"
            f"Examine whether this feature is legitimately available at prediction time."
        )


def format_leakage_badge(item: Any) -> str:
    """
    Renders a Neo-Brutalist HTML badge token for leakage severity.
    Accepts string severity, Issue object, or evidence dictionary.
    """
    if isinstance(item, str):
        severity = item
        subtype = ""
    elif hasattr(item, "severity") and hasattr(item, "evidence"):
        severity = item.severity
        subtype = item.evidence.get("subtype", "")
    elif isinstance(item, dict):
        subtype = item.get("subtype", "")
        severity = "high" if subtype in ("target_copy", "target_scaled", "target_inverted", "index_leak", "perfect_predictor") else "medium"
    else:
        severity = "low"
        subtype = ""

    if subtype == "time_order_risk":
        return '<span class="chip medium" style="background:#FEF3C7;color:#92400E;">TIME-SPLIT RISK</span>'
    elif severity == "high":
        return '<span class="chip high" style="background:#FEE2E2;color:#991B1B;">CRITICAL LEAK</span>'
    elif severity == "medium":
        return '<span class="chip medium" style="background:#FEF3C7;color:#92400E;">SUSPICIOUS LEAK</span>'
    return '<span class="chip low" style="background:#EBFBEE;color:#065F46;">LEAK-FREE</span>'
