"""
Dataset Doctor: Outliers Plain-English Explanations & Badge Formatters (Phases 7 & 8).
Provides human-readable diagnostic context and badge chips for outlier issues.
"""
from typing import Any, Dict
from doctor.base import Issue


def explain_outliers_issue(issue: Issue) -> str:
    """
    Generates an actionable, clear Markdown explanation of an outlier issue.
    Highlights domain physical impossibilities, unit mix-ups, skewness context,
    and why certain outliers should be left untouched.
    """
    ev = issue.evidence or {}
    subtype = ev.get("subtype", "extreme_tail")
    col = issue.column or "column"
    count = ev.get("count", 0)
    pct = ev.get("pct", 0.0)
    is_skewed = ev.get("is_skewed", False)
    skew = ev.get("skew", 0.0)
    samples = ev.get("sample_values", [])
    bounds = ev.get("bounds", {})

    lines = []

    if subtype == "impossible_values":
        lines.append(f"### 🚨 Domain-Impossible Values in `{col}`")
        lines.append(
            f"**{count} value{'s' if count != 1 else ''} ({pct:.1%})** fall outside the physical, "
            f"logical, or contractual limits for this feature."
        )
        reason = ev.get("impossible_reason", "Values violate domain constraints.")
        lines.append(f"- **Violation**: {reason}")
        if samples:
            sample_str = ", ".join([f"`{s}`" for s in samples[:8]])
            lines.append(f"- **Sample invalid values**: {sample_str}")
        lines.append(
            "\n> **Recommendation Insight**: Impossible values are recorded errors, not valid extreme events. "
            "They should be converted to missing (`set_to_null`) and handed to the Null Advisor for principled imputation."
        )

    elif subtype == "unit_mixup":
        lines.append(f"### 📏 Possible Unit Mix-up in `{col}`")
        lines.append(
            f"**{count} value{'s' if count != 1 else ''}** appear to be recorded in the wrong unit of measurement "
            f"(e.g., millimeters instead of centimeters, or grams instead of kilograms)."
        )
        ratio = ev.get("median_ratio", 0)
        lines.append(f"- **Magnitude Jump**: Values are approximately **{ratio:,.0f}x** the typical median magnitude.")
        if samples:
            sample_str = ", ".join([f"`{s}`" for s in samples[:5]])
            lines.append(f"- **Sample anomalous values**: {sample_str}")
        lines.append(
            "\n> **Recommendation Insight**: Unit mix-ups severely distort distance-based models and linear coefficients. "
            "Verify data entry protocols or rescale/convert values to null."
        )

    else:
        # Extreme tail or statistical outliers
        lines.append(f"### 📈 Extreme Statistical Values in `{col}`")
        lines.append(
            f"Found **{count} values ({pct:.1%})** residing in the far statistical tails of the distribution."
        )

        if is_skewed:
            lines.append(
                f"- **Skewness Alert**: This feature is naturally skewed (**skew = {skew:.2f}**). "
                f"Standard $1.5\\times\\text{{IQR}}$ formulas systematically over-flag legitimate tails on skewed data. "
                f"Dataset Doctor uses robust Median Absolute Deviation (MAD) and 1%/99% percentile thresholds."
            )
        else:
            lines.append(
                f"- **Distribution**: Feature is approximately symmetric (**skew = {skew:.2f}**)."
            )

        if bounds:
            p1 = bounds.get("p1")
            p99 = bounds.get("p99")
            iqr_low = bounds.get("iqr_lower")
            iqr_high = bounds.get("iqr_upper")
            lines.append(
                f"- **Reference Bounds**: 1st percentile: `{p1:,.2f}`, 99th percentile: `{p99:,.2f}` | "
                f"IQR Bounds: `[{iqr_low:,.2f}, {iqr_high:,.2f}]`"
            )

        if samples:
            sample_str = ", ".join([f"`{s:,.2f}`" if isinstance(s, (int, float)) else f"`{s}`" for s in samples[:6]])
            lines.append(f"- **Sample tail values**: {sample_str}")

        lines.append(
            "\n> **Model Impact**: Tree-based algorithms (RandomForest, LightGBM, XGBoost) are invariant to monotonic scaling "
            "and largely unaffected by tail extremes. If your target is rare, extreme values often carry critical signal—"
            "leaving them untouched (`leave`) is frequently the safest, most honest choice."
        )

    return "\n".join(lines)


def format_outliers_badge(issue: Issue) -> str:
    """Returns a short UI badge chip representation for an outlier issue."""
    ev = issue.evidence or {}
    subtype = ev.get("subtype", "extreme_tail")
    count = ev.get("count", 0)

    if subtype == "impossible_values":
        return f"{count} IMPOSSIBLE (HIGH)"
    elif subtype == "unit_mixup":
        return f"{count} UNIT MIX-UP (MEDIUM)"
    elif issue.severity == "high":
        return f"{count} EXTREME (HIGH)"
    elif issue.severity == "medium":
        return f"{count} EXTREME (MEDIUM)"
    else:
        return f"{count} TAIL (LOW)"
