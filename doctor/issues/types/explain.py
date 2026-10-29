"""
Dataset Doctor: Plain-English Type Issue Explanations & Badge Formatters (Phase 6).
Provides human-readable summaries and visual badge labels for:
- Numbers stored as text (currencies, percentages, unit strings)
- Ambiguous date formats (day-first vs month-first uncertainty)
- Standard date formats
- Yes/No and binary representations
- Constant / zero-variance columns
- High-cardinality ID-like columns
- Mixed type columns
"""
from typing import Any, Dict


def explain_types_issue(evidence: Dict[str, Any], kind: str) -> str:
    """
    Returns an informative, plain-language explanation of a detected data type anomaly.
    """
    col = evidence.get("column", "column")

    if kind == "text_number":
        symbols = evidence.get("detected_symbols", [])
        sym_str = ", ".join(repr(s) for s in symbols) if symbols else "symbols/commas"
        sample_str = ", ".join(repr(v) for v in evidence.get("sample_values", [])[:3])
        return (
            f"Column '{col}' contains numeric values formatted as text with characters such as {sym_str} "
            f"(e.g., {sample_str}). Machine learning algorithms and scalers treat these as arbitrary strings "
            "or categorical categories instead of numeric magnitudes unless converted."
        )

    elif kind == "ambiguous_date":
        sample_str = ", ".join(repr(v) for v in evidence.get("sample_values", [])[:3])
        return (
            f"Column '{col}' contains date strings where both day and month components are 12 or less (e.g., {sample_str}). "
            "It cannot be determined automatically whether this follows Day-First (DD/MM/YYYY) or "
            "Month-First (MM/DD/YYYY) format without user confirmation."
        )

    elif kind == "date":
        fmt = evidence.get("detected_format", "standard")
        return (
            f"Column '{col}' contains parseable date timestamps in format {fmt}. "
            "Parsing as datetime enables proper temporal validation, feature engineering, and time-split validation."
        )

    elif kind == "boolean":
        uniques = evidence.get("unique_values", [])
        return (
            f"Column '{col}' represents a binary yes/no flag with values {uniques}. "
            "Encoding into binary indicators (0 and 1) provides clean numeric signal for modeling."
        )

    elif kind == "constant":
        const_val = evidence.get("constant_value")
        return (
            f"Column '{col}' has zero variance: all non-null entries have the single identical value {repr(const_val)}. "
            "Zero-variance features offer zero predictive power, degrade optimization convergence, and waste memory."
        )

    elif kind == "id_like":
        pct = evidence.get("unique_ratio_pct", 100.0)
        return (
            f"Column '{col}' has {pct}% unique values and resembles an identifier (ID/UUID). "
            "Identifiers should not be passed to predictive models because models can memorize row identity "
            "rather than generalizable patterns."
        )

    elif kind == "high_cardinality":
        n_unique = evidence.get("unique_count", 0)
        return (
            f"Column '{col}' contains {n_unique} distinct categorical values. "
            "High cardinality categories cause one-hot encoding feature explosion or severe sparsity."
        )

    elif kind == "mixed_type":
        samples = evidence.get("sample_values", [])
        sample_str = ", ".join(repr(v) for v in samples[:4])
        return (
            f"Column '{col}' contains a mixture of numeric values and text words (e.g., {sample_str}). "
            "Mixed data types trigger silent NaN coercion or unexpected type casting during model training."
        )

    return f"Type anomaly detected in column '{col}'."


def format_types_badge(evidence: Dict[str, Any]) -> str:
    """
    Returns a compact badge label for tables and diagnostic cards.
    """
    kind = evidence.get("kind", "")
    if kind == "text_number":
        symbols = evidence.get("detected_symbols", [])
        sym_label = f" ({', '.join(symbols[:2])})" if symbols else ""
        return f"TEXT NUMBER{sym_label}"
    elif kind == "ambiguous_date":
        return "AMBIGUOUS DATE (ASK USER)"
    elif kind == "date":
        return "DATE TIMESTAMP"
    elif kind == "boolean":
        return "BOOLEAN (YES/NO)"
    elif kind == "constant":
        return "CONSTANT (ZERO VARIANCE)"
    elif kind == "id_like":
        return "ID / IDENTIFIER"
    elif kind == "high_cardinality":
        return f"HIGH CARDINALITY ({evidence.get('unique_count', '')})"
    elif kind == "mixed_type":
        return "MIXED (NUM+TEXT)"
    return "TYPE ISSUE"
