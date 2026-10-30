"""
Dataset Doctor: Data Types Advisor (Phase 6).
Evaluates remediation options for:
- Text numbers (convert to float/int stripping symbols)
- Ambiguous dates (prompts user for day-first vs month-first, never guesses silently)
- Unambiguous dates (parse datetime)
- Booleans (encode binary 0/1)
- Constant zero-variance columns (drop column)
- ID-like identifiers (drop column)
- Mixed types (coerce numeric or leave)
Implements the Advisor interface from doctor/base.py.
"""
from typing import Any, Dict, List, Optional
import pandas as pd

from doctor.base import Advisor, DetectionContext, Issue, Option, Recommendation


class TypeAdvisor(Advisor):
    """
    Recommends safe, reproducible data type corrections and asks the user
    for ambiguous format clarifications.
    """
    id = "types"

    def advise(
        self,
        issue: Issue,
        train_df: pd.DataFrame,
        target: Optional[str] = None,
        ctx: Optional[DetectionContext] = None,
    ) -> Recommendation:
        kind = issue.evidence.get("kind", "text_number")
        col = issue.column or "column"

        if kind == "text_number":
            symbols = issue.evidence.get("detected_symbols", [])
            sym_desc = ", ".join(repr(s) for s in symbols) if symbols else "formatting symbols"
            options = [
                Option(
                    id="convert_to_numeric",
                    label="Convert to Numeric (Recommended)",
                    params={"strip_symbols": True, "target_dtype": "float64"},
                    explanation=f"Strip symbols ({sym_desc}) and parse into clean floating point numbers.",
                ),
                Option(
                    id="leave",
                    label="Leave as Text (Categorical)",
                    params={},
                    explanation="Retain column as raw strings without type casting.",
                ),
            ]
            return Recommendation(
                issue=issue,
                recommended="convert_to_numeric",
                reason=(
                    f"Column '{col}' represents numeric magnitudes masked by symbols ({sym_desc}). "
                    "Converting to numeric enables mathematical comparison, scaling, and continuous model fitting."
                ),
                rule_hits=["strip_symbols_and_parse_numeric"],
                options=options,
                confidence="clear",
            )

        elif kind == "ambiguous_date":
            # Safety Rule: Ambiguous dates are flagged and the user is asked, never guessed!
            options = [
                Option(
                    id="parse_date_day_first",
                    label="Parse as Day-First (DD/MM/YYYY)",
                    params={"dayfirst": True},
                    explanation="Interpret date components assuming the day comes before the month.",
                ),
                Option(
                    id="parse_date_month_first",
                    label="Parse as Month-First (MM/DD/YYYY)",
                    params={"dayfirst": False},
                    explanation="Interpret date components assuming the month comes before the day.",
                ),
                Option(
                    id="leave",
                    label="Leave as Raw Text",
                    params={},
                    explanation="Keep date strings untouched for manual handling.",
                ),
            ]
            return Recommendation(
                issue=issue,
                recommended="parse_date_day_first",
                reason=(
                    f"Date strings in '{col}' have day and month values both <= 12. "
                    "Automatic format determination cannot guarantee correctness. "
                    "User must confirm whether Day-First (DD/MM/YYYY) or Month-First (MM/DD/YYYY) was intended."
                ),
                rule_hits=["ambiguous_date_requires_user_disambiguation"],
                options=options,
                confidence="close call",  # Explicitly close call / user decision required
            )

        elif kind == "date":
            fmt = issue.evidence.get("detected_format", "standard")
            options = [
                Option(
                    id="parse_date",
                    label="Parse Datetime (Recommended)",
                    params={"format": fmt},
                    explanation=f"Convert string timestamps into standard pandas datetime64 using format {fmt}.",
                ),
                Option(
                    id="leave",
                    label="Leave as Text",
                    params={},
                    explanation="Keep original date strings unmodified.",
                ),
            ]
            return Recommendation(
                issue=issue,
                recommended="parse_date",
                reason=(
                    f"Column '{col}' matches unambiguous date format {fmt}. "
                    "Parsing as datetime enables chronological sorting, temporal splitting, and feature extraction."
                ),
                rule_hits=["parse_unambiguous_datetime"],
                options=options,
                confidence="clear",
            )

        elif kind == "boolean":
            options = [
                Option(
                    id="encode_boolean",
                    label="Encode to Binary (0 / 1) (Recommended)",
                    params={"true_value": 1, "false_value": 0},
                    explanation="Map affirmative values (yes/true/y) to 1 and negative values (no/false/n) to 0.",
                ),
                Option(
                    id="leave",
                    label="Leave as String Labels",
                    params={},
                    explanation="Keep original text values without integer encoding.",
                ),
            ]
            return Recommendation(
                issue=issue,
                recommended="encode_boolean",
                reason=(
                    f"Column '{col}' consists of binary yes/no flags. "
                    "Encoding into 0/1 integers provides a clean, standard numerical representation for ML models."
                ),
                rule_hits=["encode_binary_boolean_flags"],
                options=options,
                confidence="clear",
            )

        elif kind == "constant":
            const_val = issue.evidence.get("constant_value")
            options = [
                Option(
                    id="drop_column",
                    label="Drop Constant Column (Recommended)",
                    params={"column": col},
                    explanation=f"Remove column '{col}' because it has zero variance (always {repr(const_val)}).",
                ),
                Option(
                    id="leave",
                    label="Keep Constant Column",
                    params={},
                    explanation="Retain zero-variance feature in training dataset.",
                ),
            ]
            return Recommendation(
                issue=issue,
                recommended="drop_column",
                reason=(
                    f"Column '{col}' has zero variance: every non-null row has value {repr(const_val)}. "
                    "Zero-variance features provide zero mutual information to models and unnecessarily inflate memory usage."
                ),
                rule_hits=["drop_zero_variance_columns"],
                options=options,
                confidence="clear",
            )

        elif kind == "id_like":
            options = [
                Option(
                    id="drop_column",
                    label="Drop Identifier Column (Recommended)",
                    params={"column": col},
                    explanation="Drop high-cardinality unique IDs to avoid overfitting and target memorization.",
                ),
                Option(
                    id="leave",
                    label="Keep Identifier",
                    params={},
                    explanation="Retain identifier column for tracking or join keys.",
                ),
            ]
            return Recommendation(
                issue=issue,
                recommended="drop_column",
                reason=(
                    f"Column '{col}' has near 100% unique values and functions as an identifier. "
                    "Identifiers provide no generalizable learning patterns and cause tree models to memorize row identities."
                ),
                rule_hits=["drop_high_cardinality_identifiers"],
                options=options,
                confidence="clear",
            )

        elif kind == "mixed_type":
            options = [
                Option(
                    id="coerce_numeric",
                    label="Coerce to Numeric (Invalid to NaN)",
                    params={"errors": "coerce"},
                    explanation="Parse numeric entries and convert text words to NaN (to be handled by Null Advisor).",
                ),
                Option(
                    id="leave",
                    label="Leave as Categorical Text (Recommended)",
                    params={},
                    explanation="Treat column as a categorical string feature to avoid dropping text labels.",
                ),
            ]
            return Recommendation(
                issue=issue,
                recommended="leave",
                reason=(
                    f"Column '{col}' contains genuine numbers mixed with text words. "
                    "Coercing blindly to numeric would erase descriptive text labels. "
                    "Manual inspection is advised before forcing numeric conversion."
                ),
                rule_hits=["preserve_mixed_labels_or_review"],
                options=options,
                confidence="close call",
            )

        # Fallback default
        return Recommendation(
            issue=issue,
            recommended="leave",
            reason=f"Type issue detected in '{col}'. Retaining column unchanged.",
            rule_hits=[],
            options=[Option(id="leave", label="Leave Untouched", params={}, explanation="No change applied.")],
            confidence="not enough data",
        )
