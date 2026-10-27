"""
Dataset Doctor: Duplicates Advisor (Phase 5).
Evaluates remediation options for exact duplicate rows, conflicting duplicate rows,
near duplicates, and pre-balanced datasets.
Implements the Advisor interface from doctor/base.py.
"""
from typing import Any, Dict, List, Optional
import pandas as pd

from doctor.base import Advisor, DetectionContext, Issue, Option, Recommendation


class DuplicateAdvisor(Advisor):
    """
    Recommends transparent, leak-free duplicate resolution strategies.
    Rules:
    - Exact duplicates default to keep_first.
    - Conflicting duplicates default to drop_all_conflicting (eliminating label noise).
    - Pre-balanced datasets display a high-severity warning recommending split-first.
    """
    id = "duplicates"

    def advise(
        self,
        issue: Issue,
        train_df: pd.DataFrame,
        target: Optional[str] = None,
        ctx: Optional[DetectionContext] = None,
    ) -> Recommendation:
        kind = issue.evidence.get("kind", "exact")

        if kind == "exact":
            options = [
                Option(
                    id="keep_first",
                    label="Keep First Occurrence (Recommended)",
                    params={"keep": "first"},
                    explanation="Retain the first appearance of each duplicate set and drop redundant copies.",
                ),
                Option(
                    id="keep_last",
                    label="Keep Last Occurrence",
                    params={"keep": "last"},
                    explanation="Retain the last appearance of each duplicate set.",
                ),
                Option(
                    id="leave",
                    label="Leave Duplicates Untouched",
                    params={},
                    explanation="Retain all duplicate rows without modification.",
                ),
            ]
            return Recommendation(
                issue=issue,
                recommended="keep_first",
                reason=(
                    "Exact duplicate rows provide no additional learning signal and can cause data leakage "
                    "if split across training and test partitions. Keeping the first occurrence deduplicates "
                    "honestly without losing unique information."
                ),
                rule_hits=["deduplicate_exact_rows", "preserve_first_occurrence"],
                options=options,
                confidence="clear",
            )

        elif kind == "conflicting":
            target_col = issue.evidence.get("target_column") or target or "target"
            options = [
                Option(
                    id="drop_all_conflicting",
                    label="Drop All Conflicting Rows (Recommended)",
                    params={"target_column": target_col},
                    explanation=(
                        f"Drop all rows involved in conflicting sets where identical features produce different '{target_col}' values."
                    ),
                ),
                Option(
                    id="leave",
                    label="Keep Conflicting Rows (Manual Review)",
                    params={},
                    explanation="Retain contradictory rows for manual inspection.",
                ),
            ]
            return Recommendation(
                issue=issue,
                recommended="drop_all_conflicting",
                reason=(
                    f"Identical feature vectors with conflicting '{target_col}' labels represent contradictory training signal. "
                    "A supervised model cannot learn both outcomes simultaneously from identical inputs. "
                    "Dropping these ambiguous rows prevents corrupted model weights."
                ),
                rule_hits=["eliminate_conflicting_targets", "remove_label_noise"],
                options=options,
                confidence="clear",
            )

        elif kind == "near":
            options = [
                Option(
                    id="leave",
                    label="Leave Untouched (Recommended)",
                    params={},
                    explanation="Retain near duplicates until confirmed by domain inspection.",
                ),
                Option(
                    id="normalize_and_drop",
                    label="Normalize Text and Deduplicate",
                    params={},
                    explanation="Strip whitespace and convert text to lowercase before dropping duplicates.",
                ),
            ]
            return Recommendation(
                issue=issue,
                recommended="leave",
                reason=(
                    "Near-duplicates differ only by casing or whitespace. Inspect sample pairs to confirm "
                    "they represent true duplicates before applying normalization."
                ),
                rule_hits=["flag_near_duplicates_for_review"],
                options=options,
                confidence="clear",
            )

        elif kind == "pre_balanced":
            options = [
                Option(
                    id="leave",
                    label="Acknowledge Pre-Balancing Warning",
                    params={},
                    explanation="Re-split from original un-balanced raw data, balancing only on training set.",
                )
            ]
            return Recommendation(
                issue=issue,
                recommended="leave",
                reason=(
                    "Dataset exhibits 50/50 balance with duplicated minority rows, indicating synthetic oversampling "
                    "before train/test split. To prevent test leakage, split original raw data first, then apply balancing."
                ),
                rule_hits=["pre_split_balancing_detected", "enforce_split_first"],
                options=options,
                confidence="clear",
            )

        # Fallback default
        return Recommendation(
            issue=issue,
            recommended="leave",
            reason="No automated action recommended.",
            rule_hits=[],
            options=[Option(id="leave", label="Leave Untouched", params={}, explanation="Do nothing.")],
            confidence="clear",
        )
