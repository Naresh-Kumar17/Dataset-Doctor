"""
Dataset Doctor: Data Leakage Advisor Engine (Phase 12).
Recommends:
- Evidence-based column drop for target clones, derivative scalar multiples, index leaks, and post-event proxies.
- Time-ordered split recommendation for chronological monotonic sequences.
- User acknowledgement override options for legitimate domain features.
"""
from typing import Any, Dict, List, Optional
import pandas as pd
from doctor.base import Advisor, DetectionContext, Issue, Option, Recommendation


class LeakageAdvisor(Advisor):
    id: str = "leakage"

    def advise(
        self,
        issue: Issue,
        train_df: pd.DataFrame,
        target: Optional[str] = None,
        ctx: Optional[DetectionContext] = None,
        measure: bool = True,
    ) -> Recommendation:
        ev = issue.evidence or {}
        subtype = ev.get("subtype", "target_copy")
        col = issue.column or "column"
        target_col = target or ev.get("target_column", "target")

        if subtype == "time_order_risk":
            options = [
                Option(
                    id="time_split",
                    label="Enforce Chronological Time-Split",
                    params={"time_column": col},
                    explanation=(
                        f"Sorts dataset by '{col}' and reserves the most recent 20% strictly for testing, "
                        f"preventing future information from leaking into past predictions."
                    ),
                ),
                Option(
                    id="leave",
                    label="Keep Standard Random Split",
                    params={},
                    explanation="Retains random shuffling if you have domain proof that observations are time-independent.",
                ),
            ]
            recommended = "time_split"
            reason = (
                f"Column '{col}' exhibits monotonic chronological progression. "
                f"Switching to a time-ordered partition prevents future-to-past data snooping."
            )
            confidence = "clear"

        else:
            options = [
                Option(
                    id="drop_column",
                    label=f"Drop Column '{col}' (Leak-Free)",
                    params={"column": col},
                    explanation=(
                        f"Permanently removes '{col}' from training and evaluation features. "
                        f"Zero rows are dropped; test partition size is 100% preserved."
                    ),
                ),
                Option(
                    id="keep_with_acknowledgement",
                    label=f"Keep Column '{col}' (User Acknowledgement)",
                    params={"column": col},
                    explanation=(
                        f"Leaves '{col}' in the dataset. Choose this only if you can mathematically prove "
                        f"that this value is fully finalized and available at real-time inference before '{target_col}' occurs."
                    ),
                ),
            ]
            recommended = "drop_column"
            reason = (
                f"Column '{col}' is classified as a {subtype.replace('_', ' ')} (evidence: {issue.title}). "
                f"Dropping it prevents catastrophic generalization failure in production."
            )
            confidence = "clear"

        return Recommendation(
            issue=issue,
            recommended=recommended,
            reason=reason,
            rule_hits=[f"subtype_{subtype}", f"severity_{issue.severity}"],
            options=options,
            confidence=confidence,
        )
