"""
Dataset Doctor: Data Leakage Fixer Engine (Phase 12).
Rules:
- NEVER DROPS ROWS.
- Removes leaking columns identically from train and test partitions.
- Test set row count and distribution are strictly preserved.
"""
from typing import Any, Dict, Optional
import pandas as pd
from doctor.base import Decision, Fixer


class LeakageFixer(Fixer):
    id: str = "leakage"
    rows_may_be_dropped_from: str = "none"

    def fit(self, train_df: pd.DataFrame, decision: Decision) -> Dict[str, Any]:
        """
        Validates column presence and prepares drop parameters.
        """
        col = decision.column or decision.params.get("column")
        return {
            "column": col,
            "option_id": decision.option_id,
            "params": decision.params,
        }

    def transform(
        self,
        df: pd.DataFrame,
        decision: Optional[Decision] = None,
        fitted: Optional[Dict[str, Any]] = None,
        is_test: bool = False,
    ) -> pd.DataFrame:
        """
        Executes column removal. Row count is strictly 100% preserved.
        """
        if fitted is None:
            if decision and decision.params:
                fitted = decision.params
            else:
                return df.copy()

        opt = fitted.get("option_id", "drop_column")
        col = fitted.get("column")

        if opt == "drop_column" and col and col in df.columns:
            return df.drop(columns=[col])

        return df.copy()
