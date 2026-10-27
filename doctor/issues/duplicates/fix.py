"""
Dataset Doctor: Duplicates Fixer (Phase 5).
Applies duplicate remediation strategies:
- keep_first: retain the first occurrence of each duplicate set.
- keep_last: retain the last occurrence of each duplicate set.
- drop_all_conflicting: drop all rows with identical features but contradictory targets.
- normalize_and_drop: normalize text casing/whitespace before deduplication.
- leave: retain all rows unchanged.
Safety rules:
- Pre-split deduplication cleans the dataset before partitioning.
- If called on test set (is_test=True), test rows are never dropped.
"""
from typing import Any, Dict, Optional
import pandas as pd

from doctor.base import Decision, Fixer


class DuplicateFixer(Fixer):
    """
    Applies deduplication transformations adhering to dataset safety rules.
    """
    id = "duplicates"
    rows_may_be_dropped_from = "train"  # Test set never loses rows in post-split steps

    def fit(self, train_df: pd.DataFrame, decision: Decision) -> Dict[str, Any]:
        """
        Captures parameters and configuration for deduplication.
        """
        return {
            "option_id": decision.option_id,
            "params": decision.params or {},
            "target_column": decision.params.get("target_column") if decision.params else None,
            "train_row_count": len(train_df),
        }

    def transform(self, df: pd.DataFrame, fitted: Dict[str, Any], is_test: bool = False) -> pd.DataFrame:
        """
        Applies deduplication to the dataset.
        If is_test is True, row-dropping is bypassed to preserve test set integrity.
        """
        if is_test:
            # Test set never loses rows
            return df.copy()

        option_id = fitted.get("option_id", "keep_first")

        if option_id == "keep_first":
            return df.drop_duplicates(keep="first")

        elif option_id == "keep_last":
            return df.drop_duplicates(keep="last")

        elif option_id == "drop_all_conflicting":
            target_col = fitted.get("target_column")
            if target_col and target_col in df.columns and len(df.columns) > 1:
                feat_cols = [c for c in df.columns if c != target_col]
                feat_dups = df[df.duplicated(subset=feat_cols, keep=False)]
                if not feat_dups.empty:
                    nunique = feat_dups.groupby(feat_cols)[target_col].transform("nunique")
                    conflicting_idx = feat_dups[nunique > 1].index
                    return df.drop(index=conflicting_idx)
            return df.copy()

        elif option_id == "normalize_and_drop":
            clean_df = df.copy()
            for col in clean_df.columns:
                s = clean_df[col]
                if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
                    clean_df[col] = s.astype(str).str.strip().str.lower()
            return clean_df.drop_duplicates(keep="first")

        elif option_id == "leave":
            return df.copy()

        return df.copy()
