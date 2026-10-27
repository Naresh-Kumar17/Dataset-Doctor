"""
Dataset Doctor: Class Imbalance Fixer & Resampling Engine (Phase 9).
Rules:
- RESAMPLING IS STRICTLY RESTRICTED TO TRAINING DATA ONLY.
- The test partition is never resampled; test row count and class ratio are strictly preserved.
- Supported strategies: 'leave', 'class_weight', 'random_oversample', 'smote', 'random_undersample', 'threshold_tuning'.
- SMOTE safety guard: Refuses to run on unencoded categorical/object features, warning user to encode first.
"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from doctor.base import Decision, Fixer

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


class ImbalanceFixer(Fixer):
    id: str = "imbalance"
    rows_may_be_dropped_from: str = "train"

    def fit(self, train_df: pd.DataFrame, decision: Decision) -> Dict[str, Any]:
        """
        Validates remediation strategy and prepares resampling parameters on train_df.
        """
        target_col = decision.column or decision.params.get("target_column")
        if not target_col or target_col not in train_df.columns:
            # Fallback to candidate target
            candidates = [c for c in train_df.columns if "target" in c.lower()]
            target_col = candidates[0] if candidates else train_df.columns[-1]

        y = train_df[target_col]
        counts = y.value_counts()
        minority_class = counts.index[-1]

        # Check for unencoded categoricals
        feature_cols = [c for c in train_df.columns if c != target_col]
        non_numeric_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train_df[c])]

        strategy = decision.option_id
        if strategy == "smote" and non_numeric_cols:
            raise ValueError(
                f"SMOTE cannot be applied directly to unencoded non-numeric features: {non_numeric_cols}. "
                f"Encode categoricals first (or choose 'random_oversample' or 'class_weight')."
            )

        return {
            "target_column": target_col,
            "strategy": strategy,
            "minority_class": str(minority_class),
            "original_counts": {str(k): int(v) for k, v in counts.items()},
            "non_numeric_cols": non_numeric_cols,
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
        Applies remediation. If is_test=True, strictly returns untouched copy.
        """
        # TEST SET INVARIANT: NEVER RESAMPLE TEST SET
        if is_test:
            return df.copy()

        if fitted is None:
            if decision and decision.params:
                fitted = decision.params
            else:
                return df.copy()

        strategy = fitted.get("strategy", "leave")
        target_col = fitted.get("target_column")

        if strategy in ["leave", "class_weight", "threshold_tuning"]:
            # Data rows remain unaltered (class weights and threshold tuning are model-level)
            return df.copy()

        if not target_col or target_col not in df.columns:
            return df.copy()

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # If only 1 class or empty, return as is
        if y.nunique() < 2:
            return df.copy()

        seed = 42

        if strategy == "random_oversample":
            if HAS_IMBLEARN:
                ros = RandomOverSampler(random_state=seed)
                X_res, y_res = ros.fit_resample(X, y)
                df_res = pd.DataFrame(X_res, columns=X.columns)
                df_res[target_col] = y_res
                return df_res
            else:
                # Manual deterministic fallback
                max_count = y.value_counts().max()
                dfs = []
                for cls, group in df.groupby(target_col):
                    oversampled = group.sample(max_count, replace=True, random_state=seed)
                    dfs.append(oversampled)
                return pd.concat(dfs, ignore_index=True)

        elif strategy == "random_undersample":
            if HAS_IMBLEARN:
                rus = RandomUnderSampler(random_state=seed)
                X_res, y_res = rus.fit_resample(X, y)
                df_res = pd.DataFrame(X_res, columns=X.columns)
                df_res[target_col] = y_res
                return df_res
            else:
                # Manual deterministic fallback
                min_count = y.value_counts().min()
                dfs = []
                for cls, group in df.groupby(target_col):
                    undersampled = group.sample(min_count, replace=False, random_state=seed)
                    dfs.append(undersampled)
                return pd.concat(dfs, ignore_index=True)

        elif strategy == "smote":
            non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
            if non_numeric:
                raise ValueError(
                    f"SMOTE cannot be applied directly to unencoded features: {non_numeric}. "
                    f"Encode categoricals before applying SMOTE."
                )

            if HAS_IMBLEARN:
                # k_neighbors must not exceed minority class count - 1
                min_c = y.value_counts().min()
                k_neighbors = min(5, max(1, min_c - 1))
                smote = SMOTE(k_neighbors=k_neighbors, random_state=seed)
                X_res, y_res = smote.fit_resample(X, y)
                df_res = pd.DataFrame(X_res, columns=X.columns)
                df_res[target_col] = y_res
                return df_res
            else:
                raise RuntimeError("imbalanced-learn is required for SMOTE")

        return df.copy()
