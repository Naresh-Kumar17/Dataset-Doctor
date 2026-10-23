"""
Dataset Doctor: Missing Values Transformer & Fixer.
Rules (Phase 4):
- fit() learns parameters strictly from training data only.
- transform() applies learned parameters to train or test.
- Row-dropping is strictly restricted to train sets; test rows are NEVER dropped.
- On test data when drop_rows was selected for train, missing test values are filled
  with the train-fitted fallback to preserve test set row counts and schema integrity.
- Test values change ONLY in cells that were null (or confirmed tokens/sentinels).
"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from doctor.base import Decision, Fixer


class NullFixer(Fixer):
    """
    Applies data transformations to remediate missing values in a leak-free manner.
    """
    id: str = "nulls_fixer"
    rows_may_be_dropped_from: str = "train"

    def fit(self, train_df: pd.DataFrame, decision: Decision) -> Dict[str, Any]:
        """
        Learns transformation parameters strictly from train_df.
        """
        col = decision.column
        option = decision.option_id
        params = decision.params or {}

        if not col or col not in train_df.columns:
            return {"option": "leave", "column": col}

        series = train_df[col]
        # Replace explicit hidden tokens / sentinels if specified in decision
        tokens_to_null = params.get("tokens_to_null", [])
        if tokens_to_null:
            series = series.replace(tokens_to_null, np.nan)

        valid_vals = series.dropna()
        is_numeric = pd.api.types.is_numeric_dtype(series)

        # 1. Calculate train-only baseline statistics
        if is_numeric and not valid_vals.empty:
            train_median = float(valid_vals.median())
            train_mean = float(valid_vals.mean())
            mode_s = valid_vals.mode()
            train_mode = float(mode_s.iloc[0]) if not mode_s.empty else train_median
        else:
            train_median = None
            train_mean = None
            mode_s = valid_vals.mode() if not valid_vals.empty else pd.Series(["UNKNOWN"])
            train_mode = mode_s.iloc[0] if not mode_s.empty else "UNKNOWN"

        fitted: Dict[str, Any] = {
            "option": option,
            "column": col,
            "is_numeric": is_numeric,
            "tokens_to_null": tokens_to_null,
            "train_median": train_median,
            "train_mean": train_mean,
            "train_mode": train_mode,
        }

        # 2. Option-specific parameter determination
        if option == "drop_rows":
            # Test fallback ensures test rows are never dropped
            fallback = train_median if is_numeric else train_mode
            fitted["test_fallback"] = fallback

        elif option == "drop_column":
            pass  # Drops column entirely

        elif option == "mean":
            fitted["fill_value"] = train_mean if train_mean is not None else 0.0

        elif option == "median":
            fitted["fill_value"] = train_median if train_median is not None else 0.0

        elif option == "mode":
            fitted["fill_value"] = train_mode

        elif option == "constant":
            user_val = params.get("value")
            if user_val is None:
                user_val = 0 if is_numeric else "Unknown"
            elif is_numeric:
                try:
                    user_val = float(user_val)
                except (ValueError, TypeError):
                    user_val = 0.0
            fitted["fill_value"] = user_val

        elif option == "missing_label":
            fitted["fill_value"] = "MISSING"

        elif option == "add_indicator":
            indicator_col = f"{col}_was_missing"
            sub_fill = params.get("sub_fill", "median" if is_numeric else "mode")
            if sub_fill == "mean" and train_mean is not None:
                fill_val = train_mean
            elif sub_fill == "mode":
                fill_val = train_mode
            else:
                fill_val = train_median if train_median is not None else train_mode

            fitted["indicator_col"] = indicator_col
            fitted["fill_value"] = fill_val

        elif option == "ffill_interpolate":
            fallback = train_median if is_numeric else train_mode
            fitted["fallback"] = fallback

        elif option == "leave":
            pass

        return fitted

    def transform(self, df: pd.DataFrame, fitted: Dict[str, Any], is_test: bool = False) -> pd.DataFrame:
        """
        Applies learned parameters to train or test set.
        """
        col = fitted.get("column")
        option = fitted.get("option", "leave")
        if not col or col not in df.columns or option == "leave":
            return df.copy()

        df_out = df.copy()

        # Step 1: Normalize any designated tokens/sentinels to NaN first
        tokens_to_null = fitted.get("tokens_to_null", [])
        if tokens_to_null:
            df_out[col] = df_out[col].replace(tokens_to_null, np.nan)

        # Step 2: Apply chosen remediation
        if option == "drop_rows":
            if not is_test:
                # Training partition: drop rows with nulls in this column
                df_out = df_out.dropna(subset=[col])
            else:
                # Test partition: strictly preserve 100% of test rows; impute with train fallback
                df_out[col] = df_out[col].fillna(fitted.get("test_fallback"))

        elif option == "drop_column":
            if col in df_out.columns:
                df_out = df_out.drop(columns=[col])

        elif option in ("mean", "median", "mode", "constant", "missing_label"):
            fill_val = fitted.get("fill_value")
            # Preserve column dtype compatibility if possible
            if fitted.get("is_numeric") and fill_val is not None:
                try:
                    fill_val = float(fill_val)
                except (ValueError, TypeError):
                    pass
            df_out[col] = df_out[col].fillna(fill_val)

        elif option == "add_indicator":
            indicator_col = fitted.get("indicator_col", f"{col}_was_missing")
            # Add 0/1 binary indicator
            df_out[indicator_col] = df_out[col].isna().astype(int)
            # Impute the original column with learned fill_value
            fill_val = fitted.get("fill_value")
            df_out[col] = df_out[col].fillna(fill_val)

        elif option == "ffill_interpolate":
            fallback = fitted.get("fallback")
            if fitted.get("is_numeric"):
                df_out[col] = (
                    df_out[col]
                    .interpolate(method="linear", limit_direction="both")
                    .ffill()
                    .bfill()
                    .fillna(fallback if fallback is not None else 0.0)
                )
            else:
                df_out[col] = df_out[col].ffill().bfill().fillna(fallback if fallback is not None else "MISSING")

        return df_out
