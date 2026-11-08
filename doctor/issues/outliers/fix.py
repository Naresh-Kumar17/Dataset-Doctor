"""
Dataset Doctor: Outliers Remediation Engine (Phase 8).
Implements principled, leak-free transformations for outlier issues:
1. set_to_null: Replaces impossible values with NaN so Null Advisor can impute.
2. clip_iqr: Learns [Q1 - 1.5*IQR, Q3 + 1.5*IQR] strictly from train; clips both train and test.
3. clip_percentile: Learns [P1, P99] strictly from train; clips both train and test.
4. log_transform: Applies log1p(x + shift) where shift is learned from train.
5. drop_rows: Drops outlier rows strictly from TRAIN partition. TEST ROWS ARE NEVER DROPPED.
6. leave: Preserves original values without modification.
"""
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from doctor.base import Decision, Fixer


class OutlierFixer(Fixer):
    """
    Executes outlier transformations learned strictly from training data.
    """
    id: str = "outliers_fixer"
    rows_may_be_dropped_from: str = "train"

    def fit(self, train_df: pd.DataFrame, decision: Decision) -> Dict[str, Any]:
        """
        Learns transformation parameters (bounds, quantiles, shifts) strictly from train_df.
        """
        col = decision.column
        option = decision.option_id
        params = decision.params or {}

        if col not in train_df.columns:
            return {}

        series = train_df[col].dropna()
        if len(series) == 0:
            return {}

        vals = series.to_numpy(dtype=float)

        if option == "clip_iqr":
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            iqr = q75 - q25
            lower = float(params.get("lower", q25 - 1.5 * iqr))
            upper = float(params.get("upper", q75 + 1.5 * iqr))
            return {"lower": lower, "upper": upper}

        elif option == "clip_percentile":
            p1 = float(params.get("lower", np.percentile(vals, 1)))
            p99 = float(params.get("upper", np.percentile(vals, 99)))
            return {"lower": p1, "upper": p99}

        elif option == "log_transform":
            min_val = float(np.min(vals))
            shift = float(abs(min_val) + 1.0) if min_val < 0 else 0.0
            return {"shift": shift}

        elif option == "set_to_null":
            # Learn or extract impossible condition
            condition = params.get("condition", "auto")
            min_bound = params.get("min_bound", None)
            max_bound = params.get("max_bound", None)
            return {
                "condition": condition,
                "min_bound": min_bound,
                "max_bound": max_bound,
                "invalid_values": params.get("invalid_values", []),
            }

        elif option == "drop_rows":
            # Compute train bounds outside of which train rows will be dropped
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            iqr = q75 - q25
            lower = float(params.get("lower", q25 - 3.0 * iqr))
            upper = float(params.get("upper", q75 + 3.0 * iqr))
            return {"lower": lower, "upper": upper}

        return {}

    def transform(
        self,
        df: pd.DataFrame,
        decision: Decision,
        fitted_params: Dict[str, Any],
        is_test: bool = False,
    ) -> pd.DataFrame:
        """
        Applies fitted outlier remediation to df.
        Test integrity rule: is_test=True NEVER drops rows.
        """
        col = decision.column
        option = decision.option_id
        out_df = df.copy()

        if col not in out_df.columns or option == "leave":
            return out_df

        if option == "set_to_null":
            # Convert impossible/invalid values to NaN
            min_bound = fitted_params.get("min_bound")
            max_bound = fitted_params.get("max_bound")
            invalid_vals = fitted_params.get("invalid_values", [])

            # Check age rule if col is age
            if col.lower() == "age" or col.lower().endswith("_age"):
                mask = (out_df[col] < 0) | (out_df[col] > 120)
                out_df.loc[mask, col] = np.nan
            elif any(t in col.lower() for t in ["pct", "percent", "discount_pct", "rate"]):
                mask = (out_df[col] < 0) | (out_df[col] > 100)
                out_df.loc[mask, col] = np.nan
            elif min_bound is not None or max_bound is not None:
                mask = pd.Series(False, index=out_df.index)
                if min_bound is not None:
                    mask = mask | (out_df[col] < min_bound)
                if max_bound is not None:
                    mask = mask | (out_df[col] > max_bound)
                out_df.loc[mask, col] = np.nan
            elif invalid_vals:
                out_df.loc[out_df[col].isin(invalid_vals), col] = np.nan
            else:
                # Default fallback: check physical impossibility
                if col.lower() in ["age", "height_cm", "salary", "price"]:
                    if col.lower() == "age":
                        out_df.loc[(out_df[col] < 0) | (out_df[col] > 120), col] = np.nan
                    elif out_df[col].dtype.kind in "biufc":
                        out_df.loc[out_df[col] < 0, col] = np.nan

        elif option in ("clip_iqr", "clip_percentile"):
            lower = fitted_params.get("lower")
            upper = fitted_params.get("upper")
            if lower is not None and upper is not None:
                out_df[col] = out_df[col].clip(lower=lower, upper=upper)

        elif option == "log_transform":
            shift = fitted_params.get("shift", 0.0)
            numeric_col = pd.to_numeric(out_df[col], errors="coerce")
            out_df[col] = np.log1p(numeric_col + shift)

        elif option == "drop_rows":
            # STRICT GUARD: Test rows are NEVER dropped!
            if is_test:
                return out_df

            lower = fitted_params.get("lower")
            upper = fitted_params.get("upper")
            if lower is not None and upper is not None:
                keep_mask = (out_df[col] >= lower) & (out_df[col] <= upper)
                out_df = out_df[keep_mask].copy()

        return out_df
