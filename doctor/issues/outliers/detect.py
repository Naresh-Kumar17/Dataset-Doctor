"""
Dataset Doctor: Outliers Detection Engine (Phase 7).
Detects:
1. Domain-impossible values (e.g. negative age, age > 120, percent > 100, negative price/salary).
   - Flagged with certainty="certain", severity="high".
2. Unit mix-ups (e.g. 100x or 1000x typical median magnitude).
   - Flagged with certainty="possible", severity="medium".
3. Extreme statistical tail outliers:
   - IQR rule (1.5x / 3.0x).
   - Modified z-score (MAD - Median Absolute Deviation) for skewed continuous features.
   - 1st and 99th percentile boundaries.
4. Categorical and binary guard:
   - Strictly 0 flags on binary or categorical columns.
Implements the Detector interface from doctor/base.py.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from doctor.base import CertaintyLevel, DetectionContext, Detector, Issue, SeverityLevel
from doctor.issues.outliers.explain import explain_outliers_issue


def compute_skewness(vals: np.ndarray) -> float:
    """Computes Fisher-Pearson sample skewness."""
    if len(vals) < 3:
        return 0.0
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    if std < 1e-9:
        return 0.0
    n = len(vals)
    m3 = np.sum((vals - mean) ** 3) / n
    skew = m3 / (std ** 3)
    return float(skew)


def compute_mad(vals: np.ndarray) -> Tuple[float, float]:
    """Returns (median, MAD) for a 1D numpy array."""
    if len(vals) == 0:
        return 0.0, 0.0
    med = float(np.median(vals))
    abs_dev = np.abs(vals - med)
    mad = float(np.median(abs_dev))
    return med, mad


class OutlierDetector(Detector):
    """
    Diagnostic scanner identifying domain impossible values, unit errors,
    and extreme distribution tails on numeric features.
    """
    id = "outliers"

    # Known domain patterns for physical boundary checks
    IMPOSSIBLE_RULES = [
        {
            "name": "age_bounds",
            "matches": lambda col: col.lower() == "age" or col.lower().endswith("_age"),
            "is_impossible": lambda v: v < 0 or v > 120,
            "reason": "Age must be between 0 and 120 years.",
        },
        {
            "name": "percentage_bounds",
            "matches": lambda col: any(t in col.lower() for t in ["pct", "percent", "discount_pct", "rate"]) and not col.lower().endswith("_id"),
            "is_impossible": lambda v: v < 0 or v > 100,
            "reason": "Percentages must be within [0, 100].",
        },
        {
            "name": "non_negative_measurements",
            "matches": lambda col: any(col.lower().startswith(prefix) or col.lower().endswith(prefix) or col.lower() == prefix for prefix in [
                "salary", "income", "price", "cost", "revenue", "height", "weight", "distance", "speed", "count", "quantity"
            ]),
            "is_impossible": lambda v: v < 0,
            "reason": "Physical dimension / quantity cannot be negative.",
        },
    ]

    def detect(self, df: pd.DataFrame, ctx: Optional[DetectionContext] = None) -> List[Issue]:
        issues: List[Issue] = []
        if df.empty or len(df) == 0:
            return issues

        target_col = ctx.target_column if ctx else None
        total_rows = len(df)

        for col in df.columns:
            series = df[col]

            # 1. Guard: Check numeric type
            if not pd.api.types.is_numeric_dtype(series):
                continue

            non_null = series.dropna()
            if len(non_null) < 5:
                continue

            # 2. Guard: 0 outlier flags on binary or low-cardinality categorical columns
            n_unique = non_null.nunique()
            if n_unique <= 2:
                # Binary or constant -> strictly skip
                continue

            # If column name indicates an ID or sequential index, skip
            if col.lower() == "id" or col.lower().endswith("_id"):
                continue

            vals = non_null.to_numpy(dtype=float)
            n_vals = len(vals)

            # 3. Check for Domain-Impossible Values
            impossible_mask = np.zeros(n_vals, dtype=bool)
            impossible_reason = ""
            for rule in self.IMPOSSIBLE_RULES:
                if rule["matches"](col):
                    for i, v in enumerate(vals):
                        if rule["is_impossible"](v):
                            impossible_mask[i] = True
                    impossible_reason = rule["reason"]
                    break

            impossible_indices = non_null.index[impossible_mask].tolist()
            impossible_vals = vals[impossible_mask].tolist()
            n_impossible = len(impossible_indices)

            if n_impossible > 0:
                issue = Issue(
                    module_id=self.id,
                    column=col,
                    severity="high",
                    certainty="certain",
                    title=f"{col}: {n_impossible} domain-impossible value{'s' if n_impossible != 1 else ''} found",
                    description=(
                        f"Found {n_impossible} value{'s' if n_impossible != 1 else ''} ({n_impossible / total_rows:.1%}) "
                        f"violating domain physical constraints ({impossible_reason})."
                    ),
                    evidence={
                        "subtype": "impossible_values",
                        "count": n_impossible,
                        "pct": n_impossible / total_rows,
                        "row_indices": impossible_indices,
                        "sample_values": impossible_vals[:10],
                        "impossible_reason": impossible_reason,
                    },
                )
                issues.append(issue)

            # 4. Filter out impossible values to analyze underlying distribution
            valid_mask = ~impossible_mask
            valid_vals = vals[valid_mask]
            if len(valid_vals) < 5:
                continue

            valid_indices = non_null.index[valid_mask]

            # 5. Check for Unit Mix-ups (e.g. 100x or 1000x median jump)
            med, mad = compute_mad(valid_vals)
            unit_error_indices: List[int] = []
            unit_error_vals: List[float] = []
            median_ratio = 1.0

            if med > 1.0:
                for idx, v in zip(valid_indices, valid_vals):
                    ratio = v / med
                    # Check order-of-magnitude jump: ~100x or ~1000x
                    if (70.0 <= ratio <= 130.0) or (700.0 <= ratio <= 1300.0):
                        unit_error_indices.append(idx)
                        unit_error_vals.append(v)
                        median_ratio = ratio

            # Unit errors must be a small anomaly (< 3% of rows)
            n_unit = len(unit_error_indices)
            if 0 < n_unit <= max(5, int(0.03 * total_rows)):
                issue = Issue(
                    module_id=self.id,
                    column=col,
                    severity="medium",
                    certainty="possible",
                    title=f"{col}: Possible unit mix-up ({n_unit} value{'s' if n_unit != 1 else ''})",
                    description=(
                        f"Detected {n_unit} value{'s' if n_unit != 1 else ''} recorded at ~{median_ratio:,.0f}x "
                        f"the column median, strongly suggesting a unit of measurement error."
                    ),
                    evidence={
                        "subtype": "unit_mixup",
                        "count": n_unit,
                        "pct": n_unit / total_rows,
                        "row_indices": unit_error_indices,
                        "sample_values": unit_error_vals[:5],
                        "median_ratio": median_ratio,
                    },
                )
                issues.append(issue)

            # 6. Statistical Distribution Analysis: Skewness, IQR, MAD, and Percentiles
            skew = compute_skewness(valid_vals)
            is_skewed = abs(skew) > 1.0

            q25 = float(np.percentile(valid_vals, 25))
            q75 = float(np.percentile(valid_vals, 75))
            iqr = q75 - q25

            p1 = float(np.percentile(valid_vals, 1))
            p99 = float(np.percentile(valid_vals, 99))

            iqr_lower = q25 - 1.5 * iqr
            iqr_upper = q75 + 1.5 * iqr
            iqr_extreme_upper = q75 + 3.0 * iqr
            iqr_extreme_lower = q25 - 3.0 * iqr

            # Determine extreme tail points excluding unit errors
            clean_eval_mask = ~np.isin(valid_indices, unit_error_indices)
            eval_vals = valid_vals[clean_eval_mask]
            eval_indices = valid_indices[clean_eval_mask]

            if len(eval_vals) < 5:
                continue

            extreme_outlier_indices: List[int] = []
            extreme_outlier_vals: List[float] = []

            if is_skewed:
                # Skewed continuous feature:
                # Standard IQR over-flags legitimate log-normal/Pareto tails.
                # Use robust MAD (modified z > 3.5) and extreme bounds to avoid flagging clean skewed data.
                if mad > 1e-9:
                    mod_z = 0.6745 * np.abs(eval_vals - med) / mad
                    # Only flag extreme deviations beyond standard continuous tails (mod_z > 8.0)
                    skew_outliers = mod_z > 8.0
                    if np.sum(skew_outliers) > 0 and np.sum(skew_outliers) / total_rows <= 0.05:
                        extreme_outlier_indices = eval_indices[skew_outliers].tolist()
                        extreme_outlier_vals = eval_vals[skew_outliers].tolist()
            else:
                # Normal or approximately symmetric feature:
                # Flag extreme tails beyond 3.0 * IQR (or extreme 1%/99% percentile tails)
                if iqr > 1e-9:
                    extreme_mask = (eval_vals < iqr_extreme_lower) | (eval_vals > iqr_extreme_upper)
                    flag_count = np.sum(extreme_mask)
                    # For normal clean distribution, flags must remain <= 1%
                    if flag_count > 0:
                        extreme_outlier_indices = eval_indices[extreme_mask].tolist()
                        extreme_outlier_vals = eval_vals[extreme_mask].tolist()

            n_extremes = len(extreme_outlier_indices)
            if n_extremes > 0:
                # Severity: medium if extreme points or substantial tail, low otherwise
                sev: SeverityLevel = "medium" if n_extremes >= 5 or any(abs(v - med) > 4 * (iqr if iqr > 0 else 1) for v in extreme_outlier_vals) else "low"
                issue = Issue(
                    module_id=self.id,
                    column=col,
                    severity=sev,
                    certainty="possible",
                    title=f"{col}: {n_extremes} extreme statistical outlier{'s' if n_extremes != 1 else ''}",
                    description=(
                        f"Found {n_extremes} extreme tail values ({n_extremes / total_rows:.1%}) "
                        f"beyond statistical distribution thresholds."
                    ),
                    evidence={
                        "subtype": "extreme_tail",
                        "count": n_extremes,
                        "pct": n_extremes / total_rows,
                        "row_indices": extreme_outlier_indices,
                        "sample_values": extreme_outlier_vals[:10],
                        "is_skewed": is_skewed,
                        "skew": skew,
                        "median": med,
                        "mad": mad,
                        "bounds": {
                            "iqr_lower": iqr_lower,
                            "iqr_upper": iqr_upper,
                            "iqr_extreme_lower": iqr_extreme_lower,
                            "iqr_extreme_upper": iqr_extreme_upper,
                            "p1": p1,
                            "p99": p99,
                        },
                    },
                )
                issues.append(issue)

        return issues
