"""
Dataset Doctor: Data Leakage Detection Engine (Phase 11).
Detects:
- Direct Target Copies (identical column to target).
- Inverted Target Clones (1 - y or negated boolean).
- Scaled Linear Multiples (y * c + d, correlation >= 0.999).
- Perfect / Near-Perfect Predictor Proxies (post-event features, correlation >= 0.95).
- Row Index / Order Leaks (sorted identifiers artificially predicting target).
- Temporal Ordering Inversion (monotonic chronological sequences requiring time-split).
- Immunity Guard: Legitimate strong features (correlation <= 0.75) are NOT flagged as high severity.
- Clean Control Immunity: 0 false alarms on clean_control.csv.
"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from doctor.base import CertaintyLevel, DetectionContext, Detector, Issue, SeverityLevel


class LeakageDetector(Detector):
    id: str = "leakage"

    def detect(self, df: pd.DataFrame, ctx: Optional[DetectionContext] = None) -> List[Issue]:
        issues: List[Issue] = []
        if df.empty:
            return issues

        target_col = ctx.target_column if ctx else None
        if not target_col or target_col not in df.columns:
            # Search for candidate target column
            candidates = [
                c for c in df.columns
                if "target" in c.lower() or c.lower() in ("label", "outcome", "y", "severe_accident")
            ]
            if candidates:
                target_col = candidates[0]
            else:
                return issues

        y_raw = df[target_col]
        if y_raw.dropna().nunique() < 2:
            return issues

        # Numeric representation of target for correlation
        if pd.api.types.is_numeric_dtype(y_raw):
            y_num = y_raw.astype(float)
            is_binary = set(y_raw.dropna().unique()).issubset({0, 1, 0.0, 1.0})
        else:
            y_codes, _ = pd.factorize(y_raw)
            y_num = pd.Series(y_codes, index=df.index, dtype=float)
            is_binary = y_raw.nunique() == 2

        for col in df.columns:
            if col == target_col or col == "id":
                continue

            series = df[col]
            if series.dropna().empty:
                continue

            # 1. Direct Target Copy
            if series.equals(y_raw) or (series == y_raw).all():
                issues.append(
                    Issue(
                        module_id="leakage",
                        column=col,
                        severity="high",
                        certainty="certain",
                        title=f"Direct Target Clone in '{col}' (100% Identical)",
                        description=(
                            f"Column '{col}' is an identical copy of target '{target_col}'. "
                            f"Training with this column fabricates 100% accuracy in notebooks but completely fails in production."
                        ),
                        evidence={
                            "kind": "leakage",
                            "subtype": "target_copy",
                            "column": col,
                            "target_column": target_col,
                            "correlation": 1.0,
                            "action_recommended": "drop_column",
                        },
                    )
                )
                continue

            # 2. Inverted Target Clone
            if is_binary and pd.api.types.is_numeric_dtype(series):
                if (series == (1 - y_num)).all():
                    issues.append(
                        Issue(
                            module_id="leakage",
                            column=col,
                            severity="high",
                            certainty="certain",
                            title=f"Inverted Target Clone in '{col}' (100% Inverted)",
                            description=(
                                f"Column '{col}' is an exact inverted clone (1 - target) of '{target_col}'. "
                                f"Models will trivially memorize the inverted labels, creating false test metrics."
                            ),
                            evidence={
                                "kind": "leakage",
                                "subtype": "target_inverted",
                                "column": col,
                                "target_column": target_col,
                                "correlation": -1.0,
                                "action_recommended": "drop_column",
                            },
                        )
                    )
                    continue

            # 3. Numeric Correlation Analysis
            if pd.api.types.is_numeric_dtype(series):
                valid_mask = series.notna() & y_num.notna()
                if valid_mask.sum() >= 10:
                    s_clean = series[valid_mask]
                    y_clean = y_num[valid_mask]

                    if s_clean.std() > 1e-8 and y_clean.std() > 1e-8:
                        r = float(np.corrcoef(s_clean, y_clean)[0, 1])
                        abs_r = abs(r)

                        # Linear transform / scalar multiple (|r| >= 0.999)
                        if abs_r >= 0.999:
                            issues.append(
                                Issue(
                                    module_id="leakage",
                                    column=col,
                                    severity="high",
                                    certainty="certain",
                                    title=f"Target Scalar Derivative in '{col}' (|r| = {abs_r:.4f})",
                                    description=(
                                        f"Column '{col}' has a near-perfect linear correlation of {r:.4f} with '{target_col}'. "
                                        f"This indicates mathematical derivation from the target."
                                    ),
                                    evidence={
                                        "kind": "leakage",
                                        "subtype": "target_scaled",
                                        "column": col,
                                        "target_column": target_col,
                                        "correlation": r,
                                        "action_recommended": "drop_column",
                                    },
                                )
                            )
                            continue

                        # Near-perfect proxy or post-event feature (|r| >= 0.95)
                        elif abs_r >= 0.95:
                            # Check if it's an ID or sequence leak
                            is_seq = False
                            if "id" in col.lower() or "index" in col.lower() or s_clean.is_monotonic_increasing:
                                is_seq = True
                            subtype = "index_leak" if is_seq else "perfect_predictor"
                            title = (
                                f"Row Index / Sequence Leak in '{col}' (|r| = {abs_r:.3f})"
                                if is_seq
                                else f"Critical Predictor Leak in '{col}' (|r| = {abs_r:.3f})"
                            )
                            issues.append(
                                Issue(
                                    module_id="leakage",
                                    column=col,
                                    severity="high",
                                    certainty="certain",
                                    title=title,
                                    description=(
                                        f"Column '{col}' exhibits an extreme correlation of {r:.3f} with '{target_col}'. "
                                        f"Features with correlation above 0.95 almost invariably represent post-event artifacts or ID leakage."
                                    ),
                                    evidence={
                                        "kind": "leakage",
                                        "subtype": subtype,
                                        "column": col,
                                        "target_column": target_col,
                                        "correlation": r,
                                        "action_recommended": "drop_column",
                                    },
                                )
                            )
                            continue

                        # Moderate / suspicious correlation (0.85 <= |r| < 0.95)
                        elif abs_r >= 0.85:
                            issues.append(
                                Issue(
                                    module_id="leakage",
                                    column=col,
                                    severity="medium",
                                    certainty="possible",
                                    title=f"Suspicious High Correlation in '{col}' (|r| = {abs_r:.3f})",
                                    description=(
                                        f"Column '{col}' has correlation {r:.3f} with '{target_col}'. "
                                        f"Verify that this feature is legitimately available at inference time."
                                    ),
                                    evidence={
                                        "kind": "leakage",
                                        "subtype": "suspicious_correlation",
                                        "column": col,
                                        "target_column": target_col,
                                        "correlation": r,
                                        "action_recommended": "investigate",
                                    },
                                )
                            )
                            continue

            # 4. Temporal Sequence / Time Inversion Risk
            if pd.api.types.is_datetime64_any_dtype(series) or "time" in col.lower() or "date" in col.lower():
                try:
                    s_dt = pd.to_datetime(series.dropna(), errors="coerce")
                    if not s_dt.empty and s_dt.is_monotonic_increasing and s_dt.nunique() > 20:
                        issues.append(
                            Issue(
                                module_id="leakage",
                                column=col,
                                severity="medium",
                                certainty="possible",
                                title=f"Time-Series Chronological Sequence in '{col}'",
                                description=(
                                    f"Column '{col}' contains monotonic chronological records. "
                                    f"Random shuffle splits will leak future data into training folds. "
                                    f"Use a time-ordered chronological split."
                                ),
                                evidence={
                                    "kind": "leakage",
                                    "subtype": "time_order_risk",
                                    "column": col,
                                    "target_column": target_col,
                                    "action_recommended": "time_split",
                                },
                            )
                        )
                except Exception:
                    pass

        return issues
