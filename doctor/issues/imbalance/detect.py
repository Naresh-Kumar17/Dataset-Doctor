"""
Dataset Doctor: Class Imbalance Detection Engine (Phase 9).
Detects:
- Class distribution and class shares.
- Minority class proportion and imbalance ratio.
- Flags severity=high when minority class < 10% (ratio >= 9.0).
- Flags severity=medium when minority class < 20% (ratio >= 4.0).
- Stays quiet on balanced targets (minority >= 20%).
- Surfaces user confirmation prompt: "Is X% the true real-world rate?"
"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from doctor.base import CertaintyLevel, DetectionContext, Detector, Issue, SeverityLevel


class ImbalanceDetector(Detector):
    id: str = "imbalance"

    def detect(self, df: pd.DataFrame, ctx: Optional[DetectionContext] = None) -> List[Issue]:
        issues: List[Issue] = []
        if df.empty:
            return issues

        target_col = ctx.target_column if ctx else None
        
        # If target column is specified, analyze it
        candidates = []
        if target_col and target_col in df.columns:
            candidates.append(target_col)
        else:
            # Check columns named 'target' or low-cardinality discrete columns (<= 10 unique)
            for c in df.columns:
                if "target" in c.lower() or df[c].nunique() in [2, 3, 4, 5]:
                    candidates.append(c)

        for col in candidates:
            series = df[col].dropna()
            if series.empty or series.nunique() < 2:
                continue

            # Skip high-cardinality continuous numeric columns unless named 'target'
            if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10 and "target" not in col.lower():
                continue

            value_counts = series.value_counts(ascending=True)
            total = len(series)
            n_classes = len(value_counts)

            minority_class = value_counts.index[0]
            minority_count = int(value_counts.iloc[0])
            minority_share = float(minority_count / total)

            majority_class = value_counts.index[-1]
            majority_count = int(value_counts.iloc[-1])
            imbalance_ratio = float(majority_count / max(minority_count, 1))

            class_shares = {str(k): float(v / total) for k, v in value_counts.items()}
            class_counts = {str(k): int(v) for k, v in value_counts.items()}

            # Determine severity
            if minority_share < 0.10:
                severity: SeverityLevel = "high"
                title = f"Severe Class Imbalance in '{col}' ({minority_share:.1%} minority)"
            elif minority_share < 0.20:
                severity: SeverityLevel = "medium"
                title = f"Moderate Class Imbalance in '{col}' ({minority_share:.1%} minority)"
            else:
                # Balanced (>= 20% minority share) -> Stay quiet
                continue

            desc = (
                f"Column '{col}' has {n_classes} classes with an imbalance ratio of {imbalance_ratio:.1f}:1. "
                f"Minority class '{minority_class}' represents only {minority_count} of {total} rows ({minority_share:.1%}). "
                f"Models evaluated by accuracy alone will appear falsely accurate while completely failing on the minority class."
            )

            evidence: Dict[str, Any] = {
                "column": col,
                "n_classes": n_classes,
                "is_multiclass": n_classes > 2,
                "minority_class": str(minority_class),
                "minority_count": minority_count,
                "minority_share": minority_share,
                "majority_class": str(majority_class),
                "majority_count": majority_count,
                "imbalance_ratio": imbalance_ratio,
                "class_counts": class_counts,
                "class_shares": class_shares,
                "prompt_real_world": f"Is {minority_share:.1%} the true rate in the real world?",
            }

            issues.append(
                Issue(
                    module_id="imbalance",
                    column=col,
                    severity=severity,
                    certainty="certain",
                    title=title,
                    description=desc,
                    evidence=evidence,
                )
            )

        return issues
