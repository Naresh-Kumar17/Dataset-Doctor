"""
Dataset Doctor: Duplicates Detector (Phase 5).
Detects:
1. Exact duplicate rows (count, percent, affected rows).
2. Near-duplicate rows (differing only by casing, whitespace, or tiny numeric differences).
3. Conflicting duplicate rows (identical features, different target labels).
4. Duplicate identifiers in ID-like columns.
5. Pre-balanced dataset warning (50/50 class ratio with repeated minority feature vectors).
Implements the Detector interface from doctor/base.py.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from doctor.base import CertaintyLevel, DetectionContext, Detector, Issue, SeverityLevel
from doctor.issues.duplicates.explain import explain_duplicates_issue


class DuplicateDetector(Detector):
    """
    Scans a DataFrame for exact duplicates, near duplicates, conflicting labels,
    and synthetic pre-balancing.
    """
    id = "duplicates"

    def detect(self, df: pd.DataFrame, ctx: Optional[DetectionContext] = None) -> List[Issue]:
        issues: List[Issue] = []
        if df.empty or len(df) == 0:
            return issues

        target_col = ctx.target_column if ctx else None
        total_rows = len(df)

        # ---------------------------------------------------------------------
        # 1. Exact Duplicate Rows
        # ---------------------------------------------------------------------
        exact_surplus_mask = df.duplicated(keep="first")
        exact_all_mask = df.duplicated(keep=False)
        surplus_count = int(exact_surplus_mask.sum())
        total_involved = int(exact_all_mask.sum())

        if surplus_count > 0:
            pct = round((surplus_count / total_rows) * 100, 2)
            severity: SeverityLevel = "high" if (pct > 10.0 or surplus_count >= 50) else "medium"
            surplus_indices = df.index[exact_surplus_mask].tolist()

            evidence = {
                "kind": "exact",
                "total_rows": total_rows,
                "surplus_copies": surplus_count,
                "total_involved_rows": total_involved,
                "duplicate_pct": pct,
                "affected_row_indices": surplus_indices[:100],  # store first 100 for display
            }

            desc = explain_duplicates_issue(evidence, kind="exact")
            issues.append(
                Issue(
                    module_id=self.id,
                    column=None,
                    severity=severity,
                    certainty="certain",
                    title="Exact duplicate rows detected",
                    description=desc,
                    evidence=evidence,
                )
            )

        # ---------------------------------------------------------------------
        # 2. Conflicting Duplicates (identical features, differing target)
        # ---------------------------------------------------------------------
        if target_col and target_col in df.columns and len(df.columns) > 1:
            feature_cols = [c for c in df.columns if c != target_col]
            # Identify rows with identical feature values
            feat_dup_mask = df.duplicated(subset=feature_cols, keep=False)
            if feat_dup_mask.any():
                feat_dups = df[feat_dup_mask]
                # Group by features to check if target has multiple distinct values
                conflicting_indices: List[int] = []
                conflicting_groups = 0

                # Use transform to check nunique per group efficiently
                target_nunique_per_group = feat_dups.groupby(feature_cols)[target_col].transform("nunique")
                is_conflicting_row = target_nunique_per_group > 1
                if is_conflicting_row.any():
                    conflicting_df = feat_dups[is_conflicting_row]
                    conflicting_indices = conflicting_df.index.tolist()
                    conflicting_groups = int(conflicting_df.groupby(feature_cols).ngroups)

                if conflicting_indices:
                    evidence_conf = {
                        "kind": "conflicting",
                        "total_rows": total_rows,
                        "conflicting_rows_count": len(conflicting_indices),
                        "conflicting_pairs_count": conflicting_groups,
                        "target_column": target_col,
                        "affected_row_indices": conflicting_indices[:100],
                    }
                    desc_conf = explain_duplicates_issue(evidence_conf, kind="conflicting")
                    issues.append(
                        Issue(
                            module_id=self.id,
                            column=target_col,
                            severity="high",  # Conflicting targets are always high severity
                            certainty="certain",
                            title=f"Conflicting duplicate rows (identical features, different '{target_col}')",
                            description=desc_conf,
                            evidence=evidence_conf,
                        )
                    )

        # ---------------------------------------------------------------------
        # 3. Near-Duplicate Rows (casing, whitespace, tiny float differences)
        # ---------------------------------------------------------------------
        df_norm = self._normalize_dataframe_for_near_check(df)
        norm_surplus_mask = df_norm.duplicated(keep="first")
        # Near duplicates are surplus in normalized data but NOT exact duplicates in original data
        near_surplus_mask = norm_surplus_mask & (~exact_surplus_mask)
        near_count = int(near_surplus_mask.sum())

        if near_count > 0:
            evidence_near = {
                "kind": "near",
                "total_rows": total_rows,
                "near_duplicate_count": near_count,
                "variation_types": ["case / whitespace / numeric tolerance"],
                "affected_row_indices": df.index[near_surplus_mask].tolist()[:100],
            }
            desc_near = explain_duplicates_issue(evidence_near, kind="near")
            issues.append(
                Issue(
                    module_id=self.id,
                    column=None,
                    severity="medium",
                    certainty="possible",  # Near duplicates are strictly classified as possible
                    title="Near-duplicate rows detected",
                    description=desc_near,
                    evidence=evidence_near,
                )
            )

        # ---------------------------------------------------------------------
        # 4. Duplicate IDs in ID-like Columns
        # ---------------------------------------------------------------------
        for col in df.columns:
            col_lower = str(col).lower()
            is_id_named = col_lower in ("id", "uuid", "guid") or col_lower.endswith(("_id", "-id", "id"))
            if is_id_named:
                id_dup_mask = df[col].duplicated(keep="first")
                id_dup_cnt = int(id_dup_mask.sum())
                # Only flag if there are duplicates and it behaves as an identifier (> 50% unique values)
                if id_dup_cnt > 0 and (df[col].nunique() / total_rows) > 0.5:
                    evidence_id = {
                        "kind": "id",
                        "column": col,
                        "duplicate_id_count": id_dup_cnt,
                        "unique_count": int(df[col].nunique()),
                        "total_rows": total_rows,
                        "affected_row_indices": df.index[id_dup_mask].tolist()[:50],
                    }
                    desc_id = explain_duplicates_issue(evidence_id, kind="id")
                    issues.append(
                        Issue(
                            module_id=self.id,
                            column=col,
                            severity="high",
                            certainty="certain",
                            title=f"Duplicate unique identifiers in '{col}'",
                            description=desc_id,
                            evidence=evidence_id,
                        )
                    )

        # ---------------------------------------------------------------------
        # 5. Pre-Balanced Dataset Warning
        # ---------------------------------------------------------------------
        if target_col and target_col in df.columns:
            target_series = df[target_col].dropna()
            # Check if binary or low-cardinality discrete target
            val_counts = target_series.value_counts(normalize=True)
            if len(val_counts) == 2:
                share0 = float(val_counts.iloc[0])
                share1 = float(val_counts.iloc[1])
                is_balanced_50_50 = abs(share0 - 0.50) <= 0.03  # 47% to 53%
                if is_balanced_50_50:
                    feature_cols = [c for c in df.columns if c != target_col]
                    class0_val = val_counts.index[0]
                    class1_val = val_counts.index[1]

                    df_c0 = df[df[target_col] == class0_val]
                    df_c1 = df[df[target_col] == class1_val]

                    dup_c0 = int(df_c0.duplicated(subset=feature_cols, keep="first").sum())
                    dup_c1 = int(df_c1.duplicated(subset=feature_cols, keep="first").sum())

                    # If one class has multiple repeated rows (> 10 or > 10% of class size)
                    # while being 50/50 balanced, this is synthetic oversampling leakage
                    max_dup = max(dup_c0, dup_c1)
                    if max_dup >= 10:
                        repeated_class = class0_val if dup_c0 > dup_c1 else class1_val
                        evidence_pb = {
                            "kind": "pre_balanced",
                            "target_column": target_col,
                            "target_ratio_str": f"{share0*100:.1f}% / {share1*100:.1f}%",
                            "repeated_minority_rows": max_dup,
                            "oversampled_class": str(repeated_class),
                            "class0_duplicates": dup_c0,
                            "class1_duplicates": dup_c1,
                            "total_rows": total_rows,
                        }
                        desc_pb = explain_duplicates_issue(evidence_pb, kind="pre_balanced")
                        issues.append(
                            Issue(
                                module_id=self.id,
                                column=target_col,
                                severity="high",
                                certainty="possible",
                                title="Suspicious pre-balancing detected (possible test leakage)",
                                description=desc_pb,
                                evidence=evidence_pb,
                            )
                        )

        return issues

    def _normalize_dataframe_for_near_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes string columns (lowercase, strip whitespace) and float columns
        (rounded to 4 decimals) to uncover near-duplicate records.
        """
        df_norm = pd.DataFrame(index=df.index)
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
                # String normalization: strip whitespace and convert to lowercase
                df_norm[col] = s.astype(str).str.strip().str.lower()
            elif pd.api.types.is_float_dtype(s.dtype):
                # Float normalization: round to 4 decimal places to catch precision fuzz
                df_norm[col] = s.round(4)
            else:
                df_norm[col] = s
        return df_norm
