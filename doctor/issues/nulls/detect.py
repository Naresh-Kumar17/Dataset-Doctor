"""
Dataset Doctor: Null / Missing Values Detector (Phase 2).
Finds every missing value, including hidden tokens and sentinels, without altering data.
Implements the Detector interface from doctor/base.py.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from doctor.base import CertaintyLevel, DetectionContext, Detector, Issue, SeverityLevel
from doctor.issues.nulls.explain import explain_null_issue


# Clear missing tokens: certainty is 'certain'
CLEAR_STRING_TOKENS: Set[str] = {
    "",
    "?",
    "-",
    "null",
    "none",
    "nan",
    "n/a",
    "n.a.",
}

# Ambiguous tokens: certainty is 'possible' (e.g. NA could be Namibia, -999 could be sentinel)
AMBIGUOUS_STRING_TOKENS: Set[str] = {
    "na",
}

NUMERIC_SENTINELS: Set[float] = {
    -999.0,
    -9999.0,
    9999.0,
    99999.0,
}


def _check_string_tokens(series: pd.Series) -> Tuple[pd.Series, pd.Series, Dict[str, int], Dict[str, int]]:
    """
    Scans a series for hidden string tokens and sentinels.
    Returns:
        (certain_mask, possible_mask, hidden_token_counts, sentinel_counts)
    """
    n = len(series)
    certain_mask = pd.Series(False, index=series.index)
    possible_mask = pd.Series(False, index=series.index)
    hidden_counts: Dict[str, int] = {}
    sentinel_counts: Dict[str, int] = {}

    # Fast check: skip if entirely numeric non-string without sentinel-like min/max
    is_object_or_string = series.dtype == object or isinstance(series.dtype, pd.StringDtype)

    if is_object_or_string:
        # String vectorized processing
        s_clean = series.astype(str).str.strip()
        s_lower = s_clean.str.lower()

        # 1. Clear empty or whitespace
        empty_mask = s_clean == ""
        if empty_mask.any():
            cnt = int(empty_mask.sum())
            certain_mask |= empty_mask
            hidden_counts["<empty>"] = cnt

        # 2. Clear tokens
        for token in ["?", "-", "n/a", "n.a.", "null", "none", "nan"]:
            token_mask = s_lower == token
            if token_mask.any():
                cnt = int(token_mask.sum())
                certain_mask |= token_mask
                # Keep original case display if possible
                hidden_counts[token.upper() if token in ["n/a", "n.a."] else token] = cnt

        # 3. Ambiguous 'NA' / 'na'
        na_mask = (s_clean == "NA") | (s_clean == "na")
        if na_mask.any():
            cnt = int(na_mask.sum())
            possible_mask |= na_mask
            hidden_counts["NA"] = cnt

        # 4. Text-based sentinels (e.g. "-999", "9999")
        for num_sent in ["-999", "-9999", "9999", "99999", "-1"]:
            sent_mask = s_clean == num_sent
            if sent_mask.any():
                cnt = int(sent_mask.sum())
                possible_mask |= sent_mask
                sentinel_counts[num_sent] = cnt

    return certain_mask, possible_mask, hidden_counts, sentinel_counts


def _check_numeric_sentinels(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Checks numeric series for sentinel values (-999, 9999, -1, 0).
    Sentinels are strictly flagged as 'possible'.
    """
    sentinel_mask = pd.Series(False, index=series.index)
    sentinel_counts: Dict[str, int] = {}

    if not pd.api.types.is_numeric_dtype(series.dtype):
        return sentinel_mask, sentinel_counts

    # High-confidence sentinels: -999, 9999, -9999
    for val in [-999, 9999, -9999, 99999]:
        matches = series == val
        if matches.any():
            cnt = int(matches.sum())
            sentinel_mask |= matches
            sentinel_counts[str(val)] = cnt

    # -1 sentinel: only if all valid values are otherwise >= 0
    neg_one = series == -1
    if neg_one.any():
        non_sentinel = series[~neg_one].dropna()
        if len(non_sentinel) > 0 and (non_sentinel >= 0).all():
            cnt = int(neg_one.sum())
            sentinel_mask |= neg_one
            sentinel_counts["-1"] = cnt

    return sentinel_mask, sentinel_counts


def _test_target_relationship(
    missing_indicator: pd.Series,
    target_series: pd.Series,
    task_type: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Statistical test checking if missingness is correlated with the target variable.
    Classification: Chi-square test of independence.
    Regression: Two-sample t-test between missing and non-missing subsets.
    """
    # Requires variation in missingness and valid target
    if missing_indicator.nunique() < 2 or len(target_series) == 0:
        return None

    valid_mask = target_series.notna()
    if valid_mask.sum() < 10:
        return None

    sub_miss = missing_indicator[valid_mask]
    sub_target = target_series[valid_mask]

    is_classification = (
        task_type == "classification"
        or sub_target.nunique() <= 10
        or not pd.api.types.is_numeric_dtype(sub_target.dtype)
    )

    try:
        if is_classification:
            # Chi-square test
            contingency = pd.crosstab(sub_miss, sub_target)
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                return None
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
            is_related = bool(p_val < 0.05)
            return {
                "related": is_related,
                "test": "chi2",
                "p_value": float(p_val),
                "statistic": float(chi2),
                "significant": is_related,
            }
        else:
            # Regression t-test
            vals_missing = sub_target[sub_miss == True]
            vals_present = sub_target[sub_miss == False]
            if len(vals_missing) < 3 or len(vals_present) < 3:
                return None
            t_stat, p_val = stats.ttest_ind(vals_missing, vals_present, equal_var=False, nan_policy="omit")
            is_related = bool(p_val < 0.05)
            return {
                "related": is_related,
                "test": "t_test",
                "p_value": float(p_val),
                "statistic": float(t_stat),
                "significant": is_related,
            }
    except Exception:
        return None


class NullDetector(Detector):
    """
    Phase 2 Null & Missing Value Detector.
    Scans for true NaNs, hidden strings, and sentinel numbers across all columns.
    Change nothing in the DataFrame; strictly read-only diagnostics.
    """
    id: str = "nulls"

    def detect(self, df: pd.DataFrame, ctx: Optional[DetectionContext] = None) -> List[Issue]:
        issues: List[Issue] = []
        n_rows, n_cols = df.shape
        if n_rows == 0:
            return issues

        target_col = ctx.target_column if ctx else None
        task_type = ctx.task_type if ctx else None
        target_series = df[target_col] if (target_col and target_col in df.columns) else None

        # Track full dataset missing masks for exact row loss & pattern checks
        col_certain_masks: Dict[str, pd.Series] = {}
        col_possible_masks: Dict[str, pd.Series] = {}
        col_all_missing_masks: Dict[str, pd.Series] = {}
        col_sentinel_masks: Dict[str, pd.Series] = {}

        # 1. First pass: detect per-column missingness
        column_summaries = []

        for col in df.columns:
            series = df[col]

            # A. True NaNs & empty string cells (empty CSV cells with keep_default_na=False)
            nan_mask = series.isna()
            if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
                s_trimmed = series.astype(str).str.strip()
                empty_cell_mask = s_trimmed == ""
            else:
                empty_cell_mask = pd.Series(False, index=series.index)

            true_missing_mask = nan_mask | empty_cell_mask
            nan_count = int(true_missing_mask.sum())

            # B. Hidden string tokens
            certain_str_mask, possible_str_mask, hidden_counts, text_sentinels = _check_string_tokens(series)

            # C. Numeric sentinels
            num_sentinel_mask, num_sentinel_counts = _check_numeric_sentinels(series)

            # Combine sentinels
            all_sentinel_counts = {**text_sentinels, **num_sentinel_counts}
            combined_sentinel_mask = pd.Series(False, index=series.index)
            if text_sentinels:
                # Include string sentinel masks
                for num_sent in text_sentinels:
                    combined_sentinel_mask |= (series.astype(str).str.strip() == num_sent)
            combined_sentinel_mask |= num_sentinel_mask

            # Combined certainty masks
            certain_mask = true_missing_mask | certain_str_mask
            possible_mask = possible_str_mask | num_sentinel_mask

            # If a cell is already certain, remove from possible
            possible_mask = possible_mask & (~certain_mask)

            all_missing = certain_mask | possible_mask

            col_certain_masks[col] = certain_mask
            col_possible_masks[col] = possible_mask
            col_all_missing_masks[col] = all_missing
            col_sentinel_masks[col] = combined_sentinel_mask

            total_missing = int(all_missing.sum())
            if total_missing > 0:
                column_summaries.append({
                    "column": col,
                    "nan_count": nan_count,
                    "certain_mask": certain_mask,
                    "possible_mask": possible_mask,
                    "all_missing": all_missing,
                    "sentinel_mask": combined_sentinel_mask,
                    "hidden_counts": hidden_counts,
                    "sentinel_counts": all_sentinel_counts,
                    "total_missing": total_missing,
                })

        # 2. Row-level loss calculation across whole dataset
        # Genuine row loss dropped by cleaning pipeline (certain missing values + identified sentinels)
        # Ambiguous tokens like 'NA' (Namibia country code) are preserved unless explicitly chosen
        if col_certain_masks:
            row_drop_mask = pd.DataFrame({
                c: (col_certain_masks[c] | col_sentinel_masks[c])
                for c in df.columns
            }).any(axis=1)
            rows_lost = int(row_drop_mask.sum())
            rows_lost_pct = round((rows_lost / n_rows) * 100, 2)

            missing_counts_per_row = pd.DataFrame({
                c: (col_certain_masks[c] | col_sentinel_masks[c])
                for c in df.columns
            }).sum(axis=1)
            row_counts_by_missing = {
                "1": int((missing_counts_per_row == 1).sum()),
                "2": int((missing_counts_per_row == 2).sum()),
                "3+": int((missing_counts_per_row >= 3).sum()),
            }
        else:
            rows_lost = 0
            rows_lost_pct = 0.0
            row_counts_by_missing = {"1": 0, "2": 0, "3+": 0}

        row_loss_impact = {
            "rows_lost": rows_lost,
            "rows_lost_pct": rows_lost_pct,
            "row_counts_by_missing": row_counts_by_missing,
            "rows_kept_if_dropped": n_rows - rows_lost,
        }

        # 3. Build Issue objects per affected column
        for cs in column_summaries:
            col = cs["column"]
            total_missing = cs["total_missing"]
            missing_pct = round((total_missing / n_rows) * 100, 2)
            is_target = bool(target_col and col == target_col)

            # Severity rules:
            # - high: target column missing, OR > 60% missing
            # - medium: 5% - 60% missing
            # - low: < 5% missing
            if is_target or missing_pct > 60.0:
                severity: SeverityLevel = "high"
            elif missing_pct >= 5.0:
                severity: SeverityLevel = "medium"
            else:
                severity: SeverityLevel = "low"

            # Certainty rules:
            # - certain: if true NaN or clear tokens found
            # - possible: if ONLY ambiguous tokens found
            has_certain = cs["certain_mask"].any()
            certainty: CertaintyLevel = "certain" if has_certain else "possible"

            # Missingness pattern test against target (Evidence only)
            target_pattern = None
            if target_series is not None and not is_target:
                target_pattern = _test_target_relationship(
                    cs["all_missing"],
                    target_series,
                    task_type=task_type
                )

            # Cross-column missingness correlation
            correlated_cols = []
            for other_col, other_mask in col_all_missing_masks.items():
                if other_col != col and other_mask.sum() > 0:
                    try:
                        overlap = (cs["all_missing"] & other_mask).sum()
                        if overlap > 0:
                            # Simple jaccard overlap
                            union = (cs["all_missing"] | other_mask).sum()
                            if (overlap / union) > 0.5:
                                correlated_cols.append(other_col)
                    except Exception:
                        pass

            evidence = {
                "total_rows": n_rows,
                "column": col,
                "nan_count": cs["nan_count"],
                "nan_pct": round((cs["nan_count"] / n_rows) * 100, 2),
                "hidden_token_counts": cs["hidden_counts"],
                "sentinel_counts": cs["sentinel_counts"],
                "total_missing_count": total_missing,
                "total_missing_pct": missing_pct,
                "is_target": is_target,
                "target_pattern": target_pattern,
                "correlated_missing_cols": correlated_cols,
                "row_loss_impact": row_loss_impact,
            }

            title = f"Null values in '{col}' ({missing_pct:.1f}% missing)"
            description = explain_null_issue(evidence, column=col)

            issues.append(
                Issue(
                    module_id="nulls",
                    column=col,
                    severity=severity,
                    certainty=certainty,
                    title=title,
                    description=description,
                    evidence=evidence,
                )
            )

        return issues
