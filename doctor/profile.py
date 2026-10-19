"""
Dataset Profiler for Dataset Doctor.
Computes lightweight, comprehensive summary stats, memory footprint,
column data types, sample values, and target suggestions.
"""
from typing import Any, Dict, List
import numpy as np
import pandas as pd


def format_memory_size(bytes_val: int) -> str:
    """Formats bytes into a readable unit (KB, MB, GB)."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.1f} KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"


def infer_column_type(series: pd.Series) -> str:
    """Infers high-level semantic column type."""
    # Check if all values are empty or NaN
    non_null = series.dropna()
    if len(non_null) == 0:
        return "empty"

    # Numeric types
    if pd.api.types.is_numeric_dtype(series):
        # Check if boolean-like (0 and 1 only)
        unique_vals = set(non_null.unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            return "boolean"
        return "numeric"

    # Datetime types
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # String / object analysis
    sample = non_null.astype(str).str.strip()
    unique_vals_str = set(sample.unique())

    # Boolean strings
    if unique_vals_str.issubset({"0", "1", "true", "false", "yes", "no", "y", "n", "True", "False", "Yes", "No", "Y", "N"}):
        return "boolean"

    # Unique ratio
    n_unique = len(unique_vals_str)
    n_rows = len(series)

    # ID-like: high cardinality strings
    if n_rows > 30 and (n_unique / n_rows) > 0.90:
        return "id_like"

    if n_unique <= 25 or (n_rows > 0 and (n_unique / n_rows) < 0.2):
        return "categorical"

    return "text"


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Produces a summary profile of a pandas DataFrame.
    """
    n_rows = len(df)
    n_cols = len(df.columns)
    mem_usage = int(df.memory_usage(deep=True).sum())

    columns_info: List[Dict[str, Any]] = []
    columns_with_missing = 0

    for col in df.columns:
        series = df[col]
        # Null count considers NaN and empty strings
        is_null_mask = series.isna() | (series.astype(str).str.strip() == "")
        null_count = int(is_null_mask.sum())
        null_pct = round((null_count / n_rows * 100), 2) if n_rows > 0 else 0.0

        if null_count > 0:
            columns_with_missing += 1

        inferred = infer_column_type(series)
        unique_count = int(series.nunique(dropna=True))

        # Sample values: take up to 4 representative unique non-null values
        sample_vals = series.dropna().unique()[:4]
        sample_display = [str(x) for x in sample_vals]

        columns_info.append({
            "name": col,
            "dtype": str(series.dtype),
            "inferred_type": inferred,
            "unique_count": unique_count,
            "null_count": null_count,
            "null_percent": null_pct,
            "sample_values": sample_display,
        })

    # Preview rows (first 10 rows converted to strings for display)
    preview_df = df.head(10).copy()
    preview_rows = preview_df.fillna("").astype(str).to_dict(orient="records")

    # Target heuristic
    suggested_target = None
    target_candidates = ["target", "label", "class", "outcome", "severe_accident", "y", "churn", "survived"]
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in target_candidates:
        if cand in lower_cols:
            suggested_target = lower_cols[cand]
            break

    # If no obvious name match, check the last column
    if not suggested_target and len(df.columns) > 1:
        last_col = df.columns[-1]
        last_unique = df[last_col].nunique()
        if last_unique >= 2 and last_unique <= 50:
            suggested_target = last_col

    # Suggest task type based on suggested target
    suggested_task_type = "classification"
    if suggested_target:
        t_series = df[suggested_target]
        t_unique = t_series.nunique()
        if pd.api.types.is_numeric_dtype(t_series) and t_unique > 20:
            suggested_task_type = "regression"
        else:
            suggested_task_type = "classification"

    return {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "memory_bytes": mem_usage,
        "memory_display": format_memory_size(mem_usage),
        "columns_with_missing": columns_with_missing,
        "columns": columns_info,
        "preview_columns": list(df.columns),
        "preview_rows": preview_rows,
        "suggested_target": suggested_target,
        "suggested_task_type": suggested_task_type,
    }
