"""
Dataset Doctor: Data Types Fixer (Phase 6).
Applies data type remediation strategies:
- convert_to_numeric: strips currency symbols ($), commas, percentages (%), and units (kg), then casts to float64.
- parse_date_day_first: parses dates assuming Day-First (DD/MM/YYYY).
- parse_date_month_first: parses dates assuming Month-First (MM/DD/YYYY).
- parse_date: parses standard unambiguous datetime timestamps.
- encode_boolean: converts binary string flags (yes/no, true/false) into 0 and 1 indicators.
- drop_column: drops constant or identifier columns (applies to both train and test without altering row count).
- coerce_numeric: coerces mixed values to float, setting invalid strings to NaN.
- leave: retains column unchanged.

Safety rules:
- Test set rows are never dropped (is_test=True preserves exact row count).
- Learned conversion parameters and detected symbols are fitted on train only.
"""
from typing import Any, Dict, List, Optional
import re
import numpy as np
import pandas as pd

from doctor.base import Decision, Fixer
from doctor.issues.types.detect import TypeDetector


class TypeFixer(Fixer):
    """
    Applies deterministic and leak-free type transformations.
    """
    id = "types"
    rows_may_be_dropped_from = "train"

    # Standard boolean dictionary
    BOOL_MAP = {
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "true": 1,
        "false": 0,
        "t": 1,
        "f": 0,
        "1": 1,
        "0": 0,
    }

    def fit(self, train_df: pd.DataFrame, decision: Decision) -> Dict[str, Any]:
        """
        Learns transformation configuration and symbols from train_df.
        """
        col = decision.column
        opt = decision.option_id
        params = decision.params or {}

        fitted: Dict[str, Any] = {
            "option_id": opt,
            "column": col,
            "params": params,
        }

        if opt == "convert_to_numeric" and col and col in train_df.columns:
            # Check what symbols are present in train_df
            sample = train_df[col].dropna().astype(str).iloc[:100]
            detected_symbols = set()
            for val in sample:
                for sym in TypeDetector.CURRENCY_SYMBOLS:
                    if sym in val:
                        detected_symbols.add(sym)
                if "%" in val:
                    detected_symbols.add("%")
                if "," in val:
                    detected_symbols.add(",")
                v_lower = val.lower()
                for u in TypeDetector.UNIT_SUFFIXES:
                    if v_lower.endswith(u) or f" {u}" in v_lower:
                        detected_symbols.add(u)
            fitted["symbols"] = sorted(list(detected_symbols))

        elif opt in ("parse_date_day_first", "parse_date_month_first", "parse_date"):
            fitted["dayfirst"] = bool(params.get("dayfirst", opt == "parse_date_day_first"))

        elif opt == "encode_boolean":
            fitted["mapping"] = self.BOOL_MAP

        elif opt == "drop_column":
            fitted["column_to_drop"] = col

        return fitted

    def transform(self, df: pd.DataFrame, fitted: Dict[str, Any], is_test: bool = False) -> pd.DataFrame:
        """
        Applies fitted type transformations to a DataFrame.
        """
        col = fitted.get("column")
        opt = fitted.get("option_id", "leave")

        if opt == "leave" or not col:
            return df.copy()

        if col not in df.columns:
            return df.copy()

        data = df.copy()

        if opt == "convert_to_numeric":
            data[col] = self._apply_numeric_conversion(data[col])

        elif opt == "parse_date_day_first":
            data[col] = pd.to_datetime(data[col], dayfirst=True, errors="coerce")

        elif opt == "parse_date_month_first":
            data[col] = pd.to_datetime(data[col], dayfirst=False, errors="coerce")

        elif opt == "parse_date":
            dayfirst = fitted.get("dayfirst", False)
            data[col] = pd.to_datetime(data[col], dayfirst=dayfirst, errors="coerce")

        elif opt == "encode_boolean":
            mapping = fitted.get("mapping", self.BOOL_MAP)
            data[col] = (
                data[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(mapping)
                .fillna(0)
                .astype(int)
            )

        elif opt == "drop_column":
            data = data.drop(columns=[col])

        elif opt == "coerce_numeric":
            data[col] = pd.to_numeric(data[col], errors="coerce")

        return data

    @classmethod
    def _apply_numeric_conversion(cls, series: pd.Series) -> pd.Series:
        """
        Strips currencies, commas, percentages, unit suffixes, and converts to float.
        """
        def clean_val(v):
            if pd.isna(v):
                return np.nan
            v_str = str(v).strip()
            # Guard: If value is already purely numeric string or int/float
            try:
                return float(v_str)
            except ValueError:
                pass

            cleaned = TypeDetector._strip_symbols(v_str)
            try:
                return float(cleaned)
            except ValueError:
                return np.nan

        return series.apply(clean_val).astype(float)
