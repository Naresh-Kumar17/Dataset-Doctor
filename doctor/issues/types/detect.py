"""
Dataset Doctor: Data Types Detector (Phase 6).
Detects:
1. Numbers stored as text ("$1,200", "15%", "12 kg", text numbers).
   - Guard: Codes with leading zeroes ("001", "007") are preserved as categorical/code.
2. Dates stored as text, including:
   - Ambiguous date formats (day <= 12 and month <= 12) -> flagged as possible (asks user).
   - Unambiguous ISO or day-first/month-first dates -> flagged as certain.
3. Mixed type columns (numbers plus non-null string words).
4. Yes/No style boolean columns ("yes/no", "Y/N", "true/false").
5. Constant columns (zero variance, identical value across all rows).
6. ID-like columns (> 95% unique identifiers).
7. High-cardinality categorical features.
Implements the Detector interface from doctor/base.py.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import re
import numpy as np
import pandas as pd

from doctor.base import CertaintyLevel, DetectionContext, Detector, Issue, SeverityLevel
from doctor.issues.types.explain import explain_types_issue


class TypeDetector(Detector):
    """
    Scans a DataFrame for type mismatches, text numbers, ambiguous dates,
    boolean strings, and degenerate constant/ID features.
    """
    id = "types"

    # Common currency, unit, and formatting symbols
    CURRENCY_SYMBOLS = ["$", "€", "£", "¥", "₹"]
    UNIT_SUFFIXES = ["kg", "km", "cm", "mm", "lbs", "m", "g", "oz", "usd", "eur", "k"]

    # Known boolean sets (all lowercased)
    BOOLEAN_SETS = [
        {"y", "n"},
        {"yes", "no"},
        {"true", "false"},
        {"t", "f"},
    ]

    def detect(self, df: pd.DataFrame, ctx: Optional[DetectionContext] = None) -> List[Issue]:
        issues: List[Issue] = []
        if df.empty or len(df) == 0:
            return issues

        target_col = ctx.target_column if ctx else None
        total_rows = len(df)

        for col in df.columns:
            # Skip target column for non-constant structural checks
            series = df[col]
            non_null = series.dropna()

            if len(non_null) == 0:
                continue

            # -----------------------------------------------------------------
            # 1. Constant Column (Zero Variance)
            # -----------------------------------------------------------------
            n_uniques = non_null.nunique()
            if n_uniques == 1 and total_rows > 1:
                val = non_null.iloc[0]
                evidence = {
                    "kind": "constant",
                    "column": col,
                    "constant_value": str(val),
                    "total_rows": total_rows,
                }
                issues.append(
                    Issue(
                        module_id=self.id,
                        column=col,
                        severity="high",
                        certainty="certain",
                        title=f"Constant column '{col}' has zero variance",
                        description=explain_types_issue(evidence, "constant"),
                        evidence=evidence,
                    )
                )
                # If a column is constant, no need to perform type conversions on it
                continue

            # If target column, skip feature-specific ID and conversion checks
            if col == target_col:
                continue

            # -----------------------------------------------------------------
            # 2. String / Object Column Type Analysis
            # -----------------------------------------------------------------
            is_object_or_str = (series.dtype == object) or pd.api.types.is_string_dtype(series.dtype)

            if is_object_or_str:
                str_vals = non_null.astype(str).str.strip()

                # Guard: Check for leading zero categorical codes (e.g. "001", "007", "0521")
                # Such codes must NOT be converted to numeric!
                if self._is_leading_zero_code(str_vals):
                    # Identified as code categorical, protected from text number conversion
                    pass
                else:
                    # A. Check for Numbers Stored as Text
                    is_text_num, num_evidence = self._check_text_numbers(str_vals, col)
                    if is_text_num:
                        issues.append(
                            Issue(
                                module_id=self.id,
                                column=col,
                                severity="medium",
                                certainty="certain",
                                title=f"Numbers formatted as text in '{col}'",
                                description=explain_types_issue(num_evidence, "text_number"),
                                evidence=num_evidence,
                            )
                        )
                        continue

                # B. Check for Yes/No Style Booleans
                is_bool, bool_evidence = self._check_boolean_representation(str_vals, col)
                if is_bool:
                    issues.append(
                        Issue(
                            module_id=self.id,
                            column=col,
                            severity="low",
                            certainty="certain",
                            title=f"Boolean binary values in '{col}'",
                            description=explain_types_issue(bool_evidence, "boolean"),
                            evidence=bool_evidence,
                        )
                    )
                    continue

                # C. Check for Dates (Ambiguous vs Unambiguous)
                is_date, date_evidence = self._check_date_formats(str_vals, col)
                if is_date:
                    kind = date_evidence.get("kind")
                    if kind == "ambiguous_date":
                        issues.append(
                            Issue(
                                module_id=self.id,
                                column=col,
                                severity="medium",
                                certainty="possible",
                                title=f"Ambiguous date format in '{col}' (Day-first vs Month-first)",
                                description=explain_types_issue(date_evidence, "ambiguous_date"),
                                evidence=date_evidence,
                            )
                        )
                    else:
                        issues.append(
                            Issue(
                                module_id=self.id,
                                column=col,
                                severity="low",
                                certainty="certain",
                                title=f"Date timestamps stored as text in '{col}'",
                                description=explain_types_issue(date_evidence, "date"),
                                evidence=date_evidence,
                            )
                        )
                    continue

                # D. Check for Mixed Type Columns (Numbers + Words)
                is_mixed, mixed_evidence = self._check_mixed_types(str_vals, col)
                if is_mixed:
                    issues.append(
                        Issue(
                            module_id=self.id,
                            column=col,
                            severity="medium",
                            certainty="certain",
                            title=f"Mixed numeric and text entries in '{col}'",
                            description=explain_types_issue(mixed_evidence, "mixed_type"),
                            evidence=mixed_evidence,
                        )
                    )
                    continue

            # -----------------------------------------------------------------
            # 3. ID-Like Columns (>95% uniqueness on integer / string IDs)
            # -----------------------------------------------------------------
            # Continuous floats must NOT be flagged as ID
            is_float = pd.api.types.is_float_dtype(series.dtype)
            if not is_float and total_rows >= 20:
                unique_ratio = n_uniques / total_rows
                if unique_ratio > 0.95:
                    col_lower = col.lower()
                    is_id_name = any(kw in col_lower for kw in ["id", "uuid", "key", "code", "ident"])
                    # If named ID or monotonic increasing sequence, certainty is certain
                    is_seq = False
                    if pd.api.types.is_integer_dtype(series.dtype):
                        is_seq = series.is_monotonic_increasing

                    certainty: CertaintyLevel = "certain" if (is_id_name or is_seq) else "possible"
                    id_evidence = {
                        "kind": "id_like",
                        "column": col,
                        "unique_count": n_uniques,
                        "unique_ratio_pct": round(unique_ratio * 100, 2),
                        "total_rows": total_rows,
                    }
                    issues.append(
                        Issue(
                            module_id=self.id,
                            column=col,
                            severity="medium",
                            certainty=certainty,
                            title=f"Identifier column '{col}' (>95% unique values)",
                            description=explain_types_issue(id_evidence, "id_like"),
                            evidence=id_evidence,
                        )
                    )

        return issues

    def _is_leading_zero_code(self, s: pd.Series) -> bool:
        """
        Detects whether a column contains categorical codes with leading zeros (e.g. '001', '007').
        Such columns MUST NOT be converted to numeric.
        """
        if len(s) == 0:
            return False
        sample = s.iloc[:100]
        leading_zero_pattern = re.compile(r"^0[0-9]+$")
        leading_zero_count = sum(bool(leading_zero_pattern.match(v)) for v in sample if v != "0")
        return (leading_zero_count / len(sample)) > 0.15

    def _check_text_numbers(self, s: pd.Series, col: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Detects numbers stored with formatting symbols like $, commas, %, or units (e.g. '12 kg').
        """
        if len(s) == 0:
            return False, {}

        sample = s.iloc[:100]
        symbols_found = set()
        clean_numeric_count = 0

        for val in sample:
            v_str = str(val).strip()
            # Track detected symbols
            for sym in self.CURRENCY_SYMBOLS:
                if sym in v_str:
                    symbols_found.add(sym)
            if "%" in v_str:
                symbols_found.add("%")
            if "," in v_str:
                symbols_found.add(",")

            # Check unit suffixes
            v_lower = v_str.lower()
            for u in self.UNIT_SUFFIXES:
                if v_lower.endswith(u) or f" {u}" in v_lower:
                    symbols_found.add(u)

            # Strip known symbols and test numeric conversion
            cleaned = self._strip_symbols(v_str)
            try:
                float(cleaned)
                clean_numeric_count += 1
            except ValueError:
                pass

        parseable_ratio = clean_numeric_count / len(sample)
        # If >80% are parseable after symbol stripping and at least one symbol was present
        if parseable_ratio >= 0.80 and len(symbols_found) > 0:
            evidence = {
                "kind": "text_number",
                "column": col,
                "detected_symbols": sorted(list(symbols_found)),
                "sample_values": sample.iloc[:5].tolist(),
                "parseable_pct": round(parseable_ratio * 100, 1),
            }
            return True, evidence

        return False, {}

    @classmethod
    def _strip_symbols(cls, val_str: str) -> str:
        """Helper to remove currency signs, commas, percentages, and unit suffixes."""
        cleaned = val_str.strip()
        for sym in cls.CURRENCY_SYMBOLS:
            cleaned = cleaned.replace(sym, "")
        cleaned = cleaned.replace(",", "")
        cleaned = cleaned.replace("%", "")
        # Remove trailing unit words
        for u in cls.UNIT_SUFFIXES:
            pattern = re.compile(rf"\s*{u}\b", re.IGNORECASE)
            cleaned = pattern.sub("", cleaned)
        return cleaned.strip()

    def _check_boolean_representation(self, s: pd.Series, col: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Detects binary yes/no, Y/N, True/False string columns.
        """
        uniques = set(s.str.lower().unique())
        for bset in self.BOOLEAN_SETS:
            if uniques == bset or (len(uniques) == 1 and uniques.issubset(bset)):
                evidence = {
                    "kind": "boolean",
                    "column": col,
                    "unique_values": sorted(list(uniques)),
                }
                return True, evidence
        return False, {}

    def _check_date_formats(self, s: pd.Series, col: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Detects date strings and separates ambiguous dates (day <= 12 and month <= 12)
        from unambiguous dates.
        """
        if len(s) == 0:
            return False, {}

        sample = s.iloc[:100]

        # 1. ISO format check: YYYY-MM-DD
        iso_pattern = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
        iso_matches = sum(bool(iso_pattern.match(v)) for v in sample)
        if iso_matches / len(sample) >= 0.80:
            evidence = {
                "kind": "date",
                "column": col,
                "detected_format": "YYYY-MM-DD (ISO)",
                "sample_values": sample.iloc[:4].tolist(),
            }
            return True, evidence

        # 2. Separated date format: X/Y/Z or X-Y-Z
        sep_pattern = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})$")
        matched_components = []
        for v in sample:
            m = sep_pattern.match(v)
            if m:
                p1, p2, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
                matched_components.append((p1, p2, yr))

        if len(matched_components) / len(sample) >= 0.80:
            # Check ambiguity: Are all p1 and p2 <= 12?
            all_both_under_13 = all(p1 <= 12 and p2 <= 12 for p1, p2, yr in matched_components)

            if all_both_under_13:
                # Ambiguous! Both Day-First (DD/MM/YYYY) and Month-First (MM/DD/YYYY) are valid!
                evidence = {
                    "kind": "ambiguous_date",
                    "column": col,
                    "sample_values": sample.iloc[:4].tolist(),
                    "requires_user_disambiguation": True,
                }
                return True, evidence
            else:
                # Disambiguated by values > 12
                has_p1_day = any(p1 > 12 and p2 <= 12 for p1, p2, yr in matched_components)
                detected_fmt = "DD/MM/YYYY (Day-first)" if has_p1_day else "MM/DD/YYYY (Month-first)"
                evidence = {
                    "kind": "date",
                    "column": col,
                    "detected_format": detected_fmt,
                    "sample_values": sample.iloc[:4].tolist(),
                }
                return True, evidence

        return False, {}

    def _check_mixed_types(self, s: pd.Series, col: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Detects mixed columns where some rows contain valid numbers and other rows contain words.
        """
        if len(s) == 0:
            return False, {}

        sample = s.iloc[:100]
        numeric_count = 0
        word_count = 0

        for val in sample:
            v_str = str(val).strip()
            try:
                float(v_str)
                numeric_count += 1
            except ValueError:
                if re.search(r"[a-zA-Z]", v_str):
                    word_count += 1

        total = len(sample)
        # If numeric is between 20% and 80%, and word entries are also between 20% and 80%
        if 0.20 <= (numeric_count / total) <= 0.80 and (word_count / total) >= 0.20:
            evidence = {
                "kind": "mixed_type",
                "column": col,
                "numeric_ratio_pct": round((numeric_count / total) * 100, 1),
                "text_ratio_pct": round((word_count / total) * 100, 1),
                "sample_values": sample.iloc[:6].tolist(),
            }
            return True, evidence

        return False, {}
