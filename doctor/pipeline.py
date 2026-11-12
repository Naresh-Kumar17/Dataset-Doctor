"""
Dataset Doctor: Remediation Pipeline & Decision Log.
Rules:
- Maintains a sequential audit trail of decisions made by user or recommended automation.
- Stores fitted transformers (learned strictly from train) for reproducibility.
- Exports a standalone, dependency-free reproduce.py script that replays the exact
  data cleaning sequence with learned parameters.
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import re
from doctor.base import Decision
from doctor.issues.nulls.fix import NullFixer
from doctor.issues.duplicates.fix import DuplicateFixer
from doctor.issues.types.fix import TypeFixer
from doctor.issues.outliers.fix import OutlierFixer
from doctor.issues.imbalance.fix import ImbalanceFixer


@dataclass
class PipelineStep:
    """A fitted transformation step in the remediation pipeline."""
    step_id: str
    issue_id: str
    column: Optional[str]
    option_id: str
    decided_by: str
    fitted_params: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class RemediationPipeline:
    """
    Manages the sequential execution, decision logging, and export of dataset remediations.
    """
    def __init__(self):
        self.decisions: List[Decision] = []
        self.steps: List[PipelineStep] = []
        self._null_fixer = NullFixer()
        self._duplicate_fixer = DuplicateFixer()
        self._type_fixer = TypeFixer()
        self._outlier_fixer = OutlierFixer()
        self._imbalance_fixer = ImbalanceFixer()

    def record_decision(self, decision: Decision, train_df: Optional[pd.DataFrame] = None) -> PipelineStep:
        """
        Records a decision and fits its transformer on train_df if provided.
        """
        self.decisions.append(decision)
        fitted = {}
        if train_df is not None:
            if decision.issue_id.startswith("null") or decision.issue_id == "null_fixer":
                fitted = self._null_fixer.fit(train_df, decision)
            elif decision.issue_id == "duplicates" or decision.issue_id.startswith("dup"):
                fitted = self._duplicate_fixer.fit(train_df, decision)
            elif decision.issue_id == "types" or decision.issue_id.startswith("type"):
                fitted = self._type_fixer.fit(train_df, decision)
            elif decision.issue_id == "outliers" or decision.issue_id.startswith("outlier"):
                fitted = self._outlier_fixer.fit(train_df, decision)
            elif decision.issue_id == "imbalance" or decision.issue_id.startswith("imbalance"):
                fitted = self._imbalance_fixer.fit(train_df, decision)

        step = PipelineStep(
            step_id=f"step_{len(self.steps) + 1}",
            issue_id=decision.issue_id,
            column=decision.column,
            option_id=decision.option_id,
            decided_by=decision.decided_by,
            fitted_params=fitted,
            timestamp=decision.timestamp,
        )
        self.steps.append(step)
        return step

    def transform(self, df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
        """
        Applies all fitted pipeline steps in sequential order.
        """
        df_curr = df.copy()
        for step in self.steps:
            if step.fitted_params is not None:
                if step.issue_id.startswith("null"):
                    df_curr = self._null_fixer.transform(df_curr, step.fitted_params, is_test=is_test)
                elif step.issue_id == "duplicates" or step.issue_id.startswith("dup"):
                    df_curr = self._duplicate_fixer.transform(df_curr, step.fitted_params, is_test=is_test)
                elif step.issue_id == "types" or step.issue_id.startswith("type"):
                    df_curr = self._type_fixer.transform(df_curr, step.fitted_params, is_test=is_test)
                elif step.issue_id == "outliers" or step.issue_id.startswith("outlier"):
                    # Create temporary Decision object for fixer
                    temp_dec = Decision(
                        issue_id=step.issue_id,
                        column=step.column,
                        option_id=step.option_id,
                        params=step.fitted_params,
                        decided_by=step.decided_by,
                    )
                    df_curr = self._outlier_fixer.transform(df_curr, temp_dec, step.fitted_params, is_test=is_test)
                elif step.issue_id == "imbalance" or step.issue_id.startswith("imbalance"):
                    temp_dec = Decision(
                        issue_id=step.issue_id,
                        column=step.column,
                        option_id=step.option_id,
                        params=step.fitted_params,
                        decided_by=step.decided_by,
                    )
                    df_curr = self._imbalance_fixer.transform(df_curr, temp_dec, step.fitted_params, is_test=is_test)
        return df_curr

    def get_decision_log(self) -> List[Dict[str, Any]]:
        """Returns JSON-serializable decision audit trail."""
        return [
            {
                "step_id": s.step_id,
                "column": s.column,
                "option_id": s.option_id,
                "decided_by": s.decided_by,
                "timestamp": s.timestamp,
                "fitted_params": {k: v for k, v in s.fitted_params.items() if k not in ("tokens_to_null",)},
            }
            for s in self.steps
        ]

    def export_reproduce_code(self, raw_data_path: str = "data.csv") -> str:
        """
        Generates standalone, executable Python script reproduce.py.
        """
        lines = [
            '"""',
            'Auto-generated by Dataset Doctor: Reproducible Data Cleaning Pipeline.',
            f'Generated on: {datetime.now().isoformat()}',
            '"""',
            'import pandas as pd',
            'import numpy as np',
            'import re',
            '',
            'def clean_dataset(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:',
            '    """Applies the exact cleaning decisions learned during Dataset Doctor diagnosis."""',
            '    data = df.copy()',
            '',
        ]

        for s in self.steps:
            col = s.column
            opt = s.option_id
            params = s.fitted_params

            col_clean = re.sub(r"[\r\n]", "", str(col))
            opt_clean = re.sub(r"[\r\n]", "", str(opt))
            decided_clean = re.sub(r"[\r\n]", "", str(s.decided_by))
            step_clean = re.sub(r"[\r\n]", "", str(s.step_id))
            lines.append(f'    # Step: {step_clean} | Column: {col_clean} | Method: {opt_clean} | Decided by: {decided_clean}')
            if col and params.get("tokens_to_null"):
                lines.append(f'    if {repr(col)} in data.columns:')
                lines.append(f'        data[{repr(col)}] = data[{repr(col)}].replace({repr(params["tokens_to_null"])}, np.nan)')

            if opt == "drop_rows":
                fallback = params.get("test_fallback")
                fb_repr = repr(fallback)
                lines.append(f'    if not is_test:')
                lines.append(f'        data = data.dropna(subset=[{repr(col)}])')
                lines.append(f'    else:')
                lines.append(f'        data[{repr(col)}] = data[{repr(col)}].fillna({fb_repr})')

            elif opt == "drop_column":
                lines.append(f'    if {repr(col)} in data.columns:')
                lines.append(f'        data = data.drop(columns=[{repr(col)}])')

            elif opt in ("mean", "median", "mode", "constant", "missing_label"):
                fill_val = params.get("fill_value")
                val_repr = repr(fill_val)
                lines.append(f'    if {repr(col)} in data.columns:')
                lines.append(f'        data[{repr(col)}] = data[{repr(col)}].fillna({val_repr})')

            elif opt == "add_indicator":
                fill_val = params.get("fill_value")
                ind_col = params.get("indicator_col", f"{col}_was_missing")
                lines.append(f'    if {repr(col)} in data.columns:')
                lines.append(f'        data[{repr(ind_col)}] = data[{repr(col)}].isna().astype(int)')
                lines.append(f'        data[{repr(col)}] = data[{repr(col)}].fillna({repr(fill_val)})')

            elif opt == "ffill_interpolate":
                fallback = params.get("fallback")
                lines.append(f'    if {repr(col)} in data.columns:')
                if params.get("is_numeric"):
                    lines.append(f'        data[{repr(col)}] = data[{repr(col)}].interpolate(method="linear").ffill().bfill().fillna({repr(fallback)})')
                else:
                    lines.append(f'        data[{repr(col)}] = data[{repr(col)}].ffill().bfill().fillna({repr(fallback)})')

            elif opt == "keep_first":
                lines.append('    if not is_test:')
                lines.append('        data = data.drop_duplicates(keep="first")')

            elif opt == "keep_last":
                lines.append('    if not is_test:')
                lines.append('        data = data.drop_duplicates(keep="last")')

            elif opt == "drop_all_conflicting":
                tgt = params.get("target_column")
                lines.append(f'    if not is_test and {repr(tgt)} in data.columns and len(data.columns) > 1:')
                lines.append(f'        feat_cols = [c for c in data.columns if c != {repr(tgt)}]')
                lines.append('        feat_dups = data[data.duplicated(subset=feat_cols, keep=False)]')
                lines.append('        if not feat_dups.empty:')
                lines.append(f'            nunique = feat_dups.groupby(feat_cols)[{repr(tgt)}].transform("nunique")')
                lines.append('            data = data.drop(index=feat_dups[nunique > 1].index)')

            elif opt == "normalize_and_drop":
                lines.append('    if not is_test:')
                lines.append('        for c in data.columns:')
                lines.append('            if data[c].dtype == object or pd.api.types.is_string_dtype(data[c].dtype):')
                lines.append('                data[c] = data[c].astype(str).str.strip().str.lower()')
                lines.append('        data = data.drop_duplicates(keep="first")')

            elif opt == "convert_to_numeric":
                lines.append(f'    if {repr(col)} in data.columns:')
                lines.append('        def _strip_symbols_fn(val):')
                lines.append('            if pd.isna(val): return np.nan')
                lines.append('            s = str(val).strip()')
                lines.append('            for sym in ["$", "€", "£", "¥", "₹", "%", ","]:')
                lines.append('                s = s.replace(sym, "")')
                lines.append('            for u in ["kg", "km", "cm", "mm", "lbs", "m", "g", "oz", "usd", "eur", "k"]:')
                lines.append('                s = re.sub(rf"\\s*{u}\\b", "", s, flags=re.IGNORECASE)')
                lines.append('            try: return float(s.strip())')
                lines.append('            except: return np.nan')
                lines.append(f'        data[{repr(col)}] = data[{repr(col)}].apply(_strip_symbols_fn).astype(float)')

            elif opt == "parse_date_day_first":
                lines.append(f'    if {repr(col)} in data.columns:')
                lines.append(f'        data[{repr(col)}] = pd.to_datetime(data[{repr(col)}], dayfirst=True, errors="coerce")')

            elif opt == "parse_date_month_first":
                lines.append(f'    if {repr(col)} in data.columns:')
                lines.append(f'        data[{repr(col)}] = pd.to_datetime(data[{repr(col)}], dayfirst=False, errors="coerce")')

            elif opt == "parse_date":
                dfirst = bool(params.get("dayfirst", False))
                lines.append(f'    if {repr(col)} in data.columns:')
                lines.append(f'        data[{repr(col)}] = pd.to_datetime(data[{repr(col)}], dayfirst={dfirst}, errors="coerce")')

            elif opt == "encode_boolean":
                lines.append(f'    if {repr(col)} in data.columns:')
                lines.append('        _bmap = {"yes": 1, "no": 0, "y": 1, "n": 0, "true": 1, "false": 0, "t": 1, "f": 0, "1": 1, "0": 0}')
                lines.append(f'        data[{repr(col)}] = data[{repr(col)}].astype(str).str.strip().str.lower().map(_bmap).fillna(0).astype(int)')

            elif opt == "coerce_numeric":
                lines.append(f'    if {repr(col)} in data.columns:')
                lines.append(f'        data[{repr(col)}] = pd.to_numeric(data[{repr(col)}], errors="coerce")')

            elif opt == "set_to_null":
                lines.append(f'    if {repr(col)} in data.columns:')
                if col and (col.lower() == "age" or col.lower().endswith("_age")):
                    lines.append(f'        data.loc[(data[{repr(col)}] < 0) | (data[{repr(col)}] > 120), {repr(col)}] = np.nan')
                elif col and any(t in col.lower() for t in ["pct", "percent", "discount", "rate"]):
                    lines.append(f'        data.loc[(data[{repr(col)}] < 0) | (data[{repr(col)}] > 100), {repr(col)}] = np.nan')
                else:
                    lines.append(f'        data.loc[data[{repr(col)}] < 0, {repr(col)}] = np.nan')

            elif opt in ("clip_iqr", "clip_percentile"):
                lower = params.get("lower")
                upper = params.get("upper")
                lines.append(f'    if {repr(col)} in data.columns:')
                lines.append(f'        data[{repr(col)}] = data[{repr(col)}].clip(lower={lower}, upper={upper})')

            elif opt == "log_transform":
                shift = params.get("shift", 0.0)
                lines.append(f'    if {repr(col)} in data.columns:')
                lines.append(f'        data[{repr(col)}] = np.log1p(pd.to_numeric(data[{repr(col)}], errors="coerce") + {shift})')

            elif opt == "drop_rows":
                lower = params.get("lower")
                upper = params.get("upper")
                lines.append('    if not is_test:')
                lines.append(f'        if {repr(col)} in data.columns:')
                lines.append(f'            data = data[(data[{repr(col)}] >= {lower}) & (data[{repr(col)}] <= {upper})].copy()')

            elif opt == "random_oversample":
                lines.append('    if not is_test:')
                lines.append(f'        if {repr(col)} in data.columns:')
                lines.append(f'            max_c = data[{repr(col)}].value_counts().max()')
                lines.append(f'            dfs = [grp.sample(max_c, replace=True, random_state=42) for _, grp in data.groupby({repr(col)})]')
                lines.append('            data = pd.concat(dfs, ignore_index=True)')

            elif opt == "random_undersample":
                lines.append('    if not is_test:')
                lines.append(f'        if {repr(col)} in data.columns:')
                lines.append(f'            min_c = data[{repr(col)}].value_counts().min()')
                lines.append(f'            dfs = [grp.sample(min_c, replace=False, random_state=42) for _, grp in data.groupby({repr(col)})]')
                lines.append('            data = pd.concat(dfs, ignore_index=True)')

            elif opt == "class_weight":
                lines.append(f'    # Model-level: class_weight="balanced" applied to classifier when training on {repr(col)}')

            elif opt == "smote":
                lines.append('    if not is_test:')
                lines.append(f'        if {repr(col)} in data.columns:')
                lines.append('            from imblearn.over_sampling import SMOTE')
                lines.append(f'            _X = data.drop(columns=[{repr(col)}])')
                lines.append(f'            _y = data[{repr(col)}]')
                lines.append('            _sm = SMOTE(random_state=42)')
                lines.append('            _X_res, _y_res = _sm.fit_resample(_X, _y)')
                lines.append('            data = pd.DataFrame(_X_res, columns=_X.columns)')
                lines.append(f'            data[{repr(col)}] = _y_res')

            elif opt in ("threshold_tuning", "leave"):
                lines.append(f'    # Strategy "{opt}" on {repr(col)} preserves natural data distribution.')

            lines.append('')

        lines.extend([
            '    return data',
            '',
            'if __name__ == "__main__":',
            f'    raw_df = pd.read_csv({repr(raw_data_path)})',
            '    cleaned_df = clean_dataset(raw_df, is_test=False)',
            f'    print(f"Cleaned dataset shape: {{cleaned_df.shape}}")',
            '    cleaned_df.to_csv("cleaned_data.csv", index=False)',
        ])

        return "\n".join(lines)

    def export_reproduce_file(self, output_path: str, raw_data_path: str = "data.csv") -> Path:
        """Writes reproduce.py to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        code = self.export_reproduce_code(raw_data_path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        return path
