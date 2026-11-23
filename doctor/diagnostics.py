"""
Dataset Doctor: Model Error Analysis & Subgroup Diagnostics (Phase 16).
Rules:
- Confusion matrix and error rate calculations derived strictly from model predictions on test partition.
- Isolates top high-confidence prediction errors (False Positives and False Negatives) with original context.
- Slicing / Subgroup analysis: Identifies blind spots where error rate is disproportionately high (> 1.5x average).
- Confidence separation analysis: Measures calibration gap between correct predictions and errors.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


@dataclass
class SliceError:
    feature: str
    slice_value: str
    sample_count: int
    error_count: int
    error_rate: float
    is_blind_spot: bool


@dataclass
class ErrorProfile:
    total_samples: int
    total_errors: int
    overall_error_rate: float
    tp: int
    fp: int
    tn: int
    fn: int
    false_positive_rate: float
    false_negative_rate: float
    top_false_positives: List[Dict[str, Any]]
    top_false_negatives: List[Dict[str, Any]]
    blind_spots: List[SliceError]
    slice_analysis: List[SliceError]
    mean_correct_confidence: float
    mean_error_confidence: float


class ModelErrorAnalyzer:
    """
    Diagnoses model failure modes on the locked test partition.
    """
    def __init__(self, top_n_errors: int = 5):
        self.top_n_errors = top_n_errors

    def analyze(
        self,
        model: Any,
        test_df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
    ) -> ErrorProfile:
        X_test_raw = test_df.drop(columns=[target_column])
        y_test = test_df[target_column].values

        # Encode features aligned with training
        X_encoded = pd.get_dummies(X_test_raw, drop_first=True)
        for col in X_encoded.columns:
            if X_encoded[col].isna().any():
                X_encoded[col] = X_encoded[col].fillna(0)

        # Align columns if feature_columns is provided
        if feature_columns:
            missing_cols = [c for c in feature_columns if c not in X_encoded.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0, index=X_encoded.index, columns=missing_cols)
                X_encoded = pd.concat([X_encoded, missing_df], axis=1)
            X_encoded = X_encoded[feature_columns]

        X_test = X_encoded.values

        # Predictions
        preds = model.predict(X_test)
        has_proba = hasattr(model, "predict_proba")
        probs = model.predict_proba(X_test)[:, 1] if has_proba else preds.astype(float)

        # Confusion matrix
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        total_samples = len(y_test)
        total_errors = fp + fn
        overall_error_rate = float(total_errors / max(1, total_samples))

        fpr = float(fp / max(1, fp + tn))
        fnr = float(fn / max(1, fn + tp))

        # Confidence tracking
        correct_mask = (preds == y_test)
        error_mask = ~correct_mask

        confidences = np.where(preds == 1, probs, 1.0 - probs)
        mean_correct_conf = float(np.mean(confidences[correct_mask])) if np.any(correct_mask) else 0.0
        mean_error_conf = float(np.mean(confidences[error_mask])) if np.any(error_mask) else 0.0

        # Top False Positives (predicted 1, actual 0, ranked by highest prob)
        fp_indices = np.where((preds == 1) & (y_test == 0))[0]
        fp_sorted = fp_indices[np.argsort(-probs[fp_indices])][:self.top_n_errors]

        top_fps = []
        for idx in fp_sorted:
            row_dict = test_df.iloc[idx].to_dict()
            row_dict["_predicted_prob"] = round(float(probs[idx]), 4)
            row_dict["_error_type"] = "False Positive"
            top_fps.append(row_dict)

        # Top False Negatives (predicted 0, actual 1, ranked by lowest prob)
        fn_indices = np.where((preds == 0) & (y_test == 1))[0]
        fn_sorted = fn_indices[np.argsort(probs[fn_indices])][:self.top_n_errors]

        top_fns = []
        for idx in fn_sorted:
            row_dict = test_df.iloc[idx].to_dict()
            row_dict["_predicted_prob"] = round(float(probs[idx]), 4)
            row_dict["_error_type"] = "False Negative"
            top_fns.append(row_dict)

        # Slicing Analysis: check categorical features for subgroup blind spots
        slice_results: List[SliceError] = []
        blind_spots: List[SliceError] = []

        cat_cols = [
            c for c in test_df.columns
            if c != target_column and (
                pd.api.types.is_string_dtype(test_df[c].dtype)
                or isinstance(test_df[c].dtype, pd.CategoricalDtype)
                or test_df[c].dtype == object
                or isinstance(test_df[c].dtype, pd.StringDtype)
                or (not pd.api.types.is_float_dtype(test_df[c].dtype) and test_df[c].nunique() <= 15)
            )
        ]

        for col in cat_cols:
            for val, grp in test_df.groupby(col):
                grp_indices = grp.index
                pos_in_test = [test_df.index.get_loc(i) for i in grp_indices]
                grp_errors = int(np.sum(error_mask[pos_in_test]))
                grp_total = len(grp)
                if grp_total >= 5:  # Minimum sample size for slice assessment
                    grp_err_rate = float(grp_errors / grp_total)
                    is_blind = bool(grp_err_rate > 1.5 * overall_error_rate and grp_errors > 0)
                    se = SliceError(
                        feature=col,
                        slice_value=str(val),
                        sample_count=grp_total,
                        error_count=grp_errors,
                        error_rate=round(grp_err_rate, 4),
                        is_blind_spot=is_blind,
                    )
                    slice_results.append(se)
                    if is_blind:
                        blind_spots.append(se)

        return ErrorProfile(
            total_samples=total_samples,
            total_errors=total_errors,
            overall_error_rate=round(overall_error_rate, 4),
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            false_positive_rate=round(fpr, 4),
            false_negative_rate=round(fnr, 4),
            top_false_positives=top_fps,
            top_false_negatives=top_fns,
            blind_spots=blind_spots,
            slice_analysis=slice_results,
            mean_correct_confidence=round(mean_correct_conf, 4),
            mean_error_confidence=round(mean_error_conf, 4),
        )
