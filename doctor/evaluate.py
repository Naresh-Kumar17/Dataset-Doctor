"""
Dataset Doctor: Fixed Model Evaluation & Noise Estimation Engine.
Rules:
- Fixed Model: RandomForest (200 trees, fixed seed; fast mode with fewer trees for advisor checks).
- Preprocessing: Self-contained preprocessing learns values ONLY from train (median for numerics,
  one-hot encoding for categoricals).
- Cross-Validation: 5-fold Stratified CV on TRAIN only (reports mean and std for noise estimation).
- Final Test: Single honest evaluation on the untouched TEST set.
- Metrics: Accuracy, Precision, Recall, F1 (minority class for imbalanced problems), and PR-AUC.
- Imbalance Guard: If minority class < 20%, accuracy is flagged as "not reliable here".
- Noise Guard: is_significant() checks whether a score delta exceeds the CV standard deviation.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


@dataclass
class EvalResult:
    """Encapsulates the complete evaluation result including CV and test scores."""
    task_type: Literal["classification", "regression"]
    primary_metric: str
    train_cv_mean: Dict[str, float]
    train_cv_std: Dict[str, float]
    test_scores: Dict[str, float]
    minority_share: Optional[float] = None
    minority_label: Optional[Any] = None
    accuracy_reliable: bool = True
    warnings: List[str] = field(default_factory=list)
    majority_baseline: Optional[Dict[str, float]] = None


class Preprocessor:
    """
    Leak-free simple preprocessor. Fits transformations strictly on train
    and applies them to train or test.
    """
    def __init__(self):
        self.numeric_medians: Dict[str, float] = {}
        self.categorical_cols: List[str] = []
        self.numeric_cols: List[str] = []
        self.encoded_columns: List[str] = []

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()
        self.numeric_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
        self.categorical_cols = [c for c in X_df.columns if c not in self.numeric_cols]

        # 1. Learn medians on numeric columns
        for col in self.numeric_cols:
            med = float(X_df[col].median()) if not X_df[col].dropna().empty else 0.0
            self.numeric_medians[col] = med
            X_df[col] = X_df[col].fillna(med)

        # 2. Impute categoricals with __MISSING__
        for col in self.categorical_cols:
            X_df[col] = X_df[col].fillna("__MISSING__").astype(str)

        # 3. One-hot encode categoricals
        if self.categorical_cols:
            X_encoded = pd.get_dummies(X_df, columns=self.categorical_cols, drop_first=True)
        else:
            X_encoded = X_df

        self.encoded_columns = list(X_encoded.columns)
        return X_encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()
        # 1. Apply learned medians
        for col in self.numeric_cols:
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(self.numeric_medians.get(col, 0.0))
            else:
                X_df[col] = self.numeric_medians.get(col, 0.0)

        # 2. Apply categoricals
        for col in self.categorical_cols:
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna("__MISSING__").astype(str)
            else:
                X_df[col] = "__MISSING__"

        # 3. One-hot encode
        if self.categorical_cols:
            X_encoded = pd.get_dummies(X_df, columns=self.categorical_cols, drop_first=True)
        else:
            X_encoded = X_df

        # Align columns with train set (fill unseen with 0)
        X_aligned = X_encoded.reindex(columns=self.encoded_columns, fill_value=0)
        return X_aligned


def _compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray], pos_label: Any) -> Dict[str, float]:
    """Computes standard evaluation metrics for binary classification."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))

    pr_auc_score = 0.0
    if y_prob is not None:
        try:
            # Binary classification probability
            prec_arr, rec_arr, _ = precision_recall_curve(y_true == pos_label, y_prob)
            pr_auc_score = float(auc(rec_arr, prec_arr))
        except Exception:
            pr_auc_score = 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "pr_auc": round(pr_auc_score, 4),
    }


def _compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes standard evaluation metrics for regression."""
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
    }


def evaluate_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    task_type: Literal["classification", "regression"] = "classification",
    fast: bool = False,
    random_seed: int = 42,
    cv_folds: int = 5,
) -> EvalResult:
    """
    Evaluates a dataset split using fixed Random Forest models with 5-fold CV on train
    and a final test score on the untouched test partition.
    """
    warnings: List[str] = []
    total_rows = len(train_df) + len(test_df)
    if total_rows < 500:
        warnings.append(f"Dataset has {total_rows} rows (< 500). Comparisons may be noisy.")

    X_train_raw = train_df.drop(columns=[target])
    y_train_raw = train_df[target].values
    X_test_raw = test_df.drop(columns=[target])
    y_test_raw = test_df[target].values

    n_trees = 50 if fast else 200

    if task_type == "classification":
        # 1. Analyze class balance and minority class
        unique_classes, counts = np.unique(y_train_raw, return_counts=True)
        minority_idx = int(np.argmin(counts))
        minority_label = unique_classes[minority_idx]
        minority_share = float(counts[minority_idx] / len(y_train_raw))

        accuracy_reliable = minority_share >= 0.20
        primary_metric = "f1" if not accuracy_reliable else "f1"
        if not accuracy_reliable:
            warnings.append(
                f"Imbalanced target: minority class is {minority_share * 100:.1f}% (< 20%). "
                "Accuracy is NOT reliable here; use minority F1 and Recall."
            )

        # 2. Majority class baseline calculation
        majority_idx = int(np.argmax(counts))
        majority_label = unique_classes[majority_idx]
        y_maj_pred = np.full_like(y_test_raw, fill_value=majority_label)
        majority_baseline = {
            "accuracy": round(float(accuracy_score(y_test_raw, y_maj_pred)), 4),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "pr_auc": 0.0,
        }

        # 3. 5-Fold Stratified Cross-Validation on TRAIN
        min_class_samples = int(np.min(counts))
        actual_splits = min(cv_folds, min_class_samples)
        if actual_splits < 2:
            actual_splits = 2

        skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_seed)
        fold_metrics: List[Dict[str, float]] = []

        for train_fold_idx, val_fold_idx in skf.split(X_train_raw, y_train_raw):
            X_tr = X_train_raw.iloc[train_fold_idx]
            y_tr = y_train_raw[train_fold_idx]
            X_val = X_train_raw.iloc[val_fold_idx]
            y_val = y_train_raw[val_fold_idx]

            prep = Preprocessor()
            X_tr_enc = prep.fit_transform(X_tr)
            X_val_enc = prep.transform(X_val)

            clf = RandomForestClassifier(n_estimators=n_trees, random_state=random_seed, n_jobs=-1)
            clf.fit(X_tr_enc, y_tr)
            y_pred = clf.predict(X_val_enc)

            y_prob = None
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_val_enc)
                if minority_label in clf.classes_:
                    pos_col = list(clf.classes_).index(minority_label)
                    y_prob = probs[:, pos_col]

            metrics = _compute_classification_metrics(y_val, y_pred, y_prob, pos_label=minority_label)
            fold_metrics.append(metrics)

        # Calculate CV mean and std
        train_cv_mean = {}
        train_cv_std = {}
        metric_keys = ["accuracy", "precision", "recall", "f1", "pr_auc"]
        for k in metric_keys:
            vals = [fm[k] for fm in fold_metrics]
            train_cv_mean[k] = round(float(np.mean(vals)), 4)
            train_cv_std[k] = round(float(np.std(vals)), 4)

        # 4. Final Untouched Test Evaluation
        full_prep = Preprocessor()
        X_train_full = full_prep.fit_transform(X_train_raw)
        X_test_full = full_prep.transform(X_test_raw)

        final_clf = RandomForestClassifier(n_estimators=n_trees, random_state=random_seed, n_jobs=-1)
        final_clf.fit(X_train_full, y_train_raw)
        y_test_pred = final_clf.predict(X_test_full)

        y_test_prob = None
        if hasattr(final_clf, "predict_proba"):
            probs = final_clf.predict_proba(X_test_full)
            if minority_label in final_clf.classes_:
                pos_col = list(final_clf.classes_).index(minority_label)
                y_test_prob = probs[:, pos_col]

        test_scores = _compute_classification_metrics(y_test_raw, y_test_pred, y_test_prob, pos_label=minority_label)

        return EvalResult(
            task_type="classification",
            primary_metric=primary_metric,
            train_cv_mean=train_cv_mean,
            train_cv_std=train_cv_std,
            test_scores=test_scores,
            minority_share=minority_share,
            minority_label=minority_label,
            accuracy_reliable=accuracy_reliable,
            warnings=warnings,
            majority_baseline=majority_baseline,
        )

    else:
        # Regression task
        primary_metric = "r2"
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        fold_metrics = []

        for train_fold_idx, val_fold_idx in kf.split(X_train_raw, y_train_raw):
            X_tr = X_train_raw.iloc[train_fold_idx]
            y_tr = y_train_raw[train_fold_idx]
            X_val = X_train_raw.iloc[val_fold_idx]
            y_val = y_train_raw[val_fold_idx]

            prep = Preprocessor()
            X_tr_enc = prep.fit_transform(X_tr)
            X_val_enc = prep.transform(X_val)

            reg = RandomForestRegressor(n_estimators=n_trees, random_state=random_seed, n_jobs=-1)
            reg.fit(X_tr_enc, y_tr)
            y_pred = reg.predict(X_val_enc)

            metrics = _compute_regression_metrics(y_val, y_pred)
            fold_metrics.append(metrics)

        train_cv_mean = {}
        train_cv_std = {}
        metric_keys = ["r2", "rmse", "mae"]
        for k in metric_keys:
            vals = [fm[k] for fm in fold_metrics]
            train_cv_mean[k] = round(float(np.mean(vals)), 4)
            train_cv_std[k] = round(float(np.std(vals)), 4)

        # Untouched test evaluation
        full_prep = Preprocessor()
        X_train_full = full_prep.fit_transform(X_train_raw)
        X_test_full = full_prep.transform(X_test_raw)

        final_reg = RandomForestRegressor(n_estimators=n_trees, random_state=random_seed, n_jobs=-1)
        final_reg.fit(X_train_full, y_train_raw)
        y_test_pred = final_reg.predict(X_test_full)
        test_scores = _compute_regression_metrics(y_test_raw, y_test_pred)

        return EvalResult(
            task_type="regression",
            primary_metric=primary_metric,
            train_cv_mean=train_cv_mean,
            train_cv_std=train_cv_std,
            test_scores=test_scores,
            accuracy_reliable=True,
            warnings=warnings,
        )


def is_significant(
    metric: str,
    before_result: EvalResult,
    after_result: EvalResult,
) -> Tuple[bool, str]:
    """
    Determines if a metric delta is statistically meaningful or within noise.
    A change smaller than the CV standard deviation is classified as "within noise".
    """
    before_score = before_result.train_cv_mean.get(metric, before_result.test_scores.get(metric, 0.0))
    after_score = after_result.train_cv_mean.get(metric, after_result.test_scores.get(metric, 0.0))
    std = before_result.train_cv_std.get(metric, 0.0)

    delta = after_score - before_score
    if abs(delta) <= std or abs(delta) < 1e-4:
        return False, "within noise"
    elif delta > 0:
        return True, "improved"
    else:
        return True, "degraded"


# Alias for programmatic SDK
evaluate_model = evaluate_split
