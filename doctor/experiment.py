"""
Dataset Doctor: Multi-Model Experimentation & Comparison Engine (Phase 15).
Rules:
- Evaluates multiple model architectures against the same locked test split.
- Models: Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting.
- Multi-metric matrix: Accuracy, Balanced Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC.
- Overfitting Guard: Flags significant train vs test generalization gaps.
- Minority Metric Guard: Warns when high overall accuracy masks near-zero minority recall.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


@dataclass
class ModelResult:
    model_name: str
    model_family: str
    train_accuracy: float
    test_accuracy: float
    train_f1: float
    test_f1: float
    test_balanced_accuracy: float
    test_precision: float
    test_recall: float
    test_roc_auc: Optional[float]
    test_pr_auc: Optional[float]
    is_overfitting: bool
    accuracy_illusion_warning: bool
    fitted_model: Any = field(repr=False, default=None)


@dataclass
class ExperimentComparison:
    results: List[ModelResult]
    best_model_f1: str
    best_model_pr_auc: str
    best_overall_model: str
    has_accuracy_illusion: bool
    feature_columns: List[str]
    target_column: str


class MultiModelExperiment:
    """
    Trains and compares multiple classifiers on a clean, split dataset.
    """
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def _prepare_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        X_train_raw = train_df.drop(columns=[target_column])
        y_train = train_df[target_column].values
        X_test_raw = test_df.drop(columns=[target_column])
        y_test = test_df[target_column].values

        # Simple robust encoding: convert categoricals using pandas get_dummies aligned
        X_combined = pd.concat([X_train_raw, X_test_raw], axis=0)
        # Select numeric and categorical
        X_encoded = pd.get_dummies(X_combined, drop_first=True)
        # Fill any remaining NaNs with column median/mode for training safety
        for col in X_encoded.columns:
            if X_encoded[col].isna().any():
                fill_val = X_encoded[col].median() if pd.api.types.is_numeric_dtype(X_encoded[col]) else 0
                X_encoded[col] = X_encoded[col].fillna(fill_val)

        n_train = len(train_df)
        X_train = X_encoded.iloc[:n_train].values
        X_test = X_encoded.iloc[n_train:].values
        feature_names = list(X_encoded.columns)

        return X_train, y_train, X_test, y_test, feature_names

    def run_experiment(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str,
    ) -> ExperimentComparison:
        X_train, y_train, X_test, y_test, feat_cols = self._prepare_data(train_df, test_df, target_column)

        models = [
            ("Logistic Regression", "linear", LogisticRegression(max_iter=1000, random_state=self.random_seed)),
            ("Decision Tree", "tree", DecisionTreeClassifier(max_depth=5, random_state=self.random_seed)),
            ("Random Forest", "ensemble", RandomForestClassifier(n_estimators=50, max_depth=5, random_state=self.random_seed)),
            ("Gradient Boosting", "boosting", HistGradientBoostingClassifier(max_iter=50, random_state=self.random_seed)),
        ]

        results = []
        any_illusion = False

        for name, family, model in models:
            # Fit strictly on train
            model.fit(X_train, y_train)

            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)

            train_acc = float(accuracy_score(y_train, train_preds))
            test_acc = float(accuracy_score(y_test, test_preds))

            # Minority class is assumed label 1 for binary
            train_f1 = float(f1_score(y_train, train_preds, zero_division=0))
            test_f1 = float(f1_score(y_test, test_preds, zero_division=0))

            test_bal_acc = float(balanced_accuracy_score(y_test, test_preds))
            test_prec = float(precision_score(y_test, test_preds, zero_division=0))
            test_rec = float(recall_score(y_test, test_preds, zero_division=0))

            test_roc_auc = None
            test_pr_auc = None
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(X_test)[:, 1]
                    test_roc_auc = float(roc_auc_score(y_test, probs))
                    test_pr_auc = float(average_precision_score(y_test, probs))
                except Exception:
                    pass

            # Overfitting detection: significant drop in F1 or accuracy
            is_overfit = bool((train_f1 - test_f1 > 0.18) or (train_acc - test_acc > 0.15))

            # Accuracy illusion: accuracy is high but minority recall is poor
            illusion = bool((test_acc >= 0.85 and test_rec <= 0.50) or (test_acc - test_bal_acc >= 0.15))
            if illusion:
                any_illusion = True

            results.append(ModelResult(
                model_name=name,
                model_family=family,
                train_accuracy=train_acc,
                test_accuracy=test_acc,
                train_f1=train_f1,
                test_f1=test_f1,
                test_balanced_accuracy=test_bal_acc,
                test_precision=test_prec,
                test_recall=test_rec,
                test_roc_auc=test_roc_auc,
                test_pr_auc=test_pr_auc,
                is_overfitting=is_overfit,
                accuracy_illusion_warning=illusion,
                fitted_model=model,
            ))

        # Determine best models
        best_f1 = max(results, key=lambda r: r.test_f1).model_name
        best_pr = max(results, key=lambda r: (r.test_pr_auc or 0.0)).model_name
        best_overall = max(results, key=lambda r: (r.test_f1 * 0.6 + (r.test_pr_auc or 0.0) * 0.4)).model_name

        return ExperimentComparison(
            results=results,
            best_model_f1=best_f1,
            best_model_pr_auc=best_pr,
            best_overall_model=best_overall,
            has_accuracy_illusion=any_illusion,
            feature_columns=feat_cols,
            target_column=target_column,
        )
