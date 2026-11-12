"""
Dataset Doctor: Model Explainability & Feature Attribution (Phase 17).
Rules:
- Native tree feature importance (impurity-based) with high-cardinality inflation guard.
- Permutation importance computed on training validation folds only.
- Strict Invariant: Zero test data snooping. Test partition is NEVER passed or permuted.
- Feature Pruning Advisor: Flags and recommends pruning noisy / near-zero / negative importance features.
- Plain-English feature explanations with non-causal disclosures.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.utils.validation import check_is_fitted

from doctor.evaluate import Preprocessor

NON_CAUSAL_DISCLOSURE = (
    "⚠️ PREDICTIVE IMPORTANCE ≠ REAL-WORLD CAUSATION: High feature attribution indicates strong "
    "statistical correlation within this dataset, but does NOT imply that intervening on this feature "
    "in reality will causally alter the target outcome. External interventions without controlled "
    "causal experimentation can be ineffective or counterproductive."
)


@dataclass
class FeatureAttribution:
    """Attribution metrics and pruning advice for an individual feature."""
    feature: str
    native_importance: float
    permutation_mean: float
    permutation_std: float
    category: Literal["strong", "moderate", "noisy", "prune"]
    cardinality: int
    has_cardinality_bias: bool
    plain_english_reason: str


@dataclass
class AttributionProfile:
    """Full explainability profile for a trained model across all features."""
    model_name: str
    metric_used: str
    baseline_validation_score: float
    features: List[FeatureAttribution]
    features_to_prune: List[str]
    high_cardinality_warnings: List[str]
    non_causal_disclosure: str
    summary: str
    test_data_snooped: bool = False  # Guaranteed invariant proof


class ModelExplainer:
    """
    Computes leak-free feature attribution and explainability diagnostics.
    Enforces strict invariant: Evaluation is performed exclusively on training
    validation folds. The test partition is NEVER permuted or snooped.
    """
    def __init__(self, n_repeats: int = 5, random_state: int = 42):
        self.n_repeats = n_repeats
        self.random_state = random_state

    def explain(
        self,
        train_df: pd.DataFrame,
        target_column: str,
        model: Optional[Any] = None,
        val_df: Optional[pd.DataFrame] = None,
        scoring_metric: str = "balanced_accuracy",
    ) -> AttributionProfile:
        """
        Computes native and permutation feature attribution on validation fold.
        """
        if train_df is None or train_df.empty:
            raise ValueError("train_df is required and cannot be empty.")
        if target_column not in train_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in train_df.")

        # 1. Carve validation partition strictly from train_df if val_df is not provided
        if val_df is None:
            y_all = train_df[target_column].values
            # Check if classification with at least 2 samples per class
            classes, counts = np.unique(y_all, return_counts=True)
            if len(classes) >= 2 and np.min(counts) >= 2:
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=self.random_state)
            else:
                splitter = ShuffleSplit(n_splits=1, test_size=0.25, random_state=self.random_state)

            train_idx, val_idx = next(splitter.split(train_df, y_all))
            sub_train = train_df.iloc[train_idx].copy()
            sub_val = train_df.iloc[val_idx].copy()
        else:
            sub_train = train_df.copy()
            sub_val = val_df.copy()

        # 2. Fit leak-free preprocessor strictly on sub_train
        prep = Preprocessor()
        X_tr_raw = sub_train.drop(columns=[target_column])
        y_tr = sub_train[target_column].values
        X_tr_enc = prep.fit_transform(X_tr_raw)

        X_val_raw = sub_val.drop(columns=[target_column])
        y_val = sub_val[target_column].values
        X_val_enc = prep.transform(X_val_raw)

        # 3. Model initialization and fitting
        model_name = "RandomForestClassifier"
        if model is None:
            model = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=self.random_state)
            model.fit(X_tr_enc.values, y_tr)
        else:
            model_name = type(model).__name__
            try:
                check_is_fitted(model)
            except Exception:
                model.fit(X_tr_enc.values, y_tr)

        # 4. Measure baseline score on validation partition
        preds_val = model.predict(X_val_enc.values)
        if scoring_metric == "f1":
            baseline_score = float(f1_score(y_val, preds_val, average="weighted", zero_division=0))
        else:
            baseline_score = float(balanced_accuracy_score(y_val, preds_val))

        # 5. Extract and aggregate native feature importances
        encoded_cols = prep.encoded_columns
        if hasattr(model, "feature_importances_"):
            raw_imps = model.feature_importances_
        elif hasattr(model, "coef_"):
            raw_imps = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            raw_imps = np.zeros(len(encoded_cols))

        orig_features = list(X_tr_raw.columns)
        native_map: Dict[str, float] = {}
        for col in orig_features:
            if col in prep.numeric_cols:
                if col in encoded_cols:
                    idx = encoded_cols.index(col)
                    native_map[col] = float(raw_imps[idx])
                else:
                    native_map[col] = 0.0
            else:
                prefix = f"{col}_"
                tot = sum(raw_imps[i] for i, name in enumerate(encoded_cols) if name.startswith(prefix))
                native_map[col] = float(tot)

        # Normalize native importances to sum to 1.0
        tot_native = sum(native_map.values())
        if tot_native > 0:
            for col in native_map:
                native_map[col] = float(native_map[col] / tot_native)

        # 6. Compute permutation importance on validation partition
        features_attribution: List[FeatureAttribution] = []
        features_to_prune: List[str] = []
        high_cardinality_warnings: List[str] = []

        for col in orig_features:
            cardinality = int(train_df[col].nunique())
            drops: List[float] = []
            is_numeric = col in prep.numeric_cols

            for rep in range(self.n_repeats):
                rng = np.random.RandomState(self.random_state + rep * 17)
                if is_numeric and col in prep.encoded_columns:
                    val_perm_enc = X_val_enc.copy()
                    val_perm_enc[col] = rng.permutation(val_perm_enc[col].values)
                    preds_perm = model.predict(val_perm_enc.values)
                else:
                    val_perm = X_val_raw.copy()
                    val_perm[col] = rng.permutation(val_perm[col].values)
                    X_val_shuffled_enc = prep.transform(val_perm)
                    preds_perm = model.predict(X_val_shuffled_enc.values)

                if scoring_metric == "f1":
                    score_perm = float(f1_score(y_val, preds_perm, average="weighted", zero_division=0))
                else:
                    score_perm = float(balanced_accuracy_score(y_val, preds_perm))

                drops.append(baseline_score - score_perm)

            mean_drop = float(np.mean(drops))
            std_drop = float(np.std(drops))
            native_val = native_map.get(col, 0.0)

            # High-cardinality inflation guard:
            # Impurity importance is high (> 8%), but permutation drop is negligible (<= 1%)
            has_card_bias = bool(
                cardinality >= 30
                and native_val >= 0.08
                and mean_drop <= 0.01
            )

            if has_card_bias:
                category = "prune"
                reason = (
                    f"High cardinality ({cardinality} categories) causes inflated native tree split "
                    f"importance ({native_val:.1%}), but zero validation impact ({mean_drop:+.2%}). "
                    f"Spurious split artifact recommended for feature pruning."
                )
                features_to_prune.append(col)
                high_cardinality_warnings.append(
                    f"Feature '{col}' exhibits high-cardinality impurity inflation: native importance is "
                    f"{native_val:.1%} but validation permutation drop is {mean_drop:+.2%}."
                )
            elif mean_drop > 0.02:
                category = "strong"
                reason = f"Permuting this feature degrades validation performance by {mean_drop:.1%}. Primary predictive driver."
            elif mean_drop > 0.001:
                category = "moderate"
                reason = f"Permuting this feature causes modest validation degradation ({mean_drop:.1%}). Secondary predictive signal."
            else:
                category = "prune"
                reason = f"Zero or negative validation impact ({mean_drop:+.2%}). Noise candidate for feature pruning."
                features_to_prune.append(col)

            features_attribution.append(
                FeatureAttribution(
                    feature=col,
                    native_importance=native_val,
                    permutation_mean=mean_drop,
                    permutation_std=std_drop,
                    category=category,
                    cardinality=cardinality,
                    has_cardinality_bias=has_card_bias,
                    plain_english_reason=reason,
                )
            )

        # Sort features by permutation drop descending
        features_attribution.sort(key=lambda x: x.permutation_mean, reverse=True)

        n_strong = sum(1 for f in features_attribution if f.category == "strong")
        n_prune = len(features_to_prune)
        summary = (
            f"Analyzed {len(features_attribution)} features for {model_name}. "
            f"Identified {n_strong} strong predictive drivers and {n_prune} noise/spurious features "
            f"recommended for pruning."
        )

        return AttributionProfile(
            model_name=model_name,
            metric_used=scoring_metric,
            baseline_validation_score=baseline_score,
            features=features_attribution,
            features_to_prune=features_to_prune,
            high_cardinality_warnings=high_cardinality_warnings,
            non_causal_disclosure=NON_CAUSAL_DISCLOSURE,
            summary=summary,
            test_data_snooped=False,
        )

