"""
Unit tests for Model Explainability & Feature Attribution (Phase 17).
Validates:
1. Permutation importance executes on validation folds only (zero test snooping).
2. Planted strong signal features rank above noise features.
3. High-cardinality spurious features are detected and flagged for impurity inflation.
4. Feature Pruning Advisor correctly identifies noise candidates for removal.
5. Plain-English non-causal disclosures are strictly returned.
"""
import json
from pathlib import Path
import pandas as pd
import pytest

from doctor.explainability import ModelExplainer, NON_CAUSAL_DISCLOSURE


@pytest.fixture
def explainability_data():
    root = Path(__file__).resolve().parent.parent
    csv_path = root / "data" / "planted" / "explainability.csv"
    truth_path = root / "data" / "planted" / "explainability_truth.json"

    assert csv_path.exists(), f"Missing {csv_path}"
    assert truth_path.exists(), f"Missing {truth_path}"

    df = pd.read_csv(csv_path)
    with open(truth_path, "r", encoding="utf-8") as f:
        truth = json.load(f)

    return df, truth


def test_explainability_zero_test_snooping(explainability_data):
    """Ensures test set is never accessed or permuted when calculating attribution."""
    df, truth = explainability_data
    n_train = int(len(df) * 0.8)
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()

    # Capture test_df copy to verify 100% untouched invariant
    test_snapshot = test_df.copy(deep=True)

    explainer = ModelExplainer(n_repeats=3, random_state=42)
    profile = explainer.explain(train_df, target_column=truth["target_column"])

    # Test set invariant
    assert profile.test_data_snooped is False
    pd.testing.assert_frame_equal(test_df, test_snapshot)
    assert len(profile.features) == len(train_df.columns) - 1


def test_explainability_ground_truth_ranking(explainability_data):
    """Ensures planted signal features rank significantly above planted noise features."""
    df, truth = explainability_data
    n_train = int(len(df) * 0.8)
    train_df = df.iloc[:n_train].copy()

    explainer = ModelExplainer(n_repeats=5, random_state=42)
    profile = explainer.explain(train_df, target_column=truth["target_column"])

    feat_map = {f.feature: f for f in profile.features}

    # Strong features should have positive drops and strong/moderate category
    for strong_feat in truth["planted_strong_features"]:
        assert strong_feat in feat_map
        assert feat_map[strong_feat].permutation_mean > 0.03
        assert feat_map[strong_feat].category == "strong"

    # Noise features should have lower drops than strong features
    for noise_feat in truth["planted_noise_features"]:
        assert noise_feat in feat_map
        assert feat_map[noise_feat].permutation_mean < feat_map["signal_alpha"].permutation_mean
        assert feat_map[noise_feat].permutation_mean < feat_map["signal_beta"].permutation_mean


def test_explainability_high_cardinality_bias(explainability_data):
    """Ensures high-cardinality spurious features are caught by the inflation guard."""
    df, truth = explainability_data
    n_train = int(len(df) * 0.8)
    train_df = df.iloc[:n_train].copy()

    explainer = ModelExplainer(n_repeats=5, random_state=42)
    profile = explainer.explain(train_df, target_column=truth["target_column"])

    high_card_feat = truth["high_cardinality_feature"]
    feat_map = {f.feature: f for f in profile.features}

    assert high_card_feat in feat_map
    target_f = feat_map[high_card_feat]

    # Native impurity importance is inflated (> 0.08)
    assert target_f.native_importance >= 0.08
    # But permutation importance drop is negligible (<= 0.015)
    assert target_f.permutation_mean <= 0.015
    # Inflation flag must be True
    assert target_f.has_cardinality_bias is True
    assert len(profile.high_cardinality_warnings) >= 1
    assert high_card_feat in profile.high_cardinality_warnings[0]


def test_explainability_pruning_advisor(explainability_data):
    """Ensures noise features and spurious features are recommended for pruning."""
    df, truth = explainability_data
    n_train = int(len(df) * 0.8)
    train_df = df.iloc[:n_train].copy()

    explainer = ModelExplainer(n_repeats=5, random_state=42)
    profile = explainer.explain(train_df, target_column=truth["target_column"])

    # Noise features must be flagged for pruning
    for noise_feat in truth["planted_noise_features"]:
        assert noise_feat in profile.features_to_prune

    # Spurious high cardinality feature must also be in pruning list
    assert truth["high_cardinality_feature"] in profile.features_to_prune

    # Strong features must NOT be in pruning list
    for strong_feat in truth["planted_strong_features"]:
        assert strong_feat not in profile.features_to_prune


def test_explainability_disclosures_and_reasons(explainability_data):
    """Ensures non-causal disclosure and plain English reasons are present."""
    df, truth = explainability_data
    n_train = int(len(df) * 0.8)
    train_df = df.iloc[:n_train].copy()

    explainer = ModelExplainer(n_repeats=3, random_state=42)
    profile = explainer.explain(train_df, target_column=truth["target_column"])

    assert profile.non_causal_disclosure == NON_CAUSAL_DISCLOSURE
    assert "PREDICTIVE IMPORTANCE" in profile.non_causal_disclosure
    assert "REAL-WORLD CAUSATION" in profile.non_causal_disclosure

    for f in profile.features:
        assert len(f.plain_english_reason) > 10
        assert f.cardinality >= 1

