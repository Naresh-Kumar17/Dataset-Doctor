"""
Unit tests for Phase 16: Model Error Analysis & Diagnostics.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from doctor.diagnostics import ModelErrorAnalyzer


def test_error_analysis_diagnostics():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_cat": np.random.choice(["X", "Y", "Z"], size=n),
        "target": np.random.choice([0, 1], p=[0.8, 0.2], size=n),
    })

    train_df = df.iloc[:150].copy()
    test_df = df.iloc[150:].copy()

    X_train = pd.get_dummies(train_df.drop(columns=["target"]), drop_first=True)
    y_train = train_df["target"].values
    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X_train.values, y_train)

    analyzer = ModelErrorAnalyzer(top_n_errors=5)
    prof = analyzer.analyze(clf, test_df, target_column="target", feature_columns=list(X_train.columns))

    # Confusion matrix checks
    assert (prof.tp + prof.fp + prof.tn + prof.fn) == len(test_df)
    assert 0.0 <= prof.false_positive_rate <= 1.0
    assert 0.0 <= prof.false_negative_rate <= 1.0

    # Top errors
    assert isinstance(prof.top_false_positives, list)
    assert isinstance(prof.top_false_negatives, list)

    # Subgroup slicing
    assert len(prof.slice_analysis) > 0
    cat_slices = [s for s in prof.slice_analysis if s.feature == "feature_cat"]
    assert len(cat_slices) > 0

    # Confidence separation
    assert 0.0 <= prof.mean_correct_confidence <= 1.0
    assert 0.0 <= prof.mean_error_confidence <= 1.0
