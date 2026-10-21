"""
Unit tests for doctor/evaluate.py and doctor/compare.py (Phase 3: Evaluation Engine).
Verifies:
1. Fixed Random Forest evaluates 5-fold CV on train and single score on untouched test.
2. Imbalance safety guard: Minority class < 20% flags accuracy as unreliable.
3. Majority baseline: Always-predict-majority model shows ~85% accuracy and 0.0 minority recall on accident data.
4. Noise estimation: is_significant() classifies score differences within CV std as 'within noise'.
5. Small dataset guard: Datasets < 500 rows produce noisy comparison warning.
6. Split-order demo: Oversampling before split inflates test recall vs split-first (synthetic leakage proof).
7. Regression evaluation: Computes R2, RMSE, and MAE.
"""
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from doctor.split import split_dataset
from doctor.evaluate import evaluate_split, is_significant, EvalResult
from doctor.compare import compare_evaluations, run_split_order_demo


ACCIDENT_CSV = Path(__file__).resolve().parent.parent / "data" / "samples" / "accident.csv"


@pytest.fixture
def accident_df():
    """Loads the real road accident dataset (2000 rows, 85/15 ratio)."""
    assert ACCIDENT_CSV.exists(), f"Accident dataset not found at {ACCIDENT_CSV}"
    return pd.read_csv(ACCIDENT_CSV)


def test_evaluate_split_classification_fast_mode(accident_df):
    """Gate 3: 5-fold CV on train and final score on test using fast mode."""
    split = split_dataset(accident_df, target="severe_accident", random_seed=42)
    res = evaluate_split(
        train_df=split.train_df,
        test_df=split.test_df,
        target="severe_accident",
        task_type="classification",
        fast=True,
        random_seed=42,
    )

    assert res.task_type == "classification"
    assert "accuracy" in res.train_cv_mean
    assert "f1" in res.train_cv_mean
    assert "recall" in res.train_cv_mean
    assert "pr_auc" in res.train_cv_mean

    # CV std exists and is non-negative
    assert res.train_cv_std["f1"] >= 0.0

    # Test scores exist
    assert "accuracy" in res.test_scores
    assert "f1" in res.test_scores

    # Imbalance guard fires on 85/15 ratio
    assert res.minority_share is not None
    assert res.minority_share < 0.20
    assert res.accuracy_reliable is False
    assert any("not reliable" in w.lower() for w in res.warnings)


def test_always_predict_majority_baseline(accident_df):
    """Gate 3: Always-predict-majority model shows ~85% accuracy and 0.0 minority recall."""
    split = split_dataset(accident_df, target="severe_accident", random_seed=42)
    res = evaluate_split(
        train_df=split.train_df,
        test_df=split.test_df,
        target="severe_accident",
        fast=True,
        random_seed=42,
    )

    baseline = res.majority_baseline
    assert baseline is not None
    assert abs(baseline["accuracy"] - 0.85) < 0.02
    assert baseline["recall"] == 0.0
    assert baseline["f1"] == 0.0


def test_is_significant_noise_detection():
    """Gate 3: Changes smaller than CV std are labelled 'within noise'."""
    before = EvalResult(
        task_type="classification",
        primary_metric="f1",
        train_cv_mean={"f1": 0.50, "accuracy": 0.85},
        train_cv_std={"f1": 0.04, "accuracy": 0.02},
        test_scores={"f1": 0.50, "accuracy": 0.85},
    )

    # After is +0.02, but std is 0.04 -> within noise
    after_small = EvalResult(
        task_type="classification",
        primary_metric="f1",
        train_cv_mean={"f1": 0.52, "accuracy": 0.86},
        train_cv_std={"f1": 0.04, "accuracy": 0.02},
        test_scores={"f1": 0.52, "accuracy": 0.86},
    )

    sig, label = is_significant("f1", before, after_small)
    assert sig is False
    assert label == "within noise"

    # After is +0.08, which exceeds std 0.04 -> improved
    after_large = EvalResult(
        task_type="classification",
        primary_metric="f1",
        train_cv_mean={"f1": 0.58, "accuracy": 0.89},
        train_cv_std={"f1": 0.04, "accuracy": 0.02},
        test_scores={"f1": 0.58, "accuracy": 0.89},
    )

    sig2, label2 = is_significant("f1", before, after_large)
    assert sig2 is True
    assert label2 == "improved"


def test_small_dataset_noise_warning():
    """Gate 3: Datasets with fewer than 500 rows produce noisy comparison warning."""
    df_small = pd.DataFrame({
        "feature": range(200),
        "target": [0] * 160 + [1] * 40,
    })
    split = split_dataset(df_small, target="target", random_seed=42)
    res = evaluate_split(split.train_df, split.test_df, target="target", fast=True, random_seed=42)

    assert any("< 500" in w or "fewer than 500" in w for w in res.warnings)


def test_split_order_demo_reveals_leakage(accident_df):
    """
    Gate 3: Planted check: oversampling before split produces higher test recall
    than split-first on the same data.
    """
    demo = run_split_order_demo(accident_df, target="severe_accident", random_seed=42)

    assert demo["leakage_detected"] is True
    flawed_rec = demo["flawed_oversample_before_split"]["recall"]
    honest_rec = demo["honest_split_first"]["recall"]

    # Flawed recall is artificially high (> 0.90) due to cloned instances in test set
    assert flawed_rec > honest_rec
    assert flawed_rec > 0.85
    assert len(demo["explanation"]) == 3


def test_compare_evaluations_summary():
    """compare_evaluations builds structured summary table with noise labels."""
    before = EvalResult(
        task_type="classification",
        primary_metric="f1",
        train_cv_mean={"f1": 0.40, "accuracy": 0.85},
        train_cv_std={"f1": 0.05, "accuracy": 0.02},
        test_scores={"f1": 0.40, "accuracy": 0.85},
    )
    after = EvalResult(
        task_type="classification",
        primary_metric="f1",
        train_cv_mean={"f1": 0.42, "accuracy": 0.86},
        train_cv_std={"f1": 0.05, "accuracy": 0.02},
        test_scores={"f1": 0.42, "accuracy": 0.86},
    )

    summary = compare_evaluations(before, after)
    assert summary.primary_metric == "f1"
    f1_comp = [m for m in summary.metrics if m.metric == "f1"][0]
    assert f1_comp.label == "within noise"
    assert "within CV noise" in summary.verdict


def test_regression_evaluation():
    """Continuous target produces R2, RMSE, and MAE."""
    np.random.seed(42)
    n = 600
    x = np.random.uniform(1, 50, size=n)
    y = x * 3.0 + np.random.normal(0, 2, size=n)
    df = pd.DataFrame({"x": x, "target": y})

    split = split_dataset(df, target="target", task_type="regression", random_seed=42)
    res = evaluate_split(split.train_df, split.test_df, target="target", task_type="regression", fast=True, random_seed=42)

    assert res.task_type == "regression"
    assert "r2" in res.train_cv_mean
    assert "rmse" in res.train_cv_mean
    assert "mae" in res.train_cv_mean
    assert res.train_cv_mean["r2"] > 0.80
