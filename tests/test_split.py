"""
Unit tests for doctor/split.py (Phase 3: Train/Test Splitting Engine).
Verifies:
1. Stratified split maintains class ratio within 1% on both train and test partitions.
2. Determinism: Same random seed produces identical split; different seed produces different split.
3. Partition isolation: Zero row overlap between train and test partitions.
4. Target hygiene: Missing target rows are dropped prior to splitting and exact count reported.
5. Inferred tasks: Regression correctly detected and performed without stratification errors.
6. Error handling: Non-existent target column raises clear ValueError.
"""
import pytest
import numpy as np
import pandas as pd
from doctor.split import split_dataset, infer_task_type


@pytest.fixture
def imbalanced_classification_df():
    """Generates a synthetic imbalanced binary classification dataset (85/15 ratio)."""
    np.random.seed(42)
    n_rows = 1000
    n_minority = int(n_rows * 0.15)
    n_majority = n_rows - n_minority
    
    y = np.array([0] * n_majority + [1] * n_minority)
    np.random.shuffle(y)
    
    feature1 = np.random.randn(n_rows)
    feature2 = np.random.choice(["cat_A", "cat_B", "cat_C"], size=n_rows)
    
    return pd.DataFrame({"feat1": feature1, "feat2": feature2, "target": y})


@pytest.fixture
def regression_df():
    """Generates synthetic continuous regression dataset."""
    np.random.seed(42)
    n = 300
    x = np.random.uniform(10, 100, size=n)
    y = x * 2.5 + np.random.normal(0, 5, size=n)
    return pd.DataFrame({"x": x, "price": y})


def test_stratified_split_preserves_class_ratio(imbalanced_classification_df):
    """Gate 3: Stratified split keeps class ratio within 1% (0.01) on both parts."""
    df = imbalanced_classification_df
    split = split_dataset(df, target="target", task_type="classification", test_size=0.2, random_seed=42)
    
    full_ratio_1 = float((df["target"] == 1).mean())
    train_ratio_1 = float((split.train_df["target"] == 1).mean())
    test_ratio_1 = float((split.test_df["target"] == 1).mean())
    
    assert abs(train_ratio_1 - full_ratio_1) < 0.01, f"Train ratio {train_ratio_1} deviated from {full_ratio_1}"
    assert abs(test_ratio_1 - full_ratio_1) < 0.01, f"Test ratio {test_ratio_1} deviated from {full_ratio_1}"
    assert split.minority_share is not None
    assert abs(split.minority_share - 0.15) < 0.01


def test_split_determinism(imbalanced_classification_df):
    """Gate 3: Same seed gives same split. Different seed gives different split."""
    df = imbalanced_classification_df
    split1 = split_dataset(df, target="target", random_seed=42)
    split2 = split_dataset(df, target="target", random_seed=42)
    split3 = split_dataset(df, target="target", random_seed=999)
    
    # Identical seed -> identical row indices
    assert split1.train_indices == split2.train_indices
    assert split1.test_indices == split2.test_indices
    
    # Different seed -> different row indices
    assert split1.train_indices != split3.train_indices


def test_zero_partition_overlap(imbalanced_classification_df):
    """Gate 3: No row appears in both train and test (row index check)."""
    df = imbalanced_classification_df
    split = split_dataset(df, target="target", random_seed=42)
    
    train_idx_set = set(split.train_indices)
    test_idx_set = set(split.test_indices)
    
    assert len(train_idx_set.intersection(test_idx_set)) == 0
    assert len(train_idx_set) + len(test_idx_set) == len(df)


def test_missing_target_rows_dropped_and_reported():
    """Gate 3: Rows with missing target dropped before split and exact count reported."""
    df = pd.DataFrame({
        "feature": range(100),
        "target": [0] * 80 + [1] * 15 + [None, np.nan, "null", "", "?"]
    })
    split = split_dataset(df, target="target", random_seed=42)
    
    assert split.dropped_target_rows == 5
    assert len(split.train_df) + len(split.test_df) == 95
    assert not split.train_df["target"].isna().any()
    assert not split.test_df["target"].isna().any()


def test_regression_split(regression_df):
    """Regression split runs correctly without stratification errors."""
    split = split_dataset(regression_df, target="price", random_seed=42)
    
    assert split.task_type == "regression"
    assert split.train_shape[0] == 240
    assert split.test_shape[0] == 60
    assert len(set(split.train_indices).intersection(set(split.test_indices))) == 0


def test_missing_target_column_raises_error(imbalanced_classification_df):
    """Attempting to split with non-existent target column raises ValueError."""
    with pytest.raises(ValueError, match="not found in dataframe columns"):
        split_dataset(imbalanced_classification_df, target="non_existent_column")
