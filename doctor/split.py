"""
Dataset Doctor: Central Train/Test Splitting Engine.
Rules:
- split.py is the ONLY place that splits data.
- Drops rows with missing target values first and reports count.
- Classification uses stratified splitting (keeping class ratios within 1%).
- Regression uses plain random splitting.
- Fixed seed ensures deterministic splits; different seeds produce different splits.
- Preserves and returns original row indices for locking, audit, and GUI display.
- Verifies zero overlap between train and test row partitions.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataSplit:
    """Encapsulates the locked partition of a dataset into train and test sets."""
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_indices: List[Any]
    test_indices: List[Any]
    target_column: str
    task_type: Literal["classification", "regression"]
    dropped_target_rows: int
    random_seed: int
    test_size: float = 0.2
    minority_share: Optional[float] = None
    class_distribution_train: Optional[Dict[Any, float]] = None
    class_distribution_test: Optional[Dict[Any, float]] = None

    @property
    def train_shape(self) -> Tuple[int, int]:
        return self.train_df.shape

    @property
    def test_shape(self) -> Tuple[int, int]:
        return self.test_df.shape


def infer_task_type(series: pd.Series) -> Literal["classification", "regression"]:
    """Infers whether the target represents classification or regression."""
    s_clean = series.dropna()
    if s_clean.empty:
        return "classification"
    
    unique_count = s_clean.nunique()
    if unique_count <= 10:
        return "classification"
    
    # Check if numeric
    if pd.api.types.is_numeric_dtype(s_clean):
        # If float or high unique integers, regression
        return "regression"
    
    return "classification"


def split_dataset(
    df: pd.DataFrame,
    target: str,
    task_type: Optional[str] = None,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> DataSplit:
    """
    Splits a DataFrame into train and test partitions.
    
    Parameters:
    - df: Full input DataFrame
    - target: Name of the target column
    - task_type: "classification" or "regression" (auto-inferred if None)
    - test_size: Fraction of data reserved for test set (default 0.2 = 80/20)
    - random_seed: Random seed for deterministic reproducibility
    
    Returns:
    - DataSplit dataclass containing train_df, test_df, indices, and distributions.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe columns: {list(df.columns)}")
    
    # 1. Drop rows with missing target values
    target_series = df[target]
    missing_target_mask = target_series.isna()
    if target_series.dtype == "object":
        missing_target_mask = missing_target_mask | target_series.astype(str).str.strip().isin(
            ["", "nan", "NaN", "null", "None", "N/A", "NA", "?"]
        )
    
    dropped_target_rows = int(missing_target_mask.sum())
    clean_df = df[~missing_target_mask].copy()
    
    if len(clean_df) < 5:
        raise ValueError(
            f"Dataset has only {len(clean_df)} valid rows after dropping missing targets. "
            "Minimum 5 rows required for train/test split."
        )
    
    # 2. Determine task type
    inferred_task = infer_task_type(clean_df[target]) if not task_type else task_type.lower()
    if inferred_task not in ("classification", "regression"):
        raise ValueError(f"Invalid task_type '{inferred_task}'. Must be 'classification' or 'regression'.")
    
    # 3. Perform Split
    minority_share = None
    dist_train = None
    dist_test = None
    
    if inferred_task == "classification":
        class_counts = clean_df[target].value_counts()
        total_valid = len(clean_df)
        proportions = (class_counts / total_valid).to_dict()
        minority_share = float(class_counts.min() / total_valid)
        
        # Check if any class has fewer than 2 samples (stratify requires at least 2 per class)
        can_stratify = int(class_counts.min()) >= 2 and len(class_counts) > 1
        stratify_arr = clean_df[target] if can_stratify else None
        
        train_df, test_df = train_test_split(
            clean_df,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify_arr,
        )
        
        # Compute distributions
        dist_train = {k: round(v, 4) for k, v in train_df[target].value_counts(normalize=True).items()}
        dist_test = {k: round(v, 4) for k, v in test_df[target].value_counts(normalize=True).items()}
        
    else:  # regression
        train_df, test_df = train_test_split(
            clean_df,
            test_size=test_size,
            random_state=random_seed,
            shuffle=True,
        )
    
    # 4. Extract indices and verify separation
    train_indices = train_df.index.tolist()
    test_indices = test_df.index.tolist()
    
    # Strict safety assertion: zero row overlap between train and test
    overlap = set(train_indices).intersection(set(test_indices))
    if overlap:
        raise RuntimeError(f"Split error: {len(overlap)} rows present in both train and test partitions.")
    
    return DataSplit(
        train_df=train_df,
        test_df=test_df,
        train_indices=train_indices,
        test_indices=test_indices,
        target_column=target,
        task_type=inferred_task,
        dropped_target_rows=dropped_target_rows,
        random_seed=random_seed,
        test_size=test_size,
        minority_share=minority_share,
        class_distribution_train=dist_train,
        class_distribution_test=dist_test,
    )
