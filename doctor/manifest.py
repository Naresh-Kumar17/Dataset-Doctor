"""
Dataset Doctor: Experiment Manifest & Data Provenance Engine (Phase 19).
Captures machine-readable cryptographic provenance and guarantees bit-for-bit reproducibility:
- Raw dataset SHA-256 hash
- Stratified partition indices hash
- Sequential remediation decision log with learned parameters
- Clean transformed dataset SHA-256 hash
- Model hyperparameters and evaluation metrics
"""
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import sklearn


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """
    Computes a deterministic SHA-256 hash of a pandas DataFrame's contents,
    index, columns, and data values.
    """
    # Use pandas util hash_pandas_object or deterministic CSV representation
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def compute_bytes_hash(data: bytes) -> str:
    """Computes a SHA-256 hash of raw bytes."""
    return hashlib.sha256(data).hexdigest()


@dataclass
class DecisionRecord:
    step_id: int
    issue_id: str
    column: Optional[str]
    option_id: str
    params: Dict[str, Any]
    learned_parameters: Dict[str, Any]
    decided_by: str


@dataclass
class SplitProvenance:
    strategy: str
    test_size: float
    random_state: int
    train_rows: int
    test_rows: int
    train_index_hash: str
    test_index_hash: str
    target_column: Optional[str] = None
    class_ratio_train: Optional[Dict[str, float]] = None
    class_ratio_test: Optional[Dict[str, float]] = None


@dataclass
class ExperimentManifest:
    manifest_version: str
    created_at: str
    dataset_name: str
    raw_dataset_hash: str
    raw_rows: int
    raw_cols: int
    columns: List[str]
    split_provenance: SplitProvenance
    decision_log: List[Dict[str, Any]]
    clean_train_hash: str
    clean_test_hash: str
    model_name: str
    evaluation_metrics: Dict[str, float]
    train_cv_mean: Dict[str, float]
    system_environment: Dict[str, str]

    @property
    def decisions(self) -> List[Dict[str, Any]]:
        return self.decision_log

    def to_json(self, indent: int = 2) -> str:
        """Serializes manifest to standard JSON string."""
        data = asdict(self)
        return json.dumps(data, indent=indent)

    def save(self, path: Union[str, Path]) -> None:
        """Saves manifest to a file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


class ManifestGenerator:
    """
    Generates and verifies bit-for-bit experiment manifests.
    """
    def __init__(self, manifest_version: str = "1.0.0"):
        self.manifest_version = manifest_version

    def generate(
        self,
        raw_df: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        clean_train_df: pd.DataFrame,
        clean_test_df: pd.DataFrame,
        target_column: Optional[str],
        pipeline: Any,
        eval_result: Optional[Any] = None,
        dataset_name: str = "dataset.csv",
        random_state: int = 42,
        test_size: float = 0.2,
    ) -> ExperimentManifest:
        """
        Builds a comprehensive ExperimentManifest capturing full data provenance.
        """
        raw_hash = compute_dataframe_hash(raw_df)
        train_idx_bytes = np.array(list(train_df.index)).tobytes()
        test_idx_bytes = np.array(list(test_df.index)).tobytes()
        train_idx_hash = hashlib.sha256(train_idx_bytes).hexdigest()
        test_idx_hash = hashlib.sha256(test_idx_bytes).hexdigest()

        # Class distributions
        train_ratios = None
        test_ratios = None
        if target_column and target_column in train_df.columns:
            train_ratios = {str(k): float(v) for k, v in train_df[target_column].value_counts(normalize=True).items()}
            test_ratios = {str(k): float(v) for k, v in test_df[target_column].value_counts(normalize=True).items()}

        split_prov = SplitProvenance(
            strategy="stratified" if train_ratios and len(train_ratios) <= 10 else "random",
            test_size=test_size,
            random_state=random_state,
            train_rows=len(train_df),
            test_rows=len(test_df),
            train_index_hash=train_idx_hash,
            test_index_hash=test_idx_hash,
            target_column=target_column,
            class_ratio_train=train_ratios,
            class_ratio_test=test_ratios,
        )

        # Decision log serialization
        decision_log: List[Dict[str, Any]] = []
        if hasattr(pipeline, "steps"):
            for idx, step in enumerate(pipeline.steps):
                decision_log.append({
                    "step_id": idx + 1,
                    "issue_id": step.issue_id,
                    "column": step.column,
                    "option_id": step.option_id,
                    "params": getattr(step, "fitted_params", getattr(step, "params", {})),
                    "decided_by": getattr(step, "decided_by", "system"),
                })
        elif hasattr(pipeline, "decisions"):
            for idx, dec in enumerate(pipeline.decisions):
                decision_log.append({
                    "step_id": idx + 1,
                    "issue_id": dec.issue_id,
                    "column": dec.column,
                    "option_id": dec.option_id,
                    "params": getattr(dec, "params", {}),
                    "decided_by": getattr(dec, "decided_by", "system"),
                })

        clean_tr_hash = compute_dataframe_hash(clean_train_df)
        clean_te_hash = compute_dataframe_hash(clean_test_df)

        metrics = eval_result.test_scores if eval_result else {}
        cv_mean = eval_result.train_cv_mean if eval_result else {}

        env = {
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
            "platform": platform.platform(),
        }

        return ExperimentManifest(
            manifest_version=self.manifest_version,
            created_at=datetime.now().isoformat(),
            dataset_name=dataset_name,
            raw_dataset_hash=raw_hash,
            raw_rows=len(raw_df),
            raw_cols=len(raw_df.columns),
            columns=list(raw_df.columns),
            split_provenance=split_prov,
            decision_log=decision_log,
            clean_train_hash=clean_tr_hash,
            clean_test_hash=clean_te_hash,
            model_name="RandomForestClassifier",
            evaluation_metrics=metrics,
            train_cv_mean=cv_mean,
            system_environment=env,
        )

    def verify_reproducibility(
        self,
        manifest: ExperimentManifest,
        reproduced_train_df: pd.DataFrame,
        reproduced_test_df: pd.DataFrame,
    ) -> Tuple[bool, List[str]]:
        """
        Verifies that reproduced DataFrames match the manifest bit-for-bit.
        """
        mismatches: List[str] = []

        rep_tr_hash = compute_dataframe_hash(reproduced_train_df)
        if rep_tr_hash != manifest.clean_train_hash:
            mismatches.append(
                f"Train DataFrame hash mismatch: expected {manifest.clean_train_hash[:12]}..., got {rep_tr_hash[:12]}..."
            )

        rep_te_hash = compute_dataframe_hash(reproduced_test_df)
        if rep_te_hash != manifest.clean_test_hash:
            mismatches.append(
                f"Test DataFrame hash mismatch: expected {manifest.clean_test_hash[:12]}..., got {rep_te_hash[:12]}..."
            )

        if len(reproduced_train_df) != manifest.split_provenance.train_rows:
            mismatches.append(
                f"Train rows mismatch: expected {manifest.split_provenance.train_rows}, got {len(reproduced_train_df)}"
            )

        if len(reproduced_test_df) != manifest.split_provenance.test_rows:
            mismatches.append(
                f"Test rows mismatch: expected {manifest.split_provenance.test_rows}, got {len(reproduced_test_df)}"
            )

        is_reproducible = len(mismatches) == 0
        return is_reproducible, mismatches

