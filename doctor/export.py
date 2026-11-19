"""
Dataset Doctor: Standalone Script & Experiment Export Engine (Phase 19).
Exports standalone Python reproduction scripts with hardcoded learned parameters
and zero reliance on the `doctor` package.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from doctor.manifest import ExperimentManifest


def export_standalone_script(
    pipeline: Any,
    manifest: Optional[ExperimentManifest] = None,
    raw_csv_path: str = "dataset.csv",
    train_csv_out: str = "train_clean.csv",
    test_csv_out: str = "test_clean.csv",
    target_column: Optional[str] = "target",
) -> str:
    """
    Generates a production-ready, 100% standalone reproduce.py script.
    Executes in pure Python with only pandas, numpy, and scikit-learn.
    """
    raw_csv_path_clean = str(raw_csv_path).replace("\\", "/")
    train_csv_out_clean = str(train_csv_out).replace("\\", "/")
    test_csv_out_clean = str(test_csv_out).replace("\\", "/")
    pipeline_code = pipeline.export_reproduce_code() if hasattr(pipeline, "export_reproduce_code") else ""

    manifest_metadata = ""
    if manifest:
        manifest_metadata = f"""
# =====================================================================
# EXPERIMENT MANIFEST & CRYPTOGRAPHIC PROVENANCE
# Manifest Version:    {manifest.manifest_version}
# Raw Dataset SHA-256: {manifest.raw_dataset_hash}
# Clean Train SHA-256: {manifest.clean_train_hash}
# Clean Test SHA-256:  {manifest.clean_test_hash}
# Target Column:       {manifest.split_provenance.target_column}
# Train Rows:          {manifest.split_provenance.train_rows}
# Test Rows:           {manifest.split_provenance.test_rows}
# =====================================================================
"""

    script = f'''"""
Standalone Dataset Doctor Reproduction Script.
Auto-generated for bit-for-bit deterministic reproducibility.
Zero external doctor package dependencies required.
"""
import sys
import hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

{manifest_metadata}

{pipeline_code}


def compute_sha256(df: pd.DataFrame) -> str:
    """Computes exact SHA-256 hash of DataFrame."""
    return hashlib.sha256(df.to_csv(index=True).encode("utf-8")).hexdigest()


def main():
    print("Dataset Doctor: Executing Standalone Reproduction...")
    raw_path = "{raw_csv_path_clean}"
    
    try:
        raw_df = pd.read_csv(raw_path, keep_default_na=False)
    except Exception as e:
        print(f"Error loading {{raw_path}}: {{e}}")
        sys.exit(1)

    print(f"Loaded raw dataset: {{len(raw_df)}} rows x {{len(raw_df.columns)}} columns")

    # Apply deterministic cleaning
    clean_df = clean_dataset(raw_df, is_test=False)
    print(f"Transformed dataset: {{len(clean_df)}} rows x {{len(clean_df.columns)}} columns")

    # Save clean dataset
    clean_df.to_csv("{train_csv_out_clean}", index=False)
    print("Clean dataset saved successfully to {train_csv_out_clean}")
    print(f"Clean DataFrame SHA-256: {{compute_sha256(clean_df)}}")


if __name__ == "__main__":
    main()
'''
    return script

