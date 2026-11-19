"""
Unit tests for Experiment Manifest & Bit-for-Bit Reproducibility (Phase 19).
"""
import json
from pathlib import Path
import pytest
import pandas as pd

from doctor.api import DatasetDoctor
from doctor.manifest import ManifestGenerator, compute_dataframe_hash
from doctor.export import export_standalone_script


@pytest.fixture
def clean_control_csv():
    root = Path(__file__).resolve().parent.parent
    return str(root / "data" / "planted" / "clean_control.csv")


def test_dataframe_hash_determinism():
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df3 = pd.DataFrame({"a": [1, 2, 4], "b": ["x", "y", "z"]})

    h1 = compute_dataframe_hash(df1)
    h2 = compute_dataframe_hash(df2)
    h3 = compute_dataframe_hash(df3)

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # Valid SHA-256 hex string


def test_manifest_generation(clean_control_csv):
    doc = DatasetDoctor.from_csv(clean_control_csv, target="target")
    train_df, test_df = doc.split(test_size=0.2, random_state=42)
    eval_res = doc.evaluate(fast_mode=True)
    pipeline = doc.auto_remediate()

    clean_train = pipeline.transform(train_df, is_test=False)
    clean_test = pipeline.transform(test_df, is_test=True)

    gen = ManifestGenerator(manifest_version="1.0.0")
    manifest = gen.generate(
        raw_df=doc.df,
        train_df=train_df,
        test_df=test_df,
        clean_train_df=clean_train,
        clean_test_df=clean_test,
        target_column="target",
        pipeline=pipeline,
        eval_result=eval_res,
        dataset_name="clean_control.csv",
    )

    assert manifest.manifest_version == "1.0.0"
    assert manifest.raw_rows == 500
    assert manifest.raw_cols == 6
    assert len(manifest.clean_train_hash) == 64
    assert len(manifest.clean_test_hash) == 64
    assert manifest.split_provenance.train_rows == 400
    assert manifest.split_provenance.test_rows == 100
    assert "accuracy" in manifest.evaluation_metrics

    # Test serialization
    json_str = manifest.to_json()
    parsed = json.loads(json_str)
    assert parsed["dataset_name"] == "clean_control.csv"
    assert parsed["split_provenance"]["strategy"] == "stratified"


def test_bit_for_bit_verification(clean_control_csv):
    doc = DatasetDoctor.from_csv(clean_control_csv, target="target")
    train_df, test_df = doc.split(test_size=0.2, random_state=42)
    pipeline = doc.auto_remediate()

    clean_train = pipeline.transform(train_df, is_test=False)
    clean_test = pipeline.transform(test_df, is_test=True)

    gen = ManifestGenerator()
    manifest = gen.generate(
        raw_df=doc.df,
        train_df=train_df,
        test_df=test_df,
        clean_train_df=clean_train,
        clean_test_df=clean_test,
        target_column="target",
        pipeline=pipeline,
    )

    # Identical DataFrames should pass verification
    is_valid, mismatches = gen.verify_reproducibility(manifest, clean_train, clean_test)
    assert is_valid is True
    assert len(mismatches) == 0

    # Tampered DataFrame must fail verification
    tampered_train = clean_train.copy()
    tampered_train.iloc[0, 0] = "XX"
    is_valid_tampered, mismatches_tampered = gen.verify_reproducibility(manifest, tampered_train, clean_test)
    assert is_valid_tampered is False
    assert len(mismatches_tampered) > 0


def test_standalone_script_export(clean_control_csv, tmp_path):
    doc = DatasetDoctor.from_csv(clean_control_csv, target="target")
    pipeline = doc.auto_remediate()

    script = export_standalone_script(pipeline, raw_csv_path=clean_control_csv)
    assert "def clean_dataset" in script
    assert "def compute_sha256" in script
    assert "pd.read_csv" in script

    # Verify script executes standalone
    exec_scope = {}
    exec(script, exec_scope)
    assert "clean_dataset" in exec_scope

    clean_func = exec_scope["clean_dataset"]
    reproduced_df = clean_func(doc.df, is_test=False)
    expected_clean_df = pipeline.transform(doc.df, is_test=False)

    pd.testing.assert_frame_equal(reproduced_df, expected_clean_df)

