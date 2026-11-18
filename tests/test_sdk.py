"""
Unit tests for Dataset Doctor Python Core SDK (Phase 18).
"""
from pathlib import Path
import pytest
import pandas as pd

from doctor import DatasetDoctor, DatasetLoadConfig


@pytest.fixture
def clean_control_csv():
    root = Path(__file__).resolve().parent.parent
    return str(root / "data" / "planted" / "clean_control.csv")


@pytest.fixture
def leakage_csv():
    root = Path(__file__).resolve().parent.parent
    return str(root / "data" / "planted" / "leakage.csv")


def test_sdk_from_csv_and_profile(clean_control_csv):
    doc = DatasetDoctor.from_csv(clean_control_csv, target="target")
    assert doc.df is not None
    assert len(doc.df) == 500
    assert doc.target == "target"
    assert doc.task_type == "classification"

    profile = doc.profile()
    assert profile.n_rows == 500
    assert profile.n_cols == 6
    assert profile.target_column == "target"


def test_sdk_audit_quality(clean_control_csv, leakage_csv):
    # Clean control should score 100/100 and allow training
    doc_clean = DatasetDoctor.from_csv(clean_control_csv, target="target")
    audit_clean = doc_clean.audit()
    assert audit_clean.quality_score == 100
    assert audit_clean.can_train is True
    assert audit_clean.readiness == "TRAINING_READY"

    # Leakage file should detect direct target copy and block training
    doc_leak = DatasetDoctor.from_csv(leakage_csv, target="target")
    audit_leak = doc_leak.audit()
    assert audit_leak.can_train is False
    assert audit_leak.readiness == "BLOCKED_CRITICAL"
    assert len(audit_leak.blocker_reasons) > 0


def test_sdk_split_and_evaluate(clean_control_csv):
    doc = DatasetDoctor.from_csv(clean_control_csv, target="target")
    train_df, test_df = doc.split(test_size=0.2, random_state=42)

    assert len(train_df) == 400
    assert len(test_df) == 100
    # Zero overlap check
    assert set(train_df.index).isdisjoint(set(test_df.index))

    # Evaluate baseline
    eval_res = doc.evaluate(fast_mode=True)
    assert "accuracy" in eval_res.test_scores
    assert "f1" in eval_res.test_scores
    assert "f1" in eval_res.train_cv_mean


def test_sdk_check_leakage(leakage_csv, clean_control_csv):
    doc_leak = DatasetDoctor.from_csv(leakage_csv, target="target")
    leaks = doc_leak.check_leakage()
    assert len(leaks) >= 4
    high_leaks = [l for l in leaks if l.severity == "high"]
    assert len(high_leaks) >= 3

    doc_clean = DatasetDoctor.from_csv(clean_control_csv, target="target")
    clean_leaks = doc_clean.check_leakage()
    assert len(clean_leaks) == 0


def test_sdk_explainability(clean_control_csv):
    doc = DatasetDoctor.from_csv(clean_control_csv, target="target")
    attr = doc.explain(n_repeats=2, random_state=42)
    assert len(attr.features) > 0
    assert attr.test_data_snooped is False
    assert "PREDICTIVE IMPORTANCE" in attr.non_causal_disclosure

