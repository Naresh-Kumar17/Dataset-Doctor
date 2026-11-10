"""
Dataset Doctor: Unit Tests for Class Imbalance (Phase 9 & 10).
Covers:
- ImbalanceDetector on planted datasets (90/10, 99/1, multiclass, balanced control).
- 0 certain false alarms on clean_control.csv.
- ImbalanceAdvisor rule shortlists and 5-fold Stratified CV measurement.
- ImbalanceFixer test partition preservation invariant (zero test rows resampled/dropped).
- SMOTE unencoded categorical safety guard.
- RemediationPipeline integration and standalone reproduce.py code generation and execution.
- Plain-English explanations and Neo-brutalist badge formatters.
"""
import os
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest

from doctor.base import Decision, DetectionContext, Issue
from doctor.issues.imbalance.detect import ImbalanceDetector
from doctor.issues.imbalance.advise import ImbalanceAdvisor
from doctor.issues.imbalance.fix import ImbalanceFixer, HAS_IMBLEARN
from doctor.issues.imbalance.explain import explain_imbalance_issue, format_imbalance_badge
from doctor.pipeline import RemediationPipeline
from doctor.loader import load_csv


@pytest.fixture
def planted_imbalance_df():
    csv_path = "data/planted/imbalance.csv"
    if not os.path.exists(csv_path):
        from tools.make_planted import make_imbalance_planted
        make_imbalance_planted()
    df, _ = load_csv(csv_path)
    return df


@pytest.fixture
def clean_control_df():
    csv_path = "data/planted/clean_control.csv"
    if not os.path.exists(csv_path):
        from tools.make_planted import make_clean_control
        make_clean_control()
    df, _ = load_csv(csv_path)
    return df


def test_imbalance_detection_planted_targets(planted_imbalance_df):
    detector = ImbalanceDetector()
    issues = detector.detect(planted_imbalance_df)
    issues_by_col = {i.column: i for i in issues}

    # 1. 90/10 target: severe imbalance, high severity
    assert "target_90_10" in issues_by_col
    iss_90 = issues_by_col["target_90_10"]
    assert iss_90.severity == "high"
    assert iss_90.certainty == "certain"
    assert iss_90.evidence["minority_share"] == 0.1
    assert iss_90.evidence["imbalance_ratio"] == 9.0

    # 2. 99/1 target: extreme imbalance, high severity
    assert "target_99_1" in issues_by_col
    iss_99 = issues_by_col["target_99_1"]
    assert iss_99.severity == "high"
    assert iss_99.certainty == "certain"
    assert iss_99.evidence["minority_share"] == 0.01
    assert iss_99.evidence["imbalance_ratio"] == 99.0

    # 3. Multiclass target: high severity
    assert "target_multiclass" in issues_by_col
    iss_multi = issues_by_col["target_multiclass"]
    assert iss_multi.severity == "high"
    assert iss_multi.evidence["is_multiclass"] is True

    # 4. Balanced control (50/50): 0 issues (stays completely silent)
    assert "target_balanced" not in issues_by_col


def test_imbalance_detection_zero_certain_on_clean_control(clean_control_df):
    detector = ImbalanceDetector()
    ctx = DetectionContext(target_column="target", task_type="classification")
    issues = detector.detect(clean_control_df, ctx=ctx)
    certain_issues = [i for i in issues if i.certainty == "certain"]
    assert len(certain_issues) == 0


def test_imbalance_evidence_warnings(planted_imbalance_df):
    detector = ImbalanceDetector()
    ctx = DetectionContext(target_column="target_90_10", task_type="classification")
    issues = detector.detect(planted_imbalance_df, ctx=ctx)
    assert len(issues) == 1
    ev = issues[0].evidence
    assert "accuracy_paradox_warning" in ev
    assert "prompt_real_world" in ev
    assert "90.0%" in ev["accuracy_paradox_warning"] or "90%" in ev["accuracy_paradox_warning"]


def test_imbalance_advisor_evaluation(planted_imbalance_df):
    detector = ImbalanceDetector()
    advisor = ImbalanceAdvisor()
    ctx = DetectionContext(target_column="target_90_10", task_type="classification")
    issues = detector.detect(planted_imbalance_df, ctx=ctx)
    assert len(issues) == 1

    rec = advisor.advise(issues[0], planted_imbalance_df, target="target_90_10", ctx=ctx, measure=True)
    assert rec.recommended in ("class_weight", "random_oversample", "leave")
    assert rec.options is not None
    assert len(rec.options) >= 3

    # Check trade-off measurement metrics
    for opt in rec.options:
        if opt.measured:
            assert "f1_mean" in opt.measured
            assert "recall_mean" in opt.measured
            assert "precision_mean" in opt.measured
            assert "trade_off" in opt.measured


def test_imbalance_fixer_test_set_invariant(planted_imbalance_df):
    fixer = ImbalanceFixer()
    n_train = int(len(planted_imbalance_df) * 0.8)
    train_df = planted_imbalance_df.iloc[:n_train].copy()
    test_df = planted_imbalance_df.iloc[n_train:].copy()

    orig_test_len = len(test_df)
    orig_test_counts = test_df["target_90_10"].value_counts().to_dict()

    dec = Decision(
        issue_id="imbalance",
        column="target_90_10",
        option_id="random_oversample",
        params={},
        decided_by="user",
    )
    fitted = fixer.fit(train_df, dec)

    # Train gets resampled
    train_res = fixer.transform(train_df, dec, fitted, is_test=False)
    assert len(train_res) > len(train_df)

    # Test set is 100% UNCHANGED
    test_res = fixer.transform(test_df, dec, fitted, is_test=True)
    assert len(test_res) == orig_test_len
    assert test_res["target_90_10"].value_counts().to_dict() == orig_test_counts


def test_imbalance_fixer_class_weight_preserves_rows(planted_imbalance_df):
    fixer = ImbalanceFixer()
    dec = Decision(
        issue_id="imbalance",
        column="target_90_10",
        option_id="class_weight",
        params={},
        decided_by="user",
    )
    fitted = fixer.fit(planted_imbalance_df, dec)
    transformed = fixer.transform(planted_imbalance_df, dec, fitted, is_test=False)
    assert len(transformed) == len(planted_imbalance_df)


def test_imbalance_fixer_smote_categorical_guard(planted_imbalance_df):
    fixer = ImbalanceFixer()
    dec = Decision(
        issue_id="imbalance",
        column="target_90_10",
        option_id="smote",
        params={},
        decided_by="user",
    )
    # planted_imbalance_df contains 'feature_cat' which is string object
    with pytest.raises(ValueError) as excinfo:
        fitted = fixer.fit(planted_imbalance_df, dec)
        fixer.transform(planted_imbalance_df, dec, fitted, is_test=False)
    assert "unencoded" in str(excinfo.value).lower() or "smote" in str(excinfo.value).lower()


def test_imbalance_pipeline_integration(planted_imbalance_df):
    pipeline = RemediationPipeline()
    n_train = 800
    train_df = planted_imbalance_df.iloc[:n_train].copy()
    test_df = planted_imbalance_df.iloc[n_train:].copy()

    dec = Decision(
        issue_id="imbalance",
        column="target_90_10",
        option_id="random_oversample",
        params={},
        decided_by="user",
    )
    pipeline.record_decision(dec, train_df=train_df)
    clean_train = pipeline.transform(train_df, is_test=False)
    clean_test = pipeline.transform(test_df, is_test=True)

    assert len(clean_train) > len(train_df)
    assert len(clean_test) == len(test_df)

    log = pipeline.get_decision_log()
    assert len(log) == 1
    assert log[0]["column"] == "target_90_10"
    assert log[0]["option_id"] == "random_oversample"


def test_imbalance_reproduce_code_execution(planted_imbalance_df):
    pipeline = RemediationPipeline()
    dec = Decision(
        issue_id="imbalance",
        column="target_90_10",
        option_id="random_oversample",
        params={},
        decided_by="user",
    )
    pipeline.record_decision(dec, train_df=planted_imbalance_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "raw.csv")
        script_path = os.path.join(tmpdir, "reproduce.py")
        planted_imbalance_df.to_csv(csv_path, index=False)

        pipeline.export_reproduce_file(script_path, raw_data_path=csv_path)

        # Run generated reproduce.py
        res = subprocess.run([sys.executable, script_path], cwd=tmpdir, capture_output=True, text=True)
        assert res.returncode == 0, f"reproduce.py failed: {res.stderr}"
        assert os.path.exists(os.path.join(tmpdir, "cleaned_data.csv"))


def test_imbalance_explanation_and_badges():
    mock_issue = Issue(
        module_id="imbalance",
        column="target_90_10",
        severity="high",
        certainty="certain",
        title="Severe Class Imbalance in 'target_90_10'",
        description="Test desc",
        evidence={
            "minority_share": 0.10,
            "imbalance_ratio": 9.0,
            "minority_class": "1",
            "majority_class": "0",
            "prompt_real_world": "Is 10.0% the true rate in the real world?",
        },
    )

    exp = explain_imbalance_issue(mock_issue)
    assert "accuracy" in exp.lower()
    assert "illusion" in exp.lower()
    assert "recall" in exp.lower()

    badge_high = format_imbalance_badge("high", minority_share=0.10)
    assert "SEVERE IMBALANCE" in badge_high
    assert "10.0%" in badge_high

    badge_med = format_imbalance_badge("medium", minority_share=0.15)
    assert "MODERATE IMBALANCE" in badge_med

    badge_dict = format_imbalance_badge({"minority_share": 0.05})
    assert "SEVERE IMBALANCE" in badge_dict

