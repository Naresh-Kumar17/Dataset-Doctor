"""
Unit tests for Data Leakage Detection, Advisor, and Fixer (Phase 11 & Phase 12).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from doctor.base import Decision, DetectionContext
from doctor.issues.leakage.detect import LeakageDetector
from doctor.issues.leakage.advise import LeakageAdvisor
from doctor.issues.leakage.fix import LeakageFixer
from doctor.issues.leakage.explain import explain_leakage_issue, format_leakage_badge
from doctor.pipeline import RemediationPipeline

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
LEAKAGE_CSV = ROOT_DIR / "data" / "planted" / "leakage.csv"
CLEAN_CSV = ROOT_DIR / "data" / "planted" / "clean_control.csv"


@pytest.fixture(scope="module")
def leakage_df():
    if not LEAKAGE_CSV.exists():
        from tools.make_planted import make_leakage_planted
        make_leakage_planted()
    return pd.read_csv(LEAKAGE_CSV)


@pytest.fixture(scope="module")
def clean_df():
    if not CLEAN_CSV.exists():
        from tools.make_planted import make_clean_control
        make_clean_control()
    return pd.read_csv(CLEAN_CSV)


def test_leakage_detector_catches_all_planted_leaks(leakage_df):
    detector = LeakageDetector()
    ctx = DetectionContext(target_column="target", task_type="classification")
    issues = detector.detect(leakage_df, ctx=ctx)

    issue_cols = {i.column for i in issues}
    expected_leaks = {"target_copy", "target_scaled", "target_inverted", "id_leak", "payout_amount", "timestamp_col"}
    for exp in expected_leaks:
        assert exp in issue_cols, f"Expected {exp} to be detected as a leakage issue"

    # Verify high severity for actual leaks
    issues_by_col = {i.column: i for i in issues}
    for col in ["target_copy", "target_scaled", "target_inverted", "id_leak", "payout_amount"]:
        assert issues_by_col[col].severity == "high"


def test_leakage_detector_zero_false_alarms_on_clean_control(clean_df):
    detector = LeakageDetector()
    ctx = DetectionContext(target_column="target", task_type="classification")
    issues = detector.detect(clean_df, ctx=ctx)
    assert len(issues) == 0, f"Expected 0 issues on clean control, got: {[i.column for i in issues]}"


def test_leakage_detector_legit_strong_immunity(leakage_df):
    detector = LeakageDetector()
    ctx = DetectionContext(target_column="target", task_type="classification")
    issues = detector.detect(leakage_df, ctx=ctx)
    issues_by_col = {i.column: i for i in issues}

    # legit_strong has r ~ 0.65, must NOT be flagged as high severity leak
    if "legit_strong" in issues_by_col:
        assert issues_by_col["legit_strong"].severity != "high"


def test_leakage_detector_no_target_specified(leakage_df):
    detector = LeakageDetector()
    ctx = DetectionContext(target_column=None)
    # Detects temporal or ID columns even without explicit target, or gracefully returns
    issues = detector.detect(leakage_df, ctx=ctx)
    assert isinstance(issues, list)


def test_leakage_advisor_recommendations(leakage_df):
    detector = LeakageDetector()
    advisor = LeakageAdvisor()
    ctx = DetectionContext(target_column="target", task_type="classification")
    issues = detector.detect(leakage_df, ctx=ctx)

    for iss in issues:
        rec = advisor.advise(iss, leakage_df, target="target", ctx=ctx)
        assert rec is not None
        if iss.evidence.get("subtype") == "time_order_risk":
            assert rec.recommended == "time_split"
        else:
            assert rec.recommended == "drop_column"


def test_leakage_fixer_drops_column(leakage_df):
    fixer = LeakageFixer()
    dec = Decision(
        issue_id="leakage",
        column="target_copy",
        option_id="drop_column",
        params={},
        decided_by="user",
    )
    fitted = fixer.fit(leakage_df, dec)
    cleaned = fixer.transform(leakage_df, dec, fitted, is_test=False)
    assert "target_copy" not in cleaned.columns
    assert "target_copy" in leakage_df.columns


def test_leakage_fixer_preserves_all_rows_and_classes(leakage_df):
    fixer = LeakageFixer()
    n_train = 800
    train_df = leakage_df.iloc[:n_train].copy()
    test_df = leakage_df.iloc[n_train:].copy()

    orig_train_len = len(train_df)
    orig_test_len = len(test_df)
    orig_test_target_dist = test_df["target"].value_counts().to_dict()

    dec = Decision(
        issue_id="leakage",
        column="id_leak",
        option_id="drop_column",
        params={},
        decided_by="user",
    )
    fitted = fixer.fit(train_df, dec)
    train_out = fixer.transform(train_df, dec, fitted, is_test=False)
    test_out = fixer.transform(test_df, dec, fitted, is_test=True)

    # Invariants
    assert len(train_out) == orig_train_len
    assert len(test_out) == orig_test_len
    assert test_out["target"].value_counts().to_dict() == orig_test_target_dist
    assert "id_leak" not in train_out.columns
    assert "id_leak" not in test_out.columns


def test_leakage_pipeline_export_reproduce_code(tmp_path, leakage_df):
    pipeline = RemediationPipeline()
    dec1 = Decision(
        issue_id="leakage",
        column="target_copy",
        option_id="drop_column",
        params={},
        decided_by="advisor",
    )
    dec2 = Decision(
        issue_id="leakage",
        column="id_leak",
        option_id="drop_column",
        params={},
        decided_by="advisor",
    )
    pipeline.record_decision(dec1, train_df=leakage_df)
    pipeline.record_decision(dec2, train_df=leakage_df)

    cleaned = pipeline.transform(leakage_df, is_test=False)
    assert "target_copy" not in cleaned.columns
    assert "id_leak" not in cleaned.columns

    script_path = tmp_path / "reproduce.py"
    data_path = tmp_path / "data.csv"
    leakage_df.to_csv(data_path, index=False)
    pipeline.export_reproduce_file(str(script_path), raw_data_path=str(data_path))

    # Verify exported reproduce script runs cleanly
    import subprocess
    import sys
    res = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True, cwd=str(tmp_path))
    assert res.returncode == 0
    assert (tmp_path / "cleaned_data.csv").exists()


def test_leakage_explain_and_badges(leakage_df):
    detector = LeakageDetector()
    ctx = DetectionContext(target_column="target", task_type="classification")
    issues = detector.detect(leakage_df, ctx=ctx)
    assert len(issues) > 0

    for iss in issues:
        exp = explain_leakage_issue(iss)
        assert isinstance(exp, str) and len(exp) > 0
        badge = format_leakage_badge(iss.evidence)
        assert "badge" in badge or "chip" in badge or "LEAK" in badge or "TIME" in badge


def test_leakage_user_override(leakage_df):
    pipeline = RemediationPipeline()
    dec = Decision(
        issue_id="leakage",
        column="target_copy",
        option_id="keep_with_acknowledgement",
        params={},
        decided_by="user_override",
    )
    pipeline.record_decision(dec, train_df=leakage_df)
    res = pipeline.transform(leakage_df, is_test=False)
    assert "target_copy" in res.columns
    assert len(res) == len(leakage_df)
