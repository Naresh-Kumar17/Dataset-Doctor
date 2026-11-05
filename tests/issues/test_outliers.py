"""
Comprehensive Unit Test Suite for Outliers Module (Phases 7 & 8).
Verifies:
- Domain-impossible value detection and severity (Gate 7)
- Unit mix-up detection (Gate 7)
- Robust MAD handling of skewed continuous features (Gate 7)
- Categorical and binary column immunity (Gate 7)
- Clean control immunity (Gate 7)
- Rule-based and CV-based recommendations (Gate 8)
- Minority class extreme protection (Gate 8)
- Train-derived clipping bounds and test partition row preservation (Gate 8)
- Standalone reproduce.py code generation and execution (Gate 8)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from doctor.base import Decision, DetectionContext, Issue
from doctor.issues.outliers.detect import OutlierDetector, compute_mad, compute_skewness
from doctor.issues.outliers.advise import OutlierAdvisor
from doctor.issues.outliers.fix import OutlierFixer
from doctor.issues.outliers.explain import explain_outliers_issue, format_outliers_badge
from doctor.pipeline import RemediationPipeline


@pytest.fixture
def sample_outliers_df():
    rng = np.random.default_rng(42)
    n = 200
    age = rng.integers(25, 65, size=n).astype(float)
    age[5] = -5.0
    age[12] = 145.0

    height = rng.normal(170.0, 5.0, size=n)
    height[20] = 170000.0  # 1000x jump

    salary = rng.normal(50000.0, 5000.0, size=n)
    salary[30] = 250000.0
    salary[31] = 280000.0

    cat_col = rng.choice(["A", "B", "C"], size=n)
    bin_col = rng.choice([0, 1], size=n)
    target = rng.choice([0, 1], p=[0.9, 0.1], size=n)

    return pd.DataFrame({
        "age": age,
        "height_cm": height,
        "salary": salary,
        "department": cat_col,
        "is_active": bin_col,
        "target": target,
    })


def test_detect_impossible_age_values(sample_outliers_df):
    detector = OutlierDetector()
    issues = detector.detect(sample_outliers_df)
    age_issue = next((i for i in issues if i.column == "age"), None)

    assert age_issue is not None
    assert age_issue.severity == "high"
    assert age_issue.certainty == "certain"
    assert age_issue.evidence["subtype"] == "impossible_values"
    assert age_issue.evidence["count"] == 2
    assert -5.0 in age_issue.evidence["sample_values"]
    assert 145.0 in age_issue.evidence["sample_values"]


def test_detect_unit_mixup(sample_outliers_df):
    detector = OutlierDetector()
    issues = detector.detect(sample_outliers_df)
    height_issue = next((i for i in issues if i.column == "height_cm" and i.evidence["subtype"] == "unit_mixup"), None)

    assert height_issue is not None
    assert height_issue.severity == "medium"
    assert height_issue.certainty == "possible"
    assert height_issue.evidence["count"] == 1
    assert 170000.0 in height_issue.evidence["sample_values"]


def test_categorical_and_binary_columns_ignored(sample_outliers_df):
    detector = OutlierDetector()
    issues = detector.detect(sample_outliers_df)
    flagged_cols = [i.column for i in issues]

    assert "department" not in flagged_cols
    assert "is_active" not in flagged_cols
    assert "target" not in flagged_cols


def test_clean_control_zero_false_alarms():
    clean_csv = Path(__file__).resolve().parent.parent.parent / "data" / "planted" / "clean_control.csv"
    if clean_csv.exists():
        df_clean = pd.read_csv(clean_csv)
        detector = OutlierDetector()
        issues = detector.detect(df_clean, ctx=DetectionContext(target_column="target"))
        assert len(issues) == 0


def test_skewed_column_mad_guard():
    rng = np.random.default_rng(42)
    clean_skewed = rng.lognormal(mean=9.0, sigma=0.8, size=500)
    df = pd.DataFrame({"income": clean_skewed, "target": rng.choice([0, 1], size=500)})

    detector = OutlierDetector()
    issues = detector.detect(df)
    income_issue = next((i for i in issues if i.column == "income"), None)

    if income_issue:
        flag_pct = income_issue.evidence["count"] / len(df)
        assert flag_pct <= 0.05
        assert income_issue.evidence["is_skewed"] is True


def test_advisor_impossible_recommends_set_to_null(sample_outliers_df):
    detector = OutlierDetector()
    advisor = OutlierAdvisor()
    issues = detector.detect(sample_outliers_df)
    age_issue = next((i for i in issues if i.column == "age"), None)

    rec = advisor.advise(age_issue, sample_outliers_df, target="target", measure=False)
    assert rec.recommended == "set_to_null"
    assert rec.confidence == "clear"


def test_advisor_minority_extreme_protection():
    rng = np.random.default_rng(42)
    n = 300
    salary = rng.normal(50000.0, 5000.0, size=n)
    target = np.zeros(n, dtype=int)
    # Rare target (10%)
    minority_idx = rng.choice(n, size=30, replace=False)
    target[minority_idx] = 1

    # Plant extremes aligned with minority class
    extremes_idx = list(minority_idx[:8])
    for idx in extremes_idx:
        salary[idx] = 300000.0

    df = pd.DataFrame({"salary": salary, "target": target})
    detector = OutlierDetector()
    advisor = OutlierAdvisor()

    issues = detector.detect(df, ctx=DetectionContext(target_column="target"))
    sal_issue = next((i for i in issues if i.column == "salary"), None)

    assert sal_issue is not None
    rec = advisor.advise(sal_issue, df, target="target", measure=False)
    assert rec.recommended == "leave"
    assert "minority" in rec.reason.lower() or "rare" in rec.reason.lower()


def test_fixer_clip_bounds_learned_from_train():
    train_df = pd.DataFrame({"val": [10.0, 12.0, 14.0, 15.0, 16.0, 18.0, 20.0, 100.0]})
    test_df = pd.DataFrame({"val": [11.0, 13.0, 15.0, 250.0]})

    fixer = OutlierFixer()
    dec = Decision(issue_id="outliers", column="val", option_id="clip_iqr", params={}, decided_by="user")
    fitted = fixer.fit(train_df, dec)

    train_trans = fixer.transform(train_df, dec, fitted, is_test=False)
    test_trans = fixer.transform(test_df, dec, fitted, is_test=True)

    learned_upper = fitted["upper"]
    assert train_trans["val"].max() <= learned_upper
    assert test_trans["val"].max() <= learned_upper
    assert len(test_trans) == len(test_df)


def test_fixer_test_set_never_drops_rows():
    train_df = pd.DataFrame({"val": [10.0, 12.0, 14.0, 15.0, 16.0, 500.0]})
    test_df = pd.DataFrame({"val": [11.0, 13.0, 500.0]})

    fixer = OutlierFixer()
    dec = Decision(issue_id="outliers", column="val", option_id="drop_rows", params={}, decided_by="user")
    fitted = fixer.fit(train_df, dec)

    train_trans = fixer.transform(train_df, dec, fitted, is_test=False)
    test_trans = fixer.transform(test_df, dec, fitted, is_test=True)

    assert len(train_trans) < len(train_df)
    assert len(test_trans) == len(test_df)  # Strict rule: 0 test rows dropped!


def test_fixer_set_to_null():
    df = pd.DataFrame({"age": [25.0, -10.0, 45.0, 150.0, 30.0]})
    fixer = OutlierFixer()
    dec = Decision(issue_id="outliers", column="age", option_id="set_to_null", params={}, decided_by="user")
    fitted = fixer.fit(df, dec)
    trans = fixer.transform(df, dec, fitted, is_test=False)

    assert pd.isna(trans["age"].iloc[1])
    assert pd.isna(trans["age"].iloc[3])
    assert trans["age"].iloc[0] == 25.0


def test_pipeline_outliers_and_reproduce_code():
    train_df = pd.DataFrame({
        "age": [25.0, -5.0, 40.0, 130.0, 50.0],
        "salary": [40000.0, 45000.0, 50000.0, 55000.0, 300000.0],
    })

    pipeline = RemediationPipeline()
    dec1 = Decision(issue_id="outliers", column="age", option_id="set_to_null", params={}, decided_by="user")
    dec2 = Decision(issue_id="outliers", column="salary", option_id="clip_iqr", params={}, decided_by="user")

    pipeline.record_decision(dec1, train_df)
    pipeline.record_decision(dec2, train_df)

    clean_pipeline = pipeline.transform(train_df, is_test=False)
    code = pipeline.export_reproduce_code()

    scope = {"pd": pd, "np": np}
    exec(code, scope)
    clean_func = scope["clean_dataset"]
    clean_reproduced = clean_func(train_df, is_test=False)

    pd.testing.assert_frame_equal(clean_pipeline, clean_reproduced)


def test_explain_and_format_badges():
    issue_imp = Issue(
        module_id="outliers",
        column="age",
        severity="high",
        certainty="certain",
        title="Impossible Age",
        description="",
        evidence={"subtype": "impossible_values", "count": 10, "pct": 0.02, "sample_values": [-5, 145]},
    )
    badge_imp = format_outliers_badge(issue_imp)
    exp_imp = explain_outliers_issue(issue_imp)

    assert "10 IMPOSSIBLE" in badge_imp
    assert "Domain-Impossible" in exp_imp
    assert "set_to_null" in exp_imp
