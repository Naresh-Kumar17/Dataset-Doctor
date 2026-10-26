"""
Unit and integration tests for NullFixer and RemediationPipeline (Gate 4B).
Tests:
- Zero nulls remaining in train and test after imputation.
- Test row count strictly preserved (no test rows dropped).
- Test cell mutations isolated only to previously missing cells.
- Learned values derived strictly from train partition (no data leakage).
- Persistent decision log and reproducible script export.
- User overrides changing remediation method.
"""
import numpy as np
import pandas as pd
import pytest
from doctor.base import Decision
from doctor.issues.nulls.fix import NullFixer
from doctor.pipeline import RemediationPipeline


@pytest.fixture
def fixer():
    return NullFixer()


def test_fixer_median_imputation_zero_nulls(fixer):
    """Median imputation leaves 0 nulls in train and test."""
    train = pd.DataFrame({"age": [20.0, 30.0, np.nan, 40.0]})
    test = pd.DataFrame({"age": [25.0, np.nan, 35.0]})

    dec = Decision(issue_id="nulls", column="age", option_id="median", params={}, decided_by="recommended_accepted")
    fitted = fixer.fit(train, dec)
    train_fixed = fixer.transform(train, fitted, is_test=False)
    test_fixed = fixer.transform(test, fitted, is_test=True)

    assert train_fixed["age"].isna().sum() == 0
    assert test_fixed["age"].isna().sum() == 0
    assert len(test_fixed) == len(test)
    assert test_fixed["age"].iloc[1] == 30.0  # train median: [20, 30, 40] -> 30.0


def test_fixer_test_rows_never_dropped(fixer):
    """When drop_rows is selected, rows are dropped from train only; test rows are preserved."""
    train = pd.DataFrame({"feat": [1.0, np.nan, 3.0, 4.0]})
    test = pd.DataFrame({"feat": [10.0, np.nan, 30.0]})

    dec = Decision(issue_id="nulls", column="feat", option_id="drop_rows", params={}, decided_by="user")
    fitted = fixer.fit(train, dec)

    train_fixed = fixer.transform(train, fitted, is_test=False)
    test_fixed = fixer.transform(test, fitted, is_test=True)

    assert len(train_fixed) == 3
    assert len(test_fixed) == 3  # Test rows preserved 100%
    assert test_fixed["feat"].isna().sum() == 0  # Imputed with train fallback


def test_fixer_learned_strictly_from_train_no_leakage(fixer):
    """Learned parameters reflect train distribution only, not combined dataset."""
    train = pd.DataFrame({"num": [10.0, 20.0, 30.0, np.nan]})
    test = pd.DataFrame({"num": [500.0, 500.0, np.nan]})

    dec = Decision(issue_id="nulls", column="num", option_id="median", params={}, decided_by="user")
    fitted = fixer.fit(train, dec)
    test_fixed = fixer.transform(test, fitted, is_test=True)

    # Train median is 20.0. If full data leaked, median would be 30.0.
    assert test_fixed["num"].iloc[2] == 20.0


def test_fixer_non_null_cells_unaltered(fixer):
    """Only cells that were originally null are modified."""
    train = pd.DataFrame({"val": [100.0, np.nan, 300.0]})
    test = pd.DataFrame({"val": [150.0, 250.0, np.nan]})

    dec = Decision(issue_id="nulls", column="val", option_id="mean", params={}, decided_by="user")
    fitted = fixer.fit(train, dec)
    test_fixed = fixer.transform(test, fitted, is_test=True)

    assert test_fixed["val"].iloc[0] == 150.0
    assert test_fixed["val"].iloc[1] == 250.0
    assert test_fixed["val"].iloc[2] == 200.0  # (100 + 300) / 2 = 200.0


def test_fixer_add_indicator(fixer):
    """add_indicator appends binary indicator column and imputes original feature."""
    train = pd.DataFrame({"income": [50000.0, np.nan, 70000.0]})
    test = pd.DataFrame({"income": [np.nan, 80000.0]})

    dec = Decision(issue_id="nulls", column="income", option_id="add_indicator", params={}, decided_by="user")
    fitted = fixer.fit(train, dec)
    test_fixed = fixer.transform(test, fitted, is_test=True)

    assert "income_was_missing" in test_fixed.columns
    assert test_fixed["income_was_missing"].tolist() == [1, 0]
    assert test_fixed["income"].isna().sum() == 0


def test_remediation_pipeline_and_reproduce_export():
    """Pipeline logs choices and exports runnable reproduce.py."""
    pipeline = RemediationPipeline()
    train = pd.DataFrame({"x": [1.0, 2.0, np.nan], "cat": ["A", np.nan, "B"]})

    d1 = Decision(issue_id="nulls", column="x", option_id="median", params={}, decided_by="recommended_accepted")
    d2 = Decision(issue_id="nulls", column="cat", option_id="mode", params={}, decided_by="user")

    pipeline.record_decision(d1, train)
    pipeline.record_decision(d2, train)

    log = pipeline.get_decision_log()
    assert len(log) == 2
    assert log[0]["column"] == "x"
    assert log[1]["option_id"] == "mode"

    code = pipeline.export_reproduce_code()
    assert "def clean_dataset(df" in code
    assert "x" in code
    assert "cat" in code

    # Execute exported script in sandbox
    scope = {"pd": pd, "np": np}
    exec(code, scope)
    clean_fn = scope["clean_dataset"]
    cleaned = clean_fn(train, is_test=False)
    assert cleaned["x"].isna().sum() == 0
    assert cleaned["cat"].isna().sum() == 0


def test_fixer_normalizes_string_tokens_and_sentinels(fixer):
    """Hidden tokens like '?', 'N/A', empty strings, and sentinels are converted and imputed."""
    from doctor.issues.nulls.detect import NullDetector
    from doctor.base import DetectionContext

    df = pd.DataFrame({
        "status": ["valid", "?", "valid", "", "N/A"],
        "reading": [10.0, -999.0, 30.0, 40.0, -999.0],
    })

    dec_status = Decision(issue_id="nulls", column="status", option_id="mode", params={}, decided_by="recommended_accepted")
    dec_reading = Decision(issue_id="nulls", column="reading", option_id="median", params={"tokens_to_null": [-999.0]}, decided_by="recommended_accepted")

    fitted_s = fixer.fit(df, dec_status)
    df_step1 = fixer.transform(df, fitted_s, is_test=False)

    fitted_r = fixer.fit(df_step1, dec_reading)
    df_step2 = fixer.transform(df_step1, fitted_r, is_test=False)

    # All '?', '', 'N/A' must be replaced with 'valid'
    assert (df_step2["status"] == "valid").all()
    # All -999.0 sentinels must be replaced with median (30.0)
    assert not (df_step2["reading"] == -999.0).any()
    assert (df_step2["reading"] == 30.0).sum() == 3

    # Verification: NullDetector finds 0 issues on cleaned dataframe
    detector = NullDetector()
    issues = detector.detect(df_step2, ctx=DetectionContext(target_column=None, task_type="unsupervised"))
    assert len(issues) == 0

