"""
Unit tests for Phase 5: Duplicates Detection, Advice, and Fixer.
Verifies Gate 5 specifications:
- Exact duplicates detection & resolution
- Near duplicates (casing, whitespace, precision fuzz) flagged as possible
- Conflicting duplicate rows (identical features, different target)
- Duplicate unique IDs
- Synthetic pre-balancing test leakage detection
- Zero false alarms on clean control
- Test partition isolation (no test row loss)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from doctor.base import Decision, DetectionContext, Issue
from doctor.issues.duplicates.detect import DuplicateDetector
from doctor.issues.duplicates.advise import DuplicateAdvisor
from doctor.issues.duplicates.fix import DuplicateFixer
from doctor.pipeline import RemediationPipeline

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "id": range(1, 101),
        "name": [f"Person_{i}" for i in range(1, 101)],
        "score": np.random.RandomState(42).randint(50, 100, size=100),
        "target": np.random.RandomState(42).choice([0, 1], p=[0.8, 0.2], size=100),
    })


def test_detect_exact_duplicates(clean_df):
    detector = DuplicateDetector()
    # Duplicate 5 rows exactly
    dups = clean_df.iloc[:5].copy()
    dirty_df = pd.concat([clean_df, dups], ignore_index=True)

    issues = detector.detect(dirty_df)
    exact_issue = next((i for i in issues if i.evidence.get("kind") == "exact"), None)

    assert exact_issue is not None
    assert exact_issue.certainty == "certain"
    assert exact_issue.evidence["surplus_copies"] == 5
    assert exact_issue.evidence["total_involved_rows"] == 10


def test_detect_conflicting_duplicates():
    detector = DuplicateDetector()
    # 2 rows with identical features but different targets
    df = pd.DataFrame({
        "feature_a": [10, 20, 10],
        "feature_b": ["red", "blue", "red"],
        "target": [0, 1, 1],  # Row 0 and Row 2 have identical features but target 0 vs 1!
    })
    ctx = DetectionContext(target_column="target", task_type="classification")
    issues = detector.detect(df, ctx=ctx)

    conf_issue = next((i for i in issues if i.evidence.get("kind") == "conflicting"), None)
    assert conf_issue is not None
    assert conf_issue.severity == "high"
    assert conf_issue.certainty == "certain"
    assert conf_issue.evidence["conflicting_rows_count"] == 2
    assert conf_issue.evidence["conflicting_pairs_count"] == 1


def test_detect_near_duplicates():
    detector = DuplicateDetector()
    df = pd.DataFrame({
        "name": ["Alice", "alice ", "Bob", "bob"],
        "city": ["New York", "new york", "Paris", "Paris"],
        "amount": [100.0, 100.00001, 50.0, 50.0],
    })
    issues = detector.detect(df)
    near_issue = next((i for i in issues if i.evidence.get("kind") == "near"), None)

    assert near_issue is not None
    assert near_issue.certainty == "possible"
    assert near_issue.evidence["near_duplicate_count"] == 2


def test_detect_duplicate_ids():
    detector = DuplicateDetector()
    df = pd.DataFrame({
        "user_id": [101, 102, 103, 101, 105],
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    issues = detector.detect(df)
    id_issue = next((i for i in issues if i.evidence.get("kind") == "id"), None)

    assert id_issue is not None
    assert id_issue.severity == "high"
    assert id_issue.column == "user_id"
    assert id_issue.evidence["duplicate_id_count"] == 1


def test_pre_balanced_warning_detection():
    detector = DuplicateDetector()
    # Build 50/50 dataset with repeated minority rows
    base_minority = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [10, 20, 30], "target": [1, 1, 1]})
    # Oversample minority by repeating rows 10 times (30 rows total)
    minority_oversampled = pd.concat([base_minority] * 10, ignore_index=True)
    # Majority has 30 unique rows
    majority = pd.DataFrame({"feat1": range(100, 130), "feat2": range(200, 230), "target": [0] * 30})

    prebalanced_df = pd.concat([majority, minority_oversampled], ignore_index=True)
    ctx = DetectionContext(target_column="target", task_type="classification")
    issues = detector.detect(prebalanced_df, ctx=ctx)

    pb_issue = next((i for i in issues if i.evidence.get("kind") == "pre_balanced"), None)
    assert pb_issue is not None
    assert pb_issue.severity == "high"
    assert pb_issue.certainty == "possible"


def test_clean_control_zero_false_alarms():
    clean_csv = ROOT_DIR / "data" / "planted" / "clean_control.csv"
    if not clean_csv.exists():
        from tools.make_planted import make_clean_control
        make_clean_control()

    df_clean = pd.read_csv(clean_csv)
    detector = DuplicateDetector()
    issues = detector.detect(df_clean, ctx=DetectionContext(target_column="target", task_type="classification"))

    # Must find 0 exact, 0 conflicting, and 0 pre-balanced issues on clean control
    dup_kinds = [i.evidence.get("kind") for i in issues if i.evidence.get("kind") in ("exact", "conflicting", "pre_balanced")]
    assert len(dup_kinds) == 0


def test_fixer_keep_first_and_keep_last():
    fixer = DuplicateFixer()
    df = pd.DataFrame({
        "col": ["A", "B", "A"],
        "val": [1, 2, 3],
    })
    # Drop duplicates on "col" or whole row
    df_row_dups = pd.DataFrame({
        "col": ["A", "B", "A"],
        "val": [1, 2, 1],
    }, index=[10, 20, 30])

    dec_first = Decision(issue_id="duplicates", column=None, option_id="keep_first", params={}, decided_by="user")
    fitted_first = fixer.fit(df_row_dups, dec_first)
    res_first = fixer.transform(df_row_dups, fitted_first, is_test=False)
    assert len(res_first) == 2
    assert list(res_first.index) == [10, 20]

    dec_last = Decision(issue_id="duplicates", column=None, option_id="keep_last", params={}, decided_by="user")
    fitted_last = fixer.fit(df_row_dups, dec_last)
    res_last = fixer.transform(df_row_dups, fitted_last, is_test=False)
    assert len(res_last) == 2
    assert list(res_last.index) == [20, 30]


def test_fixer_drop_all_conflicting():
    fixer = DuplicateFixer()
    df = pd.DataFrame({
        "feat": [1, 2, 1, 3],
        "target": [0, 1, 1, 0],
    })
    # Rows 0 and 2 have identical feature '1' but different target 0 vs 1
    dec = Decision(
        issue_id="duplicates",
        column="target",
        option_id="drop_all_conflicting",
        params={"target_column": "target"},
        decided_by="user",
    )
    fitted = fixer.fit(df, dec)
    res = fixer.transform(df, fitted, is_test=False)

    assert len(res) == 2
    assert list(res["feat"].values) == [2, 3]


def test_fixer_test_set_isolation():
    fixer = DuplicateFixer()
    df_test = pd.DataFrame({
        "col": ["A", "A", "B"],
        "target": [0, 0, 1],
    })
    dec = Decision(issue_id="duplicates", column=None, option_id="keep_first", params={}, decided_by="user")
    fitted = fixer.fit(df_test, dec)
    res_test = fixer.transform(df_test, fitted, is_test=True)

    # Test set rows are strictly untouched!
    assert len(res_test) == 3


def test_advisor_recommendations():
    advisor = DuplicateAdvisor()
    df = pd.DataFrame({"a": [1, 1], "b": [2, 2]})

    exact_issue = Issue(
        module_id="duplicates", column=None, severity="medium", certainty="certain",
        title="Exact duplicates", description="", evidence={"kind": "exact", "surplus_copies": 1}
    )
    rec = advisor.advise(exact_issue, df, target=None)
    assert rec.recommended == "keep_first"

    conf_issue = Issue(
        module_id="duplicates", column="target", severity="high", certainty="certain",
        title="Conflicting duplicates", description="", evidence={"kind": "conflicting", "target_column": "target"}
    )
    rec_conf = advisor.advise(conf_issue, df, target="target")
    assert rec_conf.recommended == "drop_all_conflicting"


def test_pipeline_integration():
    pipeline = RemediationPipeline()
    train_data = pd.DataFrame({
        "feat": [10, 20, 10, 30],
        "target": [1, 0, 1, 0],
    })
    dec = Decision(issue_id="duplicates", column=None, option_id="keep_first", params={}, decided_by="user")
    pipeline.record_decision(dec, train_data)

    cleaned = pipeline.transform(train_data, is_test=False)
    assert len(cleaned) == 3

    reproduce_code = pipeline.export_reproduce_code()
    assert "drop_duplicates" in reproduce_code

    exec_scope = {"pd": pd, "np": np}
    exec(reproduce_code, exec_scope)
    clean_func = exec_scope.get("clean_dataset")
    reproduced_df = clean_func(train_data, is_test=False)
    assert len(reproduced_df) == 3
