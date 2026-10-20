"""
Dataset Doctor: Phase 2 Null Detection Unit and Integration Tests.
Verifies all Gate 2 criteria:
- 100% of planted missing cells found (true NaNs, hidden tokens, sentinels).
- 0 false alarms on clean_control.csv (country 'NA' flagged at most as possible).
- Row loss number is exact.
- Planted target-related missingness is reported; random missingness is not.
- Target column missingness flagged as high severity.
- Performance on 100,000 rows < 5.0 seconds.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pytest

from doctor.base import DetectionContext
from doctor.issues.nulls.detect import NullDetector
from doctor.issues.nulls.explain import explain_null_issue, format_evidence_badge
from doctor.loader import load_csv

PLANTED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "planted"
NULLS_CSV = PLANTED_DIR / "nulls.csv"
CLEAN_CSV = PLANTED_DIR / "clean_control.csv"


@pytest.fixture(scope="module", autouse=True)
def ensure_planted_datasets():
    """Ensures planted ground truth datasets exist before tests execute."""
    if not NULLS_CSV.exists() or not CLEAN_CSV.exists():
        from tools.make_planted import make_clean_control, make_nulls_planted
        make_clean_control()
        make_nulls_planted()


def test_true_nan_and_empty_cells_detected():
    df = pd.DataFrame({
        "vals": [1.0, np.nan, 3.0, np.nan, 5.0],
        "notes": ["ok", "", "done", "   ", "yes"],
    })
    detector = NullDetector()
    issues = detector.detect(df)
    issues_dict = {iss.column: iss for iss in issues}

    assert "vals" in issues_dict
    assert issues_dict["vals"].evidence["nan_count"] == 2
    assert issues_dict["vals"].evidence["total_missing_pct"] == 40.0
    assert issues_dict["vals"].certainty == "certain"

    assert "notes" in issues_dict
    assert issues_dict["notes"].evidence["nan_count"] == 2
    assert issues_dict["notes"].evidence["total_missing_pct"] == 40.0
    assert issues_dict["notes"].certainty == "certain"


def test_hidden_tokens_case_insensitive():
    df = pd.DataFrame({
        "code": ["valid", "?", "-", "N/A", "n/a", "NULL", "none", "nan", "good"]
    })
    detector = NullDetector()
    issues = detector.detect(df)
    assert len(issues) == 1
    issue = issues[0]
    hidden = issue.evidence["hidden_token_counts"]

    assert hidden.get("?") == 1
    assert hidden.get("-") == 1
    assert hidden.get("N/A") == 2
    assert hidden.get("null") == 1
    assert hidden.get("none") == 1
    assert hidden.get("nan") == 1
    assert issue.certainty == "certain"


def test_sentinels_flagged_strictly_as_possible():
    df = pd.DataFrame({
        "score": [100, 200, -999, 400, 9999],
        "duration": [10, 20, 30, -1, 50],
    })
    detector = NullDetector()
    issues = detector.detect(df)
    issues_dict = {iss.column: iss for iss in issues}

    assert "score" in issues_dict
    assert issues_dict["score"].certainty == "possible"
    sent_score = issues_dict["score"].evidence["sentinel_counts"]
    assert sent_score.get("-999") == 1
    assert sent_score.get("9999") == 1

    assert "duration" in issues_dict
    assert issues_dict["duration"].certainty == "possible"
    sent_dur = issues_dict["duration"].evidence["sentinel_counts"]
    assert sent_dur.get("-1") == 1


def test_gate_planted_nulls_found_100_percent():
    df, _ = load_csv(str(NULLS_CSV), filename="nulls.csv")
    detector = NullDetector()
    issues = detector.detect(df, ctx=DetectionContext(target_column="target", task_type="classification"))
    by_col = {iss.column: iss for iss in issues}

    # age: 50 NaNs
    assert "age" in by_col
    assert by_col["age"].evidence["nan_count"] == 50

    # phone: 60 N/A
    assert "phone" in by_col
    assert by_col["phone"].evidence["hidden_token_counts"].get("N/A") == 60

    # score: 40 -999 sentinels
    assert "score" in by_col
    assert by_col["score"].evidence["sentinel_counts"].get("-999") == 40
    assert by_col["score"].certainty == "possible"

    # notes: 700 NaNs (>60% -> high severity)
    assert "notes" in by_col
    assert by_col["notes"].evidence["nan_count"] == 700
    assert by_col["notes"].severity == "high"


def test_clean_control_zero_false_alarms():
    df_clean, _ = load_csv(str(CLEAN_CSV), filename="clean_control.csv")
    detector = NullDetector()
    issues = detector.detect(df_clean, ctx=DetectionContext(target_column="target", task_type="classification"))

    certain_issues = [iss for iss in issues if iss.certainty == "certain"]
    assert len(certain_issues) == 0, f"Expected 0 certain issues on clean control, got: {certain_issues}"

    country_issues = [iss for iss in issues if iss.column == "country"]
    for iss in country_issues:
        assert iss.certainty == "possible", "Country 'NA' (Namibia) must be flagged at most as possible"


def test_exact_row_loss_matches_clean_pipeline():
    df, _ = load_csv(str(NULLS_CSV), filename="nulls.csv")
    detector = NullDetector()
    issues = detector.detect(df, ctx=DetectionContext(target_column="target", task_type="classification"))

    reported_loss = issues[0].evidence["row_loss_impact"]["rows_lost"]

    # Calculate exact rows lost when dropping true missing and confirmed sentinels
    expected_loss = int(pd.DataFrame({
        c: (df[c].isna() | df[c].isin(["N/A", "-999", -999, ""]))
        for c in ["age", "phone", "score", "notes", "income", "noise_col"]
    }).any(axis=1).sum())

    assert reported_loss == expected_loss
    assert reported_loss == 796


def test_planted_target_pattern_reported():
    df, _ = load_csv(str(NULLS_CSV), filename="nulls.csv")
    detector = NullDetector()
    issues = detector.detect(df, ctx=DetectionContext(target_column="target", task_type="classification"))
    by_col = {iss.column: iss for iss in issues}

    # Planted target-related missingness in income
    assert "income" in by_col
    pattern_income = by_col["income"].evidence.get("target_pattern")
    assert pattern_income is not None
    assert pattern_income["related"] is True
    assert pattern_income["p_value"] < 0.05

    # Planted random missingness in noise_col (MCAR)
    assert "noise_col" in by_col
    pattern_noise = by_col["noise_col"].evidence.get("target_pattern")
    assert pattern_noise is not None
    assert pattern_noise["related"] is False
    assert pattern_noise["p_value"] >= 0.05


def test_target_column_missingness_high_severity():
    df = pd.DataFrame({
        "feat": [1, 2, 3, 4, 5],
        "my_target": [0, np.nan, 1, 0, 1],  # 20% missing, but is target!
    })
    detector = NullDetector()
    issues = detector.detect(df, ctx=DetectionContext(target_column="my_target", task_type="classification"))
    by_col = {iss.column: iss for iss in issues}

    assert "my_target" in by_col
    assert by_col["my_target"].severity == "high"
    assert by_col["my_target"].evidence["is_target"] is True


def test_performance_under_5_seconds_on_100k_rows():
    np.random.seed(42)
    n = 100000
    df_large = pd.DataFrame({
        "num1": np.random.choice([1.0, 2.0, np.nan], size=n),
        "cat1": np.random.choice(["active", "N/A", "inactive"], size=n),
        "sent1": np.random.choice([10, -999, 20], size=n),
        "clean_num": np.random.normal(100, 15, size=n),
        "target": np.random.choice([0, 1], size=n),
    })

    detector = NullDetector()
    t0 = time.perf_counter()
    issues = detector.detect(df_large, ctx=DetectionContext(target_column="target", task_type="classification"))
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, f"Detection took {elapsed:.2f}s, expected < 5.0s"
    assert len(issues) >= 3


def test_explain_functions():
    evidence = {
        "total_rows": 1000,
        "total_missing_count": 120,
        "total_missing_pct": 12.0,
        "nan_count": 100,
        "hidden_token_counts": {"N/A": 20},
        "target_pattern": {"related": False, "p_value": 0.61},
    }
    explanation = explain_null_issue(evidence, column="income")
    assert "120 of 1000 rows missing (12.0%)" in explanation
    assert "Hidden tokens: 'N/A': 20" in explanation
    assert "Missing is NOT linked to target (p = 0.61)" in explanation

    badge = format_evidence_badge(evidence)
    assert "100 NaN" in badge
    assert "20 hidden" in badge
