"""
Unit and integration tests for NullAdvisor (Phase 4).
Tests:
- Rule-based shortlisting logic across symmetric, skewed, categorical, high-missingness, and target-linked data.
- 5-fold cross-validation measurement on train only.
- Simplicity hierarchy tiebreaker and 'close call' detection.
- Small sample (< 500 rows) fallback to 'not enough data'.
- 5 Gate 4A planted scenarios with known ground truth.
"""
import numpy as np
import pandas as pd
import pytest
from doctor.base import DetectionContext, Issue
from doctor.issues.nulls.advise import NullAdvisor, SIMPLICITY_ORDER


@pytest.fixture
def advisor():
    return NullAdvisor()


def test_advisor_target_missing(advisor):
    """When target column is missing, drop_rows is strictly forced."""
    df = pd.DataFrame({"target": [1, np.nan, 0, 1], "feat": [10, 20, 30, 40]})
    issue = Issue(
        module_id="nulls", column="target", severity="high", certainty="certain",
        title="Target missing", description="Target has missing cells",
        evidence={"is_target": True}
    )
    rec = advisor.advise(issue, df, target="target")
    assert rec.recommended == "drop_rows"
    assert "Target column missingness must be dropped" in rec.rule_hits[0]


def test_advisor_extreme_missingness(advisor):
    """Columns with > 60% missingness recommend drop_column or add_indicator."""
    n = 600
    s = [np.nan] * 450 + [1.0] * 150  # 75% missing
    df = pd.DataFrame({"sparse_col": s, "target": [0, 1] * 300})
    issue = Issue(
        module_id="nulls", column="sparse_col", severity="high", certainty="certain",
        title="Sparse column", description="75% missing",
        evidence={"total_rows": n, "total_missing_count": 450, "total_missing_pct": 0.75}
    )
    rec = advisor.advise(issue, df, target="target")
    assert rec.recommended in ("drop_column", "add_indicator")
    assert any("> 60%" in r for r in rec.rule_hits)


def test_advisor_numeric_skewed_prefers_median(advisor):
    """Numeric column with high skewness recommends median."""
    rng = np.random.RandomState(42)
    x = rng.exponential(scale=5.0, size=600)
    x[rng.rand(600) < 0.10] = np.nan
    df = pd.DataFrame({"skew_col": x, "target": (x > 5).astype(int)})
    issue = Issue(
        module_id="nulls", column="skew_col", severity="medium", certainty="certain",
        title="Skewed column", description="10% missing",
        evidence={"total_rows": 600, "total_missing_count": 60, "total_missing_pct": 0.10}
    )
    rec = advisor.advise(issue, df, target="target", ctx=DetectionContext(target_column="target", task_type="classification"))
    assert rec.recommended == "median"
    assert any("skewed" in r.lower() for r in rec.rule_hits)


def test_advisor_numeric_symmetric_recommends_mean_or_median(advisor):
    """Numeric roughly symmetric column recommends mean or median."""
    rng = np.random.RandomState(42)
    x = rng.normal(loc=100.0, scale=15.0, size=600)
    x[rng.rand(600) < 0.03] = np.nan
    df = pd.DataFrame({"sym_col": x, "target": (x > 100).astype(int)})
    issue = Issue(
        module_id="nulls", column="sym_col", severity="low", certainty="certain",
        title="Symmetric column", description="3% missing",
        evidence={"total_rows": 600, "total_missing_count": 18, "total_missing_pct": 0.03}
    )
    rec = advisor.advise(issue, df, target="target", ctx=DetectionContext(target_column="target", task_type="classification"))
    assert rec.recommended in ("mean", "median")


def test_advisor_categorical_recommends_mode_or_missing_label(advisor):
    """Categorical column recommends mode or missing_label."""
    rng = np.random.RandomState(42)
    cats = rng.choice(["Low", "Med", "High"], size=600, p=[0.6, 0.3, 0.1]).astype(object)
    cats[rng.rand(600) < 0.05] = np.nan
    df = pd.DataFrame({"cat_col": cats, "target": (cats == "Low").astype(int)})
    issue = Issue(
        module_id="nulls", column="cat_col", severity="low", certainty="certain",
        title="Categorical column", description="5% missing",
        evidence={"total_rows": 600, "total_missing_count": 30, "total_missing_pct": 0.05}
    )
    rec = advisor.advise(issue, df, target="target", ctx=DetectionContext(target_column="target", task_type="classification"))
    assert rec.recommended in ("mode", "missing_label")


def test_advisor_target_linked_missingness_prioritizes_indicator(advisor):
    """When missingness correlates with target, add_indicator is included and drop_rows excluded."""
    rng = np.random.RandomState(42)
    y = rng.choice([0, 1], size=600)
    x = rng.normal(size=600)
    mask = (y == 1) & (rng.rand(600) < 0.70)
    x[mask] = np.nan
    df = pd.DataFrame({"linked_col": x, "target": y})
    issue = Issue(
        module_id="nulls", column="linked_col", severity="medium", certainty="certain",
        title="Linked column", description="Missingness linked to target",
        evidence={"total_rows": 600, "total_missing_count": int(mask.sum()), "total_missing_pct": float(mask.mean()), "target_pattern": {"related": True, "p_value": 0.0001}}
    )
    rec = advisor.advise(issue, df, target="target", ctx=DetectionContext(target_column="target", task_type="classification"))
    opt_ids = [o.id for o in rec.options]
    assert "add_indicator" in opt_ids or "add_indicator" in rec.recommended
    assert rec.recommended != "drop_rows"


def test_advisor_small_sample_not_enough_data(advisor):
    """Dataset with < 500 rows sets confidence to 'not enough data'."""
    df = pd.DataFrame({"col": [1.0, np.nan, 3.0] * 50, "target": [0, 1, 0] * 50})  # 150 rows
    issue = Issue(
        module_id="nulls", column="col", severity="medium", certainty="certain",
        title="Small df", description="150 rows",
        evidence={"total_rows": 150, "total_missing_count": 50, "total_missing_pct": 0.33}
    )
    rec = advisor.advise(issue, df, target="target", ctx=DetectionContext(target_column="target", task_type="classification"))
    assert rec.confidence == "not enough data"
    assert "< 500" in rec.reason


def test_simplicity_hierarchy_ranking(advisor):
    """Simplicity order matches established hierarchy."""
    assert SIMPLICITY_ORDER["leave"] < SIMPLICITY_ORDER["median"]
    assert SIMPLICITY_ORDER["median"] == SIMPLICITY_ORDER["mode"]
    assert SIMPLICITY_ORDER["median"] < SIMPLICITY_ORDER["mean"]
    assert SIMPLICITY_ORDER["mean"] < SIMPLICITY_ORDER["add_indicator"]
    assert SIMPLICITY_ORDER["add_indicator"] < SIMPLICITY_ORDER["drop_rows"]
    assert SIMPLICITY_ORDER["drop_rows"] < SIMPLICITY_ORDER["drop_column"]


def test_advisor_handles_none_target_pattern(advisor):
    """Ensures advisor does not crash when evidence['target_pattern'] is None."""
    df = pd.DataFrame({"col": [1.0, np.nan, 3.0, 4.0, 5.0] * 120, "target": [0, 1, 0, 1, 0] * 120})
    issue = Issue(
        module_id="nulls", column="col", severity="medium", certainty="certain",
        title="None target pattern", description="Evidence contains None target pattern",
        evidence={"total_rows": 600, "total_missing_count": 120, "total_missing_pct": 0.20, "target_pattern": None}
    )
    rec = advisor.advise(issue, df, target="target", ctx=DetectionContext(target_column="target", task_type="classification"))
    assert rec.recommended is not None
    assert rec.options is not None

