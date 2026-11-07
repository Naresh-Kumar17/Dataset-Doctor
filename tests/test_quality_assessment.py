"""
Unit tests for Phase 14: Unified Data-to-Model Quality Assessment.
"""
import pandas as pd
from doctor.base import Decision
from doctor.quality import DataQualityAssessor


def test_quality_assessment_clean_dataset():
    df = pd.DataFrame({
        "age": [25, 30, 45, 50, 35, 40],
        "score": [10.5, 20.0, 15.2, 18.0, 22.1, 19.4],
        "target": [0, 1, 0, 1, 0, 1],
    })
    assessor = DataQualityAssessor()
    report = assessor.assess(df, target_column="target", task_type="classification")

    assert report.overall_score == 100
    assert report.readiness == "TRAINING_READY"
    can_train, msg = report.can_train_model(allow_override=False)
    assert can_train is True
    assert len(report.critical_blockers) == 0
    assert set(report.domain_summaries.keys()) == {
        "nulls", "duplicates", "types", "outliers", "imbalance", "leakage"
    }


def test_quality_assessment_blocks_on_leakage():
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "target": [0, 1, 0, 1, 0, 1],
        "target_copy": [0, 1, 0, 1, 0, 1],  # Direct target leakage
    })
    assessor = DataQualityAssessor()
    report = assessor.assess(df, target_column="target", task_type="classification")

    assert report.readiness == "BLOCKED_CRITICAL"
    assert len(report.critical_blockers) > 0
    assert report.overall_score < 100

    # Training blocked without override
    can_train, msg = report.can_train_model(allow_override=False)
    assert can_train is False
    assert "BLOCKED" in msg

    # Training permitted with explicit override
    can_override, override_msg = report.can_train_model(allow_override=True)
    assert can_override is True
    assert "override" in override_msg.lower()


def test_quality_assessment_remediations_clear_blocker():
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "target": [0, 1, 0, 1, 0, 1],
        "target_copy": [0, 1, 0, 1, 0, 1],
    })
    assessor = DataQualityAssessor()
    decisions = [
        Decision(issue_id="leakage", column="target_copy", option_id="drop_column", params={}, decided_by="user")
    ]
    report = assessor.assess(df, target_column="target", task_type="classification", decisions=decisions)

    assert report.readiness == "TRAINING_READY"
    assert len(report.critical_blockers) == 0
    assert report.overall_score == 100
    assert report.total_issues_resolved >= 1
