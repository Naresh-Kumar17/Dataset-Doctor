"""
Unit tests for AI Diagnostic Consultant & Second Opinion Agent.
"""
from pathlib import Path
import pandas as pd
import pytest

from doctor.ai_consultant import AIConsultant, AIConsultationReport
from doctor.quality import DataQualityAssessor
from doctor.profile import profile_dataframe


@pytest.fixture
def leakage_df():
    csv_p = Path(__file__).resolve().parent.parent / "data" / "planted" / "leakage.csv"
    return pd.read_csv(csv_p)


@pytest.fixture
def clean_df():
    csv_p = Path(__file__).resolve().parent.parent / "data" / "planted" / "clean_control.csv"
    return pd.read_csv(csv_p)


def test_ai_consultant_report_dataclass():
    report = AIConsultationReport(
        clinical_impression="Test impression",
        patient_status="GUARDED",
        survival_forecast="Moderate risk",
        semantic_leakage_review=[{"column": "leak", "risk_level": "HIGH", "rationale": "test"}],
        fairness_bias_analysis="Balanced",
        prescriptions=["Drop leak"],
        discharge_letter_markdown="# Medical Summary",
        provider="Offline Expert",
    )
    d = report.to_dict()
    assert d["patient_status"] == "GUARDED"
    assert len(d["semantic_leakage_review"]) == 1
    assert "Medical Summary" in d["discharge_letter_markdown"]
    assert "timestamp" in d


def test_ai_consultant_offline_leakage_dataset(leakage_df):
    assessor = DataQualityAssessor()
    quality = assessor.assess(leakage_df, target_column="target", task_type="classification")
    profile = profile_dataframe(leakage_df)

    consultant = AIConsultant()
    report = consultant.consult(
        df=leakage_df,
        profile=profile,
        quality_assessment=quality,
        target_column="target",
        provider="offline",
    )

    assert report.patient_status in ("CRITICAL", "GUARDED")
    assert report.provider == "Built-In Expert Diagnostic Synthesizer (Offline)"
    assert len(report.prescriptions) > 0
    assert "CLINICAL CONSULTATION & DISCHARGE SUMMARY" in report.discharge_letter_markdown

    # Verify semantic leakage detection caught suspicious columns
    flagged_cols = [r["column"] for r in report.semantic_leakage_review]
    assert any("target" in col for col in flagged_cols)
    assert any("payout" in col or "copy" in col for col in flagged_cols)


def test_ai_consultant_offline_clean_dataset(clean_df):
    assessor = DataQualityAssessor()
    quality = assessor.assess(clean_df, target_column="target", task_type="classification")
    profile = profile_dataframe(clean_df)

    consultant = AIConsultant()
    report = consultant.consult(
        df=clean_df,
        profile=profile,
        quality_assessment=quality,
        target_column="target",
        provider="offline",
    )

    assert report.patient_status in ("STABLE", "EXCELLENT")
    assert "CLINICAL CONSULTATION" in report.discharge_letter_markdown


def test_ai_consultant_fallback_on_invalid_key(clean_df):
    """When an invalid Gemini or OpenAI key is passed, consultant gracefully falls back to offline engine."""
    consultant = AIConsultant()
    report = consultant.consult(
        df=clean_df,
        target_column="target",
        provider="gemini",
        api_key_override="invalid-test-key",
    )
    # Must not raise an unhandled exception; must gracefully return offline report
    assert report is not None
    assert report.provider == "Built-In Expert Diagnostic Synthesizer (Offline)"
