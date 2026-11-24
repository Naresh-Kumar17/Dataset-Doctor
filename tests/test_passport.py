"""
Unit tests for Dataset Medical Passport cryptographic PDF generator.
Verifies PDF generation, cryptographic SHA-256 fingerprinting,
and Latin-1 text sanitization.
"""
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from doctor.passport import DatasetMedicalPassport, _sanitize_for_latin1
from doctor.quality import DataQualityAssessor
from doctor.ai_consultant import AIConsultant


def test_sanitize_for_latin1():
    """Verify unicode characters, emojis, and curly quotes are sanitized safely."""
    raw = "Doctor’s opinion: “Clean dataset” • \U0001f6e1 [FAIL] 100% ✓ \u26a1"
    clean = _sanitize_for_latin1(raw)
    assert "Doctor's" in clean
    assert '"Clean dataset"' in clean
    assert "[SHIELD]" in clean
    assert "[PASS]" in clean
    assert "[STRESS]" in clean
    # Must encode cleanly to latin-1 without exception
    encoded = clean.encode("latin-1")
    assert len(encoded) > 0


def test_calculate_sha256_deterministic():
    """Verify SHA-256 calculation is deterministic and 64 characters hex."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    h1 = DatasetMedicalPassport.calculate_sha256(df)
    h2 = DatasetMedicalPassport.calculate_sha256(df)
    assert len(h1) == 64
    assert h1 == h2


def test_generate_pdf_basic():
    """Verify generating a basic medical passport returns valid PDF bytes."""
    df = pd.DataFrame({"age": [25, 40, 65, 30], "target": [0, 1, 1, 0]})
    passport = DatasetMedicalPassport()
    pdf_bytes = passport.generate_pdf(
        df=df,
        filename="patients.csv",
        target_column="target",
        task_type="classification",
    )
    assert isinstance(pdf_bytes, (bytes, bytearray))
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 1000


def test_generate_pdf_with_full_clinical_reports():
    """Verify passport generation integrates QualityAssessment and AIConsultant reports."""
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    assessor = DataQualityAssessor()
    quality = assessor.assess(df, target_column="target")

    consultant = AIConsultant()
    ai_report = consultant.consult(df=df, quality_assessment=quality, target_column="target", provider="offline")

    passport = DatasetMedicalPassport()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "test_passport.pdf"
        pdf_bytes = passport.generate_pdf(
            df=df,
            filename="clinical_trial.csv",
            target_column="target",
            task_type="classification",
            quality_assessment=quality,
            ai_consultation=ai_report,
            output_path=out_file,
        )
        assert out_file.exists()
        assert out_file.stat().st_size > 1000
        assert pdf_bytes.startswith(b"%PDF-")
