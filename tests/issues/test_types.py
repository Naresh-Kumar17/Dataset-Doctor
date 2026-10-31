"""
Dataset Doctor: Data Types Module Unit Tests (Phase 6).
Tests:
- Detection of numbers formatted as text (currencies, percentages, unit strings).
- Protection of categorical codes with leading zeros (e.g. '001', '007') from numeric casting.
- Ambiguous date format detection (both parts <= 12 flagged as possible).
- Unambiguous ISO and disambiguated date detection (certainty=certain).
- Boolean representation recognition (yes/no, true/false).
- Constant zero-variance column detection (severity=high).
- ID-like identifier detection (>95% uniqueness).
- 0 false alarms on clean_control.csv.
- Advisor recommendations and user disambiguation prompts.
- Fixer transformations (numeric conversion, date parsing, boolean encoding, column dropping).
- Test set safety (is_test=True never drops rows).
- Standalone reproduce.py code export and reproduction fidelity.
"""
import json
import re
import numpy as np
import pandas as pd
import pytest

from doctor.base import Decision, DetectionContext, Issue
from doctor.issues.types.detect import TypeDetector
from doctor.issues.types.advise import TypeAdvisor
from doctor.issues.types.fix import TypeFixer
from doctor.pipeline import RemediationPipeline


@pytest.fixture
def sample_types_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "price_usd": ["$1,200.50", "$45.00", "$150.00", "$89.99", "$2,100.00"],
        "discount_pct": ["15%", "5%", "20%", "10%", "50%"],
        "weight_kg": ["12 kg", "4.5 kg", "25 kg", "1.2 kg", "8.0 kg"],
        "ambiguous_event_date": ["03/04/2021", "05/06/2022", "01/02/2020", "07/08/2021", "10/11/2022"],
        "iso_signup_date": ["2021-04-15", "2022-08-20", "2023-11-28", "2020-01-05", "2022-06-12"],
        "subscribed_flag": ["yes", "no", "yes", "no", "yes"],
        "constant_status": ["ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE"],
        "product_code": ["001", "002", "007", "015", "099"],
        "target": [0, 1, 0, 0, 1],
    })


def test_detect_text_numbers(sample_types_df):
    detector = TypeDetector()
    issues = detector.detect(sample_types_df, ctx=DetectionContext(target_column="target"))
    issues_by_col = {i.column: i for i in issues if i.column}

    for col in ["price_usd", "discount_pct", "weight_kg"]:
        assert col in issues_by_col, f"Expected text number issue for '{col}'"
        iss = issues_by_col[col]
        assert iss.evidence.get("kind") == "text_number"
        assert iss.certainty == "certain"
        assert iss.severity == "medium"


def test_leading_zero_codes_protected(sample_types_df):
    detector = TypeDetector()
    issues = detector.detect(sample_types_df, ctx=DetectionContext(target_column="target"))
    issues_by_col = {i.column: i for i in issues if i.column}

    # product_code has values '001', '002', '007'
    # Must NOT be detected as text_number!
    if "product_code" in issues_by_col:
        assert issues_by_col["product_code"].evidence.get("kind") != "text_number"


def test_detect_ambiguous_date(sample_types_df):
    detector = TypeDetector()
    issues = detector.detect(sample_types_df, ctx=DetectionContext(target_column="target"))
    issues_by_col = {i.column: i for i in issues if i.column}

    assert "ambiguous_event_date" in issues_by_col
    iss = issues_by_col["ambiguous_event_date"]
    assert iss.evidence.get("kind") == "ambiguous_date"
    assert iss.certainty == "possible"  # Explicitly possible / user decision required
    assert iss.evidence.get("requires_user_disambiguation") is True


def test_detect_unambiguous_date(sample_types_df):
    detector = TypeDetector()
    issues = detector.detect(sample_types_df, ctx=DetectionContext(target_column="target"))
    issues_by_col = {i.column: i for i in issues if i.column}

    assert "iso_signup_date" in issues_by_col
    iss = issues_by_col["iso_signup_date"]
    assert iss.evidence.get("kind") == "date"
    assert iss.certainty == "certain"


def test_detect_boolean(sample_types_df):
    detector = TypeDetector()
    issues = detector.detect(sample_types_df, ctx=DetectionContext(target_column="target"))
    issues_by_col = {i.column: i for i in issues if i.column}

    assert "subscribed_flag" in issues_by_col
    iss = issues_by_col["subscribed_flag"]
    assert iss.evidence.get("kind") == "boolean"
    assert iss.certainty == "certain"


def test_detect_constant_column(sample_types_df):
    detector = TypeDetector()
    issues = detector.detect(sample_types_df, ctx=DetectionContext(target_column="target"))
    issues_by_col = {i.column: i for i in issues if i.column}

    assert "constant_status" in issues_by_col
    iss = issues_by_col["constant_status"]
    assert iss.evidence.get("kind") == "constant"
    assert iss.severity == "high"
    assert iss.certainty == "certain"


def test_detect_id_like():
    detector = TypeDetector()
    df = pd.DataFrame({
        "customer_id": [f"ID_{i:04d}" for i in range(100)],
        "val": range(100),
    })
    issues = detector.detect(df)
    issues_by_col = {i.column: i for i in issues if i.column}

    assert "customer_id" in issues_by_col
    iss = issues_by_col["customer_id"]
    assert iss.evidence.get("kind") == "id_like"


def test_clean_control_zero_false_alarms():
    from pathlib import Path
    clean_csv = Path(__file__).resolve().parent.parent.parent / "data" / "planted" / "clean_control.csv"
    if clean_csv.exists():
        df_clean = pd.read_csv(clean_csv)
        detector = TypeDetector()
        issues = detector.detect(df_clean, ctx=DetectionContext(target_column="target"))
        defects = [
            i for i in issues
            if i.evidence.get("kind") in ("text_number", "ambiguous_date", "constant", "mixed_type")
        ]
        assert len(defects) == 0, f"Expected 0 false alarms on control, got {defects}"


def test_advisor_recommendations(sample_types_df):
    detector = TypeDetector()
    advisor = TypeAdvisor()
    issues = detector.detect(sample_types_df, ctx=DetectionContext(target_column="target"))
    issues_by_col = {i.column: i for i in issues if i.column}

    # Text number recommends convert_to_numeric
    rec_num = advisor.advise(issues_by_col["price_usd"], sample_types_df, target="target")
    assert rec_num.recommended == "convert_to_numeric"
    assert rec_num.confidence == "clear"

    # Ambiguous date requires user prompt, close call
    rec_ambig = advisor.advise(issues_by_col["ambiguous_event_date"], sample_types_df, target="target")
    assert rec_ambig.confidence == "close call"
    option_ids = [o.id for o in rec_ambig.options]
    assert "parse_date_day_first" in option_ids
    assert "parse_date_month_first" in option_ids

    # Boolean recommends encode_boolean
    rec_bool = advisor.advise(issues_by_col["subscribed_flag"], sample_types_df, target="target")
    assert rec_bool.recommended == "encode_boolean"

    # Constant recommends drop_column
    rec_const = advisor.advise(issues_by_col["constant_status"], sample_types_df, target="target")
    assert rec_const.recommended == "drop_column"


def test_fixer_convert_to_numeric(sample_types_df):
    fixer = TypeFixer()
    dec = Decision(issue_id="types", column="price_usd", option_id="convert_to_numeric", params={}, decided_by="user")
    fitted = fixer.fit(sample_types_df, dec)
    fixed = fixer.transform(sample_types_df, fitted, is_test=False)

    assert pd.api.types.is_float_dtype(fixed["price_usd"].dtype)
    assert np.isclose(fixed["price_usd"].iloc[0], 1200.50)
    assert np.isclose(fixed["price_usd"].iloc[1], 45.00)


def test_fixer_date_parsing_day_vs_month_first():
    df = pd.DataFrame({"event_date": ["03/04/2021"]})
    fixer = TypeFixer()

    # Day-first: 03 is day, 04 is month (April 3, 2021)
    dec_day = Decision(issue_id="types", column="event_date", option_id="parse_date_day_first", params={"dayfirst": True}, decided_by="user")
    df_day = fixer.transform(df, fixer.fit(df, dec_day))
    assert df_day["event_date"].iloc[0].day == 3
    assert df_day["event_date"].iloc[0].month == 4

    # Month-first: 03 is month, 04 is day (March 4, 2021)
    dec_mo = Decision(issue_id="types", column="event_date", option_id="parse_date_month_first", params={"dayfirst": False}, decided_by="user")
    df_mo = fixer.transform(df, fixer.fit(df, dec_mo))
    assert df_mo["event_date"].iloc[0].day == 4
    assert df_mo["event_date"].iloc[0].month == 3


def test_fixer_encode_boolean(sample_types_df):
    fixer = TypeFixer()
    dec = Decision(issue_id="types", column="subscribed_flag", option_id="encode_boolean", params={}, decided_by="user")
    fitted = fixer.fit(sample_types_df, dec)
    fixed = fixer.transform(sample_types_df, fitted, is_test=False)

    assert pd.api.types.is_integer_dtype(fixed["subscribed_flag"].dtype)
    assert set(fixed["subscribed_flag"].unique()).issubset({0, 1})
    assert fixed["subscribed_flag"].iloc[0] == 1  # "yes" -> 1
    assert fixed["subscribed_flag"].iloc[1] == 0  # "no" -> 0


def test_fixer_drop_column(sample_types_df):
    fixer = TypeFixer()
    dec = Decision(issue_id="types", column="constant_status", option_id="drop_column", params={"column": "constant_status"}, decided_by="user")
    fitted = fixer.fit(sample_types_df, dec)
    fixed = fixer.transform(sample_types_df, fitted, is_test=False)

    assert "constant_status" not in fixed.columns
    assert len(fixed) == len(sample_types_df)


def test_fixer_test_set_integrity(sample_types_df):
    fixer = TypeFixer()
    dec = Decision(issue_id="types", column="constant_status", option_id="drop_column", params={"column": "constant_status"}, decided_by="user")
    fitted = fixer.fit(sample_types_df, dec)
    fixed_test = fixer.transform(sample_types_df, fitted, is_test=True)

    assert len(fixed_test) == len(sample_types_df)
    assert "constant_status" not in fixed_test.columns


def test_pipeline_reproduce_code_execution(sample_types_df):
    pipeline = RemediationPipeline()
    dec_price = Decision(issue_id="types", column="price_usd", option_id="convert_to_numeric", params={}, decided_by="user")
    dec_bool = Decision(issue_id="types", column="subscribed_flag", option_id="encode_boolean", params={}, decided_by="user")
    dec_const = Decision(issue_id="types", column="constant_status", option_id="drop_column", params={"column": "constant_status"}, decided_by="user")

    pipeline.record_decision(dec_price, sample_types_df)
    pipeline.record_decision(dec_bool, sample_types_df)
    pipeline.record_decision(dec_const, sample_types_df)

    clean_df = pipeline.transform(sample_types_df)
    reproduce_code = pipeline.export_reproduce_code()

    exec_scope = {"pd": pd, "np": np, "re": re}
    exec(reproduce_code, exec_scope)
    clean_func = exec_scope.get("clean_dataset")
    reproduced_df = clean_func(sample_types_df, is_test=False)

    assert clean_df.equals(reproduced_df)
