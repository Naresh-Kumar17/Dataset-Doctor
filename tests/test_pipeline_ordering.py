"""
Unit tests for Phase 13: 10-Step Deterministic Pipeline & Causal Ordering.
"""
import pandas as pd
import numpy as np
from doctor.base import Decision
from doctor.pipeline import (
    RemediationPipeline,
    CANONICAL_ORDER,
    get_stage_index,
)


def test_canonical_ordering_rules():
    assert CANONICAL_ORDER == ["types", "nulls", "duplicates", "outliers", "imbalance", "leakage"]
    assert get_stage_index("types") < get_stage_index("nulls")
    assert get_stage_index("nulls") < get_stage_index("duplicates")
    assert get_stage_index("duplicates") < get_stage_index("outliers")
    assert get_stage_index("outliers") < get_stage_index("imbalance")
    assert get_stage_index("imbalance") < get_stage_index("leakage")


def test_pipeline_order_validation():
    # Valid canonical steps
    dec_type = Decision(issue_id="types", column="price", option_id="convert_to_numeric", params={}, decided_by="user")
    dec_outlier = Decision(issue_id="outliers", column="price", option_id="clip_percentile", params={"lower": 10, "upper": 100}, decided_by="user")

    pipe = RemediationPipeline()
    pipe.record_decision(dec_type)
    pipe.record_decision(dec_outlier)
    assert pipe.is_canonical_order()
    val = pipe.validate_ordering()
    assert val["is_valid"] is True
    assert len(val["violations"]) == 0

    # Inverted non-canonical steps (outliers before types)
    pipe_inv = RemediationPipeline()
    pipe_inv.record_decision(dec_outlier)
    pipe_inv.record_decision(dec_type)
    assert not pipe_inv.is_canonical_order()
    val_inv = pipe_inv.validate_ordering()
    assert val_inv["is_valid"] is False
    assert len(val_inv["violations"]) > 0

    # Auto-reorder
    pipe_inv.reorder_to_canonical()
    assert pipe_inv.is_canonical_order()
    assert pipe_inv.steps[0].issue_id == "types"
    assert pipe_inv.steps[1].issue_id == "outliers"


def test_pipeline_end_to_end_transform():
    df = pd.DataFrame({
        "id": [1, 2, 3, 3, 4],
        "cost": ["$10.00", "$20.00", "$30.00", "$30.00", "$999.00"],
        "leak_col": [1, 0, 1, 1, 0],
        "target": [1, 0, 1, 1, 0],
    })

    pipe = RemediationPipeline()
    pipe.record_decision(Decision(issue_id="types", column="cost", option_id="convert_to_numeric", params={}, decided_by="user"), train_df=df)
    pipe.record_decision(Decision(issue_id="duplicates", column=None, option_id="keep_first", params={}, decided_by="user"), train_df=df)
    pipe.record_decision(Decision(issue_id="outliers", column="cost", option_id="clip_percentile", params={"lower": 10.0, "upper": 100.0}, decided_by="user"), train_df=df)
    pipe.record_decision(Decision(issue_id="leakage", column="leak_col", option_id="drop_column", params={}, decided_by="user"), train_df=df)

    clean_df = pipe.transform(df, is_test=False)
    assert "leak_col" not in clean_df.columns
    assert len(clean_df) == 4  # 1 duplicate dropped
    assert pd.api.types.is_numeric_dtype(clean_df["cost"])
    assert clean_df["cost"].max() <= 100.0

    # Test transform preserves row count strictly
    test_df = pd.DataFrame({
        "id": [10, 10],
        "cost": ["$15.00", "$15.00"],
        "leak_col": [1, 1],
        "target": [1, 1],
    })
    clean_test = pipe.transform(test_df, is_test=True)
    assert len(clean_test) == 2  # Test rows never dropped by deduplication
    assert "leak_col" not in clean_test.columns


def test_pipeline_before_after_comparison():
    raw_df = pd.DataFrame({
        "a": [1, 2, None],
        "target": [0, 1, 0],
    })
    clean_df = pd.DataFrame({
        "a": [1.0, 2.0, 1.5],
        "target": [0, 1, 0],
    })
    pipe = RemediationPipeline()
    cmp_res = pipe.compare_before_after(raw_df, clean_df, target_column="target")
    assert cmp_res["nulls_before"] == 1
    assert cmp_res["nulls_after"] == 0
    assert cmp_res["nulls_eliminated"] == 1
    assert cmp_res["rows_delta"] == 0
