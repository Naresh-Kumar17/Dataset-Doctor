"""
Unit tests for doctor/profile.py.
"""
from pathlib import Path
import pandas as pd
from doctor.profile import profile_dataframe, format_memory_size

SAMPLE_ACCIDENT = Path(__file__).resolve().parent.parent / "data" / "samples" / "accident.csv"


def test_profile_accident_dataset():
    assert SAMPLE_ACCIDENT.exists()
    df = pd.read_csv(SAMPLE_ACCIDENT)
    profile = profile_dataframe(df)

    assert profile["n_rows"] == 2000
    assert profile["n_columns"] == 9
    assert profile["memory_bytes"] > 0
    assert "KB" in profile["memory_display"] or "MB" in profile["memory_display"]
    assert profile["suggested_target"] == "severe_accident"
    assert profile["suggested_task_type"] == "classification"
    assert len(profile["preview_rows"]) == 10
    assert len(profile["columns"]) == 9


def test_format_memory_size():
    assert format_memory_size(500) == "500 B"
    assert format_memory_size(1500) == "1.5 KB"
    assert format_memory_size(2 * 1024 * 1024) == "2.0 MB"
