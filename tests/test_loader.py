"""
Unit tests for doctor/loader.py verifying Gate 1 requirements.
"""
from pathlib import Path
import pytest
import pandas as pd

from doctor.loader import (
    load_csv,
    FileTooLargeError,
    EmptyFileError,
    NoDataRowsError,
)

SAMPLE_ACCIDENT = Path(__file__).resolve().parent.parent / "data" / "samples" / "accident.csv"


def test_load_accident_csv_matches_pandas():
    """Row count, column count, and dtypes match plain pandas read."""
    assert SAMPLE_ACCIDENT.exists()
    df_loader, meta = load_csv(str(SAMPLE_ACCIDENT), filename="accident.csv")
    df_pandas = pd.read_csv(SAMPLE_ACCIDENT, keep_default_na=False)

    assert len(df_loader) == len(df_pandas)
    assert list(df_loader.columns) == list(df_pandas.columns)
    assert (df_loader.dtypes == df_pandas.dtypes).all()
    assert meta["rows"] == 2000
    assert meta["columns"] == 9


def test_load_latin1_encoded_file(tmp_path):
    """File with Latin-1 / Western European accented characters loads correctly."""
    latin1_file = tmp_path / "latin1_test.csv"
    content = "city,description,price\nMünchen,Café,15\nZürich,Façade,25\n"
    latin1_file.write_bytes(content.encode("latin-1"))

    df, meta = load_csv(str(latin1_file), filename="latin1_test.csv")
    assert len(df) == 2
    assert "München" in df["city"].values
    assert "Café" in df["description"].values
    assert meta["encoding"].lower() in ["latin-1", "iso-8859-1"]


def test_file_too_large_rejected(tmp_path):
    """File exceeding the size limit is rejected with a clear message."""
    # Test with custom max_size of 1 MB
    fake_large_file = tmp_path / "large.csv"
    fake_large_file.write_bytes(b"a,b\n1,2\n" * 100_000)  # ~800 KB

    # 1. Custom 100 KB limit throws FileTooLargeError
    with pytest.raises(FileTooLargeError) as exc_info:
        load_csv(fake_large_file.read_bytes(), filename="large.csv", max_size=100 * 1024)
    assert "exceeds the maximum allowed size" in str(exc_info.value)

    # 2. 60 MB synthetic check
    sixty_mb_bytes = b"x" * (60 * 1024 * 1024)
    with pytest.raises(FileTooLargeError) as exc_60:
        load_csv(sixty_mb_bytes, filename="huge.csv")
    assert "exceeds the maximum allowed size of 50 MB" in str(exc_60.value)


def test_empty_file_rejected():
    """Empty 0-byte file and whitespace-only file raise EmptyFileError with clear message."""
    with pytest.raises(EmptyFileError) as exc_0:
        load_csv(b"", filename="empty.csv")
    assert "is empty (0 bytes)" in str(exc_0.value)

    with pytest.raises(EmptyFileError) as exc_space:
        load_csv(b"   \n\n   \t", filename="spaces.csv")
    assert "contains only whitespace" in str(exc_space.value) or "is empty" in str(exc_space.value)


def test_header_only_file_rejected():
    """CSV with header row but zero data rows raises NoDataRowsError without crashing."""
    header_only = b"age,speed,severity\n"
    with pytest.raises(NoDataRowsError) as exc_info:
        load_csv(header_only, filename="headers_only.csv")
    assert "zero data rows" in str(exc_info.value)


def test_keep_default_na_preserves_valid_tokens():
    """Ensures keep_default_na=False preserves 'NA' (e.g. Namibia) as a string and not NaN."""
    csv_bytes = b"country_code,val\nNA,100\nUS,200\nDE,300\n"
    df, _ = load_csv(csv_bytes, filename="countries.csv")
    assert df["country_code"].iloc[0] == "NA"
    assert not pd.isna(df["country_code"].iloc[0])
