"""
Unit tests for Dataset Doctor Terminal CLI (Phase 18).
"""
import json
from pathlib import Path
import pytest

from doctor.cli import main


@pytest.fixture
def clean_csv():
    root = Path(__file__).resolve().parent.parent
    return str(root / "data" / "planted" / "clean_control.csv")


@pytest.fixture
def leak_csv():
    root = Path(__file__).resolve().parent.parent
    return str(root / "data" / "planted" / "leakage.csv")


def test_cli_version(capsys):
    ret = main(["version"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Dataset Doctor CLI version" in captured.out


def test_cli_profile_stdout(clean_csv, capsys):
    ret = main(["profile", clean_csv, "--target", "target"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "DATASET DOCTOR: PROFILE" in captured.out
    assert "500 rows x 6 columns" in captured.out


def test_cli_profile_json(clean_csv, capsys):
    ret = main(["profile", clean_csv, "--target", "target", "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["n_rows"] == 500
    assert data["n_cols"] == 6
    assert data["target_column"] == "target"


def test_cli_audit_report(clean_csv, tmp_path):
    report_file = tmp_path / "report.json"
    ret = main(["audit", clean_csv, "--target", "target", "--out", str(report_file)])
    assert ret == 0
    assert report_file.exists()

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["quality_score"] == 100
    assert data["can_train"] is True


def test_cli_audit_strict_mode(leak_csv, clean_csv):
    # Leakage file in strict mode must exit with code 1
    ret_leak = main(["audit", leak_csv, "--target", "target", "--strict"])
    assert ret_leak == 1

    # Clean file in strict mode must exit with code 0
    ret_clean = main(["audit", clean_csv, "--target", "target", "--strict"])
    assert ret_clean == 0


def test_cli_check_leaks_strict(leak_csv, clean_csv):
    # Strict mode should exit 1 when leaks detected
    ret_leak = main(["check-leaks", leak_csv, "--target", "target", "--strict"])
    assert ret_leak == 1

    # Clean file should exit 0
    ret_clean = main(["check-leaks", clean_csv, "--target", "target", "--strict"])
    assert ret_clean == 0


def test_cli_evaluate(clean_csv, capsys):
    ret = main(["evaluate", clean_csv, "--target", "target", "--fast"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "MODEL EVALUATION" in captured.out
    assert "TRAIN 5-FOLD CV SCORES" in captured.out
    assert "HONEST TEST SET PERFORMANCE" in captured.out

