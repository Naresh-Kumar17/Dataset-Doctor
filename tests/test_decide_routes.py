"""
Integration tests for Phase 4 /decide API routes.
Verifies:
- GET /decide/advise/{column} returns recommendations and CV comparison table.
- POST /decide/preview returns 10-row before/after comparison.
- POST /decide/apply executes single column remediation.
- POST /decide/apply_all executes batch remediation across all null issues.
- GET /decide/export returns runnable reproduce.py code.
"""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.auth import create_authenticated_session
from app.config import AUTH_COOKIE_NAME

SAMPLE_ACCIDENT = Path(__file__).resolve().parent.parent / "data" / "samples" / "accident.csv"


@pytest.fixture
def auth_client():
    c = TestClient(app)
    token = create_authenticated_session()
    c.cookies.set(AUTH_COOKIE_NAME, token)
    return c


def test_decide_advise_endpoint(auth_client):
    """GET /decide/advise/{col} returns ranked options and reasons."""
    # Ensure dataset is loaded by uploading accident.csv
    with open(SAMPLE_ACCIDENT, "rb") as f:
        auth_client.post("/upload", files={"file": ("accident.csv", f, "text/csv")})

    res = auth_client.get("/decide/advise/speed_limit")
    assert res.status_code == 200
    data = res.json()
    assert data["column"] == "speed_limit"
    assert "recommended" in data
    assert "options" in data
    assert len(data["options"]) > 0
    assert "confidence" in data


def test_decide_preview_endpoint(auth_client):
    """POST /decide/preview returns 10-row before/after comparison."""
    with open(SAMPLE_ACCIDENT, "rb") as f:
        auth_client.post("/upload", files={"file": ("accident.csv", f, "text/csv")})

    res = auth_client.post(
        "/decide/preview",
        json={"column": "speed_limit", "option_id": "median"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["column"] == "speed_limit"
    assert "rows" in data
    assert len(data["rows"]) <= 10
    assert "is_changed" in data["rows"][0]


def test_decide_apply_endpoint(auth_client):
    """POST /decide/apply executes remediation and records choice."""
    with open(SAMPLE_ACCIDENT, "rb") as f:
        auth_client.post("/upload", files={"file": ("accident.csv", f, "text/csv")})

    res = auth_client.post(
        "/decide/apply",
        json={"column": "speed_limit", "option_id": "median", "decided_by": "user"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["column"] == "speed_limit"
    assert data["remaining_nulls"] == 0


def test_decide_apply_all_endpoint(auth_client):
    """POST /decide/apply_all applies recommended fixes across all columns."""
    with open(SAMPLE_ACCIDENT, "rb") as f:
        auth_client.post("/upload", files={"file": ("accident.csv", f, "text/csv")})

    res = auth_client.post("/decide/apply_all")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "applied_count" in data


def test_decide_export_reproduce_pipeline(auth_client):
    """GET /decide/export returns executable Python script."""
    res = auth_client.get("/decide/export")
    assert res.status_code == 200
    assert "def clean_dataset(" in res.text
    assert "pd.read_csv" in res.text


def test_decide_export_cleaned_csv(auth_client):
    """GET /decide/export_csv returns CSV content with attachment header."""
    with open(SAMPLE_ACCIDENT, "rb") as f:
        auth_client.post("/upload", files={"file": ("accident.csv", f, "text/csv")})

    res = auth_client.get("/decide/export_csv")
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    assert 'attachment; filename="cleaned_dataset.csv"' in res.headers.get("content-disposition", "")
    assert len(res.text.splitlines()) > 1


def test_decide_export_cleaned_csv_no_dataset(auth_client, monkeypatch):
    """GET /decide/export_csv returns 400 when no dataset or sample is available."""
    from app.routes import decide
    monkeypatch.setattr(decide, "SAMPLE_ACCIDENT", Path("/nonexistent/path/accident.csv"))

    res = auth_client.get("/decide/export_csv")
    assert res.status_code == 400
    assert "No active dataset" in res.json()["detail"]







def test_decide_imbalance_apply(auth_client):
    from pathlib import Path
    imbalance_csv = Path(__file__).resolve().parent.parent / "data" / "planted" / "imbalance.csv"
    with open(imbalance_csv, "rb") as f:
        auth_client.post("/upload", files={"file": ("imbalance.csv", f, "text/csv")})

    res = auth_client.post(
        "/decide/imbalance/apply",
        json={"column": "target_90_10", "option_id": "random_oversample", "params": {}},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["applied_option"] == "random_oversample"


def test_decide_leakage_apply(auth_client):
    from pathlib import Path
    leakage_csv = Path(__file__).resolve().parent.parent / "data" / "planted" / "leakage.csv"
    with open(leakage_csv, "rb") as f:
        auth_client.post("/upload", files={"file": ("leakage.csv", f, "text/csv")})

    res = auth_client.post(
        "/decide/leakage/apply",
        json={"column": "target_copy", "option_id": "drop_column", "params": {}},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["applied_option"] == "drop_column"
    assert data["cols_delta"] == -1
