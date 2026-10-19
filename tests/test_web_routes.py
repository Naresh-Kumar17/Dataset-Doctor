"""
Integration tests for FastAPI web routes verifying Phase 1 requirements.
"""
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
SAMPLE_ACCIDENT = Path(__file__).resolve().parent.parent / "data" / "samples" / "accident.csv"


def test_home_upload_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "DATASET DOCTOR" in response.text
    assert "DROP CSV HERE" in response.text


def test_upload_accident_and_view_profile():
    # 1. Upload accident.csv
    with open(SAMPLE_ACCIDENT, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": ("accident.csv", f, "text/csv")},
            follow_redirects=False
        )
    assert response.status_code == 303
    assert response.headers["location"] == "/profile"
    assert "session_id" in response.cookies

    session_cookie = response.cookies["session_id"]
    client.cookies.set("session_id", session_cookie)

    # 2. View profile page
    profile_resp = client.get("/profile")
    assert profile_resp.status_code == 200
    assert "DATASET PROFILE" in profile_resp.text
    assert "2000" in profile_resp.text
    assert "severe_accident" in profile_resp.text

    # 3. Post lock target
    target_resp = client.post(
        "/target",
        data={"target_column": "severe_accident", "task_type": "classification"},
        follow_redirects=False
    )
    assert target_resp.status_code == 303


def test_bench_page_lists_all_modules():
    response = client.get("/bench")
    assert response.status_code == 200
    assert "AUTOMATED TEST BENCH" in response.text
    assert "Null / Missing Values" in response.text
    assert "Class Imbalance" in response.text
    assert "NOT BUILT" in response.text
    assert "GATE LOG HISTORY" in response.text


def test_individual_module_page():
    response = client.get("/module/nulls")
    assert response.status_code == 200
    assert "Null / Missing Values" in response.text
    assert "Phase 2" in response.text


def test_upload_empty_file_returns_friendly_error():
    response = client.post(
        "/upload",
        files={"file": ("empty.csv", b"", "text/csv")}
    )
    assert response.status_code == 400
    assert "empty (0 bytes)" in response.text


def test_upload_header_only_file_returns_friendly_error():
    response = client.post(
        "/upload",
        files={"file": ("headers.csv", b"col1,col2,col3\n", "text/csv")}
    )
    assert response.status_code == 400
    assert "zero data rows" in response.text
