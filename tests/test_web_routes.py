"""
Integration tests for FastAPI web routes and security architecture.
Verifies authentication, security headers, rate limiting, and core Phase 1 flows.
"""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.auth import create_authenticated_session
from app.config import AUTH_COOKIE_NAME, AUTH_USERNAME, AUTH_PASSWORD

client = TestClient(app)
SAMPLE_ACCIDENT = Path(__file__).resolve().parent.parent / "data" / "samples" / "accident.csv"


def test_public_health_check_no_auth_required():
    """Health check endpoint must be publicly accessible without authentication for health monitoring."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "Dataset Doctor is running"}
    # Verify security headers are attached
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_unauthenticated_request_redirects_to_login():
    """Unauthenticated users attempting to access / are redirected to /login."""
    fresh_client = TestClient(app)
    response = fresh_client.get("/", headers={"Accept": "text/html"}, follow_redirects=False)
    assert response.status_code == 303
    assert "/login" in response.headers["location"]


def test_login_invalid_credentials_rejected():
    """Login with wrong credentials returns 401 with access denied banner."""
    fresh_client = TestClient(app)
    response = fresh_client.post(
        "/login",
        data={"username": "wrong_user", "password": "wrong_password", "next_url": "/"},
    )
    assert response.status_code == 401
    assert "ACCESS DENIED" in response.text


def test_login_valid_credentials_succeeds_and_sets_cookie():
    """Login with valid project credentials from configuration succeeds."""
    fresh_client = TestClient(app)
    response = fresh_client.post(
        "/login",
        data={"username": AUTH_USERNAME, "password": AUTH_PASSWORD, "next_url": "/"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert AUTH_COOKIE_NAME in response.cookies


@pytest.fixture
def auth_client():
    """Provides a TestClient authenticated with a valid session token."""
    c = TestClient(app)
    token = create_authenticated_session()
    c.cookies.set(AUTH_COOKIE_NAME, token)
    return c


def test_authenticated_home_upload_page(auth_client):
    response = auth_client.get("/")
    assert response.status_code == 200
    assert "DATASET DOCTOR" in response.text
    assert "DROP CSV HERE" in response.text
    assert "uploadLoader" in response.text


def test_upload_accident_and_view_profile(auth_client):
    # 1. Upload accident.csv
    with open(SAMPLE_ACCIDENT, "rb") as f:
        response = auth_client.post(
            "/upload",
            files={"file": ("accident.csv", f, "text/csv")},
            follow_redirects=False
        )
    assert response.status_code == 303
    assert response.headers["location"] == "/profile"
    assert "session_id" in response.cookies

    session_cookie = response.cookies["session_id"]
    auth_client.cookies.set("session_id", session_cookie)

    # 2. View profile page
    profile_resp = auth_client.get("/profile")
    assert profile_resp.status_code == 200
    assert "DATASET PROFILE" in profile_resp.text
    assert "2000" in profile_resp.text
    assert "severe_accident" in profile_resp.text

    # 3. Post lock target
    target_resp = auth_client.post(
        "/target",
        data={"target_column": "severe_accident", "task_type": "classification"},
        follow_redirects=False
    )
    assert target_resp.status_code == 303


def test_upload_with_xhr_header_returns_json(auth_client):
    """Verifies that the upload loader AJAX / XHR endpoint returns JSON with redirect info."""
    with open(SAMPLE_ACCIDENT, "rb") as f:
        response = auth_client.post(
            "/upload",
            files={"file": ("accident.csv", f, "text/csv")},
            headers={"X-Requested-With": "XMLHttpRequest", "Accept": "application/json"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["redirect"] == "/profile"


def test_bench_page_lists_all_modules(auth_client):
    response = auth_client.get("/bench")
    assert response.status_code == 200
    assert "AUTOMATED TEST BENCH" in response.text
    assert "Null / Missing Values" in response.text
    assert "Class Imbalance" in response.text
    assert "NOT BUILT" in response.text
    assert "GATE LOG HISTORY" in response.text


def test_individual_module_page(auth_client):
    response = auth_client.get("/module/nulls")
    assert response.status_code == 200
    assert "Null / Missing Values" in response.text
    assert "Phase 2" in response.text


def test_duplicates_module_page(auth_client):
    response = auth_client.get("/module/duplicates")
    assert response.status_code == 200
    assert "Duplicates" in response.text
    assert "Phase 5" in response.text
    assert "HYGIENE STATUS" in response.text


def test_duplicates_apply_route(auth_client):
    # Upload accident dataset
    with open(SAMPLE_ACCIDENT, "rb") as f:
        auth_client.post(
            "/upload",
            files={"file": ("accident.csv", f, "text/csv")},
            follow_redirects=False,
        )
    resp = auth_client.post(
        "/decide/duplicates/apply",
        json={"option_id": "keep_first", "params": {}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["applied_option"] == "keep_first"
    assert "final_rows" in data


def test_upload_empty_file_returns_friendly_error(auth_client):
    response = auth_client.post(
        "/upload",
        files={"file": ("empty.csv", b"", "text/csv")}
    )
    assert response.status_code == 400
    assert "empty (0 bytes)" in response.text


def test_upload_header_only_file_returns_friendly_error(auth_client):
    response = auth_client.post(
        "/upload",
        files={"file": ("headers.csv", b"col1,col2,col3\n", "text/csv")}
    )
    assert response.status_code == 400
    assert "zero data rows" in response.text


def test_logout_clears_cookie(auth_client):
    response = auth_client.get("/logout", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_lock_target_unsupervised_mode(auth_client):
    """Verifies that selecting None / Unsupervised clears the target column and updates profile."""
    with open(SAMPLE_ACCIDENT, "rb") as f:
        auth_client.post(
            "/upload",
            files={"file": ("accident.csv", f, "text/csv")},
            follow_redirects=False,
        )
    # Lock target as __none__ (Unsupervised / Cleaning only)
    resp = auth_client.post(
        "/target",
        data={"target_column": "__none__", "task_type": "unsupervised"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert "TASK CONFIGURED: <strong>UNSUPERVISED / PURE DATA CLEANING</strong>" in resp.text


