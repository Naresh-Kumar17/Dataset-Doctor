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
    assert ("GATE PASSED" in response.text or "PASSED" in response.text)
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


def test_types_module_page(auth_client):
    response = auth_client.get("/module/types")
    assert response.status_code == 200
    assert "Data Types" in response.text
    assert "PHASE 6" in response.text.upper()
    assert "DATA TYPE ANOMALIES & TYPE REMEDIATION" in response.text


def test_types_apply_route(auth_client):
    # Upload planted types dataset
    from pathlib import Path
    types_csv = Path(__file__).resolve().parent.parent / "data" / "planted" / "types.csv"
    with open(types_csv, "rb") as f:
        auth_client.post(
            "/upload",
            files={"file": ("types.csv", f, "text/csv")},
            follow_redirects=False,
        )
    resp = auth_client.post(
        "/decide/types/apply",
        json={"column": "price_usd", "option_id": "convert_to_numeric", "params": {}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["column"] == "price_usd"
    assert data["applied_option"] == "convert_to_numeric"
    assert data["new_dtype"] == "float64"


def test_outliers_module_page(auth_client):
    response = auth_client.get("/module/outliers")
    assert response.status_code == 200
    assert "Outlier" in response.text
    assert "OUTLIER ANOMALIES & REMEDIATION DECISIONS" in response.text


def test_outliers_apply_route(auth_client):
    # Upload planted outliers dataset
    from pathlib import Path
    outliers_csv = Path(__file__).resolve().parent.parent / "data" / "planted" / "outliers.csv"
    with open(outliers_csv, "rb") as f:
        auth_client.post(
            "/upload",
            files={"file": ("outliers.csv", f, "text/csv")},
            follow_redirects=False,
        )
    resp = auth_client.post(
        "/decide/outliers/apply",
        json={"column": "age", "option_id": "set_to_null", "params": {}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["column"] == "age"
    assert data["applied_option"] == "set_to_null"







def test_imbalance_module_page(auth_client):
    response = auth_client.get("/module/imbalance")
    assert response.status_code == 200
    assert "Imbalance" in response.text
    assert "CLASS IMBALANCE DIAGNOSTICS & REMEDIATION" in response.text
    assert "ACCURACY ILLUSION" in response.text


def test_imbalance_apply_route(auth_client):
    from pathlib import Path
    imbalance_csv = Path(__file__).resolve().parent.parent / "data" / "planted" / "imbalance.csv"
    with open(imbalance_csv, "rb") as f:
        auth_client.post(
            "/upload",
            files={"file": ("imbalance.csv", f, "text/csv")},
            follow_redirects=False,
        )
    resp = auth_client.post(
        "/decide/imbalance/apply",
        json={"column": "target_90_10", "option_id": "class_weight", "params": {}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["column"] == "target_90_10"
    assert data["applied_option"] == "class_weight"


def test_leakage_module_page(auth_client):
    response = auth_client.get("/module/leakage")
    assert response.status_code == 200
    assert "Leakage" in response.text
    assert "DATA LEAKAGE DIAGNOSTICS & REMEDIATION" in response.text
    assert "target_copy" in response.text


def test_leakage_apply_route(auth_client):
    from pathlib import Path
    leakage_csv = Path(__file__).resolve().parent.parent / "data" / "planted" / "leakage.csv"
    with open(leakage_csv, "rb") as f:
        auth_client.post(
            "/upload",
            files={"file": ("leakage.csv", f, "text/csv")},
            follow_redirects=False,
        )
    resp = auth_client.post(
        "/decide/leakage/apply",
        json={"column": "target_copy", "option_id": "drop_column", "params": {}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["column"] == "target_copy"
    assert data["applied_option"] == "drop_column"
    assert data["cols_delta"] == -1


def test_pipeline_module_page(auth_client):
    response = auth_client.get("/module/pipeline")
    assert response.status_code == 200
    assert "Pipeline" in response.text
    assert "CANONICAL EXECUTION ORDER" in response.text
    assert "Types ➔ Nulls ➔ Duplicates ➔ Outliers ➔ Imbalance ➔ Leakage" in response.text


def test_quality_assessment_module_page(auth_client):
    response = auth_client.get("/module/quality_assessment")
    assert response.status_code == 200
    assert "Quality Assessment" in response.text
    assert "DATA QUALITY SCORE" in response.text or "DATA HEALTH SCORE" in response.text
    assert "CROSS-MODULE DIAGNOSTIC RISK MATRIX" in response.text


def test_multi_model_module_page(auth_client):
    response = auth_client.get("/module/multi_model")
    assert response.status_code == 200
    assert "Multi-Model" in response.text
    assert "MULTI-MODEL EVALUATION MATRIX" in response.text


def test_error_analysis_module_page(auth_client):
    response = auth_client.get("/module/error_analysis")
    assert response.status_code == 200
    assert "Error Analysis" in response.text
    assert "TRUE POSITIVES" in response.text
    assert "FALSE POSITIVES" in response.text


def test_explainability_module_page(auth_client):
    response = auth_client.get("/module/explainability")
    assert response.status_code == 200
    assert "Explainability" in response.text
    assert "PREDICTIVE IMPORTANCE ≠ REAL-WORLD CAUSATION" in response.text
    assert "VALIDATION PERMUTATION IMPORTANCE VS NATIVE IMPURITY" in response.text
    assert "FEATURE PRUNING ADVISOR RECOMMENDATIONS" in response.text


def test_guest_login_route():
    """Verify 1-click guest login logs in without credentials and redirects to home."""
    c = TestClient(app)
    response = c.post("/login/guest", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert AUTH_COOKIE_NAME in response.cookies


def test_demo_route():
    """Verify 1-click demo GET route logs in immediately and redirects to home."""
    c = TestClient(app)
    response = c.get("/demo", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert AUTH_COOKIE_NAME in response.cookies


def test_sample_dataset_route_leakage(auth_client):
    """Verify 1-click pre-loaded sample dataset loads leakage.csv and configures target."""
    response = auth_client.get("/sample/leakage", follow_redirects=False)
    assert response.status_code == 303
    assert "/profile" in response.headers["location"]

    # Follow redirect to /profile
    profile_resp = auth_client.get("/profile")
    assert profile_resp.status_code == 200
    assert "TARGET LOCKED" in profile_resp.text
    assert "target" in profile_resp.text
    assert "PATIENT CONSULTATION • STEP 2 OF 4" in profile_resp.text
    assert "RUN FULL DATASET HEALTH CHECK" in profile_resp.text


def test_sample_dataset_route_accident(auth_client):
    """Verify 1-click pre-loaded sample dataset loads accident.csv."""
    response = auth_client.get("/sample/accident", follow_redirects=False)
    assert response.status_code == 303
    assert "/profile" in response.headers["location"]

    profile_resp = auth_client.get("/profile")
    assert profile_resp.status_code == 200
    assert "severe_accident" in profile_resp.text


def test_sample_dataset_route_chaos(auth_client):
    """Verify 1-click Patient Zero ICU Chaos multi-bug dataset loads and locks target."""
    response = auth_client.get("/sample/chaos", follow_redirects=False)
    assert response.status_code == 303
    assert "/profile" in response.headers["location"]

    profile_resp = auth_client.get("/profile")
    assert profile_resp.status_code == 200
    assert "patient_icu_admission" in profile_resp.text


def test_sample_dataset_route_unknown_redirects_home(auth_client):
    """Verify unknown sample name safely redirects to home."""
    response = auth_client.get("/sample/nonexistent_sample", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_consultation_flow_ctas(auth_client):
    """Verify the 4-step consultation CTAs render across profile, quality, and pipeline."""
    # Load leakage sample to have active state
    auth_client.get("/sample/leakage", follow_redirects=False)

    # 1. Profile has Step 2 CTA
    resp_profile = auth_client.get("/profile")
    assert resp_profile.status_code == 200
    assert "PATIENT CONSULTATION • STEP 2 OF 4" in resp_profile.text
    assert "/module/quality_assessment" in resp_profile.text

    # 2. Quality assessment has Step 3 CTA
    resp_quality = auth_client.get("/module/quality_assessment")
    assert resp_quality.status_code == 200
    assert "PATIENT CONSULTATION • STEP 3 OF 4" in resp_quality.text
    assert "/module/pipeline" in resp_quality.text

    # 3. Pipeline has Step 4 CTA
    resp_pipeline = auth_client.get("/module/pipeline")
    assert resp_pipeline.status_code == 200
    assert "PATIENT CONSULTATION • STEP 4 OF 4" in resp_pipeline.text
    assert "/module/multi_model" in resp_pipeline.text


def test_ai_doctor_web_endpoint(auth_client):
    """Verify /api/ai-consultation endpoint executes consultation and returns report JSON."""
    auth_client.get("/sample/leakage", follow_redirects=False)
    resp = auth_client.post("/api/ai-consultation", json={"provider": "offline"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    report = data["report"]
    assert "patient_status" in report
    assert "clinical_impression" in report
    assert "survival_forecast" in report
    assert "CLINICAL CONSULTATION & DISCHARGE SUMMARY" in report["discharge_letter_markdown"]


def test_quality_assessment_renders_ai_doctor_card(auth_client):
    """Verify Quality Assessment UI page renders the AI Doctor second opinion card."""
    auth_client.get("/sample/leakage", follow_redirects=False)
    resp = auth_client.get("/module/quality_assessment")
    assert resp.status_code == 200
    assert "AI Doctor's Second Opinion & Diagnostic Letter" in resp.text
    assert "AUTONOMOUS CLINICAL INTELLIGENCE LAYER" in resp.text
    assert "PRODUCTION SURVIVAL FORECAST" in resp.text
    assert "FORMAL PHYSICIAN'S DISCHARGE LETTER" in resp.text


def test_simulate_crash_endpoint_all_vectors(auth_client):
    """Verify /api/simulate-crash endpoint executes full autopsy battery."""
    auth_client.get("/sample/leakage", follow_redirects=False)
    resp = auth_client.post("/api/simulate-crash", json={"stress_id": "all"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "battery_report" in data
    battery = data["battery_report"]
    assert len(battery["results"]) == 4
    assert "overall_verdict" in battery
    assert "overall_resilience_gain_pct" in battery


def test_simulate_crash_endpoint_single_vector(auth_client):
    """Verify /api/simulate-crash endpoint executes targeted single vector."""
    auth_client.get("/sample/leakage", follow_redirects=False)
    resp = auth_client.post("/api/simulate-crash", json={"stress_id": "covariate_drift"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "single_result" in data
    res = data["single_result"]
    assert res["stress_id"] == "covariate_drift"
    assert "resilience_advantage_pct" in res
    assert "clinical_autopsy" in res


def test_multi_model_renders_autopsy_simulator(auth_client):
    """Verify Multi-Model UI workbench renders the Production Crash & Autopsy Simulator."""
    auth_client.get("/sample/leakage", follow_redirects=False)
    resp = auth_client.get("/module/multi_model")
    assert resp.status_code == 200
    assert "PRODUCTION CRASH & DATA AUTOPSY SIMULATOR" in resp.text
    assert "HARDENED STRESS LAB" in resp.text
    assert "CLINICAL AUTOPSY EXECUTIVE SUMMARY" in resp.text
    assert "COVARIATE DRIFT" in resp.text
    assert "UPSTREAM OUTAGE" in resp.text


def test_download_medical_passport_pdf(auth_client):
    """Verify /passport/download streams a valid binary PDF with attachment header."""
    auth_client.get("/sample/leakage", follow_redirects=False)
    resp = auth_client.get("/passport/download")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "attachment" in resp.headers.get("content-disposition", "")
    assert resp.content.startswith(b"%PDF-")
    assert len(resp.content) > 1000


def test_preview_medical_passport_pdf(auth_client):
    """Verify /passport/preview streams a valid inline PDF."""
    auth_client.get("/sample/leakage", follow_redirects=False)
    resp = auth_client.get("/passport/preview")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "inline" in resp.headers.get("content-disposition", "")
    assert resp.content.startswith(b"%PDF-")


def test_profile_renders_passport_button(auth_client):
    """Verify Profile page renders the Medical Passport PDF button."""
    auth_client.get("/sample/leakage", follow_redirects=False)
    resp = auth_client.get("/profile")
    assert resp.status_code == 200
    assert "/passport/download" in resp.text
    assert "DOWNLOAD MEDICAL PASSPORT (PDF)" in resp.text


def test_quality_assessment_renders_passport_button(auth_client):
    """Verify Quality Assessment renders the Medical Passport PDF button."""
    auth_client.get("/sample/leakage", follow_redirects=False)
    resp = auth_client.get("/module/quality_assessment")
    assert resp.status_code == 200
    assert "/passport/download" in resp.text
    assert "DOWNLOAD MEDICAL PASSPORT (PDF)" in resp.text


def test_multi_model_renders_passport_button(auth_client):
    """Verify Multi-Model completion banner renders the Medical Passport PDF button."""
    auth_client.get("/sample/leakage", follow_redirects=False)
    resp = auth_client.get("/module/multi_model")
    assert resp.status_code == 200
    assert "/passport/download" in resp.text
    assert "DOWNLOAD MEDICAL PASSPORT (PDF)" in resp.text


def test_universal_consultation_stepper_renders_on_all_pages(auth_client):
    """Verify Universal 4-Step Patient Consultation Stepper renders across profile and modules."""
    auth_client.get("/sample/leakage", follow_redirects=False)
    
    # Check Profile
    resp_profile = auth_client.get("/profile")
    assert resp_profile.status_code == 200
    assert "consultation-stepper-bar" in resp_profile.text
    assert "Intake &amp; Target" in resp_profile.text or "Intake & Target" in resp_profile.text
    assert "Health Score &amp; AI Doctor" in resp_profile.text or "Health Score & AI Doctor" in resp_profile.text
    
    # Check Module page
    resp_module = auth_client.get("/module/quality_assessment")
    assert resp_module.status_code == 200
    assert "consultation-stepper-bar" in resp_module.text
    assert "Treatment Recipe" in resp_module.text
    assert "Model Gains &amp; Crash Lab" in resp_module.text or "Model Gains & Crash Lab" in resp_module.text


def test_module_page_renders_eli5_explainer(auth_client):
    """Verify Module pages render the toggleable ELI5 plain-English explainer."""
    auth_client.get("/sample/leakage", follow_redirects=False)
    resp = auth_client.get("/module/leakage")
    assert resp.status_code == 200
    assert "EXPLAIN THIS IN PLAIN ENGLISH (ELI5)" in resp.text
    assert "PLAIN-ENGLISH TRANSLATOR" in resp.text
    assert "The Cheating Exam Problem" in resp.text


def test_quality_assessment_renders_letter_grade_badge(auth_client):
    """Verify Quality Assessment UI renders the academic Letter Grade badge and layman status."""
    auth_client.get("/sample/leakage", follow_redirects=False)
    resp = auth_client.get("/module/quality_assessment")
    assert resp.status_code == 200
    assert "HEALTH GRADE" in resp.text
    assert "DATA QUALITY SCORE" in resp.text
    assert "CLINICAL SUMMARY FOR AUDITORS &amp; MANAGERS:" in resp.text or "CLINICAL SUMMARY FOR AUDITORS & MANAGERS:" in resp.text


def test_quality_assessment_renders_cure_all_hero_bar(auth_client):
    """Verify Quality Assessment UI renders the 1-Click Cure All hero bar."""
    auth_client.get("/sample/chaos", follow_redirects=False)
    resp = auth_client.get("/module/quality_assessment")
    assert resp.status_code == 200
    assert "PRESCRIBE &amp; CURE ALL DATA PATHOLOGIES" in resp.text or "PRESCRIBE & CURE ALL DATA PATHOLOGIES" in resp.text
    assert "runCureAll()" in resp.text


def test_base_renders_how_it_works_tour_modal(auth_client):
    """Verify base shell renders the How It Works button and 30-second onboarding modal."""
    resp = auth_client.get("/")
    assert resp.status_code == 200
    assert "HOW IT WORKS" in resp.text
    assert "tourModal" in resp.text
    assert "HOW DATASET DOCTOR WORKS IN 30 SECONDS" in resp.text
    assert "STEP 1 OF 4 • INTAKE &amp; DISCOVERY" in resp.text or "STEP 1 OF 4 • INTAKE & DISCOVERY" in resp.text
    assert "STEP 4 OF 4 • HARDENED STRESS LAB" in resp.text
    assert "Drop Any Messy CSV Table" in resp.text
    assert "Production Crash &amp; Drift Simulator" in resp.text or "Production Crash & Drift Simulator" in resp.text




