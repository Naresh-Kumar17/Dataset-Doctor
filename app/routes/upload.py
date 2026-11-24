"""
Upload and Profile routes for Dataset Doctor.
Handles file ingestion, validation, session registration, profile generation,
and target column specification. Hardened with sanitization, rate limiting,
and support for real-time GUI upload progress tracking.
"""
from pathlib import Path
from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from doctor.loader import LoaderError, load_csv, MAX_FILE_SIZE_MB, MAX_FILE_SIZE_BYTES
from doctor.profile import profile_dataframe
from doctor.registry import get_all_modules
from app.session import create_session, get_session
from app.security import sanitize_filename, sanitize_text_input, rate_limiter, get_client_ip
from app.config import RATE_LIMIT_UPLOAD

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


def is_xhr(request: Request) -> bool:
    return (
        request.headers.get("x-requested-with") == "XMLHttpRequest"
        or "application/json" in request.headers.get("accept", "")
    )


@router.get("/", response_class=HTMLResponse)
def get_upload_page(request: Request):
    session_id = request.cookies.get("session_id")
    session = get_session(session_id)
    active_filename = session.metadata.get("filename") if session and session.metadata else None
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "active_filename": active_filename,
            "error": None,
            "max_file_size_mb": MAX_FILE_SIZE_MB,
        }
    )


@router.get("/sample/{sample_name}")
def get_load_sample_dataset(request: Request, sample_name: str):
    """Loads a pre-configured real-world sample dataset with 1 click for instant demo and evaluation."""
    root_data = Path(__file__).resolve().parent.parent.parent / "data"
    sample_map = {
        "leakage": (
            root_data / "planted" / "leakage.csv",
            "customer_churn_leakage.csv",
            "target",
            "classification",
        ),
        "nulls": (
            root_data / "planted" / "nulls.csv",
            "hospital_readmission_nulls.csv",
            "target",
            "classification",
        ),
        "imbalance": (
            root_data / "planted" / "imbalance.csv",
            "credit_card_fraud_imbalance.csv",
            "target",
            "classification",
        ),
        "accident": (
            root_data / "samples" / "accident.csv",
            "traffic_accident_severity.csv",
            "severe_accident",
            "classification",
        ),
        "chaos": (
            root_data / "samples" / "patient_zero_pathology.csv",
            "patient_zero_icu_chaos.csv",
            "patient_icu_admission",
            "classification",
        ),
        "patient_zero": (
            root_data / "samples" / "patient_zero_pathology.csv",
            "patient_zero_icu_chaos.csv",
            "patient_icu_admission",
            "classification",
        ),
    }

    if sample_name not in sample_map:
        return RedirectResponse(url="/", status_code=303)

    csv_path, display_name, target_col, task_type = sample_map[sample_name]
    if not csv_path.exists():
        return RedirectResponse(url="/", status_code=303)

    with open(csv_path, "rb") as f:
        raw_bytes = f.read()

    df, metadata = load_csv(raw_bytes, filename=display_name)
    profile = profile_dataframe(df)

    session = create_session()
    session.df = df
    session.metadata = metadata
    session.profile = profile
    session.target_column = target_col if target_col in df.columns else profile.get("suggested_target")
    session.task_type = task_type

    response = RedirectResponse(url="/profile?locked=1&sample=" + sample_name, status_code=303)
    response.set_cookie(
        key="session_id",
        value=session.session_id,
        max_age=3600,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return response


@router.post("/upload")
async def post_upload_file(request: Request, file: UploadFile = File(...)):
    # 1. IP Rate Limiting check
    client_ip = get_client_ip(request)
    allowed, retry_after = rate_limiter.is_allowed(f"upload:{client_ip}", RATE_LIMIT_UPLOAD, 60)
    if not allowed:
        err_msg = f"Upload rate limit exceeded. Please wait {retry_after} seconds before uploading another file."
        if is_xhr(request):
            return JSONResponse(status_code=429, content={"status": "error", "message": err_msg})
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "active_filename": None,
                "error": err_msg,
                "max_file_size_mb": MAX_FILE_SIZE_MB,
            },
            status_code=429,
        )

    # 2. Filename Sanitization against path traversal
    filename = sanitize_filename(file.filename or "dataset.csv")

    # 3. Stream upload in 1MB chunks to prevent memory exhaustion DoS
    CHUNK_SIZE = 1024 * 1024
    chunks = []
    total_bytes = 0
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_FILE_SIZE_BYTES:
            err_msg = f"File '{filename}' exceeds the maximum allowed size of {MAX_FILE_SIZE_MB} MB."
            if is_xhr(request):
                return JSONResponse(status_code=413, content={"status": "error", "message": err_msg})
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "active_filename": None,
                    "error": err_msg,
                    "max_file_size_mb": MAX_FILE_SIZE_MB,
                },
                status_code=413,
            )
        chunks.append(chunk)

    raw_bytes = b"".join(chunks)

    try:
        df, metadata = load_csv(raw_bytes, filename=filename)
        profile = profile_dataframe(df)
    except LoaderError as e:
        session_id = request.cookies.get("session_id")
        session = get_session(session_id)
        active_filename = session.metadata.get("filename") if session and session.metadata else None
        if is_xhr(request):
            return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "active_filename": active_filename,
                "error": str(e),
                "max_file_size_mb": MAX_FILE_SIZE_MB,
            },
            status_code=400,
        )
    except Exception as e:
        err_str = f"Unexpected error reading file: {e}"
        if is_xhr(request):
            return JSONResponse(status_code=400, content={"status": "error", "message": err_str})
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "active_filename": None,
                "error": err_str,
                "max_file_size_mb": MAX_FILE_SIZE_MB,
            },
            status_code=400,
        )

    # 3. Create new session
    session = create_session()
    session.df = df
    session.metadata = metadata
    session.profile = profile
    session.target_column = profile.get("suggested_target")
    session.task_type = profile.get("suggested_task_type")

    # If client requested JSON (e.g. XHR loader), return JSON with redirect URL
    if is_xhr(request):
        response = JSONResponse(
            status_code=200,
            content={"status": "success", "redirect": "/profile"}
        )
    else:
        response = RedirectResponse(url="/profile", status_code=303)

    response.set_cookie(
        key="session_id",
        value=session.session_id,
        max_age=3600,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return response


@router.get("/profile", response_class=HTMLResponse)
def get_profile_page(request: Request, locked: int = 0):
    session_id = request.cookies.get("session_id")
    session = get_session(session_id)

    if not session or session.df is None or session.profile is None:
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="profile.html",
        context={
            "active_filename": session.metadata.get("filename") if session.metadata else "dataset.csv",
            "current_page": "profile",
            "modules": get_all_modules(),
            "profile": session.profile,
            "target_column": session.target_column,
            "task_type": session.task_type,
            "target_locked": (locked == 1) or bool(session.target_column),
        }
    )


@router.post("/target")
def post_lock_target(
    request: Request,
    target_column: str = Form(...),
    task_type: str = Form(...)
):
    session_id = request.cookies.get("session_id")
    session = get_session(session_id)

    if not session or session.df is None:
        return RedirectResponse(url="/", status_code=303)

    # Sanitize and validate inputs
    clean_target = sanitize_text_input(target_column, 120)
    clean_task = sanitize_text_input(task_type, 30).lower()

    if clean_target in ("__none__", "", "none") or clean_task == "unsupervised":
        session.target_column = None
        session.task_type = "unsupervised"
        return RedirectResponse(url="/profile?locked=1", status_code=303)

    if clean_target not in session.df.columns:
        # Invalid column choice
        return RedirectResponse(url="/profile", status_code=303)

    if clean_task not in ("classification", "regression"):
        clean_task = "classification"

    session.target_column = clean_target
    session.task_type = clean_task

    return RedirectResponse(url="/profile?locked=1", status_code=303)
