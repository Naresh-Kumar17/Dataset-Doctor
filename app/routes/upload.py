"""
Upload and Profile routes for Dataset Doctor.
Handles file ingestion, validation, session registration, profile generation,
and target column specification.
"""
from pathlib import Path
from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from doctor.loader import LoaderError, load_csv
from doctor.profile import profile_dataframe
from doctor.registry import get_all_modules
from app.session import create_session, get_session

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


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
        }
    )


@router.post("/upload", response_class=HTMLResponse)
async def post_upload_file(request: Request, file: UploadFile = File(...)):
    raw_bytes = await file.read()
    filename = file.filename or "dataset.csv"

    try:
        df, metadata = load_csv(raw_bytes, filename=filename)
        profile = profile_dataframe(df)
    except LoaderError as e:
        session_id = request.cookies.get("session_id")
        session = get_session(session_id)
        active_filename = session.metadata.get("filename") if session and session.metadata else None
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "active_filename": active_filename,
                "error": str(e),
            },
            status_code=400,
        )
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "active_filename": None,
                "error": f"Unexpected error reading file: {e}",
            },
            status_code=400,
        )

    # Create new session
    session = create_session()
    session.df = df
    session.metadata = metadata
    session.profile = profile
    session.target_column = profile.get("suggested_target")
    session.task_type = profile.get("suggested_task_type")

    response = RedirectResponse(url="/profile", status_code=303)
    response.set_cookie(
        key="session_id",
        value=session.session_id,
        max_age=3600,
        httponly=True,
        samesite="lax",
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

    if not session:
        return RedirectResponse(url="/", status_code=303)

    session.target_column = target_column
    session.task_type = task_type

    return RedirectResponse(url="/profile?locked=1", status_code=303)
