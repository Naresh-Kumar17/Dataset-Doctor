"""
Test Bench routes for Dataset Doctor.
Provides an empirical test dashboard for running regression gates against planted-truth datasets.
"""
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from doctor.registry import get_all_modules, update_module_status
from app.session import get_session

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def get_gate_log_text() -> str:
    log_path = ROOT_DIR / "docs" / "gate_log.md"
    if log_path.exists():
        return log_path.read_text(encoding="utf-8")
    return "No gate runs logged yet."


@router.get("/bench", response_class=HTMLResponse)
def get_bench_page(request: Request):
    session_id = request.cookies.get("session_id")
    session = get_session(session_id)
    active_filename = session.metadata.get("filename") if session and session.metadata else None

    return templates.TemplateResponse(
        request=request,
        name="bench.html",
        context={
            "active_filename": active_filename,
            "current_page": "bench",
            "modules": get_all_modules(),
            "gate_log_content": get_gate_log_text(),
        }
    )


@router.post("/bench/run/{module_name}")
def post_run_gate(module_name: str, request: Request):
    return RedirectResponse(url="/bench", status_code=303)
