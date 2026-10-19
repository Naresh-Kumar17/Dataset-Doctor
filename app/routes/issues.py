"""
Diagnostic module routes for Dataset Doctor.
Renders individual issue module status, detection results, and advisor cards.
"""
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from doctor.registry import get_all_modules, get_module
from app.session import get_session

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


@router.get("/module/{module_id}", response_class=HTMLResponse)
def get_module_page(module_id: str, request: Request):
    module = get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found.")

    session_id = request.cookies.get("session_id")
    session = get_session(session_id)
    active_filename = session.metadata.get("filename") if session and session.metadata else None

    return templates.TemplateResponse(
        request=request,
        name="module.html",
        context={
            "active_filename": active_filename,
            "current_page": module_id,
            "modules": get_all_modules(),
            "module": module,
        }
    )
