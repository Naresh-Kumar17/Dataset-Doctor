"""
Diagnostic module routes for Dataset Doctor.
Renders individual issue module status, detection results, and advisor cards.
"""
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from doctor.registry import get_all_modules, get_module
from doctor.base import DetectionContext
from doctor.issues.nulls.detect import NullDetector
from doctor.issues.nulls.explain import format_evidence_badge
from doctor.loader import load_csv
from app.session import get_session

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))
SAMPLE_ACCIDENT = Path(__file__).resolve().parent.parent.parent / "data" / "samples" / "accident.csv"


@router.get("/module/{module_id}", response_class=HTMLResponse)
def get_module_page(module_id: str, request: Request):
    module = get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found.")

    session_id = request.cookies.get("session_id")
    session = get_session(session_id)
    active_filename = session.metadata.get("filename") if session and session.metadata else None

    df = None
    target_col = None
    task_type = None

    if session and session.df is not None:
        df = session.df
        target_col = session.target_column
        task_type = session.task_type
    elif SAMPLE_ACCIDENT.exists():
        # Fallback to accident dataset for preview
        try:
            df, _ = load_csv(str(SAMPLE_ACCIDENT), filename="accident.csv")
            active_filename = active_filename or "accident.csv (Sample)"
            target_col = "severe_accident"
            task_type = "classification"
        except Exception:
            df = None

    issues = []
    summary_counts = {"high": 0, "medium": 0, "low": 0, "total": 0, "certain": 0, "possible": 0}
    row_loss = None

    if module_id == "nulls" and df is not None:
        detector = NullDetector()
        ctx = DetectionContext(target_column=target_col, task_type=task_type)
        issues = detector.detect(df, ctx=ctx)

        for iss in issues:
            summary_counts[iss.severity] = summary_counts.get(iss.severity, 0) + 1
            summary_counts[iss.certainty] = summary_counts.get(iss.certainty, 0) + 1
            summary_counts["total"] += 1

        if issues:
            row_loss = issues[0].evidence.get("row_loss_impact")

    return templates.TemplateResponse(
        request=request,
        name="module.html",
        context={
            "active_filename": active_filename,
            "current_page": module_id,
            "modules": get_all_modules(),
            "module": module,
            "issues": issues,
            "summary_counts": summary_counts,
            "row_loss": row_loss,
            "has_data": df is not None,
            "format_badge": format_evidence_badge,
        }
    )
