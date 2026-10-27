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
SAMPLE_NULLS = Path(__file__).resolve().parent.parent.parent / "data" / "planted" / "nulls.csv"
SAMPLE_DUPLICATES = Path(__file__).resolve().parent.parent.parent / "data" / "planted" / "duplicates.csv"


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
    elif module_id in ("nulls", "nulls_advisor") and SAMPLE_NULLS.exists():
        # Fallback to nulls planted dataset to show actual missing value diagnostics and advice
        try:
            df, _ = load_csv(str(SAMPLE_NULLS), filename="nulls.csv (Sample Preview)")
            active_filename = active_filename or "nulls.csv (Sample Preview)"
            target_col = "target"
            task_type = "classification"
            if session:
                session.df = df
                session.target_column = target_col
                session.task_type = task_type
                session.metadata = {"filename": "nulls.csv (Sample Preview)"}
        except Exception:
            df = None
    elif module_id == "duplicates" and SAMPLE_DUPLICATES.exists():
        # Fallback to duplicates planted dataset to show actual duplicates diagnostics
        try:
            df, _ = load_csv(str(SAMPLE_DUPLICATES), filename="duplicates.csv (Sample Preview)")
            active_filename = active_filename or "duplicates.csv (Sample Preview)"
            target_col = "target"
            task_type = "classification"
            if session:
                session.df = df
                session.target_column = target_col
                session.task_type = task_type
                session.metadata = {"filename": "duplicates.csv (Sample Preview)"}
        except Exception:
            df = None
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

    split_info = None
    eval_result = None
    split_demo = None

    if module_id in ("nulls", "nulls_advisor") and df is not None:
        from doctor.issues.nulls.advise import NullAdvisor
        detector = NullDetector()
        advisor = NullAdvisor()
        ctx = DetectionContext(target_column=target_col, task_type=task_type)
        issues = detector.detect(df, ctx=ctx)

        for iss in issues:
            summary_counts[iss.severity] = summary_counts.get(iss.severity, 0) + 1
            summary_counts[iss.certainty] = summary_counts.get(iss.certainty, 0) + 1
            summary_counts["total"] += 1

            # Attach recommendation and fill advice directly to the issue
            if iss.column and iss.column in df.columns:
                try:
                    rec = advisor.advise(iss, df, target=target_col, ctx=ctx, measure=False)
                    fill_preview = None
                    if rec.recommended in ("median", "mean"):
                        col_clean = pd.to_numeric(df[iss.column], errors="coerce").dropna()
                        if not col_clean.empty:
                            val = float(col_clean.median() if rec.recommended == "median" else col_clean.mean())
                            fill_preview = f"Fill with {rec.recommended.upper()} ({val:.2f})"
                        else:
                            fill_preview = f"Fill with {rec.recommended.upper()}"
                    elif rec.recommended == "mode":
                        mode_s = df[iss.column].dropna().mode()
                        mode_val = mode_s.iloc[0] if not mode_s.empty else "N/A"
                        fill_preview = f"Fill with MODE ('{mode_val}')"
                    elif rec.recommended == "missing_label":
                        fill_preview = "Fill with new category 'MISSING'"
                    elif rec.recommended == "add_indicator":
                        fill_preview = f"Add indicator '{iss.column}_was_missing' + fill"
                    elif rec.recommended == "drop_column":
                        fill_preview = "Drop column (>60% missing)"
                    elif rec.recommended == "drop_rows":
                        fill_preview = "Drop rows with missing target"
                    else:
                        fill_preview = f"Apply '{rec.recommended}'"

                    iss.recommendation = {
                        "option": rec.recommended,
                        "reason": rec.reason,
                        "fill_preview": fill_preview,
                        "confidence": rec.confidence,
                        "options": rec.options,
                    }
                except Exception:
                    iss.recommendation = None
            else:
                iss.recommendation = None

        if issues:
            row_loss = issues[0].evidence.get("row_loss_impact")

    elif module_id == "duplicates" and df is not None:
        from doctor.issues.duplicates.detect import DuplicateDetector
        from doctor.issues.duplicates.explain import format_duplicates_badge
        detector = DuplicateDetector()
        ctx = DetectionContext(target_column=target_col, task_type=task_type)
        issues = detector.detect(df, ctx=ctx)

        for iss in issues:
            summary_counts[iss.severity] = summary_counts.get(iss.severity, 0) + 1
            summary_counts[iss.certainty] = summary_counts.get(iss.certainty, 0) + 1
            summary_counts["total"] += 1

    elif module_id == "split_eval" and df is not None:
        target = target_col or (df.columns[-1] if not df.empty else None)
        if target and target in df.columns:
            try:
                from doctor.split import split_dataset
                from doctor.evaluate import evaluate_split
                from doctor.compare import run_split_order_demo

                split_info = split_dataset(df, target=target, task_type=task_type, random_seed=42)
                eval_result = evaluate_split(
                    split_info.train_df,
                    split_info.test_df,
                    target=target,
                    task_type=split_info.task_type,
                    fast=True,
                    random_seed=42,
                )
                if split_info.task_type == "classification":
                    split_demo = run_split_order_demo(df, target=target, random_seed=42)
            except Exception:
                split_info = None
                eval_result = None
                split_demo = None

    def badge_formatter(evidence: dict) -> str:
        if "kind" in evidence:
            from doctor.issues.duplicates.explain import format_duplicates_badge
            return format_duplicates_badge(evidence)
        return format_evidence_badge(evidence)

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
            "split_info": split_info,
            "eval_result": eval_result,
            "split_demo": split_demo,
            "target_col": target_col,
            "has_data": df is not None,
            "format_badge": badge_formatter,
        }
    )
