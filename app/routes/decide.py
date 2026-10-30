"""
Dataset Doctor: Interactive Decision & Remediation Routes (Phase 4).
Handles:
- Generating advisor recommendations for missing values.
- 10-row Before / After interactive preview.
- Applying single decisions (recommended or user override).
- Batch applying all recommended remediations.
- Exporting decision log and reproduce.py pipeline.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from doctor.base import Decision, DetectionContext, Issue
from doctor.issues.nulls.advise import NullAdvisor
from doctor.issues.nulls.detect import NullDetector
from doctor.issues.nulls.fix import NullFixer
from doctor.pipeline import RemediationPipeline
from app.session import create_session, get_session
from app.security import sanitize_text_input

router = APIRouter(prefix="/decide", tags=["decide"])
SAMPLE_ACCIDENT = Path(__file__).resolve().parent.parent.parent / "data" / "samples" / "accident.csv"
SAMPLE_NULLS = Path(__file__).resolve().parent.parent.parent / "data" / "planted" / "nulls.csv"


class ApplyDecisionRequest(BaseModel):
    column: str
    option_id: str
    params: Optional[Dict[str, Any]] = None
    decided_by: Optional[str] = "user"


class PreviewRequest(BaseModel):
    column: str
    option_id: str
    params: Optional[Dict[str, Any]] = None


def _get_active_df_and_target(request: Request, column: Optional[str] = None) -> tuple[pd.DataFrame, Optional[str], Optional[str], Optional[Any]]:
    session_id = request.cookies.get("session_id")
    session = get_session(session_id)
    if session and session.df is not None:
        if column is None or column in session.df.columns:
            return session.df, session.target_column, session.task_type, session

    # Check if requested column specifically exists in SAMPLE_NULLS
    if column is not None and SAMPLE_NULLS.exists():
        from doctor.loader import load_csv
        try:
            df_nulls, _ = load_csv(str(SAMPLE_NULLS), filename="nulls.csv")
            if column in df_nulls.columns:
                if session is None:
                    session = create_session()
                session.df = df_nulls
                session.target_column = "target"
                session.task_type = "classification"
                session.metadata = {"filename": "nulls.csv (Sample Preview)"}
                return df_nulls, "target", "classification", session
        except Exception:
            pass

    if SAMPLE_ACCIDENT.exists():
        from doctor.loader import load_csv
        df, _ = load_csv(str(SAMPLE_ACCIDENT), filename="accident.csv")
        if session is None:
            session = create_session()
        if session.df is None:
            session.df = df
            session.target_column = "severe_accident"
            session.task_type = "classification"
            session.metadata = {"filename": "accident.csv (Sample)"}
        return df, "severe_accident", "classification", session

    raise HTTPException(status_code=400, detail="No active dataset uploaded or session found.")


@router.get("/advise/{column}")
def get_column_advice(column: str, request: Request):
    """
    Evaluates remediation options for a specific column and returns ranked recommendations.
    """
    df, target_col, task_type, _ = _get_active_df_and_target(request, column=column)

    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found in dataset.")

    detector = NullDetector()
    ctx = DetectionContext(target_column=target_col, task_type=task_type)
    issues = detector.detect(df, ctx=ctx)

    matching_issue = next((iss for iss in issues if iss.column == column), None)
    if not matching_issue:
        # Construct synthetic issue if nulls present
        missing_cnt = int(df[column].isna().sum())
        missing_pct = float(missing_cnt / max(1, len(df)))
        matching_issue = Issue(
            module_id="nulls",
            column=column,
            severity="low" if missing_pct < 0.10 else "medium",
            certainty="certain",
            title=f"Missing values in {column}",
            description=f"{missing_cnt} missing values detected.",
            evidence={"total_rows": len(df), "total_missing_count": missing_cnt, "total_missing_pct": missing_pct},
        )

    advisor = NullAdvisor()
    rec = advisor.advise(matching_issue, df, target=target_col, ctx=ctx)

    return {
        "column": column,
        "recommended": rec.recommended,
        "reason": rec.reason,
        "rule_hits": rec.rule_hits,
        "confidence": rec.confidence,
        "options": [
            {
                "id": o.id,
                "label": o.label,
                "explanation": o.explanation,
                "measured": o.measured or {},
            }
            for o in rec.options
        ],
    }


@router.post("/preview")
def get_fix_preview(req: PreviewRequest, request: Request):
    """
    Generates a 10-row before/after preview comparing original vs transformed cell values.
    """
    df, target_col, _, _ = _get_active_df_and_target(request, column=req.column)
    col = req.column

    if col not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{col}' not found.")

    decision = Decision(
        issue_id="nulls",
        column=col,
        option_id=req.option_id,
        params=req.params or {},
        decided_by="user",
    )

    fixer = NullFixer()
    fitted = fixer.fit(df, decision)
    df_fixed = fixer.transform(df, fitted, is_test=False)

    # Find rows with missing values (including ? and string tokens) to highlight changes
    col_s = df[col]
    tokens_to_null = fitted.get("tokens_to_null", [])
    is_missing_mask = col_s.isna()
    if tokens_to_null:
        is_missing_mask = is_missing_mask | col_s.isin(tokens_to_null)
    if col_s.dtype == object or isinstance(col_s.dtype, pd.StringDtype):
        is_missing_mask = is_missing_mask | col_s.isin(["", "?", "-", "n/a", "N/A"]) | (col_s.astype(str).str.strip() == "")

    missing_indices = df[is_missing_mask].index.tolist()
    if len(missing_indices) >= 10:
        sample_indices = missing_indices[:10]
    else:
        other_indices = [i for i in df.index if i not in missing_indices][:10 - len(missing_indices)]
        sample_indices = missing_indices + other_indices

    preview_rows = []
    for idx in sample_indices:
        orig_val = df.loc[idx, col]
        is_orig_nan = bool(is_missing_mask.loc[idx])

        if req.option_id == "drop_rows" and idx not in df_fixed.index:
            fixed_val = "[ROW DROPPED]"
            changed = True
        elif req.option_id == "drop_column":
            fixed_val = "[COLUMN DROPPED]"
            changed = True
        elif idx in df_fixed.index and col in df_fixed.columns:
            fixed_val = df_fixed.loc[idx, col]
            changed = is_orig_nan or (str(orig_val) != str(fixed_val))
        else:
            fixed_val = "N/A"
            changed = False

        preview_rows.append({
            "index": int(idx),
            "original": None if is_orig_nan else (float(orig_val) if isinstance(orig_val, (np.floating, float)) else str(orig_val)),
            "transformed": None if pd.isna(fixed_val) and fixed_val != "[ROW DROPPED]" else (float(fixed_val) if isinstance(fixed_val, (np.floating, float)) else str(fixed_val)),
            "is_changed": bool(changed),
        })

    return {
        "column": col,
        "option_id": req.option_id,
        "rows": preview_rows,
        "original_null_count": int(is_missing_mask.sum()),
        "fixed_null_count": int(df_fixed[col].isna().sum()) if col in df_fixed.columns else 0,
        "rows_retained": len(df_fixed),
    }


@router.post("/apply")
def apply_decision(req: ApplyDecisionRequest, request: Request):
    """
    Applies remediation to the active session dataset and updates decision log.
    """
    df, target_col, task_type, session = _get_active_df_and_target(request, column=req.column)
    col = req.column

    if col not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{col}' not found.")

    params = dict(req.params or {})
    # Automatically extract hidden tokens and sentinels for this column from issues
    detector = NullDetector()
    ctx = DetectionContext(target_column=target_col, task_type=task_type)
    detected_issues = detector.detect(df, ctx=ctx)
    col_issue = next((iss for iss in detected_issues if iss.column == col), None)
    if col_issue and "tokens_to_null" not in params:
        ev = col_issue.evidence or {}
        tokens = list(ev.get("hidden_token_counts", {}).keys())
        tokens = ["" if t == "<empty>" else t for t in tokens]
        try:
            sentinels = [float(s) for s in ev.get("sentinel_counts", {}).keys()]
        except Exception:
            sentinels = []
        params["tokens_to_null"] = tokens + sentinels

    clean_decided_by = sanitize_text_input(req.decided_by or "user", 50)
    decision = Decision(
        issue_id="nulls",
        column=col,
        option_id=req.option_id,
        params=params,
        decided_by=clean_decided_by,
    )

    fixer = NullFixer()
    fitted = fixer.fit(df, decision)
    df_transformed = fixer.transform(df, fitted, is_test=False)

    # Persist in session
    if session:
        session.df = df_transformed
        session.decisions.append({
            "column": col,
            "option_id": req.option_id,
            "params": params,
            "decided_by": req.decided_by or "user",
            "fitted_params": {k: v for k, v in fitted.items() if k != "tokens_to_null"},
        })

    remaining_nulls = int(df_transformed[col].isna().sum()) if col in df_transformed.columns else 0

    resp = JSONResponse({
        "status": "success",
        "column": col,
        "applied_option": req.option_id,
        "remaining_nulls": remaining_nulls,
        "total_rows": len(df_transformed),
        "total_columns": len(df_transformed.columns),
        "message": f"Successfully applied '{req.option_id}' to column '{col}'.",
    })
    if session:
        resp.set_cookie(
            "session_id",
            session.session_id,
            max_age=3600,
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",
        )
    return resp


@router.post("/apply_all")
def apply_all_recommendations(request: Request):
    """
    Applies recommended remediations across all columns with missing values.
    """
    df, target_col, task_type, session = _get_active_df_and_target(request)

    detector = NullDetector()
    ctx = DetectionContext(target_column=target_col, task_type=task_type)
    issues = detector.detect(df, ctx=ctx)

    if not issues:
        return JSONResponse({"status": "success", "message": "No null issues detected to fix.", "applied_count": 0})

    advisor = NullAdvisor()
    fixer = NullFixer()
    pipeline = RemediationPipeline()
    applied_steps = []

    current_df = df.copy()

    for iss in issues:
        col = iss.column
        if not col or col not in current_df.columns:
            continue

        ev = iss.evidence or {}
        tokens = list(ev.get("hidden_token_counts", {}).keys())
        tokens = ["" if t == "<empty>" else t for t in tokens]
        try:
            sentinels = [float(s) for s in ev.get("sentinel_counts", {}).keys()]
        except Exception:
            sentinels = []
        tokens_to_null = tokens + sentinels

        # Fast non-blocking recommendation calculation for batch apply to prevent proxy timeout
        rec = advisor.advise(iss, current_df, target=target_col, ctx=ctx, measure=False)
        dec = Decision(
            issue_id="nulls",
            column=col,
            option_id=rec.recommended,
            params={"tokens_to_null": tokens_to_null},
            decided_by="recommended_accepted",
        )
        step = pipeline.record_decision(dec, current_df)
        current_df = fixer.transform(current_df, step.fitted_params, is_test=False)
        applied_steps.append({"column": col, "option": rec.recommended, "reason": rec.reason})

    # Save to session
    if session:
        session.df = current_df
        for s in pipeline.get_decision_log():
            session.decisions.append(s)

    resp = JSONResponse({
        "status": "success",
        "applied_count": len(applied_steps),
        "applied_steps": applied_steps,
        "final_rows": len(current_df),
        "final_columns": len(current_df.columns),
        "message": f"Batch applied recommendations to {len(applied_steps)} column(s).",
    })
    if session:
        resp.set_cookie(
            "session_id",
            session.session_id,
            max_age=3600,
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",
        )
    return resp


class ApplyDuplicateDecisionRequest(BaseModel):
    option_id: str
    params: Optional[Dict[str, Any]] = None
    decided_by: Optional[str] = "user"


@router.post("/duplicates/apply")
def apply_duplicate_decision(req: ApplyDuplicateDecisionRequest, request: Request):
    """
    Applies deduplication remediation (keep_first, keep_last, drop_all_conflicting, etc.).
    """
    df, target_col, _, session = _get_active_df_and_target(request)
    from doctor.issues.duplicates.fix import DuplicateFixer
    fixer = DuplicateFixer()
    params = req.params or {}
    if "target_column" not in params and target_col:
        params["target_column"] = target_col

    clean_decided_by = sanitize_text_input(req.decided_by or "user", 50)
    decision = Decision(
        issue_id="duplicates",
        column=None,
        option_id=req.option_id,
        params=params,
        decided_by=clean_decided_by,
    )
    fitted = fixer.fit(df, decision)
    df_transformed = fixer.transform(df, fitted, is_test=False)
    rows_dropped = len(df) - len(df_transformed)

    if session:
        session.df = df_transformed
        session.decisions.append({
            "issue_id": "duplicates",
            "column": None,
            "option_id": req.option_id,
            "params": params,
            "decided_by": req.decided_by or "user",
            "fitted_params": fitted,
        })

    return {
        "status": "success",
        "applied_option": req.option_id,
        "rows_dropped": rows_dropped,
        "final_rows": len(df_transformed),
        "message": f"Successfully applied '{req.option_id}'. Dropped {rows_dropped} rows.",
    }


class ApplyTypeDecisionRequest(BaseModel):
    column: str
    option_id: str
    params: Optional[Dict[str, Any]] = None
    decided_by: Optional[str] = "user"


@router.post("/types/apply")
def apply_type_decision(req: ApplyTypeDecisionRequest, request: Request):
    """
    Applies data type remediation (convert_to_numeric, parse_date, encode_boolean, drop_column, etc.).
    """
    df, target_col, _, session = _get_active_df_and_target(request, column=req.column)
    from doctor.issues.types.fix import TypeFixer
    fixer = TypeFixer()
    params = req.params or {}

    clean_decided_by = sanitize_text_input(req.decided_by or "user", 50)
    decision = Decision(
        issue_id="types",
        column=req.column,
        option_id=req.option_id,
        params=params,
        decided_by=clean_decided_by,
    )
    fitted = fixer.fit(df, decision)
    df_transformed = fixer.transform(df, fitted, is_test=False)

    if session:
        session.df = df_transformed
        session.decisions.append({
            "issue_id": "types",
            "column": req.column,
            "option_id": req.option_id,
            "params": params,
            "decided_by": req.decided_by or "user",
            "fitted_params": fitted,
        })

    return {
        "status": "success",
        "column": req.column,
        "applied_option": req.option_id,
        "new_dtype": str(df_transformed[req.column].dtype) if req.column in df_transformed.columns else "dropped",
        "message": f"Successfully applied '{req.option_id}' to column '{req.column}'.",
    }


@router.get("/export")
def export_reproduce_pipeline(request: Request):
    """
    Exports a standalone reproduce.py script reflecting all decisions made.
    """
    session_id = request.cookies.get("session_id")
    session = get_session(session_id)

    pipeline = RemediationPipeline()
    if session and session.decisions:
        for dec_dict in session.decisions:
            dec = Decision(
                issue_id=dec_dict.get("issue_id", "nulls"),
                column=dec_dict.get("column"),
                option_id=dec_dict.get("option_id", "leave"),
                params=dec_dict.get("params", {}),
                decided_by=dec_dict.get("decided_by", "user"),
            )
            step = pipeline.record_decision(dec)
            step.fitted_params = dec_dict.get("fitted_params", {})

    code = pipeline.export_reproduce_code(raw_data_path="raw_data.csv")
    return PlainTextResponse(code, media_type="text/x-python")


@router.get("/export_csv")
def export_cleaned_csv(request: Request):
    """
    Exports the current in-memory cleaned/transformed dataset as a downloadable CSV.
    Neutralizes spreadsheet formula injection (=, +, -, @) on export.
    """
    df, _, _, _ = _get_active_df_and_target(request)
    export_df = df.copy()
    for c in export_df.columns:
        s = export_df[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
            export_df[c] = s.astype(str).apply(
                lambda v: f"'{v}" if v and str(v).startswith(("=", "+", "-", "@")) else v
            )
    csv_str = export_df.to_csv(index=False)
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="cleaned_dataset.csv"'},
    )

