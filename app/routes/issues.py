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
SAMPLE_TYPES = Path(__file__).resolve().parent.parent.parent / "data" / "planted" / "types.csv"
SAMPLE_OUTLIERS = Path(__file__).resolve().parent.parent.parent / "data" / "planted" / "outliers.csv"
SAMPLE_IMBALANCE = Path(__file__).resolve().parent.parent.parent / "data" / "planted" / "imbalance.csv"
SAMPLE_LEAKAGE = Path(__file__).resolve().parent.parent.parent / "data" / "planted" / "leakage.csv"
SAMPLE_PIPELINE = Path(__file__).resolve().parent.parent.parent / "data" / "planted" / "pipeline.csv"
SAMPLE_EXPLAINABILITY = Path(__file__).resolve().parent.parent.parent / "data" / "planted" / "explainability.csv"


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
    elif module_id == "types" and SAMPLE_TYPES.exists():
        # Fallback to types planted dataset to show actual data types diagnostics
        try:
            df, _ = load_csv(str(SAMPLE_TYPES), filename="types.csv (Sample Preview)")
            active_filename = active_filename or "types.csv (Sample Preview)"
            target_col = "target"
            task_type = "classification"
            if session:
                session.df = df
                session.target_column = target_col
                session.task_type = task_type
                session.metadata = {"filename": "types.csv (Sample Preview)"}
        except Exception:
            df = None
    elif module_id in ("outliers", "outliers_advisor") and SAMPLE_OUTLIERS.exists():
        # Fallback to outliers planted dataset to show actual outliers diagnostics
        try:
            df, _ = load_csv(str(SAMPLE_OUTLIERS), filename="outliers.csv (Sample Preview)")
            active_filename = active_filename or "outliers.csv (Sample Preview)"
            target_col = "target"
            task_type = "classification"
            if session:
                session.df = df
                session.target_column = target_col
                session.task_type = task_type
                session.metadata = {"filename": "outliers.csv (Sample Preview)"}
        except Exception:
            df = None
    elif module_id in ("imbalance", "imbalance_advisor") and SAMPLE_IMBALANCE.exists():
        # Fallback to imbalance planted dataset to show class distribution diagnostics
        try:
            df, _ = load_csv(str(SAMPLE_IMBALANCE), filename="imbalance.csv (Sample Preview)")
            active_filename = active_filename or "imbalance.csv (Sample Preview)"
            target_col = "target_90_10"
            task_type = "classification"
            if session:
                session.df = df
                session.target_column = target_col
                session.task_type = task_type
                session.metadata = {"filename": "imbalance.csv (Sample Preview)"}
        except Exception:
            df = None
    elif module_id in ("leakage", "leakage_advisor") and SAMPLE_LEAKAGE.exists():
        # Fallback to leakage planted dataset to show data leakage diagnostics
        try:
            df, _ = load_csv(str(SAMPLE_LEAKAGE), filename="leakage.csv (Sample Preview)")
            active_filename = active_filename or "leakage.csv (Sample Preview)"
            target_col = "target"
            task_type = "classification"
            if session:
                session.df = df
                session.target_column = target_col
                session.task_type = task_type
                session.metadata = {"filename": "leakage.csv (Sample Preview)"}
        except Exception:
            df = None
    elif module_id == "pipeline" and SAMPLE_PIPELINE.exists():
        try:
            df, _ = load_csv(str(SAMPLE_PIPELINE), filename="pipeline.csv (Sample Preview)")
            active_filename = active_filename or "pipeline.csv (Sample Preview)"
            target_col = "target"
            task_type = "classification"
            if session:
                session.df = df
                session.target_column = target_col
                session.task_type = task_type
                session.metadata = {"filename": "pipeline.csv (Sample Preview)"}
        except Exception:
            df = None
    elif module_id == "quality_assessment" and SAMPLE_LEAKAGE.exists():
        try:
            df, _ = load_csv(str(SAMPLE_LEAKAGE), filename="leakage.csv (Sample Preview)")
            active_filename = active_filename or "leakage.csv (Sample Preview)"
            target_col = "target"
            task_type = "classification"
            if session:
                session.df = df
                session.target_column = target_col
                session.task_type = task_type
                session.metadata = {"filename": "leakage.csv (Sample Preview)"}
        except Exception:
            df = None
    elif module_id in ("multi_model", "error_analysis") and SAMPLE_IMBALANCE.exists():
        try:
            df, _ = load_csv(str(SAMPLE_IMBALANCE), filename="imbalance.csv (Sample Preview)")
            active_filename = active_filename or "imbalance.csv (Sample Preview)"
            target_col = "target_90_10"
            task_type = "classification"
            if session:
                session.df = df
                session.target_column = target_col
                session.task_type = task_type
                session.metadata = {"filename": "imbalance.csv (Sample Preview)"}
        except Exception:
            df = None
    elif module_id == "explainability" and SAMPLE_EXPLAINABILITY.exists():
        try:
            df, _ = load_csv(str(SAMPLE_EXPLAINABILITY), filename="explainability.csv (Sample Preview)")
            active_filename = active_filename or "explainability.csv (Sample Preview)"
            target_col = "target"
            task_type = "classification"
            if session:
                session.df = df
                session.target_column = target_col
                session.task_type = task_type
                session.metadata = {"filename": "explainability.csv (Sample Preview)"}
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
    pipeline_data = None
    quality_report = None
    experiment_result = None
    error_profile = None
    attribution_profile = None

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

    elif module_id == "types" and df is not None:
        from doctor.issues.types.detect import TypeDetector
        from doctor.issues.types.advise import TypeAdvisor
        detector = TypeDetector()
        advisor = TypeAdvisor()
        ctx = DetectionContext(target_column=target_col, task_type=task_type)
        issues = detector.detect(df, ctx=ctx)

        for iss in issues:
            summary_counts[iss.severity] = summary_counts.get(iss.severity, 0) + 1
            summary_counts[iss.certainty] = summary_counts.get(iss.certainty, 0) + 1
            summary_counts["total"] += 1

            try:
                rec = advisor.advise(iss, df, target=target_col, ctx=ctx)
                iss.recommendation = {
                    "option": rec.recommended,
                    "reason": rec.reason,
                    "confidence": rec.confidence,
                    "options": rec.options,
                }
            except Exception:
                iss.recommendation = None

    elif module_id in ("outliers", "outliers_advisor") and df is not None:
        from doctor.issues.outliers.detect import OutlierDetector
        from doctor.issues.outliers.advise import OutlierAdvisor
        detector = OutlierDetector()
        advisor = OutlierAdvisor()
        ctx = DetectionContext(target_column=target_col, task_type=task_type)
        issues = detector.detect(df, ctx=ctx)

        for iss in issues:
            summary_counts[iss.severity] = summary_counts.get(iss.severity, 0) + 1
            summary_counts[iss.certainty] = summary_counts.get(iss.certainty, 0) + 1
            summary_counts["total"] += 1

            try:
                rec = advisor.advise(iss, df, target=target_col, ctx=ctx, measure=False)
                iss.recommendation = {
                    "option": rec.recommended,
                    "reason": rec.reason,
                    "confidence": rec.confidence,
                    "options": rec.options,
                }
            except Exception:
                iss.recommendation = None
    elif module_id in ("imbalance", "imbalance_advisor") and df is not None:
        from doctor.issues.imbalance.detect import ImbalanceDetector
        from doctor.issues.imbalance.advise import ImbalanceAdvisor
        detector = ImbalanceDetector()
        advisor = ImbalanceAdvisor()
        ctx = DetectionContext(target_column=target_col, task_type=task_type)
        issues = detector.detect(df, ctx=ctx)

        for iss in issues:
            summary_counts[iss.severity] = summary_counts.get(iss.severity, 0) + 1
            summary_counts[iss.certainty] = summary_counts.get(iss.certainty, 0) + 1
            summary_counts["total"] += 1

            try:
                rec = advisor.advise(iss, df, target=target_col, ctx=ctx, measure=False)
                iss.recommendation = {
                    "option": rec.recommended,
                    "reason": rec.reason,
                    "confidence": rec.confidence,
                    "options": rec.options,
                }
            except Exception:
                iss.recommendation = None

    elif module_id in ("leakage", "leakage_advisor") and df is not None:
        from doctor.issues.leakage.detect import LeakageDetector
        from doctor.issues.leakage.advise import LeakageAdvisor
        detector = LeakageDetector()
        advisor = LeakageAdvisor()
        ctx = DetectionContext(target_column=target_col, task_type=task_type)
        issues = detector.detect(df, ctx=ctx)

        for iss in issues:
            summary_counts[iss.severity] = summary_counts.get(iss.severity, 0) + 1
            summary_counts[iss.certainty] = summary_counts.get(iss.certainty, 0) + 1
            summary_counts["total"] += 1

            try:
                rec = advisor.advise(iss, df, target=target_col, ctx=ctx, measure=False)
                iss.recommendation = {
                    "option": rec.recommended,
                    "reason": rec.reason,
                    "confidence": rec.confidence,
                    "options": rec.options,
                }
            except Exception:
                iss.recommendation = None

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

    elif module_id == "pipeline" and df is not None:
        try:
            from doctor.pipeline import RemediationPipeline, CANONICAL_ORDER
            from doctor.base import Decision
            p = RemediationPipeline()
            if session and hasattr(session, "pipeline") and session.pipeline and session.pipeline.steps:
                p = session.pipeline
            else:
                if "price" in df.columns:
                    p.record_decision(Decision(issue_id="types", column="price", option_id="convert_to_numeric", params={}, decided_by="advisor"), train_df=df)
                if "notes" in df.columns:
                    p.record_decision(Decision(issue_id="nulls", column="notes", option_id="drop_column", params={}, decided_by="advisor"), train_df=df)
                p.record_decision(Decision(issue_id="duplicates", column=None, option_id="keep_first", params={}, decided_by="advisor"), train_df=df)
                if "price" in df.columns:
                    p.record_decision(Decision(issue_id="outliers", column="price", option_id="clip_percentile", params={"lower": 100, "upper": 2000}, decided_by="advisor"), train_df=df)
                if "target_copy" in df.columns:
                    p.record_decision(Decision(issue_id="leakage", column="target_copy", option_id="drop_column", params={}, decided_by="advisor"), train_df=df)

            cleaned_preview = p.transform(df)
            cmp = p.compare_before_after(df, cleaned_preview, target_column=target_col)
            reproduce_code = p.export_reproduce_code(active_filename or "data.csv")
            ordering_val = p.validate_ordering()

            pipeline_data = {
                "steps": [
                    {
                        "step_id": s.step_id,
                        "issue_id": s.issue_id,
                        "column": s.column,
                        "option_id": s.option_id,
                        "decided_by": s.decided_by,
                    }
                    for s in p.steps
                ],
                "canonical_order": CANONICAL_ORDER,
                "is_canonical": p.is_canonical_order(),
                "ordering_validation": ordering_val,
                "comparison": cmp,
                "reproduce_code": reproduce_code,
            }
        except Exception:
            pipeline_data = None

    elif module_id == "quality_assessment" and df is not None:
        try:
            from doctor.quality import DataQualityAssessor
            assessor = DataQualityAssessor()
            decisions = session.decisions if (session and hasattr(session, "decisions")) else None
            quality_report = assessor.assess(df, target_column=target_col, task_type=task_type, decisions=decisions)
        except Exception:
            quality_report = None

    elif module_id == "multi_model" and df is not None:
        try:
            from doctor.experiment import MultiModelExperiment
            experimenter = MultiModelExperiment(random_seed=42)
            n_train = int(len(df) * 0.8)
            tr = df.iloc[:n_train].copy()
            te = df.iloc[n_train:].copy()
            experiment_result = experimenter.run_experiment(tr, te, target_column=target_col)
        except Exception:
            experiment_result = None

    elif module_id == "error_analysis" and df is not None:
        try:
            from doctor.diagnostics import ModelErrorAnalyzer
            from sklearn.ensemble import RandomForestClassifier
            import pandas as pd
            n_train = int(len(df) * 0.8)
            tr = df.iloc[:n_train].copy()
            te = df.iloc[n_train:].copy()
            target = target_col or df.columns[-1]
            X_tr = pd.get_dummies(tr.drop(columns=[target]), drop_first=True)
            y_tr = tr[target].values
            clf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
            clf.fit(X_tr.values, y_tr)
            analyzer = ModelErrorAnalyzer(top_n_errors=5)
            error_profile = analyzer.analyze(clf, te, target_column=target, feature_columns=list(X_tr.columns))
        except Exception:
            error_profile = None

    elif module_id == "explainability" and df is not None:
        try:
            from doctor.explainability import ModelExplainer
            explainer = ModelExplainer(n_repeats=5, random_state=42)
            n_train = int(len(df) * 0.8)
            tr = df.iloc[:n_train].copy()
            target = target_col or "target"
            attribution_profile = explainer.explain(tr, target_column=target)
        except Exception:
            attribution_profile = None

    def badge_formatter(evidence: dict) -> str:
        kind = evidence.get("kind", "")
        subtype = evidence.get("subtype", "")
        if subtype in ("impossible_values", "unit_mixup", "extreme_tail") or "is_skewed" in evidence:
            from doctor.issues.outliers.explain import format_outliers_badge
            from doctor.base import Issue
            mock_iss = Issue(
                module_id="outliers",
                column=None,
                severity="high" if subtype == "impossible_values" else "medium",
                certainty="certain" if subtype == "impossible_values" else "possible",
                title="",
                description="",
                evidence=evidence,
            )
            return format_outliers_badge(mock_iss)
        elif kind in ("text_number", "ambiguous_date", "date", "boolean", "constant", "id_like", "high_cardinality", "mixed_type"):
            from doctor.issues.types.explain import format_types_badge
            return format_types_badge(evidence)
        elif "imbalance_ratio" in evidence or "minority_share" in evidence or evidence.get("kind") == "imbalance":
            from doctor.issues.imbalance.explain import format_imbalance_badge
            return format_imbalance_badge(evidence)
        elif "leak_type" in evidence or evidence.get("kind") == "leakage" or evidence.get("subtype") in ("target_copy", "target_scaled", "target_inverted", "index_leak", "time_order_risk"):
            from doctor.issues.leakage.explain import format_leakage_badge
            return format_leakage_badge(evidence)
        elif "kind" in evidence:
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
            "pipeline_data": pipeline_data,
            "quality_report": quality_report,
            "experiment_result": experiment_result,
            "error_profile": error_profile,
            "attribution_profile": attribution_profile,
            "target_col": target_col,
            "has_data": df is not None,
            "format_badge": badge_formatter,
        }
    )
