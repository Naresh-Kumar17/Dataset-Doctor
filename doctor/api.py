"""
Dataset Doctor: Python Core SDK (Phase 18).
Provides a unified, headless Python interface for scripts, notebooks, and pipelines:
- DatasetDoctor.from_csv(path, target=None)
- doc.profile()
- doc.audit()
- doc.split()
- doc.evaluate()
- doc.check_leakage()
- doc.explain()
- doc.auto_remediate()
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from doctor.base import DetectionContext, Issue
from doctor.loader import load_csv, DatasetLoadConfig
from doctor.profile import profile_dataset, DatasetProfile
from doctor.split import split_dataset, SplitConfig
from doctor.evaluate import evaluate_model, EvalResult
from doctor.pipeline import RemediationPipeline
from doctor.quality import DataQualityAssessor, QualityAssessment
from doctor.issues.leakage.detect import LeakageDetector
from doctor.explainability import ModelExplainer, AttributionProfile


class DatasetDoctor:
    """
    Headless SDK interface for Dataset Doctor.
    Enforces the 'Split First, Protect Test Always' discipline.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        task_type: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        self.df = df.copy()
        self.target = target
        self.task_type = task_type
        self.filename = filename or "dataset.csv"

        # Split state
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.split_config: Optional[SplitConfig] = None

        # Evaluator & pipeline state
        self.pipeline: RemediationPipeline = RemediationPipeline()
        self.last_assessment: Optional[QualityAssessment] = None
        self.last_eval_result: Optional[EvalResult] = None

        # Auto-detect task type if target given but task_type not specified
        if self.target and self.target in self.df.columns and not self.task_type:
            target_series = self.df[self.target].dropna()
            n_unique = target_series.nunique()
            if n_unique == 2 or pd.api.types.is_object_dtype(target_series) or pd.api.types.is_string_dtype(target_series):
                self.task_type = "classification"
            elif pd.api.types.is_numeric_dtype(target_series) and n_unique > 20:
                self.task_type = "regression"
            else:
                self.task_type = "classification"

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        target: Optional[str] = None,
        config: Optional[DatasetLoadConfig] = None,
    ) -> "DatasetDoctor":
        """Loads a dataset safely with encoding detection and boundary validation."""
        path_obj = Path(path)
        df, _ = load_csv(str(path_obj), filename=path_obj.name, config=config)
        return cls(df, target=target, filename=path_obj.name)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        target: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> "DatasetDoctor":
        """Creates an SDK instance from an existing in-memory DataFrame."""
        return cls(df, target=target, task_type=task_type)

    def profile(self) -> DatasetProfile:
        """Generates an honest schema, null tokens, and distributional profile."""
        return profile_dataset(self.df, target_column=self.target)

    def audit(self, allow_override: bool = False) -> QualityAssessment:
        """
        Executes complete cross-module risk matrix assessment across all 6 diagnostic domains.
        Calculates 0-100 overall Data Quality Score and checks training blockers.
        """
        assessor = DataQualityAssessor()
        assessment = assessor.assess(
            self.df,
            target_column=self.target,
            task_type=self.task_type,
            decisions=self.pipeline.decisions,
        )
        self.last_assessment = assessment
        return assessment

    def check_leakage(self) -> List[Issue]:
        """Runs fast single-feature correlation and target duplicate leakage detection."""
        detector = LeakageDetector()
        ctx = DetectionContext(target_column=self.target, task_type=self.task_type)
        return detector.detect(self.df, ctx=ctx)

    def split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes a deterministic train/test split.
        Locks the test partition cryptographically and preserves it strictly.
        """
        self.split_config = SplitConfig(
            test_size=test_size,
            random_state=random_state,
            stratify=stratify if self.task_type == "classification" else False,
        )
        split_res = split_dataset(
            self.df,
            target=self.target,
            config=self.split_config,
        )
        self.train_df = split_res.train_df
        self.test_df = split_res.test_df
        return self.train_df, self.test_df

    def evaluate(self, fast_mode: bool = False) -> EvalResult:
        """
        Trains fixed baseline model strictly on train partition and evaluates
        against the untouched, locked test partition with 5-fold CV noise bounds.
        """
        if self.train_df is None or self.test_df is None:
            self.split()

        assert self.train_df is not None and self.test_df is not None
        assert self.target is not None, "Evaluation requires target column to be set."

        self.last_eval_result = evaluate_model(
            self.train_df,
            self.test_df,
            target=self.target,
            task_type=self.task_type or "classification",
            fast=fast_mode,
        )
        return self.last_eval_result

    def explain(self, n_repeats: int = 5, random_state: int = 42) -> AttributionProfile:
        """
        Calculates leak-free permutation feature attribution on validation folds only.
        Test partition is strictly never permuted or snooped.
        """
        if self.train_df is None:
            self.split()

        assert self.train_df is not None
        assert self.target is not None, "Explainability requires target column to be set."

        explainer = ModelExplainer(n_repeats=n_repeats, random_state=random_state)
        return explainer.explain(self.train_df, target_column=self.target)

    def auto_remediate(self) -> RemediationPipeline:
        """
        Builds a canonical 10-step RemediationPipeline populated with evidence-backed fixes.
        """
        from doctor.issues.nulls.detect import NullDetector
        from doctor.issues.nulls.advise import NullAdvisor
        from doctor.issues.duplicates.detect import DuplicateDetector
        from doctor.issues.duplicates.advise import DuplicateAdvisor
        from doctor.issues.types.detect import TypeDetector
        from doctor.issues.types.advise import TypeAdvisor
        from doctor.issues.outliers.detect import OutlierDetector
        from doctor.issues.outliers.advise import OutlierAdvisor
        from doctor.issues.leakage.detect import LeakageDetector
        from doctor.issues.leakage.advise import LeakageAdvisor

        ctx = DetectionContext(target_column=self.target, task_type=self.task_type)
        pipe = RemediationPipeline()

        from doctor.base import Decision

        # Step 1: Types
        type_issues = TypeDetector().detect(self.df, ctx)
        type_adv = TypeAdvisor()
        for iss in type_issues:
            rec = type_adv.advise(iss, self.df, target=self.target, ctx=ctx)
            if rec.recommended != "leave":
                pipe.record_decision(Decision(issue_id="types", column=iss.column, option_id=rec.recommended, params={}, decided_by="recommended_accepted"), self.df)

        # Step 2: Duplicates
        dup_issues = DuplicateDetector().detect(self.df, ctx)
        dup_adv = DuplicateAdvisor()
        for iss in dup_issues:
            rec = dup_adv.advise(iss, self.df, target=self.target, ctx=ctx)
            if rec.recommended != "leave":
                pipe.record_decision(Decision(issue_id="duplicates", column=iss.column, option_id=rec.recommended, params={}, decided_by="recommended_accepted"), self.df)

        # Step 3: Outliers
        outlier_issues = OutlierDetector().detect(self.df, ctx)
        outlier_adv = OutlierAdvisor()
        for iss in outlier_issues:
            rec = outlier_adv.advise(iss, self.df, target=self.target, ctx=ctx)
            if rec.recommended != "leave":
                pipe.record_decision(Decision(issue_id="outliers", column=iss.column, option_id=rec.recommended, params={}, decided_by="recommended_accepted"), self.df)

        # Step 4: Nulls
        null_issues = NullDetector().detect(self.df, ctx)
        null_adv = NullAdvisor()
        for iss in null_issues:
            rec = null_adv.advise(iss, self.df, target=self.target, ctx=ctx, measure=False)
            if rec.recommended != "leave":
                pipe.record_decision(Decision(issue_id="nulls", column=iss.column, option_id=rec.recommended, params={}, decided_by="recommended_accepted"), self.df)

        # Step 5: Leakage
        leak_issues = LeakageDetector().detect(self.df, ctx)
        leak_adv = LeakageAdvisor()
        for iss in leak_issues:
            rec = leak_adv.advise(iss, self.df, target=self.target, ctx=ctx)
            if rec.recommended == "drop_column":
                pipe.record_decision(Decision(issue_id="leakage", column=iss.column, option_id=rec.recommended, params={}, decided_by="recommended_accepted"), self.df)

        self.pipeline = pipe
        return self.pipeline

    def ai_consult(
        self,
        provider: str = "auto",
        api_key: Optional[str] = None,
    ) -> Any:
        """
        Invokes the AI Diagnostic Consultant to produce a clinical discharge summary,
        semantic leakage review, and production survival forecast.
        """
        from doctor.ai_consultant import AIConsultant
        consultant = AIConsultant()
        assessment = self.audit() if not self.last_assessment else self.last_assessment
        profile_obj = self.profile()
        profile_dict = profile_obj.to_dict() if hasattr(profile_obj, "to_dict") else {}
        return consultant.consult(
            df=self.df,
            profile=profile_dict,
            quality_assessment=assessment,
            target_column=self.target,
            provider=provider,
            api_key_override=api_key,
        )

    def export_script(self, out_path: Union[str, Path] = "reproduce.py") -> str:
        """Exports a standalone reproduce.py script with all learned parameters."""
        code_str = self.pipeline.export_reproduce_code()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(code_str)
        return code_str

