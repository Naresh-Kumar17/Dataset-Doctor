"""
Dataset Doctor: Unified Data-to-Model Quality Assessment (Phase 14).
Rules:
- Aggregates diagnostic findings across all 6 diagnostic domains:
  Nulls, Duplicates, Types, Outliers, Imbalance, Leakage.
- Computes an evidence-weighted Data Quality Score (0 - 100).
- Enforces strict training readiness gates: Unresolved critical issues (target leakage,
  target corruption) strictly block model training unless overridden with explicit confirmation.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from doctor.base import Decision, DetectionContext, Issue
from doctor.issues.nulls.detect import NullDetector
from doctor.issues.duplicates.detect import DuplicateDetector
from doctor.issues.types.detect import TypeDetector
from doctor.issues.outliers.detect import OutlierDetector
from doctor.issues.imbalance.detect import ImbalanceDetector
from doctor.issues.leakage.detect import LeakageDetector


@dataclass
class DomainSummary:
    domain_id: str
    domain_name: str
    status: str  # "CLEAN", "WARNING", "CRITICAL"
    total_issues: int
    unresolved_issues: int
    resolved_issues: int
    findings: List[str] = field(default_factory=list)


@dataclass
class QualityAssessment:
    overall_score: int  # 0 to 100
    readiness: str  # "TRAINING_READY", "NEEDS_ATTENTION", "BLOCKED_CRITICAL"
    domain_summaries: Dict[str, DomainSummary]
    critical_blockers: List[str]
    unresolved_issues: List[Issue]
    total_issues_found: int
    total_issues_resolved: int
    raw_issues: Dict[str, List[Issue]] = field(default_factory=dict)
    letter_grade: str = "A"
    layman_verdict: str = "HEALTHY"
    layman_summary: str = ""

    def can_train_model(self, allow_override: bool = False) -> Tuple[bool, str]:
        """
        Determines whether the dataset passes the gate to proceed to model training.
        """
        if self.readiness == "BLOCKED_CRITICAL":
            if allow_override:
                return True, "Model training allowed via explicit user override (high risk)."
            blockers_str = "; ".join(self.critical_blockers)
            return False, f"Model training BLOCKED due to critical integrity hazards: {blockers_str}"
        elif self.readiness == "NEEDS_ATTENTION":
            if allow_override or self.overall_score >= 60:
                return True, "Proceeding with caution; unaddressed warnings remain."
            return False, "Model training paused: Significant unaddressed data anomalies remain."
        return True, "Dataset passes all quality checks. Ready for leak-free model training."

    @property
    def quality_score(self) -> int:
        return self.overall_score

    @property
    def can_train(self) -> bool:
        can, _ = self.can_train_model()
        return can

    @property
    def blocker_reasons(self) -> List[str]:
        return self.critical_blockers


class DataQualityAssessor:
    """
    Evaluates dataset readiness across all 6 diagnostic dimensions.
    """
    def __init__(self):
        self._null_detector = NullDetector()
        self._duplicate_detector = DuplicateDetector()
        self._type_detector = TypeDetector()
        self._outlier_detector = OutlierDetector()
        self._imbalance_detector = ImbalanceDetector()
        self._leakage_detector = LeakageDetector()

    def assess(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None,
        decisions: Optional[List[Decision]] = None,
        precomputed_issues: Optional[Dict[str, List[Issue]]] = None,
    ) -> QualityAssessment:
        ctx = DetectionContext(target_column=target_column, task_type=task_type or "classification")

        # 1. Run all 6 detectors (or use precomputed_issues if already scanned)
        if precomputed_issues is not None:
            all_detected = precomputed_issues
        else:
            null_issues = self._null_detector.detect(df, ctx=ctx)
            dup_issues = self._duplicate_detector.detect(df, ctx=ctx)
            type_issues = self._type_detector.detect(df, ctx=ctx)
            outlier_issues = self._outlier_detector.detect(df, ctx=ctx)
            imbalance_issues = self._imbalance_detector.detect(df, ctx=ctx) if target_column else []
            leakage_issues = self._leakage_detector.detect(df, ctx=ctx) if target_column else []

            all_detected = {
                "nulls": null_issues,
                "duplicates": dup_issues,
                "types": type_issues,
                "outliers": outlier_issues,
                "imbalance": imbalance_issues,
                "leakage": leakage_issues,
            }

        # 2. Check resolved issues via decisions
        decisions_list = decisions or []
        resolved_keys = set()
        for dec in decisions_list:
            if dec.option_id not in ("leave", "skip"):
                resolved_keys.add((dec.issue_id, dec.column))
                resolved_keys.add((dec.issue_id, None))
                if dec.issue_id.startswith("leak"):
                    resolved_keys.add(("leakage", dec.column))
                if dec.issue_id.startswith("null"):
                    resolved_keys.add(("nulls", dec.column))

        domain_titles = {
            "nulls": "Missing Values & Nulls",
            "duplicates": "Duplicates & Conflicts",
            "types": "Data Types & Normalization",
            "outliers": "Outliers & Extreme Values",
            "imbalance": "Class Distribution & Imbalance",
            "leakage": "Data Leakage & Proxies",
        }

        domain_summaries: Dict[str, DomainSummary] = {}
        unresolved_all: List[Issue] = []
        critical_blockers: List[str] = []
        total_found = 0
        total_resolved = 0

        # Score calculation parameters
        penalty = 0

        for dom_id, issues in all_detected.items():
            dom_total = len(issues)
            total_found += dom_total
            dom_resolved = 0
            dom_unresolved_issues = []
            findings = []

            has_critical = False
            has_warning = False

            for iss in issues:
                is_resolved = (dom_id, iss.column) in resolved_keys or (iss.module_id, iss.column) in resolved_keys
                if is_resolved:
                    dom_resolved += 1
                    total_resolved += 1
                else:
                    dom_unresolved_issues.append(iss)
                    unresolved_all.append(iss)

                    # Informational or possible issues do not penalize clean score
                    is_informational = (
                        iss.certainty == "possible"
                        or iss.evidence.get("kind") == "id_like"
                        or (dom_id == "imbalance" and iss.evidence.get("minority_share", 0.0) >= 0.15)
                    )

                    # Assess penalties and blockers
                    if dom_id == "leakage" and iss.severity == "high":
                        has_critical = True
                        critical_blockers.append(f"Direct target leakage in column '{iss.column}' ({iss.title})")
                        penalty += 25
                    elif iss.severity == "high":
                        has_warning = True
                        penalty += 15
                        if dom_id == "nulls" and iss.column == target_column:
                            has_critical = True
                            critical_blockers.append(f"Target column '{target_column}' contains unhandled missing values")
                    elif not is_informational:
                        if iss.severity == "medium":
                            has_warning = True
                            penalty += 6
                        else:
                            penalty += 2

                findings.append(f"{iss.title} ({iss.severity.upper()})")

            # Determine domain status
            if has_critical:
                status = "CRITICAL"
            elif has_warning:
                status = "WARNING"
            elif any(iss.certainty == "certain" and iss.evidence.get("kind") != "id_like" for iss in dom_unresolved_issues):
                status = "WARNING"
            else:
                status = "CLEAN"

            domain_summaries[dom_id] = DomainSummary(
                domain_id=dom_id,
                domain_name=domain_titles.get(dom_id, dom_id),
                status=status,
                total_issues=dom_total,
                unresolved_issues=len(dom_unresolved_issues),
                resolved_issues=dom_resolved,
                findings=findings[:5],
            )

        # Compute overall quality score (0 to 100)
        overall_score = max(0, min(100, 100 - penalty))

        # Determine overall training readiness
        if len(critical_blockers) > 0:
            readiness = "BLOCKED_CRITICAL"
        elif any(ds.status == "WARNING" for ds in domain_summaries.values()) or overall_score < 75:
            readiness = "NEEDS_ATTENTION"
        else:
            readiness = "TRAINING_READY"

        # Determine Academic Letter Grade & Layman Clinical Status
        if readiness == "BLOCKED_CRITICAL":
            letter_grade = "F"
            layman_verdict = "QUARANTINED (CRITICAL CHEATING / INTEGRITY BLOCK)"
            layman_summary = "Cheating data or severe integrity failure detected. Model training is blocked to prevent production collapse."
        elif overall_score >= 95:
            letter_grade = "A+"
            layman_verdict = "PRISTINE (FULLY PRODUCTION CERTIFIED)"
            layman_summary = "Dataset exhibits pristine health with zero critical risks. Certified for high-stakes production deployment."
        elif overall_score >= 85:
            letter_grade = "A"
            layman_verdict = "HEALTHY (PRODUCTION READY)"
            layman_summary = "Dataset is clean, representative, and safe for machine learning model training."
        elif overall_score >= 75:
            letter_grade = "B"
            layman_verdict = "GOOD (STABLE WITH PRECAUTIONS)"
            layman_summary = "Minor statistical anomalies or missingness detected. Safe for training with standard precautions."
        elif overall_score >= 60:
            letter_grade = "C"
            layman_verdict = "FAIR (CLINICAL TREATMENT ADVISED)"
            layman_summary = "Notable data flaws (e.g. missing values or class skew). Treatment recipe recommended before training."
        elif overall_score >= 45:
            letter_grade = "D"
            layman_verdict = "POOR (HIGH DRIFT RISK)"
            layman_summary = "Significant data corruption detected. Models trained on this data will be unstable in production."
        else:
            letter_grade = "F"
            layman_verdict = "FAIL (SEVERELY CONTAMINATED)"
            layman_summary = "Severe data quality failure. Complete automated clinical remediation required before training."

        return QualityAssessment(
            overall_score=overall_score,
            readiness=readiness,
            domain_summaries=domain_summaries,
            critical_blockers=critical_blockers,
            unresolved_issues=unresolved_all,
            total_issues_found=total_found,
            total_issues_resolved=total_resolved,
            raw_issues=all_detected,
            letter_grade=letter_grade,
            layman_verdict=layman_verdict,
            layman_summary=layman_summary,
        )
