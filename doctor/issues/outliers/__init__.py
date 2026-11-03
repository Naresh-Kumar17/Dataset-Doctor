"""
Dataset Doctor: Outliers Diagnostic and Remediation Package (Phases 7 & 8).
Exports:
- OutlierDetector: Identifies domain impossible values, unit errors, and extreme tails.
- OutlierAdvisor: Recommends evidence-backed remediation strategies.
- OutlierFixer: Transforms dataset partitions using parameters learned strictly from train.
- explain_outliers_issue, format_outliers_badge: Formats plain English explanations and badges.
"""
from doctor.issues.outliers.detect import OutlierDetector
from doctor.issues.outliers.advise import OutlierAdvisor
from doctor.issues.outliers.fix import OutlierFixer
from doctor.issues.outliers.explain import explain_outliers_issue, format_outliers_badge

__all__ = [
    "OutlierDetector",
    "OutlierAdvisor",
    "OutlierFixer",
    "explain_outliers_issue",
    "format_outliers_badge",
]
