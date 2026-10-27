"""
Dataset Doctor: Class Imbalance Package (Phase 9).
"""
from doctor.issues.imbalance.detect import ImbalanceDetector
from doctor.issues.imbalance.advise import ImbalanceAdvisor
from doctor.issues.imbalance.fix import ImbalanceFixer
from doctor.issues.imbalance.explain import explain_imbalance_issue, format_imbalance_badge

__all__ = [
    "ImbalanceDetector",
    "ImbalanceAdvisor",
    "ImbalanceFixer",
    "explain_imbalance_issue",
    "format_imbalance_badge",
]
