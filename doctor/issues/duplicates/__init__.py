"""
Dataset Doctor: Duplicates Module.
"""
from doctor.issues.duplicates.detect import DuplicateDetector
from doctor.issues.duplicates.advise import DuplicateAdvisor
from doctor.issues.duplicates.fix import DuplicateFixer
from doctor.issues.duplicates.explain import explain_duplicates_issue, format_duplicates_badge

__all__ = [
    "DuplicateDetector",
    "DuplicateAdvisor",
    "DuplicateFixer",
    "explain_duplicates_issue",
    "format_duplicates_badge",
]
