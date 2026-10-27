"""
Dataset Doctor: Class Imbalance Plain-English Explanations & Badge Formatters.
"""
from typing import Optional
from doctor.base import Issue


def explain_imbalance_issue(issue: Issue) -> str:
    """
    Generates a clear, non-technical plain-English explanation of the class imbalance finding.
    """
    ev = issue.evidence
    col = issue.column or "target"
    minority_share = ev.get("minority_share", 0.1)
    imbalance_ratio = ev.get("imbalance_ratio", 10.0)
    minority_class = ev.get("minority_class", "1")
    majority_class = ev.get("majority_class", "0")

    naive_acc = (1.0 - minority_share) * 100.0

    lines = [
        f"**Why this is dangerous:**",
        f"A dumb baseline model predicting '{majority_class}' for every single row will score **{naive_acc:.1f}% accuracy**, yet achieve **0% recall** on the minority class ('{minority_class}').",
        f"When minority cases are rare ({minority_share:.1%}, ratio {imbalance_ratio:.1f}:1), overall accuracy is an illusion that conceals total model failure.",
        "",
        f"**What you must do:**",
        f"1. **Never judge by accuracy alone.** Rely on minority Recall, Precision, F1, and Precision-Recall AUC (PR-AUC).",
        f"2. **Resample training data only.** Any resampling (SMOTE, random oversampling, undersampling) must be kept inside training folds. The test set must retain its true real-world class distribution.",
        f"3. **Ask the domain question:** *{ev.get('prompt_real_world', 'Is this the true rate in the real world?')}* If the sample was artificially collected, model probabilities will need calibration.",
    ]
    return "\n".join(lines)


def format_imbalance_badge(severity: str, minority_share: Optional[float] = None) -> str:
    """
    Renders a Neo-Brutalist HTML badge token for imbalance severity.
    """
    pct_text = f" ({minority_share:.1%})" if minority_share is not None else ""
    if severity == "high":
        return f'<span class="chip high">SEVERE IMBALANCE{pct_text}</span>'
    elif severity == "medium":
        return f'<span class="chip medium">MODERATE IMBALANCE{pct_text}</span>'
    return f'<span class="chip low">BALANCED{pct_text}</span>'
