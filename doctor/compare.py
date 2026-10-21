"""
Dataset Doctor: Comparison Engine & Split-Order Evidence Demonstration.
Rules:
- Compares before/after evaluation results across all metrics.
- Flags changes smaller than the CV standard deviation as "within noise".
- Demonstrates why balancing before splitting leaks synthetic test instances,
  inflating test recall compared to honest split-first training.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler

from doctor.evaluate import EvalResult, Preprocessor, is_significant
from doctor.split import split_dataset


@dataclass
class MetricComparison:
    """Represents a comparison row for a single metric."""
    metric: str
    before: float
    after: float
    delta: float
    cv_std: float
    is_significant: bool
    label: Literal["improved", "degraded", "within noise"]


@dataclass
class ComparisonSummary:
    """Structured summary comparing model performance before and after a fix."""
    task_type: str
    primary_metric: str
    metrics: List[MetricComparison]
    verdict: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "primary_metric": self.primary_metric,
            "verdict": self.verdict,
            "warnings": self.warnings,
            "metrics": [
                {
                    "metric": m.metric,
                    "before": m.before,
                    "after": m.after,
                    "delta": round(m.delta, 4),
                    "cv_std": round(m.cv_std, 4),
                    "is_significant": m.is_significant,
                    "label": m.label,
                }
                for m in self.metrics
            ],
        }


def compare_evaluations(before: EvalResult, after: EvalResult) -> ComparisonSummary:
    """
    Compares two EvalResult objects and produces a ComparisonSummary.
    Labels each metric change as 'improved', 'degraded', or 'within noise'.
    """
    metric_keys = list(before.test_scores.keys())
    comparisons: List[MetricComparison] = []
    
    primary_metric = before.primary_metric
    primary_sig = False
    primary_label = "within noise"

    for m in metric_keys:
        b_val = before.test_scores.get(m, 0.0)
        a_val = after.test_scores.get(m, 0.0)
        delta = round(a_val - b_val, 4)
        std = before.train_cv_std.get(m, 0.0)

        sig, label = is_significant(m, before, after)
        if m == primary_metric:
            primary_sig = sig
            primary_label = label

        comparisons.append(
            MetricComparison(
                metric=m,
                before=b_val,
                after=a_val,
                delta=delta,
                cv_std=std,
                is_significant=sig,
                label=label,
            )
        )

    # Determine overall verdict based on primary metric
    if not primary_sig:
        verdict = f"Change in {primary_metric} ({delta:+.4f}) is within CV noise (±{before.train_cv_std.get(primary_metric, 0.0):.4f})."
    elif primary_label == "improved":
        verdict = f"Statistically significant improvement in {primary_metric} ({delta:+.4f})."
    else:
        verdict = f"Performance degraded in {primary_metric} ({delta:+.4f})."

    all_warnings = list(set(before.warnings + after.warnings))

    return ComparisonSummary(
        task_type=before.task_type,
        primary_metric=primary_metric,
        metrics=comparisons,
        verdict=verdict,
        warnings=all_warnings,
    )


def run_split_order_demo(df: pd.DataFrame, target: str, random_seed: int = 42) -> Dict[str, Any]:
    """
    Demonstrates the difference between flawed pre-split balancing and honest post-split balancing.
    
    Proof:
    - Flawed: Oversampling before split clones minority rows into both train and test.
      The model memorizes synthetic test instances, creating artificially inflated recall.
    - Honest: Splitting first ensures the test set maintains real data distribution (no clones).
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    clean_df = df.dropna(subset=[target]).copy()
    y_raw = clean_df[target]
    minority_class = y_raw.value_counts().idxmin()

    # --- Setup 1: Flawed (Oversample before split) ---
    X_raw = clean_df.drop(columns=[target])
    prep_flawed = Preprocessor()
    X_enc = prep_flawed.fit_transform(X_raw)

    ros = RandomOverSampler(random_state=random_seed)
    X_over, y_over = ros.fit_resample(X_enc, y_raw)

    X_tr_flawed, X_te_flawed, y_tr_flawed, y_te_flawed = train_test_split(
        X_over, y_over, test_size=0.2, random_state=random_seed, stratify=y_over
    )

    clf_flawed = RandomForestClassifier(n_estimators=200, random_state=random_seed, n_jobs=-1)
    clf_flawed.fit(X_tr_flawed, y_tr_flawed)
    pred_flawed = clf_flawed.predict(X_te_flawed)

    flawed_metrics = {
        "accuracy": round(float(accuracy_score(y_te_flawed, pred_flawed)), 4),
        "recall": round(float(recall_score(y_te_flawed, pred_flawed, pos_label=minority_class, zero_division=0)), 4),
        "f1": round(float(f1_score(y_te_flawed, pred_flawed, pos_label=minority_class, zero_division=0)), 4),
    }

    # --- Setup 2: Honest (Split first, oversample train only) ---
    split = split_dataset(clean_df, target=target, task_type="classification", random_seed=random_seed)
    prep_honest = Preprocessor()
    X_train_enc = prep_honest.fit_transform(split.train_df.drop(columns=[target]))
    X_test_enc = prep_honest.transform(split.test_df.drop(columns=[target]))
    y_train = split.train_df[target].values
    y_test = split.test_df[target].values

    # Resample train partition only
    X_train_res, y_train_res = ros.fit_resample(X_train_enc, y_train)

    clf_honest = RandomForestClassifier(n_estimators=200, random_state=random_seed, n_jobs=-1)
    clf_honest.fit(X_train_res, y_train_res)
    pred_honest = clf_honest.predict(X_test_enc)

    honest_metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred_honest)), 4),
        "recall": round(float(recall_score(y_test, pred_honest, pos_label=minority_class, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred_honest, pos_label=minority_class, zero_division=0)), 4),
    }

    # --- Setup 3: Baseline (Split first, no fix) ---
    clf_base = RandomForestClassifier(n_estimators=200, random_state=random_seed, n_jobs=-1)
    clf_base.fit(X_train_enc, y_train)
    pred_base = clf_base.predict(X_test_enc)

    base_metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred_base)), 4),
        "recall": round(float(recall_score(y_test, pred_base, pos_label=minority_class, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred_base, pos_label=minority_class, zero_division=0)), 4),
    }

    leakage_detected = flawed_metrics["recall"] > honest_metrics["recall"]

    explanation = [
        "1. Balancing before splitting clones synthetic minority rows across both train and test partitions.",
        "2. The model achieves artificially high test recall (~0.99) simply by memorizing copies of training instances.",
        "3. Splitting first preserves an unpolluted test distribution, reflecting honest generalization on real unseen data.",
    ]

    return {
        "leakage_detected": leakage_detected,
        "flawed_oversample_before_split": flawed_metrics,
        "honest_split_first": honest_metrics,
        "baseline_no_fix": base_metrics,
        "test_distribution_honest": split.class_distribution_test,
        "explanation": explanation,
    }
