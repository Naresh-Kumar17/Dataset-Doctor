"""
Phase 0 Evidence: Accident Dataset Redo
Verifies that balancing before train/test split creates synthetic test leakage,
and documents the honest numbers when splitting first.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler


def run_accident_redo():
    data_path = Path(__file__).resolve().parent.parent / "data" / "samples" / "accident.csv"
    df = pd.read_csv(data_path)
    X = pd.get_dummies(df.drop(columns=["severe_accident"]), drop_first=True)
    y = df["severe_accident"]

    # --- Flawed Setup: Oversample before split ---
    ros = RandomOverSampler(random_state=42)
    X_over, y_over = ros.fit_resample(X, y)
    X_tr_flawed, X_te_flawed, y_tr_flawed, y_te_flawed = train_test_split(
        X_over, y_over, test_size=0.2, random_state=42, stratify=y_over
    )
    clf_flawed = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_flawed.fit(X_tr_flawed, y_tr_flawed)
    pred_flawed = clf_flawed.predict(X_te_flawed)

    # --- Correct Setup: Split first (train only modified) ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Baseline: no fix
    clf_base = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_base.fit(X_tr, y_tr)
    pred_base = clf_base.predict(X_te)

    # Class weights on train only
    clf_cw = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    clf_cw.fit(X_tr, y_tr)
    pred_cw = clf_cw.predict(X_te)

    # Oversample train only
    X_tr_res, y_tr_res = ros.fit_resample(X_tr, y_tr)
    clf_ros = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_ros.fit(X_tr_res, y_tr_res)
    pred_ros = clf_ros.predict(X_te)

    summary = {
        "oversample_before_split": {
            "accuracy": round(accuracy_score(y_te_flawed, pred_flawed), 4),
            "recall": round(recall_score(y_te_flawed, pred_flawed), 4),
            "f1": round(f1_score(y_te_flawed, pred_flawed), 4),
        },
        "split_first_no_fix": {
            "accuracy": round(accuracy_score(y_te, pred_base), 4),
            "recall": round(recall_score(y_te, pred_base), 4),
            "f1": round(f1_score(y_te, pred_base), 4),
        },
        "split_first_class_weights": {
            "accuracy": round(accuracy_score(y_te, pred_cw), 4),
            "recall": round(recall_score(y_te, pred_cw), 4),
            "f1": round(f1_score(y_te, pred_cw), 4),
        },
        "split_first_oversample_train": {
            "accuracy": round(accuracy_score(y_te, pred_ros), 4),
            "recall": round(recall_score(y_te, pred_ros), 4),
            "f1": round(f1_score(y_te, pred_ros), 4),
        },
        "test_class_distribution": dict(y_te.value_counts(normalize=True).round(3)),
    }

    explanation = (
        "1. Oversampling before splitting clones minority rows into both training and test partitions.\n"
        "2. The model achieves high test recall simply by memorizing training instances that appear in the test set.\n"
        "3. Splitting first preserves an unpolluted 85/15 test distribution, revealing real generalization performance."
    )

    print("Accident Redo Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nWhy old score was inflated:")
    print(explanation)

    return summary, explanation


if __name__ == "__main__":
    run_accident_redo()
