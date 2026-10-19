"""
Phase 0 First Test:
1. Load accident CSV.
2. Flawed setup: oversample before split (demonstrates data leakage).
3. Correct setup: Stratified 80/20 split first.
4. Train RandomForest baseline on train; evaluate on untouched test.
5. Retrain with class_weight="balanced" on train; evaluate on test.
6. Retrain with oversample train only; evaluate on test.
7. Print comparison of accuracy, severe recall, and severe F1.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler


def run_first_test():
    data_path = Path(__file__).resolve().parent.parent / "data" / "samples" / "accident.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Accident dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    X = pd.get_dummies(df.drop(columns=["severe_accident"]), drop_first=True)
    y = df["severe_accident"]

    print("=" * 60)
    print("PHASE 0: FIRST TEST (ACCIDENT BENCHMARK)")
    print(f"Total rows: {len(df)} | Class distribution: {dict(y.value_counts(normalize=True).round(3))}")
    print("=" * 60)

    # 1. Old (Flawed): Oversample BEFORE split
    ros_before = RandomOverSampler(random_state=42)
    X_over, y_over = ros_before.fit_resample(X, y)
    X_train_flawed, X_test_flawed, y_train_flawed, y_test_flawed = train_test_split(
        X_over, y_over, test_size=0.2, random_state=42, stratify=y_over
    )
    clf_flawed = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_flawed.fit(X_train_flawed, y_train_flawed)
    y_pred_flawed = clf_flawed.predict(X_test_flawed)
    
    acc_flawed = accuracy_score(y_test_flawed, y_pred_flawed)
    rec_flawed = recall_score(y_test_flawed, y_pred_flawed)
    f1_flawed = f1_score(y_test_flawed, y_pred_flawed)

    # 2. Correct: Split FIRST (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    test_ratio = dict(y_test.value_counts(normalize=True).round(3))

    # 2a. Baseline (no fix)
    clf_base = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_base.fit(X_train, y_train)
    y_pred_base = clf_base.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    rec_base = recall_score(y_test, y_pred_base)
    f1_base = f1_score(y_test, y_pred_base)

    # 2b. Class weights on train only
    clf_weighted = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    clf_weighted.fit(X_train, y_train)
    y_pred_weighted = clf_weighted.predict(X_test)
    acc_weighted = accuracy_score(y_test, y_pred_weighted)
    rec_weighted = recall_score(y_test, y_pred_weighted)
    f1_weighted = f1_score(y_test, y_pred_weighted)

    # 2c. Oversample train only
    ros_train = RandomOverSampler(random_state=42)
    X_train_over, y_train_over = ros_train.fit_resample(X_train, y_train)
    clf_over = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_over.fit(X_train_over, y_train_over)
    y_pred_over = clf_over.predict(X_test)
    acc_over = accuracy_score(y_test, y_pred_over)
    rec_over = recall_score(y_test, y_pred_over)
    f1_over = f1_score(y_test, y_pred_over)

    results = {
        "Oversample before split (old, wrong)": {"acc": acc_flawed, "recall": rec_flawed, "f1": f1_flawed},
        "Split first, no fix": {"acc": acc_base, "recall": rec_base, "f1": f1_base},
        "Split first, class weights": {"acc": acc_weighted, "recall": rec_weighted, "f1": f1_weighted},
        "Split first, oversample train only": {"acc": acc_over, "recall": rec_over, "f1": f1_over},
    }

    print(f"\nTest set true class distribution (honest split): {test_ratio}")
    print("\nRESULTS TABLE:")
    print(f"{'Setup':<38} | {'Accuracy':<10} | {'Severe Recall':<14} | {'Severe F1':<10}")
    print("-" * 80)
    for setup, m in results.items():
        print(f"{setup:<38} | {m['acc']:<10.4f} | {m['recall']:<14.4f} | {m['f1']:<10.4f}")

    return results, test_ratio


if __name__ == "__main__":
    run_first_test()
