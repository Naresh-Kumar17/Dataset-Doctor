"""
Gate runner for Dataset Doctor.
Usage:
    python -m doctor.bench nulls
    python -m doctor.bench --all
Evaluates modules against planted-problem datasets with pre-defined ground truth,
records results in bench_results.json, and prints pass/fail outcomes.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from doctor.registry import get_all_modules, get_module, update_module_status

ROOT_DIR = Path(__file__).resolve().parent.parent
BENCH_RESULTS_FILE = ROOT_DIR / "bench_results.json"


def run_nulls_gate() -> tuple[bool, dict]:
    """
    Executes Gate 2 evaluation for Null / Missing Values.
    Verifies:
    1. 100% of planted missing cells found (hidden tokens & sentinels included).
    2. 0 false alarms on clean_control.csv (country 'NA' flagged at most as possible).
    3. Row-loss number is exact.
    4. Planted target-related missingness is reported; planted random missingness is not.
    5. Runs in under 5.0 seconds on 100,000 rows.
    """
    import time
    import numpy as np
    import pandas as pd
    from doctor.loader import load_csv
    from doctor.base import DetectionContext
    from doctor.issues.nulls.detect import NullDetector

    details = {"checks": [], "planted": 0, "found": 0, "missed": 0, "false_alarms": 0}
    detector = NullDetector()

    # 1. Load Planted Data & Truth
    nulls_csv = ROOT_DIR / "data" / "planted" / "nulls.csv"
    truth_json = ROOT_DIR / "data" / "planted" / "nulls_truth.json"
    clean_csv = ROOT_DIR / "data" / "planted" / "clean_control.csv"

    if not nulls_csv.exists() or not truth_json.exists() or not clean_csv.exists():
        from tools.make_planted import make_clean_control, make_nulls_planted
        make_clean_control()
        make_nulls_planted()

    with open(truth_json, "r", encoding="utf-8") as f:
        truth = json.load(f)

    df_planted, _ = load_csv(str(nulls_csv), filename="nulls.csv")
    ctx = DetectionContext(target_column="target", task_type="classification")
    issues = detector.detect(df_planted, ctx=ctx)
    issues_by_col = {issue.column: issue for issue in issues if issue.column}

    # Gate 2.1: Planted vs Found
    planted_total = 0
    found_total = 0
    missed_total = 0

    # Age check (50 NaNs)
    age_issue = issues_by_col.get("age")
    age_found = age_issue.evidence.get("nan_count", 0) if age_issue else 0
    planted_total += 50
    found_total += min(age_found, 50)
    missed_total += max(0, 50 - age_found)
    check_age = age_found == 50
    details["checks"].append({"name": "Age 50 planted NaNs found", "passed": check_age, "found": age_found, "expected": 50})

    # Phone check (60 'N/A' hidden tokens)
    phone_issue = issues_by_col.get("phone")
    phone_found = phone_issue.evidence.get("hidden_token_counts", {}).get("N/A", 0) if phone_issue else 0
    planted_total += 60
    found_total += min(phone_found, 60)
    missed_total += max(0, 60 - phone_found)
    check_phone = phone_found == 60
    details["checks"].append({"name": "Phone 60 planted 'N/A' hidden tokens found", "passed": check_phone, "found": phone_found, "expected": 60})

    # Score check (40 '-999' sentinels)
    score_issue = issues_by_col.get("score")
    score_found = score_issue.evidence.get("sentinel_counts", {}).get("-999", 0) if score_issue else 0
    planted_total += 40
    found_total += min(score_found, 40)
    missed_total += max(0, 40 - score_found)
    check_score = score_found == 40 and score_issue.certainty == "possible"
    details["checks"].append({"name": "Score 40 planted '-999' sentinels found as possible", "passed": check_score, "found": score_found, "expected": 40})

    # Notes check (700 NaNs, severity high)
    notes_issue = issues_by_col.get("notes")
    notes_found = notes_issue.evidence.get("nan_count", 0) if notes_issue else 0
    planted_total += 700
    found_total += min(notes_found, 700)
    missed_total += max(0, 700 - notes_found)
    check_notes = notes_found == 700 and notes_issue.severity == "high"
    details["checks"].append({"name": "Notes 700 planted NaNs found with high severity", "passed": check_notes, "found": notes_found, "expected": 700})

    # Gate 2.2: 0 false alarms on clean_control.csv
    df_clean, _ = load_csv(str(clean_csv), filename="clean_control.csv")
    clean_issues = detector.detect(df_clean, ctx=DetectionContext(target_column="target", task_type="classification"))
    clean_certain = [iss for iss in clean_issues if iss.certainty == "certain"]
    country_issues = [iss for iss in clean_issues if iss.column == "country"]
    country_ok = all(iss.certainty == "possible" for iss in country_issues)
    false_alarms = len(clean_certain)
    check_clean = (false_alarms == 0) and country_ok
    details["checks"].append({"name": "0 false alarms on clean_control.csv (country 'NA' at most possible)", "passed": check_clean, "false_alarms": false_alarms})

    # Gate 2.3: Row loss number is exact
    expected_rows_lost = int(pd.DataFrame({
        c: (df_planted[c].isna() | df_planted[c].isin(["N/A", "-999", -999, ""]))
        for c in ["age", "phone", "score", "notes", "income", "noise_col"]
    }).any(axis=1).sum())
    reported_rows_lost = age_issue.evidence["row_loss_impact"]["rows_lost"] if age_issue else 0
    check_row_loss = (reported_rows_lost == expected_rows_lost)
    details["checks"].append({"name": "Row-loss count is exact", "passed": check_row_loss, "reported": reported_rows_lost, "expected": expected_rows_lost})


    # Gate 2.4: Target-related missingness reported; random missingness not reported
    income_issue = issues_by_col.get("income")
    income_pattern = income_issue.evidence.get("target_pattern") if income_issue else None
    income_related = income_pattern.get("related", False) if income_pattern else False

    noise_issue = issues_by_col.get("noise_col")
    noise_pattern = noise_issue.evidence.get("target_pattern") if noise_issue else None
    noise_related = noise_pattern.get("related", False) if noise_pattern else False

    check_pattern = income_related and (not noise_related)
    details["checks"].append({"name": "Planted target-related missingness reported; random missingness quiet", "passed": check_pattern, "income_p": income_pattern.get("p_value") if income_pattern else None, "noise_p": noise_pattern.get("p_value") if noise_pattern else None})

    # Gate 2.5: Benchmark under 5.0 seconds on 100,000 rows
    np.random.seed(42)
    n_bench = 100000
    df_large = pd.DataFrame({
        "a": np.random.choice([1.0, 2.0, np.nan], size=n_bench),
        "b": np.random.choice(["valid", "N/A", "test"], size=n_bench),
        "c": np.random.choice([100, -999, 200], size=n_bench),
        "d": np.random.choice([0, 1], size=n_bench),
        "e": np.random.normal(50, 10, size=n_bench),
    })
    t0 = time.perf_counter()
    detector.detect(df_large, ctx=DetectionContext(target_column="d", task_type="classification"))
    elapsed = time.perf_counter() - t0
    check_speed = elapsed < 5.0
    details["checks"].append({"name": f"Performance on 100k rows < 5.0s (measured: {elapsed:.3f}s)", "passed": check_speed, "elapsed_seconds": round(elapsed, 3)})

    details["planted"] = planted_total
    details["found"] = found_total
    details["missed"] = missed_total
    details["false_alarms"] = false_alarms

    gate_passed = all(c["passed"] for c in details["checks"])
    return gate_passed, details


def run_split_eval_gate() -> tuple[bool, dict]:
    """
    Executes Gate 3 evaluation for Split + Evaluation Engine.
    Verifies:
    1. Stratified split keeps class ratio within 1% on both parts.
    2. Determinism: Same seed gives same split, different seed gives different split.
    3. Partition isolation: Zero row overlap between train and test partitions.
    4. Planted check: Oversampling before split inflates test recall vs split-first.
    5. Always-predict-majority model shows ~85% accuracy, 0 minority recall, and accuracy not reliable.
    6. Noise guard: Changes smaller than CV std labelled 'within noise'.
    7. Small dataset guard: Datasets < 500 rows produce noisy comparison warning.
    """
    import numpy as np
    import pandas as pd
    from doctor.split import split_dataset
    from doctor.evaluate import evaluate_split, is_significant, EvalResult
    from doctor.compare import run_split_order_demo

    details = {"checks": [], "planted": 7, "found": 0, "missed": 0, "false_alarms": 0}

    accident_csv = ROOT_DIR / "data" / "samples" / "accident.csv"
    if not accident_csv.exists():
        return False, {"checks": [{"name": "accident.csv exists", "passed": False}], "planted": 7, "found": 0, "missed": 7, "false_alarms": 0}

    df_accident = pd.read_csv(accident_csv)

    # 1. Stratified split ratio preservation
    split1 = split_dataset(df_accident, target="severe_accident", random_seed=42)
    full_ratio = float((df_accident["severe_accident"] == 1).mean())
    tr_ratio = float((split1.train_df["severe_accident"] == 1).mean())
    te_ratio = float((split1.test_df["severe_accident"] == 1).mean())
    chk1 = abs(tr_ratio - full_ratio) < 0.01 and abs(te_ratio - full_ratio) < 0.01
    details["checks"].append({
        "name": "Stratified split preserves class ratio within 1% on train and test",
        "passed": chk1,
        "train_ratio": round(tr_ratio, 4),
        "test_ratio": round(te_ratio, 4),
        "target_ratio": round(full_ratio, 4),
    })

    # 2. Determinism
    split_same = split_dataset(df_accident, target="severe_accident", random_seed=42)
    split_diff = split_dataset(df_accident, target="severe_accident", random_seed=999)
    chk2 = (split1.train_indices == split_same.train_indices) and (split1.train_indices != split_diff.train_indices)
    details["checks"].append({
        "name": "Deterministic split (same seed identical, different seed different)",
        "passed": chk2,
    })

    # 3. Partition isolation (no row in both train and test)
    overlap = set(split1.train_indices).intersection(set(split1.test_indices))
    chk3 = len(overlap) == 0 and len(split1.train_indices) + len(split1.test_indices) == len(df_accident)
    details["checks"].append({
        "name": "Zero partition overlap (no row in both train and test)",
        "passed": chk3,
    })

    # 4. Planted check: oversample before split gives higher recall than split first
    demo = run_split_order_demo(df_accident, target="severe_accident", random_seed=42)
    flawed_rec = demo["flawed_oversample_before_split"]["recall"]
    honest_rec = demo["honest_split_first"]["recall"]
    chk4 = demo["leakage_detected"] and flawed_rec > honest_rec
    details["checks"].append({
        "name": f"Synthetic leakage detected: pre-split balancing inflates recall ({flawed_rec:.4f} vs honest {honest_rec:.4f})",
        "passed": chk4,
    })

    # 5. Always-predict-majority model baseline
    eval_res = evaluate_split(split1.train_df, split1.test_df, target="severe_accident", fast=True, random_seed=42)
    base = eval_res.majority_baseline
    chk5 = (
        base is not None
        and abs(base["accuracy"] - 0.85) < 0.02
        and base["recall"] == 0.0
        and eval_res.accuracy_reliable is False
    )
    details["checks"].append({
        "name": "Majority model baseline (~85% accuracy, 0 minority recall, accuracy not reliable banner)",
        "passed": chk5,
    })

    # 6. Change smaller than CV std is labelled 'within noise'
    dummy_before = EvalResult(
        task_type="classification",
        primary_metric="f1",
        train_cv_mean={"f1": 0.50},
        train_cv_std={"f1": 0.04},
        test_scores={"f1": 0.50},
    )
    dummy_after = EvalResult(
        task_type="classification",
        primary_metric="f1",
        train_cv_mean={"f1": 0.52},
        train_cv_std={"f1": 0.04},
        test_scores={"f1": 0.52},
    )
    sig, label = is_significant("f1", dummy_before, dummy_after)
    chk6 = (sig is False) and (label == "within noise")
    details["checks"].append({
        "name": "Delta smaller than CV std labelled 'within noise'",
        "passed": chk6,
    })

    # 7. Dataset under 500 rows shows noisy comparison warning
    df_small = pd.DataFrame({"feat": range(200), "target": [0] * 160 + [1] * 40})
    split_small = split_dataset(df_small, target="target", random_seed=42)
    eval_small = evaluate_split(split_small.train_df, split_small.test_df, target="target", fast=True, random_seed=42)
    chk7 = any("< 500" in w or "fewer than 500" in w for w in eval_small.warnings)
    details["checks"].append({
        "name": "Dataset < 500 rows generates noisy comparison warning",
        "passed": chk7,
    })

    passed_count = sum(1 for c in details["checks"] if c["passed"])
    details["found"] = passed_count
    details["missed"] = len(details["checks"]) - passed_count
    gate_passed = all(c["passed"] for c in details["checks"])
    return gate_passed, details


def run_benchmark(target: str = "--all"):
    print("=" * 65)
    print("DATASET DOCTOR: AUTOMATED TEST BENCH")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    modules = get_all_modules()
    results = {}

    if target != "--all":
        mod = get_module(target)
        if not mod:
            print(f"Error: Unknown module '{target}'. Available: {[m.id for m in modules]}")
            sys.exit(1)
        targets = [mod]
    else:
        targets = modules

    all_passed = True

    for mod in targets:
        print(f"\nMODULE: {mod.name} (id: {mod.id}, phase: {mod.phase})")

        if mod.id == "nulls":
            passed, gate_details = run_nulls_gate()
            status = "gate_passed" if passed else "gate_failed"
            now_iso = datetime.now().isoformat()
            update_module_status("nulls", status, gate_details=gate_details, timestamp=now_iso)

            print(f"  Planted: {gate_details['planted']} | Found: {gate_details['found']} | Missed: {gate_details['missed']} | False alarms: {gate_details['false_alarms']}")
            for chk in gate_details["checks"]:
                mark = "[PASS]" if chk["passed"] else "[FAIL]"
                print(f"    {mark} {chk['name']}")

            print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
            if not passed:
                all_passed = False

            results[mod.id] = {
                "phase": mod.phase,
                "status": status,
                "result": "PASS" if passed else "FAIL",
                "gate_details": gate_details,
            }
        elif mod.id == "split_eval":
            passed, gate_details = run_split_eval_gate()
            status = "gate_passed" if passed else "gate_failed"
            now_iso = datetime.now().isoformat()
            update_module_status("split_eval", status, gate_details=gate_details, timestamp=now_iso)

            print(f"  Planted: {gate_details['planted']} | Found: {gate_details['found']} | Missed: {gate_details['missed']} | False alarms: {gate_details['false_alarms']}")
            for chk in gate_details["checks"]:
                mark = "[PASS]" if chk["passed"] else "[FAIL]"
                print(f"    {mark} {chk['name']}")

            print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
            if not passed:
                all_passed = False

            results[mod.id] = {
                "phase": mod.phase,
                "status": status,
                "result": "PASS" if passed else "FAIL",
                "gate_details": gate_details,
            }
        else:
            if mod.status == "not_built":
                print(f"  Status: NOT BUILT (Scheduled for Phase {mod.phase})")
                results[mod.id] = {
                    "phase": mod.phase,
                    "status": "not_built",
                    "result": "SKIPPED",
                }
            elif mod.status == "gate_passed":
                print("  RESULT: PASS")
                results[mod.id] = {
                    "phase": mod.phase,
                    "status": "gate_passed",
                    "result": "PASS",
                }
            elif mod.status == "gate_failed":
                print("  RESULT: FAIL")
                all_passed = False
                results[mod.id] = {
                    "phase": mod.phase,
                    "status": "gate_failed",
                    "result": "FAIL",
                }

    with open(BENCH_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "last_run": datetime.now().isoformat(),
                "results": results,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 65)
    print(f"Benchmark summary saved to {BENCH_RESULTS_FILE.name}")
    print("=" * 65)
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Dataset Doctor Gate Runner")
    parser.add_argument(
        "module",
        nargs="?",
        default="--all",
        help="Module name to run (e.g. nulls, duplicates) or --all",
    )
    parser.add_argument(
        "--all",
        dest="run_all",
        action="store_true",
        help="Run all modules",
    )
    args = parser.parse_args()

    target = "--all" if args.run_all or args.module == "--all" else args.module
    success = run_benchmark(target)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
