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


def run_nulls_advisor_gate() -> tuple[bool, dict]:
    """
    Executes Gate 4A (Advisor) and Gate 4B (Fixer) evaluation.
    """
    import numpy as np
    import pandas as pd
    from doctor.base import Decision, DetectionContext, Issue
    from doctor.issues.nulls.advise import NullAdvisor
    from doctor.issues.nulls.fix import NullFixer
    from doctor.pipeline import RemediationPipeline

    details = {"checks": [], "planted": 14, "found": 0, "missed": 0, "false_alarms": 0}
    advisor = NullAdvisor()
    fixer = NullFixer()
    rng = np.random.RandomState(42)

    # -------------------------------------------------------------------------
    # Gate 4A: Advisor Ground Truth Scenarios (5 planted scenarios)
    # -------------------------------------------------------------------------
    advisor_matches = 0

    # Scenario 1: Numeric, 3% random missing, symmetric -> mean or median
    n = 600
    x1 = rng.normal(loc=50.0, scale=10.0, size=n)
    mask1 = rng.rand(n) < 0.03
    x1_missing = x1.copy()
    x1_missing[mask1] = np.nan
    y = (x1 > 50).astype(int)
    df1 = pd.DataFrame({"feat_sym": x1_missing, "target": y})
    issue1 = Issue(
        module_id="nulls", column="feat_sym", severity="low", certainty="certain",
        title="Missing values in feat_sym", description="3% missing",
        evidence={"total_rows": n, "total_missing_count": int(mask1.sum()), "total_missing_pct": float(mask1.mean())}
    )
    rec1 = advisor.advise(issue1, df1, target="target", ctx=DetectionContext(target_column="target", task_type="classification"))
    scen1_ok = rec1.recommended in ("mean", "median")
    if scen1_ok:
        advisor_matches += 1
    details["checks"].append({
        "name": f"Scenario 1 (Numeric, 3% random missing, symmetric): recommends 'mean' or 'median' (got '{rec1.recommended}')",
        "passed": scen1_ok,
    })

    # Scenario 2: Numeric, 10% missing, heavily skewed -> median
    x2 = rng.exponential(scale=5.0, size=n)  # Skewness ~ 2.0
    mask2 = rng.rand(n) < 0.10
    x2_missing = x2.copy()
    x2_missing[mask2] = np.nan
    df2 = pd.DataFrame({"feat_skew": x2_missing, "target": y})
    issue2 = Issue(
        module_id="nulls", column="feat_skew", severity="medium", certainty="certain",
        title="Missing values in feat_skew", description="10% missing in skewed column",
        evidence={"total_rows": n, "total_missing_count": int(mask2.sum()), "total_missing_pct": float(mask2.mean())}
    )
    rec2 = advisor.advise(issue2, df2, target="target", ctx=DetectionContext(target_column="target", task_type="classification"))
    scen2_ok = rec2.recommended == "median"
    if scen2_ok:
        advisor_matches += 1
    details["checks"].append({
        "name": f"Scenario 2 (Numeric, 10% missing, heavily skewed): recommends 'median' (got '{rec2.recommended}')",
        "passed": scen2_ok,
    })

    # Scenario 3: Categorical, 5% missing -> mode or missing_label
    cats = rng.choice(["Alpha", "Beta", "Gamma"], size=n, p=[0.7, 0.2, 0.1])
    cats_obj = cats.astype(object)
    mask3 = rng.rand(n) < 0.05
    cats_obj[mask3] = np.nan
    df3 = pd.DataFrame({"feat_cat": cats_obj, "target": y})
    issue3 = Issue(
        module_id="nulls", column="feat_cat", severity="low", certainty="certain",
        title="Missing values in feat_cat", description="5% missing",
        evidence={"total_rows": n, "total_missing_count": int(mask3.sum()), "total_missing_pct": float(mask3.mean())}
    )
    rec3 = advisor.advise(issue3, df3, target="target", ctx=DetectionContext(target_column="target", task_type="classification"))
    scen3_ok = rec3.recommended in ("mode", "missing_label")
    if scen3_ok:
        advisor_matches += 1
    details["checks"].append({
        "name": f"Scenario 3 (Categorical, 5% missing): recommends 'mode' or 'missing_label' (got '{rec3.recommended}')",
        "passed": scen3_ok,
    })

    # Scenario 4: Column 75% missing -> drop_column or add_indicator
    x4 = rng.normal(size=n).astype(object)
    mask4 = rng.rand(n) < 0.75
    x4[mask4] = np.nan
    df4 = pd.DataFrame({"feat_sparse": x4, "target": y})
    issue4 = Issue(
        module_id="nulls", column="feat_sparse", severity="high", certainty="certain",
        title="Missing values in feat_sparse", description="75% missing",
        evidence={"total_rows": n, "total_missing_count": int(mask4.sum()), "total_missing_pct": 0.75}
    )
    rec4 = advisor.advise(issue4, df4, target="target", ctx=DetectionContext(target_column="target", task_type="classification"))
    scen4_ok = rec4.recommended in ("drop_column", "add_indicator")
    if scen4_ok:
        advisor_matches += 1
    details["checks"].append({
        "name": f"Scenario 4 (Column 75% missing): recommends 'drop_column' or 'add_indicator' (got '{rec4.recommended}')",
        "passed": scen4_ok,
    })

    # Scenario 5: Missingness depends on target -> includes add_indicator, excludes drop_rows
    x5 = rng.normal(size=n)
    # Missing preferentially when target == 1
    mask5 = (y == 1) & (rng.rand(n) < 0.60)
    x5_missing = x5.copy()
    x5_missing[mask5] = np.nan
    df5 = pd.DataFrame({"feat_target_link": x5_missing, "target": y})
    issue5 = Issue(
        module_id="nulls", column="feat_target_link", severity="medium", certainty="certain",
        title="Missingness linked to target", description="Missingness correlated with target",
        evidence={
            "total_rows": n, "total_missing_count": int(mask5.sum()), "total_missing_pct": float(mask5.mean()),
            "target_pattern": {"related": True, "p_value": 0.0001}
        }
    )
    rec5 = advisor.advise(issue5, df5, target="target", ctx=DetectionContext(target_column="target", task_type="classification"))
    opt_ids5 = [o.id for o in rec5.options]
    scen5_ok = ("add_indicator" in rec5.recommended or "add_indicator" in opt_ids5) and (rec5.recommended != "drop_rows")
    if scen5_ok:
        advisor_matches += 1
    details["checks"].append({
        "name": f"Scenario 5 (Missingness linked to target): includes 'add_indicator', does NOT recommend 'drop_rows' (got '{rec5.recommended}')",
        "passed": scen5_ok,
    })

    # Gate 4A summary: at least 4 of 5 match
    chk_4a = advisor_matches >= 4
    details["checks"].append({
        "name": f"Gate 4A: At least 4 of 5 advisor recommendations match expected ground truth ({advisor_matches}/5 matched)",
        "passed": chk_4a,
    })

    # Every recommendation shows reason and measured scores
    all_recs = [rec1, rec2, rec3, rec4, rec5]
    chk_reason_scores = all(
        len(r.reason) > 5 and len(r.options) > 0 and all("score_mean" in (o.measured or {}) for o in r.options)
        for r in all_recs
    )
    details["checks"].append({
        "name": "Every recommendation provides a clear reason and measured CV scores",
        "passed": chk_reason_scores,
    })

    # Close call detection check
    # When options have virtually indistinguishable scores within noise, confidence should be 'close call'
    has_close_call = any(r.confidence == "close call" for r in all_recs)
    # If not triggered in random planted data, test explicitly with equal-score mock options
    if not has_close_call:
        from doctor.base import Option
        mock_opts = [
            Option(id="mean", label="Mean", measured={"score_mean": 0.850, "score_std": 0.020}),
            Option(id="median", label="Median", measured={"score_mean": 0.852, "score_std": 0.020}),
        ]
        best_id, reason, conf = advisor._rank_and_select(["mean", "median"], mock_opts, train_size=600, primary_reason="Testing", can_measure=True)
        has_close_call = (conf == "close call") and (best_id == "median")
    details["checks"].append({
        "name": "Close call flag and simplicity tie-breaker trigger when score delta is within CV std",
        "passed": has_close_call,
    })

    # -------------------------------------------------------------------------
    # Gate 4B: Fixer Verification
    # -------------------------------------------------------------------------
    # Build clean train / test splits for fixer tests
    train_raw = pd.DataFrame({
        "val": [10.0, 20.0, 30.0, np.nan, 50.0, np.nan],
        "cat": ["A", "B", "A", np.nan, "A", "C"],
        "target": [0, 0, 1, 1, 0, 1],
    })
    test_raw = pd.DataFrame({
        "val": [15.0, np.nan, 35.0, np.nan],
        "cat": ["A", np.nan, "C", "A"],
        "target": [0, 1, 1, 0],
    })

    # Check 1: After fix, 0 nulls remain in train and test (median fill)
    dec_med = Decision(issue_id="nulls", column="val", option_id="median", params={}, decided_by="recommended_accepted")
    fitted_med = fixer.fit(train_raw, dec_med)
    train_fixed = fixer.transform(train_raw, fitted_med, is_test=False)
    test_fixed = fixer.transform(test_raw, fitted_med, is_test=True)
    chk_zero_nulls = bool((train_fixed["val"].isna().sum() == 0) and (test_fixed["val"].isna().sum() == 0))
    details["checks"].append({
        "name": "Gate 4B: After fix, 0 nulls remain in train and test (imputation)",
        "passed": chk_zero_nulls,
    })

    # Check 2: Test row count is strictly unchanged
    chk_test_len = bool(len(test_fixed) == len(test_raw))
    details["checks"].append({
        "name": "Gate 4B: Test row count is strictly unchanged (0 test rows dropped)",
        "passed": chk_test_len,
    })

    # Check 3: Test values changed ONLY in cells that were null
    non_null_mask = test_raw["val"].notna()
    chk_unaffected_cells = bool((test_fixed.loc[non_null_mask, "val"].values == test_raw.loc[non_null_mask, "val"].values).all())
    details["checks"].append({
        "name": "Gate 4B: Test values changed only in cells that were originally null",
        "passed": chk_unaffected_cells,
    })

    # Check 4: Learned values come from train: build data where train median differs from full data median
    # Train: [10, 20, 30] -> median = 20.0
    # Full data including test: [10, 20, 30, 100, 100] -> median = 30.0
    train_skew_data = pd.DataFrame({"val": [10.0, 20.0, 30.0, np.nan]})
    test_skew_data = pd.DataFrame({"val": [100.0, 100.0, np.nan]})
    dec_leak_check = Decision(issue_id="nulls", column="val", option_id="median", params={}, decided_by="user")
    fitted_leak = fixer.fit(train_skew_data, dec_leak_check)
    transformed_test = fixer.transform(test_skew_data, fitted_leak, is_test=True)
    imputed_test_val = transformed_test["val"].iloc[2]
    chk_train_median_leak = bool(imputed_test_val == 20.0)  # Must equal train median (20), NOT combined (30)
    details["checks"].append({
        "name": f"Gate 4B: Learned values come strictly from train (expected 20.0, got {imputed_test_val})",
        "passed": chk_train_median_leak,
    })

    # Check 5: Decision log contains choice and reproduce.py rebuilds identical result
    pipeline = RemediationPipeline()
    pipeline.record_decision(dec_med, train_raw)
    log = pipeline.get_decision_log()
    has_log = len(log) == 1 and log[0]["column"] == "val" and log[0]["option_id"] == "median"

    reproduce_code = pipeline.export_reproduce_code()
    # Execute reproduce function locally in sandbox dict
    exec_scope = {"pd": pd, "np": np}
    exec(reproduce_code, exec_scope)
    clean_func = exec_scope.get("clean_dataset")
    reproduced_df = clean_func(train_raw, is_test=False)
    chk_reproduce = bool(has_log and (reproduced_df["val"].isna().sum() == 0) and (reproduced_df["val"].iloc[3] == fitted_med["fill_value"]))
    details["checks"].append({
        "name": "Gate 4B: Decision log persists and exported reproduce.py code rebuilds identical result",
        "passed": chk_reproduce,
    })

    # Check 6: User override successfully changes applied method
    dec_override = Decision(issue_id="nulls", column="val", option_id="constant", params={"value": -1.0}, decided_by="user")
    fitted_override = fixer.fit(train_raw, dec_override)
    train_overridden = fixer.transform(train_raw, fitted_override, is_test=False)
    chk_override = bool(train_overridden["val"].iloc[3] == -1.0)
    details["checks"].append({
        "name": "Gate 4B: User override successfully changes the applied remediation method",
        "passed": chk_override,
    })

    passed_count = sum(1 for c in details["checks"] if c["passed"])
    details["found"] = passed_count
    details["missed"] = len(details["checks"]) - passed_count
    gate_passed = all(c["passed"] for c in details["checks"])
    return gate_passed, details


def run_duplicates_gate() -> tuple[bool, dict]:
    """
    Executes Gate 5 evaluation for Duplicates (detect + advise + fix).
    Verifies:
    1. Planted: 50 exact, 20 near, 10 conflicting pairs. All exact and conflicting found; near ones flagged as possible.
    2. 0 false alarms on control (rows that look similar but differ in real values).
    3. After the fix, no duplicates remain and row count equals original minus planted exact duplicates.
    4. The pre-balanced warning fires on a file built by oversampling to 50/50, and does not fire on the control.
    """
    import numpy as np
    import pandas as pd
    from doctor.loader import load_csv
    from doctor.base import Decision, DetectionContext
    from doctor.issues.duplicates.detect import DuplicateDetector
    from doctor.issues.duplicates.advise import DuplicateAdvisor
    from doctor.issues.duplicates.fix import DuplicateFixer

    details = {"checks": [], "planted": 80, "found": 0, "missed": 0, "false_alarms": 0}
    detector = DuplicateDetector()
    advisor = DuplicateAdvisor()
    fixer = DuplicateFixer()

    dup_csv = ROOT_DIR / "data" / "planted" / "duplicates.csv"
    truth_json = ROOT_DIR / "data" / "planted" / "duplicates_truth.json"
    clean_csv = ROOT_DIR / "data" / "planted" / "clean_control.csv"
    pb_csv = ROOT_DIR / "data" / "planted" / "prebalanced_planted.csv"

    if not dup_csv.exists() or not truth_json.exists() or not clean_csv.exists() or not pb_csv.exists():
        from tools.make_planted import make_clean_control, make_duplicates_planted, make_prebalanced_planted
        if not clean_csv.exists():
            make_clean_control()
        make_duplicates_planted()
        make_prebalanced_planted()

    with open(truth_json, "r", encoding="utf-8") as f:
        truth = json.load(f)

    df_planted, _ = load_csv(str(dup_csv), filename="duplicates.csv")
    ctx = DetectionContext(target_column="target", task_type="classification")
    issues = detector.detect(df_planted, ctx=ctx)
    issues_by_kind = {i.evidence.get("kind"): i for i in issues if "kind" in i.evidence}

    # 1. 50 exact duplicates found
    exact_issue = issues_by_kind.get("exact")
    exact_found = exact_issue.evidence.get("surplus_copies", 0) if exact_issue else 0
    chk_exact = (exact_found == 50) and (exact_issue.certainty == "certain")
    details["checks"].append({
        "name": f"50 planted exact duplicates found as certain (found {exact_found})",
        "passed": chk_exact,
        "found": exact_found,
        "expected": 50,
    })

    # 2. 10 conflicting pairs found with high severity
    conf_issue = issues_by_kind.get("conflicting")
    conf_found = conf_issue.evidence.get("conflicting_pairs_count", 0) if conf_issue else 0
    chk_conf = (conf_found == 10) and (conf_issue.severity == "high") and (conf_issue.certainty == "certain")
    details["checks"].append({
        "name": f"10 planted conflicting pairs found with high severity (found {conf_found})",
        "passed": chk_conf,
        "found": conf_found,
        "expected": 10,
    })

    # 3. 20 near duplicates found and flagged strictly as 'possible'
    near_issue = issues_by_kind.get("near")
    near_found = near_issue.evidence.get("near_duplicate_count", 0) if near_issue else 0
    chk_near = (near_found == 20) and (near_issue.certainty == "possible")
    details["checks"].append({
        "name": f"20 planted near duplicates found and flagged as 'possible' (found {near_found})",
        "passed": chk_near,
        "found": near_found,
        "expected": 20,
    })

    # 4. 0 false alarms on clean_control.csv
    df_clean, _ = load_csv(str(clean_csv), filename="clean_control.csv")
    clean_issues = detector.detect(df_clean, ctx=DetectionContext(target_column="target", task_type="classification"))
    clean_duplicate_issues = [i for i in clean_issues if i.evidence.get("kind") in ("exact", "conflicting", "pre_balanced")]
    false_alarms = len(clean_duplicate_issues)
    chk_clean = false_alarms == 0
    details["checks"].append({
        "name": "0 false alarms on clean_control.csv (0 exact or conflicting duplicates)",
        "passed": chk_clean,
        "false_alarms": false_alarms,
    })

    # 5. Fixer test: After fix, no duplicate rows remain and row count equals original minus planted exact duplicates
    orig_len = len(df_planted)
    dec_first = Decision(issue_id="duplicates", column=None, option_id="keep_first", params={}, decided_by="recommended_accepted")
    fitted_first = fixer.fit(df_planted, dec_first)
    df_fixed = fixer.transform(df_planted, fitted_first, is_test=False)
    no_dups_remain = int(df_fixed.duplicated().sum()) == 0
    expected_fixed_len = orig_len - 50
    chk_fixer_exact = no_dups_remain and (len(df_fixed) == expected_fixed_len)
    details["checks"].append({
        "name": f"After keep_first fix, 0 duplicates remain and row count equals original minus 50 ({len(df_fixed)} == {expected_fixed_len})",
        "passed": chk_fixer_exact,
        "rows_after_fix": len(df_fixed),
        "expected_rows": expected_fixed_len,
    })

    # 6. Test set integrity: is_test=True never loses rows
    test_result = fixer.transform(df_planted, fitted_first, is_test=True)
    chk_test_safety = len(test_result) == orig_len
    details["checks"].append({
        "name": "Test set integrity: test rows are never dropped during deduplication",
        "passed": chk_test_safety,
    })

    # 7. Conflicting duplicates fixer: drop_all_conflicting removes ambiguous rows
    dec_conf = Decision(issue_id="duplicates", column="target", option_id="drop_all_conflicting", params={"target_column": "target"}, decided_by="user")
    fitted_conf = fixer.fit(df_planted, dec_conf)
    df_no_conf = fixer.transform(df_planted, fitted_conf, is_test=False)
    conf_check_after = detector.detect(df_no_conf, ctx=ctx)
    has_conf_after = any(i.evidence.get("kind") == "conflicting" for i in conf_check_after)
    chk_fixer_conf = not has_conf_after
    details["checks"].append({
        "name": "Conflicting rows fixer eliminates all conflicting label pairs",
        "passed": chk_fixer_conf,
    })

    # 8. Pre-balanced warning fires on 50/50 oversampled file
    df_pb = pd.read_csv(pb_csv)
    pb_issues = detector.detect(df_pb, ctx=DetectionContext(target_column="severe_accident", task_type="classification"))
    pb_warning_fired = any(i.evidence.get("kind") == "pre_balanced" for i in pb_issues)
    chk_pb_fire = pb_warning_fired is True
    details["checks"].append({
        "name": "Pre-balanced warning fires on 50/50 oversampled dataset",
        "passed": chk_pb_fire,
    })

    # 9. Pre-balanced warning does NOT fire on clean_control.csv
    pb_control_fired = any(i.evidence.get("kind") == "pre_balanced" for i in clean_issues)
    chk_pb_quiet = pb_control_fired is False
    details["checks"].append({
        "name": "Pre-balanced warning stays quiet on clean control",
        "passed": chk_pb_quiet,
    })

    details["planted"] = 80
    passed_count = sum(1 for c in details["checks"] if c["passed"])
    details["found"] = passed_count
    details["missed"] = len(details["checks"]) - passed_count
    details["false_alarms"] = false_alarms
    gate_passed = all(c["passed"] for c in details["checks"])
    return gate_passed, details


def run_types_gate() -> tuple[bool, dict]:
    """
    Executes Gate 6 evaluation for Data Types (detect + advise + fix).
    Verifies:
    1. All planted type problems found (text numbers, ambiguous dates, booleans, constants).
    2. 0 false alarms on control (no text numbers, ambiguous dates, constants, or mixed types).
    3. Ambiguous dates are flagged and user is asked, never guessed (certainty 'possible').
    4. Categorical column with codes like '001' is NOT turned into a number by mistake.
    5. After conversion, a numeric column has the same values as the planted truth.
    6. Boolean flags correctly convert to 0/1.
    7. Constant columns drop cleanly without changing test row counts.
    8. Standalone reproduce.py reproduces the exact cleaned types.
    """
    import json
    import re
    import numpy as np
    import pandas as pd
    from doctor.loader import load_csv
    from doctor.base import Decision, DetectionContext
    from doctor.issues.types.detect import TypeDetector
    from doctor.issues.types.advise import TypeAdvisor
    from doctor.issues.types.fix import TypeFixer
    from doctor.pipeline import RemediationPipeline

    details = {"checks": [], "planted": 7, "found": 0, "missed": 0, "false_alarms": 0}
    detector = TypeDetector()
    advisor = TypeAdvisor()
    fixer = TypeFixer()

    types_csv = ROOT_DIR / "data" / "planted" / "types.csv"
    truth_json = ROOT_DIR / "data" / "planted" / "types_truth.json"
    clean_csv = ROOT_DIR / "data" / "planted" / "clean_control.csv"

    if not types_csv.exists() or not truth_json.exists():
        from tools.make_planted import make_types_planted
        make_types_planted()
    if not clean_csv.exists():
        from tools.make_planted import make_clean_control
        make_clean_control()

    with open(truth_json, "r", encoding="utf-8") as f:
        truth = json.load(f)

    df_types, _ = load_csv(str(types_csv), filename="types.csv")
    ctx = DetectionContext(target_column="target", task_type="classification")
    issues = detector.detect(df_types, ctx=ctx)
    issues_by_col = {i.column: i for i in issues if i.column}

    # 1. Text numbers found (price_usd, discount_pct, weight_kg)
    text_num_cols = [col for col, iss in issues_by_col.items() if iss.evidence.get("kind") == "text_number"]
    chk_text_nums = all(c in text_num_cols for c in ["price_usd", "discount_pct", "weight_kg"])
    details["checks"].append({
        "name": f"All planted text numbers found ({', '.join(text_num_cols)})",
        "passed": bool(chk_text_nums),
        "found_columns": text_num_cols,
    })

    # 2. Ambiguous date flagged as possible (never guessed)
    ambig_iss = issues_by_col.get("ambiguous_event_date")
    chk_ambig = bool(ambig_iss and ambig_iss.evidence.get("kind") == "ambiguous_date" and ambig_iss.certainty == "possible")
    details["checks"].append({
        "name": "Ambiguous dates flagged and user is asked, never guessed (certainty=possible)",
        "passed": chk_ambig,
        "certainty": ambig_iss.certainty if ambig_iss else None,
    })

    # 3. Categorical code like '001' is NOT converted to number by mistake
    code_iss = issues_by_col.get("product_code")
    chk_code_safe = bool(code_iss is None or code_iss.evidence.get("kind") != "text_number")
    details["checks"].append({
        "name": "Categorical column with codes like '001' is NOT turned into a number by mistake",
        "passed": chk_code_safe,
    })

    # 4. Unambiguous date found as certain
    iso_iss = issues_by_col.get("iso_signup_date")
    chk_iso = bool(iso_iss and iso_iss.evidence.get("kind") == "date" and iso_iss.certainty == "certain")
    details["checks"].append({
        "name": "Unambiguous ISO date timestamp found with certainty=certain",
        "passed": chk_iso,
    })

    # 5. Boolean representation found as certain
    bool_iss = issues_by_col.get("subscribed_flag")
    chk_bool = bool(bool_iss and bool_iss.evidence.get("kind") == "boolean" and bool_iss.certainty == "certain")
    details["checks"].append({
        "name": "Yes/No boolean representation detected with certainty=certain",
        "passed": chk_bool,
    })

    # 6. Constant column found as high severity certain
    const_iss = issues_by_col.get("constant_status")
    chk_const = bool(const_iss and const_iss.evidence.get("kind") == "constant" and const_iss.severity == "high")
    details["checks"].append({
        "name": "Constant zero-variance column detected with severity=high",
        "passed": chk_const,
    })

    # 7. 0 false alarms on clean_control.csv for type defects
    df_clean, _ = load_csv(str(clean_csv), filename="clean_control.csv")
    clean_issues = detector.detect(df_clean, ctx=DetectionContext(target_column="target", task_type="classification"))
    type_defects_on_control = [
        iss for iss in clean_issues
        if iss.evidence.get("kind") in ("text_number", "ambiguous_date", "constant", "mixed_type")
    ]
    false_alarms = len(type_defects_on_control)
    chk_clean = (false_alarms == 0)
    details["checks"].append({
        "name": "0 false alarms on clean_control.csv (no false text numbers, ambiguous dates, or constants)",
        "passed": chk_clean,
        "false_alarms": false_alarms,
    })

    # 8. Advisor check on ambiguous date: requires user confirmation
    if ambig_iss:
        rec_ambig = advisor.advise(ambig_iss, df_types, target="target", ctx=ctx)
        chk_advisor_ambig = bool(rec_ambig.confidence == "close call" and any(o.id == "parse_date_day_first" for o in rec_ambig.options) and any(o.id == "parse_date_month_first" for o in rec_ambig.options))
    else:
        chk_advisor_ambig = False
    details["checks"].append({
        "name": "Advisor presents explicit Day-First and Month-First options for ambiguous dates",
        "passed": chk_advisor_ambig,
    })

    # 9. Fixer check: numeric column values match planted truth
    dec_price = Decision(issue_id="types", column="price_usd", option_id="convert_to_numeric", params={}, decided_by="recommended_accepted")
    fitted_price = fixer.fit(df_types, dec_price)
    fixed_types = fixer.transform(df_types, fitted_price, is_test=False)
    expected_price_5 = truth["expected_truth_values"]["price_usd_first_5"]
    actual_price_5 = fixed_types["price_usd"].iloc[:5].tolist()
    chk_numeric_match = bool(np.allclose(actual_price_5, expected_price_5))
    details["checks"].append({
        "name": "After convert_to_numeric, numeric column values match planted truth exactly",
        "passed": chk_numeric_match,
        "first_5_actual": actual_price_5,
        "first_5_expected": expected_price_5,
    })

    # 10. Fixer check: boolean encoding produces 0 and 1
    dec_bool = Decision(issue_id="types", column="subscribed_flag", option_id="encode_boolean", params={}, decided_by="recommended_accepted")
    fitted_bool = fixer.fit(df_types, dec_bool)
    fixed_bool = fixer.transform(df_types, fitted_bool, is_test=False)
    chk_bool_vals = bool(set(fixed_bool["subscribed_flag"].unique()).issubset({0, 1}))
    details["checks"].append({
        "name": "After encode_boolean, values are standard binary 0 and 1",
        "passed": chk_bool_vals,
    })

    # 11. Fixer check: constant column dropped cleanly
    dec_const = Decision(issue_id="types", column="constant_status", option_id="drop_column", params={"column": "constant_status"}, decided_by="recommended_accepted")
    fitted_const = fixer.fit(df_types, dec_const)
    fixed_const = fixer.transform(df_types, fitted_const, is_test=False)
    chk_const_dropped = bool("constant_status" not in fixed_const.columns and len(fixed_const) == len(df_types))
    details["checks"].append({
        "name": "After drop_column, constant column is removed without altering row count",
        "passed": chk_const_dropped,
    })

    # 12. Pipeline export: standalone reproduce code executes and produces identical clean types
    pipeline = RemediationPipeline()
    pipeline.record_decision(dec_price, df_types)
    pipeline.record_decision(dec_bool, df_types)
    pipeline.record_decision(dec_const, df_types)
    pipeline_clean_df = pipeline.transform(df_types, is_test=False)

    reproduce_code = pipeline.export_reproduce_code()
    exec_scope = {"pd": pd, "np": np, "re": re}
    exec(reproduce_code, exec_scope)
    clean_func = exec_scope.get("clean_dataset")
    reproduced_df = clean_func(df_types, is_test=False)
    chk_reproduce = bool(pipeline_clean_df.equals(reproduced_df))
    details["checks"].append({
        "name": "Decision log persists and exported reproduce.py code rebuilds identical clean dataset",
        "passed": chk_reproduce,
    })

    details["planted"] = 7
    passed_count = sum(1 for c in details["checks"] if c["passed"])
    details["found"] = passed_count
    details["missed"] = len(details["checks"]) - passed_count
    details["false_alarms"] = false_alarms
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
        elif mod.id == "nulls_advisor":
            passed, gate_details = run_nulls_advisor_gate()
            status = "gate_passed" if passed else "gate_failed"
            now_iso = datetime.now().isoformat()
            update_module_status("nulls_advisor", status, gate_details=gate_details, timestamp=now_iso)

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
        elif mod.id == "duplicates":
            passed, gate_details = run_duplicates_gate()
            status = "gate_passed" if passed else "gate_failed"
            now_iso = datetime.now().isoformat()
            update_module_status("duplicates", status, gate_details=gate_details, timestamp=now_iso)

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
        elif mod.id == "types":
            passed, gate_details = run_types_gate()
            status = "gate_passed" if passed else "gate_failed"
            now_iso = datetime.now().isoformat()
            update_module_status("types", status, gate_details=gate_details, timestamp=now_iso)

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
            default=lambda o: bool(o) if type(o).__name__ in ("bool_", "bool") else (int(o) if isinstance(o, (int, np.integer)) else (float(o) if isinstance(o, (float, np.floating)) else str(o))),
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
