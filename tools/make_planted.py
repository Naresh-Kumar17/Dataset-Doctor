"""
Dataset Doctor: Planted-Problem and Clean Control Data Generator.
Creates deterministic ground-truth datasets for gate validation.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
PLANTED_DIR = ROOT_DIR / "data" / "planted"
PLANTED_DIR.mkdir(parents=True, exist_ok=True)


def make_clean_control():
    """
    Creates data/planted/clean_control.csv and truth JSON.
    Has NO real missing values, but includes tricky legitimate values like
    country code 'NA' (Namibia) and real '0' readings.
    A good detector must stay quiet (0 false alarms; 'NA' flagged at most as possible).
    """
    np.random.seed(42)
    n_rows = 500

    countries = ["NA", "US", "DE", "FR", "JP", "BR", "ZA", "IN", "CA", "AU"]
    country_col = [countries[i % len(countries)] for i in range(n_rows)]

    df_clean = pd.DataFrame({
        "id": np.arange(1, n_rows + 1),
        "customer_name": [f"Person_{i}" for i in range(1, n_rows + 1)],
        "country": country_col,
        "age": np.random.randint(18, 75, size=n_rows),
        "balance_change": np.random.choice([0.0, 10.5, -5.2, 0.0, 100.0, 0.0, 50.0], size=n_rows),
        "target": np.random.choice([0, 1], p=[0.8, 0.2], size=n_rows),
    })

    clean_csv = PLANTED_DIR / "clean_control.csv"
    df_clean.to_csv(clean_csv, index=False)

    clean_truth = {
        "dataset": "clean_control",
        "rows": n_rows,
        "expected_certain_issues": 0,
        "must_not_flag_certain": ["country", "id", "customer_name", "age", "balance_change", "target"],
        "allowed_possible": ["country"],  # 'NA' (Namibia) can be flagged at most as possible
    }

    clean_truth_path = PLANTED_DIR / "clean_control_truth.json"
    with open(clean_truth_path, "w", encoding="utf-8") as f:
        json.dump(clean_truth, f, indent=2)

    print(f"Created {clean_csv.name} ({n_rows} rows) and {clean_truth_path.name}")
    return clean_csv, clean_truth_path


def make_nulls_planted():
    """
    Creates data/planted/nulls.csv and truth JSON following IMPLEMENTATION.md section 6.1.
    Planted issues:
    - age: 50 true NaN
    - phone: 60 hidden tokens 'N/A'
    - score: 40 sentinels -999
    - notes: 700 true NaN (>60% -> high severity)
    - income: 100 missing values strongly correlated with target (target-related missingness)
    - noise_col: 50 missing values uniformly at random (independent of target)
    - country: legitimate countries including 'NA' (must not flag as certain)
    - id: sequence 1..1000 (must not flag)
    """
    np.random.seed(42)
    n_rows = 1000

    # 1. Base clean columns
    target = np.random.choice([0, 1], p=[0.8, 0.2], size=n_rows)  # 20% minority share
    ids = np.arange(1, n_rows + 1)
    countries = ["NA", "US", "DE", "FR", "JP", "GB"]
    country_col = [countries[i % len(countries)] for i in range(n_rows)]

    # 2. Age: 50 true NaN
    age = np.random.randint(20, 70, size=n_rows).astype(float)
    nan_age_idx = np.random.choice(n_rows, size=50, replace=False)
    age[nan_age_idx] = np.nan

    # 3. Phone: 60 hidden token 'N/A'
    phone = [f"555-01{i:03d}" for i in range(n_rows)]
    hidden_phone_idx = np.random.choice(n_rows, size=60, replace=False)
    for idx in hidden_phone_idx:
        phone[idx] = "N/A"

    # 4. Score: 40 sentinel -999
    score = np.random.randint(100, 900, size=n_rows)
    sentinel_score_idx = np.random.choice(n_rows, size=40, replace=False)
    score[sentinel_score_idx] = -999

    # 5. Notes: 700 true NaN (>60% -> high severity)
    notes = [f"Customer notes entry {i}" for i in range(n_rows)]
    nan_notes_idx = np.random.choice(n_rows, size=700, replace=False)
    notes = [None if i in nan_notes_idx else notes[i] for i in range(n_rows)]

    # 6. Income: Planted target-related missingness
    # When target == 1, 60% are missing. When target == 0, only 5% are missing.
    income = np.random.normal(55000, 12000, size=n_rows)
    for i in range(n_rows):
        if target[i] == 1:
            if np.random.rand() < 0.60:
                income[i] = np.nan
        else:
            if np.random.rand() < 0.05:
                income[i] = np.nan

    # 7. Noise_col: Planted purely random missingness (MCAR)
    noise_col = np.random.normal(10, 2, size=n_rows)
    random_miss_idx = np.random.choice(n_rows, size=50, replace=False)
    noise_col[random_miss_idx] = np.nan

    df = pd.DataFrame({
        "id": ids,
        "target": target,
        "country": country_col,
        "age": age,
        "phone": phone,
        "score": score,
        "notes": notes,
        "income": income,
        "noise_col": noise_col,
    })

    nulls_csv = PLANTED_DIR / "nulls.csv"
    df.to_csv(nulls_csv, index=False)

    truth = {
        "module": "nulls",
        "rows": n_rows,
        "planted": [
            {"column": "age", "kind": "nan", "count": 50},
            {"column": "phone", "kind": "hidden_token", "token": "N/A", "count": 60},
            {"column": "score", "kind": "sentinel", "token": "-999", "count": 40},
            {"column": "notes", "kind": "nan", "count": 700, "expected_severity": "high"},
            {"column": "income", "kind": "target_related_missingness", "expect_pattern": True},
            {"column": "noise_col", "kind": "random_missingness", "expect_pattern": False},
        ],
        "must_not_flag": ["id", "country"],
        "allowed_possible": ["country"],
    }

    truth_path = PLANTED_DIR / "nulls_truth.json"
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)

    print(f"Created {nulls_csv.name} ({n_rows} rows) and {truth_path.name}")
    return nulls_csv, truth_path


def make_duplicates_planted():
    """
    Creates data/planted/duplicates.csv and truth JSON following IMPLEMENTATION.md Gate 5:
    - 50 exact duplicate rows
    - 20 near-duplicate rows (differing only in text casing, whitespace, or tiny numeric differences)
    - 10 conflicting pairs (identical features, inverted target value)
    """
    rng = np.random.RandomState(42)
    n_base = 500

    cities = ["Berlin", "Tokyo", "Paris", "New York", "London", "Toronto", "Sydney", "Singapore"]
    base_df = pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(1, n_base + 1)],
        "name": [f"Customer_{i}" for i in range(1, n_base + 1)],
        "city": [cities[i % len(cities)] for i in range(n_base)],
        "score": rng.randint(200, 850, size=n_base),
        "amount": np.round(rng.uniform(10.0, 500.0, size=n_base), 2),
        "target": rng.choice([0, 1], p=[0.75, 0.25], size=n_base),
    })

    # 1. 50 exact duplicate rows: randomly select 50 rows and copy exactly
    exact_idx = rng.choice(n_base, size=50, replace=False)
    exact_rows = base_df.iloc[exact_idx].copy()

    # 2. 20 near duplicates: select 20 distinct rows and introduce minor variations
    near_idx = rng.choice([i for i in range(n_base) if i not in exact_idx], size=20, replace=False)
    near_rows = base_df.iloc[near_idx].copy()
    # 10 rows with case / whitespace variations
    for i in range(10):
        orig_row = near_rows.iloc[i]
        near_rows.iloc[i, near_rows.columns.get_loc("name")] = orig_row["name"].upper()
        near_rows.iloc[i, near_rows.columns.get_loc("city")] = f"  {orig_row['city']}  "
    # 10 rows with tiny float fuzz (< 1e-4) that rounds to same 4 decimals
    for i in range(10, 20):
        orig_row = near_rows.iloc[i]
        near_rows.iloc[i, near_rows.columns.get_loc("amount")] = round(orig_row["amount"] + 0.00003, 5)

    # 3. 10 conflicting pairs: select 10 rows, duplicate all features identically, but flip target!
    conflicting_idx = rng.choice([i for i in range(n_base) if i not in exact_idx and i not in near_idx], size=10, replace=False)
    conflicting_rows = base_df.iloc[conflicting_idx].copy()
    conflicting_rows["target"] = 1 - conflicting_rows["target"]

    # Combine into full planted dataset
    # Total rows = 500 (base) + 50 (exact) + 20 (near) + 10 (conflicting) = 580 rows
    full_df = pd.concat([base_df, exact_rows, near_rows, conflicting_rows], ignore_index=True)

    dup_csv = PLANTED_DIR / "duplicates.csv"
    full_df.to_csv(dup_csv, index=False)

    truth = {
        "module": "duplicates",
        "rows": len(full_df),
        "base_rows": n_base,
        "planted": {
            "exact_duplicate_surplus": 50,
            "near_duplicates": 20,
            "conflicting_pairs": 10,
        },
        "must_not_flag_on_control": True,
    }

    truth_path = PLANTED_DIR / "duplicates_truth.json"
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)

    print(f"Created {dup_csv.name} ({len(full_df)} rows) and {truth_path.name}")
    return dup_csv, truth_path


def make_prebalanced_planted():
    """
    Creates a pre-balanced dataset by synthetically oversampling an imbalanced dataset
    to exactly 50/50 balance before any train/test split.
    """
    sample_csv = ROOT_DIR / "data" / "samples" / "accident.csv"
    if sample_csv.exists():
        df = pd.read_csv(sample_csv)
    else:
        rng = np.random.RandomState(42)
        n = 1000
        df = pd.DataFrame({
            "feat1": rng.normal(0, 1, size=n),
            "feat2": rng.randint(0, 10, size=n),
            "severe_accident": rng.choice([0, 1], p=[0.85, 0.15], size=n),
        })

    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X = df.drop(columns=["severe_accident"])
    y = df["severe_accident"]
    X_res, y_res = ros.fit_resample(X, y)
    df_prebalanced = pd.concat([X_res, y_res], axis=1)

    pb_csv = PLANTED_DIR / "prebalanced_planted.csv"
    df_prebalanced.to_csv(pb_csv, index=False)
    print(f"Created {pb_csv.name} ({len(df_prebalanced)} rows, 50/50 balanced)")
    return pb_csv


def make_types_planted():
    """
    Creates data/planted/types.csv and truth JSON following IMPLEMENTATION.md Gate 6:
    - Numbers stored as text: "$1,200.50", "15%", "12 kg"
    - Ambiguous dates: "03/04/2021" (both day and month <= 12)
    - Unambiguous dates: "2021-04-15" (ISO format)
    - Yes/no booleans: "yes", "no"
    - Constant column: "ACTIVE" across all rows
    - Code categorical with leading zeros: "001", "007" (MUST NOT convert to number!)
    - Mixed type: numbers plus words ("100", "cancelled", etc.)
    """
    rng = np.random.RandomState(42)
    n_rows = 500

    # 1. Base ID and target
    ids = np.arange(1, n_rows + 1)
    target = rng.choice([0, 1], p=[0.75, 0.25], size=n_rows)

    # 2. Text numbers: Currency ($ and comma)
    base_prices = np.round(rng.uniform(10.0, 2500.0, size=n_rows), 2)
    price_usd = [f"${p:,.2f}" for p in base_prices]

    # 3. Text numbers: Percentages (%)
    base_discounts = rng.choice([5.0, 10.0, 15.0, 20.0, 25.0, 50.0], size=n_rows)
    discount_pct = [f"{int(d)}%" for d in base_discounts]

    # 4. Text numbers: Units (kg)
    base_weights = np.round(rng.uniform(1.0, 50.0, size=n_rows), 1)
    weight_kg = [f"{w} kg" for w in base_weights]

    # 5. Ambiguous dates: e.g. "03/04/2021" (day <= 12 and month <= 12)
    ambig_months = rng.choice(np.arange(1, 13), size=n_rows)
    ambig_days = rng.choice(np.arange(1, 13), size=n_rows)
    ambig_years = rng.choice([2020, 2021, 2022, 2023], size=n_rows)
    ambiguous_event_date = [f"{d:02d}/{m:02d}/{y}" for d, m, y in zip(ambig_days, ambig_months, ambig_years)]

    # 6. Unambiguous dates: ISO format "YYYY-MM-DD"
    iso_months = rng.choice(np.arange(1, 13), size=n_rows)
    iso_days = rng.choice(np.arange(1, 29), size=n_rows)
    iso_years = rng.choice([2021, 2022, 2023], size=n_rows)
    iso_signup_date = [f"{y}-{m:02d}-{d:02d}" for y, m, d in zip(iso_years, iso_months, iso_days)]

    # 7. Yes/No booleans
    subscribed_flag = rng.choice(["yes", "no", "yes", "no", "yes"], size=n_rows)

    # 8. Constant column (zero variance)
    constant_status = ["ACTIVE"] * n_rows

    # 9. Categorical codes with leading zeros (e.g. "001", "002", "007")
    # Must NOT be converted to numeric!
    code_numbers = rng.choice([1, 2, 7, 15, 23, 99], size=n_rows)
    product_code = [f"{c:03d}" for c in code_numbers]

    # 10. Mixed type column (numbers + text labels)
    mixed_options = ["100", "250", "50", "cancelled", "unknown", "pending", "500"]
    mixed_notes = rng.choice(mixed_options, size=n_rows)

    df_types = pd.DataFrame({
        "id": ids,
        "price_usd": price_usd,
        "discount_pct": discount_pct,
        "weight_kg": weight_kg,
        "ambiguous_event_date": ambiguous_event_date,
        "iso_signup_date": iso_signup_date,
        "subscribed_flag": subscribed_flag,
        "constant_status": constant_status,
        "product_code": product_code,
        "mixed_notes": mixed_notes,
        "target": target,
    })

    types_csv = PLANTED_DIR / "types.csv"
    df_types.to_csv(types_csv, index=False)

    truth = {
        "module": "types",
        "rows": n_rows,
        "planted": {
            "text_numbers": ["price_usd", "discount_pct", "weight_kg"],
            "ambiguous_dates": ["ambiguous_event_date"],
            "unambiguous_dates": ["iso_signup_date"],
            "booleans": ["subscribed_flag"],
            "constants": ["constant_status"],
            "mixed_types": ["mixed_notes"],
        },
        "protected_categoricals": ["product_code"],
        "must_not_flag_on_control": ["text_number", "ambiguous_date", "constant", "mixed_type"],
        "expected_truth_values": {
            "price_usd_first_5": base_prices[:5].tolist(),
            "discount_pct_first_5": [float(d) for d in base_discounts[:5]],
            "weight_kg_first_5": base_weights[:5].tolist(),
            "subscribed_flag_first_5": [1 if v == "yes" else 0 for v in subscribed_flag[:5]],
        },
    }

    truth_path = PLANTED_DIR / "types_truth.json"
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)

    print(f"Created {types_csv.name} ({n_rows} rows) and {truth_path.name}")
    return types_csv, truth_path


def make_outliers_planted():
    """
    Creates data/planted/outliers.csv and truth JSON following IMPLEMENTATION.md section 7 & 8.
    Planted issues:
    - age: 10 impossible values (-5, -12, 145, 210, 999, etc.) -> severity=high
    - height_cm: 1 unit error (170,000 cm - 1000x median mix-up) -> severity=medium
    - extreme_salary: 20 extreme-but-real values (high valid earners, 10 aligned with minority target)
    - clean_skewed_income: heavily skewed clean log-normal column -> must not flag > 5% as outliers
    - clean_gaussian: clean normal column N(100, 15) -> must stay under 1% flagged
    - department: categorical column -> 0 flags
    - is_active: binary column -> 0 flags
    - target: imbalanced binary classification (10% minority)
    """
    rng = np.random.default_rng(42)
    n_rows = 600

    # 1. Target with 10% minority
    target = np.zeros(n_rows, dtype=int)
    minority_indices = rng.choice(n_rows, size=60, replace=False)
    target[minority_indices] = 1

    # 2. Age: 10 impossible values
    age = rng.integers(21, 68, size=n_rows).astype(float)
    impossible_age_indices = [10, 25, 40, 80, 120, 180, 240, 310, 420, 500]
    impossible_age_values = [-5.0, -12.0, 145.0, 210.0, 999.0, -1.0, 150.0, 300.0, -99.0, 130.0]
    for idx, val in zip(impossible_age_indices, impossible_age_values):
        age[idx] = val

    # 3. Height in cm: 1 unit mix-up (170,000 mm entered instead of 170 cm)
    height_cm = np.round(rng.normal(170.0, 7.5, size=n_rows), 1)
    unit_error_index = 35
    height_cm[unit_error_index] = 170000.0  # 1000x median

    # 4. Extreme salary: 20 extreme-but-real values
    # Base salaries ~ 40,000 to 90,000
    base_salary = np.round(rng.normal(55000.0, 12000.0, size=n_rows), 2)
    base_salary = np.clip(base_salary, 28000.0, 98000.0)

    # Pick 20 extreme rows, 10 of which are in minority class
    avail_min = [i for i in minority_indices if i not in impossible_age_indices and i != unit_error_index]
    avail_maj = [i for i in range(n_rows) if i not in minority_indices and i not in impossible_age_indices and i != unit_error_index]

    extreme_indices = list(avail_min[:10]) + list(avail_maj[:10])
    extreme_values = [
        260000.0, 285000.0, 310000.0, 330000.0, 355000.0,
        375000.0, 390000.0, 415000.0, 430000.0, 450000.0,
        270000.0, 290000.0, 315000.0, 340000.0, 360000.0,
        380000.0, 400000.0, 420000.0, 440000.0, 460000.0
    ]
    for idx, val in zip(extreme_indices, extreme_values):
        base_salary[idx] = val

    # 5. Heavily skewed but clean log-normal income
    clean_skewed_income = np.round(rng.lognormal(mean=9.5, sigma=0.85, size=n_rows), 2)

    # 6. Clean Gaussian column
    clean_gaussian = np.round(rng.normal(loc=100.0, scale=15.0, size=n_rows), 2)

    # 7. Categorical column (must never flag)
    depts = ["Sales", "Engineering", "Marketing", "HR", "Support", "Finance"]
    department = [depts[i % len(depts)] for i in range(n_rows)]

    # 8. Binary column (must never flag)
    is_active = (rng.uniform(size=n_rows) > 0.3).astype(int)

    df_outliers = pd.DataFrame({
        "id": np.arange(1, n_rows + 1),
        "age": age,
        "height_cm": height_cm,
        "extreme_salary": base_salary,
        "clean_skewed_income": clean_skewed_income,
        "clean_gaussian": clean_gaussian,
        "department": department,
        "is_active": is_active,
        "target": target,
    })

    outliers_csv = PLANTED_DIR / "outliers.csv"
    df_outliers.to_csv(outliers_csv, index=False)

    truth = {
        "module": "outliers",
        "rows": n_rows,
        "planted": {
            "impossible_age": {
                "column": "age",
                "count": 10,
                "indices": [int(i) for i in impossible_age_indices],
                "values": [float(v) for v in impossible_age_values],
                "expected_severity": "high",
            },
            "unit_error_height": {
                "column": "height_cm",
                "count": 1,
                "index": int(unit_error_index),
                "value": 170000.0,
                "expected_severity": "medium",
            },
            "extreme_salary": {
                "column": "extreme_salary",
                "count": 20,
                "indices": [int(i) for i in extreme_indices],
                "minority_target_count": 10,
                "expected_severity": "medium",
            },
        },
        "clean_controls": {
            "clean_skewed_income": {"max_flag_pct": 0.05},
            "clean_gaussian": {"max_flag_pct": 0.01},
            "department": {"max_flag_pct": 0.0},
            "is_active": {"max_flag_pct": 0.0},
        },
    }

    truth_path = PLANTED_DIR / "outliers_truth.json"
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer, int)) else float(o))

    print(f"Created {outliers_csv.name} ({n_rows} rows) and {truth_path.name}")
    return outliers_csv, truth_path


def make_imbalance_planted():
    """
    Creates data/planted/imbalance.csv and truth JSON following IMPLEMENTATION.md Gate 9.
    Planted scenarios:
    - target_90_10: 90% class 0, 10% class 1 (minority share = 0.10, ratio = 9.0, severity = high)
    - target_99_1: 99% class 0, 1% class 1 (minority share = 0.01, ratio = 99.0, severity = high)
    - target_multiclass: 70% A, 25% B, 5% C (minority share = 0.05, severity = high)
    - target_balanced: 50% class 0, 50% class 1 (control: balanced, 0 issues)
    - feature_cat: unencoded categorical string column (tier_1, tier_2, tier_3) to test SMOTE safety
    """
    np.random.seed(42)
    n_rows = 1000

    # 1. Informative features
    f1 = np.random.normal(0, 1, size=n_rows)
    f2 = np.random.normal(5, 2, size=n_rows)
    f_cat = np.random.choice(["tier_1", "tier_2", "tier_3"], size=n_rows)

    # 2. Planted targets
    # 90/10 target: 900 zeros, 100 ones
    target_90_10 = np.array([0] * 900 + [1] * 100)
    # 99/1 target: 990 zeros, 10 ones
    target_99_1 = np.array([0] * 990 + [1] * 10)
    # Multiclass target: 700 A, 250 B, 50 C
    target_multiclass = np.array(["A"] * 700 + ["B"] * 250 + ["C"] * 50)
    # Balanced control: 500 zeros, 500 ones
    target_balanced = np.array([0] * 500 + [1] * 500)

    # Permute indices together so features stay matched
    perm = np.random.permutation(n_rows)

    df_imbalance = pd.DataFrame({
        "feature_1": f1[perm],
        "feature_2": f2[perm],
        "feature_cat": f_cat[perm],
        "target_90_10": target_90_10[perm],
        "target_99_1": target_99_1[perm],
        "target_multiclass": target_multiclass[perm],
        "target_balanced": target_balanced[perm],
    })

    imbalance_csv = PLANTED_DIR / "imbalance.csv"
    df_imbalance.to_csv(imbalance_csv, index=False)

    truth = {
        "module": "imbalance",
        "rows": n_rows,
        "planted": {
            "target_90_10": {
                "column": "target_90_10",
                "minority_share": 0.10,
                "imbalance_ratio": 9.0,
                "expected_severity": "high",
                "is_imbalanced": True,
            },
            "target_99_1": {
                "column": "target_99_1",
                "minority_share": 0.01,
                "imbalance_ratio": 99.0,
                "expected_severity": "high",
                "is_imbalanced": True,
            },
            "target_multiclass": {
                "column": "target_multiclass",
                "minority_share": 0.05,
                "imbalance_ratio": 14.0,
                "expected_severity": "high",
                "is_imbalanced": True,
            },
            "target_balanced": {
                "column": "target_balanced",
                "minority_share": 0.50,
                "imbalance_ratio": 1.0,
                "expected_severity": "none",
                "is_imbalanced": False,
            },
        },
    }

    truth_path = PLANTED_DIR / "imbalance_truth.json"
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer, int)) else float(o))

    print(f"Created {imbalance_csv.name} ({n_rows} rows) and {truth_path.name}")
    return imbalance_csv, truth_path




def make_leakage_planted():
    """
    Creates data/planted/leakage.csv and truth JSON following Gate 11 specifications.
    Planted leakage defects:
    - target_copy: Direct clone of target (correlation = 1.0, severity = high)
    - target_scaled: Scalar linear transform y * 2.5 + 10 (correlation = 1.0, severity = high)
    - target_inverted: Inverted binary clone 1 - y (correlation = -1.0, severity = high)
    - id_leak: Row index/sequence correlating 0.96 with target (severity = high)
    - payout_amount: Post-event feature correlating 0.99 with target (severity = high)
    - legit_strong: Legitimate honest feature correlating ~0.68 (must NOT be flagged high)
    - timestamp_col: Monotonic increasing time series (triggers time-split warning)
    - noise_1, noise_2: Random Gaussian noise controls (zero correlation)
    """
    np.random.seed(42)
    n_rows = 1000

    # 1. Base classification target (70% class 0, 30% class 1)
    target = np.random.choice([0, 1], p=[0.70, 0.30], size=n_rows)

    # 2. Planted leakage columns
    target_copy = target.copy()
    target_scaled = target * 2.5 + 10.0
    target_inverted = 1 - target

    # Row index leak: sorted sequence correlating > 0.95 with target
    id_leak = target * 1000.0 + np.random.normal(0, 80, size=n_rows)

    # Post-event payout amount: only positive when target=1 with slight noise (r > 0.98)
    payout_amount = np.where(target == 1, 5000.0 + np.random.normal(0, 50, size=n_rows), 0.0)

    # Legitimate strong feature: honest correlation ~ 0.68
    legit_strong = 1.8 * target + np.random.normal(0, 1.0, size=n_rows)

    # Noise controls
    noise_1 = np.random.normal(0, 1.0, size=n_rows)
    noise_2 = np.random.normal(10, 2.5, size=n_rows)

    # Monotonic timestamps
    base_time = pd.Timestamp("2024-01-01 08:00:00")
    timestamps = [str(base_time + pd.Timedelta(minutes=15 * i)) for i in range(n_rows)]

    df_leakage = pd.DataFrame({
        "id": np.arange(1, n_rows + 1),
        "target": target,
        "target_copy": target_copy,
        "target_scaled": target_scaled,
        "target_inverted": target_inverted,
        "id_leak": id_leak,
        "payout_amount": payout_amount,
        "legit_strong": legit_strong,
        "noise_1": noise_1,
        "noise_2": noise_2,
        "timestamp_col": timestamps,
    })

    leakage_csv = PLANTED_DIR / "leakage.csv"
    df_leakage.to_csv(leakage_csv, index=False)

    truth = {
        "module": "leakage",
        "rows": n_rows,
        "target_column": "target",
        "planted": {
            "target_copy": {
                "column": "target_copy",
                "subtype": "target_copy",
                "expected_severity": "high",
                "expected_certainty": "certain",
                "is_leak": True,
            },
            "target_scaled": {
                "column": "target_scaled",
                "subtype": "target_scaled",
                "expected_severity": "high",
                "expected_certainty": "certain",
                "is_leak": True,
            },
            "target_inverted": {
                "column": "target_inverted",
                "subtype": "target_inverted",
                "expected_severity": "high",
                "expected_certainty": "certain",
                "is_leak": True,
            },
            "id_leak": {
                "column": "id_leak",
                "subtype": "index_leak",
                "expected_severity": "high",
                "expected_certainty": "certain",
                "is_leak": True,
            },
            "payout_amount": {
                "column": "payout_amount",
                "subtype": "perfect_predictor",
                "expected_severity": "high",
                "expected_certainty": "certain",
                "is_leak": True,
            },
            "legit_strong": {
                "column": "legit_strong",
                "subtype": "legitimate_feature",
                "expected_severity": "none",
                "is_leak": False,
            },
            "timestamp_col": {
                "column": "timestamp_col",
                "subtype": "time_order_risk",
                "expected_severity": "medium",
                "is_leak": True,
            },
        },
    }

    truth_path = PLANTED_DIR / "leakage_truth.json"
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer, int)) else float(o))

    print(f"Created {leakage_csv.name} ({n_rows} rows) and {truth_path.name}")
    return leakage_csv, truth_path

def main():
    print("Generating planted datasets...")
    make_clean_control()
    make_nulls_planted()
    make_duplicates_planted()
    make_prebalanced_planted()
    make_types_planted()
    make_outliers_planted()
    make_imbalance_planted()
    make_leakage_planted()
    print("All planted datasets successfully generated.")


if __name__ == "__main__":
    main()



