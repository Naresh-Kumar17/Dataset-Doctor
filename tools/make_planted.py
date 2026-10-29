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


def main():
    print("Generating planted datasets...")
    make_clean_control()
    make_nulls_planted()
    make_duplicates_planted()
    make_prebalanced_planted()
    make_types_planted()
    print("All planted datasets successfully generated.")


if __name__ == "__main__":
    main()


