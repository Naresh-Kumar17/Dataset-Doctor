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


def main():
    print("Generating planted datasets...")
    make_clean_control()
    make_nulls_planted()
    print("All planted datasets successfully generated.")


if __name__ == "__main__":
    main()
