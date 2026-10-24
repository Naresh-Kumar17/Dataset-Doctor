# GitHub Issue: Phase 4 — Null Advisor + Fixer (Interactive Remediation Engine)

## Title
`[Phase 4] Null Advisor + Fixer: Rule-Based Reasoning, CV-Measured Remediation, Interactive Decision Cards & Leak-Free Transformation`

---

## 1. Description & Goal
Build the end-to-end remediation engine for missing values. For each column with nulls, hidden tokens, or confirmed sentinels, the system:
1. **Recommends** a method based on data properties and target signal.
2. **Explains why** using plain-English reasoning and rules that fired.
3. **Tests alternative options** using honest 5-fold cross-validation on **train data only**.
4. **Presents interactive decision cards** in the web UI with before/after previews.
5. **Applies choices leak-free** (learning statistics strictly on train, applying to train and test).
6. **Records every decision** in a persistent decision log for auditability and `reproduce.py` pipeline generation.

---

## 2. Supported Fix Options
| Option ID | Operation | Target Partition |
| :--- | :--- | :--- |
| `drop_rows` | Drop rows containing missing values in this column | **Train set only** (never drops test rows) |
| `drop_column` | Drop the entire column | Train and Test |
| `mean` | Impute with training set mean | Train and Test (learned from train) |
| `median` | Impute with training set median | Train and Test (learned from train) |
| `mode` | Impute with most frequent training value | Train and Test (learned from train) |
| `constant` | Impute with user-specified constant (e.g. `0`, `"Unknown"`) | Train and Test |
| `missing_label` | Impute categorical nulls with `"MISSING"` category | Train and Test |
| `ffill_interpolate` | Forward-fill or linear interpolation for time-series | Train and Test |
| `add_indicator` | Add binary `was_missing` indicator column + imputation | Train and Test |
| `leave` | Preserve NaNs for models with native missing support | Train and Test |

---

## 3. Step A: Rule-Based Shortlist Logic
- **Target is missing**: Shortlist strictly `drop_rows` (cannot learn from unlabelled samples).
- **Column > 60% missing**: Shortlist `drop_column`, `add_indicator`.
- **Missing < 5% of rows, independent of target**: Shortlist `drop_rows`, `median`/`mode`.
- **Numeric & Skewed (|skew| > 1 or outliers)**: Prioritize `median` over `mean`.
- **Numeric & Symmetric (|skew| <= 1)**: Shortlist `mean`, `median`.
- **Categorical / Text**: Shortlist `mode`, `missing_label`.
- **Missingness linked to target (p < 0.05)**: Mandate `add_indicator` + fill; **exclude** `drop_rows`.
- **Time-ordered**: Shortlist `ffill_interpolate`.
- **High row-loss guard**: If dropping rows would lose > 20% of dataset, exclude `drop_rows`.

---

## 4. Step B: Measurement & Ranking
- Evaluate shortlisted options using 5-fold CV on **train set only**.
- Record primary metric mean and standard deviation, plus rows retained.
- If score difference between top 2 options is within 1 std: mark confidence as **"close call"** and pick the simpler option (simplicity hierarchy: `leave` < `median`/`mode` < `mean` < `constant`/`missing_label` < `add_indicator` < `drop_rows` < `drop_column`).
- If dataset < 500 rows: mark confidence as **"not enough data"** and default to rule-based choice.

---

## 5. Step C: Interactive Decision Cards (UI)
- Diagnostics summary and badge showing detected null counts and patterns.
- Recommended option with plain-language rationale and triggered rules.
- Comparison table displaying CV score, std dev, and rows retained for all options.
- Action buttons: **[APPLY RECOMMENDED]**, **[CHOOSE ANOTHER]**, **[SKIP]**.
- Interactive **10-Row Before / After Preview** displaying exact cell changes.
- **"Apply Recommended to All"** batch action with confirmation modal.

---

## 6. Step D: Leak-Free Fitting & Pipeline
- `fit(train_df, decision)` computes parameters exclusively from `train_df`.
- `transform(df, fitted_params)` applies transformations to train or test.
- Every decision appends to `decision_log` with timestamp and user override tracking.

---

## 7. Acceptance Criteria (Gates 4A & 4B)

### Gate 4A: Advisor Evaluation
- [x] Planted scenario 1: Numeric, 3% random missing, symmetric -> recommends `mean` or `median`.
- [x] Planted scenario 2: Numeric, 10% missing, heavily skewed -> recommends `median`.
- [x] Planted scenario 3: Categorical, 5% missing -> recommends `mode` or `missing_label`.
- [x] Planted scenario 4: Column 75% missing -> recommends `drop_column` or `add_indicator`.
- [x] Planted scenario 5: Missingness depends on target -> includes `add_indicator`, excludes `drop_rows`.
- [x] At least 4 of 5 recommendations match ground truth (5/5 passed).
- [x] Every recommendation provides rationale and measured CV scores.
- [x] "Close call" flag triggers when score delta is smaller than CV std.

### Gate 4B: Fixer Verification
- [x] After fix: 0 missing values remain in train and test (except `leave`).
- [x] Test row count is strictly unchanged (0 test rows dropped).
- [x] Test cell modifications occur only in cells that were originally missing.
- [x] Imputed values equal training statistics (verifying full partition isolation).
- [x] User overrides successfully apply alternative remediation.
- [x] Decision log persists choices and feeds `reproduce.py` and `cleaned_dataset.csv` export.

---

## 8. Development Workflow
1. **Issue Creation**: Filed as Phase 4 specification.
2. **Branch**: `phase/04-null-advisor-fixer` branched from updated `main`.
3. **Implementation**: Advisor, Fixer, Decision routes, UI cards, and Gate runner.
4. **Testing**: Unit tests and automated benchmark `python -m doctor.bench nulls_advisor`.
5. **Pull Request**: Submitted to `main` with Gate 4 verification summary.
