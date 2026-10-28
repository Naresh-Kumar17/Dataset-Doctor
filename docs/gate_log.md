# Gate Log

## 2026-10-19  Phase 0: Setup and evidence   PASS
- pytest -q runs: 1 passed (tests/test_hello.py verified FastAPI hello endpoint)
- FastAPI root endpoint verified: `{"status": "Dataset Doctor is running"}`
- Baseline evidence established on road accident dataset (2,000 rows, 85/15 ratio):
  - Flawed (oversample before split): Accuracy 0.9574, Severe Recall 0.9941, Severe F1 0.9589
  - Honest (split first, no fix): Accuracy 0.8700, Severe Recall 0.2167, Severe F1 0.3333
  - Honest (split first, class weights): Accuracy 0.8375, Severe Recall 0.4000, Severe F1 0.4248
  - Honest (split first, oversample train only): Accuracy 0.8500, Severe Recall 0.3167, Severe F1 0.3878
- Test set preserved clean class distribution: 85% class 0 / 15% class 1 (340 vs 60 rows)
- Documented 3-line explanation of test leakage from pre-split balancing
- Competitive analysis recorded in docs/notes.md

## 2026-10-19  Phase 1: GUI shell + loader + profile   PASS
- Loader verified:
  - Accident dataset matches pandas row count (2,000), column count (9), and dtypes exactly.
  - Latin-1 encoded file decoded and loaded accurately.
  - 60 MB file size limit enforced with clear error message (FileTooLargeError).
  - Empty files (0-byte / whitespace) and header-only files handled with friendly errors (EmptyFileError, NoDataRowsError) with zero crash.
  - keep_default_na=False verified: valid tokens like 'NA' preserved without silent NaN conversion.
- Profile verified:
  - Computes row count, column count, memory usage, missing column counts, dtypes, unique counts, sample values, and 10-row preview.
  - Correctly suggested target column ('severe_accident') and task type ('classification').
- GUI Shell and Routes verified:
  - Neo-brutalist styling, responsive layout (desktop & 375px mobile), sidebar registry navigation.
  - Routes /, /upload, /profile, /target, /module/{id}, and /bench verified via automated tests (17 passed).
  - Test Bench page (/bench) opens and lists all modules with status chips.

## 2026-10-21  Phase 2: Null detection   PASS
- Automated benchmark runner verified via `python -m doctor.bench nulls` and `python -m doctor.bench --all`:
  - 100% of planted missing cells found:
    - 50 true NaNs in 'age' found with exact count.
    - 60 hidden tokens ('N/A') in 'phone' found with exact count.
    - 40 sentinel values ('-999') in 'score' found and classified strictly as 'possible'.
    - 700 NaNs in 'notes' found with 'high' severity (>60% missing threshold).
  - 0 false alarms on clean_control.csv:
    - 0 certain issues on clean control.
    - Tricky country code 'NA' (Namibia) flagged at most as 'possible'.
  - Exact row-loss number confirmed:
    - 796 rows lost (79.6%) exactly matches dropping all true missing values and confirmed sentinels.
    - Row breakdown: 17 rows with 1 null, 66 rows with 2 nulls, 713 rows with 3+ nulls.
  - Missingness pattern statistical tests verified:
    - Planted target-related missingness ('income') detected and reported (Chi-square p < 0.05).
    - Planted random missingness ('noise_col') correctly identified as not related (p >= 0.05).
  - High-performance benchmark verified:
    - 100,000 rows x 5 columns scanned in under 0.35s (threshold < 5.0s).
- Full Phase 2 test suite: 10 passed in `tests/issues/test_nulls_detect.py`.
- Neo-brutalist GUI diagnostic card & table added to `/module/nulls` with summary strip, row-loss impact banner, and evidence badges.

## 2026-10-21  Phase 3: Split + evaluation engine   PASS
- Automated benchmark runner verified via `python -m doctor.bench split_eval` and `python -m doctor.bench --all`:
  - 7 of 7 Gate 3 checks PASSED:
    - [PASS] Stratified split preserves class ratio within 1% on train and test (exact 85/15 ratio preserved).
    - [PASS] Deterministic split verified (same seed yields identical partition; different seed yields different partition).
    - [PASS] Zero partition overlap verified (strict set intersection check).
    - [PASS] Synthetic leakage detected: pre-split balancing inflates recall (0.9941 vs honest 0.3167).
    - [PASS] Always-predict-majority baseline verified (~85% accuracy, 0 minority recall, accuracy not reliable banner).
    - [PASS] Delta smaller than CV std labelled 'within noise'.
    - [PASS] Dataset < 500 rows generates noisy comparison warning.
- Full Phase 3 test suite: 13 passed in `tests/test_split.py` and `tests/test_evaluate.py`.
- Regression check: all 46 tests passed across the entire project.
- Neo-brutalist GUI dashboard integrated at `/module/split_eval` with split overview stat grid, imbalance alert banner, baseline CV table, and synthetic leakage demonstration card.

## 2026-10-23  Phase 4: Null advisor + fixer, interactive   PASS
- Automated benchmark runner verified via `python -m doctor.bench nulls_advisor` and `python -m doctor.bench --all`:
  - 14 of 14 Gate 4 checks PASSED:
    - Gate 4A (Advisor Ground Truth):
      - [PASS] Scenario 1 (Numeric, 3% random missing, symmetric): recommends 'median' or 'mean'.
      - [PASS] Scenario 2 (Numeric, 10% missing, heavily skewed): recommends 'median'.
      - [PASS] Scenario 3 (Categorical, 5% missing): recommends 'mode' or 'missing_label'.
      - [PASS] Scenario 4 (Column 75% missing): recommends 'add_indicator' or 'drop_column'.
      - [PASS] Scenario 5 (Missingness linked to target): recommends 'add_indicator', does NOT drop rows.
      - [PASS] 5 of 5 planted advisor recommendations matched expected ground truth (threshold >= 4/5).
      - [PASS] Every recommendation provides plain-English rationale and measured 5-fold CV scores.
      - [PASS] Close call flag and simplicity tie-breaker trigger when score delta is within CV std.
    - Gate 4B (Fixer Verification):
      - [PASS] After fix: 0 nulls remain in train and test partitions.
      - [PASS] Test row count is strictly unchanged (0 test rows dropped; train-fitted fallback applied).
      - [PASS] Test values changed only in cells that were originally null.
      - [PASS] Learned values derived strictly from train partition (train median 20.0 vs full data 30.0; test imputed with 20.0, zero leakage).
      - [PASS] Decision log persists sequential choices and exported `reproduce.py` builds identical result.
      - [PASS] User override successfully changes applied remediation method.
- Full Phase 4 test suite: 19 passed in `tests/issues/test_nulls_advise.py`, `tests/issues/test_nulls_fix.py`, and `tests/test_decide_routes.py`.
- Regression check: all 65 tests passed across the entire project.
- Neo-brutalist interactive decision cards integrated into `/module/nulls` with Before/After live preview, [APPLY RECOMMENDED], [CHOOSE ANOTHER], [SKIP], batch action [APPLY RECOMMENDED TO ALL], and standalone `reproduce.py` export endpoint.

## 2026-10-26  Phase 5: Duplicates (Detect + Advise + Fix)   PASS
- Automated benchmark runner verified via `python -m doctor.bench duplicates` and `python -m doctor.bench --all`:
  - 9 of 9 Gate 5 checks PASSED:
    - [PASS] 50 of 50 planted exact duplicate rows detected and flagged with certainty='certain'.
    - [PASS] 10 of 10 planted conflicting duplicate pairs (identical features, differing target) detected with severity='high'.
    - [PASS] 20 of 20 planted near-duplicate rows (casing, whitespace, precision fuzz) detected and flagged as certainty='possible'.
    - [PASS] 0 false alarms on clean_control.csv (0 exact or conflicting duplicates flagged).
    - [PASS] Fixer verification: After `keep_first` fix, exactly 0 duplicate rows remain and row count drops by exactly 50 (530 == 530).
    - [PASS] Test set safety: test rows are never dropped during deduplication (`is_test=True` preserves row count).
    - [PASS] Conflicting pairs fixer: `drop_all_conflicting` successfully removes ambiguous contradictory label pairs.
    - [PASS] Pre-balanced warning fires on 50/50 oversampled file (synthetic pre-balancing leakage guard).
    - [PASS] Pre-balanced warning stays quiet on clean control.
- Full project test suite: all 80 tests passed across the entire project (`pytest -v`).
- Neo-brutalist interactive duplicate management integrated into `/module/duplicates` with diagnostic summary cards, conflicting target alert banner, pre-balanced warning card, and quick remediation actions ([DROP EXACT DUPLICATES], [DROP CONFLICTING ROWS], [EXPORT REPRODUCE.PY]).

